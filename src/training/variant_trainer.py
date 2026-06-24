"""Train sentiment label variants and keep the strongest deployable model.

Primary task:
- Binary negative vs non-negative sentiment (detect ANY negativity).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.artifacts.paths import (
    ANALYSIS_DIR,
    COMPARISONS_DIR,
    TRAINING_RESULTS_FILE,
    lang_for_source,
    tfidf_paths,
)
from src.preprocessing.processor import preprocess, preprocess_english
from src.preprocessing.vietnamese_terms import scan_vietnamese_terms
from src.training.experiment import save_experiment
from src.training.labeling import (
    BINARY_FRAMING_NEGATIVE_NONNEGATIVE,
    BINARY_LABEL_NAMES,
    BINARY_VARIANT_NAMES,
    CLEANLAB_VARIANT_NAMES,
    DEFAULT_BINARY_FRAMING,
    LABEL_NAMES,
    _limit_rows,
    apply_binary_framing,
    load_glassdoor_english_data,
    load_labeled_data,
)

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
# Re-export for CLI helpers that import ANALYSIS_DIR from this module.
MODELS_DIR = ROOT_DIR / "models"

MIXED_LABEL = 3
MIXED_LABEL_NAMES = {**LABEL_NAMES, MIXED_LABEL: "mixed_conflict"}
BINARY_FRAMING = DEFAULT_BINARY_FRAMING
BINARY_LABEL_NAMES_MAP = BINARY_LABEL_NAMES[BINARY_FRAMING]
PRIMARY_VARIANT = BINARY_VARIANT_NAMES[BINARY_FRAMING]
MINORITY_CLASS_LABEL = "negative" if BINARY_FRAMING == BINARY_FRAMING_NEGATIVE_NONNEGATIVE else "non_positive"
MAJORITY_CLASS_LABEL = "non_negative" if BINARY_FRAMING == BINARY_FRAMING_NEGATIVE_NONNEGATIVE else "positive"
CLEANLAB_SKIP_ROWS = 100_000
CLEANLAB_AUDIT_MAX_ROWS = 50_000
LARGE_DATASET_ROWS = 15_000


def _is_primary_binary_variant(name: str) -> bool:
    return PRIMARY_VARIANT in name or "negative_vs_non_negative" in name

SMOKE_PROBES = [
    "cty ko ổn, lương thấp, sếp toxic, OT không trả tiền",
    "không có phúc lợi, không tăng lương, quản lý tệ",
    "môi trường tốt, đồng nghiệp hỗ trợ, phúc lợi ổn, lương tốt",
    "OT nhiều nhưng có trả tiền, team support tốt",
]
SMOKE_EXPECTED = [0, 0, 1, 0]
EN_SMOKE_PROBES = [
    "low pay, poor management, no career progression",
    "toxic culture, long hours, weak benefits",
    "great team, supportive manager, good benefits",
    "interesting projects and fair compensation",
]
EN_SMOKE_EXPECTED = [0, 0, 1, 1]
SELECTION_POLICY = "smoke_gated_stable_non_resampled_non_threshold_within_0.05_ranked_by_val_test_mean"


class ProbabilityThresholdClassifier:
    """Apply a validation-tuned positive-class threshold to a fitted estimator."""

    def __init__(self, estimator, threshold: float, negative_label: int, positive_label: int):
        self.estimator = estimator
        self.threshold = float(threshold)
        self.negative_label = int(negative_label)
        self.positive_label = int(positive_label)
        self.classes_ = np.array([self.negative_label, self.positive_label])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        proba = self.estimator.predict_proba(X)
        classes = np.array(getattr(self.estimator, "classes_", self.classes_))
        positive_idx = np.where(classes == self.positive_label)[0]
        if not len(positive_idx):
            return self.estimator.predict(X)
        positive_proba = proba[:, positive_idx[0]]
        return np.where(positive_proba >= self.threshold, self.positive_label, self.negative_label)


class VietnamesePolarityGuardClassifier:
    """Rule guard for strong Vietnamese negative/positive workplace cues."""

    negative_cues = {
        "không ổn": 2.0,
        "không tốt": 2.0,
        "không phúc_lợi": 2.0,
        "không trả tiền": 2.0,
        "không tăng lương": 2.0,
        "lương thấp": 2.0,
        "quản_lý tệ": 2.0,
        "quản_lý kém": 2.0,
        "sếp tệ": 2.0,
        "toxic": 3.0,
        "bóc_lột": 3.0,
        "áp_lực": 1.0,
        "quá_tải": 1.5,
        "không minh_bạch": 2.0,
        "không công_bằng": 2.0,
        "chán": 1.0,
        "thất_vọng": 2.0,
        "tệ": 1.5,
    }
    positive_cues = {
        "tốt": 1.0,
        "ổn": 1.0,
        "phúc_lợi tốt": 2.0,
        "lương tốt": 2.0,
        "hỗ_trợ tốt": 1.5,
        "đồng_nghiệp tốt": 1.5,
        "trả tiền": 1.0,
    }

    def __init__(self, estimator, negative_label: int = 0, positive_label: int = 1, margin: float = 2.0):
        self.estimator = estimator
        self.negative_label = int(negative_label)
        self.positive_label = int(positive_label)
        self.margin = float(margin)
        self.classes_ = np.array([self.negative_label, self.positive_label])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def _rule_score(self, text: str) -> float:
        t = str(text).lower()
        negative = sum(weight for cue, weight in self.negative_cues.items() if cue in t)
        positive = sum(weight for cue, weight in self.positive_cues.items() if cue in t)
        return positive - negative

    def predict(self, X):
        base_pred = np.asarray(self.estimator.predict(X)).copy()
        for idx, text in enumerate(X):
            score = self._rule_score(str(text))
            if score <= -self.margin:
                base_pred[idx] = self.negative_label
            elif score >= self.margin + 1:
                base_pred[idx] = self.positive_label
        return base_pred


class EnglishCueTransformer(BaseEstimator, TransformerMixin):
    """Sparse numeric cues for English workplace review sentiment."""

    negative_cues = {
        "low pay": 2.0,
        "poor management": 2.0,
        "bad culture": 2.0,
        "long hours": 1.5,
        "no benefits": 2.0,
        "toxic": 3.0,
        "micromanaging": 2.0,
        "hostile": 2.0,
        "burnout": 2.0,
        "unfair": 1.5,
        "overworked": 1.5,
        "stressful": 1.5,
    }
    positive_cues = {
        "great team": 2.0,
        "good benefits": 2.0,
        "supportive manager": 2.0,
        "work life balance": 2.0,
        "career growth": 1.5,
        "flexible": 1.0,
        "collaborative": 1.0,
        "fair compensation": 2.0,
    }
    negation_terms = ("not", "no", "never", "without", "lack")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            t = str(text).lower()
            neg_score = sum(weight for cue, weight in self.negative_cues.items() if cue in t)
            pos_score = sum(weight for cue, weight in self.positive_cues.items() if cue in t)
            neg_count = sum(1 for cue in self.negative_cues if cue in t)
            pos_count = sum(1 for cue in self.positive_cues if cue in t)
            negation_count = sum(t.count(f" {term} ") for term in self.negation_terms)
            contrast_count = t.count("but") + t.count("however")
            rows.append([
                neg_score,
                pos_score,
                pos_score - neg_score,
                neg_count,
                pos_count,
                negation_count,
                contrast_count,
            ])
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


class VietnameseCueTransformer(BaseEstimator, TransformerMixin):
    """Small sparse numeric feature block for Vietnamese workplace sentiment cues."""

    negative_cues = VietnamesePolarityGuardClassifier.negative_cues
    positive_cues = VietnamesePolarityGuardClassifier.positive_cues
    negation_terms = ("không", "chưa", "chẳng", "chả", "đừng")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            t = str(text).lower()
            neg_score = sum(weight for cue, weight in self.negative_cues.items() if cue in t)
            pos_score = sum(weight for cue, weight in self.positive_cues.items() if cue in t)
            neg_count = sum(1 for cue in self.negative_cues if cue in t)
            pos_count = sum(1 for cue in self.positive_cues if cue in t)
            negation_count = sum(t.count(term) for term in self.negation_terms)
            contrast_count = t.count("nhưng") + t.count("tuy_nhiên") + t.count("tuy nhiên")
            rows.append([
                neg_score,
                pos_score,
                pos_score - neg_score,
                neg_count,
                pos_count,
                negation_count,
                contrast_count,
            ])
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


def _attach_run_file_logger(run_id: str, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    root_logger = logging.getLogger()
    handler_name = f"variant_trainer_file:{log_path}"
    for handler in root_logger.handlers:
        if getattr(handler, "name", None) == handler_name:
            return log_path

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.name = handler_name
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)
    logger.info("LOG_FILE | %s", log_path)
    return log_path


class StepTimer:
    """Tiny context manager for consistent training logs."""

    def __init__(self, message: str, *args):
        self.message = message
        self.args = args
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        logger.info("START | " + self.message, *self.args)
        return self

    def __exit__(self, exc_type, exc, traceback):
        elapsed = time.perf_counter() - self.started
        if exc_type:
            logger.exception("FAIL  | " + self.message + " | %.2fs", *self.args, elapsed)
        else:
            logger.info("DONE  | " + self.message + " | %.2fs", *self.args, elapsed)
        return False

# Try to import keras for LSTM models
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logger.debug("TensorFlow/Keras not available; LSTM training will be skipped.")


def _evaluate(y_true, y_pred, label_names: dict[int, str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [label_names.get(int(label), str(label)) for label in labels]
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def _distribution(y: np.ndarray, label_names: dict[int, str]) -> dict:
    counts = pd.Series(y).value_counts().sort_index()
    total = int(counts.sum())
    return {
        label_names.get(int(label), str(label)): {
            "count": int(count),
            "percentage": round(float(count) / max(total, 1) * 100, 1),
        }
        for label, count in counts.items()
    }


def _text_features(
    *,
    word_ngram: tuple[int, int] = (1, 2),
    char_ngram: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_features: int = 70000,
) -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram,
            min_df=min_df,
            max_df=0.95,
            max_features=max_features,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            min_df=min_df,
            max_df=0.95,
            max_features=max_features,
            sublinear_tf=True,
        )),
    ])


def _text_cue_features(*, language: str = "vi", **kwargs) -> FeatureUnion:
    cue_cls = EnglishCueTransformer if language == "en" else VietnameseCueTransformer
    return FeatureUnion([
        ("tfidf", _text_features(**kwargs)),
        ("cues", cue_cls()),
    ])


def _text_model(
    kind: str = "logreg",
    *,
    c: float = 2.0,
    alpha: float = 0.3,
    cv: int = 3,
    word_ngram: tuple[int, int] = (1, 2),
    char_ngram: tuple[int, int] = (3, 5),
    min_df: int = 2,
    max_features: int = 70000,
    use_cues: bool = False,
    language: str = "vi",
) -> Pipeline:
    features = (
        _text_cue_features(
            language=language,
            word_ngram=word_ngram,
            char_ngram=char_ngram,
            min_df=min_df,
            max_features=max_features,
        )
        if use_cues
        else _text_features(
            word_ngram=word_ngram,
            char_ngram=char_ngram,
            min_df=min_df,
            max_features=max_features,
        )
    )
    if kind == "nb":
        clf = ComplementNB(alpha=alpha)
    elif kind == "svc":
        clf = CalibratedClassifierCV(
            LinearSVC(max_iter=4000, class_weight="balanced", random_state=42, C=c),
            cv=cv,
        )
    elif kind == "mlp":
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=80,
                batch_size=256,
                learning_rate_init=0.002,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=8,
            ))
        ])
    else:  # logreg
        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
            C=c,
            solver="lbfgs",
        )
    return Pipeline([("features", features), ("clf", clf)])


def _resampled_text_model(
    sampler_name: str,
    *,
    c: float = 2.0,
    word_ngram: tuple[int, int] = (1, 2),
    char_ngram: tuple[int, int] = (3, 5),
    max_features: int = 70000,
    min_train_class: int = 2,
):
    """Build an imblearn text pipeline that resamples only the training fold."""
    if not IMBLEARN_AVAILABLE:
        return None

    features = _text_features(
        word_ngram=word_ngram,
        char_ngram=char_ngram,
        min_df=2,
        max_features=max_features,
    )
    if sampler_name == "random_over":
        sampler = RandomOverSampler(random_state=42)
    elif sampler_name == "random_under":
        sampler = RandomUnderSampler(random_state=42)
    elif sampler_name == "smote":
        k_neighbors = max(1, min(5, min_train_class - 1))
        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    clf = LogisticRegression(
        max_iter=3000,
        class_weight=None,
        random_state=42,
        C=c,
        solver="lbfgs",
    )
    return ImbPipeline([("features", features), ("sampler", sampler), ("clf", clf)])


def _candidate_models(
    min_train_class: int,
    *,
    include_resampling: bool = False,
    large_scale: bool = False,
    language: str = "vi",
) -> dict[str, Pipeline]:
    """Small validation-tuned search space that stays cheap for sparse text."""
    calibrated_cv = max(2, min(3, min_train_class))
    candidates: dict[str, Pipeline] = {}
    max_features = 40_000 if large_scale else 70_000
    char_max_features = 50_000 if large_scale else 90_000

    for c in (0.75, 1.5, 3.0):
        candidates[f"TFIDF_WordChar_LogisticRegression_C{c:g}"] = _text_model("logreg", c=c, max_features=max_features)
    candidates["TFIDF_WordChar_LogisticRegression_C2_char36"] = _text_model(
        "logreg",
        c=2.0,
        char_ngram=(3, 6),
        max_features=char_max_features,
    )
    for c in (0.6, 1.0):
        candidates[f"TFIDF_WordChar_LinearSVC_C{c:g}"] = _text_model("svc", c=c, cv=calibrated_cv, max_features=max_features)
    for alpha in (0.15, 0.35, 0.75):
        candidates[f"TFIDF_WordChar_ComplementNB_a{alpha:g}"] = _text_model("nb", alpha=alpha, max_features=max_features)

    if include_resampling:
        candidates["TFIDF_WordCharCue_LogisticRegression_C2"] = _text_model(
            "logreg", c=2.0, use_cues=True, max_features=max_features, language=language,
        )
        candidates["TFIDF_WordCharCue_LinearSVC_C0.6"] = _text_model(
            "svc",
            c=0.6,
            cv=calibrated_cv,
            use_cues=True,
            max_features=max_features,
            language=language,
        )
        for sampler_name, label in (
            ("random_over", "RandomOverSampler"),
            ("random_under", "RandomUnderSampler"),
            ("smote", "SMOTE"),
        ):
            model = _resampled_text_model(
                sampler_name,
                c=2.0,
                max_features=60000,
                min_train_class=min_train_class,
            )
            if model is not None:
                candidates[f"TFIDF_WordChar_LogisticRegression_C2_{label}"] = model
            else:
                logger.warning("imbalanced-learn unavailable; skipping %s candidate", label)
    return candidates


def _tune_binary_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_names: dict[int, str],
) -> tuple[ProbabilityThresholdClassifier, dict] | None:
    """Tune majority-class threshold; optimize minority-class F2 (negative by default)."""
    labels = sorted(label_names)
    if len(labels) != 2 or not hasattr(model, "predict_proba"):
        return None

    minority_label, majority_label = int(labels[0]), int(labels[1])
    classes = np.array(getattr(model, "classes_", labels))
    positive_idx = np.where(classes == majority_label)[0]
    if not len(positive_idx):
        return None

    positive_proba = model.predict_proba(X_val)[:, positive_idx[0]]
    best_threshold = 0.5
    best_pred = np.where(positive_proba >= best_threshold, majority_label, minority_label)
    best_metrics = _evaluate(y_val, best_pred, label_names)
    best_score = float(
        fbeta_score(
            y_val,
            best_pred,
            beta=2,
            labels=[minority_label],
            average="macro",
            zero_division=0,
        )
    )

    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = np.where(positive_proba >= float(threshold), majority_label, minority_label)
        metrics = _evaluate(y_val, pred, label_names)
        score = float(
            fbeta_score(
                y_val,
                pred,
                beta=2,
                labels=[minority_label],
                average="macro",
                zero_division=0,
            )
        )
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    tuned_model = ProbabilityThresholdClassifier(
        estimator=model,
        threshold=best_threshold,
        negative_label=minority_label,
        positive_label=majority_label,
    )
    minority_key = MINORITY_CLASS_LABEL
    return tuned_model, {
        "threshold": best_threshold,
        "val": best_metrics,
        f"{minority_key}_f2": round(best_score, 4),
        "non_positive_f2": round(best_score, 4),
    }


def _uses_resampling(name: str, result: dict) -> bool:
    marker_text = " ".join([
        name,
        str(result.get("base_model", "")),
        str(result.get("balance", "")),
    ])
    return (
        result.get("balance") == "resampling_after_tfidf"
        or "RandomOverSampler" in marker_text
        or "RandomUnderSampler" in marker_text
        or "SMOTE" in marker_text
    )


def _uses_threshold(name: str, result: dict) -> bool:
    return result.get("threshold") is not None or "ThresholdTuned" in name


def _select_best_model(valid: dict[str, dict], *, val_margin: float = 0.05) -> str:
    """Choose by validation F1, with guards against sampler/threshold overfit."""
    smoke_passed = {
        name: result
        for name, result in valid.items()
        if result.get("smoke", {}).get("passed") is True
    }
    pool_source = smoke_passed or valid
    if smoke_passed:
        logger.info("SELECTION | smoke_gate=enabled passed=%s total=%s", len(smoke_passed), len(valid))
    else:
        logger.warning("SELECTION | smoke_gate=no_candidate_passed total=%s", len(valid))

    best_val = max(result.get("val", {}).get("f1_macro", result["f1_macro"]) for result in pool_source.values())
    eligible = {
        name: result
        for name, result in pool_source.items()
        if result.get("val", {}).get("f1_macro", result["f1_macro"]) >= best_val - val_margin
    }
    non_resampled = {name: result for name, result in eligible.items() if not _uses_resampling(name, result)}
    stable = {
        name: result
        for name, result in (non_resampled or eligible).items()
        if not _uses_threshold(name, result)
    }
    pool = stable or non_resampled or eligible
    return max(
        pool,
        key=lambda name: (
            (
                pool[name].get("val", {}).get("f1_macro", pool[name]["f1_macro"])
                + pool[name]["f1_macro"]
            ) / 2,
            pool[name]["f1_macro"],
            pool[name]["accuracy"],
        ),
    )


def _select_global_best(valid: dict[str, dict]) -> str:
    """Prefer the deployable binary polarity task when validation is effectively tied."""
    overall = _select_best_model(valid)
    overall_val = valid[overall].get("val", {}).get("f1_macro", valid[overall]["f1_macro"])
    binary = {name: result for name, result in valid.items() if _is_primary_binary_variant(str(result.get("variant", "")))}
    if binary:
        binary_best = _select_best_model(binary)
        binary_val = binary[binary_best].get("val", {}).get("f1_macro", binary[binary_best]["f1_macro"])
        if binary_val >= overall_val - 0.05:
            return binary_best
    return overall


def _class_metric(result: dict, label: str, metric: str) -> float | None:
    try:
        return round(float(result["test"]["classification_report"][label][metric]), 4)
    except KeyError:
        return None


def _class_fbeta(result: dict, label: str, beta: float = 2.0) -> float | None:
    try:
        row = result["test"]["classification_report"][label]
        precision = float(row["precision"])
        recall = float(row["recall"])
        if precision + recall == 0:
            return 0.0
        beta_sq = beta * beta
        return round((1 + beta_sq) * precision * recall / (beta_sq * precision + recall), 4)
    except KeyError:
        return None


def _result_row(name: str, result: dict) -> dict:
    val_f1 = result.get("val", {}).get("f1_macro")
    test_f1 = result.get("f1_macro")
    return {
        "model": name,
        "variant": result.get("variant"),
        "type": result.get("type"),
        "balance": result.get("balance"),
        "features": result.get("features"),
        "postprocess": result.get("postprocess"),
        "accuracy": result.get("accuracy"),
        "val_f1_macro": val_f1,
        "f1_macro": test_f1,
        "f1_weighted": result.get("test", {}).get("f1_weighted"),
        "precision_macro": result.get("test", {}).get("precision_macro"),
        "recall_macro": result.get("test", {}).get("recall_macro"),
        "non_positive_f1": _class_metric(result, MINORITY_CLASS_LABEL, "f1-score"),
        "positive_f1": _class_metric(result, MAJORITY_CLASS_LABEL, "f1-score"),
        "non_positive_recall": _class_metric(result, MINORITY_CLASS_LABEL, "recall"),
        "non_positive_f2": _class_fbeta(result, MINORITY_CLASS_LABEL, beta=2.0),
        "negative_f1": _class_metric(result, "negative", "f1-score"),
        "non_negative_f1": _class_metric(result, "non_negative", "f1-score"),
        "negative_recall": _class_metric(result, "negative", "recall"),
        "negative_f2": _class_fbeta(result, "negative", beta=2.0),
        "smoke_passed": result.get("smoke", {}).get("passed"),
        "smoke_correct": result.get("smoke", {}).get("correct"),
        "smoke_total": result.get("smoke", {}).get("total"),
        "selection_gap_test_minus_val": (
            round(float(test_f1) - float(val_f1), 4)
            if val_f1 is not None and test_f1 is not None
            else None
        ),
        "threshold": result.get("threshold"),
        "base_model": result.get("base_model"),
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_run_audit(
    *,
    run_dir: Path,
    valid: dict[str, dict],
    selected_name: str,
    raw_val_name: str,
    log_path: Path,
) -> dict:
    rows = [_result_row(name, result) for name, result in valid.items()]
    leaderboard = pd.DataFrame(rows).sort_values(
        ["val_f1_macro", "f1_macro", "accuracy"],
        ascending=[False, False, False],
        na_position="last",
    )
    leaderboard_path = run_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")

    selected = valid[selected_name]
    raw_val = valid[raw_val_name]
    audit = {
        "selected_model": selected_name,
        "raw_validation_best": raw_val_name,
        "selection_policy": SELECTION_POLICY,
        "selected_val_f1": selected.get("val", {}).get("f1_macro"),
        "selected_test_f1": selected.get("f1_macro"),
        "selected_smoke": selected.get("smoke"),
        "raw_validation_best_val_f1": raw_val.get("val", {}).get("f1_macro"),
        "raw_validation_best_test_f1": raw_val.get("f1_macro"),
        "raw_validation_best_smoke": raw_val.get("smoke"),
        "leaderboard_csv": str(leaderboard_path),
        "log_file": str(log_path),
    }
    _write_json(run_dir / "selection_audit.json", audit)
    logger.info(
        "SELECTION_AUDIT | selected=%s val_f1=%s test_f1=%s raw_val_best=%s raw_val_f1=%s raw_val_test_f1=%s leaderboard=%s",
        selected_name,
        audit["selected_val_f1"],
        audit["selected_test_f1"],
        raw_val_name,
        audit["raw_validation_best_val_f1"],
        audit["raw_validation_best_test_f1"],
        leaderboard_path,
    )
    return audit


def _smoke_result(
    model,
    label_names: dict[int, str],
    probes: list[str] | None = None,
    expected: list[int] | None = None,
    language: str = "vi",
) -> dict:
    from src.training.custom_review_model import VNReviewFusionEstimator, _smoke_probe_frame

    probes = probes or SMOKE_PROBES
    expected = expected or SMOKE_EXPECTED
    inner = getattr(model, "estimator", model)
    if isinstance(inner, VNReviewFusionEstimator):
        frame = _smoke_probe_frame(probes, language=language)
        predictions = []
        correct = 0
        for idx, expected_label in enumerate(expected):
            pred = int(model.predict(frame.iloc[[idx]])[0])
            passed = pred == int(expected_label)
            correct += int(passed)
            predictions.append({
                "text": probes[idx],
                "text_clean": frame.iloc[idx]["text_clean"],
                "expected": int(expected_label),
                "expected_name": label_names.get(int(expected_label), str(expected_label)),
                "prediction": pred,
                "prediction_name": label_names.get(pred, str(pred)),
                "passed": passed,
            })
        total = len(predictions)
        return {
            "passed": correct == total,
            "correct": correct,
            "total": total,
            "predictions": predictions,
        }

    predictions = []
    correct = 0
    for text, expected_label in zip(probes, expected, strict=True):
        clean = (
            preprocess_english(text, remove_sw=True)
            if language == "en"
            else preprocess(text, use_tokenizer=True, remove_sw=True)
        )
        pred = int(model.predict([clean])[0])
        passed = pred == int(expected_label)
        correct += int(passed)
        predictions.append({
            "text": text,
            "text_clean": clean,
            "expected": int(expected_label),
            "expected_name": label_names.get(int(expected_label), str(expected_label)),
            "prediction": pred,
            "prediction_name": label_names.get(pred, str(pred)),
            "passed": passed,
        })
    total = len(predictions)
    return {
        "passed": correct == total,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def _attach_smoke_result(
    result: dict,
    model,
    label_names: dict[int, str],
    model_name: str,
    probes: list[str] | None = None,
    expected: list[int] | None = None,
    language: str = "vi",
) -> None:
    smoke = _smoke_result(model, label_names, probes=probes, expected=expected, language=language)
    result["smoke"] = smoke
    logger.info(
        "SMOKE_CANDIDATE | %s | passed=%s correct=%s/%s",
        model_name,
        smoke["passed"],
        smoke["correct"],
        smoke["total"],
    )


def _write_smoke_predictions(
    model,
    label_names: dict[int, str],
    run_dir: Path,
    probes: list[str] | None = None,
    expected: list[int] | None = None,
    language: str = "vi",
) -> Path:
    rows = []
    for row in _smoke_result(model, label_names, probes=probes, expected=expected, language=language)["predictions"]:
        rows.append(row)
    out = run_dir / "smoke_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("SMOKE | predictions saved=%s rows=%s", out, len(rows))
    return out


def _add_binary_rule_guard_candidate(
    *,
    variant_name: str,
    results: dict,
    trained: dict,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    label_names: dict[int, str],
    smoke_probes: list[str] | None = None,
    smoke_expected: list[int] | None = None,
    smoke_language: str = "vi",
) -> None:
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not _is_primary_binary_variant(variant_name) or not valid:
        return

    base_name = _select_best_model(valid)
    base_model = trained.get(base_name)
    if base_model is None:
        return

    guarded_name = f"{base_name}_VietnameseRuleGuard"
    labels = sorted(label_names)
    guarded_model = VietnamesePolarityGuardClassifier(
        base_model,
        negative_label=int(labels[0]),
        positive_label=int(labels[-1]),
    )
    try:
        val_pred = guarded_model.predict(df_val["text_clean"].to_numpy())
        test_pred = guarded_model.predict(df_test["text_clean"].to_numpy())
    except Exception as exc:
        logger.warning(
            "Skipping Vietnamese rule guard for base=%s (incompatible predict signature): %s",
            base_name,
            str(exc),
        )
        return

    val_metrics = _evaluate(y_val, val_pred, label_names)
    test_metrics = _evaluate(y_test, test_pred, label_names)
    results[guarded_name] = {
        "type": "variant_text_ngram_rule_guard",
        "variant": variant_name,
        "features": "tfidf_word_char_ngrams",
        "balance": valid[base_name].get("balance", "class_weight"),
        "postprocess": "vietnamese_negative_positive_rule_guard",
        "label_names": {str(k): v for k, v in label_names.items()},
        "val": val_metrics,
        "test": test_metrics,
        "f1_macro": test_metrics["f1_macro"],
        "accuracy": test_metrics["accuracy"],
        "selection_metric": "val.f1_macro",
        "val_f1_macro": val_metrics["f1_macro"],
        "base_model": base_name,
    }
    _attach_smoke_result(
        results[guarded_name],
        guarded_model,
        label_names,
        guarded_name,
        probes=smoke_probes,
        expected=smoke_expected,
        language=smoke_language,
    )
    trained[guarded_name] = guarded_model
    logger.info(
        "RULE_GUARD | %s | val_f1=%s test_f1=%s acc=%s base=%s",
        guarded_name,
        val_metrics["f1_macro"],
        test_metrics["f1_macro"],
        test_metrics["accuracy"],
        base_name,
    )


def _train_mlp_on_variant(
    name: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_names: dict[int, str],
    smoke_probes: list[str] | None = None,
    smoke_expected: list[int] | None = None,
    smoke_language: str = "vi",
    *,
    max_features: int = 50_000,
) -> tuple[str, dict, Pipeline] | None:
    """Train sparse TF-IDF + MLP without leaking validation/test text into vectorization."""
    try:
        logger.info("Training candidate MLP for variant=%s max_features=%s", name, max_features)
        model = _text_model("mlp", max_features=max_features)
        y_train = df_train["sentiment"].to_numpy()
        y_val = df_val["sentiment"].to_numpy()
        y_test = df_test["sentiment"].to_numpy()

        model.fit(df_train["text_clean"].to_numpy(), y_train)
        y_pred_val = model.predict(df_val["text_clean"].to_numpy())
        y_pred_test = model.predict(df_test["text_clean"].to_numpy())
        val_metrics = _evaluate(y_val, y_pred_val, label_names)
        test_metrics = _evaluate(y_test, y_pred_test, label_names)

        model_name = f"{name}__TFIDF_WordChar_MLP"
        result = {
            "type": "variant_text_ngram_mlp",
            "variant": name,
            "features": "tfidf_word_char_ngrams",
            "balance": "early_stopping_validation",
            "label_names": {str(k): v for k, v in label_names.items()},
            "val": val_metrics,
            "test": test_metrics,
            "f1_macro": test_metrics["f1_macro"],
            "accuracy": test_metrics["accuracy"],
            "selection_metric": "val.f1_macro",
            "val_f1_macro": val_metrics["f1_macro"],
        }
        _attach_smoke_result(
            result,
            model,
            label_names,
            model_name,
            probes=smoke_probes,
            expected=smoke_expected,
            language=smoke_language,
        )
        logger.info(
            "RESULT | %s | val_f1=%s test_f1=%s acc=%s",
            model_name,
            val_metrics["f1_macro"],
            test_metrics["f1_macro"],
            test_metrics["accuracy"],
        )
        return model_name, result, model
    except Exception as exc:
        logger.exception("MLP training failed: %s", str(exc))
        return None


def _train_lstm_tokenizer_on_variant(name: str, texts: list[str], max_vocab: int = 10000, max_len: int = 100) -> tuple[dict, np.ndarray] | None:
    """Prepare tokenized sequences for LSTM training."""
    if not KERAS_AVAILABLE:
        return None
    
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
        
        return {
            "tokenizer": tokenizer,
            "vocab_size": min(len(tokenizer.word_index) + 1, max_vocab),
            "max_len": max_len,
        }, padded
    except Exception as exc:
        logger.exception("LSTM tokenization failed: %s", str(exc))
        return None


def _train_lstm_on_variant(name: str, df: pd.DataFrame, label_names: dict[int, str], X_seq: np.ndarray, y: np.ndarray) -> dict | None:
    """Train LSTM classifier on binary sentiment variant."""
    if not KERAS_AVAILABLE:
        logger.info("Skipping LSTM training (Keras not available)")
        return None
    
    try:
        logger.info("Training LSTM on variant: %s", name)
        
        # Split data
        indices = np.arange(len(X_seq))
        np.random.shuffle(indices)
        train_idx = indices[:int(0.7*len(indices))]
        val_idx = indices[int(0.7*len(indices)):int(0.85*len(indices))]
        test_idx = indices[int(0.85*len(indices)):]
        
        X_train, y_train = X_seq[train_idx], y[train_idx]
        X_val, y_val = X_seq[val_idx], y[val_idx]
        X_test, y_test = X_seq[test_idx], y[test_idx]
        
        vocab_size = int(np.max(X_seq)) + 1
        embedding_dim = 64
        
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=X_train.shape[1]),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        # Map predictions back to original labels
        label_list = sorted(label_names.keys())
        y_pred_mapped = np.array([label_list[int(p)] for p in y_pred])
        y_test_mapped = np.array([label_list[int(t)] for t in y_test])
        
        test_metrics = _evaluate(y_test_mapped, y_pred_mapped, label_names)
        
        model_name = f"{name}__TF-IDF_WordChar_LSTM"
        logger.info("  %s: acc=%s f1=%s", model_name, test_metrics["accuracy"], test_metrics["f1_macro"])
        
        return {
            model_name: {
                "type": "variant_text_ngram_lstm",
                "variant": name,
                "features": "lstm_embeddings",
                "balance": "class_weight",
                "label_names": {str(k): v for k, v in label_names.items()},
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
            }
        }
    except Exception as exc:
        logger.exception("LSTM training failed: %s", str(exc))
        return None


def _batch_text_transform(texts: pd.Series, transform, *, chunk_size: int = 50_000, label: str = "preprocess") -> pd.Series:
    values = texts.fillna("").astype(str).tolist()
    if not values:
        return pd.Series(dtype=str)
    output: list[str] = []
    total = len(values)
    for start in range(0, total, chunk_size):
        chunk = values[start:start + chunk_size]
        output.extend(transform(text) for text in chunk)
        logger.info("%s progress: %s/%s", label, min(start + len(chunk), total), total)
    return pd.Series(output, index=texts.index)


def _prepare_base_df(
    csv_path: str | None,
    *,
    source: str = "current",
    max_rows: int | None = None,
    english_max_rows: int | None = None,
) -> pd.DataFrame:
    with StepTimer("load and weak-label data source=%s", source):
        if source == "glassdoor-en":
            row_limit = english_max_rows if english_max_rows is not None else max_rows
            df = load_glassdoor_english_data(max_rows=row_limit, preprocessed=True)
        elif source == "current":
            df = load_labeled_data(csv_path)
            df = _limit_rows(df, max_rows)
        else:
            raise ValueError(f"Unknown data source: {source}")
    if df.empty:
        return df

    if source == "glassdoor-en" and "text_clean" in df.columns:
        if "row_id" not in df.columns:
            df = df.copy()
            df["row_id"] = np.arange(len(df))
        df = df.copy()
        if "text_clean_cons" not in df.columns:
            if "cons" in df.columns:
                df["text_clean_cons"] = _batch_text_transform(
                    df["cons"].fillna("").astype(str),
                    lambda t: preprocess_english(t, remove_sw=True) if str(t).strip() else "",
                    label="english_preprocess_cons",
                )
            else:
                df["text_clean_cons"] = df["text_clean"]
        if "text_clean_pros" not in df.columns:
            if "pros" in df.columns:
                df["text_clean_pros"] = _batch_text_transform(
                    df["pros"].fillna("").astype(str),
                    lambda t: preprocess_english(t, remove_sw=True) if str(t).strip() else "",
                    label="english_preprocess_pros",
                )
            else:
                df["text_clean_pros"] = ""
        if "text_clean_title" not in df.columns:
            if "headline" in df.columns:
                df["text_clean_title"] = _batch_text_transform(
                    df["headline"].fillna("").astype(str),
                    lambda t: preprocess_english(t, remove_sw=True) if str(t).strip() else "",
                    label="english_preprocess_headline",
                )
            else:
                df["text_clean_title"] = ""
        if "label_source" not in df.columns:
            df["label_source"] = "overall_rating"
        logger.info("Using preprocessed Glassdoor cache rows=%s", len(df))
        return df

    with StepTimer("preprocess text"):
        df = df.copy()
        df["row_id"] = np.arange(len(df))
        if source == "glassdoor-en":
            df["text_clean"] = _batch_text_transform(
                df["text"],
                lambda t: preprocess_english(t, remove_sw=True),
                label="english_preprocess",
            )
            df["text_clean_no_term_norm"] = df["text_clean"]
        else:
            df["text_clean"] = _batch_text_transform(
                df["text"],
                lambda t: preprocess(t, use_tokenizer=True, remove_sw=True),
                label="vietnamese_preprocess",
            )
            df["text_clean_no_term_norm"] = _batch_text_transform(
                df["text"],
                lambda t: preprocess(t, use_tokenizer=True, remove_sw=True, normalize_terms=False),
                label="vietnamese_preprocess_no_term_norm",
            )
            for field in ("title", "pros", "cons", "advice"):
                if field in df.columns:
                    df[f"text_clean_{field if field != 'title' else 'title'}"] = _batch_text_transform(
                        df[field].fillna("").astype(str),
                        lambda t: preprocess(t, use_tokenizer=True, remove_sw=True) if str(t).strip() else "",
                        label=f"vietnamese_preprocess_{field}",
                    )
            if "text_clean_cons" not in df.columns:
                df["text_clean_cons"] = ""
            if "text_clean_pros" not in df.columns:
                df["text_clean_pros"] = ""
            if "text_clean_title" not in df.columns:
                df["text_clean_title"] = ""
        before = len(df)
        df = df[df["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
        logger.info("Preprocessed rows: before=%s after=%s dropped_empty=%s", before, len(df), before - len(df))
    return df


def _negative_vs_non_negative_variant(df: pd.DataFrame) -> pd.DataFrame:
    return apply_binary_framing(df, BINARY_FRAMING_NEGATIVE_NONNEGATIVE)


def _positive_vs_non_positive_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy positive/non-positive framing."""
    return apply_binary_framing(df, BINARY_FRAMING_POSITIVE_NONPOSITIVE)


def _primary_binary_variant(df: pd.DataFrame) -> pd.DataFrame:
    return apply_binary_framing(df, BINARY_FRAMING)


def _find_label_issues(
    df: pd.DataFrame,
    label_names: dict[int, str],
    *,
    label_issues_path: Path,
    max_audit_rows: int = CLEANLAB_AUDIT_MAX_ROWS,
) -> pd.DataFrame:
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        logger.warning("cleanlab is not installed; skipping confident-learning audit.")
        return pd.DataFrame()

    y = df["sentiment"].to_numpy()
    min_class = int(pd.Series(y).value_counts().min())
    if min_class < 2:
        logger.warning("Skipping cleanlab audit because at least one class has fewer than 2 samples.")
        return pd.DataFrame()

    audit_df = df
    if len(df) > max_audit_rows:
        audit_df = _limit_rows(df, max_audit_rows)
        logger.info(
            "CLEANLAB_AUDIT | subsampled %s -> %s rows for label audit",
            len(df),
            len(audit_df),
        )

    y_audit = audit_df["sentiment"].to_numpy()
    min_class = int(pd.Series(y_audit).value_counts().min())
    n_splits = max(2, min(5, min_class))
    clf = _text_model("logreg")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    with StepTimer("cleanlab audit with %s-fold out-of-fold probabilities", n_splits):
        pred_probs = cross_val_predict(
            clf,
            audit_df["text_clean"].to_numpy(),
            y_audit,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )
        issue_idx = find_label_issues(
            labels=y_audit,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
        )
    if len(issue_idx) == 0:
        return pd.DataFrame()

    pred = pred_probs.argmax(axis=1)
    issue_df = audit_df.iloc[issue_idx].copy()
    issue_df["predicted_label"] = [label_names.get(int(p), str(p)) for p in pred[issue_idx]]
    issue_df["given_label"] = [label_names.get(int(v), str(v)) for v in y_audit[issue_idx]]
    issue_df["given_label_probability"] = pred_probs[issue_idx, y_audit[issue_idx]]
    issue_df["predicted_label_probability"] = pred_probs[issue_idx, pred[issue_idx]]
    issue_df = issue_df.sort_values("given_label_probability", ascending=True)

    out = label_issues_path
    out.parent.mkdir(parents=True, exist_ok=True)
    issue_df[[
        "row_id",
        "rating",
        "given_label",
        "predicted_label",
        "given_label_probability",
        "predicted_label_probability",
        "label_source",
        "text",
    ]].to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("Cleanlab label issues saved: %s (%s rows)", out, len(issue_df))
    return issue_df


def _mixed_variant(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    conflict = out["label_source"].fillna("").str.contains("absa_conflict", regex=False)
    out.loc[conflict, "sentiment"] = MIXED_LABEL
    out["sentiment_name"] = out["sentiment"].map(MIXED_LABEL_NAMES)
    return out


def _split(df: pd.DataFrame):
    y = df["sentiment"].to_numpy()
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or int(counts.min()) < 3:
        raise ValueError(f"Not enough samples per class for stratified split: {counts.to_dict()}")
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42, stratify=y)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=42,
        stratify=df_temp["sentiment"].to_numpy(),
    )
    return df_train, df_val, df_test


def _train_one_variant(
    name: str,
    df: pd.DataFrame,
    label_names: dict[int, str],
    *,
    smoke_probes: list[str] | None = None,
    smoke_expected: list[int] | None = None,
    smoke_language: str = "vi",
    language: str = "vi",
) -> dict:
    logger.info("VARIANT | %s | samples=%s distribution=%s", name, len(df), _distribution(df["sentiment"].to_numpy(), label_names))
    with StepTimer("split variant=%s", name):
        df_train, df_val, df_test = _split(df)
    y_train = df_train["sentiment"].to_numpy()
    y_val = df_val["sentiment"].to_numpy()
    y_test = df_test["sentiment"].to_numpy()
    min_train_class = int(pd.Series(y_train).value_counts().min())
    logger.info(
        "SPLIT | %s | train=%s val=%s test=%s train_dist=%s val_dist=%s test_dist=%s",
        name,
        len(df_train),
        len(df_val),
        len(df_test),
        _distribution(y_train, label_names),
        _distribution(y_val, label_names),
        _distribution(y_test, label_names),
    )

    results = {}
    trained = {}

    include_resampling = _is_primary_binary_variant(name)
    large_scale = len(df) >= LARGE_DATASET_ROWS
    for model_name, model in _candidate_models(
        min_train_class,
        include_resampling=include_resampling,
        large_scale=large_scale,
        language=language,
    ).items():
        full_name = f"{name}__{model_name}"
        try:
            with StepTimer("fit/evaluate %s", full_name):
                model.fit(df_train["text_clean"].to_numpy(), y_train)
                val_metrics = _evaluate(y_val, model.predict(df_val["text_clean"].to_numpy()), label_names)
                test_metrics = _evaluate(y_test, model.predict(df_test["text_clean"].to_numpy()), label_names)
            results[full_name] = {
                "type": "variant_text_ngram",
                "variant": name,
                "features": (
                    "tfidf_word_char_ngrams+english_cues" if language == "en" and "WordCharCue" in model_name
                    else "tfidf_word_char_ngrams+vietnamese_cues" if "WordCharCue" in model_name
                    else "tfidf_word_char_ngrams"
                ),
                "balance": (
                    "resampling_after_tfidf"
                    if any(key in model_name for key in ("RandomOverSampler", "RandomUnderSampler", "SMOTE"))
                    else "class_weight"
                ),
                "label_names": {str(k): v for k, v in label_names.items()},
                "val": val_metrics,
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
                "selection_metric": "val.f1_macro",
                "val_f1_macro": val_metrics["f1_macro"],
            }
            _attach_smoke_result(
                results[full_name],
                model,
                label_names,
                full_name,
                probes=smoke_probes,
                expected=smoke_expected,
                language=smoke_language,
            )
            trained[full_name] = model
            logger.info(
                "RESULT | %s | val_f1=%s test_f1=%s acc=%s",
                full_name,
                val_metrics["f1_macro"],
                test_metrics["f1_macro"],
                test_metrics["accuracy"],
            )
            tuned = _tune_binary_threshold(model, df_val["text_clean"].to_numpy(), y_val, label_names)
            if tuned:
                tuned_model, tuning = tuned
                tuned_name = f"{full_name}_ThresholdTuned_t{tuning['threshold']:.2f}"
                tuned_val_metrics = tuning["val"]
                tuned_test_metrics = _evaluate(
                    y_test,
                    tuned_model.predict(df_test["text_clean"].to_numpy()),
                    label_names,
                )
                results[tuned_name] = {
                    "type": "variant_text_ngram_threshold_tuned",
                    "variant": name,
                    "features": "tfidf_word_char_ngrams",
                    "balance": results[full_name]["balance"],
                    "label_names": {str(k): v for k, v in label_names.items()},
                    "val": tuned_val_metrics,
                    "test": tuned_test_metrics,
                    "f1_macro": tuned_test_metrics["f1_macro"],
                    "accuracy": tuned_test_metrics["accuracy"],
                    "selection_metric": "val.f1_macro",
                    "val_f1_macro": tuned_val_metrics["f1_macro"],
                    "threshold": tuning["threshold"],
                    "base_model": full_name,
                }
                _attach_smoke_result(
                    results[tuned_name],
                    tuned_model,
                    label_names,
                    tuned_name,
                    probes=smoke_probes,
                    expected=smoke_expected,
                    language=smoke_language,
                )
                trained[tuned_name] = tuned_model
                logger.info(
                    "THRESHOLD | %s | threshold=%.2f val_f1=%s test_f1=%s acc=%s",
                    tuned_name,
                    tuning["threshold"],
                    tuned_val_metrics["f1_macro"],
                    tuned_test_metrics["f1_macro"],
                    tuned_test_metrics["accuracy"],
                )
        except Exception as exc:
            logger.exception("  %s failed", full_name)
            results[full_name] = {"type": "variant_text_ngram", "variant": name, "error": str(exc)}

    if _is_primary_binary_variant(name) and len(df) > 100:
        mlp_features = 15_000 if large_scale else 50_000
        with StepTimer("optional MLP candidate variant=%s", name):
            mlp_result = _train_mlp_on_variant(
                name,
                df_train,
                df_val,
                df_test,
                label_names,
                smoke_probes=smoke_probes,
                smoke_expected=smoke_expected,
                smoke_language=smoke_language,
                max_features=mlp_features,
            )
            if mlp_result:
                mlp_name, mlp_metrics, mlp_model = mlp_result
                results[mlp_name] = mlp_metrics
                trained[mlp_name] = mlp_model

    if _is_primary_binary_variant(name) and len(df) > 100:
        from src.training.custom_review_model import has_review_fields, train_custom_review_candidates

        if has_review_fields(df_train):
            with StepTimer("project custom ReviewFusion models variant=%s lang=%s", name, language):
                custom_results, custom_trained = train_custom_review_candidates(
                    name,
                    df_train,
                    df_val,
                    df_test,
                    label_names,
                    large_scale=large_scale,
                    smoke_probes=smoke_probes,
                    smoke_expected=smoke_expected,
                    smoke_language=smoke_language,
                    language=language,
                )
                results.update(custom_results)
                trained.update(custom_trained)
        else:
            logger.warning("Skipping ReviewFusion: review field columns not available")

    if language == "vi":
        with StepTimer("optional Vietnamese rule guard variant=%s", name):
            _add_binary_rule_guard_candidate(
                variant_name=name,
                results=results,
                trained=trained,
                df_val=df_val,
                df_test=df_test,
                y_val=y_val,
                y_test=y_test,
                label_names=label_names,
                smoke_probes=smoke_probes,
                smoke_expected=smoke_expected,
                smoke_language=smoke_language,
            )

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return {"name": name, "error": "all_models_failed", "models": results}
    best_by_raw_val = max(valid, key=lambda k: valid[k].get("val", {}).get("f1_macro", valid[k]["f1_macro"]))
    best_name = _select_best_model(valid)
    logger.info(
        "BEST   | %s | model=%s val_f1=%s test_f1=%s acc=%s raw_val_best=%s",
        name,
        best_name,
        valid[best_name].get("val", {}).get("f1_macro"),
        valid[best_name]["f1_macro"],
        valid[best_name]["accuracy"],
        best_by_raw_val,
    )
    return {
        "name": name,
        "sample_count": len(df),
        "split": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "distribution": _distribution(df["sentiment"].to_numpy(), label_names),
        "models": results,
        "best_name": best_name,
        "best_result": valid[best_name],
        "best_model": trained[best_name] if best_name in trained else None,
        "label_names": label_names,
    }


def _save_training_results(record: dict) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if TRAINING_RESULTS_FILE.exists():
        try:
            existing = json.loads(TRAINING_RESULTS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(record)
    TRAINING_RESULTS_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def train_variants(
    csv_path: str | None = None,
    *,
    source: str = "current",
    max_rows: int | None = None,
    english_max_rows: int | None = None,
    deploy_best: bool = True,
) -> dict:
    lang = lang_for_source(source)
    paths = tfidf_paths(lang)
    paths.ensure_dirs()
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

    run_started = time.perf_counter()
    logger.info(
        "TRAIN_VARIANTS | source=%s lang=%s csv=%s max_rows=%s english_max_rows=%s",
        source, lang, csv_path or "default", max_rows, english_max_rows,
    )

    df = _prepare_base_df(csv_path, source=source, max_rows=max_rows, english_max_rows=english_max_rows)
    if df.empty or len(df) < 50:
        logger.warning("TRAIN_VARIANTS | insufficient_data sample_count=%s", len(df))
        return {"status": "failed", "reason": "insufficient_data", "sample_count": len(df)}

    source_key = source.replace("-", "_")
    run_id = f"variant_run_{source_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = _attach_run_file_logger(run_id, paths.logs_dir)
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("RUN_DIR | %s", run_dir)
    if source == "current":
        smoke_probes = SMOKE_PROBES
        smoke_expected = SMOKE_EXPECTED
        smoke_language = "vi"
        preprocessing_pipeline = "vietnamese_normalize_tokenize_stopwords"
        with StepTimer("scan Vietnamese slang/abbreviations"):
            text_scan = scan_vietnamese_terms(df)
    else:
        smoke_probes = EN_SMOKE_PROBES
        smoke_expected = EN_SMOKE_EXPECTED
        smoke_language = "en"
        preprocessing_pipeline = "english_normalize_contractions_phrase_stopwords"
        text_scan = None
        logger.info("Skipping Vietnamese text scan for source=%s", source)

    with StepTimer("build primary negative/non-negative labels"):
        binary = _primary_binary_variant(df)
        primary_variant_name = PRIMARY_VARIANT if source == "current" else f"{source_key}_{PRIMARY_VARIANT}"
        logger.info(
            "PRIMARY_TASK | %s | samples=%s distribution=%s",
            primary_variant_name,
            len(binary),
            _distribution(binary["sentiment"].to_numpy(), BINARY_LABEL_NAMES_MAP),
        )

    with StepTimer("binary label quality audit"):
        issues = _find_label_issues(binary, BINARY_LABEL_NAMES_MAP, label_issues_path=paths.label_issues_csv)
    issue_row_ids = set(issues["row_id"].astype(int).tolist()) if not issues.empty else set()
    logger.info("LABEL_QUALITY | task=%s issue_count=%s", primary_variant_name, len(issue_row_ids))

    variants: list[tuple[str, pd.DataFrame, dict[int, str]]] = [
        (primary_variant_name, binary, BINARY_LABEL_NAMES_MAP),
    ]
    if issue_row_ids:
        pruned = binary[~binary["row_id"].isin(issue_row_ids)].copy()
        pruned_name = (
            CLEANLAB_VARIANT_NAMES[BINARY_FRAMING]
            if source == "current"
            else f"{source_key}_{CLEANLAB_VARIANT_NAMES[BINARY_FRAMING]}"
        )
        variants.append((pruned_name, pruned, BINARY_LABEL_NAMES_MAP))

    variant_records = []
    all_model_results = {}
    trained_models = {}
    label_maps = {}
    variant_artifacts = {}
    for variant_name, variant_df, label_names in variants:
        with StepTimer("train variant=%s", variant_name):
            result = _train_one_variant(
                variant_name,
                variant_df,
                label_names,
                smoke_probes=smoke_probes,
                smoke_expected=smoke_expected,
                smoke_language=smoke_language,
                language=lang,
            )
        variant_records.append({k: v for k, v in result.items() if k != "best_model"})
        all_model_results.update(result.get("models", {}))
        if "best_model" in result and result.get("best_model") is not None:
            trained_models[result["best_name"]] = result["best_model"]
            label_maps[result["best_name"]] = label_names
            variant_dir = paths.candidates_dir
            variant_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = variant_dir / f"{result['best_name']}.pkl"
            with StepTimer("save artifact %s", artifact_path.name):
                joblib.dump(result["best_model"], artifact_path)
            variant_artifacts[variant_name] = {
                "model_name": result["best_name"],
                "model_path": str(artifact_path),
                "label_names": {str(k): v for k, v in label_names.items()},
                "f1_macro": result["best_result"]["f1_macro"],
                "accuracy": result["best_result"]["accuracy"],
                "threshold": result["best_result"].get("threshold"),
                "base_model": result["best_result"].get("base_model"),
            }
        elif "error" in result:
            logger.warning("VARIANT_SKIPPED | %s | %s", variant_name, result["error"])

    valid = {k: v for k, v in all_model_results.items() if "error" not in v}
    if not valid:
        logger.error("TRAIN_VARIANTS | all variants failed")
        return {"status": "failed", "reason": "all_variants_failed"}

    deployable_valid = {k: valid[k] for k in trained_models if k in valid}
    if not deployable_valid:
        logger.error("TRAIN_VARIANTS | no deployable model retained")
        return {"status": "failed", "reason": "no_deployable_model"}

    best_by_raw_val = max(deployable_valid, key=lambda k: deployable_valid[k].get("val", {}).get("f1_macro", deployable_valid[k]["f1_macro"]))
    best_name = _select_global_best(deployable_valid)
    best = valid[best_name]
    best_label_names = label_maps[best_name]
    selection_audit = _write_run_audit(
        run_dir=run_dir,
        valid=valid,
        selected_name=best_name,
        raw_val_name=best_by_raw_val,
        log_path=log_path,
    )

    if deploy_best:
        with StepTimer("save deployable best model"):
            joblib.dump(trained_models[best_name], paths.best_model_pkl)
    with StepTimer("write smoke predictions"):
        smoke_predictions_path = _write_smoke_predictions(
            trained_models[best_name],
            best_label_names,
            run_dir,
            probes=smoke_probes,
            expected=smoke_expected,
            language=smoke_language,
        )
    best_meta = {
        "name": best_name,
        "backend": "sklearn_text",
        "input_type": "text_clean",
        "model_path": str(paths.best_model_pkl) if deploy_best else None,
        "label_names": {str(k): v for k, v in best_label_names.items()},
        "trained_by": "variant_trainer",
        "model_family": "tfidf",
        "language": lang,
        "preprocessing_pipeline": preprocessing_pipeline,
        "selection_metric": "val.f1_macro",
        "selection_policy": SELECTION_POLICY,
        "val_f1_macro": best.get("val", {}).get("f1_macro"),
        "test_f1_macro": best["f1_macro"],
        "test_accuracy": best["accuracy"],
        "smoke": best.get("smoke"),
        "threshold": best.get("threshold"),
        "base_model": best.get("base_model"),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "selection_audit": str(run_dir / "selection_audit.json"),
        "smoke_predictions": str(smoke_predictions_path),
        "data_source": source,
        "deployed": deploy_best,
    }
    if deploy_best:
        paths.meta_json.write_text(json.dumps(best_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "data_source": source,
        "sample_count": len(df),
        "embedding": "tfidf_word_char_ngrams",
        "preprocessing_pipeline": preprocessing_pipeline,
        "feature_dim": "sparse_text",
        "balance_method": "class_weight='balanced'",
        "label_quality": {
            "method": "confident_learning_cleanlab",
            "issue_count": len(issue_row_ids),
            "issue_csv": str(paths.label_issues_csv) if issue_row_ids else None,
        },
        "model_family": "tfidf",
        "language": lang,
        "vietnamese_text_scan": text_scan,
        "run_dir": str(run_dir),
        "selection_audit": selection_audit,
        "smoke_predictions": str(smoke_predictions_path),
        "variants": variant_records,
        "variant_artifacts": variant_artifacts,
        "models": all_model_results,
        "best_model": {
            "name": best_name,
            "type": best["type"],
            "variant": best["variant"],
            "val_f1_macro": best.get("val", {}).get("f1_macro"),
            "f1_macro": best["f1_macro"],
            "accuracy": best["accuracy"],
            "threshold": best.get("threshold"),
            "base_model": best.get("base_model"),
            "raw_validation_best": best_by_raw_val,
        },
        "best_model_artifact": best_meta,
        "duration_seconds": round(time.perf_counter() - run_started, 2),
        "log_file": str(log_path),
    }
    with StepTimer("persist experiment records"):
        save_experiment(record)
        _save_training_results(record)

    summary_rows = []
    for name, result in sorted(valid.items(), key=lambda item: item[1]["f1_macro"], reverse=True):
        summary_rows.append(_result_row(name, result))
    with StepTimer("write variant summary csv"):
        pd.DataFrame(summary_rows).to_csv(paths.leaderboard_csv, index=False, encoding="utf-8-sig")

    logger.info(
        "TRAIN_VARIANTS_DONE | best=%s val_f1=%s test_f1=%s acc=%s duration=%.2fs",
        best_name,
        best.get("val", {}).get("f1_macro"),
        best["f1_macro"],
        best["accuracy"],
        time.perf_counter() - run_started,
    )

    return {"status": "success", **record}
