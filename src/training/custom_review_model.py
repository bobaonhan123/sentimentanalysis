"""Project-specific Vietnamese company review sentiment model (VNReviewFusion).

Designed for weak-labeled Glassdoor-style reviews where:
- Most body text lives in `cons`, not `pros`
- Star rating and ABSA lexicon signals disagree on borderline cases
- Informal Vietnamese workplace slang benefits from field-aware + cue features

Architecture:
  full_text TF-IDF (word+char) + cons_char TF-IDF + pros_char TF-IDF
  + Vietnamese polarity cues + structured metadata (rating, ABSA, field stats)
  -> linear head (LogReg / LinearSVC) with optional rule guard
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src.training.labeling import _absa_score, _keyword_score, english_keyword_score

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "text_clean",
    "text_clean_cons",
    "text_clean_pros",
    "text_clean_title",
    "rating",
    "label_source",
)


class ReviewMetadataTransformer(BaseEstimator, TransformerMixin):
    """Numeric signals derived from review structure and weak-label confidence."""

    def __init__(self, language: str = "vi"):
        self.language = language

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        rows = []
        for row in X.itertuples(index=False):
            title = getattr(row, "text_clean_title", "") or ""
            pros = getattr(row, "text_clean_pros", "") or ""
            cons = getattr(row, "text_clean_cons", "") or ""
            full = getattr(row, "text_clean", "") or ""
            rating = float(getattr(row, "rating", 3.0) or 3.0)
            label_source = str(getattr(row, "label_source", "") or "")

            if self.language == "en":
                kw = english_keyword_score(full)
                absa = english_keyword_score(" ".join([title, pros, cons]))
            else:
                suppress_sarcasm = rating <= 3.0
                absa = _absa_score(title, pros, cons, "", suppress_sarcasm=suppress_sarcasm)
                kw = _keyword_score(title, pros, cons, "", suppress_sarcasm=suppress_sarcasm)
            cons_len = len(str(cons).split())
            pros_len = len(str(pros).split())
            ratio = pros_len / max(cons_len, 1)
            t = full.lower()
            if self.language == "en":
                contrast = t.count("but") + t.count("however")
            else:
                contrast = t.count("nhưng") + t.count("tuy_nhiên") + t.count("tuy nhiên")
            weak_override = 1.0 if "absa" in label_source or "override" in label_source else 0.0
            rows.append([
                (rating - 3.0) / 2.0,
                np.log1p(cons_len),
                np.log1p(pros_len),
                ratio,
                absa,
                kw,
                absa + kw,
                float(bool(str(pros).strip())),
                float(bool(str(cons).strip())),
                contrast,
                weak_override,
                float(rating <= 2.0),
                float(rating >= 4.0),
            ])
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


class VNReviewFusionEstimator(BaseEstimator, ClassifierMixin):
    """Field-aware fusion classifier for workplace reviews (VI or EN)."""

    def __init__(
        self,
        *,
        clf_kind: Literal["logreg", "svc"] = "logreg",
        c: float = 1.5,
        use_rating: bool = True,
        use_guard: bool = False,
        max_features: int = 35_000,
        language: str = "vi",
    ):
        self.clf_kind = clf_kind
        self.c = c
        self.use_rating = use_rating
        self.use_guard = use_guard
        self.max_features = max_features
        self.language = language

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("VNReviewFusionEstimator expects a pandas DataFrame with review fields")

    def _build_vectorizers(self) -> dict:
        half = max(10_000, self.max_features // 2)
        third = max(8_000, self.max_features // 3)
        return {
            "full": FeatureUnion([
                ("word", TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=half,
                    sublinear_tf=True,
                )),
                ("char", TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.95,
                    max_features=half,
                    sublinear_tf=True,
                )),
            ]),
            "cons": TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 6),
                min_df=1,
                max_df=0.95,
                max_features=third,
                sublinear_tf=True,
            ),
            "pros": TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 6),
                min_df=1,
                max_df=0.95,
                max_features=max(5_000, third // 2),
                sublinear_tf=True,
            ),
        }

    def _stack_features(self, df: pd.DataFrame, *, fit: bool) -> sparse.csr_matrix:
        from src.training.variant_trainer import EnglishCueTransformer, VietnameseCueTransformer

        cue_cls = EnglishCueTransformer if self.language == "en" else VietnameseCueTransformer
        blocks = []
        for key, column in (
            ("full", "text_clean"),
            ("cons", "text_clean_cons"),
            ("pros", "text_clean_pros"),
        ):
            texts = df[column].fillna("").astype(str).to_numpy()
            if fit:
                if not any(str(t).strip() for t in texts):
                    continue
            elif key not in self.vectorizers_:
                continue
            vec = self.vectorizers_[key]
            if fit:
                blocks.append(vec.fit_transform(texts))
            else:
                blocks.append(vec.transform(texts))

        if fit:
            self.cue_transformer_ = cue_cls()
            cue_block = self.cue_transformer_.fit_transform(df["text_clean"].fillna("").astype(str))
        else:
            cue_block = self.cue_transformer_.transform(df["text_clean"].fillna("").astype(str))
        blocks.append(cue_block)

        if self.use_rating:
            if fit:
                self.meta_transformer_ = ReviewMetadataTransformer(language=self.language)
                meta_block = self.meta_transformer_.fit_transform(df)
            else:
                meta_block = self.meta_transformer_.transform(df)
            blocks.append(meta_block)

        return sparse.hstack(blocks).tocsr()

    def _build_classifier(self):
        if self.clf_kind == "svc":
            return CalibratedClassifierCV(
                LinearSVC(max_iter=4000, class_weight="balanced", random_state=42, C=self.c),
                cv=3,
            )
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
            C=self.c,
            solver="lbfgs",
        )

    def fit(self, X, y):
        df = self._as_frame(X)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"VNReviewFusion missing columns: {missing}")

        self.vectorizers_ = self._build_vectorizers()
        features = self._stack_features(df, fit=True)
        self.base_clf_ = self._build_classifier()
        self.base_clf_.fit(features, y)
        self.estimator_ = self.base_clf_
        self.classes_ = np.asarray(sorted(set(y)))
        return self

    def _apply_rule_guard(self, df: pd.DataFrame, base_pred: np.ndarray) -> np.ndarray:
        if self.language != "vi":
            return base_pred
        from src.training.variant_trainer import VietnamesePolarityGuardClassifier

        guard = VietnamesePolarityGuardClassifier(
            self.base_clf_,
            negative_label=0,
            positive_label=1,
            margin=2.0,
        )
        out = np.asarray(base_pred).copy()
        texts = df["text_clean"].fillna("").astype(str).to_numpy()
        for idx, text in enumerate(texts):
            score = guard._rule_score(text)
            if score <= -guard.margin:
                out[idx] = 0
            elif score >= guard.margin + 1:
                out[idx] = 1
        return out

    def predict_proba(self, X):
        df = self._as_frame(X)
        features = self._stack_features(df, fit=False)
        return self.base_clf_.predict_proba(features)

    def predict(self, X):
        df = self._as_frame(X)
        features = self._stack_features(df, fit=False)
        base_pred = self.base_clf_.predict(features)
        if self.use_guard:
            return self._apply_rule_guard(df, base_pred)
        return base_pred


def custom_review_candidates(*, large_scale: bool = False, language: str = "vi") -> dict[str, VNReviewFusionEstimator]:
    """Named project models to compare against generic TF-IDF / FastText baselines."""
    max_features = 28_000 if large_scale else 40_000
    prefix = "Custom_ENReviewFusion" if language == "en" else "Custom_VNReviewFusion"
    return {
        f"{prefix}_LogReg": VNReviewFusionEstimator(
            clf_kind="logreg",
            c=1.5,
            use_rating=True,
            use_guard=False,
            max_features=max_features,
            language=language,
        ),
        f"{prefix}_LogReg_Guard": VNReviewFusionEstimator(
            clf_kind="logreg",
            c=1.5,
            use_rating=True,
            use_guard=language == "vi",
            max_features=max_features,
            language=language,
        ),
        f"{prefix}_LinearSVC": VNReviewFusionEstimator(
            clf_kind="svc",
            c=0.8,
            use_rating=True,
            use_guard=False,
            max_features=max_features,
            language=language,
        ),
        f"{prefix}_NoRating_LogReg": VNReviewFusionEstimator(
            clf_kind="logreg",
            c=2.0,
            use_rating=False,
            use_guard=False,
            max_features=max_features,
            language=language,
        ),
    }


def has_review_fields(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in REQUIRED_COLUMNS)


def _smoke_probe_frame(probes: list[str], *, language: str = "vi") -> pd.DataFrame:
    from src.preprocessing.processor import preprocess, preprocess_english

    rows = []
    for text in probes:
        clean = (
            preprocess_english(text, remove_sw=True)
            if language == "en"
            else preprocess(text, use_tokenizer=True, remove_sw=True)
        )
        rows.append({
            "text_clean": clean,
            "text_clean_cons": clean,
            "text_clean_pros": "",
            "text_clean_title": "",
            "rating": 3.0,
            "label_source": "smoke_probe",
        })
    return pd.DataFrame(rows)


def _tune_binary_threshold_df(model, df_val: pd.DataFrame, y_val, label_names: dict[int, str]):
    from src.training.variant_trainer import ProbabilityThresholdClassifier, _evaluate

    labels = sorted(label_names)
    if len(labels) != 2 or not hasattr(model, "predict_proba"):
        return None
    negative_label, positive_label = int(labels[0]), int(labels[1])
    classes = np.array(getattr(model, "classes_", labels))
    positive_idx = np.where(classes == positive_label)[0]
    if not len(positive_idx):
        return None
    positive_proba = model.predict_proba(df_val)[:, positive_idx[0]]
    best_threshold = 0.5
    best_pred = np.where(positive_proba >= best_threshold, positive_label, negative_label)
    best_metrics = _evaluate(y_val, best_pred, label_names)
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = np.where(positive_proba >= float(threshold), positive_label, negative_label)
        metrics = _evaluate(y_val, pred, label_names)
        if metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    tuned = ProbabilityThresholdClassifier(
        estimator=model,
        threshold=best_threshold,
        negative_label=negative_label,
        positive_label=positive_label,
    )
    return tuned, {"threshold": best_threshold, "val": best_metrics}


def _attach_custom_smoke_result(
    result: dict,
    model,
    label_names: dict[int, str],
    model_name: str,
    *,
    probes: list[str] | None = None,
    expected: list[int] | None = None,
    language: str = "vi",
) -> None:
    from src.training.variant_trainer import SMOKE_EXPECTED, SMOKE_PROBES

    probes = probes or SMOKE_PROBES
    expected = expected or SMOKE_EXPECTED
    frame = _smoke_probe_frame(probes, language=language)
    predictions = []
    correct = 0
    for idx, expected_label in enumerate(expected):
        pred = int(model.predict(frame.iloc[[idx]])[0])
        passed = pred == int(expected_label)
        correct += int(passed)
        predictions.append({
            "text": probes[idx],
            "expected": int(expected_label),
            "prediction": pred,
            "passed": passed,
        })
    result["smoke"] = {
        "passed": correct == len(predictions),
        "correct": correct,
        "total": len(predictions),
        "predictions": predictions,
    }
    logger.info(
        "SMOKE_CANDIDATE | %s | passed=%s correct=%s/%s",
        model_name,
        result["smoke"]["passed"],
        correct,
        len(predictions),
    )


def train_custom_review_candidates(
    variant_name: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_names: dict[int, str],
    *,
    large_scale: bool = False,
    smoke_probes: list[str] | None = None,
    smoke_expected: list[int] | None = None,
    smoke_language: str = "vi",
    language: str = "vi",
    tune_threshold_fn=None,
    evaluate_fn=None,
    attach_smoke_fn=None,
) -> tuple[dict[str, dict], dict[str, VNReviewFusionEstimator]]:
    """Fit all VNReviewFusion variants; returns (results, trained_models)."""
    from src.training.variant_trainer import _evaluate

    evaluate = evaluate_fn or _evaluate
    tune_threshold = tune_threshold_fn or _tune_binary_threshold_df
    attach_smoke = attach_smoke_fn or _attach_custom_smoke_result

    results: dict[str, dict] = {}
    trained: dict[str, VNReviewFusionEstimator] = {}
    y_train = df_train["sentiment"].to_numpy()
    y_val = df_val["sentiment"].to_numpy()
    y_test = df_test["sentiment"].to_numpy()

    for model_name, model in custom_review_candidates(large_scale=large_scale, language=language).items():
        full_name = f"{variant_name}__{model_name}"
        try:
            logger.info("Training project model %s", full_name)
            model.fit(df_train, y_train)
            val_metrics = evaluate(y_val, model.predict(df_val), label_names)
            test_metrics = evaluate(y_test, model.predict(df_test), label_names)
            fusion_prefix = "ENReviewFusion" if language == "en" else "VNReviewFusion"
            cue_label = "english_cues" if language == "en" else "vi_cues"
            results[full_name] = {
                "type": f"custom_{language}_review_fusion",
                "variant": variant_name,
                "features": f"field_tfidf+{cue_label}+rating_metadata",
                "model_family": fusion_prefix,
                "balance": "class_weight",
                "postprocess": "vietnamese_rule_guard" if model.use_guard else None,
                "label_names": {str(k): v for k, v in label_names.items()},
                "val": val_metrics,
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
                "selection_metric": "val.f1_macro",
                "val_f1_macro": val_metrics["f1_macro"],
            }
            attach_smoke(
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
            tuned = tune_threshold(model, df_val, y_val, label_names)
            if tuned:
                tuned_model, tuning = tuned
                tuned_name = f"{full_name}_ThresholdTuned_t{tuning['threshold']:.2f}"
                tuned_test = evaluate(y_test, tuned_model.predict(df_test), label_names)
                results[tuned_name] = {
                    "type": "custom_vn_review_fusion_threshold_tuned",
                    "variant": variant_name,
                    "features": results[full_name]["features"],
                    "balance": "class_weight",
                    "postprocess": results[full_name]["postprocess"],
                    "label_names": results[full_name]["label_names"],
                    "val": tuning["val"],
                    "test": tuned_test,
                    "f1_macro": tuned_test["f1_macro"],
                    "accuracy": tuned_test["accuracy"],
                    "val_f1_macro": tuning["val"]["f1_macro"],
                    "threshold": tuning["threshold"],
                    "base_model": full_name,
                }
                attach_smoke(
                    results[tuned_name],
                    tuned_model,
                    label_names,
                    tuned_name,
                    probes=smoke_probes,
                    expected=smoke_expected,
                    language=smoke_language,
                )
                trained[tuned_name] = tuned_model
        except Exception as exc:
            logger.exception("Project model %s failed", full_name)
            results[full_name] = {"type": "custom_vn_review_fusion", "variant": variant_name, "error": str(exc)}

    return results, trained
