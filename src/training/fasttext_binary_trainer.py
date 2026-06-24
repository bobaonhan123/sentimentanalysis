"""Binary negative vs non-negative training on FastText 305-dim + sklearn/MLP models."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.artifacts.paths import fasttext_paths
from src.preprocessing.processor import preprocess, preprocess_english
from src.training.labeling import (
    BINARY_VARIANT_NAMES,
    DEFAULT_BINARY_FRAMING,
    _limit_rows,
    load_glassdoor_english_data,
    load_labeled_data,
)
from src.training.balancing import balance_with_class_weight
from src.training.trainer import (
    _build_features,
    _ensure_fasttext_model,
    _evaluate,
    _fit_classifier,
    _get_ml_models,
    _sample_weights,
)
from src.training.variant_trainer import (
    BINARY_LABEL_NAMES_MAP,
    ProbabilityThresholdClassifier,
    _attach_run_file_logger,
    _negative_vs_non_negative_variant,
    _result_row,
    _write_json,
)

logger = logging.getLogger(__name__)

BINARY_FRAMING = DEFAULT_BINARY_FRAMING


class StepTimer:
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


def _binary_evaluate(y_true, y_pred, label_names: dict[int, str] | None = None) -> dict:
    metrics = _evaluate(y_true, y_pred)
    if label_names:
        report = metrics.get("classification_report") or {}
        minority = report.get("negative") or report.get("non_positive") or report.get("0") or {}
        majority = report.get("non_negative") or report.get("positive") or report.get("1") or {}
        metrics["negative_f1"] = minority.get("f1-score")
        metrics["non_negative_f1"] = majority.get("f1-score")
        metrics["negative_recall"] = minority.get("recall")
        metrics["non_positive_f1"] = minority.get("f1-score")
        metrics["positive_f1"] = majority.get("f1-score")
        metrics["non_positive_recall"] = minority.get("recall")
        if minority.get("precision") is not None and minority.get("recall") is not None:
            p, r = float(minority["precision"]), float(minority["recall"])
            f2 = round((5 * p * r / (4 * p + r)) if (p + r) else 0.0, 4)
            metrics["negative_f2"] = f2
            metrics["non_positive_f2"] = f2
    return metrics


def _tune_binary_threshold(model, X_val, y_val, label_names: dict[int, str]):
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
    best_metrics = _binary_evaluate(y_val, best_pred, label_names)
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
        metrics = _binary_evaluate(y_val, pred, label_names)
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
    tuned = ProbabilityThresholdClassifier(
        estimator=model,
        threshold=best_threshold,
        negative_label=minority_label,
        positive_label=majority_label,
    )
    return tuned, {
        "threshold": best_threshold,
        "val": best_metrics,
        "negative_f2": round(best_score, 4),
        "non_positive_f2": round(best_score, 4),
    }


def _load_binary_frame(
    *,
    csv_path: str | None = None,
    source: str = "current",
    max_rows: int | None = None,
    english_max_rows: int = 0,
) -> tuple[pd.DataFrame, str]:
    normalized = (source or "current").strip().lower()
    if normalized in {"glassdoor-en", "glassdoor_en", "en", "english"}:
        row_limit = english_max_rows if english_max_rows else max_rows
        df = load_glassdoor_english_data(max_rows=row_limit, preprocessed=True)
        source_key = "glassdoor_en"
    else:
        df = load_labeled_data(csv_path)
        df = _limit_rows(df, max_rows)
        source_key = "current"
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient labeled data: {len(df)} rows")
    is_english = source_key == "glassdoor_en"
    if "text_clean" not in df.columns:
        df = df.copy()
        preprocess_fn = preprocess_english if is_english else (
            lambda t: preprocess(t, use_tokenizer=True, remove_sw=True)
        )
        df["text_clean"] = df["text"].apply(preprocess_fn)
        df = df[df["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
    binary = _negative_vs_non_negative_variant(df)
    return binary, source_key


def train_fasttext_binary(
    *,
    csv_path: str | None = None,
    source: str = "current",
    max_rows: int | None = None,
    english_max_rows: int = 0,
    deploy_best: bool = False,
) -> dict:
    """Train FastText embedding + sklearn/MLP models on binary sentiment."""
    lang = "en" if (source or "").strip().lower() in {"glassdoor-en", "glassdoor_en", "en", "english"} else "vi"
    paths = fasttext_paths(lang)
    paths.ensure_dirs()
    run_id = f"fasttext_binary_{source.replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = _attach_run_file_logger(run_id, paths.logs_dir)
    logger.info("RUN_DIR | %s lang=%s", run_dir, lang)

    with StepTimer("load binary labeled data source=%s", source):
        df, source_key = _load_binary_frame(
            csv_path=csv_path,
            source=source,
            max_rows=max_rows,
            english_max_rows=english_max_rows,
        )
    label_names = BINARY_LABEL_NAMES_MAP
    variant_name = BINARY_VARIANT_NAMES[BINARY_FRAMING]
    y = df["sentiment"].to_numpy()
    class_weights = balance_with_class_weight(y)

    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42, stratify=y)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=42,
        stratify=df_temp["sentiment"].to_numpy(),
    )
    y_train = df_train["sentiment"].to_numpy()
    y_val = df_val["sentiment"].to_numpy()
    y_test = df_test["sentiment"].to_numpy()

    with StepTimer("build FastText 305-dim features"):
        ft_model = _ensure_fasttext_model(lang)
        X_train = _build_features(ft_model, df_train["text_clean"].to_numpy())
        X_val = _build_features(ft_model, df_val["text_clean"].to_numpy())
        X_test = _build_features(ft_model, df_test["text_clean"].to_numpy())
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    results: dict[str, dict] = {}
    trained: dict[str, object] = {}
    train_sample_weight = _sample_weights(y_train, class_weights)

    for name, model in _get_ml_models().items():
        full_name = f"FastText_{name}"
        try:
            with StepTimer("fit/evaluate %s", full_name):
                _fit_classifier(model, X_train, y_train, sample_weight=train_sample_weight)
                val_metrics = _binary_evaluate(y_val, model.predict(X_val), label_names)
                test_metrics = _binary_evaluate(y_test, model.predict(X_test), label_names)
            results[full_name] = {
                "type": "fasttext_features",
                "variant": f"{source_key}_{variant_name}",
                "features": "fasttext_300d+handcrafted_5d",
                "balance": "class_weight",
                "label_names": {str(k): v for k, v in label_names.items()},
                "val": val_metrics,
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
            trained[full_name] = model
            logger.info(
                "RESULT | %s | val_f1=%s test_f1=%s acc=%s",
                full_name,
                val_metrics["f1_macro"],
                test_metrics["f1_macro"],
                test_metrics["accuracy"],
            )
            tuned = _tune_binary_threshold(model, X_val, y_val, label_names)
            if tuned:
                tuned_model, tuning = tuned
                tuned_name = f"{full_name}_ThresholdTuned_t{tuning['threshold']:.2f}"
                tuned_test = _binary_evaluate(y_test, tuned_model.predict(X_test), label_names)
                results[tuned_name] = {
                    "type": "fasttext_features_threshold_tuned",
                    "variant": results[full_name]["variant"],
                    "features": results[full_name]["features"],
                    "balance": "class_weight",
                    "label_names": results[full_name]["label_names"],
                    "val": tuning["val"],
                    "test": tuned_test,
                    "f1_macro": tuned_test["f1_macro"],
                    "accuracy": tuned_test["accuracy"],
                    "val_f1_macro": tuning["val"]["f1_macro"],
                    "threshold": tuning["threshold"],
                    "non_positive_f2": tuning.get("non_positive_f2"),
                    "base_model": full_name,
                }
                trained[tuned_name] = tuned_model
        except Exception as exc:
            logger.exception("%s failed", full_name)
            results[full_name] = {"type": "fasttext_features", "error": str(exc)}

    ensemble_models = {
        n: m for n, m in trained.items()
        if hasattr(m, "predict_proba") and "ThresholdTuned" not in n and "error" not in results.get(n, {})
    }
    if len(ensemble_models) >= 2:
        try:
            with StepTimer("fit/evaluate FastText_Ensemble_SoftVote"):
                ensemble = VotingClassifier(
                    estimators=[(n, m) for n, m in ensemble_models.items()],
                    voting="soft",
                )
                ensemble.fit(X_train, y_train)
                val_metrics = _binary_evaluate(y_val, ensemble.predict(X_val), label_names)
                test_metrics = _binary_evaluate(y_test, ensemble.predict(X_test), label_names)
                full_name = "FastText_Ensemble_SoftVote"
                results[full_name] = {
                    "type": "fasttext_ensemble",
                    "variant": f"{source_key}_{variant_name}",
                    "features": "fasttext_300d+handcrafted_5d",
                    "balance": "soft_vote",
                    "label_names": {str(k): v for k, v in label_names.items()},
                    "val": val_metrics,
                    "test": test_metrics,
                    "f1_macro": test_metrics["f1_macro"],
                    "accuracy": test_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "members": list(ensemble_models.keys()),
                }
                trained[full_name] = ensemble
        except Exception as exc:
            logger.exception("FastText ensemble failed: %s", exc)

    valid = {k: v for k, v in results.items() if "error" not in v}
    rows = [_result_row(name, result) for name, result in valid.items()]
    leaderboard = pd.DataFrame(rows).sort_values(
        ["val_f1_macro", "f1_macro", "accuracy"],
        ascending=[False, False, False],
        na_position="last",
    )
    leaderboard_path = run_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    paths.leaderboard_csv.write_text(leaderboard.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8-sig")

    best_name = max(valid, key=lambda k: valid[k].get("val_f1_macro", valid[k]["f1_macro"]))
    payload = {
        "status": "success",
        "run_id": run_id,
        "language": lang,
        "model_family": "fasttext",
        "source": source_key,
        "sample_count": len(df),
        "split": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "label_names": {str(k): v for k, v in label_names.items()},
        "best_name": best_name,
        "best_result": valid[best_name],
        "leaderboard_csv": str(leaderboard_path),
        "log_file": str(log_path),
        "models": results,
    }
    _write_json(run_dir / "results.json", payload)
    logger.info(
        "FASTTEXT_BINARY_DONE | best=%s val_f1=%s test_f1=%s",
        best_name,
        valid[best_name].get("val_f1_macro"),
        valid[best_name]["f1_macro"],
    )
    return payload
