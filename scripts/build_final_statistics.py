#!/usr/bin/env python3
"""Build final cross-language model statistics (one row per algorithm/strategy)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts.paths import STATISTICS_DIR, fasttext_paths, phobert_binary_results_path, tfidf_paths

METRIC_COLUMNS = [
    "n_samples",
    "val_f1_macro",
    "test_f1_macro",
    "test_accuracy",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "negative_f1",
    "non_negative_f1",
    "negative_recall",
    "negative_f2",
    "non_positive_f1",
    "positive_f1",
    "non_positive_recall",
    "non_positive_f2",
    "smoke_passed",
    "threshold",
    "best_strategy",
    "source_run_id",
    "source_model_name",
    "pipeline",
]

RESAMPLING_TOKENS = ("RandomOverSampler", "RandomUnderSampler", "SMOTE")


def _latest_run_dir(family_paths) -> Path | None:
    runs_dir = family_paths.runs_dir
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    def sort_key(p: Path) -> tuple[int, float]:
        leaderboard = p / "leaderboard.csv"
        has_board = leaderboard.exists() and leaderboard.stat().st_size > 0
        return (1 if has_board else 0, p.stat().st_mtime)

    return max(candidates, key=sort_key)


def _load_leaderboard(family_paths, lang: str, pipeline: str) -> tuple[pd.DataFrame, str | None]:
    run_dir = _latest_run_dir(family_paths)
    if run_dir is None:
        return pd.DataFrame(), None
    leaderboard = run_dir / "leaderboard.csv"
    if not leaderboard.exists():
        return pd.DataFrame(), run_dir.name
    df = pd.read_csv(leaderboard)
    df["language"] = lang
    df["run_id"] = run_dir.name
    df["pipeline"] = pipeline
    return df, run_dir.name


def _model_key(full_name: str) -> str:
    """Group hyperparameter / threshold / resampling variants under one model row."""
    model_part = full_name.split("__", 1)[-1] if "__" in full_name else full_name
    model_part = re.sub(r"_ThresholdTuned_t[\d.]+$", "", model_part)
    for token in RESAMPLING_TOKENS:
        model_part = model_part.replace(f"_{token}", "")
    if not model_part.startswith("Custom_VNReviewFusion"):
        model_part = re.sub(r"_C[\d.]+$", "", model_part)
        model_part = re.sub(r"_a[\d.]+$", "", model_part)
    model_part = re.sub(r"_char36$", "_char36", model_part)
    return model_part


def _algorithm_name(model_key: str) -> str:
    """Human-readable model name: FeatureBackend_Classifier (max ~4 words)."""
    if model_key in {"PhoBERT", "PhoBERT-base-v2"}:
        return "PhoBERT"
    if model_key in {"DistilBERT", "DistilBERT-base-uncased"}:
        return "DistilBERT"

    key = model_key
    if key.startswith("Custom_VNReviewFusion_"):
        return f"VNReviewFusion_{key.replace('Custom_VNReviewFusion_', '', 1)}"
    if key.startswith("Custom_ENReviewFusion_"):
        return f"ENReviewFusion_{key.replace('Custom_ENReviewFusion_', '', 1)}"
    if key.startswith("FastText_"):
        return key
    if "VietnameseRuleGuard" in key or "RuleGuard" in key:
        base = _algorithm_name(re.sub(r"_VietnameseRuleGuard$", "", key))
        return f"{base}_RuleGuard"
    if "WordCharCue" in key:
        for clf in ("LinearSVC", "LogisticRegression", "ComplementNB", "MLP"):
            if clf in key:
                return f"TFIDF_WordCharCue_{clf}"
    if key.startswith("TFIDF_"):
        for clf in ("LinearSVC", "LogisticRegression", "ComplementNB", "MLP"):
            if clf in key:
                return f"TFIDF_WordChar_{clf}"
    if key == "TFIDF_WordCharCue_Cleanlab":
        return "TFIDF_WordCharCue_Cleanlab"
    return model_key


def _feature_backend(model_key: str, pipeline: str) -> str:
    key = model_key.lower()
    if pipeline == "phobert" or model_key in {"PhoBERT", "DistilBERT"}:
        return "transformer"
    if pipeline == "fasttext" or model_key.startswith("FastText_"):
        return "fasttext_305d"
    if "ENReviewFusion" in model_key or "enreviewfusion" in model_key.lower():
        return "field_tfidf+en_cues+metadata"
    if "VNReviewFusion" in model_key or "vnreviewfusion" in model_key.lower():
        return "field_tfidf+vi_cues+metadata"
    if "wordcharcue" in key:
        return "tfidf_word_char+cues"
    return "tfidf_word_char"


def _infer_model_family(model_key: str, pipeline: str) -> str:
    """Granular family name — one row per distinct algorithm in statistics."""
    if pipeline == "phobert":
        if model_key in {"DistilBERT", "DistilBERT-base-uncased"}:
            return "DistilBERT"
        return "PhoBERT"
    return _algorithm_name(model_key)


def _strategy_label(row: pd.Series) -> str:
    parts = []
    variant = str(row.get("variant", ""))
    if "cleanlab" in variant:
        parts.append("label_filtered")
    elif "positive_vs_non_positive" in variant or "negative_vs_non_negative" in variant:
        parts.append("weak_label_full")
    balance = str(row.get("balance", ""))
    if balance == "resampling_after_tfidf":
        for token in RESAMPLING_TOKENS:
            if token in str(row.get("model", "")):
                parts.append(token)
                break
    elif balance in {"class_weight", "early_stopping_validation", "soft_vote"}:
        parts.append(balance)
    if pd.notna(row.get("threshold")) and str(row.get("threshold", "")).strip():
        parts.append("threshold_tuned")
    else:
        parts.append("default_threshold")
    if str(row.get("features", "")):
        parts.append(str(row.get("features")))
    return " + ".join(parts)


def _best_per_model(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    working["model_key"] = working["model"].map(_model_key)
    working["model_family"] = working.apply(
        lambda row: _infer_model_family(row["model_key"], str(row.get("pipeline", "tfidf"))),
        axis=1,
    )
    working["algorithm"] = working["model_key"].map(_algorithm_name)
    working["feature_backend"] = working.apply(
        lambda row: _feature_backend(row["model_key"], str(row.get("pipeline", "tfidf"))),
        axis=1,
    )
    working["best_strategy"] = working.apply(_strategy_label, axis=1)
    working = working.sort_values(["model_key", "f1_macro"], ascending=[True, False])
    return working.groupby("model_key", as_index=False).first()


def _prefix_metrics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model_key", "model_family", "algorithm", "feature_backend"])
    rename_map = {
        "f1_macro": f"{prefix}test_f1_macro",
        "accuracy": f"{prefix}test_accuracy",
        "val_f1_macro": f"{prefix}val_f1_macro",
        "f1_weighted": f"{prefix}f1_weighted",
        "precision_macro": f"{prefix}precision_macro",
        "recall_macro": f"{prefix}recall_macro",
        "negative_f1": f"{prefix}negative_f1",
        "non_negative_f1": f"{prefix}non_negative_f1",
        "negative_recall": f"{prefix}negative_recall",
        "negative_f2": f"{prefix}negative_f2",
        "non_positive_f1": f"{prefix}non_positive_f1",
        "positive_f1": f"{prefix}positive_f1",
        "non_positive_recall": f"{prefix}non_positive_recall",
        "non_positive_f2": f"{prefix}non_positive_f2",
        "smoke_passed": f"{prefix}smoke_passed",
        "threshold": f"{prefix}threshold",
        "best_strategy": f"{prefix}best_strategy",
        "run_id": f"{prefix}source_run_id",
        "model": f"{prefix}source_model_name",
        "pipeline": f"{prefix}pipeline",
    }
    out = df.rename(columns=rename_map)
    out[f"{prefix}n_samples"] = None
    keep = ["model_key", "model_family", "algorithm", "feature_backend"] + [f"{prefix}{col}" for col in METRIC_COLUMNS]
    for col in keep:
        if col not in out.columns:
            out[col] = None
    return out[keep]


def _custom_project_row(vi_df: pd.DataFrame) -> pd.DataFrame:
    """Best row among project-specific VI cues / cleanlab / rule-guard strategies."""
    if vi_df.empty:
        return pd.DataFrame()
    mask = (
        vi_df["model"].astype(str).str.contains("WordCharCue|RuleGuard|rule_guard", case=False, regex=True)
        | vi_df["variant"].astype(str).str.contains("cleanlab", case=False, regex=True)
    )
    subset = vi_df[mask].copy()
    if subset.empty:
        return pd.DataFrame()
    best = subset.sort_values(["f1_macro", "val_f1_macro"], ascending=[False, False]).iloc[0]
    return pd.DataFrame([{
        "model_key": "TFIDF_WordCharCue_Cleanlab",
        "model_family": "TFIDF_WordCharCue_Cleanlab",
        "algorithm": "TFIDF_WordCharCue_Cleanlab",
        "feature_backend": "tfidf_word_char+cues",
        "vi_n_samples": None,
        "vi_val_f1_macro": best.get("val_f1_macro"),
        "vi_test_f1_macro": best.get("f1_macro"),
        "vi_test_accuracy": best.get("accuracy"),
        "vi_f1_weighted": best.get("f1_weighted"),
        "vi_precision_macro": best.get("precision_macro"),
        "vi_recall_macro": best.get("recall_macro"),
        "vi_non_positive_f1": best.get("non_positive_f1"),
        "vi_positive_f1": best.get("positive_f1"),
        "vi_non_positive_recall": best.get("non_positive_recall"),
        "vi_non_positive_f2": best.get("non_positive_f2"),
        "vi_smoke_passed": best.get("smoke_passed"),
        "vi_threshold": best.get("threshold"),
        "vi_best_strategy": _strategy_label(best),
        "vi_source_run_id": best.get("run_id"),
        "vi_source_model_name": best.get("model"),
        "vi_pipeline": best.get("pipeline", "tfidf"),
    }])


def _transformer_row(lang: str = "vi") -> pd.DataFrame:
    results_path = phobert_binary_results_path(lang)
    if not results_path.exists():
        return pd.DataFrame()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    test = payload.get("threshold_test") or payload.get("test") or {}
    val = payload.get("val") or {}
    report = test.get("classification_report") or {}
    model_key = payload.get("model_key") or ("DistilBERT" if lang == "en" else "PhoBERT")
    model_family = payload.get("model_family") or model_key
    return pd.DataFrame([{
        "model_key": model_key,
        "model_family": model_family,
        "algorithm": model_key,
        "feature_backend": "transformer",
        f"{lang}_n_samples": payload.get("sample_count"),
        f"{lang}_val_f1_macro": val.get("f1_macro"),
        f"{lang}_test_f1_macro": test.get("f1_macro"),
        f"{lang}_test_accuracy": test.get("accuracy"),
        f"{lang}_f1_weighted": test.get("f1_weighted"),
        f"{lang}_precision_macro": test.get("precision_macro"),
        f"{lang}_recall_macro": test.get("recall_macro"),
        f"{lang}_non_positive_f1": (report.get("non_positive") or {}).get("f1-score"),
        f"{lang}_positive_f1": (report.get("positive") or {}).get("f1-score"),
        f"{lang}_non_positive_recall": (report.get("non_positive") or {}).get("recall"),
        f"{lang}_smoke_passed": None,
        f"{lang}_threshold": payload.get("threshold"),
        f"{lang}_best_strategy": f"finetune_{payload.get('epochs', 1)}epoch_threshold",
        f"{lang}_source_run_id": payload.get("run_id"),
        f"{lang}_source_model_name": payload.get("model_name"),
        f"{lang}_pipeline": "phobert",
    }])


def _detect_sample_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        from src.training.labeling import GLASSDOOR_PROCESSED_PARQUET, load_labeled_data

        counts["vi"] = len(load_labeled_data())
        if GLASSDOOR_PROCESSED_PARQUET.exists():
            counts["en"] = len(pd.read_parquet(GLASSDOOR_PROCESSED_PARQUET, columns=["text"]))
    except Exception:
        pass
    return counts


def _merge_language_frames(frames: list[pd.DataFrame], prefix: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["model_key", "model_family", "algorithm", "feature_backend"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=["model_key", "model_family", "algorithm", "feature_backend"])
    best = _best_per_model(combined)
    return _prefix_metrics(best, prefix)


def _best_per_family(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per model_family — the best strategy by test F1."""
    if df.empty or "model_family" not in df.columns:
        return df
    working = df.copy()
    score_cols = [c for c in ("vi_test_f1_macro", "en_test_f1_macro") if c in working.columns]
    if score_cols:
        working["_family_score"] = working[score_cols].max(axis=1)
        sort_col = "_family_score"
    else:
        sort_col = "model_key"
    working = working.sort_values(["model_family", sort_col], ascending=[True, False], na_position="last")
    out = working.groupby("model_family", as_index=False).first()
    return out.drop(columns=["_family_score"], errors="ignore")


def build_final_statistics(
    *,
    vi_run_id: str | None = None,
    en_run_id: str | None = None,
    sample_counts: dict[str, int] | None = None,
    per_family_only: bool = False,
) -> dict:
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)

    vi_tfidf_df, vi_tfidf_run = _load_leaderboard(tfidf_paths("vi"), "vi", "tfidf")
    en_tfidf_df, en_tfidf_run = _load_leaderboard(tfidf_paths("en"), "en", "tfidf")
    vi_ft_df, vi_ft_run = _load_leaderboard(fasttext_paths("vi"), "vi", "fasttext")
    en_ft_df, en_ft_run = _load_leaderboard(fasttext_paths("en"), "en", "fasttext")

    if vi_run_id:
        vi_path = tfidf_paths("vi").run_dir(vi_run_id) / "leaderboard.csv"
        if vi_path.exists():
            vi_tfidf_df = pd.read_csv(vi_path)
            vi_tfidf_df["language"] = "vi"
            vi_tfidf_df["run_id"] = vi_run_id
            vi_tfidf_df["pipeline"] = "tfidf"
    if en_run_id:
        en_path = tfidf_paths("en").run_dir(en_run_id) / "leaderboard.csv"
        if en_path.exists():
            en_tfidf_df = pd.read_csv(en_path)
            en_tfidf_df["language"] = "en"
            en_tfidf_df["run_id"] = en_run_id
            en_tfidf_df["pipeline"] = "tfidf"

    vi_prefixed = _merge_language_frames([vi_tfidf_df, vi_ft_df], "vi_")
    en_prefixed = _merge_language_frames([en_tfidf_df, en_ft_df], "en_")

    if sample_counts is None:
        sample_counts = _detect_sample_counts()
    if sample_counts:
        if "vi" in sample_counts and not vi_prefixed.empty:
            vi_prefixed["vi_n_samples"] = sample_counts["vi"]
        if "en" in sample_counts and not en_prefixed.empty:
            en_prefixed["en_n_samples"] = sample_counts["en"]

    merged = pd.merge(
        vi_prefixed,
        en_prefixed,
        on=["model_key", "model_family", "algorithm", "feature_backend"],
        how="outer",
    )

    custom_row = _custom_project_row(pd.concat([vi_tfidf_df, vi_ft_df], ignore_index=True, sort=False))
    transformer_rows = pd.concat(
        [_transformer_row("vi"), _transformer_row("en")],
        ignore_index=True,
        sort=False,
    )
    extras = [df for df in (custom_row, transformer_rows) if not df.empty]
    if extras:
        merged = pd.concat([merged] + extras, ignore_index=True, sort=False)
        merged = merged.drop_duplicates(subset=["model_key"], keep="first")

    if per_family_only:
        merged = _best_per_family(merged)

    sort_cols = ["model_family", "model_key"]
    for col in ("vi_test_f1_macro", "en_test_f1_macro"):
        if col in merged.columns:
            sort_cols.append(col)
    merged = merged.sort_values(
        by=sort_cols,
        ascending=[True, True] + [False] * (len(sort_cols) - 2),
        na_position="last",
    ).reset_index(drop=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = STATISTICS_DIR / f"model_statistics_{stamp}.csv"
    json_path = STATISTICS_DIR / f"model_statistics_{stamp}.json"
    latest_csv = STATISTICS_DIR / "model_statistics_latest.csv"
    latest_json = STATISTICS_DIR / "model_statistics_latest.json"

    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    merged.to_csv(latest_csv, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": datetime.now().isoformat(),
        "per_family_only": per_family_only,
        "vi_tfidf_run_id": vi_run_id or vi_tfidf_run,
        "vi_fasttext_run_id": vi_ft_run,
        "en_fasttext_run_id": en_ft_run,
        "en_run_id": en_run_id or en_tfidf_run,
        "row_count": len(merged),
        "csv": str(csv_path),
        "rows": merged.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final model statistics table")
    parser.add_argument("--vi-run-id", default=None)
    parser.add_argument("--en-run-id", default=None)
    parser.add_argument(
        "--per-family-only",
        action="store_true",
        help="Collapse to one best row per model_family (legacy broad grouping)",
    )
    args = parser.parse_args()
    payload = build_final_statistics(
        vi_run_id=args.vi_run_id,
        en_run_id=args.en_run_id,
        per_family_only=args.per_family_only,
    )
    print(json.dumps({"csv": payload["csv"], "row_count": payload["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
