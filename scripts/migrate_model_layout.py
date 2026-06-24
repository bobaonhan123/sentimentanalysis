#!/usr/bin/env python3
"""One-time migration to models/{family}/{lang}/ layout."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ANALYSIS = ROOT / "analysis"


def _move(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    print(f"move {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    shutil.move(str(src), str(dst))


def _move_glob(src_dir: Path, pattern: str, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    for path in sorted(src_dir.glob(pattern)):
        _move(path, dst_dir / path.name)


def migrate() -> None:
    tfidf_vi = MODELS / "tfidf" / "vi"
    tfidf_en = MODELS / "tfidf" / "en"
    phobert_vi = MODELS / "phobert" / "vi"
    comparisons = MODELS / "comparisons"
    fasttext_vi = MODELS / "fasttext" / "vi"

    for folder in (
        tfidf_vi / "production",
        tfidf_vi / "candidates",
        tfidf_vi / "runs",
        tfidf_vi / "logs",
        tfidf_en / "candidates",
        tfidf_en / "runs",
        tfidf_en / "logs",
        phobert_vi / "production",
        phobert_vi / "runs",
        phobert_vi / "logs",
        fasttext_vi / "production",
        comparisons,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    _move(MODELS / "best_model.pkl", tfidf_vi / "production" / "best_model.pkl")
    _move(MODELS / "best_model_meta.json", tfidf_vi / "production" / "meta.json")

    variants = MODELS / "variants"
    if variants.exists():
        for path in variants.glob("*.pkl"):
            if path.name.startswith("glassdoor_en_"):
                _move(path, tfidf_en / "candidates" / path.name)
            else:
                _move(path, tfidf_vi / "candidates" / path.name)
        if variants.exists() and not any(variants.iterdir()):
            variants.rmdir()

    runs = ANALYSIS / "runs"
    if runs.exists():
        for path in runs.iterdir():
            if not path.is_dir():
                continue
            if "glassdoor_en" in path.name:
                _move(path, tfidf_en / "runs" / path.name)
            else:
                _move(path, tfidf_vi / "runs" / path.name)
        if runs.exists() and not any(runs.iterdir()):
            runs.rmdir()

    logs = ANALYSIS / "logs"
    if logs.exists():
        for path in logs.glob("variant_run_*"):
            if "glassdoor_en" in path.name:
                _move(path, tfidf_en / "logs" / path.name)
            else:
                _move(path, tfidf_vi / "logs" / path.name)
        for path in logs.glob("phobert_binary_run_*"):
            _move(path, phobert_vi / "logs" / path.name)

    phobert_legacy = ANALYSIS / "phobert_binary_outputs"
    _move(
        phobert_legacy / "best_phobert_binary_model",
        phobert_vi / "production" / "best",
    )
    if (phobert_legacy / "phobert_binary_results.json").exists():
        _move(
            phobert_legacy / "phobert_binary_results.json",
            phobert_vi / "production" / "results.json",
        )
    if (phobert_legacy / "phobert_binary_history.csv").exists():
        _move(
            phobert_legacy / "phobert_binary_history.csv",
            phobert_vi / "production" / "history.csv",
        )
    if phobert_legacy.exists() and not any(phobert_legacy.iterdir()):
        phobert_legacy.rmdir()

    _move_glob(ANALYSIS, "glassdoor_dataset_comparison_*.json", comparisons)
    _move_glob(ANALYSIS, "glassdoor_dataset_comparison_*.csv", comparisons)
    _move(ANALYSIS / "variant_training_summary.csv", tfidf_vi / "leaderboard_latest.csv")
    _move(ANALYSIS / "label_issues_cleanlab.csv", tfidf_vi / "label_issues_cleanlab.csv")
    _move(MODELS / "cc.vi.300.bin", fasttext_vi / "cc.vi.300.bin")

    print("Migration complete.")


if __name__ == "__main__":
    migrate()
