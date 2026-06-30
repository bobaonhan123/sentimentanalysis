#!/usr/bin/env python3
"""Unified experiment pipeline: VI + EN positive/non-positive models and statistics.

Runs all model families on Vietnamese (~72k) and English Glassdoor (~838k) data:
  - TF-IDF variants (LogReg, LinearSVC, ComplementNB, MLP, ReviewFusion, cues, cleanlab, resampling)
  - FastText 305-d + sklearn/MLP (VI: cc.vi.300.bin, EN: cc.en.300.bin)
  - Transformer fine-tune (VI: PhoBERT, EN: DistilBERT)

After training, builds a statistics table with one row per distinct algorithm/strategy.

Usage:
  # Preferred CLI
  python run.py run-full-experiments [--smoke]

  # Direct script (same pipeline)
  python scripts/run_full_experiment_pipeline.py --smoke
  python scripts/run_full_experiment_pipeline.py
  python scripts/run_full_experiment_pipeline.py --skip-vi-phobert --skip-en-fasttext
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts.paths import STATISTICS_DIR
from src.training.labeling import VI_DATA_CANDIDATES, load_glassdoor_english_data, load_labeled_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_full_experiment_pipeline")

# Smoke defaults: enough rows for stratified split + cleanlab, but fast on CPU.
SMOKE_VI_ROWS = 600
SMOKE_EN_ROWS = 1500
SMOKE_TRANSFORMER_EXAMPLES = 256
SMOKE_TRANSFORMER_EPOCHS = 1


def _resolve_vi_csv(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for candidate in VI_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Vietnamese CSV not found in data/vi/raw or data_post_processing")


def _manifest_path() -> Path:
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    return STATISTICS_DIR / "full_experiment_manifest.json"


def _load_build_statistics():
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("build_final_statistics", ROOT / "scripts" / "build_final_statistics.py")
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_full_experiment_pipeline(
    *,
    vi_csv: str | None = None,
    smoke: bool = False,
    vi_max_rows: int | None = None,
    en_max_rows: int | None = None,
    skip_vi_tfidf: bool = False,
    skip_en_tfidf: bool = False,
    skip_vi_phobert: bool = False,
    skip_en_phobert: bool = False,
    skip_vi_fasttext: bool = False,
    skip_en_fasttext: bool = False,
    phobert_epochs: int | None = None,
    phobert_batch_size: int = 8,
    phobert_max_examples: int | None = None,
    deploy_vi_best: bool | None = None,
) -> dict:
    """Orchestrate all model trainers and build per-algorithm statistics."""
    started = datetime.now().isoformat()
    vi_path = _resolve_vi_csv(vi_csv)

    if smoke:
        vi_max_rows = vi_max_rows if vi_max_rows is not None else SMOKE_VI_ROWS
        en_max_rows = en_max_rows if en_max_rows is not None else SMOKE_EN_ROWS
        phobert_max_examples = (
            phobert_max_examples if phobert_max_examples is not None else SMOKE_TRANSFORMER_EXAMPLES
        )
        phobert_epochs = phobert_epochs if phobert_epochs is not None else SMOKE_TRANSFORMER_EPOCHS
        deploy_vi_best = False if deploy_vi_best is None else deploy_vi_best
        logger.info(
            "SMOKE_MODE | vi_rows=%s en_rows=%s transformer_examples=%s epochs=%s",
            vi_max_rows,
            en_max_rows,
            phobert_max_examples,
            phobert_epochs,
        )
    else:
        vi_max_rows = vi_max_rows if vi_max_rows is not None else 0
        en_max_rows = en_max_rows if en_max_rows is not None else 0
        phobert_max_examples = phobert_max_examples if phobert_max_examples is not None else 0
        phobert_epochs = phobert_epochs if phobert_epochs is not None else 3
        deploy_vi_best = True if deploy_vi_best is None else deploy_vi_best

    vi_df = load_labeled_data(vi_path)
    if vi_max_rows and vi_max_rows > 0:
        from src.training.labeling import _limit_rows

        vi_df = _limit_rows(vi_df, vi_max_rows)
    en_df = load_glassdoor_english_data(max_rows=en_max_rows or None, preprocessed=True)

    manifest = {
        "started_at": started,
        "mode": "smoke" if smoke else "full",
        "vi_csv": str(vi_path),
        "vi_rows": len(vi_df),
        "en_rows": len(en_df),
        "vi_max_rows": vi_max_rows,
        "en_max_rows": en_max_rows,
        "runs": {},
        "status": "running",
    }
    _manifest_path().write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("PIPELINE | mode=%s vi_rows=%s en_rows=%s", manifest["mode"], len(vi_df), len(en_df))

    if not skip_vi_tfidf:
        from src.training.variant_trainer import train_variants

        logger.info("Starting Vietnamese TF-IDF + custom models on %s rows", len(vi_df))
        manifest["runs"]["vi_tfidf"] = train_variants(
            csv_path=str(vi_path),
            source="current",
            max_rows=vi_max_rows or None,
            deploy_best=deploy_vi_best,
        )

    if not skip_en_tfidf:
        from src.training.variant_trainer import train_variants

        logger.info("Starting English TF-IDF + ReviewFusion on %s rows", len(en_df))
        manifest["runs"]["en_tfidf"] = train_variants(
            source="glassdoor-en",
            max_rows=None,
            english_max_rows=en_max_rows or None,
            deploy_best=False,
        )

    if not skip_vi_phobert:
        from src.training.phobert_binary_trainer import train_phobert_binary

        logger.info("Starting Vietnamese PhoBERT on up to %s examples", phobert_max_examples or "all")
        manifest["runs"]["vi_phobert"] = train_phobert_binary(
            csv_path=str(vi_path),
            language="vi",
            max_examples=phobert_max_examples or None,
            epochs=phobert_epochs,
            batch_size=phobert_batch_size,
            max_len=256 if smoke else 256,
        )

    if not skip_en_phobert:
        from src.training.phobert_binary_trainer import train_phobert_binary

        logger.info("Starting English DistilBERT on up to %s examples", phobert_max_examples or "all")
        manifest["runs"]["en_phobert"] = train_phobert_binary(
            language="en",
            max_examples=phobert_max_examples or None,
            epochs=phobert_epochs,
            batch_size=phobert_batch_size,
            max_len=256 if smoke else 256,
        )

    if not skip_vi_fasttext:
        from src.training.fasttext_binary_trainer import train_fasttext_binary

        logger.info("Starting Vietnamese FastText+sklearn/MLP on %s rows", len(vi_df))
        manifest["runs"]["vi_fasttext"] = train_fasttext_binary(
            csv_path=str(vi_path),
            source="current",
            max_rows=vi_max_rows or None,
            deploy_best=False,
        )

    if not skip_en_fasttext:
        from src.training.fasttext_binary_trainer import train_fasttext_binary

        logger.info("Starting English FastText+sklearn/MLP on %s rows", len(en_df))
        manifest["runs"]["en_fasttext"] = train_fasttext_binary(
            source="glassdoor-en",
            english_max_rows=en_max_rows or None,
            deploy_best=False,
        )

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["status"] = "success"
    _manifest_path().write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    stats_mod = _load_build_statistics()
    vi_run_id = (manifest["runs"].get("vi_tfidf") or {}).get("run_id")
    en_run_id = (manifest["runs"].get("en_tfidf") or {}).get("run_id")
    stats = stats_mod.build_final_statistics(
        vi_run_id=vi_run_id,
        en_run_id=en_run_id,
        sample_counts={"vi": len(vi_df), "en": len(en_df)},
        per_family_only=False,
    )
    manifest["statistics"] = stats
    _manifest_path().write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified VI+EN experiment pipeline")
    parser.add_argument("--vi-csv", default=None, help="Vietnamese reviews CSV path")
    parser.add_argument("--smoke", action="store_true", help="Quick run on small data samples")
    parser.add_argument("--vi-max-rows", type=int, default=None, help="Limit Vietnamese rows (0=all)")
    parser.add_argument("--en-max-rows", type=int, default=None, help="Limit English rows (0=all)")
    parser.add_argument("--skip-vi-tfidf", action="store_true")
    parser.add_argument("--skip-en-tfidf", action="store_true")
    parser.add_argument("--skip-vi-phobert", action="store_true")
    parser.add_argument("--skip-en-phobert", action="store_true")
    parser.add_argument("--skip-vi-fasttext", action="store_true")
    parser.add_argument("--skip-en-fasttext", action="store_true")
    parser.add_argument("--phobert-epochs", type=int, default=None)
    parser.add_argument("--phobert-batch-size", type=int, default=8)
    parser.add_argument("--phobert-max-examples", type=int, default=None)
    parser.add_argument("--no-deploy-vi-best", action="store_true")
    args = parser.parse_args()

    manifest = run_full_experiment_pipeline(
        vi_csv=args.vi_csv,
        smoke=args.smoke,
        vi_max_rows=args.vi_max_rows,
        en_max_rows=args.en_max_rows,
        skip_vi_tfidf=args.skip_vi_tfidf,
        skip_en_tfidf=args.skip_en_tfidf,
        skip_vi_phobert=args.skip_vi_phobert,
        skip_en_phobert=args.skip_en_phobert,
        skip_vi_fasttext=args.skip_vi_fasttext,
        skip_en_fasttext=args.skip_en_fasttext,
        phobert_epochs=args.phobert_epochs,
        phobert_batch_size=args.phobert_batch_size,
        phobert_max_examples=args.phobert_max_examples,
        deploy_vi_best=False if args.no_deploy_vi_best else None,
    )
    print(json.dumps(
        {
            "status": manifest.get("status"),
            "mode": manifest.get("mode"),
            "vi_rows": manifest.get("vi_rows"),
            "en_rows": manifest.get("en_rows"),
            "statistics_csv": (manifest.get("statistics") or {}).get("csv"),
            "statistics_rows": (manifest.get("statistics") or {}).get("row_count"),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
