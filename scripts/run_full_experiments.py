#!/usr/bin/env python3
"""Run full-scale experiments on Vietnamese + English datasets.

Thin wrapper around scripts/run_full_experiment_pipeline.py for backward compatibility.
Prefer: python run.py run-full-experiments
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = spec_from_file_location(
    "run_full_experiment_pipeline",
    ROOT / "scripts" / "run_full_experiment_pipeline.py",
)
_pipeline_mod = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pipeline_mod)
run_full_experiment_pipeline = _pipeline_mod.run_full_experiment_pipeline


def run_full_experiments(**kwargs) -> dict:
    """Backward-compatible alias."""
    return run_full_experiment_pipeline(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-scale VI + EN experiments")
    parser.add_argument("--vi-csv", default=None, help="Vietnamese reviews CSV path")
    parser.add_argument("--smoke", action="store_true", help="Quick run on small data samples")
    parser.add_argument("--skip-vi-tfidf", action="store_true")
    parser.add_argument("--skip-en-tfidf", action="store_true")
    parser.add_argument("--skip-vi-phobert", action="store_true")
    parser.add_argument("--skip-vi-fasttext", action="store_true")
    parser.add_argument("--phobert-epochs", type=int, default=None)
    parser.add_argument("--phobert-batch-size", type=int, default=8)
    parser.add_argument("--no-deploy-vi-best", action="store_true")
    args = parser.parse_args()

    manifest = run_full_experiment_pipeline(
        vi_csv=args.vi_csv,
        smoke=args.smoke,
        skip_vi_tfidf=args.skip_vi_tfidf,
        skip_en_tfidf=args.skip_en_tfidf,
        skip_vi_phobert=args.skip_vi_phobert,
        skip_vi_fasttext=args.skip_vi_fasttext,
        phobert_epochs=args.phobert_epochs,
        phobert_batch_size=args.phobert_batch_size,
        deploy_vi_best=not args.no_deploy_vi_best if not args.smoke else False,
    )
    print(json.dumps(
        {
            "status": manifest.get("status"),
            "mode": manifest.get("mode"),
            "vi_rows": manifest.get("vi_rows"),
            "en_rows": manifest.get("en_rows"),
            "statistics": manifest.get("statistics"),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
