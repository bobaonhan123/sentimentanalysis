#!/usr/bin/env python3
"""After VI TF-IDF finishes, run FastText+MLP and PhoBERT, then rebuild statistics."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PID_FILE = ROOT / "models" / "statistics" / "vi_tfidf.pid"
LOG_FILE = ROOT / "models" / "statistics" / "vi_tfidf_72k_run.log"
VI_VARIANT_LOGS = ROOT / "models" / "tfidf" / "vi" / "logs"


def _latest_vi_variant_log() -> Path | None:
    if not VI_VARIANT_LOGS.exists():
        return None
    logs = sorted(VI_VARIANT_LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _latest_completed_vi_run_id() -> str | None:
    from src.artifacts.paths import tfidf_paths

    paths = tfidf_paths("vi")
    runs = sorted(paths.runs_dir.glob("variant_run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in runs:
        if (run_dir / "leaderboard.csv").exists():
            return run_dir.name
    return None


def _vi_training_done() -> bool:
    latest = _latest_vi_variant_log()
    if latest is not None and "TRAIN_VARIANTS_DONE" in latest.read_text(encoding="utf-8", errors="ignore"):
        return True
    # Manual completion path: leaderboard + production meta from a finished 72k run.
    from src.artifacts.paths import tfidf_paths

    paths = tfidf_paths("vi")
    if paths.meta_json.exists() and paths.best_model_pkl.exists():
        runs = sorted(paths.runs_dir.glob("variant_run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in runs:
            leaderboard = run_dir / "leaderboard.csv"
            if leaderboard.exists() and leaderboard.stat().st_size > 0:
                return True
    return False


def _pid_alive(pid: int) -> bool:
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    if not _vi_training_done():
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            while _pid_alive(pid):
                time.sleep(60)
        # Log may flush shortly after the training process exits.
        for _ in range(30):
            if _vi_training_done():
                break
            time.sleep(10)
        if not _vi_training_done():
            raise SystemExit("VI TF-IDF run is not marked done in log.")

    from src.training.fasttext_binary_trainer import train_fasttext_binary
    from src.training.phobert_binary_trainer import train_phobert_binary
    from importlib.util import module_from_spec, spec_from_file_location

    fasttext_result = train_fasttext_binary(source="current", deploy_best=False)
    phobert_result = train_phobert_binary(
        language="vi",
        max_examples=0,
        epochs=3,
        batch_size=8,
        max_len=256,
    )

    spec = spec_from_file_location("build_final_statistics", ROOT / "scripts" / "build_final_statistics.py")
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    stats = mod.build_final_statistics(vi_run_id=_latest_completed_vi_run_id())

    manifest = {
        "status": "success",
        "vi_fasttext": fasttext_result.get("run_id"),
        "vi_phobert": phobert_result.get("run_id"),
        "statistics_csv": stats["csv"],
        "row_count": stats["row_count"],
    }
    out = ROOT / "models" / "statistics" / "vi_followup_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
