#!/usr/bin/env python3
"""Rebuild statistics after the English full-scale run completes."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PID_FILE = ROOT / "models" / "statistics" / "en_tfidf.pid"
LOG_FILE = ROOT / "models" / "statistics" / "en_tfidf_run.log"


def _pid_alive(pid: int) -> bool:
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("build_final_statistics", ROOT / "scripts" / "build_final_statistics.py")
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        while _pid_alive(pid):
            time.sleep(60)
    if not LOG_FILE.exists() or "TRAIN_VARIANTS_DONE" not in LOG_FILE.read_text(encoding="utf-8", errors="ignore"):
        raise SystemExit("English TF-IDF run is not marked done in log.")

    payload = mod.build_final_statistics()
    manifest = {
        "status": "success",
        "statistics_csv": payload["csv"],
        "row_count": payload["row_count"],
        "vi_tfidf_run_id": payload.get("vi_tfidf_run_id"),
        "vi_fasttext_run_id": payload.get("vi_fasttext_run_id"),
        "en_run_id": payload.get("en_run_id"),
    }
    out = ROOT / "models" / "statistics" / "full_experiment_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
