"""Experiment logging — save/load training results as JSON for Streamlit."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "models"
EXPERIMENTS_FILE = EXPERIMENTS_DIR / "experiments.json"


def _ensure_dir():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def load_experiments() -> list[dict]:
    """Load all experiment records."""
    _ensure_dir()
    if not EXPERIMENTS_FILE.exists():
        return []
    try:
        return json.loads(EXPERIMENTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []


def save_experiment(record: dict):
    """Append an experiment record and save."""
    _ensure_dir()
    experiments = load_experiments()
    record.setdefault("timestamp", datetime.now().isoformat())
    experiments.append(record)
    EXPERIMENTS_FILE.write_text(
        json.dumps(experiments, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(f"Saved experiment: {record.get('run_id', 'unknown')}")


def get_best_experiment(metric: str = "f1_macro") -> dict | None:
    """Return experiment with highest value for given metric."""
    experiments = load_experiments()
    if not experiments:
        return None
    return max(experiments, key=lambda e: e.get("best_model", {}).get(metric, 0))
