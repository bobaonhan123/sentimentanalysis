"""Central layout for model artifacts and run outputs.

Layout:
    models/
    ├── phobert/{vi,en}/
    │   ├── production/best/          # deployed HF checkpoint
    │   ├── runs/{run_id}/            # per-run metrics + history
    │   └── logs/
    ├── tfidf/{vi,en}/
    │   ├── production/               # best_model.pkl + meta.json
    │   ├── candidates/               # variant .pkl files
    │   ├── runs/{run_id}/
    │   └── logs/
    ├── fasttext/{vi,en}/
    │   ├── production/
    │   ├── cc.{vi,en}.300.bin
    │   └── runs/
    └── comparisons/                  # cross-dataset comparison exports
"""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_ROOT = ROOT_DIR / "models"
COMPARISONS_DIR = MODELS_ROOT / "comparisons"
STATISTICS_DIR = MODELS_ROOT / "statistics"
EXPERIMENTS_FILE = MODELS_ROOT / "experiments.json"

# Reporting scripts still read from analysis/; training appends here.
ANALYSIS_DIR = ROOT_DIR / "analysis"
TRAINING_RESULTS_FILE = ANALYSIS_DIR / "training_results.json"


def lang_for_source(source: str) -> str:
    """Map CLI dataset source to language folder code."""
    normalized = (source or "current").strip().lower()
    if normalized in {"glassdoor-en", "glassdoor_en", "en", "english"}:
        return "en"
    return "vi"


class ModelFamilyPaths:
    """Paths under models/{family}/{lang}/."""

    def __init__(self, family: str, lang: str):
        self.family = family
        self.lang = lang
        self.base = MODELS_ROOT / family / lang

    @property
    def production_dir(self) -> Path:
        return self.base / "production"

    @property
    def candidates_dir(self) -> Path:
        return self.base / "candidates"

    @property
    def runs_dir(self) -> Path:
        return self.base / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.base / "logs"

    @property
    def best_model_pkl(self) -> Path:
        return self.production_dir / "best_model.pkl"

    @property
    def meta_json(self) -> Path:
        return self.production_dir / "meta.json"

    @property
    def leaderboard_csv(self) -> Path:
        return self.base / "leaderboard_latest.csv"

    @property
    def label_issues_csv(self) -> Path:
        return self.base / "label_issues_cleanlab.csv"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def phobert_production_checkpoint(self) -> Path:
        return self.production_dir / "best"

    def ensure_dirs(self) -> None:
        for path in (
            self.production_dir,
            self.candidates_dir,
            self.runs_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def tfidf_paths(lang: str) -> ModelFamilyPaths:
    return ModelFamilyPaths("tfidf", lang)


def phobert_paths(lang: str) -> ModelFamilyPaths:
    return ModelFamilyPaths("phobert", lang)


def fasttext_paths(lang: str = "vi") -> ModelFamilyPaths:
    return ModelFamilyPaths("fasttext", lang)


def fasttext_embedding_path(lang: str = "vi") -> Path:
    """Frozen FastText binary; kept under fasttext/{lang}/."""
    filename = f"cc.{lang}.300.bin"
    legacy = MODELS_ROOT / filename if lang == "vi" else None
    new_path = MODELS_ROOT / "fasttext" / lang / filename
    if new_path.exists():
        return new_path
    if legacy and legacy.exists():
        return legacy
    return new_path


def resolve_legacy_path(new_path: Path, *legacy_paths: Path) -> Path:
    """Prefer new layout; fall back to legacy locations when present."""
    if new_path.exists():
        return new_path
    for legacy in legacy_paths:
        if legacy.exists():
            return legacy
    return new_path


def phobert_binary_results_path(lang: str = "vi") -> Path:
    paths = phobert_paths(lang)
    new_path = paths.production_dir / "results.json"
    legacy = ANALYSIS_DIR / "phobert_binary_outputs" / "phobert_binary_results.json"
    return resolve_legacy_path(new_path, legacy)
