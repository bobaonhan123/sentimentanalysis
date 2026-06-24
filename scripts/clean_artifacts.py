from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Conservative: delete only clearly-derived artifacts (logs/pids/runs/caches/model weights).
DELETE_GLOBS = [
    "__pycache__",
    "models/**/runs",
    "models/statistics/*.log",
    "models/statistics/*.pid",
    "models/statistics/*.csv",
    "models/statistics/*.json",
    "models/fasttext/**/*.bin",
    "models/**/*.safetensors",
    "models/**/*.pkl",
    "data/en/glassdoor/*.parquet",
]


def _rm(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def main() -> None:
    deleted: list[Path] = []
    for pattern in DELETE_GLOBS:
        for p in REPO_ROOT.glob(pattern):
            # Keep source code / configs safe: only delete inside repo root.
            if REPO_ROOT not in p.resolve().parents and p.resolve() != REPO_ROOT:
                continue
            if _rm(p):
                deleted.append(p.relative_to(REPO_ROOT))

    deleted_sorted = sorted(set(deleted), key=lambda x: (str(x)))
    if not deleted_sorted:
        print("No artifacts matched; nothing deleted.")
        return

    print("Deleted:")
    for p in deleted_sorted:
        print(f"- {p}")


if __name__ == "__main__":
    main()

