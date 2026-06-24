#!/usr/bin/env python3
"""Cache Glassdoor English dataset to local parquet."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.labeling import GLASSDOOR_CACHE_PARQUET, load_glassdoor_english_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> None:
    df = load_glassdoor_english_data(max_rows=0)
    print(f"Cached {len(df)} rows -> {GLASSDOOR_CACHE_PARQUET}")


if __name__ == "__main__":
    main()
