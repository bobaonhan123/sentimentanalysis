#!/usr/bin/env python3
"""Add text_clean to cached Glassdoor parquet in chunks."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.processor import preprocess_english
from src.training.labeling import GLASSDOOR_CACHE_PARQUET

PROCESSED_PARQUET = GLASSDOOR_CACHE_PARQUET.with_name("labeled_reviews_processed.parquet")
CHUNK_SIZE = 50_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("preprocess_glassdoor_cache")


def main() -> None:
    if not GLASSDOOR_CACHE_PARQUET.exists():
        raise FileNotFoundError(f"Missing cache: {GLASSDOOR_CACHE_PARQUET}")

    if PROCESSED_PARQUET.exists():
        logger.info("Processed cache already exists: %s", PROCESSED_PARQUET)
        return

    df = pd.read_parquet(GLASSDOOR_CACHE_PARQUET)
    logger.info("Loaded %s rows from %s", len(df), GLASSDOOR_CACHE_PARQUET)

    cleaned_chunks: list[pd.DataFrame] = []
    total = len(df)
    for start in range(0, total, CHUNK_SIZE):
        chunk = df.iloc[start:start + CHUNK_SIZE].copy()
        chunk["text_clean"] = [
            preprocess_english(text, remove_sw=True)
            for text in chunk["text"].fillna("").astype(str).tolist()
        ]
        chunk["text_clean_no_term_norm"] = chunk["text_clean"]
        chunk = chunk[chunk["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
        cleaned_chunks.append(chunk)
        logger.info("Processed chunk %s/%s kept=%s", min(start + CHUNK_SIZE, total), total, len(chunk))

    processed = pd.concat(cleaned_chunks, ignore_index=True)
    processed["row_id"] = np.arange(len(processed))
    PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(PROCESSED_PARQUET, index=False)
    logger.info("Saved %s processed rows -> %s", len(processed), PROCESSED_PARQUET)


if __name__ == "__main__":
    main()
