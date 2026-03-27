"""Persistent Bloom filter wrapper for URL / review deduplication."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

from pybloom_live import ScalableBloomFilter

from src.config import settings

def _default_bloom_path() -> Path:
    """Use volume mount path inside Airflow containers, else project root."""
    airflow_path = Path("/opt/airflow/bloom_data/bloom_filter.bin")
    if airflow_path.parent.exists():
        return airflow_path
    return Path(__file__).resolve().parents[2] / "bloom_filter.bin"

_BLOOM_PATH = _default_bloom_path()


class BloomFilter:
    def __init__(self, path: Path = _BLOOM_PATH):
        self._path = path
        if path.exists():
            with open(path, "rb") as f:
                self._bf: ScalableBloomFilter = pickle.load(f)
        else:
            self._bf = ScalableBloomFilter(
                initial_capacity=settings.bloom_capacity,
                error_rate=settings.bloom_error_rate,
                mode=ScalableBloomFilter.SMALL_SET_GROWTH,
            )

    def __contains__(self, key: str) -> bool:
        return key in self._bf

    def add(self, key: str) -> bool:
        """Add key. Returns True if key was NEW (not seen before)."""
        if key in self._bf:
            return False
        self._bf.add(key)
        return True

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._bf, f)

    @property
    def count(self) -> int:
        return self._bf.count
