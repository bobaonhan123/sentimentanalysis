"""Helpers to repair common mojibake in Vietnamese text."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

MOJIBAKE_MARKERS: tuple[str, ...] = (
    "Ã",
    "Â",
    "Ä",
    "Æ",
    "áº",
    "á»",
    "â€",
    "â€™",
    "â€œ",
    "â€",
    "â€“",
    "â€”",
    "ð",
)


def _marker_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def looks_mojibake(text: str | None) -> bool:
    if not text:
        return False
    return _marker_score(str(text)) > 0


def repair_mojibake_text(text: str | None, *, max_rounds: int = 2) -> str:
    """Repair UTF-8 text that was decoded as latin1/cp1252."""
    if text is None:
        return ""

    current = str(text)
    if not current or not looks_mojibake(current):
        return current

    for _ in range(max_rounds):
        best = current
        best_score = _marker_score(current)
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = current.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            candidate_score = _marker_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
        if best == current:
            break
        current = best
        if best_score == 0:
            break
    return current


def repair_text_iterable(values: Iterable[str]) -> list[str]:
    return [repair_mojibake_text(value) for value in values]


def repair_object_columns(df: pd.DataFrame, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return a copy with mojibake repaired in selected object/string columns."""
    if df.empty:
        return df.copy()

    repaired = df.copy()
    target_columns = list(columns) if columns is not None else list(repaired.columns)
    for column in target_columns:
        if column not in repaired.columns:
            continue
        series = repaired[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        repaired[column] = series.map(lambda value: repair_mojibake_text(value) if pd.notna(value) else value)
    return repaired
