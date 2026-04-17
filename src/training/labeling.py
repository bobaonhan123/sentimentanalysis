"""Label reviews with sentiment based on rating + weak keyword signals.

Data source: CSV file in data_post_processing/ (already crawled).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data_post_processing"
RAW_CSV = DATA_DIR / "1900_export_reviews.csv"

# ── Rating-based label ──────────────────────────────────────────
LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def rating_to_sentiment(rating: float | None) -> int | None:
    """Map 1-5 star rating to sentiment label."""
    if rating is None or pd.isna(rating):
        return None
    if rating <= 2.0:
        return LABEL_MAP["negative"]
    elif rating <= 3.0:
        return LABEL_MAP["neutral"]
    else:
        return LABEL_MAP["positive"]


# ── Weak labeling keywords ──────────────────────────────────────
_NEGATIVE_KEYWORDS = [
    "tệ", "kém", "tồi", "chán", "thất vọng", "không tốt", "lương thấp",
    "quá tải", "áp lực", "bóc lột", "không công bằng", "thiếu chuyên nghiệp",
    "hay thay đổi", "không ổn định", "môi trường độc hại", "overtime",
    "không có cơ hội", "trì trệ", "lãnh đạo kém", "quan liêu",
]
_POSITIVE_KEYWORDS = [
    "tuyệt vời", "xuất sắc", "tốt", "chuyên nghiệp", "hài lòng",
    "lương cao", "phúc lợi", "cơ hội", "phát triển", "năng động",
    "thân thiện", "hỗ trợ", "linh hoạt", "ổn định", "học hỏi",
    "sáng tạo", "đãi ngộ tốt", "đồng nghiệp tốt", "cân bằng",
]


def _keyword_score(text: str) -> float:
    """Return a score: negative < 0, positive > 0."""
    if not text or pd.isna(text):
        return 0.0
    text_lower = str(text).lower()
    pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_lower)
    neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)
    return pos_hits - neg_hits


def weak_label_adjust(rating_label: int, text: str) -> int:
    """Adjust rating-based label using keyword signals for borderline (neutral) cases."""
    if rating_label != LABEL_MAP["neutral"]:
        return rating_label

    score = _keyword_score(text)
    if score <= -2:
        return LABEL_MAP["negative"]
    elif score >= 2:
        return LABEL_MAP["positive"]
    return rating_label


# ── Load labeled dataset from CSV ───────────────────────────────

def load_labeled_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load reviews from CSV, assign sentiment labels.

    The CSV has: company, industry, rating, title, job_title,
    employee_status, location, date, pros, cons, advice, recommends.

    Text is primarily in 'cons' column (most 'pros' are null).
    We combine all available text: title + pros + cons + advice.
    """
    path = Path(csv_path) if csv_path else RAW_CSV

    if not path.exists():
        logger.error(f"CSV not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path.name}")

    records = []
    for _, row in df.iterrows():
        parts = []
        for col in ["title", "pros", "cons", "advice"]:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        text = " ".join(parts)

        if not text:
            continue

        rating_label = rating_to_sentiment(row.get("rating"))
        if rating_label is None:
            continue

        label = weak_label_adjust(rating_label, text)

        records.append({
            "text": text,
            "rating": row["rating"],
            "sentiment": label,
            "sentiment_name": LABEL_NAMES[label],
            "company": row.get("company", ""),
            "industry": row.get("industry", ""),
        })

    result = pd.DataFrame(records)
    logger.info(
        f"Labeled {len(result)} reviews — "
        f"positive: {(result['sentiment'] == 2).sum()}, "
        f"neutral: {(result['sentiment'] == 1).sum()}, "
        f"negative: {(result['sentiment'] == 0).sum()}"
    )
    return result
