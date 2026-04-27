"""Label reviews with sentiment based on rating + weak labeling (keyword + ABSA aspect signals).

Data source: CSV file in data_post_processing/ (already crawled).

Weak labeling strategy:
  1. Base label from star rating (1-2 → negative, 3 → neutral, 4-5 → positive)
  2. ABSA aspect score: count positive/negative aspect mentions across pros/cons/advice
     using the same lexicons as the ABSA module
  3. Combined score overrides rating-based label when the divergence is large enough
     (star says positive but ABSA aspect signals strongly negative, and vice-versa)
  4. Borderline neutrals (rating=3) are always re-labelled by the combined signal
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


# ── Weak labeling keywords (keyword-level) ──────────────────────
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

# ── ABSA-derived opinion lexicons (mirrors src/analysis/absa.py) ─
_ABSA_POSITIVE = [
    "tốt", "tuyệt", "ổn", "tích cực", "nhiệt tình", "rõ ràng",
    "minh bạch", "công bằng", "hỗ trợ", "vui", "thân thiện", "chuyên nghiệp",
    "năng động", "cởi mở", "hợp lý", "xứng đáng", "phù hợp", "ổn định",
    "hiệu quả", "tuyệt vời", "hài lòng", "thoải mái", "cạnh tranh",
    "tận tâm", "quan tâm",
]
_ABSA_NEGATIVE = [
    "tệ", "kém", "chậm", "áp lực", "stress", "thấp", "thiếu", "ràng buộc",
    "bất công", "drama", "độc đoán", "toxic", "khắc khe", "ì ạch", "trễ",
    "cũ kỹ", "thất vọng", "khó khăn", "quá tải", "mệt", "chán",
    "không minh bạch", "không công bằng", "không hợp lý",
]


def _keyword_score(text: str) -> float:
    """Keyword-level score: negative < 0, positive > 0."""
    if not text or pd.isna(text):
        return 0.0
    t = str(text).lower()
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in t)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in t)
    return float(pos - neg)


def _absa_score(pros: str, cons: str, advice: str) -> float:
    """ABSA-based score across pros/cons/advice fields.

    cons is weighted 1.5x because employees write more specific complaints there.
    Returns a float: negative → negative sentiment, positive → positive sentiment.
    """
    def _field_score(text: str, weight: float = 1.0) -> float:
        if not text or pd.isna(text):
            return 0.0
        t = str(text).lower()
        p = sum(1 for w in _ABSA_POSITIVE if w in t)
        n = sum(1 for w in _ABSA_NEGATIVE if w in t)
        return (p - n) * weight

    return (
        _field_score(pros, weight=1.0)
        + _field_score(cons, weight=1.5)
        + _field_score(advice, weight=0.5)
    )


def weak_label_combine(
    rating_label: int,
    pros: str,
    cons: str,
    advice: str,
) -> tuple[int, str]:
    """Combine star rating + ABSA aspect score into final weak label.

    Returns (label, reason) where reason describes what triggered any override.

    Override rules:
    - rating=3 (neutral): always re-label by combined signal
    - rating>=4 (star-positive) but ABSA score <= -3: flip to negative
      (employee rates generously but text is overwhelmingly negative)
    - rating<=2 (star-negative) but ABSA score >= 3: flip to positive
      (rare, usually a data entry mistake or sarcasm — keep but log)
    """
    kw_score = _keyword_score(" ".join(str(x) for x in [pros, cons, advice]))
    absa = _absa_score(pros, cons, advice)
    combined = kw_score + absa

    if rating_label == LABEL_MAP["neutral"]:
        if combined <= -1.5:
            return LABEL_MAP["negative"], "neutral→negative(absa)"
        elif combined >= 1.5:
            return LABEL_MAP["positive"], "neutral→positive(absa)"
        return rating_label, "neutral(unchanged)"

    if rating_label == LABEL_MAP["positive"] and combined <= -3:
        return LABEL_MAP["negative"], "positive→negative(absa_override)"

    if rating_label == LABEL_MAP["negative"] and combined >= 3:
        return LABEL_MAP["positive"], "negative→positive(absa_override)"

    return rating_label, "rating"


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

        pros_text = str(row.get("pros") or "")
        cons_text = str(row.get("cons") or "")
        advice_text = str(row.get("advice") or "")
        label, label_source = weak_label_combine(rating_label, pros_text, cons_text, advice_text)

        records.append({
            "text": text,
            "rating": row["rating"],
            "sentiment": label,
            "sentiment_name": LABEL_NAMES[label],
            "label_source": label_source,
            "company": row.get("company", ""),
            "industry": row.get("industry", ""),
        })

    result = pd.DataFrame(records)
    overrides = result[result["label_source"].str.contains("override|absa", na=False)]
    logger.info(
        f"Labeled {len(result)} reviews — "
        f"positive: {(result['sentiment'] == 2).sum()}, "
        f"neutral: {(result['sentiment'] == 1).sum()}, "
        f"negative: {(result['sentiment'] == 0).sum()} | "
        f"absa_overrides: {len(overrides)}"
    )
    return result
