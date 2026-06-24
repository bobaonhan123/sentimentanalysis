"""Label reviews with sentiment based on rating + weak labeling (keyword + ABSA aspect signals).

Data source: CSV file in data/vi/raw/ (72k Vietnamese reviews).

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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DATA = Path(__file__).resolve().parents[2]
VI_REVIEWS_CSV = ROOT_DATA / "data" / "vi" / "raw" / "1900_export_reviews.csv"
VI_DATA_CANDIDATES = [
    VI_REVIEWS_CSV,
    ROOT_DATA / "data_post_processing" / "1900_export_reviews.csv",
    ROOT_DATA / "analysis" / "1900_export_reviews.csv",
]
GLASSDOOR_EN_DATASET = "lallantop/glassdoor"
GLASSDOOR_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "en" / "glassdoor"
GLASSDOOR_CACHE_PARQUET = GLASSDOOR_CACHE_DIR / "labeled_reviews.parquet"
GLASSDOOR_PROCESSED_PARQUET = GLASSDOOR_CACHE_DIR / "labeled_reviews_processed.parquet"

_EN_NEGATIVE_KEYWORDS = [
    "toxic", "low pay", "poor management", "bad culture", "long hours", "no benefits",
    "underpaid", "micromanaging", "hostile", "burnout", "no growth", "unfair",
    "disorganized", "high turnover", "stressful", "overworked", "layoffs",
    "late paycheck", "pay cut", "no raise", "fired", "harassment", "micromanagement",
    "not recommend", "toxic culture", "unpaid overtime", "no bonus", "exhausted",
    "politics", "boring", "forced", "layoff", "terrible", "awful", "horrible",
    "worst", "miserable", "exploited", "understaffed", "no work life balance",
]
_EN_POSITIVE_KEYWORDS = [
    "great team", "good benefits", "supportive manager", "work life balance",
    "career growth", "competitive pay", "flexible", "collaborative", "innovative",
    "professional", "friendly coworkers", "learning opportunities", "fair compensation",
]
_EN_SARCASM_POSITIVE_PHRASES = [
    "no complaints", "nothing to complain", "can't complain", "cannot complain",
    "no issues", "nothing bad to say",
]
_EN_PLACEHOLDER_NEGATIVE_PATTERNS = [
    "no pros",
    "no advantages",
    "no benefits",
    "nothing good",
    "can't think of any pros",
    "cannot think of any pros",
]
_MINIMAL_CONS_TEXTS = {
    "không có", "không", "nan", "none", "n/a", "no", "nothing",
}


def _english_field_score(text: str, weight: float, *, suppress_sarcasm: bool) -> float:
    if not text or pd.isna(text):
        return 0.0
    t = str(text).lower()
    if suppress_sarcasm and any(phrase in t for phrase in _EN_SARCASM_POSITIVE_PHRASES):
        pos = 0
    else:
        pos = sum(1 for kw in _EN_POSITIVE_KEYWORDS if kw in t)
    neg = sum(1 for kw in _EN_NEGATIVE_KEYWORDS if kw in t)
    return (pos - neg) * weight


def english_keyword_score(text: str) -> float:
    """Keyword-level score for English review text."""
    if not text or pd.isna(text):
        return 0.0
    return _english_field_score(text, 1.0, suppress_sarcasm=False)


def _en_meta_negative_adjustment(headline: str, pros: str, cons: str) -> float:
    adj = 0.0
    for text, weight in ((headline, 2.0), (cons, 1.5), (pros, 1.0)):
        if not text or pd.isna(text):
            continue
        t = str(text).lower()
        if any(pattern in t for pattern in _EN_PLACEHOLDER_NEGATIVE_PATTERNS):
            adj -= 2.0 * weight
    return adj


def _en_combined_text_score(
    headline: str,
    pros: str,
    cons: str,
    *,
    rating_label: int,
) -> float:
    full_text = " ".join(str(x) for x in (headline, pros, cons) if x and not pd.isna(x)).lower()
    has_sarcasm = any(phrase in full_text for phrase in _EN_SARCASM_POSITIVE_PHRASES)
    suppress_sarcasm = rating_label <= LABEL_MAP["neutral"] or (
        rating_label == LABEL_MAP["positive"] and has_sarcasm
    )
    combined = (
        _english_field_score(headline, 2.0, suppress_sarcasm=suppress_sarcasm)
        + _english_field_score(pros, 1.0, suppress_sarcasm=suppress_sarcasm)
        + _english_field_score(cons, 1.5, suppress_sarcasm=suppress_sarcasm)
        + _en_meta_negative_adjustment(headline, pros, cons)
    )
    if rating_label == LABEL_MAP["neutral"]:
        headline_unit = _english_field_score(headline, 1.0, suppress_sarcasm=suppress_sarcasm)
        cons_unit = _english_field_score(cons, 1.0, suppress_sarcasm=suppress_sarcasm)
        if headline_unit < -1.0 and cons_unit > 0.5:
            combined -= cons_unit * 1.5
        if _is_minimal_cons(cons) and headline_unit < -0.5:
            combined -= 2.0
    return combined


def _title_has_negative_cues(title: str) -> bool:
    if not title or pd.isna(title):
        return False
    t = str(title).lower()
    if any(pattern in t for pattern in _PLACEHOLDER_NEGATIVE_PATTERNS):
        return True
    return _count_lexicon_hits(t, _NEGATIVE_KEYWORDS) >= 1


def english_weak_label_combine(
    rating_label: int,
    headline: str,
    pros: str,
    cons: str,
    rating: float | None = None,
) -> tuple[int, str]:
    """Mirror VI weak-label rules for English Glassdoor reviews."""
    combined = _en_combined_text_score(headline, pros, cons, rating_label=rating_label)
    title_neg = _title_has_negative_cues(headline)

    if rating is not None and not pd.isna(rating) and float(rating) <= 1.0:
        if rating_label == LABEL_MAP["negative"]:
            return rating_label, "rating_anchor_1star"
    if rating is not None and not pd.isna(rating) and float(rating) <= 2.0:
        if rating_label == LABEL_MAP["negative"]:
            return rating_label, "rating_anchor_low"

    if rating_label == LABEL_MAP["neutral"]:
        if combined <= -1.5:
            return LABEL_MAP["negative"], "neutral→negative(en_kw)"
        if combined >= 2.5:
            return LABEL_MAP["positive"], "neutral→positive(en_kw)"
        return rating_label, "neutral(unchanged)"

    if rating_label == LABEL_MAP["positive"]:
        neg_threshold = -2.5 if title_neg else -3.0
        if combined <= neg_threshold:
            reason = "positive→negative(en_title_hr)" if title_neg else "positive→negative(en_override)"
            return LABEL_MAP["negative"], reason
        if combined <= -0.5:
            return LABEL_MAP["neutral"], "positive→neutral(en_conflict)"

    if rating_label == LABEL_MAP["negative"]:
        if combined >= 10:
            return LABEL_MAP["positive"], "negative→positive(en_override)"
        if combined >= 5:
            return LABEL_MAP["neutral"], "negative→neutral(en_conflict)"

    return rating_label, "overall_rating"

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# Binary weak-label framing (3-class sentiment kept internally).
BINARY_FRAMING_NEGATIVE_NONNEGATIVE = "negative_vs_non_negative"
BINARY_FRAMING_POSITIVE_NONPOSITIVE = "positive_vs_non_positive"
DEFAULT_BINARY_FRAMING = BINARY_FRAMING_NEGATIVE_NONNEGATIVE

BINARY_LABEL_NAMES: dict[str, dict[int, str]] = {
    BINARY_FRAMING_NEGATIVE_NONNEGATIVE: {0: "negative", 1: "non_negative"},
    BINARY_FRAMING_POSITIVE_NONPOSITIVE: {0: "non_positive", 1: "positive"},
}

BINARY_VARIANT_NAMES: dict[str, str] = {
    BINARY_FRAMING_NEGATIVE_NONNEGATIVE: "negative_vs_non_negative",
    BINARY_FRAMING_POSITIVE_NONPOSITIVE: "positive_vs_non_positive",
}

CLEANLAB_VARIANT_NAMES: dict[str, str] = {
    BINARY_FRAMING_NEGATIVE_NONNEGATIVE: "cleanlab_pruned_negative_vs_non_negative",
    BINARY_FRAMING_POSITIVE_NONPOSITIVE: "cleanlab_pruned_positive_vs_non_positive",
}


def _vi_negative_lexicon_hits(title: str, pros: str, cons: str, advice: str) -> tuple[int, str]:
    """Count VI negative keyword/ABSA hits; return (total, field summary)."""
    lexicons = (_NEGATIVE_KEYWORDS, _ABSA_NEGATIVE)
    fields = (("title", title), ("pros", pros), ("cons", cons), ("advice", advice))
    total = 0
    hit_fields: list[str] = []
    for name, text in fields:
        if not text or pd.isna(text):
            continue
        t = str(text).lower()
        hits = sum(_count_lexicon_hits(t, lex) for lex in lexicons)
        if hits:
            total += hits
            hit_fields.append(name)
        if any(pattern in t for pattern in _PLACEHOLDER_NEGATIVE_PATTERNS):
            total += 1
            if "placeholder" not in hit_fields:
                hit_fields.append("placeholder")
    return total, "+".join(hit_fields)


def has_negative_signal(
    title: str,
    pros: str,
    cons: str,
    advice: str,
    rating: float | None = None,
) -> tuple[bool, str]:
    """True when review has ANY negative element (VI weak-label path).

    Rule of thumb: one negative cue anywhere → negative (class 0).
    Non-negative (class 1) only when rating >= 4, zero negative hits, combined >= 0.
    Ratings 1-3 are negative by default (neutral/mixed → negative).
    """
    neg_hits, hit_fields = _vi_negative_lexicon_hits(title, pros, cons, advice)
    if neg_hits > 0:
        return True, f"neg_lexicon({hit_fields})"

    if rating is not None and not pd.isna(rating) and float(rating) <= 3.0:
        return True, "rating_le_3"

    rating_label = rating_to_sentiment(rating) if rating is not None and not pd.isna(rating) else LABEL_MAP["neutral"]
    combined = _combined_text_score(title, pros, cons, advice, rating_label=rating_label)
    if combined < 0:
        return True, "combined_score_negative"

    return False, "non_negative"


def _en_negative_lexicon_hits(headline: str, pros: str, cons: str) -> tuple[int, str]:
    fields = (("headline", headline), ("pros", pros), ("cons", cons))
    total = 0
    hit_fields: list[str] = []
    for name, text in fields:
        if not text or pd.isna(text):
            continue
        t = str(text).lower()
        hits = sum(1 for kw in _EN_NEGATIVE_KEYWORDS if kw in t)
        if hits:
            total += hits
            hit_fields.append(name)
    return total, "+".join(hit_fields)


def english_has_negative_signal(
    headline: str,
    pros: str,
    cons: str,
    rating: float | None = None,
) -> tuple[bool, str]:
    """Mirror VI negative-vs-non-negative rules for English Glassdoor reviews."""
    neg_hits, hit_fields = _en_negative_lexicon_hits(headline, pros, cons)
    if neg_hits > 0:
        return True, f"neg_lexicon({hit_fields})"

    if rating is not None and not pd.isna(rating) and float(rating) <= 3.0:
        return True, "rating_le_3"

    rating_label = rating_to_sentiment(rating) if rating is not None and not pd.isna(rating) else LABEL_MAP["neutral"]
    full_text = " ".join(str(x) for x in (headline, pros, cons) if x and not pd.isna(x)).lower()
    has_sarcasm = any(phrase in full_text for phrase in _EN_SARCASM_POSITIVE_PHRASES)
    suppress_sarcasm = rating_label <= LABEL_MAP["neutral"] or (
        rating_label == LABEL_MAP["positive"] and has_sarcasm
    )
    combined = (
        _english_field_score(headline, 2.0, suppress_sarcasm=suppress_sarcasm)
        + _english_field_score(pros, 1.0, suppress_sarcasm=suppress_sarcasm)
        + _english_field_score(cons, 1.5, suppress_sarcasm=suppress_sarcasm)
    )
    if combined < 0:
        return True, "combined_score_negative"

    return False, "non_negative"


def binary_label_from_fields(
    row: pd.Series | dict,
    *,
    language: str = "vi",
) -> tuple[int, str]:
    """Assign binary label 0=negative, 1=non-negative from review fields."""
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = row
    lang = str(data.get("language") or language).lower()
    rating = data.get("rating")
    if lang == "en":
        has_neg, source = english_has_negative_signal(
            str(data.get("headline") or data.get("title") or ""),
            str(data.get("pros") or ""),
            str(data.get("cons") or ""),
            rating=rating,
        )
    else:
        has_neg, source = has_negative_signal(
            str(data.get("title") or ""),
            str(data.get("pros") or ""),
            str(data.get("cons") or ""),
            str(data.get("advice") or ""),
            rating=rating,
        )
    return (0 if has_neg else 1), source


def _dataframe_has_review_fields(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    if {"pros", "cons"}.issubset(cols):
        return True
    return bool({"title", "pros", "cons"}.issubset(cols) or {"headline", "pros", "cons"}.issubset(cols))


def map_sentiment_to_binary(
    sentiment: int | np.ndarray | pd.Series,
    framing: str = DEFAULT_BINARY_FRAMING,
) -> int | np.ndarray:
    """Map 3-class weak labels (0/1/2) to binary training labels (fallback without text fields)."""
    if framing not in BINARY_LABEL_NAMES:
        raise ValueError(f"Unknown binary framing: {framing}")
    values = np.asarray(sentiment)
    if framing == BINARY_FRAMING_NEGATIVE_NONNEGATIVE:
        # Neutral/mixed treated as negative when row-level signals are unavailable.
        mapped = np.where(values == LABEL_MAP["positive"], 1, 0)
    else:
        mapped = np.where(values == LABEL_MAP["positive"], 1, 0)
    if np.ndim(sentiment) == 0:
        return int(mapped.item())
    return mapped


def apply_binary_framing(
    df: pd.DataFrame,
    framing: str = DEFAULT_BINARY_FRAMING,
    *,
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Return a copy with binary sentiment labels and human-readable names."""
    out = df.copy()
    label_names = BINARY_LABEL_NAMES[framing]
    if framing == BINARY_FRAMING_NEGATIVE_NONNEGATIVE and _dataframe_has_review_fields(out):
        labels_sources = [binary_label_from_fields(row) for _, row in out.iterrows()]
        out[sentiment_col] = [label for label, _ in labels_sources]
        out["binary_label_source"] = [source for _, source in labels_sources]
    else:
        out[sentiment_col] = map_sentiment_to_binary(out[sentiment_col].to_numpy(), framing)
        if "binary_label_source" not in out.columns:
            out["binary_label_source"] = "sentiment_map_fallback"
    out["sentiment_name"] = out[sentiment_col].map(label_names)
    return out


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
    "không phù hợp", "không xứng đáng", "không minh bạch",
    # HR / workplace negative lexicon expansion
    "trễ lương", "đấu đá", "chính trị", "ép buộc", "nhàm chán", "cắt giảm",
    "không tăng lương", "sa thải", "đuổi việc", "quấy rối", "micromanagement",
    "không recommend", "văn hóa độc hại", "lừa đảo", "ot không lương",
    "làm thêm không lương", "thưởng ít", "không có thưởng", "áp lực cao",
    "burnout", "kiệt sức", "toxic", "layoff", "sa thải hàng loạt",
    "bóc lột sức lao động", "không có phúc lợi", "lương không đủ sống",
    # Complaints often in title; catch variants missed by generic lexicon
    "không thú vị", "trả trễ", "lương trả trễ", "không có hđ", "không có hợp đồng",
    "không ký hợp đồng", "thiếu trách nhiệm", "tăng lên thấp", "lương không tăng",
    "không có lương", "không minh bạch lương", "không có hđlđ",
    "ot nhiều", "overtime nhiều", "làm thêm nhiều", "giờ làm dài",
]
_PLACEHOLDER_NEGATIVE_PATTERNS = [
    "không có ưu điểm nào",
    "không có ưu điểm gì",
    "không có ưu điểm",
    "hoàn toàn không có ưu điểm",
    "thật sự không có ưu điểm",
    "không có gì tốt",
    "không có điểm tích cực",
    "không có gì để khen",
]

# Standalone "tốt" removed from positive lexicons — substring match falsely
# cancels "không tốt" (see labeling audit).
_POSITIVE_KEYWORDS = [
    "tuyệt vời", "xuất sắc", "chuyên nghiệp", "hài lòng",
    "lương cao", "phúc lợi", "cơ hội", "phát triển", "năng động",
    "thân thiện", "hỗ trợ", "linh hoạt", "ổn định", "học hỏi",
    "sáng tạo", "đãi ngộ tốt", "đồng nghiệp tốt", "cân bằng",
    "chịu khó", "kiên trì", "môi trường tốt", "lương tốt",
]

# Phrases that read positive but are often sarcastic in low-rated reviews.
_SARCASM_POSITIVE_PHRASES = [
    "không phàn nàn", "không có gì phàn nàn", "không có gì để chê",
    "không có gì cần phàn nàn", "không có gì chê",
]

# ── ABSA-derived opinion lexicons (mirrors src/analysis/absa.py) ─
_ABSA_POSITIVE = [
    "tuyệt", "ổn", "tích cực", "nhiệt tình", "rõ ràng",
    "minh bạch", "công bằng", "hỗ trợ", "vui", "thân thiện", "chuyên nghiệp",
    "năng động", "cởi mở", "hợp lý", "xứng đáng", "phù hợp", "ổn định",
    "hiệu quả", "tuyệt vời", "hài lòng", "thoải mái", "cạnh tranh",
    "tận tâm", "quan tâm", "chịu khó", "kiên trì",
]
_ABSA_NEGATIVE = [
    "không tốt", "không ổn", "không phù hợp", "không xứng đáng",
    "tệ", "kém", "chậm", "áp lực", "stress", "thấp", "thiếu", "ràng buộc",
    "bất công", "drama", "độc đoán", "toxic", "khắc khe", "ì ạch", "trễ",
    "cũ kỹ", "thất vọng", "khó khăn", "quá tải", "mệt", "chán",
    "không minh bạch", "không công bằng", "không hợp lý",
    # HR / workplace negative lexicon expansion
    "trễ lương", "đấu đá", "chính trị", "ép buộc", "nhàm chán", "cắt giảm",
    "không tăng lương", "sa thải", "đuổi việc", "quấy rối", "micromanagement",
    "không recommend", "văn hóa độc hại", "lừa đảo", "ot không lương",
    "làm thêm không lương", "thưởng ít", "không có thưởng", "áp lực cao",
    "burnout", "kiệt sức", "layoff", "bóc lột sức lao động",
    "không có phúc lợi", "lương không đủ sống",
]


def _count_lexicon_hits(text: str, lexicon: list[str]) -> int:
    """Longest-match substring counting to reduce nested false positives."""
    if not text or pd.isna(text):
        return 0
    t = str(text).lower()
    matched: list[tuple[int, int]] = []
    for kw in sorted(lexicon, key=len, reverse=True):
        start = 0
        while True:
            idx = t.find(kw, start)
            if idx < 0:
                break
            end = idx + len(kw)
            if not any(idx < e and end > s for s, e in matched):
                matched.append((idx, end))
            start = idx + 1
    return len(matched)


def _count_positive_hits(text: str, positive_lexicon: list[str], *, suppress_sarcasm: bool) -> int:
    """Count positive keyword hits, optionally ignoring sarcasm-prone phrases."""
    if not text or pd.isna(text):
        return 0
    t = str(text).lower()
    if suppress_sarcasm and any(phrase in t for phrase in _SARCASM_POSITIVE_PHRASES):
        return 0
    return _count_lexicon_hits(t, positive_lexicon)


def _is_minimal_cons(cons: str) -> bool:
    if not cons or pd.isna(cons):
        return True
    t = str(cons).strip().lower()
    return len(t) < 12 or t in _MINIMAL_CONS_TEXTS


def _meta_negative_adjustment(title: str, pros: str, cons: str, advice: str) -> float:
    """Placeholder/meta cons lines imply complaint even without explicit neg words."""
    adj = 0.0
    for text, weight in ((title, 2.0), (cons, 1.5), (pros, 1.0), (advice, 0.5)):
        if not text or pd.isna(text):
            continue
        t = str(text).lower()
        if any(pattern in t for pattern in _PLACEHOLDER_NEGATIVE_PATTERNS):
            adj -= 2.0 * weight
    return adj


def _field_keyword_score(text: str, weight: float, *, suppress_sarcasm: bool) -> float:
    if not text or pd.isna(text):
        return 0.0
    t = str(text).lower()
    pos = _count_positive_hits(t, _POSITIVE_KEYWORDS, suppress_sarcasm=suppress_sarcasm)
    neg = _count_lexicon_hits(t, _NEGATIVE_KEYWORDS)
    return (pos - neg) * weight


def _keyword_score(
    title: str,
    pros: str,
    cons: str,
    advice: str,
    *,
    suppress_sarcasm: bool = False,
) -> float:
    """Field-aware keyword score: title/cons weighted higher (complaints often there)."""
    return (
        _field_keyword_score(title, 2.0, suppress_sarcasm=suppress_sarcasm)
        + _field_keyword_score(pros, 1.0, suppress_sarcasm=suppress_sarcasm)
        + _field_keyword_score(cons, 1.5, suppress_sarcasm=suppress_sarcasm)
        + _field_keyword_score(advice, 0.5, suppress_sarcasm=suppress_sarcasm)
    )


def _combined_text_score(
    title: str,
    pros: str,
    cons: str,
    advice: str,
    *,
    rating_label: int,
) -> float:
    """Single lexicon path (no kw+absa double-count) plus meta-negative cues."""
    full_text = " ".join(str(x) for x in (title, pros, cons, advice) if x and not pd.isna(x)).lower()
    has_sarcasm = any(phrase in full_text for phrase in _SARCASM_POSITIVE_PHRASES)
    suppress_sarcasm = rating_label <= LABEL_MAP["neutral"] or (
        rating_label == LABEL_MAP["positive"] and has_sarcasm
    )
    combined = _keyword_score(title, pros, cons, advice, suppress_sarcasm=suppress_sarcasm) + _meta_negative_adjustment(
        title, pros, cons, advice
    )
    if rating_label == LABEL_MAP["neutral"]:
        title_unit = _field_keyword_score(title, 1.0, suppress_sarcasm=suppress_sarcasm)
        cons_unit = _field_keyword_score(cons, 1.0, suppress_sarcasm=suppress_sarcasm)
        # Pros/cons column confusion: praise in cons while title carries the complaint.
        if title_unit < -1.0 and cons_unit > 0.5:
            combined -= cons_unit * 1.5
        if _is_minimal_cons(cons) and title_unit < -0.5:
            combined -= 2.0
    return combined


def _absa_score(
    title: str,
    pros: str,
    cons: str,
    advice: str,
    *,
    suppress_sarcasm: bool = False,
) -> float:
    """ABSA-based score across review fields with cons weighted 1.5×."""
    def _field_score(text: str, weight: float = 1.0) -> float:
        if not text or pd.isna(text):
            return 0.0
        t = str(text).lower()
        p = _count_positive_hits(t, _ABSA_POSITIVE, suppress_sarcasm=suppress_sarcasm)
        n = sum(1 for w in _ABSA_NEGATIVE if w in t)
        return (p - n) * weight

    return (
        _field_score(title, weight=1.0)
        + _field_score(pros, weight=1.0)
        + _field_score(cons, weight=1.5)
        + _field_score(advice, weight=0.5)
    )


def weak_label_combine(
    rating_label: int,
    pros: str,
    cons: str,
    advice: str,
    title: str = "",
    rating: float | None = None,
) -> tuple[int, str]:
    """Combine star rating + text signals into final weak label."""
    combined = _combined_text_score(title, pros, cons, advice, rating_label=rating_label)
    title_neg = _title_has_negative_cues(title)
    has_placeholder = any(
        pattern in str(text or "").lower()
        for text in (cons, advice)
        for pattern in _PLACEHOLDER_NEGATIVE_PATTERNS
    )

    # Hard anchors for very low ratings — do not let generic positive words flip label.
    if rating is not None and not pd.isna(rating) and float(rating) <= 1.0:
        if rating_label == LABEL_MAP["negative"]:
            return rating_label, "rating_anchor_1star"
    if rating is not None and not pd.isna(rating) and float(rating) <= 2.0:
        if rating_label == LABEL_MAP["negative"]:
            return rating_label, "rating_anchor_low"

    if rating_label == LABEL_MAP["neutral"]:
        if combined <= -1.5:
            return LABEL_MAP["negative"], "neutral→negative(absa)"
        if combined >= 2.5:
            return LABEL_MAP["positive"], "neutral→positive(absa)"
        return rating_label, "neutral(unchanged)"

    if rating_label == LABEL_MAP["positive"]:
        neg_threshold = -2.5 if title_neg else -3.0
        if has_placeholder and title_neg:
            neg_threshold = -1.5
        if combined <= neg_threshold:
            if title_neg:
                return LABEL_MAP["negative"], "positive→negative(title_hr)"
            return LABEL_MAP["negative"], "positive→negative(absa_override)"
        if combined <= -0.5:
            return LABEL_MAP["neutral"], "positive→neutral(absa_conflict)"

    if rating_label == LABEL_MAP["negative"]:
        if combined >= 10:
            return LABEL_MAP["positive"], "negative→positive(absa_override)"
        if combined >= 5:
            return LABEL_MAP["neutral"], "negative→neutral(absa_conflict)"

    return rating_label, "rating"


# ── Load labeled dataset from CSV ───────────────────────────────

def load_labeled_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load reviews from CSV, assign sentiment labels.

    The CSV has: company, industry, rating, title, job_title,
    employee_status, location, date, pros, cons, advice, recommends.

    Text is primarily in 'cons' column (most 'pros' are null).
    We combine all available text: title + pros + cons + advice.
    """
    path = Path(csv_path) if csv_path else VI_REVIEWS_CSV

    if not path.exists():
        logger.error(f"CSV not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
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
        title_text = str(row.get("title") or "")
        label, label_source = weak_label_combine(
            rating_label,
            pros_text,
            cons_text,
            advice_text,
            title=title_text,
            rating=row.get("rating"),
        )
        has_neg, binary_source = has_negative_signal(
            title_text, pros_text, cons_text, advice_text, rating=row.get("rating")
        )
        binary_sentiment = 0 if has_neg else 1

        records.append({
            "text": text,
            "title": title_text,
            "pros": pros_text,
            "cons": cons_text,
            "advice": advice_text,
            "rating": row["rating"],
            "sentiment": label,
            "sentiment_name": LABEL_NAMES[label],
            "label_source": label_source,
            "binary_sentiment": binary_sentiment,
            "binary_label_source": binary_source,
            "industry": row.get("industry", ""),
            "dataset_source": "1900_vi",
            "language": "vi",
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


def _limit_rows(df: pd.DataFrame, max_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    """Stratified-ish sample by sentiment when a source is much larger than local data."""
    if not max_rows or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)

    parts = []
    remaining = int(max_rows)
    counts = df["sentiment"].value_counts().sort_index()
    for idx, (label, count) in enumerate(counts.items()):
        if idx == len(counts) - 1:
            take = remaining
        else:
            take = int(round(max_rows * int(count) / len(df)))
            take = max(1, min(take, int(count), remaining))
        parts.append(df[df["sentiment"] == label].sample(n=take, random_state=random_state))
        remaining -= take
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def _apply_english_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Re-label English rows with keyword overrides (works on cached parquet too)."""
    if df.empty:
        return df

    records = []
    for _, row in df.iterrows():
        rating_label = rating_to_sentiment(row.get("rating"))
        if rating_label is None:
            continue
        headline = str(row.get("headline") or row.get("title") or "")
        pros = str(row.get("pros") or "")
        cons = str(row.get("cons") or "")
        label, label_source = english_weak_label_combine(
            rating_label, headline, pros, cons, rating=row.get("rating")
        )
        has_neg, binary_source = english_has_negative_signal(
            headline, pros, cons, rating=row.get("rating")
        )
        updated = row.to_dict()
        updated["sentiment"] = label
        updated["sentiment_name"] = LABEL_NAMES[label]
        updated["label_source"] = label_source
        updated["binary_sentiment"] = 0 if has_neg else 1
        updated["binary_label_source"] = binary_source
        records.append(updated)
    return pd.DataFrame(records)


def load_glassdoor_english_data(max_rows: int | None = None, *, preprocessed: bool = False) -> pd.DataFrame:
    """Load English Glassdoor reviews from Hugging Face.

    The source has a firm/company identifier column named `firm`; we intentionally
    do not expose it in the returned frame so training artifacts stay license-safe.
    """
    GLASSDOOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if preprocessed and GLASSDOOR_PROCESSED_PARQUET.exists():
        logger.info("Loading processed Glassdoor parquet %s", GLASSDOOR_PROCESSED_PARQUET)
        result = _apply_english_weak_labels(pd.read_parquet(GLASSDOOR_PROCESSED_PARQUET))
        if max_rows and max_rows > 0:
            result = _limit_rows(result, max_rows)
        logger.info("Prepared %s processed English Glassdoor reviews from cache", len(result))
        return result.reset_index(drop=True)
    if GLASSDOOR_CACHE_PARQUET.exists():
        logger.info("Loading cached Glassdoor parquet %s", GLASSDOOR_CACHE_PARQUET)
        result = _apply_english_weak_labels(pd.read_parquet(GLASSDOOR_CACHE_PARQUET))
        if max_rows and max_rows > 0:
            result = _limit_rows(result, max_rows)
        logger.info(
            "Prepared %s English Glassdoor reviews from cache — positive: %s, neutral: %s, negative: %s",
            len(result),
            (result["sentiment"] == 2).sum() if not result.empty else 0,
            (result["sentiment"] == 1).sum() if not result.empty else 0,
            (result["sentiment"] == 0).sum() if not result.empty else 0,
        )
        return result.reset_index(drop=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install project dependencies before loading "
            f"{GLASSDOOR_EN_DATASET}."
        ) from exc

    logger.info("Loading Hugging Face dataset %s", GLASSDOOR_EN_DATASET)
    ds = load_dataset(GLASSDOOR_EN_DATASET, split="train")
    logger.info("Loaded %s rows from %s", len(ds), GLASSDOOR_EN_DATASET)

    records = []
    for index, row in enumerate(ds, start=1):
        parts = []
        for col in ["headline", "pros", "cons"]:
            val = row.get(col)
            if val is not None and str(val).strip():
                parts.append(str(val).strip())
        text = " ".join(parts)
        if not text:
            continue

        rating = row.get("overall_rating")
        rating_label = rating_to_sentiment(rating)
        if rating_label is None:
            continue

        headline_text = str(row.get("headline") or "").strip()
        pros_text = str(row.get("pros") or "").strip()
        cons_text = str(row.get("cons") or "").strip()
        label, label_source = english_weak_label_combine(
            rating_label, headline_text, pros_text, cons_text, rating=rating
        )
        has_neg, binary_source = english_has_negative_signal(
            headline_text, pros_text, cons_text, rating=rating
        )

        records.append({
            "text": text,
            "headline": headline_text,
            "pros": pros_text,
            "cons": cons_text,
            "rating": rating,
            "sentiment": label,
            "sentiment_name": LABEL_NAMES[label],
            "label_source": label_source,
            "binary_sentiment": 0 if has_neg else 1,
            "binary_label_source": binary_source,
            "industry": "",
            "dataset_source": "glassdoor_en",
            "language": "en",
        })
        if index % 100_000 == 0:
            logger.info("Glassdoor parse progress: scanned=%s kept=%s", index, len(records))

    result = pd.DataFrame(records)
    result = result.replace({np.nan: None})
    if not max_rows or max_rows <= 0:
        result.to_parquet(GLASSDOOR_CACHE_PARQUET, index=False)
        logger.info("Cached %s Glassdoor rows to %s", len(result), GLASSDOOR_CACHE_PARQUET)
    result = _limit_rows(result, max_rows)
    logger.info(
        "Prepared %s English Glassdoor reviews — positive: %s, neutral: %s, negative: %s",
        len(result),
        (result["sentiment"] == 2).sum() if not result.empty else 0,
        (result["sentiment"] == 1).sum() if not result.empty else 0,
        (result["sentiment"] == 0).sum() if not result.empty else 0,
    )
    return result
