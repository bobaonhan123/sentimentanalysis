"""Vietnamese stopwords tailored to the 1900_export_reviews.csv domain.

Built from frequency analysis of the dataset:
- Top-120 words by raw frequency were reviewed manually
- Function words, pronouns, particles, conjunctions → stopword
- Domain-meaningful words (lương, học, đồng, văn, dự, án...) → kept
"""
from __future__ import annotations

# ── Core Vietnamese function words (high freq, no sentiment signal) ──────────
_FUNCTION_WORDS: frozenset[str] = frozenset({
    # Verbs / auxiliaries
    "có", "được", "là", "làm", "để", "cho", "sẽ", "đã", "đang", "bị",
    "phải", "nên", "cần", "muốn", "thể", "bởi",
    # Conjunctions / particles
    "và", "với", "nhưng", "nếu", "khi", "mà", "vì", "nên", "hay", "hoặc",
    "mặc", "dù", "tuy", "nhưng", "thì", "nào", "đây", "đó", "này", "kia",
    "vậy", "thế", "ấy", "đấy",
    # Prepositions / locatives
    "trong", "trên", "dưới", "sau", "trước", "giữa", "qua", "đến",
    "vào", "ra", "lên", "xuống", "ở", "tại", "theo", "từ", "về",
    # Pronouns
    "tôi", "bạn", "anh", "chị", "em", "mình", "họ", "ta", "chúng",
    "người", "ai",
    # Quantifiers / determiners
    "các", "những", "mỗi", "mọi", "tất", "cả", "một", "nhiều", "ít",
    "một", "hai", "ba", "mấy",
    # Adverbs / degree words
    "rất", "khá", "quá", "hơn", "cũng", "vẫn", "lại", "đi", "rồi",
    "nữa", "thêm", "chỉ", "chưa", "không", "của",
    # Discourse markers
    "như", "nhau", "sự", "điều", "việc",
})

# ── Combined stopword set exported for use in processor.py ───────────────────
DOMAIN_STOPWORDS: frozenset[str] = _FUNCTION_WORDS

# ── English stopwords common in this bilingual dataset ───────────────────────
_ENGLISH_SW: frozenset[str] = frozenset({
    "the", "and", "is", "of", "to", "in", "for", "a", "an",
    "it", "at", "be", "as", "by", "we", "on", "or", "do",
})

ALL_STOPWORDS: frozenset[str] = DOMAIN_STOPWORDS | _ENGLISH_SW
