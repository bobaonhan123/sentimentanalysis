"""Text preprocessing for Vietnamese and English workplace reviews."""
from __future__ import annotations

import html
import logging
import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.preprocessing.stopwords_vi import ALL_STOPWORDS
from src.preprocessing.vietnamese_terms import normalize_vietnamese_terms

logger = logging.getLogger(__name__)

VIETNAMESE_STOPWORDS: frozenset[str] = ALL_STOPWORDS
NEGATION_WORDS: frozenset[str] = frozenset({
    "không",
    "chưa",
    "chẳng",
    "chả",
    "đừng",
    "ko",
    "k",
})
ENGLISH_NEGATION_WORDS: frozenset[str] = frozenset({
    "no",
    "not",
    "nor",
    "never",
    "none",
    "cannot",
    "without",
})
ENGLISH_DOMAIN_KEEP_WORDS: frozenset[str] = frozenset({
    "benefit",
    "benefits",
    "career",
    "culture",
    "growth",
    "hours",
    "leadership",
    "management",
    "manager",
    "overtime",
    "pay",
    "salary",
    "team",
    "toxic",
    "work",
})
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    word
    for word in ENGLISH_STOP_WORDS
    if word not in ENGLISH_NEGATION_WORDS and word not in ENGLISH_DOMAIN_KEEP_WORDS
)

_ENGLISH_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bshan't\b", "shall not"),
    (r"n't\b", " not"),
    (r"\bI'm\b", "I am"),
    (r"\bit's\b", "it is"),
    (r"\bthat's\b", "that is"),
    (r"\bthere's\b", "there is"),
    (r"\bwhat's\b", "what is"),
    (r"\bthey're\b", "they are"),
    (r"\bwe're\b", "we are"),
    (r"\byou're\b", "you are"),
    (r"\bI've\b", "I have"),
    (r"\bwe've\b", "we have"),
    (r"\bthey've\b", "they have"),
    (r"\bI'd\b", "I would"),
    (r"\bwe'd\b", "we would"),
    (r"\bthey'd\b", "they would"),
    (r"\bI'll\b", "I will"),
    (r"\bwe'll\b", "we will"),
    (r"\bthey'll\b", "they will"),
)
_ENGLISH_WORKPLACE_PHRASES: tuple[tuple[str, str], ...] = (
    ("work life balance", "work_life_balance"),
    ("work-life balance", "work_life_balance"),
    ("long hours", "long_hours"),
    ("low pay", "low_pay"),
    ("poor management", "poor_management"),
    ("senior management", "senior_management"),
    ("career progression", "career_progression"),
    ("career growth", "career_growth"),
    ("good benefits", "good_benefits"),
    ("great benefits", "great_benefits"),
    ("toxic culture", "toxic_culture"),
    ("company culture", "company_culture"),
    ("great culture", "great_culture"),
    ("no growth", "no_growth"),
    ("no career", "no_career"),
)


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text:
    - Unicode NFC normalization
    - Lowercase
    - Remove excessive whitespace
    - Remove special characters but keep Vietnamese diacritics
    - Remove URLs, emails
    """
    if not text:
        return ""

    # Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Remove HTML tags that might have leaked through
    text = re.sub(r'<[^>]+>', '', text)

    # Keep Vietnamese chars, digits, basic punctuation
    # Vietnamese diacritics: àáảãạ ăắằẳẵặ âấầẩẫậ èéẻẽẹ êếềểễệ ìíỉĩị
    # òóỏõọ ôốồổỗộ ơớờởỡợ ùúủũụ ưứừửữự ỳýỷỹỵ đ
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ.,!?;:\-]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_english_text(text: str) -> str:
    """Normalize English review text without Vietnamese term mapping."""
    if not text:
        return ""

    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)

    for pattern, replacement in _ENGLISH_CONTRACTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)

    for phrase, replacement in _ENGLISH_WORKPLACE_PHRASES:
        text = text.replace(phrase, replacement)

    text = re.sub(r"[^a-z0-9_\s.,!?;:\-']", " ", text)
    text = re.sub(r"\b(\w+)'s\b", r"\1", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove Vietnamese + English stopwords from text."""
    if not text:
        return ""
    words = text.split()
    filtered = [
        w
        for w in words
        if w.lower() not in VIETNAMESE_STOPWORDS or w.lower() in NEGATION_WORDS
    ]
    return " ".join(filtered)


def remove_english_stopwords(text: str) -> str:
    """Remove English stopwords while keeping negation and workplace sentiment cues."""
    if not text:
        return ""
    words = text.split()
    filtered = [
        w
        for w in words
        if w.lower().strip(".,!?;:-") not in ENGLISH_STOPWORDS
    ]
    return " ".join(filtered)


def tokenize_vietnamese(text: str) -> str:
    """Tokenize Vietnamese text using underthesea.
    Falls back to whitespace tokenization if underthesea is unavailable.
    """
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        logger.warning("underthesea not installed, falling back to whitespace tokenization")
        return text


def preprocess(
    text: str,
    use_tokenizer: bool = True,
    remove_sw: bool = True,
    normalize_terms: bool = True,
) -> str:
    """Full preprocessing pipeline for a single text field."""
    if not text:
        return ""
    if normalize_terms:
        text = normalize_vietnamese_terms(text)
    text = normalize_text(text)
    if use_tokenizer:
        text = tokenize_vietnamese(text)
    if remove_sw:
        text = remove_stopwords(text)
    return text


def preprocess_english(text: str, remove_sw: bool = True) -> str:
    """English preprocessing optimized for Glassdoor-style workplace reviews."""
    if not text:
        return ""
    text = normalize_english_text(text)
    if remove_sw:
        text = remove_english_stopwords(text)
    return text


def preprocess_by_language(text: str, language: str = "vi", remove_sw: bool = True) -> str:
    """Dispatch to the right language-specific preprocessing pipeline."""
    if language == "en":
        return preprocess_english(text, remove_sw=remove_sw)
    return preprocess(text, use_tokenizer=True, remove_sw=remove_sw)


# ── Batch preprocessing (DB) ─────────────────────────────────────

def preprocess_reviews(batch_size: int = 500) -> int:
    """Preprocess all reviews that haven't been processed yet.
    Updates pros_clean and cons_clean columns.
    Returns count of processed reviews.
    """
    from sqlalchemy import select
    from src.database import get_session
    from src.models import Review

    session = get_session()
    processed = 0

    try:
        # Find reviews where pros_clean IS NULL but pros IS NOT NULL (or same for cons)
        stmt = (
            select(Review)
            .where(
                (Review.pros_clean.is_(None)) & (Review.pros.isnot(None))
                | (Review.cons_clean.is_(None)) & (Review.cons.isnot(None))
            )
            .limit(batch_size)
        )

        while True:
            reviews = session.execute(stmt).scalars().all()
            if not reviews:
                break

            for review in reviews:
                if review.pros and not review.pros_clean:
                    review.pros_clean = preprocess(review.pros)
                if review.cons and not review.cons_clean:
                    review.cons_clean = preprocess(review.cons)
                processed += 1

            session.commit()
            logger.info(f"Preprocessed batch: {processed} reviews so far")

            if len(reviews) < batch_size:
                break

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    logger.info(f"Total preprocessed: {processed} reviews")
    return processed
