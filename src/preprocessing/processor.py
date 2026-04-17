"""Vietnamese text preprocessing: normalize, tokenize, remove stopwords."""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ── Vietnamese stopwords (common function words) ──────────────────
# Compact set — extend as needed
VIETNAMESE_STOPWORDS: set[str] = {
    "và", "của", "là", "có", "được", "cho", "với", "trong", "này", "đã",
    "để", "từ", "một", "không", "các", "những", "khi", "tại", "theo",
    "về", "trên", "hay", "còn", "như", "hoặc", "đó", "nào", "thì",
    "cũng", "vì", "nếu", "lại", "do", "bị", "đang", "sẽ", "rất",
    "vẫn", "ra", "lên", "xuống", "vào", "nên", "mà", "nhưng", "thế",
    "tôi", "bạn", "anh", "chị", "em", "họ", "mình", "ta",
    "ở", "qua", "đến", "sau", "trước", "giữa", "dưới",
    "rồi", "hơn", "quá", "phải", "chỉ", "cái", "gì", "ai",
    "nhiều", "ít", "mỗi", "mọi", "tất", "cả",
    "thì", "là", "bởi", "vì", "nên", "mặc", "dù",
    "the", "and", "is", "of", "to", "in", "for", "a", "an",  # English stopwords mixed in
}


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


def remove_stopwords(text: str) -> str:
    """Remove Vietnamese + English stopwords from text."""
    if not text:
        return ""
    words = text.split()
    filtered = [w for w in words if w.lower() not in VIETNAMESE_STOPWORDS]
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


def preprocess(text: str, use_tokenizer: bool = True, remove_sw: bool = True) -> str:
    """Full preprocessing pipeline for a single text field."""
    if not text:
        return ""
    text = normalize_text(text)
    if use_tokenizer:
        text = tokenize_vietnamese(text)
    if remove_sw:
        text = remove_stopwords(text)
    return text


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
