"""
Frequency analysis: top-120 whitespace-split tokens from review corpus.
Run: python analysis/top120_freq.py
"""
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

CSV_PATH = Path(__file__).parent / "1900_export_reviews.csv"
TEXT_COLS = ("pros", "cons", "advice")
TOP_N = 120

# Known stopwords already confirmed — mark for reference
KNOWN_FUNCTION = {
    "có", "được", "là", "làm", "để", "cho", "sẽ", "đã", "đang", "bị",
    "phải", "nên", "cần", "muốn", "thể", "bởi", "và", "với", "nhưng",
    "nếu", "khi", "mà", "vì", "hay", "hoặc", "mặc", "dù", "tuy", "thì",
    "nào", "đây", "đó", "này", "kia", "vậy", "thế", "ấy", "đấy",
    "trong", "trên", "dưới", "sau", "trước", "giữa", "qua", "đến",
    "vào", "ra", "lên", "xuống", "ở", "tại", "theo", "từ", "về",
    "tôi", "bạn", "anh", "chị", "em", "mình", "họ", "ta", "chúng",
    "người", "ai", "các", "những", "mỗi", "mọi", "tất", "cả", "một",
    "nhiều", "ít", "hai", "ba", "mấy", "rất", "khá", "quá", "hơn",
    "cũng", "vẫn", "lại", "đi", "rồi", "nữa", "thêm", "chỉ", "chưa",
    "không", "của", "như", "nhau", "sự", "điều", "việc",
}
KNOWN_FRAGMENTS = {
    "ty", "ot", "trường", "nghiệp", "viên", "triển", "thiện", "mái",
    "thoải", "tạo", "trình", "gian", "hỏi", "nghiệm", "lý", "trợ",
    "hội", "độ", "ràng",
}
KNOWN_ENGLISH = {"the", "and", "is", "of", "to", "in", "for", "a", "an",
                 "it", "at", "be", "as", "by", "we", "on", "or", "do"}


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text


def main():
    counter: Counter = Counter()

    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in TEXT_COLS:
                text = row.get(col) or ""
                if text:
                    counter.update(clean(text).split())

    print(f"{'Rank':>4}  {'Count':>6}  {'Word':<20}  Status")
    print("-" * 60)

    for rank, (word, count) in enumerate(counter.most_common(TOP_N), 1):
        if word in KNOWN_FUNCTION:
            status = "function-word"
        elif word in KNOWN_FRAGMENTS:
            status = "fragment"
        elif word in KNOWN_ENGLISH:
            status = "english-sw"
        else:
            status = ">>> REVIEW <<<"
        print(f"{rank:>4}  {count:>6}  {word:<20}  {status}")


if __name__ == "__main__":
    main()
