"""Vietnamese informal-term normalization and dataset scan helpers."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.common.text_repair import repair_mojibake_text

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT_DIR / "analysis"

# Keep this deliberately small and auditable. The scanner output is meant to
# show what should be promoted into this dictionary after reviewing real data.
VIETNAMESE_TERM_MAP: dict[str, str] = {
    "cty": "công ty",
    "c.ty": "công ty",
    "côngty": "công ty",
    "ok": "ổn",
    "oke": "ổn",
    "okie": "ổn",
    "okay": "ổn",
    "ko": "không",
    "k": "không",
    "khong": "không",
    "hok": "không",
    "hem": "không",
    "hong": "không",
    "kh": "không",
    "kg": "không",
    "dc": "được",
    "đc": "được",
    "duoc": "được",
    "nv": "nhân viên",
    "nlđ": "người lao động",
    "ld": "lãnh đạo",
    "sếp": "quản lý",
    "sep": "quản lý",
    "ql": "quản lý",
    "qlý": "quản lý",
    "hr": "nhân sự",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "ot": "overtime",
    "o.t": "overtime",
    "wfh": "làm việc từ xa",
    "remote": "làm việc từ xa",
    "intern": "thực tập sinh",
    "fresher": "nhân viên mới",
    "task": "công việc",
    "deadline": "hạn chót",
    "benefit": "phúc lợi",
    "benefits": "phúc lợi",
    "bonus": "thưởng",
    "review": "đánh giá",
    "training": "đào tạo",
    "trainning": "đào tạo",
    "process": "quy trình",
    "team": "nhóm",
    "support": "hỗ trợ",
    "job": "công việc",
    "project": "dự án",
    "pm": "quản lý dự án",
    "dev": "lập trình viên",
    "junior": "nhân viên mới",
    "senior": "nhân viên kinh nghiệm",
    "mentor": "người hướng dẫn",
    "onsite": "làm việc tại khách hàng",
    "outsource": "gia công phần mềm",
    "kpi": "chỉ tiêu",
    "nice": "tốt",
    "deal": "thương lượng",
    "level": "cấp bậc",
    "code": "mã nguồn",
    "work": "công việc",
    "lead": "lãnh đạo",
    "leader": "lãnh đạo",
    "manager": "quản lý",
    "mn": "mọi người",
    "mng": "mọi người",
    "ae": "anh em",
    "đn": "đồng nghiệp",
    "dongnghiep": "đồng nghiệp",
    "lương": "lương",
    "luong": "lương",
    "phuc loi": "phúc lợi",
    "lg": "lương",
    "kn": "kinh nghiệm",
    "sv": "sinh viên",
    "cx": "cũng",
    "cuxng": "cũng",
    "tot": "tốt",
    "te": "tệ",
    "thap": "thấp",
    "that vong": "thất vọng",
    "thuc tap sinh": "thực tập sinh",
    "van phong": "văn phòng",
    "moi truong": "môi trường",
    "cng ty": "công ty",
    "mtr": "môi trường",
    "mt": "môi trường",
    "môi trg": "môi trường",
    "môi tr": "môi trường",
    "mtruong": "môi trường",
    "rv": "",
    "fb": "",
    "gl": "",
    "ctt": "",
    "id": "",
}

NORMALIZED_VIETNAMESE_TERM_MAP: dict[str, str] = {
    repair_mojibake_text(term): repair_mojibake_text(replacement)
    for term, replacement in VIETNAMESE_TERM_MAP.items()
}

NEGATION_FORMS = {"ko", "k", "khong", "hok", "hem", "hong", "kh", "kg"}
REPEATED_CHAR_RE = re.compile(r"([a-zà-ỹđ])\1{2,}", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"(?u)\b[\w.+#-]{1,30}\b")


def _term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


_TERM_PATTERNS = [(_term_pattern(term), replacement) for term, replacement in sorted(
    NORMALIZED_VIETNAMESE_TERM_MAP.items(),
    key=lambda item: len(item[0]),
    reverse=True,
)]


def normalize_vietnamese_terms(text: str) -> str:
    """Expand common Vietnamese workplace slang and abbreviations."""
    if not text:
        return ""

    out = repair_mojibake_text(text)
    for pattern, replacement in _TERM_PATTERNS:
        out = pattern.sub(replacement, out)
    out = REPEATED_CHAR_RE.sub(r"\1\1", out)
    return re.sub(r"\s+", " ", out).strip()


def scan_vietnamese_terms(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    out_dir: str | Path | None = None,
    top_n: int = 250,
) -> dict:
    """Scan informal tokens and known replacements in a labeled dataset."""
    target_dir = Path(out_dir) if out_dir else ANALYSIS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    known_hits: Counter[str] = Counter()
    unknown_ascii: Counter[str] = Counter()
    repeated_hits: Counter[str] = Counter()
    label_hits: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)

    labels = df.get("sentiment_name", pd.Series(["unknown"] * len(df))).fillna("unknown")
    texts = df.get(text_col, pd.Series([""] * len(df))).fillna("")

    for idx, (text, label) in enumerate(zip(texts, labels, strict=False)):
        raw = repair_mojibake_text(text).lower()
        for term in NORMALIZED_VIETNAMESE_TERM_MAP:
            if _term_pattern(term).search(raw):
                known_hits[term] += 1
                label_hits[term][str(label)] += 1
                if len(examples[term]) < 3:
                    examples[term].append({"row": int(idx), "label": str(label), "text": raw[:240]})

        for token in TOKEN_RE.findall(raw):
            if len(token) < 2:
                continue
            if REPEATED_CHAR_RE.search(token):
                repeated_hits[token] += 1
            if token.isascii() and any(ch.isalpha() for ch in token):
                if token not in NORMALIZED_VIETNAMESE_TERM_MAP and len(token) <= 16:
                    unknown_ascii[token] += 1

    rows = []
    for term, count in known_hits.most_common():
        rows.append({
            "term": term,
            "replacement": NORMALIZED_VIETNAMESE_TERM_MAP[term],
            "count": count,
            "labels": dict(label_hits[term]),
            "examples": json.dumps(examples[term], ensure_ascii=False),
        })
    known_csv = target_dir / "vietnamese_known_terms.csv"
    pd.DataFrame(rows).to_csv(known_csv, index=False, encoding="utf-8-sig")

    unknown_csv = target_dir / "vietnamese_unknown_ascii_tokens.csv"
    pd.DataFrame(
        [{"token": token, "count": count} for token, count in unknown_ascii.most_common(top_n)]
    ).to_csv(unknown_csv, index=False, encoding="utf-8-sig")

    repeated_csv = target_dir / "vietnamese_repeated_char_tokens.csv"
    pd.DataFrame(
        [{"token": token, "count": count} for token, count in repeated_hits.most_common(top_n)]
    ).to_csv(repeated_csv, index=False, encoding="utf-8-sig")

    summary = {
        "known_terms": dict(known_hits.most_common(50)),
        "unknown_ascii_top": dict(unknown_ascii.most_common(50)),
        "repeated_char_top": dict(repeated_hits.most_common(50)),
        "known_terms_csv": str(known_csv),
        "unknown_ascii_csv": str(unknown_csv),
        "repeated_char_csv": str(repeated_csv),
    }
    summary_path = target_dir / "vietnamese_text_scan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Vietnamese text scan saved: %s", summary_path)
    return summary
