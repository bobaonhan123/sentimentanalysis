"""Aspect-Based Sentiment Analysis (ABSA) for employee reviews.

Rule-based ATE+ACD (keyword matching) and OTE+ASC (±6-token sentiment lexicon).
Reads the current review CSV, writes summary CSVs and PNG charts.

Run:  python run.py absa
      python run.py absa --csv path/to/other.csv --out analysis/
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = _ROOT / "data_post_processing" / "1900_export_reviews.csv"
DEFAULT_CSV = DATA_CSV
DEFAULT_OUT = _ROOT / "analysis"

# ── Domain definition ────────────────────────────────────────────────────────

DOMAIN_DESCRIPTION = (
    "For a multi-industry employee-review dataset, ABSA lets HR and business analysts "
    "go beyond one star rating to identify the workplace dimensions that drive praise, "
    "complaints, retention risk, and employer-brand perception. The drilldowns compare "
    "aspect sentiment over time, by industry, by company, by company review-volume group, "
    "by location, and by employee status so interventions can be targeted instead of generic."
)

# ── ATE + ACD rules (≥8 keywords per aspect) ────────────────────────────────

ASPECT_RULES: dict[str, list[str]] = {
    "Salary & Benefits": [
        "lương", "thưởng", "phúc lợi", "thu nhập", "đãi ngộ",
        "hoa hồng", "tăng lương", "bảo hiểm", "trợ cấp", "thù lao",
        "lương thưởng", "chế độ", "benefit",
    ],
    "Work Environment": [
        "môi trường", "văn hóa", "đồng nghiệp", "văn phòng",
        "không khí", "teamwork", "nhóm", "tập thể", "cởi mở",
        "thân thiện", "hòa đồng", "drama", "toxic", "culture",
    ],
    "Management & Leadership": [
        "quản lý", "lãnh đạo", "sếp", "giám đốc", "trưởng phòng",
        "cấp trên", "ban lãnh đạo", "manager", "leadership",
        "ban giám đốc", "tầm nhìn", "chính sách",
    ],
    "Career Growth": [
        "thăng tiến", "học hỏi", "phát triển", "đào tạo", "lộ trình",
        "cơ hội", "kỹ năng", "training", "kinh nghiệm", "thăng chức",
        "học được", "nâng cao", "tiến bộ", "career",
    ],
    "Work-Life Balance": [
        "áp lực", "overtime", "tăng ca", "workload", "deadline",
        "cân bằng", "giờ làm", "stress", "quá tải", "nghỉ phép",
        "nghỉ ngơi", "làm thêm", "chấm công", "giờ giấc",
    ],
}

# Extra workplace aspects for the broader 10k-row multi-industry dataset.
# Kept as an update so the original five-aspect taxonomy remains intact.
ASPECT_RULES.update({
    "Job Security & Stability": [
        "ổn định", "bền vững", "hợp đồng", "thử việc", "sa thải", "layoff",
        "cắt giảm", "biến động", "nghỉ việc", "turnover", "giữ người",
        "chính thức", "nhân sự thay đổi", "tái cơ cấu",
    ],
    "Process & Communication": [
        "quy trình", "thủ tục", "giao tiếp", "truyền thông", "phối hợp",
        "minh bạch", "rõ ràng", "paperwork", "workflow", "báo cáo",
        "trao đổi", "thông tin", "chậm phản hồi", "phê duyệt",
    ],
    "Technology & Product": [
        "công nghệ", "dự án", "sản phẩm", "code", "coding", "tech stack",
        "hệ thống", "phần mềm", "tool", "tools", "legacy", "architecture",
        "platform", "server", "database", "sản phẩm",
    ],
    "Facilities & Tools": [
        "thiết bị", "máy tính", "laptop", "văn phòng", "cơ sở vật chất",
        "bãi xe", "canteen", "đồ ăn", "chỗ ngồi", "phòng họp", "wifi",
        "máy lạnh", "trang thiết bị", "office",
    ],
    "Recruitment & Onboarding": [
        "tuyển dụng", "phỏng vấn", "onboarding", "nhận việc", "offer",
        "hr", "nhân sự", "đào tạo đầu vào", "hội nhập", "probation",
        "intern", "thực tập", "mentor", "buddy",
    ],
})

# ── OTE sentiment lexicon ────────────────────────────────────────────────────

POSITIVE_WORDS: list[str] = [
    "tốt", "tuyệt", "ổn", "tích cực", "nhiệt tình", "rõ ràng",
    "minh bạch", "công bằng", "hỗ trợ", "vui", "thân thiện", "chuyên nghiệp",
    "năng động", "cởi mở", "hợp lý", "xứng đáng", "phù hợp", "ổn định",
    "tốt bụng", "nhanh", "hiệu quả", "tuyệt vời", "khá ổn", "học được",
    "tốt đẹp", "hài lòng", "thoải mái", "cạnh tranh", "tận tâm", "quan tâm",
    "không phàn nàn", "không có gì phàn nàn", "không có gì để chê",
    "không có gì cần phàn nàn", "chịu khó", "kiên trì",
]

NEGATIVE_WORDS: list[str] = [
    "không tốt", "không ổn", "không phù hợp", "không xứng đáng",
    "tệ", "kém", "chậm", "áp lực", "stress", "thấp", "thiếu", "ràng buộc",
    "rườm rà", "không rõ ràng", "bất công", "drama", "độc đoán", "chính trị",
    "toxic", "nhàm", "không ổn", "khắc khe", "ì ạch", "trễ", "cũ kỹ",
    "không học được", "không phát triển", "thất vọng", "khó khăn",
    "không có", "quá tải", "mệt", "chán", "không minh bạch",
    "không công bằng", "không hợp lý", "bóc lột", "thiếu chuyên nghiệp",
]

# Sorted longest-first so multi-word expressions match before substrings
_ALL_OPINION = sorted(POSITIVE_WORDS + NEGATIVE_WORDS, key=len, reverse=True)
_POSITIVE_SET = set(POSITIVE_WORDS)
_NEGATION_WORDS = {"không", "ko", "k", "chưa", "chẳng", "không có", "chưa có"}


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match a keyword as a phrase instead of an arbitrary substring."""
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _is_negated_context(context: str, phrase: str) -> bool:
    """Detect simple Vietnamese negation shortly before an opinion phrase."""
    match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", context, flags=re.IGNORECASE)
    if not match:
        return False
    before_tokens = context[: match.start()].split()[-3:]
    before = " ".join(before_tokens)
    return any(neg in before_tokens or neg in before for neg in _NEGATION_WORDS)


def _compile_phrase_pattern(phrase: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)


_ASPECT_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    aspect: [(kw, _compile_phrase_pattern(kw)) for kw in keywords]
    for aspect, keywords in ASPECT_RULES.items()
}


# ── Step 1: preprocessing ────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split a review field into sentence fragments."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"[.;\n]+", text)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > 40:
            sentences.extend(s.strip() for s in re.split(r",\s*", part) if s.strip())
        else:
            sentences.append(part)
    return [s for s in sentences if len(s) > 3]


def _build_sentence_records(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        for col in ("pros", "cons", "advice"):
            for sent in _split_sentences(str(row.get(col, ""))):
                records.append({
                    "review_id": idx,
                    "company": row.get("company", ""),
                    "industry": row.get("industry", ""),
                    "location": row.get("location", ""),
                    "date": row.get("date", ""),
                    "employee_status": row.get("employee_status", ""),
                    "job_title": row.get("job_title", ""),
                    "rating": row.get("rating"),
                    "source": col,
                    "sentence": sent.lower(),
                })
    return pd.DataFrame(records)


# ── Step 2: ATE + ACD ────────────────────────────────────────────────────────

def _extract_aspects(sentence: str) -> list[tuple[str, str]]:
    """Return [(aspect, matched_keyword), ...] — one match per aspect per sentence."""
    found = []
    for aspect, patterns in _ASPECT_PATTERNS.items():
        for kw, pattern in patterns:
            if pattern.search(sentence):
                found.append((aspect, kw))
                break
    return found


def _run_ate_acd(sentences_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in sentences_df.iterrows():
        for aspect, keyword in _extract_aspects(row["sentence"]):
            records.append({**row.to_dict(), "aspect": aspect, "matched_keyword": keyword})
    return pd.DataFrame(records)


# ── Step 3: OTE + ASC ────────────────────────────────────────────────────────

def _find_opinion(sentence: str, keyword: str, window: int = 6) -> tuple[str | None, str]:
    """Locate nearest opinion word within ±window tokens of the keyword.

    Returns (opinion_word, sentiment). Sentiment is 'Neutral' when no opinion
    word is found in the window — caller applies source-column fallback.
    """
    tokens = sentence.split()
    kw_tokens = keyword.split()
    kw_idx: int | None = None
    for i in range(len(tokens) - len(kw_tokens) + 1):
        if tokens[i : i + len(kw_tokens)] == kw_tokens:
            kw_idx = i + len(kw_tokens) // 2
            break
    if kw_idx is None:
        kw_idx = len(tokens) // 2

    lo = max(0, kw_idx - window)
    hi = min(len(tokens), kw_idx + window + 1)
    context = " ".join(tokens[lo:hi])

    best_word: str | None = None
    best_dist = float("inf")
    best_sentiment = "Neutral"
    mid = len(context) // 2

    for ow in _ALL_OPINION:
        match = re.search(rf"(?<!\w){re.escape(ow)}(?!\w)", context, flags=re.IGNORECASE)
        if match:
            dist = abs(match.start() - mid)
            if dist < best_dist:
                best_dist = dist
                best_word = ow
                best_sentiment = "Positive" if ow in _POSITIVE_SET else "Negative"

                # A nearby negation flips simple opinion words. Explicit phrases
                # such as "không tốt" already have polarity in the lexicon.
                if not ow.startswith(("không ", "chưa ", "chẳng ")) and _is_negated_context(context, ow):
                    best_sentiment = "Negative" if best_sentiment == "Positive" else "Positive"

    return best_word, best_sentiment


def _fallback_sentiment(row: pd.Series) -> str:
    """Fallback when no local opinion word is found.

    In this dataset, the `cons` column often contains the full review text, not
    only disadvantages. Rating-aware fallback avoids forcing those records
    negative when the sentence has no explicit opinion term.
    """
    source = row.get("source")
    rating = row.get("rating")
    if source == "pros":
        return "Positive"
    if pd.notna(rating):
        if float(rating) >= 4:
            return "Positive"
        if float(rating) <= 2:
            return "Negative"
    if source == "cons":
        return "Negative"
    return "Neutral"


def _run_ote_asc(aspects_df: pd.DataFrame) -> pd.DataFrame:
    opinions, sentiments = [], []
    for _, row in aspects_df.iterrows():
        ow, sent = _find_opinion(row["sentence"], row["matched_keyword"])
        if ow is None:
            sent = _fallback_sentiment(row)
        opinions.append(ow)
        sentiments.append(sent)

    result = aspects_df.copy()
    result["opinion_word"] = opinions
    result["sentiment"] = sentiments
    return result


# ── Step 4: summary table ────────────────────────────────────────────────────

def _build_summary(aspects_df: pd.DataFrame) -> pd.DataFrame:
    if aspects_df.empty:
        return pd.DataFrame(columns=["Positive", "Negative", "Neutral", "Representative Opinion Terms"])

    pivot = (
        aspects_df.groupby(["aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Negative", "Neutral"], fill_value=0)
    )

    def _top_words(group: pd.DataFrame, n: int = 6) -> str:
        words = group["opinion_word"].dropna().tolist()
        return ", ".join(w for w, _ in Counter(words).most_common(n))

    pivot["Representative Opinion Terms"] = aspects_df.groupby("aspect").apply(
        _top_words, include_groups=False
    )
    pivot.index.name = "Aspect"
    return pivot


def _enrich_aspect_metadata(aspects_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Add date, industry, and company grouping fields used by drilldown reports."""
    if aspects_df.empty:
        return aspects_df.copy()

    result = aspects_df.copy()
    for col in ("company", "industry", "location", "employee_status", "job_title"):
        if col not in result.columns:
            result[col] = ""
        result[col] = result[col].fillna("").astype(str).str.strip().replace("", "Unknown")

    parsed_date = pd.to_datetime(result.get("date"), dayfirst=True, errors="coerce")
    result["review_date"] = parsed_date
    result["period_year"] = parsed_date.dt.year.astype("Int64").astype(str).replace("<NA>", "Unknown")
    result["period_quarter"] = parsed_date.dt.to_period("Q").astype(str).replace("NaT", "Unknown")
    result["period_month"] = parsed_date.dt.to_period("M").astype(str).replace("NaT", "Unknown")

    top_industries = result["industry"].value_counts().head(top_n).index
    result["industry_group"] = result["industry"].where(result["industry"].isin(top_industries), "Other industries")

    top_companies = result["company"].value_counts().head(top_n).index
    result["company_group"] = result["company"].where(result["company"].isin(top_companies), "Other companies")

    review_counts = result.groupby("company")["review_id"].nunique()

    def _volume_group(company: str) -> str:
        n = int(review_counts.get(company, 0))
        if n >= 50:
            return "High-review companies (50+)"
        if n >= 10:
            return "Mid-review companies (10-49)"
        return "Long-tail companies (<10)"

    result["company_volume_group"] = result["company"].map(_volume_group)
    return result


def _build_group_summary(aspects_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if aspects_df.empty:
        columns = group_cols + [
            "Positive", "Negative", "Neutral", "Total Mentions", "Review Count",
            "Positive %", "Negative %", "Neutral %",
        ]
        return pd.DataFrame(columns=columns)

    pivot = (
        aspects_df.groupby(group_cols + ["sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Negative", "Neutral"], fill_value=0)
    )
    pivot["Total Mentions"] = pivot[["Positive", "Negative", "Neutral"]].sum(axis=1)
    pivot["Review Count"] = aspects_df.groupby(group_cols)["review_id"].nunique()
    total = pivot["Total Mentions"].replace(0, 1)
    pivot["Positive %"] = (pivot["Positive"] / total * 100).round(1)
    pivot["Negative %"] = (pivot["Negative"] / total * 100).round(1)
    pivot["Neutral %"] = (pivot["Neutral"] / total * 100).round(1)
    return pivot.reset_index().sort_values(["Total Mentions", "Negative"], ascending=[False, False])


def _build_keyword_summary(aspects_df: pd.DataFrame) -> pd.DataFrame:
    if aspects_df.empty:
        return pd.DataFrame(columns=["aspect", "matched_keyword", "sentiment", "Mentions"])
    return (
        aspects_df.groupby(["aspect", "matched_keyword", "sentiment"])
        .size()
        .reset_index(name="Mentions")
        .sort_values(["aspect", "Mentions"], ascending=[True, False])
    )


def _save_drilldown_tables(aspects_df: pd.DataFrame, out_dir: Path) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    tables = {
        "by_time": _build_group_summary(aspects_df, ["period_month", "aspect"]),
        "by_quarter": _build_group_summary(aspects_df, ["period_quarter", "aspect"]),
        "by_industry": _build_group_summary(aspects_df, ["industry", "aspect"]),
        "by_industry_group": _build_group_summary(aspects_df, ["industry_group", "aspect"]),
        "by_company": _build_group_summary(aspects_df, ["company", "aspect"]),
        "by_company_group": _build_group_summary(aspects_df, ["company_group", "aspect"]),
        "by_company_volume": _build_group_summary(aspects_df, ["company_volume_group", "aspect"]),
        "by_location": _build_group_summary(aspects_df, ["location", "aspect"]),
        "by_employee_status": _build_group_summary(aspects_df, ["employee_status", "aspect"]),
        "keyword_summary": _build_keyword_summary(aspects_df),
    }

    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = out_dir / f"absa_{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
        logger.info(f"Drilldown saved: {path}")
    return paths, tables


def _build_insights(summary: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict:
    insights: dict[str, object] = {}
    if not summary.empty:
        pos = summary["Positive"]
        neg = summary["Negative"]
        total = (pos + neg + summary["Neutral"]).replace(0, 1)
        neg_pct = (neg / total * 100).round(1)
        insights["overall"] = {
            "most_praised_aspect": str(pos.idxmax()),
            "most_praised_mentions": int(pos.max()),
            "most_complained_aspect": str(neg.idxmax()),
            "most_complained_mentions": int(neg.max()),
            "highest_negative_ratio_aspect": str(neg_pct.idxmax()),
            "highest_negative_ratio_pct": float(neg_pct.max()),
        }

    by_time = tables.get("by_time", pd.DataFrame())
    if not by_time.empty:
        valid = by_time[(by_time["period_month"] != "Unknown") & (by_time["Total Mentions"] >= 10)].copy()
        if not valid.empty:
            latest_period = valid["period_month"].max()
            latest = valid[valid["period_month"] == latest_period].sort_values(
                ["Negative %", "Total Mentions"], ascending=[False, False]
            )
            if not latest.empty:
                row = latest.iloc[0]
                insights["latest_period_hotspot"] = {
                    "period_month": str(row["period_month"]),
                    "aspect": str(row["aspect"]),
                    "negative_pct": float(row["Negative %"]),
                    "mentions": int(row["Total Mentions"]),
                }

            changes = []
            for aspect, group in valid.groupby("aspect"):
                group = group.sort_values("period_month")
                if len(group) < 2:
                    continue
                first = group.head(3)
                last = group.tail(3)
                first_mentions = max(first["Total Mentions"].sum(), 1)
                last_mentions = max(last["Total Mentions"].sum(), 1)
                first_neg = first["Negative"].sum() / first_mentions * 100
                last_neg = last["Negative"].sum() / last_mentions * 100
                changes.append({
                    "aspect": str(aspect),
                    "start_negative_pct": round(float(first_neg), 1),
                    "latest_negative_pct": round(float(last_neg), 1),
                    "delta_pct": round(float(last_neg - first_neg), 1),
                    "latest_mentions": int(last["Total Mentions"].sum()),
                })
            if changes:
                changes_df = pd.DataFrame(changes)
                insights["largest_negative_ratio_increase"] = changes_df.sort_values("delta_pct", ascending=False).head(3).to_dict("records")
                insights["largest_negative_ratio_decrease"] = changes_df.sort_values("delta_pct", ascending=True).head(3).to_dict("records")

    by_industry = tables.get("by_industry", pd.DataFrame())
    if not by_industry.empty:
        valid = by_industry[by_industry["Total Mentions"] >= 20].sort_values(
            ["Negative %", "Total Mentions"], ascending=[False, False]
        )
        insights["industry_hotspots"] = valid.head(8).to_dict("records")

    by_company_volume = tables.get("by_company_volume", pd.DataFrame())
    if not by_company_volume.empty:
        valid = by_company_volume[by_company_volume["Total Mentions"] >= 20].sort_values(
            ["Negative %", "Total Mentions"], ascending=[False, False]
        )
        insights["company_volume_hotspots"] = valid.head(8).to_dict("records")

    return insights


# ── Step 5: charts ───────────────────────────────────────────────────────────

def _save_charts(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping charts")
        return []

    saved: list[Path] = []
    aspects = summary.index.tolist()
    x = list(range(len(aspects)))
    bar_w = 0.35

    # Grouped bar: Positive vs Negative
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_pos = ax.bar(
        [i - bar_w / 2 for i in x],
        summary["Positive"],
        width=bar_w, label="Positive", color="#4CAF50",
    )
    bars_neg = ax.bar(
        [i + bar_w / 2 for i in x],
        summary["Negative"],
        width=bar_w, label="Negative", color="#F44336",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=15, ha="right")
    ax.set_ylabel("Mentions")
    ax.set_title("ABSA: Positive vs Negative Mentions per Aspect\n(Multi-Industry Employee Reviews)")
    ax.legend()
    for bar in list(bars_pos) + list(bars_neg):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    p = out_dir / "absa_aspect_bar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Stacked % horizontal bar
    total = (summary["Positive"] + summary["Negative"] + summary["Neutral"]).replace(0, 1)
    pct_pos = summary["Positive"] / total * 100
    pct_neg = summary["Negative"] / total * 100
    pct_neu = summary["Neutral"] / total * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(aspects, pct_pos[aspects], color="#4CAF50", label="Positive")
    ax.barh(aspects, pct_neg[aspects], left=pct_pos[aspects], color="#F44336", label="Negative")
    ax.barh(
        aspects, pct_neu[aspects],
        left=(pct_pos + pct_neg)[aspects], color="#9E9E9E", label="Neutral",
    )
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Sentiment Distribution by Aspect (%)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    p = out_dir / "absa_aspect_pct.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Doc-level vs ABSA pie comparison
    return saved


def _save_comparison_chart(df_raw: pd.DataFrame, aspects_df: pd.DataFrame, out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    doc_sent = df_raw["rating"].apply(
        lambda r: "Positive" if r >= 4 else ("Negative" if r <= 2 else "Neutral")
    ).value_counts()
    absa_sent = aspects_df["sentiment"].value_counts()

    order = ["Positive", "Negative", "Neutral"]
    colors = ["#4CAF50", "#F44336", "#9E9E9E"]
    doc_vals = [doc_sent.get(l, 0) for l in order]
    absa_vals = [absa_sent.get(l, 0) for l in order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].pie(doc_vals, labels=order, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Document-Level Sentiment\n(from star rating)")
    axes[1].pie(absa_vals, labels=order, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("ABSA Aspect-Level Sentiment\n(rule-based)")
    plt.suptitle("Document-Level vs Aspect-Level Sentiment Comparison", fontsize=12)
    plt.tight_layout()
    p = out_dir / "absa_vs_doclevel.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _save_drilldown_charts(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not available - skipping drilldown charts")
        return []

    saved: list[Path] = []

    def _heatmap(table_name: str, row_col: str, title: str, filename: str, min_mentions: int = 20) -> None:
        table = tables.get(table_name, pd.DataFrame())
        if table.empty or row_col not in table.columns:
            return
        valid = table[table["Total Mentions"] >= min_mentions].copy()
        if valid.empty:
            return
        top_rows = (
            valid.groupby(row_col)["Total Mentions"].sum()
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        valid = valid[valid[row_col].isin(top_rows)]
        pivot = valid.pivot_table(index=row_col, columns="aspect", values="Negative %", aggfunc="mean")
        if pivot.empty:
            return
        pivot = pivot.loc[top_rows.intersection(pivot.index)]
        matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        mask = np.ma.getmaskarray(matrix)

        fig_h = max(4.5, min(9, 0.45 * len(pivot.index) + 1.8))
        fig_w = max(9, min(15, 1.15 * len(pivot.columns) + 3))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        cmap = plt.get_cmap("RdYlGn_r").copy()
        cmap.set_bad("#f1f1f1")
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not mask[i, j]:
                    val = float(matrix[i, j])
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7)
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Negative mentions (%)")
        plt.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    _heatmap(
        "by_industry_group",
        "industry_group",
        "ABSA Negative Ratio by Industry Group and Aspect",
        "absa_industry_heatmap.png",
    )
    _heatmap(
        "by_company_volume",
        "company_volume_group",
        "ABSA Negative Ratio by Company Review Volume and Aspect",
        "absa_company_volume_heatmap.png",
    )
    _heatmap(
        "by_employee_status",
        "employee_status",
        "ABSA Negative Ratio by Employee Status and Aspect",
        "absa_employee_status_heatmap.png",
        min_mentions=10,
    )

    time_df = tables.get("by_time", pd.DataFrame())
    if not time_df.empty:
        valid = time_df[(time_df["period_month"] != "Unknown") & (time_df["Total Mentions"] >= 5)].copy()
        if not valid.empty:
            top_aspects = (
                valid.groupby("aspect")["Total Mentions"].sum()
                .sort_values(ascending=False)
                .head(5)
                .index
            )
            valid = valid[valid["aspect"].isin(top_aspects)]
            valid["period_month_dt"] = pd.to_datetime(valid["period_month"] + "-01", errors="coerce")
            valid = valid.dropna(subset=["period_month_dt"]).sort_values("period_month_dt")
            if not valid.empty:
                cutoff = valid["period_month_dt"].max() - pd.DateOffset(months=35)
                valid = valid[valid["period_month_dt"] >= cutoff]
                fig, ax = plt.subplots(figsize=(12, 5))
                for aspect, group in valid.groupby("aspect"):
                    group = group.sort_values("period_month_dt")
                    ax.plot(group["period_month_dt"], group["Negative %"], marker="o", linewidth=1.6, label=aspect)
                ax.set_ylabel("Negative mentions (%)")
                ax.set_xlabel("Review month")
                ax.set_title("ABSA Negative Ratio Trend by Aspect")
                ax.set_ylim(0, 100)
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(fontsize=8, loc="upper left", ncol=2)
                fig.autofmt_xdate(rotation=35)
                plt.tight_layout()
                path = out_dir / "absa_time_trend.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)

    keyword_df = tables.get("keyword_summary", pd.DataFrame())
    if not keyword_df.empty:
        top_keywords = keyword_df.groupby(["aspect", "matched_keyword"])["Mentions"].sum().reset_index()
        top_keywords = top_keywords.sort_values("Mentions", ascending=False).head(18)
        if not top_keywords.empty:
            labels = top_keywords["aspect"] + ": " + top_keywords["matched_keyword"]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(labels[::-1], top_keywords["Mentions"].iloc[::-1], color="#607D8B")
            ax.set_xlabel("Mentions")
            ax.set_title("Top ABSA Matched Keywords")
            plt.tight_layout()
            path = out_dir / "absa_keyword_bar.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    return saved


# ── Main entrypoint ──────────────────────────────────────────────────────────

def run_absa(csv_path: str | Path | None = None, out_dir: str | Path | None = None) -> dict:
    """Full ABSA pipeline: load → ATE+ACD → OTE+ASC → summary → charts.

    Returns a dict with keys: status, summary_csv, charts, aspect_mentions, total_reviews.
    """
    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return {"status": "failed", "reason": f"csv_not_found: {csv_path}"}

    # 1. Load
    logger.info(f"Loading reviews from {csv_path}...")
    df = pd.read_csv(csv_path)
    for col in ("pros", "cons", "advice"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    for col in ("company", "industry", "location", "date", "employee_status", "job_title"):
        if col not in df.columns:
            df[col] = ""
    logger.info(f"Loaded {len(df):,} reviews")

    # 2. Sentence splitting
    logger.info("Splitting into sentence fragments...")
    sentences_df = _build_sentence_records(df)
    logger.info(f"Sentence fragments: {len(sentences_df):,}")

    # 3. ATE + ACD
    logger.info("Running ATE + ACD (keyword matching)...")
    aspects_df = _run_ate_acd(sentences_df)
    logger.info(f"Aspect mentions: {len(aspects_df):,}")
    if not aspects_df.empty:
        logger.info(f"Distribution:\n{aspects_df['aspect'].value_counts().to_string()}")

    # 4. OTE + ASC
    logger.info("Running OTE + ASC (±6-token sentiment lexicon)...")
    aspects_df = _run_ote_asc(aspects_df)
    aspects_df = _enrich_aspect_metadata(aspects_df)
    if not aspects_df.empty:
        logger.info(f"Sentiment distribution:\n{aspects_df['sentiment'].value_counts().to_string()}")

    # 5. Summary table
    summary = _build_summary(aspects_df)
    drilldown_paths, drilldown_tables = _save_drilldown_tables(aspects_df, out_dir)
    insights = _build_insights(summary, drilldown_tables)

    # 6. Save outputs
    summary_csv = out_dir / "absa_summary.csv"
    summary.to_csv(summary_csv, encoding="utf-8-sig")
    logger.info(f"Summary saved: {summary_csv}")

    details_csv = out_dir / "absa_details.csv"
    aspects_df.to_csv(details_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Details saved: {details_csv}")

    # 7. Charts
    chart_paths = _save_charts(summary, out_dir)
    comp_path = _save_comparison_chart(df, aspects_df, out_dir)
    if comp_path:
        chart_paths.append(comp_path)
    chart_paths.extend(_save_drilldown_charts(drilldown_tables, out_dir))
    for p in chart_paths:
        logger.info(f"Chart saved: {p}")

    # 8. Print summary table
    _print_summary(summary, len(df), len(aspects_df))

    return {
        "status": "success",
        "total_reviews": len(df),
        "total_sentences": len(sentences_df),
        "aspect_mentions": len(aspects_df),
        "summary_csv": str(summary_csv),
        "details_csv": str(details_csv),
        "drilldown_csvs": {name: str(path) for name, path in drilldown_paths.items()},
        "charts": [str(p) for p in chart_paths],
        "insights": insights,
        "summary": summary.drop(columns=["Representative Opinion Terms"]).to_dict(),
    }


def _print_summary(summary: pd.DataFrame, n_reviews: int, n_mentions: int) -> None:
    import sys
    out = sys.stdout

    def p(line: str = "") -> None:
        out.write(line + "\n")
        out.flush()

    sep = "=" * 72
    p(sep)
    p("ABSA SUMMARY - Multi-Industry Employee Reviews")
    p(sep)
    p(f"{'Aspect':<25} {'Positive':>10} {'Negative':>10} {'Neutral':>10}")
    p("-" * 72)
    for asp in summary.index:
        pos_c = summary.loc[asp, "Positive"]
        neg_c = summary.loc[asp, "Negative"]
        neu_c = summary.loc[asp, "Neutral"]
        p(f"{asp:<25} {pos_c:>10} {neg_c:>10} {neu_c:>10}")
    p("-" * 72)
    p(f"Total reviews:         {n_reviews:>10,}")
    p(f"Total aspect mentions: {n_mentions:>10,}")
    p(sep)

    if summary.empty:
        p()
        p("No aspect mentions were detected.")
        p(sep)
        return

    # Business interpretation
    pos = summary["Positive"]
    neg = summary["Negative"]
    total = (pos + neg).replace(0, 1)
    top_pos = pos.idxmax()
    top_neg = neg.idxmax()
    neg_ratio = (neg / total).sort_values(ascending=False)

    p()
    p("BUSINESS INTERPRETATION")
    p("-" * 72)
    p(f"Most praised aspect  : {top_pos} ({pos[top_pos]:,} positive mentions)")
    p(f"Most complained about: {top_neg} ({neg[top_neg]:,} negative mentions)")
    p()
    p("Negative ratio by aspect:")
    for asp, ratio in neg_ratio.items():
        bar = "#" * int(ratio * 20)
        p(f"  {asp:<30} {ratio:.0%}  {bar}")
    p(sep)
