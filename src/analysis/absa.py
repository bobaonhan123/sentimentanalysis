"""Aspect-Based Sentiment Analysis (ABSA) — Banking / Financial Services domain.

Rule-based ATE+ACD (keyword matching) and OTE+ASC (±3-token sentiment lexicon).
Reads analysis/1900_export_reviews.csv, writes summary CSV and PNG charts.

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
DEFAULT_CSV = _ROOT / "analysis" / "1900_export_reviews.csv"
DEFAULT_OUT = _ROOT / "analysis"

# ── Domain definition ────────────────────────────────────────────────────────

DOMAIN_DESCRIPTION = (
    "In the banking and financial services sector, attracting and retaining skilled "
    "employees — especially technology talent — is a critical competitive challenge. "
    "ABSA lets HR managers go beyond a single star rating to understand which workplace "
    "dimensions are driving satisfaction or turnover. Granular aspect-level insight "
    "enables targeted, cost-effective interventions rather than broad culture overhauls. "
    "Banks competing for fintech-savvy developers need to know which specific pain points "
    "are weakening their employer brand on platforms like ITViec."
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

# ── OTE sentiment lexicon ────────────────────────────────────────────────────

POSITIVE_WORDS: list[str] = [
    "tốt", "tuyệt", "ổn", "tích cực", "nhiệt tình", "rõ ràng",
    "minh bạch", "công bằng", "hỗ trợ", "vui", "thân thiện", "chuyên nghiệp",
    "năng động", "cởi mở", "hợp lý", "xứng đáng", "phù hợp", "ổn định",
    "tốt bụng", "nhanh", "hiệu quả", "tuyệt vời", "khá ổn", "học được",
    "tốt đẹp", "hài lòng", "thoải mái", "cạnh tranh", "tận tâm", "quan tâm",
]

NEGATIVE_WORDS: list[str] = [
    "tệ", "kém", "chậm", "áp lực", "stress", "thấp", "thiếu", "ràng buộc",
    "rườm rà", "không rõ ràng", "bất công", "drama", "độc đoán", "chính trị",
    "toxic", "nhàm", "không ổn", "khắc khe", "ì ạch", "trễ", "cũ kỹ",
    "không học được", "không phát triển", "thất vọng", "khó khăn",
    "không có", "quá tải", "mệt", "chán", "không minh bạch",
    "không công bằng", "không hợp lý", "khó",
]

# Sorted longest-first so multi-word expressions match before substrings
_ALL_OPINION = sorted(POSITIVE_WORDS + NEGATIVE_WORDS, key=len, reverse=True)
_POSITIVE_SET = set(POSITIVE_WORDS)


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
            for sent in _split_sentences(str(row[col])):
                records.append({
                    "review_id": idx,
                    "company": row["company"],
                    "rating": row["rating"],
                    "source": col,
                    "sentence": sent.lower(),
                })
    return pd.DataFrame(records)


# ── Step 2: ATE + ACD ────────────────────────────────────────────────────────

def _extract_aspects(sentence: str) -> list[tuple[str, str]]:
    """Return [(aspect, matched_keyword), ...] — one match per aspect per sentence."""
    found = []
    for aspect, keywords in ASPECT_RULES.items():
        for kw in keywords:
            if kw in sentence:
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

def _find_opinion(sentence: str, keyword: str, window: int = 3) -> tuple[str | None, str]:
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
        if ow in context:
            dist = abs(context.find(ow) - mid)
            if dist < best_dist:
                best_dist = dist
                best_word = ow
                best_sentiment = "Positive" if ow in _POSITIVE_SET else "Negative"

    return best_word, best_sentiment


def _run_ote_asc(aspects_df: pd.DataFrame) -> pd.DataFrame:
    opinions, sentiments = [], []
    for _, row in aspects_df.iterrows():
        ow, sent = _find_opinion(row["sentence"], row["matched_keyword"])
        if ow is None:
            # Source-column fallback
            if row["source"] == "pros":
                sent = "Positive"
            elif row["source"] == "cons":
                sent = "Negative"
            else:
                sent = "Neutral"
        opinions.append(ow)
        sentiments.append(sent)

    result = aspects_df.copy()
    result["opinion_word"] = opinions
    result["sentiment"] = sentiments
    return result


# ── Step 4: summary table ────────────────────────────────────────────────────

def _build_summary(aspects_df: pd.DataFrame) -> pd.DataFrame:
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
    ax.set_title("ABSA: Positive vs Negative Mentions per Aspect\n(Banking Employee Reviews)")
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
        df[col] = df[col].fillna("")
    logger.info(f"Loaded {len(df):,} reviews")

    # 2. Sentence splitting
    logger.info("Splitting into sentence fragments...")
    sentences_df = _build_sentence_records(df)
    logger.info(f"Sentence fragments: {len(sentences_df):,}")

    # 3. ATE + ACD
    logger.info("Running ATE + ACD (keyword matching)...")
    aspects_df = _run_ate_acd(sentences_df)
    logger.info(f"Aspect mentions: {len(aspects_df):,}")
    logger.info(f"Distribution:\n{aspects_df['aspect'].value_counts().to_string()}")

    # 4. OTE + ASC
    logger.info("Running OTE + ASC (±3-token sentiment lexicon)...")
    aspects_df = _run_ote_asc(aspects_df)
    logger.info(f"Sentiment distribution:\n{aspects_df['sentiment'].value_counts().to_string()}")

    # 5. Summary table
    summary = _build_summary(aspects_df)

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
        "charts": [str(p) for p in chart_paths],
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
    p("ABSA SUMMARY - Banking / Financial Services Employee Reviews")
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
