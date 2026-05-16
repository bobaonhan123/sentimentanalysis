"""Generate compact ABSA segment insight tables/charts for slides.

Inputs are produced by `python run.py absa`:
- analysis/absa_details.csv
- analysis/absa_by_industry.csv

Outputs:
- analysis/absa_representative_companies.csv
- analysis/absa_industry_aspect_comparison.png
- analysis/absa_time_group_comparison.csv
- analysis/absa_time_group_comparison.png
"""
from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
SLIDE_ANALYSIS_DIRS = [ROOT / "slide" / "analysis", ROOT / "slide" / "public" / "analysis"]
DETAILS = DIR / "absa_details.csv"
OUT_REP_COMPANIES = DIR / "absa_representative_companies.csv"
OUT_TIME_GROUP = DIR / "absa_time_group_comparison.csv"
OUT_FOCUS_YEAR = DIR / "absa_focus_industry_year.csv"
OUT_FOCUS_YOY = DIR / "absa_focus_industry_yoy.csv"
OUT_FOCUS_COMPANY = DIR / "absa_focus_company_aspects.csv"
OUT_RECOMMENDATIONS = DIR / "absa_business_recommendations.csv"
OUT_BUSINESS_CASE = DIR / "absa_business_case_bmbsoft_2025.csv"


def _group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    pivot = (
        df.groupby(group_cols + ["sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Negative", "Neutral"], fill_value=0)
    )
    pivot["Total Mentions"] = pivot[["Positive", "Negative", "Neutral"]].sum(axis=1)
    pivot["Review Count"] = df.groupby(group_cols)["review_id"].nunique()
    total = pivot["Total Mentions"].replace(0, 1)
    pivot["Negative %"] = (pivot["Negative"] / total * 100).round(1)
    pivot["Positive %"] = (pivot["Positive"] / total * 100).round(1)
    return pivot.reset_index()


def _clean_periods(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["period_year"] = pd.to_numeric(result["period_year"], errors="coerce")
    return result[result["period_year"].between(2023, 2026)].copy()


def _focus_industry(df: pd.DataFrame) -> str:
    recent = _clean_periods(df)
    coverage = (
        recent.groupby(["industry", "period_year"])["review_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    coverage = coverage[(coverage.get(2023.0, 0) >= 30) & (coverage.get(2024.0, 0) >= 30) & (coverage.get(2025.0, 0) >= 30)]
    if coverage.empty:
        return recent.groupby("industry")["review_id"].nunique().sort_values(ascending=False).index[0]
    totals = recent[recent["industry"].isin(coverage.index)].groupby("industry")["review_id"].nunique()
    return str(totals.sort_values(ascending=False).index[0])


def _year_label(year: float | int) -> str:
    year_int = int(year)
    return "2026 YTD" if year_int == 2026 else str(year_int)


def _short_label(value: str, width: int = 28) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, max_lines=2, placeholder="..."))


def _copy_to_slide(paths: list[Path]) -> None:
    for directory in SLIDE_ANALYSIS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, directory / path.name)


def representative_companies(df: pd.DataFrame) -> pd.DataFrame:
    """Pick representative companies inside major industries and summarize aspect pain points."""
    top_industries = (
        df.groupby("industry")["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    major = df[df["industry"].isin(top_industries)].copy()

    company_volume = (
        major.groupby(["industry", "company"])["review_id"]
        .nunique()
        .reset_index(name="Company Review Count")
    )
    company_volume["industry_rank"] = company_volume.groupby("industry")["Company Review Count"].rank(
        method="first", ascending=False
    )
    reps = company_volume[company_volume["industry_rank"] <= 2][["industry", "company", "Company Review Count"]]

    company_aspect = _group_summary(major, ["industry", "company", "aspect"])
    company_aspect = company_aspect.merge(reps, on=["industry", "company"], how="inner")
    company_aspect = company_aspect[company_aspect["Total Mentions"] >= 10].copy()

    rows = []
    for (industry, company), group in company_aspect.groupby(["industry", "company"]):
        pain = group.sort_values(["Negative %", "Total Mentions"], ascending=[False, False]).iloc[0]
        anchor = group.sort_values(["Positive %", "Total Mentions"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "Industry": industry,
                "Representative Company": company,
                "Company Review Count": int(pain["Company Review Count"]),
                "Main Pain Aspect": pain["aspect"],
                "Pain Mentions": int(pain["Total Mentions"]),
                "Pain Negative %": float(pain["Negative %"]),
                "Positive Anchor Aspect": anchor["aspect"],
                "Anchor Mentions": int(anchor["Total Mentions"]),
                "Anchor Positive %": float(anchor["Positive %"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Industry", "Company Review Count"], ascending=[True, False])


def industry_aspect_chart(df: pd.DataFrame) -> Path:
    top_industries = (
        df.groupby("industry")["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    top_aspects = (
        df.groupby("aspect").size()
        .sort_values(ascending=False)
        .head(6)
        .index
    )
    summary = _group_summary(df[df["industry"].isin(top_industries) & df["aspect"].isin(top_aspects)], ["industry", "aspect"])
    summary = summary[summary["Total Mentions"] >= 20]
    pivot = summary.pivot_table(index="industry", columns="aspect", values="Negative %", aggfunc="mean")
    pivot = pivot.reindex(index=top_industries, columns=top_aspects)

    matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    mask = np.ma.getmaskarray(matrix)
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#f1f1f1")
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Negative Ratio by Major Industry and Aspect")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not mask[i, j]:
                ax.text(j, i, f"{float(matrix[i, j]):.0f}%", ha="center", va="center", fontsize=7)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Negative mentions (%)")
    plt.tight_layout()
    path = DIR / "absa_industry_aspect_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def time_group_comparison(df: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.to_datetime(df["review_date"], errors="coerce")
    result = df.copy()
    result["year"] = parsed.dt.year
    result = result[result["year"].notna()].copy()
    result["time_group"] = pd.cut(
        result["year"],
        bins=[0, 2023, 2024, 2025, 2026],
        labels=["<=2023", "2024", "2025", "2026 YTD"],
        include_lowest=True,
    )
    summary = _group_summary(result, ["time_group", "aspect"])
    summary = summary[summary["Total Mentions"] >= 20].copy()
    return summary.sort_values(["time_group", "Negative %"], ascending=[True, False])


def time_group_chart(summary: pd.DataFrame) -> Path:
    top_aspects = (
        summary.groupby("aspect")["Total Mentions"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )
    plot_df = summary[summary["aspect"].isin(top_aspects)]
    pivot = plot_df.pivot_table(index="time_group", columns="aspect", values="Negative %", aggfunc="mean")
    pivot = pivot.reindex(index=["<=2023", "2024", "2025", "2026 YTD"], columns=top_aspects)

    fig, ax = plt.subplots(figsize=(11, 5))
    for aspect in pivot.columns:
        ax.plot(pivot.index.astype(str), pivot[aspect], marker="o", linewidth=2, label=aspect)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Negative mentions (%)")
    ax.set_xlabel("Review period")
    ax.set_title("Negative Ratio by Time Group and Aspect")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    path = DIR / "absa_time_group_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def focus_industry_year_chart(df: pd.DataFrame, industry: str) -> tuple[pd.DataFrame, Path]:
    recent = _clean_periods(df)
    focus = recent[recent["industry"] == industry].copy()
    summary = _group_summary(focus, ["period_year"]).sort_values("period_year")
    summary["Year"] = summary["period_year"].apply(_year_label)
    summary.to_csv(OUT_FOCUS_YEAR, index=False, encoding="utf-8-sig")

    x = np.arange(len(summary))
    fig, ax1 = plt.subplots(figsize=(11.5, 5.4))
    bars = ax1.bar(x, summary["Total Mentions"], color="#cbd5e1", width=0.58, label="Aspect mentions")
    ax1.set_ylabel("Aspect mentions")
    ax1.set_ylim(0, float(summary["Total Mentions"].max()) * 1.18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["Year"])
    ax1.grid(True, axis="y", alpha=0.18)
    ax1.set_axisbelow(True)
    for bar, reviews in zip(bars, summary["Review Count"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(summary["Total Mentions"]) * 0.025,
            f"{int(reviews):,} reviews",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#475569",
        )

    ax2 = ax1.twinx()
    ax2.plot(x, summary["Negative %"], color="#ef4444", marker="o", linewidth=2.6, label="Negative %")
    ax2.set_ylim(0, max(70, float(summary["Negative %"].max()) + 12))
    ax2.set_ylabel("Negative mentions (%)")
    for xi, value in zip(x, summary["Negative %"]):
        ax2.text(xi, value + 2.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=10, color="#b91c1c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
    ax1.set_title(f"{industry}: Yearly Aspect Volume and Negative Ratio")
    plt.tight_layout()
    path = DIR / "absa_focus_industry_year_mixed.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary, path


def focus_industry_yoy_chart(df: pd.DataFrame, industry: str) -> tuple[pd.DataFrame, Path]:
    recent = _clean_periods(df)
    focus = recent[(recent["industry"] == industry) & (recent["period_year"].isin([2024.0, 2025.0]))].copy()
    summary = _group_summary(focus, ["period_year", "aspect"])
    neg = summary.pivot_table(index="aspect", columns="period_year", values="Negative %", aggfunc="mean")
    mentions = summary.pivot_table(index="aspect", columns="period_year", values="Total Mentions", aggfunc="sum")
    valid = neg.index[
        (mentions.get(2024.0, pd.Series(0, index=mentions.index)) >= 40)
        & (mentions.get(2025.0, pd.Series(0, index=mentions.index)) >= 40)
    ]
    yoy = pd.DataFrame({
        "Aspect": valid,
        "Neg% 2024": neg.loc[valid, 2024.0].values,
        "Neg% 2025": neg.loc[valid, 2025.0].values,
        "Mentions 2024": mentions.loc[valid, 2024.0].values,
        "Mentions 2025": mentions.loc[valid, 2025.0].values,
    })
    yoy["Delta pp"] = yoy["Neg% 2025"] - yoy["Neg% 2024"]
    yoy = yoy.sort_values(["Delta pp", "Mentions 2025"], ascending=[False, False]).head(8)
    yoy.to_csv(OUT_FOCUS_YOY, index=False, encoding="utf-8-sig")

    plot = yoy.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    height = 0.34
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.barh(y - height / 2, plot["Neg% 2024"], height=height, color="#93c5fd", label="2024")
    ax.barh(y + height / 2, plot["Neg% 2025"], height=height, color="#ef4444", alpha=0.88, label="2025")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["Aspect"], fontsize=9)
    ax.set_xlim(0, min(100, max(82, float(plot[["Neg% 2024", "Neg% 2025"]].max().max()) + 10)))
    ax.set_xlabel("Negative mentions (%)")
    ax.set_title(f"{industry}: Year-over-Year Aspect Risk Movement")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    for i, row in plot.iterrows():
        value = float(row["Neg% 2025"])
        delta = float(row["Delta pp"])
        ax.text(
            value + 1.2,
            i + height / 2,
            f"+{delta:.1f}pp" if delta >= 0 else f"{delta:.1f}pp",
            va="center",
            fontsize=8,
            color="#991b1b" if delta >= 0 else "#15803d",
        )
    plt.tight_layout()
    path = DIR / "absa_focus_industry_yoy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return yoy, path


def focus_company_bubble_chart(df: pd.DataFrame, industry: str) -> tuple[pd.DataFrame, Path]:
    recent = _clean_periods(df)
    focus = recent[recent["industry"] == industry].copy()
    top_companies = (
        focus.groupby("company")["review_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(6)
        .index
    )
    top_aspects = (
        focus.groupby("aspect")
        .size()
        .sort_values(ascending=False)
        .head(6)
        .index
    )
    summary = _group_summary(
        focus[focus["company"].isin(top_companies) & focus["aspect"].isin(top_aspects)],
        ["company", "aspect"],
    )
    summary = summary[summary["Total Mentions"] >= 8].copy()
    summary["Company Reviews"] = summary["company"].map(focus.groupby("company")["review_id"].nunique())
    summary.to_csv(OUT_FOCUS_COMPANY, index=False, encoding="utf-8-sig")

    companies = list(top_companies)
    aspects = list(top_aspects)
    x_lookup = {aspect: i for i, aspect in enumerate(aspects)}
    y_lookup = {company: i for i, company in enumerate(companies)}
    plot = summary.copy()
    plot["x"] = plot["aspect"].map(x_lookup)
    plot["y"] = plot["company"].map(y_lookup)
    max_mentions = max(float(plot["Total Mentions"].max()), 1.0)
    sizes = 90 + (plot["Total Mentions"] / max_mentions) * 850

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    scatter = ax.scatter(
        plot["x"],
        plot["y"],
        s=sizes,
        c=plot["Negative %"],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=80,
        alpha=0.82,
        edgecolors="#334155",
        linewidths=0.5,
    )
    for _, row in plot.iterrows():
        if row["Total Mentions"] >= 12:
            ax.text(row["x"], row["y"], f"{row['Negative %']:.0f}%", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels([_short_label(a, 18) for a in aspects], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(companies)))
    ax.set_yticklabels([_short_label(c, 24) for c in companies], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(
        f"{industry}: Company Aspect Hotspots\n"
        "color/text = negative ratio; bubble size = aspect mention volume",
        fontsize=11,
        pad=10,
    )
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.86, pad=0.14)
    cbar.set_label("Negative ratio (color, %)")

    legend_sizes = [20, 60, 120]
    handles = [
        plt.scatter([], [], s=90 + size / max_mentions * 850, color="#94a3b8", alpha=0.7, edgecolors="#334155")
        for size in legend_sizes
        if size <= max_mentions
    ]
    labels = [f"{size} mentions" for size in legend_sizes if size <= max_mentions]
    if handles:
        ax.legend(
            handles,
            labels,
            title="Mention volume\n(size)",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fontsize=8,
        )
    fig.text(
        0.5,
        0.01,
        "Bubble area shows total aspect mentions; color and the number inside each bubble show negative ratio.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#475569",
    )
    plt.tight_layout(rect=(0, 0.045, 1, 1))
    path = DIR / "absa_focus_company_bubble.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary, path


def business_case_chart(
    df: pd.DataFrame,
    industry: str,
    company: str = "BMBSOFT Vietnam Co., Ltd",
    year: int = 2025,
) -> tuple[pd.DataFrame, Path]:
    """Create one company-year recommendation chart for a concrete business case."""
    recent = _clean_periods(df)
    case = recent[
        (recent["industry"] == industry)
        & (recent["company"] == company)
        & (recent["period_year"] == float(year))
    ].copy()
    if case.empty:
        year_focus = recent[(recent["industry"] == industry) & (recent["period_year"] == float(year))].copy()
        company_volume = (
            year_focus.groupby("company")
            .agg(
                Review_Count=("review_id", "nunique"),
                Total_Mentions=("review_id", "size"),
                Negative_Mentions=("sentiment", lambda x: int((x == "Negative").sum())),
            )
            .reset_index()
        )
        company_volume["Negative %"] = (
            company_volume["Negative_Mentions"] / company_volume["Total_Mentions"].replace(0, 1) * 100
        )
        candidate = company_volume[
            (company_volume["Review_Count"] >= 20) & (company_volume["Total_Mentions"] >= 80)
        ].sort_values(["Negative_Mentions", "Negative %"], ascending=[False, False])
        if candidate.empty:
            candidate = company_volume.sort_values(["Negative_Mentions", "Total_Mentions"], ascending=[False, False])
        company = str(candidate.iloc[0]["company"])
        case = year_focus[year_focus["company"] == company].copy()

    summary = _group_summary(case, ["aspect"])
    summary = summary[summary["Total Mentions"] >= 3].copy()
    summary["Company"] = company
    summary["Year"] = year
    summary["Industry"] = industry
    summary["Negative Volume"] = summary["Negative"]
    summary = summary.sort_values(["Total Mentions", "Negative Volume"], ascending=[False, False])
    summary.to_csv(OUT_BUSINESS_CASE, index=False, encoding="utf-8-sig")

    plot = summary.head(8).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.barh(y, plot["Positive"], color="#16a34a", alpha=0.82, label="Positive")
    ax.barh(y, plot["Neutral"], left=plot["Positive"], color="#f59e0b", alpha=0.82, label="Neutral")
    ax.barh(
        y,
        plot["Negative"],
        left=plot["Positive"] + plot["Neutral"],
        color="#ef4444",
        alpha=0.88,
        label="Negative",
    )
    max_total = max(float(plot["Total Mentions"].max()), 1.0)
    ax.set_xlim(0, max_total * 1.32)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(a, 22) for a in plot["aspect"]], fontsize=9)
    ax.set_xlabel("Aspect mentions")
    ax.set_title(f"{company} ({year}) - Aspect Sentiment Mix", fontsize=12, pad=10)
    ax.grid(True, axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=8)
    for i, row in plot.iterrows():
        total = float(row["Total Mentions"])
        ax.text(
            total + max_total * 0.025,
            i,
            f"{row['Negative %']:.1f}% neg",
            va="center",
            fontsize=8,
            color="#991b1b" if row["Negative %"] >= 50 else "#475569",
        )

    review_count = case["review_id"].nunique()
    negative_ratio = (case["sentiment"].eq("Negative").sum() / max(len(case), 1) * 100)
    fig.text(
        0.5,
        0.01,
        f"{review_count:,} reviews, {len(case):,} aspect mentions, {negative_ratio:.1f}% overall negative aspect mentions.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#475569",
    )
    plt.tight_layout(rect=(0, 0.045, 1, 1))
    path = DIR / "absa_business_case_bmbsoft_2025.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary, path


def business_recommendations(
    df: pd.DataFrame,
    focus: str,
    focus_year: pd.DataFrame,
    focus_yoy: pd.DataFrame,
    focus_company: pd.DataFrame,
) -> pd.DataFrame:
    overall = _group_summary(df, ["aspect"]).set_index("aspect")
    salary = overall.loc["Salary & Benefits"]
    environment = overall.loc["Work Environment"]

    peak_year = focus_year.sort_values("Negative %", ascending=False).iloc[0]
    salary_yoy = focus_yoy[focus_yoy["Aspect"] == "Salary & Benefits"].iloc[0]
    environment_yoy = focus_yoy[focus_yoy["Aspect"] == "Work Environment"].iloc[0]

    hotspot = (
        focus_company[focus_company["aspect"].isin(["Salary & Benefits", "Work Environment"])]
        .sort_values(["Negative %", "Total Mentions"], ascending=[False, False])
        .head(2)
    )
    hotspot_text = "; ".join(
        f"{row['company']} / {row['aspect']}: {row['Negative %']:.1f}% negative"
        for _, row in hotspot.iterrows()
    )

    rows = [
        {
            "Priority": "P1",
            "Recommendation": "Run compensation and benefits benchmarking first",
            "Evidence": (
                f"Salary & Benefits has {int(salary['Negative']):,} negative mentions "
                f"and {salary['Negative %']:.1f}% negative overall; in {focus}, 2025 reached "
                f"{salary_yoy['Neg% 2025']:.1f}% negative across {int(salary_yoy['Mentions 2025']):,} mentions."
            ),
            "Business Action": "Compare salary bands, bonus timing, allowance policy, and benefits clarity against direct competitors.",
            "Success Metric": "Reduce Salary & Benefits negative ratio in the next full-period review by at least 10 percentage points.",
        },
        {
            "Priority": "P1",
            "Recommendation": "Treat work environment as a culture and management issue",
            "Evidence": (
                f"Work Environment has {int(environment['Negative']):,} negative mentions overall; "
                f"in {focus}, it rose +{environment_yoy['Delta pp']:.1f}pp from 2024 to 2025."
            ),
            "Business Action": "Run team-level pulse checks, inspect workload/manager feedback, and prioritize teams with high environment negativity.",
            "Success Metric": "Lower Work Environment negative ratio and increase positive anchor terms such as friendly, supportive, and professional.",
        },
        {
            "Priority": "P2",
            "Recommendation": "Use segment and company triage instead of one generic HR action",
            "Evidence": (
                f"{focus} is the largest recent segment and peaked at {peak_year['Negative %']:.1f}% negative "
                f"in {peak_year['Year']}. Representative hotspots: {hotspot_text}."
            ),
            "Business Action": "Create a watchlist by industry, company, year, and aspect; review underlying comments before intervention.",
            "Success Metric": "Every high-risk segment has an owner, an inspected sample of reviews, and a documented action decision.",
        },
        {
            "Priority": "P2",
            "Recommendation": "Operationalize an early-warning dashboard",
            "Evidence": "Negative ratios can move sharply year over year, so static overall averages hide risk spikes.",
            "Business Action": "Trigger review when an aspect rises by more than 10pp, exceeds 50% negative, or crosses 100 recent mentions.",
            "Success Metric": "Monthly alert list produced with aspect, segment, trend delta, and top matched keywords.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    if not DETAILS.exists():
        raise FileNotFoundError(f"Missing {DETAILS}; run `python run.py absa` first.")

    df = pd.read_csv(DETAILS)
    reps = representative_companies(df)
    reps.to_csv(OUT_REP_COMPANIES, index=False, encoding="utf-8-sig")

    industry_chart = industry_aspect_chart(df)

    time_summary = time_group_comparison(df)
    time_summary.to_csv(OUT_TIME_GROUP, index=False, encoding="utf-8-sig")
    time_chart = time_group_chart(time_summary)

    focus = _focus_industry(df)
    focus_year, focus_year_chart = focus_industry_year_chart(df, focus)
    focus_yoy, focus_yoy_chart = focus_industry_yoy_chart(df, focus)
    focus_company, focus_company_chart = focus_company_bubble_chart(df, focus)
    business_case, business_case_chart_path = business_case_chart(df, focus)
    recommendations = business_recommendations(df, focus, focus_year, focus_yoy, focus_company)
    recommendations.to_csv(OUT_RECOMMENDATIONS, index=False, encoding="utf-8-sig")
    _copy_to_slide([focus_year_chart, focus_yoy_chart, focus_company_chart, business_case_chart_path])

    print(f"Representative companies: {OUT_REP_COMPANIES}")
    print(f"Industry aspect chart   : {industry_chart}")
    print(f"Time group comparison   : {OUT_TIME_GROUP}")
    print(f"Time group chart        : {time_chart}")
    print(f"Focus industry          : {focus}")
    print(f"Focus yearly summary    : {OUT_FOCUS_YEAR} ({len(focus_year)} rows)")
    print(f"Focus YoY summary       : {OUT_FOCUS_YOY} ({len(focus_yoy)} rows)")
    print(f"Focus company summary   : {OUT_FOCUS_COMPANY} ({len(focus_company)} rows)")
    print(f"Business case summary   : {OUT_BUSINESS_CASE} ({len(business_case)} rows)")
    print(f"Business recommendations: {OUT_RECOMMENDATIONS} ({len(recommendations)} rows)")
    print(f"Focus charts            : {focus_year_chart}, {focus_yoy_chart}, {focus_company_chart}")
    print(f"Business case chart     : {business_case_chart_path}")


if __name__ == "__main__":
    main()
