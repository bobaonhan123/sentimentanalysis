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

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIR = Path(__file__).resolve().parent
DETAILS = DIR / "absa_details.csv"
OUT_REP_COMPANIES = DIR / "absa_representative_companies.csv"
OUT_TIME_GROUP = DIR / "absa_time_group_comparison.csv"


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


def main() -> None:
    if not DETAILS.exists():
        raise FileNotFoundError(f"Missing {DETAILS}; run `python run.py absa` first.")

    df = pd.read_csv(DETAILS)
    reps = representative_companies(df)
    reps.to_csv(OUT_REP_COMPANIES, index=False, encoding="utf-8-sig")

    industry_chart = industry_aspect_chart(df)

    time_summary = time_group_comparison(df)
    time_summary.to_csv(OUT_TIME_GROUP, index=False, encoding="utf-8-sig")
    time_chart = time_group_chart(time_summary)

    print(f"Representative companies: {OUT_REP_COMPANIES}")
    print(f"Industry aspect chart   : {industry_chart}")
    print(f"Time group comparison   : {OUT_TIME_GROUP}")
    print(f"Time group chart        : {time_chart}")


if __name__ == "__main__":
    main()
