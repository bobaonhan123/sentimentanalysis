"""Generate slide-ready model result charts.

This script reads the existing training logs and, when available, PhoBERT
outputs from analysis/phobert_outputs/phobert_results.json. It writes charts to
analysis/ and copies them into slide/analysis and slide/public/analysis.

Run:
    py analysis/generate_slide_results.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
SLIDE_ANALYSIS_DIRS = [ROOT / "slide" / "analysis", ROOT / "slide" / "public" / "analysis"]
TRAINING_RESULTS = ANALYSIS_DIR / "training_results.json"
PHOBERT_RESULTS = ANALYSIS_DIR / "phobert_outputs" / "phobert_results.json"
RESULTS_PARTIAL = ROOT / "slide" / "partials" / "06-results.html"


def _load_runs() -> list[dict]:
    return json.loads(TRAINING_RESULTS.read_text(encoding="utf-8"))


def _latest_variant_run(runs: list[dict]) -> dict:
    variants = [r for r in runs if r.get("run_id", "").startswith("variant_run_") and "models" in r]
    if not variants:
        raise ValueError("No variant run found in analysis/training_results.json")
    return variants[-1]


def _latest_full_run(runs: list[dict]) -> dict | None:
    full_runs = [r for r in runs if "distribution_before" in r and "models" in r]
    return full_runs[-1] if full_runs else None


def _metric(result: dict, key: str) -> float:
    if key in result:
        return float(result[key])
    return float(result.get("test", {}).get(key, 0.0))


def _phobert_rows() -> list[dict]:
    if not PHOBERT_RESULTS.exists():
        return []

    payload = json.loads(PHOBERT_RESULTS.read_text(encoding="utf-8"))
    rows = []
    for source_key, label in [
        ("phobert", "PhoBERT + 3-Class"),
        ("phobert_neutralboost", "PhoBERT + NeutralBoost"),
    ]:
        result = payload.get(source_key)
        if not result:
            continue
        rows.append({
            "model": label,
            "setting": "phobert_3class",
            "accuracy": float(result["accuracy"]),
            "f1_macro": float(result["f1_macro"]),
        })
    return rows


def build_variant_rows(variant_run: dict) -> pd.DataFrame:
    records = []
    valid = {name: result for name, result in variant_run["models"].items() if "error" not in result}
    for name, result in valid.items():
        records.append({
            "model": name,
            "setting": result.get("variant", "unknown"),
            "accuracy": _metric(result, "accuracy"),
            "f1_macro": _metric(result, "f1_macro"),
        })

    records.extend(_phobert_rows())
    return pd.DataFrame(records).sort_values("f1_macro", ascending=False)


def build_selected_rows(variant_run: dict, full_run: dict | None) -> pd.DataFrame:
    rows = []
    valid = {name: result for name, result in variant_run["models"].items() if "error" not in result}

    selected_names = [
        "binary_no_neutral__TFIDF_WordChar_LinearSVC",
        "cleanlab_pruned_3class__TFIDF_WordChar_LogisticRegression",
        "original_3class__TFIDF_WordChar_LogisticRegression",
    ]
    labels = {
        "binary_no_neutral__TFIDF_WordChar_LinearSVC": "TF-IDF + LinearSVC + Binary",
        "cleanlab_pruned_3class__TFIDF_WordChar_LogisticRegression": "TF-IDF + Logistic + Cleaned 3-Class",
        "original_3class__TFIDF_WordChar_LogisticRegression": "TF-IDF + Logistic + Original 3-Class",
    }
    for name in selected_names:
        result = valid.get(name)
        if result:
            rows.append({
                "model": labels[name],
                "setting": result.get("variant", ""),
                "accuracy": _metric(result, "accuracy"),
                "f1_macro": _metric(result, "f1_macro"),
            })

    if full_run:
        full_labels = {
            "MLP_NeuralNet_Tuned": "FastText + MLP + 3-Class",
            "LSTM_NeuralNet": "FastText + LSTM + 3-Class",
            "RandomForest": "FastText + RandomForest + 3-Class",
        }
        for name, label in full_labels.items():
            result = full_run.get("models", {}).get(name)
            if result and "error" not in result:
                rows.append({
                    "model": label,
                    "setting": "original_3class",
                    "accuracy": _metric(result, "accuracy"),
                    "f1_macro": _metric(result, "f1_macro"),
                })

    rows.extend(_phobert_rows())
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)


def _family_color(model: str) -> str:
    if "PhoBERT" in model:
        return "#2563eb"
    if "MLP" in model:
        return "#f97316"
    if "FastText" in model:
        return "#a16207"
    if "Binary" in model:
        return "#16a34a"
    if "TF-IDF" in model:
        return "#64748b"
    return "#6b7280"


def _bar_chart(
    df: pd.DataFrame,
    path: Path,
    title: str,
    top_n: int | None = None,
    legend: bool = False,
) -> None:
    plot_df = df.head(top_n).iloc[::-1] if top_n else df.iloc[::-1]
    fig_h = max(4.2, 0.48 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    colors = [_family_color(model) for model in plot_df["model"]]
    bars = ax.barh(plot_df["model"], plot_df["f1_macro"], color=colors, alpha=0.9)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Macro F1")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.22)
    for bar, value in zip(bars, plot_df["f1_macro"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    if legend:
        from matplotlib.patches import Patch

        handles = [
            Patch(color="#64748b", label="TF-IDF models"),
            Patch(color="#f97316", label="FastText + MLP"),
            Patch(color="#a16207", label="Other FastText baselines"),
            Patch(color="#16a34a", label="Binary polarity setting"),
            Patch(color="#2563eb", label="PhoBERT when available"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _copy_to_slide(paths: list[Path]) -> None:
    for directory in SLIDE_ANALYSIS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, directory / path.name)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _f1(value: float) -> str:
    return f"{value:.3f}"


def _slide_label(model_name: str) -> str:
    labels = {
        "binary_no_neutral__TFIDF_WordChar_LinearSVC": "TF-IDF + LinearSVC + Binary*",
        "binary_no_neutral__TFIDF_WordChar_LogisticRegression": "TF-IDF + Logistic + Binary*",
        "cleanlab_pruned_3class__TFIDF_WordChar_LogisticRegression": "TF-IDF + Logistic + Cleaned 3-Class",
        "cleanlab_pruned_3class__TFIDF_WordChar_LinearSVC": "TF-IDF + LinearSVC + Cleaned 3-Class",
        "original_3class__TFIDF_WordChar_LogisticRegression": "TF-IDF + Logistic + Original 3-Class",
        "original_3class__TFIDF_WordChar_LinearSVC": "TF-IDF + LinearSVC + Original 3-Class",
        "mixed_conflict_4class__TFIDF_WordChar_LogisticRegression": "TF-IDF + Logistic + Mixed 4-Class",
        "mixed_conflict_4class__TFIDF_WordChar_LinearSVC": "TF-IDF + LinearSVC + Mixed 4-Class",
    }
    return labels.get(model_name, model_name)


def _result_table_rows(variant_df: pd.DataFrame) -> str:
    preferred = [
        "cleanlab_pruned_3class__TFIDF_WordChar_LogisticRegression",
        "original_3class__TFIDF_WordChar_LogisticRegression",
        "mixed_conflict_4class__TFIDF_WordChar_LogisticRegression",
        "binary_no_neutral__TFIDF_WordChar_LinearSVC",
    ]
    rows = []
    for model in preferred:
        match = variant_df[variant_df["model"] == model]
        if not match.empty:
            rows.append(match.iloc[0].to_dict())
    for _, row in variant_df[variant_df["model"].str.contains("PhoBERT", regex=False)].iterrows():
        rows.append(row.to_dict())

    best_f1 = max((float(row["f1_macro"]) for row in rows), default=0.0)
    html_rows = []
    for row in rows:
        model = str(row["model"])
        cls = ' class="highlight-row"' if float(row["f1_macro"]) == best_f1 else ""
        html_rows.append(
            f'          <tr{cls}>\n'
            f"            <td>{_slide_label(model)}</td>\n"
            f'            <td class="center">{_pct(float(row["accuracy"]))}</td>\n'
            f'            <td class="center">{_f1(float(row["f1_macro"]))}</td>\n'
            "          </tr>"
        )
    return "\n".join(html_rows)


def _selected_table_rows(selected_df: pd.DataFrame) -> str:
    best_f1 = float(selected_df["f1_macro"].max()) if not selected_df.empty else 0.0
    html_rows = []
    for _, row in selected_df.iterrows():
        cls = ' class="highlight-row"' if float(row["f1_macro"]) == best_f1 else ""
        html_rows.append(
            f'          <tr{cls}><td>{row["model"]}</td><td class="center">{_f1(float(row["f1_macro"]))}</td></tr>'
        )
    return "\n".join(html_rows)


def _replace_nth_tbody(html: str, index: int, rows_html: str) -> str:
    matches = list(re.finditer(r"<tbody>.*?</tbody>", html, flags=re.DOTALL))
    if index >= len(matches):
        raise ValueError(f"Could not find tbody index {index} in {RESULTS_PARTIAL}")
    match = matches[index]
    replacement = "<tbody>\n" + rows_html + "\n        </tbody>"
    return html[: match.start()] + replacement + html[match.end() :]


def update_results_partial(variant_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    html = RESULTS_PARTIAL.read_text(encoding="utf-8")
    html = _replace_nth_tbody(html, 0, _result_table_rows(variant_df))
    html = _replace_nth_tbody(html, 1, _selected_table_rows(selected_df))
    html = html.replace(
        "Selected representative models only.",
        "Selected representative models: TF-IDF variants plus FastText MLP/LSTM/RF baselines.",
    )
    RESULTS_PARTIAL.write_text(html, encoding="utf-8")


def main() -> None:
    runs = _load_runs()
    variant_run = _latest_variant_run(runs)
    full_run = _latest_full_run(runs)

    variant_df = build_variant_rows(variant_run)
    selected_df = build_selected_rows(variant_run, full_run)

    variant_csv = ANALYSIS_DIR / "slide_variant_results.csv"
    selected_csv = ANALYSIS_DIR / "slide_selected_model_results.csv"
    variant_df.to_csv(variant_csv, index=False, encoding="utf-8-sig")
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    variant_chart = ANALYSIS_DIR / "chart_variant_results.png"
    selected_chart = ANALYSIS_DIR / "chart_selected_model_comparison.png"
    _bar_chart(variant_df, variant_chart, "Label Variant Experiments by Macro F1", top_n=10)
    _bar_chart(
        selected_df,
        selected_chart,
        "Selected Model Comparison: TF-IDF vs FastText MLP/LSTM/RF",
        top_n=None,
        legend=True,
    )
    _copy_to_slide([variant_chart, selected_chart])
    update_results_partial(variant_df, selected_df)

    print(f"Variant table: {variant_csv}")
    print(f"Selected table: {selected_csv}")
    print(f"PhoBERT included: {PHOBERT_RESULTS.exists()}")
    print(f"Charts: {variant_chart}, {selected_chart}")


if __name__ == "__main__":
    main()
