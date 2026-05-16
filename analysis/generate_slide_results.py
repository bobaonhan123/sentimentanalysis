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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.labeling import LABEL_NAMES, load_labeled_data, rating_to_sentiment

ANALYSIS_DIR = ROOT / "analysis"
SLIDE_ANALYSIS_DIRS = [ROOT / "slide" / "analysis", ROOT / "slide" / "public" / "analysis"]
TRAINING_RESULTS = ANALYSIS_DIR / "training_results.json"
PHOBERT_RESULTS = ANALYSIS_DIR / "phobert_outputs" / "phobert_results.json"
RESULTS_PARTIAL = ROOT / "slide" / "partials" / "06-results.html"
REVIEWS_CSV = ROOT / "data_post_processing" / "1900_export_reviews.csv"

# Fallback values are the exported PhoBERT results currently used by the deck.
# With-neutral values come from the PhoBERT summary table; no-neutral values were
# provided from the binary PhoBERT run and only include metrics that were reported.
PHOBERT_WITH_NEUTRAL_FALLBACK = {
    "phobert": {
        "accuracy": 0.8220,
        "f1_macro": 0.7781,
        "f1_weighted": 0.8269,
        "precision_macro": 0.7742,
        "recall_macro": 0.7847,
    },
    "phobert_neutralboost": {
        "accuracy": 0.8240,
        "f1_macro": 0.7793,
        "f1_weighted": 0.8284,
        "precision_macro": 0.7757,
        "recall_macro": 0.7851,
        "neutral_boost": 0.9,
    },
}
PHOBERT_BINARY_FALLBACK = [
    {
        "model": "PhoBERT_Binary",
        "accuracy": 0.9646,
        "f1_macro": 0.9575,
        "negative_f1": 0.9402,
        "positive_f1": 0.9749,
    },
    {
        "model": "PhoBERT_Binary_Threshold_0.68",
        "accuracy": 0.9663,
        "f1_macro": 0.9596,
        "negative_f1": 0.9431,
        "positive_f1": 0.9760,
        "threshold": 0.68,
    },
]


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


def _metric_or_none(result: dict | None, key: str) -> float | None:
    if not result or "error" in result:
        return None
    if key in result:
        return float(result[key])
    value = result.get("test", {}).get(key)
    return None if value is None else float(value)


def _load_phobert_payload() -> dict:
    payload = {}
    if PHOBERT_RESULTS.exists():
        payload = json.loads(PHOBERT_RESULTS.read_text(encoding="utf-8"))

    for key, value in PHOBERT_WITH_NEUTRAL_FALLBACK.items():
        payload.setdefault(key, value)
    payload.setdefault("phobert_binary_variants", PHOBERT_BINARY_FALLBACK)
    return payload


def _best_phobert_with_neutral(payload: dict | None = None) -> tuple[str, dict] | tuple[None, None]:
    payload = payload or _load_phobert_payload()
    candidates = []
    for source_key, label in [
        ("phobert", "PhoBERT"),
        ("phobert_neutralboost", "PhoBERT + NeutralBoost 0.9"),
    ]:
        result = payload.get(source_key)
        if result:
            candidates.append((float(result.get("f1_macro", 0.0)), label, result))
    if not candidates:
        return None, None
    _, label, result = max(candidates, key=lambda item: item[0])
    return label, result


def _best_phobert_binary(payload: dict | None = None) -> dict | None:
    payload = payload or _load_phobert_payload()
    variants = payload.get("phobert_binary_variants") or PHOBERT_BINARY_FALLBACK
    valid = [row for row in variants if row and "f1_macro" in row]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row["f1_macro"]))


def _phobert_rows() -> list[dict]:
    payload = _load_phobert_payload()
    rows = []
    for source_key, label in [
        ("phobert", "PhoBERT"),
        ("phobert_neutralboost", "PhoBERT + NeutralBoost 0.9"),
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
    for result in payload.get("phobert_binary_variants", []):
        rows.append({
            "model": result["model"].replace("_", " "),
            "setting": "binary_no_neutral",
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
        "binary_no_neutral__TFIDF_WordChar_LinearSVC": "TF-IDF Word+Char (Binary)",
        "cleanlab_pruned_3class__TFIDF_WordChar_LogisticRegression": "TF-IDF Word+Char (Cleaned)",
        "original_3class__TFIDF_WordChar_LogisticRegression": "TF-IDF Word+Char (Original)",
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
            "MLP_NeuralNet_Tuned": "FastText+MLP",
            "LSTM_NeuralNet": "FastText+LSTM",
            "RandomForest": "FastText+RF",
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


def _phobert_metric(key: str) -> float | None:
    _, result = _best_phobert_with_neutral()
    if not result:
        return None
    if key == "recall_macro" and key not in result:
        return None
    return float(result[key]) if key in result else None


def build_experiment_rows(variant_run: dict, full_run: dict | None) -> pd.DataFrame:
    """Build the slide table requested by the deck: 7 target model families."""
    full_models = full_run.get("models", {}) if full_run else {}
    variant_models = variant_run.get("models", {})
    specs = [
        ("Logistic Regression", "LogisticRegression", "binary_no_neutral__TFIDF_WordChar_LogisticRegression"),
        ("Linear SVC", "LinearSVC", "binary_no_neutral__TFIDF_WordChar_LinearSVC"),
        ("Random Forest", "RandomForest", None),
        ("Gaussian NB", "GaussianNB", None),
        ("FastText+MLP", "MLP_NeuralNet_Tuned", "binary_no_neutral__TF-IDF_WordChar_MLP"),
        ("FastText+LSTM", "LSTM_NeuralNet", "binary_no_neutral__TF-IDF_WordChar_LSTM"),
        ("PhoBERT", None, None),
    ]

    rows = []
    for label, full_key, binary_key in specs:
        full_result = full_models.get(full_key) if full_key else None
        binary_result = variant_models.get(binary_key) if binary_key else None

        if label == "PhoBERT":
            with_acc = _phobert_metric("accuracy")
            with_recall = _phobert_metric("recall_macro")
            with_f1 = _phobert_metric("f1_macro")
            binary_result = _best_phobert_binary()
        else:
            with_acc = _metric_or_none(full_result, "accuracy")
            with_recall = _metric_or_none(full_result, "recall_macro")
            with_f1 = _metric_or_none(full_result, "f1_macro")

        rows.append({
            "model": label,
            "with_neutral_accuracy": with_acc,
            "with_neutral_recall": with_recall,
            "with_neutral_f1": with_f1,
            "no_neutral_accuracy": _metric_or_none(binary_result, "accuracy"),
            "no_neutral_recall": _metric_or_none(binary_result, "recall_macro"),
            "no_neutral_f1": _metric_or_none(binary_result, "f1_macro"),
        })

    return pd.DataFrame(rows)


def _family_color(model: str) -> str:
    if "PhoBERT" in model:
        return "#2563eb"
    if "MLP" in model:
        return "#f97316"
    if "FastText" in model:
        return "#a16207"
    if "Binary" in model:
        return "#16a34a"
    if "TF-IDF" in model or "TFIDF" in model:
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
    labels = [_slide_label(str(model)).replace("*", "") for model in plot_df["model"]]
    bars = ax.barh(labels, plot_df["f1_macro"], color=colors, alpha=0.9)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Macro F1")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.22)
    for bar, value in zip(bars, plot_df["f1_macro"]):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    if legend:
        from matplotlib.patches import Patch

        handles = [
            Patch(color="#64748b", label="TF-IDF Word+Char"),
            Patch(color="#f97316", label="FastText+MLP"),
            Patch(color="#a16207", label="Other FastText baselines"),
            Patch(color="#16a34a", label="Binary polarity setting"),
            Patch(color="#2563eb", label="PhoBERT"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _experiment_f1_chart(df: pd.DataFrame, path: Path) -> None:
    labels = df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.34
    with_f1 = df["with_neutral_f1"].astype(float).to_numpy()
    no_f1 = df["no_neutral_f1"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars1 = ax.bar(x - width / 2, with_f1, width, label="With neutral", color="#2563eb", alpha=0.88)
    bars2 = ax.bar(x + width / 2, no_f1, width, label="No neutral", color="#16a34a", alpha=0.88)
    ax.set_ylim(0.55, 1.03)
    ax.set_ylabel("Macro F1")
    ax.set_title("7 Target Models: Macro F1 by Neutral Setting")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False)

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _find_convergence_history(runs: list[dict]) -> tuple[str, dict] | None:
    candidates: list[tuple[float, str, dict]] = []
    for run in runs:
        for name, result in run.get("models", {}).items():
            history = result.get("history") if isinstance(result, dict) else None
            if history and ("val_accuracy" in history or "val_loss" in history):
                candidates.append((_metric_or_none(result, "f1_macro") or 0.0, name, history))
    if not candidates:
        return None
    _, name, history = max(candidates, key=lambda item: item[0])
    return name, history


def _convergence_chart(runs: list[dict], path: Path) -> bool:
    found = _find_convergence_history(runs)
    if not found:
        return False
    name, history = found
    epochs = np.arange(1, len(next(iter(history.values()))) + 1)

    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    if "accuracy" in history:
        ax1.plot(epochs, history["accuracy"], marker="o", color="#2563eb", label="Train accuracy")
    if "val_accuracy" in history:
        ax1.plot(epochs, history["val_accuracy"], marker="o", color="#16a34a", label="Validation accuracy")
    ax1.set_ylim(0.45, 1.0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.22)

    ax2 = ax1.twinx()
    if "loss" in history:
        ax2.plot(epochs, history["loss"], linestyle="--", color="#f97316", label="Train loss")
    if "val_loss" in history:
        ax2.plot(epochs, history["val_loss"], linestyle="--", color="#ef4444", label="Validation loss")
    ax2.set_ylabel("Loss")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right", fontsize=8)
    ax1.set_title(f"Best Neural Model Convergence: {_slide_label(name)}")

    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _copy_to_slide(paths: list[Path]) -> None:
    for directory in SLIDE_ANALYSIS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, directory / path.name)


def _distribution_frame(counts: pd.Series, label: str) -> pd.DataFrame:
    order = ["negative", "neutral", "positive"]
    counts = counts.reindex(order).fillna(0).astype(int)
    total = int(counts.sum())
    return pd.DataFrame({
        "source": label,
        "class": order,
        "count": [int(counts[c]) for c in order],
        "pct": [float(counts[c]) / max(total, 1) * 100 for c in order],
    })


def build_label_distribution_comparison() -> pd.DataFrame:
    raw = pd.read_csv(REVIEWS_CSV)
    rating_counts = raw["rating"].apply(rating_to_sentiment).map(LABEL_NAMES).value_counts()

    weak = load_labeled_data(REVIEWS_CSV)
    weak_counts = weak["sentiment_name"].value_counts()

    out = pd.concat([
        _distribution_frame(rating_counts, "Rating-only"),
        _distribution_frame(weak_counts, "Weak label"),
    ], ignore_index=True)
    changed = int((weak["sentiment"] != weak["rating"].apply(rating_to_sentiment)).sum())
    out["changed_count"] = changed
    out["changed_pct"] = changed / max(len(weak), 1) * 100
    return out


def _label_distribution_chart(df: pd.DataFrame, path: Path) -> None:
    colors = {"negative": "#F44336", "neutral": "#9E9E9E", "positive": "#4CAF50"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for ax, source in zip(axes, ["Rating-only", "Weak label"]):
        part = df[df["source"] == source].copy()
        labels = [c.capitalize() for c in part["class"]]
        values = part["count"].tolist()
        pie_colors = [colors[c] for c in part["class"]]
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=pie_colors,
            textprops={"fontsize": 9},
        )
        ax.set_title(f"{source} distribution", fontsize=12, fontweight="bold")

    changed_count = int(df["changed_count"].iloc[0])
    changed_pct = float(df["changed_pct"].iloc[0])
    fig.suptitle(
        f"Class Distribution Before vs After Keyword/ABSA Weak Labeling\n"
        f"{changed_count:,} reviews re-labeled ({changed_pct:.1f}%)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _f1(value: float) -> str:
    return f"{value:.3f}"


def _metric_cell(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def _slide_label(model_name: str) -> str:
    labels = {
        "binary_no_neutral__TFIDF_WordChar_LinearSVC": "TF-IDF Word+Char (Binary)*",
        "binary_no_neutral__TFIDF_WordChar_LogisticRegression": "TF-IDF Word+Char (Binary)",
        "cleanlab_pruned_3class__TFIDF_WordChar_LogisticRegression": "TF-IDF Word+Char (Cleaned)",
        "cleanlab_pruned_3class__TFIDF_WordChar_LinearSVC": "TF-IDF Word+Char (Cleaned)",
        "original_3class__TFIDF_WordChar_LogisticRegression": "TF-IDF Word+Char (Original)",
        "original_3class__TFIDF_WordChar_LinearSVC": "TF-IDF Word+Char (Original)",
        "mixed_conflict_4class__TFIDF_WordChar_LogisticRegression": "TF-IDF Word+Char (Mixed)",
        "mixed_conflict_4class__TFIDF_WordChar_LinearSVC": "TF-IDF Word+Char (Mixed)",
        "MLP_NeuralNet_Tuned": "FastText+MLP",
        "LSTM_NeuralNet": "FastText+LSTM",
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


def _experiment_table_rows(experiment_df: pd.DataFrame) -> str:
    """Generate HTML table rows sorted by performance and with best results highlighted.
    Sorts by no_neutral_f1 (descending), then by with_neutral_f1 (descending).
    """
    # Sort by best performance metrics
    sorted_df = experiment_df.sort_values(
        by=['no_neutral_f1', 'with_neutral_f1'],
        ascending=[False, False],
        na_position='last'
    )
    
    # Find the best scores overall
    best_overall = max(
        [
            value
            for value in pd.concat([sorted_df["with_neutral_f1"], sorted_df["no_neutral_f1"]]).tolist()
            if value is not None and not pd.isna(value)
        ],
        default=0.0,
    )
    
    rows = []
    for _, row in sorted_df.iterrows():
        row_best = any(
            value is not None and not pd.isna(value) and abs(float(value) - best_overall) < 1e-9
            for value in [row["with_neutral_f1"], row["no_neutral_f1"]]
        )
        cls = ' class="highlight-row"' if row_best else ""
        rows.append(
            f'          <tr{cls}>\n'
            f"            <td><strong>{row['model']}</strong></td>\n"
            f'            <td class="center">{_highlight_if_best(row["with_neutral_accuracy"], best_overall if row["with_neutral_f1"] and abs(float(row["with_neutral_f1"]) - best_overall) < 1e-9 else None)}</td>\n'
            f'            <td class="center">{_highlight_if_best(row["with_neutral_recall"], best_overall if row["with_neutral_f1"] and abs(float(row["with_neutral_f1"]) - best_overall) < 1e-9 else None)}</td>\n'
            f'            <td class="center">{_metric_cell_bold(row["with_neutral_f1"], best_overall)}</td>\n'
            f'            <td class="center">{_highlight_if_best(row["no_neutral_accuracy"], best_overall if row["no_neutral_f1"] and abs(float(row["no_neutral_f1"]) - best_overall) < 1e-9 else None)}</td>\n'
            f'            <td class="center">{_highlight_if_best(row["no_neutral_recall"], best_overall if row["no_neutral_f1"] and abs(float(row["no_neutral_f1"]) - best_overall) < 1e-9 else None)}</td>\n'
            f'            <td class="center">{_metric_cell_bold(row["no_neutral_f1"], best_overall)}</td>\n'
            "          </tr>"
        )
    return "\n".join(rows)


def _highlight_if_best(value, best):
    """Helper to format a metric cell, optionally in bold if it's part of the best result."""
    cell = _metric_cell(value)
    if best is not None and cell != "-":
        return f"<strong>{cell}</strong>"
    return cell


def _metric_cell_bold(value, best_overall):
    """Format a metric cell with bold if it matches the best overall."""
    if value is None or pd.isna(value):
        return "-"
    val_float = float(value)
    cell = f"{val_float:.3f}"
    if abs(val_float - best_overall) < 1e-9:
        return f"<strong style='color:#16a34a'>{cell}</strong>"
    return cell


def update_results_partial(experiment_df: pd.DataFrame, has_convergence: bool) -> None:
    convergence_slide = ""
    if has_convergence:
        convergence_slide = """
<!-- SLIDE: Model Convergence -->
<section>
  <h2>Best Neural Model Convergence</h2>
  <img
    src="analysis/chart_model_convergence.png?v=1"
    style="width:100%; max-height:555px; object-fit:contain; border-radius:10px"
    alt="Training and validation convergence chart"
  />
  <div class="chart-caption">
    Classical ML models do not have epoch curves, so this chart uses the best neural run with saved training history.
  </div>
</section>
"""

    html = f"""<!-- SLIDE: Experiment Results Table -->
<section>
  <h2>Experiment Results: 7 Target Models</h2>
  <p style="font-size: 0.6em; color: var(--c2); margin-bottom: 8px">
    Same evaluation focus: Accuracy, Macro Recall, and Macro F1. Best deployable polarity result is highlighted.
  </p>
  <table style="font-size:0.38em">
    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th colspan="3" class="center">With neutral label</th>
        <th colspan="3" class="center">No neutral label</th>
      </tr>
      <tr>
        <th class="center">Acc.</th>
        <th class="center">Recall</th>
        <th class="center">F1</th>
        <th class="center">Acc.</th>
        <th class="center">Recall</th>
        <th class="center">F1</th>
      </tr>
    </thead>
    <tbody>
{_experiment_table_rows(experiment_df)}
    </tbody>
  </table>
  <div class="chart-caption">
    PhoBERT with-neutral uses NeutralBoost 0.9. PhoBERT no-neutral uses the threshold 0.68 binary run; recall is left blank where it was not exported.
  </div>
</section>

<!-- SLIDE: Experiment F1 Chart -->
<section>
  <h2>Consistent Model Comparison</h2>
  <div class="cols" style="align-items:center">
    <div class="col-wide">
      <img
        src="analysis/chart_experiment_f1_comparison.png?v=1"
        style="width:100%; max-height:535px; object-fit:contain; border-radius:10px"
        alt="Macro F1 comparison for seven target models"
      />
    </div>
    <div class="col-narrow">
      <div class="card">
        <div class="card-title">Main Reading</div>
        <p style="font-size:0.54em; margin:0">
          Removing neutral gives the strongest polarity classifier; PhoBERT is now the best no-neutral model by macro F1.
        </p>
      </div>
      <div class="card" style="margin-top:10px">
        <div class="card-title">Model Scope</div>
        <p style="font-size:0.54em; margin:0">
          The comparison focuses on Logistic Regression, Linear SVC, Random Forest, Gaussian NB, FastText+MLP, FastText+LSTM, and PhoBERT.
        </p>
      </div>
    </div>
  </div>
</section>
{convergence_slide}
"""
    RESULTS_PARTIAL.write_text(html, encoding="utf-8")


def main() -> None:
    runs = _load_runs()
    variant_run = _latest_variant_run(runs)
    full_run = _latest_full_run(runs)

    variant_df = build_variant_rows(variant_run)
    selected_df = build_selected_rows(variant_run, full_run)
    experiment_df = build_experiment_rows(variant_run, full_run)
    label_df = build_label_distribution_comparison()

    variant_csv = ANALYSIS_DIR / "slide_variant_results.csv"
    selected_csv = ANALYSIS_DIR / "slide_selected_model_results.csv"
    label_csv = ANALYSIS_DIR / "slide_label_distribution_comparison.csv"
    experiment_csv = ANALYSIS_DIR / "slide_experiment_comparison.csv"
    variant_df.to_csv(variant_csv, index=False, encoding="utf-8-sig")
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    experiment_df.to_csv(experiment_csv, index=False, encoding="utf-8-sig")
    label_df.to_csv(label_csv, index=False, encoding="utf-8-sig")

    variant_chart = ANALYSIS_DIR / "chart_variant_results.png"
    selected_chart = ANALYSIS_DIR / "chart_selected_model_comparison.png"
    label_chart = ANALYSIS_DIR / "chart_label_distribution_comparison.png"
    experiment_chart = ANALYSIS_DIR / "chart_experiment_f1_comparison.png"
    convergence_chart = ANALYSIS_DIR / "chart_model_convergence.png"
    _label_distribution_chart(label_df, label_chart)
    _experiment_f1_chart(experiment_df, experiment_chart)
    has_convergence = _convergence_chart(runs, convergence_chart)
    _bar_chart(variant_df, variant_chart, "Label Variant Experiments by Macro F1", top_n=10)
    _bar_chart(
        selected_df,
        selected_chart,
        "Selected Model Comparison: TF-IDF Word+Char, FastText, and PhoBERT",
        top_n=None,
        legend=True,
    )
    chart_paths = [variant_chart, selected_chart, label_chart, experiment_chart]
    if has_convergence:
        chart_paths.append(convergence_chart)
    _copy_to_slide(chart_paths)
    update_results_partial(experiment_df, has_convergence)

    print(f"Variant table: {variant_csv}")
    print(f"Selected table: {selected_csv}")
    print(f"Experiment table: {experiment_csv}")
    print(f"Label distribution table: {label_csv}")
    print(f"PhoBERT JSON exported: {PHOBERT_RESULTS.exists()} (fallback metrics used when JSON is absent)")
    print(f"Charts: {', '.join(str(p) for p in chart_paths)}")


if __name__ == "__main__":
    main()
