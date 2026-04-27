"""Generate training report: analysis/training_results.json → report.md + charts.

Run:  python analysis/generate_report.py
      python analysis/generate_report.py --results analysis/training_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent
RESULTS_FILE = _DIR / "training_results.json"
REPORT_FILE = _DIR / "report.md"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found. Run `python run.py train` first.")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float) -> str:
    return f"{v:.4f}"


# ── chart 1: model comparison bar ────────────────────────────────────────────

def _chart_model_comparison(run: dict, out: Path) -> Path:
    models = {
        k: v for k, v in run["models"].items() if "error" not in v
    }
    names = list(models.keys())
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    labels  = ["Accuracy", "F1 Macro", "F1 Weighted", "Precision", "Recall"]
    colors  = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]

    x = np.arange(len(names))
    bar_w = 0.14
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [models[n]["test"][metric] for n in names]
        bars = ax.bar(x + i * bar_w, vals, width=bar_w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x + bar_w * 2)
    ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — Test Set\n(run: {run['run_id']})")
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(0.8, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = out / "chart_model_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── chart 2: per-class F1 heatmap ────────────────────────────────────────────

def _chart_per_class_f1(run: dict, out: Path) -> Path:
    models = {k: v for k, v in run["models"].items() if "error" not in v}
    classes = ["negative", "neutral", "positive"]
    data = []
    for name, res in models.items():
        row = []
        for cls in classes:
            row.append(res["test"]["classification_report"].get(cls, {}).get("f1-score", 0))
        data.append(row)

    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(7, len(models) * 0.7 + 1.5))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(list(models.keys()), fontsize=9)
    for i in range(len(models)):
        for j in range(len(classes)):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if 0.4 < arr[i, j] < 0.85 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Per-Class F1 Score — Test Set")
    plt.tight_layout()
    p = out / "chart_per_class_f1.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── chart 3: confusion matrix (best model) ───────────────────────────────────

def _chart_confusion_matrix(run: dict, out: Path) -> Path:
    best_name = run["best_model"]["name"]
    cm = np.array(run["models"][best_name]["test"]["confusion_matrix"])
    labels = ["Negative", "Neutral", "Positive"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name} (Test Set)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    p = out / "chart_confusion_matrix.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── chart 4: class distribution ──────────────────────────────────────────────

def _chart_class_distribution(run: dict, out: Path) -> Path:
    dist = run["distribution_before"]
    labels = [k.capitalize() for k in dist]
    counts = [dist[k]["count"] for k in dist]
    colors = ["#F44336", "#9E9E9E", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
    )
    ax.set_title(f"Class Distribution (n={sum(counts):,})")
    plt.tight_layout()
    p = out / "chart_class_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── chart 5: ABSA summary ─────────────────────────────────────────────────────

def _chart_absa(run: dict, out: Path) -> Path | None:
    absa = run.get("absa_summary")
    if not absa:
        return None

    aspects = list(absa.get("Positive", {}).keys())
    pos = [absa["Positive"].get(a, 0) for a in aspects]
    neg = [absa["Negative"].get(a, 0) for a in aspects]

    x = np.arange(len(aspects))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.bar(x - bar_w / 2, pos, width=bar_w, label="Positive", color="#4CAF50")
    bn = ax.bar(x + bar_w / 2, neg, width=bar_w, label="Negative", color="#F44336")
    for bar in list(bp) + list(bn):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=12, ha="right")
    ax.set_ylabel("Mentions")
    ax.set_title("ABSA: Aspect Sentiment Distribution")
    ax.legend()
    plt.tight_layout()
    p = out / "chart_absa_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── chart 6: run history (multi-run) ─────────────────────────────────────────

def _chart_history(runs: list[dict], out: Path) -> Path | None:
    if len(runs) < 2:
        return None

    rows = []
    for r in runs:
        bm = r.get("best_model", {})
        rows.append({
            "run": r["run_id"].replace("run_", ""),
            "F1 Macro": bm.get("f1_macro", 0),
            "Accuracy": bm.get("accuracy", 0),
            "Samples": r.get("sample_count", 0),
        })
    df = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["run"], df["F1 Macro"], "o-", color="#2196F3", label="F1 Macro")
    ax1.plot(df["run"], df["Accuracy"], "s--", color="#4CAF50", label="Accuracy")
    ax1.set_ylabel("Score")
    ax1.set_xlabel("Run")
    ax1.set_title("Training History — Best Model per Run")
    ax1.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right", fontsize=7)
    plt.tight_layout()
    p = out / "chart_history.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── markdown report ───────────────────────────────────────────────────────────

def _build_markdown(runs: list[dict], charts: dict[str, Path], out_dir: Path) -> str:
    latest = runs[-1]
    best = latest["best_model"]
    models = {k: v for k, v in latest["models"].items() if "error" not in v}
    dist = latest["distribution_before"]
    absa = latest.get("absa_summary", {})

    def img(p: Path | None) -> str:
        if p is None:
            return ""
        return f"![{p.stem}]({p.name})\n"

    lines: list[str] = []
    a = lines.append

    a("# Sentiment Analysis — Training Report")
    a(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    a(f"**Run ID:** `{latest['run_id']}`  ")
    a(f"**Timestamp:** {latest['timestamp']}")

    # ── Overview ──────────────────────────────────────────────────
    a("\n---\n## 1. Overview\n")
    a(f"| Field | Value |")
    a(f"|-------|-------|")
    a(f"| Dataset | {latest['sample_count']:,} reviews |")
    a(f"| Embedding | `{latest['embedding']}` |")
    a(f"| Feature dim | {latest['feature_dim']} (300 FastText + 5 handcrafted) |")
    a(f"| Train / Val / Test | {latest['split']['train']:,} / {latest['split']['val']:,} / {latest['split']['test']:,} |")
    a(f"| Balancing | {latest['balance_method']} |")
    a(f"| **Best model** | **{best['name']}** |")
    a(f"| **Best F1 Macro** | **{best['f1_macro']}** |")
    a(f"| **Best Accuracy** | **{best['accuracy']}** |")

    # ── Class distribution ────────────────────────────────────────
    a("\n---\n## 2. Class Distribution\n")
    a(img(charts.get("dist")))
    a("| Class | Count | % |")
    a("|-------|------:|--:|")
    for cls, info in dist.items():
        a(f"| {cls.capitalize()} | {info['count']:,} | {info['percentage']}% |")

    cw = latest.get("class_weights", {})
    if cw:
        a(f"\n**Class weights (balanced):** " +
          ", ".join(f"`{['negative','neutral','positive'][int(k)]}={v}`" for k, v in cw.items()))

    # ── Model comparison ──────────────────────────────────────────
    a("\n---\n## 3. Model Comparison (Test Set)\n")
    a(img(charts.get("compare")))
    a("| Model | Accuracy | F1 Macro | F1 Weighted | Precision | Recall |")
    a("|-------|--------:|--------:|----------:|--------:|------:|")
    for name, res in sorted(models.items(), key=lambda x: -x[1]["f1_macro"]):
        t = res["test"]
        star = " ★" if name == best["name"] else ""
        a(f"| **{name}**{star} | {_fmt(t['accuracy'])} | {_fmt(t['f1_macro'])} | "
          f"{_fmt(t['f1_weighted'])} | {_fmt(t['precision_macro'])} | {_fmt(t['recall_macro'])} |")

    # ── Per-class F1 heatmap ──────────────────────────────────────
    a("\n---\n## 4. Per-Class F1 Score\n")
    a(img(charts.get("heatmap")))
    a("| Model | Negative F1 | Neutral F1 | Positive F1 |")
    a("|-------|----------:|--------:|----------:|")
    for name, res in sorted(models.items(), key=lambda x: -x[1]["f1_macro"]):
        cr = res["test"]["classification_report"]
        neg = _fmt(cr.get("negative", {}).get("f1-score", 0))
        neu = _fmt(cr.get("neutral",  {}).get("f1-score", 0))
        pos = _fmt(cr.get("positive", {}).get("f1-score", 0))
        a(f"| {name} | {neg} | {neu} | {pos} |")

    a("\n> **Note:** Neutral class is hardest — smallest support (163 samples) and highest class imbalance.")

    # ── Best model detail ─────────────────────────────────────────
    a(f"\n---\n## 5. Best Model: {best['name']}\n")
    a(img(charts.get("cm")))
    best_res = models[best["name"]]
    cr = best_res["test"]["classification_report"]
    a("### Classification Report (Test Set)\n")
    a("| Class | Precision | Recall | F1 | Support |")
    a("|-------|--------:|------:|---:|-------:|")
    for cls in ["negative", "neutral", "positive"]:
        d = cr.get(cls, {})
        a(f"| {cls.capitalize()} | {_fmt(d.get('precision',0))} | {_fmt(d.get('recall',0))} | "
          f"{_fmt(d.get('f1-score',0))} | {int(d.get('support',0))} |")
    ma = cr.get("macro avg", {})
    a(f"| **Macro Avg** | {_fmt(ma.get('precision',0))} | {_fmt(ma.get('recall',0))} | "
      f"{_fmt(ma.get('f1-score',0))} | {int(ma.get('support',0))} |")

    # ── ABSA summary ──────────────────────────────────────────────
    if absa:
        a("\n---\n## 6. ABSA Summary (Aspect-Based Sentiment)\n")
        a(img(charts.get("absa")))
        aspects = list(absa.get("Positive", {}).keys())
        a("| Aspect | Positive | Negative | Neutral | Neg% |")
        a("|--------|--------:|--------:|-------:|-----:|")
        for asp in aspects:
            p = absa["Positive"].get(asp, 0)
            n = absa["Negative"].get(asp, 0)
            neu = absa["Neutral"].get(asp, 0)
            neg_pct = round(n / max(p + n, 1) * 100)
            a(f"| {asp} | {p:,} | {n:,} | {neu:,} | {neg_pct}% |")
        a(f"\n**Total aspect mentions:** {latest.get('absa_aspect_mentions', 0):,}")
        a("\n### Key Insights")
        pos_totals = {k: absa["Positive"].get(k, 0) for k in aspects}
        neg_totals = {k: absa["Negative"].get(k, 0) for k in aspects}
        top_pos = max(pos_totals, key=pos_totals.get)
        top_neg = max(neg_totals, key=neg_totals.get)
        neg_ratios = {k: neg_totals[k] / max(pos_totals[k] + neg_totals[k], 1) for k in aspects}
        worst_ratio = max(neg_ratios, key=neg_ratios.get)
        a(f"- Most praised: **{top_pos}** ({pos_totals[top_pos]:,} positive mentions)")
        a(f"- Most complained: **{top_neg}** ({neg_totals[top_neg]:,} negative mentions)")
        a(f"- Highest negative ratio: **{worst_ratio}** ({neg_ratios[worst_ratio]:.0%})")

    # ── History ───────────────────────────────────────────────────
    if len(runs) > 1:
        a("\n---\n## 7. Training History\n")
        a(img(charts.get("history")))
        a("| Run | Samples | Best Model | F1 Macro | Accuracy |")
        a("|-----|-------:|-----------|--------:|--------:|")
        for r in runs:
            bm = r.get("best_model", {})
            a(f"| `{r['run_id']}` | {r.get('sample_count',0):,} | "
              f"{bm.get('name','-')} | {bm.get('f1_macro',0)} | {bm.get('accuracy',0)} |")

    a("\n---\n*Report generated by `analysis/generate_report.py`*")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def generate(results_path: Path | None = None) -> Path:
    path = results_path or RESULTS_FILE
    runs = _load(path)
    latest = runs[-1]
    out_dir = path.parent

    print(f"Generating report for {len(runs)} run(s)...")

    charts: dict[str, Path | None] = {}
    charts["compare"] = _chart_model_comparison(latest, out_dir)
    charts["heatmap"] = _chart_per_class_f1(latest, out_dir)
    charts["cm"]      = _chart_confusion_matrix(latest, out_dir)
    charts["dist"]    = _chart_class_distribution(latest, out_dir)
    charts["absa"]    = _chart_absa(latest, out_dir)
    charts["history"] = _chart_history(runs, out_dir)

    for name, p in charts.items():
        if p:
            print(f"  Chart: {p.name}")

    md = _build_markdown(runs, charts, out_dir)
    REPORT_FILE.write_text(md, encoding="utf-8")
    print(f"  Report: {REPORT_FILE}")
    return REPORT_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None, help="Path to training_results.json")
    args = parser.parse_args()
    generate(Path(args.results) if args.results else None)
