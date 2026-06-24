#!/usr/bin/env python3
"""Deep audit of rule-based weak labeling."""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.labeling import (  # noqa: E402
    BINARY_FRAMING_NEGATIVE_NONNEGATIVE,
    BINARY_FRAMING_POSITIVE_NONPOSITIVE,
    DEFAULT_BINARY_FRAMING,
    _ABSA_NEGATIVE,
    _ABSA_POSITIVE,
    _EN_NEGATIVE_KEYWORDS,
    _EN_POSITIVE_KEYWORDS,
    _NEGATIVE_KEYWORDS,
    _POSITIVE_KEYWORDS,
    _absa_score,
    _count_positive_hits,
    _english_field_score,
    _keyword_score,
    apply_binary_framing,
    english_has_negative_signal,
    english_weak_label_combine,
    has_negative_signal,
    load_glassdoor_english_data,
    load_labeled_data,
    map_sentiment_to_binary,
    rating_to_sentiment,
    weak_label_combine,
    LABEL_MAP,
    LABEL_NAMES,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def combined_vi(row, suppress_sarcasm: bool | None = None) -> float:
    rl = row["rating_label"] if "rating_label" in row else rating_to_sentiment(row["rating"])
    if suppress_sarcasm is None:
        suppress_sarcasm = rl <= LABEL_MAP["neutral"]
    return _keyword_score(
        str(row.get("title") or ""),
        str(row.get("pros") or ""),
        str(row.get("cons") or ""),
        str(row.get("advice") or ""),
        suppress_sarcasm=suppress_sarcasm,
    ) + _absa_score(
        str(row.get("title") or ""),
        str(row.get("pros") or ""),
        str(row.get("cons") or ""),
        str(row.get("advice") or ""),
        suppress_sarcasm=suppress_sarcasm,
    )


def combined_en(row, suppress_sarcasm: bool | None = None) -> float:
    rl = row["rating_label"]
    if suppress_sarcasm is None:
        suppress_sarcasm = rl <= LABEL_MAP["neutral"]
    h = str(row.get("headline") or row.get("title") or "")
    p = str(row.get("pros") or "")
    c = str(row.get("cons") or "")
    return (
        _english_field_score(h, 1.0, suppress_sarcasm=suppress_sarcasm)
        + _english_field_score(p, 1.0, suppress_sarcasm=suppress_sarcasm)
        + _english_field_score(c, 1.5, suppress_sarcasm=suppress_sarcasm)
    )


def spot_check_overrides(df: pd.DataFrame, n: int = 100, lang: str = "vi") -> dict:
    """Heuristic spot-check: flag likely-wrong overrides."""
    changed = df[df["rating_label"] != df["sentiment"]].copy()
    if len(changed) == 0:
        return {"sampled": 0, "likely_wrong": 0, "rate": 0.0}

    sample = changed.sample(n=min(n, len(changed)), random_state=SEED)
    likely_wrong = 0
    for _, row in sample.iterrows():
        rl, sl = row["rating_label"], row["sentiment"]
        cons = str(row.get("cons") or "").lower()
        pros = str(row.get("pros") or "").lower()
        text = str(row.get("text") or "").lower()

        neg_kw = sum(1 for kw in (_NEGATIVE_KEYWORDS if lang == "vi" else _EN_NEGATIVE_KEYWORDS) if kw in cons)
        pos_kw = sum(1 for kw in (_POSITIVE_KEYWORDS if lang == "vi" else _EN_POSITIVE_KEYWORDS) if kw in (pros + cons))

        # Heuristic: flip direction contradicts dominant field signal
        if rl == 2 and sl == 0:
            likely_wrong += 0  # conservative flip down — usually OK
        elif rl == 0 and sl == 2:
            if neg_kw >= 2 and pos_kw <= 1:
                likely_wrong += 1
        elif rl == 1 and sl == 2:
            if neg_kw >= 2 and pos_kw <= neg_kw:
                likely_wrong += 1
        elif rl == 1 and sl == 0:
            if pos_kw >= 2 and neg_kw <= 1:
                likely_wrong += 1
        elif rl == 2 and sl == 1:
            if neg_kw >= 3 and pos_kw == 0:
                likely_wrong += 1
        elif rl == 0 and sl == 1:
            if pos_kw >= 3 and neg_kw == 0:
                likely_wrong += 1

    return {
        "sampled": len(sample),
        "likely_wrong": likely_wrong,
        "rate": likely_wrong / len(sample) if sample.shape[0] else 0.0,
    }


def audit_vi() -> None:
    print("=" * 70)
    print("VIETNAMESE AUDIT")
    print("=" * 70)

    # Substring bug
    print("\n--- Substring: 'tốt' in 'không tốt' ---")
    for t in ["công ty không tốt", "rất tốt", "không tốt nhưng cũng không tệ"]:
        pos_kw = _count_positive_hits(t, _POSITIVE_KEYWORDS, suppress_sarcasm=False)
        pos_absa = _count_positive_hits(t, _ABSA_POSITIVE, suppress_sarcasm=False)
        neg_kw = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in t.lower())
        neg_absa = sum(1 for w in _ABSA_NEGATIVE if w in t.lower())
        net = (pos_kw - neg_kw) + (pos_absa - neg_absa)
        print(f"  '{t}' => pos_kw={pos_kw} neg_kw={neg_kw} pos_absa={pos_absa} neg_absa={neg_absa} net={net}")

    vi = load_labeled_data()
    vi["rating_label"] = vi["rating"].apply(rating_to_sentiment)
    print(f"\nTotal: {len(vi)}")

    print("\n--- Distribution shift ---")
    for lbl in [0, 1, 2]:
        r = (vi["rating_label"] == lbl).sum()
        w = (vi["sentiment"] == lbl).sum()
        print(f"  {LABEL_NAMES[lbl]}: rating={r} weak={w} delta={w - r:+d}")

    print("\n--- label_source ---")
    for src, cnt in vi["label_source"].value_counts().items():
        print(f"  {src}: {cnt}")

    pairs = [
        ("4-5★→negative", (vi["rating_label"] == 2) & (vi["sentiment"] == 0)),
        ("4-5★→neutral", (vi["rating_label"] == 2) & (vi["sentiment"] == 1)),
        ("1-2★→positive", (vi["rating_label"] == 0) & (vi["sentiment"] == 2)),
        ("1-2★→neutral", (vi["rating_label"] == 0) & (vi["sentiment"] == 1)),
        ("3★→positive", (vi["rating_label"] == 1) & (vi["sentiment"] == 2)),
        ("3★→negative", (vi["rating_label"] == 1) & (vi["sentiment"] == 0)),
    ]
    print("\n--- Override directions ---")
    for name, mask in pairs:
        print(f"  {name}: {mask.sum()}")

    # 4-5★ still positive with negative combined
    high_pos = vi[(vi["rating"] >= 4) & (vi["sentiment"] == 2)]
    neg_comb = []
    for _, row in high_pos.iterrows():
        c = combined_vi(row, suppress_sarcasm=False)
        if c < 0:
            neg_comb.append((row, c))
    neg_comb.sort(key=lambda x: x[1])
    print(f"\n--- 4-5★ still POSITIVE with combined<0: {len(neg_comb)} ---")
    for row, c in neg_comb[:10]:
        print(f"  ★{row['rating']} comb={c:.1f} src={row['label_source']}")
        print(f"    cons: {str(row['cons'])[:120]}")

    # 1-2★ flipped positive
    neg_pos = vi[(vi["rating_label"] == 0) & (vi["sentiment"] == 2)]
    print(f"\n--- 1-2★→positive ({len(neg_pos)}) ---")
    for _, row in neg_pos.head(10).iterrows():
        c = combined_vi(row)
        print(f"  ★{row['rating']} comb={c:.1f} src={row['label_source']}")
        print(f"    text: {str(row['text'])[:130]}")

    # 3★→positive
    neu_pos = vi[(vi["rating_label"] == 1) & (vi["sentiment"] == 2)]
    print(f"\n--- 3★→positive ({len(neu_pos)}) ---")
    for _, row in neu_pos.head(10).iterrows():
        c = combined_vi(row)
        print(f"  comb={c:.1f} src={row['label_source']}")
        print(f"    cons: {str(row['cons'])[:90]} | pros: {str(row['pros'])[:60]}")

    # Double counting
    sample = vi.head(8000)
    kw_s, ab_s = [], []
    for _, row in sample.iterrows():
        suppress = row["rating_label"] <= 1
        kw_s.append(
            _keyword_score(row["title"], row["pros"], row["cons"], row["advice"], suppress_sarcasm=suppress)
        )
        ab_s.append(_absa_score(row["title"], row["pros"], row["cons"], row["advice"], suppress_sarcasm=suppress))
    print(f"\n--- kw/absa correlation (8k): {np.corrcoef(kw_s, ab_s)[0,1]:.3f} ---")

    # Sarcasm at 4★
    four_star = vi[vi["rating"] == 4]
    sarcasm_phrases = ["không phàn nàn", "không có gì phàn nàn", "không có gì để chê"]
    sarc_4 = four_star[
        four_star["text"].str.lower().apply(lambda t: any(p in t for p in sarcasm_phrases))
    ]
    print(f"\n--- 4★ with sarcasm phrase (no suppression): {len(sarc_4)} ---")
    for _, row in sarc_4.head(5).iterrows():
        c_sup = combined_vi(row, suppress_sarcasm=True)
        c_no = combined_vi(row, suppress_sarcasm=False)
        print(f"  comb suppress={c_sup:.1f} no_suppress={c_no:.1f} sent={LABEL_NAMES[row['sentiment']]}")

    sc = spot_check_overrides(vi, 100, "vi")
    print(f"\n--- Spot-check 100 overrides (heuristic): ~{sc['rate']*100:.0f}% likely wrong ({sc['likely_wrong']}/{sc['sampled']}) ---")

    # Binary framing
    print("\n--- Binary framing (negative vs non-negative) ---")
    for framing in [DEFAULT_BINARY_FRAMING, BINARY_FRAMING_POSITIVE_NONPOSITIVE]:
        b = apply_binary_framing(vi, framing)
        neg = int((b["sentiment"] == 0).sum())
        non_neg = int((b["sentiment"] == 1).sum())
        total = len(b)
        print(f"  {framing}: negative={neg} ({neg/total:.1%}) non_negative={non_neg} ({non_neg/total:.1%})")
    print("\n--- binary_label_source (default framing) ---")
    b_default = apply_binary_framing(vi)
    for src, cnt in b_default["binary_label_source"].value_counts().head(12).items():
        print(f"  {src}: {cnt}")


def audit_en() -> None:
    print("\n" + "=" * 70)
    print("ENGLISH GLASSDOOR AUDIT")
    print("=" * 70)
    en = load_glassdoor_english_data(max_rows=None)
    en["rating_label"] = en["rating"].apply(rating_to_sentiment)
    print(f"Total: {len(en)}")

    print("\n--- Distribution shift ---")
    for lbl in [0, 1, 2]:
        r = (en["rating_label"] == lbl).sum()
        w = (en["sentiment"] == lbl).sum()
        print(f"  {LABEL_NAMES[lbl]}: rating={r} weak={w} delta={w - r:+d}")

    print("\n--- label_source ---")
    for src, cnt in en["label_source"].value_counts().items():
        print(f"  {src}: {cnt}")

    pairs = [
        ("4-5★→negative", (en["rating_label"] == 2) & (en["sentiment"] == 0)),
        ("4-5★→neutral", (en["rating_label"] == 2) & (en["sentiment"] == 1)),
        ("1-2★→positive", (en["rating_label"] == 0) & (en["sentiment"] == 2)),
        ("1-2★→neutral", (en["rating_label"] == 0) & (en["sentiment"] == 1)),
        ("3★→positive", (en["rating_label"] == 1) & (en["sentiment"] == 2)),
        ("3★→negative", (en["rating_label"] == 1) & (en["sentiment"] == 0)),
    ]
    for name, mask in pairs:
        print(f"  {name}: {mask.sum()}")

    pos_neg = en[(en["rating_label"] == 2) & (en["sentiment"] == 0)]
    print(f"\n--- EN 4-5★→negative samples ---")
    for _, row in pos_neg.head(5).iterrows():
        print(f"  ★{row['rating']} comb={combined_en(row):.1f} cons={str(row.get('cons',''))[:100]}")

    neg_pos = en[(en["rating_label"] == 0) & (en["sentiment"] == 2)]
    print(f"\n--- EN 1-2★→positive ({len(neg_pos)}) ---")
    for _, row in neg_pos.head(5).iterrows():
        print(f"  ★{row['rating']} src={row['label_source']} cons={str(row.get('cons',''))[:100]}")

    sc = spot_check_overrides(en, 100, "en")
    print(f"\n--- Spot-check 100 EN overrides: ~{sc['rate']*100:.0f}% likely wrong ({sc['likely_wrong']}/{sc['sampled']}) ---")

    print("\n--- Binary framing (negative vs non-negative) ---")
    b = apply_binary_framing(en)
    neg = int((b["sentiment"] == 0).sum())
    non_neg = int((b["sentiment"] == 1).sum())
    total = len(b)
    print(f"  negative={neg} ({neg/total:.1%}) non_negative={non_neg} ({non_neg/total:.1%})")


def parity_check() -> None:
    print("\n" + "=" * 70)
    print("VI vs EN PARITY (threshold on combined score)")
    print("=" * 70)
    # Same thresholds but VI combined is ~2x EN because kw+absa
    cases = [
        (2, -4, "4-5★ strong neg text"),
        (2, -2, "4-5★ mild neg text"),
        (0, 8, "1-2★ strong pos text"),
        (0, 3, "1-2★ mild pos text"),
        (1, 3, "3★ pos text"),
        (1, -2, "3★ neg text"),
    ]
    for rl, comb, desc in cases:
        vi_label, vi_src = None, None
        if rl == 1:
            if comb <= -1.5:
                vi_label = 0
            elif comb >= 2.5:
                vi_label = 2
            else:
                vi_label = 1
        elif rl == 2:
            if comb <= -3:
                vi_label = 0
            elif comb <= -1:
                vi_label = 1
            else:
                vi_label = 2
        elif rl == 0:
            if comb >= 7:
                vi_label = 2
            elif comb >= 2:
                vi_label = 1
            else:
                vi_label = 0
        # EN would need comb/2 roughly for same keyword hits
        en_equiv = comb / 2
        print(f"  {desc}: VI comb={comb} -> {LABEL_NAMES[vi_label]} | EN needs comb~{en_equiv:.1f} for same kw signal")


if __name__ == "__main__":
    audit_vi()
    audit_en()
    parity_check()
