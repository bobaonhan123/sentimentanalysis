"""Train MLP model on the binary (no-neutral) variant and append results to training_results.json.

This script is intentionally separate from the full variant_trainer so it can run quickly
without triggering cleanlab or reprocessing all variants.

Run:
    py analysis/train_binary_models.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.processor import preprocess
from src.training.labeling import LABEL_NAMES, load_labeled_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ANALYSIS_DIR = ROOT / "analysis"
TRAINING_RESULTS = ANALYSIS_DIR / "training_results.json"
DATA_CSV = ROOT / "data_post_processing" / "1900_export_reviews.csv"

BINARY_LABEL_NAMES = {0: "negative", 2: "positive"}


def _evaluate(y_true, y_pred, label_names: dict) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [label_names.get(int(l), str(l)) for l in labels]
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
        ),
    }


def _tfidf_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95,
            max_features=70000, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.95,
            max_features=70000, sublinear_tf=True,
        )),
    ])


def main():
    logger.info("Loading data...")
    df = load_labeled_data(str(DATA_CSV))
    if df.empty:
        logger.error("No data loaded!")
        sys.exit(1)

    df = df.copy()
    logger.info("Preprocessing %d rows...", len(df))
    df["text_clean"] = df["text"].apply(lambda t: preprocess(t, use_tokenizer=True, remove_sw=True))
    df = df[df["text_clean"].str.strip().astype(bool)].reset_index(drop=True)

    # Filter binary: remove neutral
    binary_df = df[df["sentiment"].isin([0, 2])].copy()
    logger.info("Binary dataset: %d samples (neg=%d, pos=%d)",
                len(binary_df),
                (binary_df["sentiment"] == 0).sum(),
                (binary_df["sentiment"] == 2).sum())

    texts = binary_df["text_clean"].to_numpy()
    y = binary_df["sentiment"].to_numpy()

    # Remap labels: 0 -> 0 (neg), 2 -> 1 (pos) for binary classifiers
    y_bin = np.where(y == 2, 1, 0)

    X_train_t, X_test_t, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    _, _, y_train_bin, y_test_bin = train_test_split(
        texts, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    model_results = {}

    # ── TF-IDF + MLP ──────────────────────────────────────────
    logger.info("Training TF-IDF + MLP...")
    vec = _tfidf_features()
    X_train_tfidf = vec.fit_transform(X_train_t)
    X_test_tfidf = vec.transform(X_test_t)

    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_train_tfidf)
    X_te_s = scaler.transform(X_test_tfidf)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        batch_size=64,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
    )
    mlp.fit(X_tr_s, y_train)
    y_pred_mlp = mlp.predict(X_te_s)
    mlp_metrics = _evaluate(y_test, y_pred_mlp, BINARY_LABEL_NAMES)
    logger.info("  MLP: acc=%.4f f1=%.4f", mlp_metrics["accuracy"], mlp_metrics["f1_macro"])

    model_results["binary_no_neutral__TF-IDF_WordChar_MLP"] = {
        "type": "variant_text_ngram_mlp",
        "variant": "binary_no_neutral",
        "features": "tfidf_word_char_ngrams",
        "balance": "class_weight",
        "label_names": {str(k): v for k, v in BINARY_LABEL_NAMES.items()},
        "test": mlp_metrics,
        "f1_macro": mlp_metrics["f1_macro"],
        "accuracy": mlp_metrics["accuracy"],
        "recall_macro": mlp_metrics["recall_macro"],
    }

    # ── Try LSTM if Keras available ────────────────────────────
    try:
        logger.info("Attempting LSTM training...")
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.callbacks import EarlyStopping

        MAX_VOCAB = 15000
        MAX_LEN = 100

        tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train_t)

        X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_t), maxlen=MAX_LEN, padding="post")
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_t), maxlen=MAX_LEN, padding="post")

        vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB)

        model = Sequential([
            Embedding(vocab_size, 64, input_length=MAX_LEN),
            Bidirectional(LSTM(64, dropout=0.2)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
        
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(
            X_train_seq, y_train_bin,
            epochs=15, batch_size=64,
            validation_split=0.15,
            callbacks=[es],
            verbose=1,
        )

        y_pred_probs = model.predict(X_test_seq, verbose=0).flatten()
        y_pred_bin = (y_pred_probs > 0.5).astype(int)
        # Remap back to original label space: 0->0 (neg), 1->2 (pos)
        y_pred_mapped = np.where(y_pred_bin == 1, 2, 0)

        lstm_metrics = _evaluate(y_test, y_pred_mapped, BINARY_LABEL_NAMES)
        logger.info("  LSTM: acc=%.4f f1=%.4f", lstm_metrics["accuracy"], lstm_metrics["f1_macro"])

        model_results["binary_no_neutral__TF-IDF_WordChar_LSTM"] = {
            "type": "variant_text_ngram_lstm",
            "variant": "binary_no_neutral",
            "features": "bilstm_embeddings",
            "balance": "class_weight",
            "label_names": {str(k): v for k, v in BINARY_LABEL_NAMES.items()},
            "test": lstm_metrics,
            "f1_macro": lstm_metrics["f1_macro"],
            "accuracy": lstm_metrics["accuracy"],
            "recall_macro": lstm_metrics["recall_macro"],
        }
    except ImportError:
        logger.warning("TensorFlow/Keras not available; LSTM skipped.")
    except Exception as exc:
        logger.exception("LSTM training failed: %s", exc)

    # ── Append to training_results.json ───────────────────────
    existing: list[dict] = []
    if TRAINING_RESULTS.exists():
        existing = json.loads(TRAINING_RESULTS.read_text(encoding="utf-8"))

    # Find the latest variant_run and inject results into it
    variant_runs = [i for i, r in enumerate(existing) if r.get("run_id", "").startswith("variant_run_")]
    if variant_runs:
        latest_idx = variant_runs[-1]
        existing[latest_idx].setdefault("models", {}).update(model_results)
        logger.info("Injected %d model results into latest variant_run (index %d)", len(model_results), latest_idx)
    else:
        # Create a new record
        run_id = f"variant_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        existing.append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(binary_df),
            "embedding": "tfidf_word_char_ngrams",
            "models": model_results,
        })

    TRAINING_RESULTS.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Saved to %s", TRAINING_RESULTS)
    logger.info("Done!")


if __name__ == "__main__":
    main()
