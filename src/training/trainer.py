"""Full training pipeline: CSV → preprocess → FastText embeddings (frozen) + extra features → train 5 models + ensemble → evaluate.

Strategy:
  - FastText pretrained Vietnamese (cc.vi.300.bin) — FROZEN, sentence embeddings (300-dim)
  - Extra features: text length, word count, exclamation ratio, positive/negative keyword ratio (5-dim)
  - Final features: 300 + 5 = 305-dim
  - Train 5 classifiers: LogReg, SVM, RandomForest, GaussianNB, MLP
  - Ensemble: soft voting across all 5 → 6th "model" in comparison
  - Balancing: class_weight='balanced' (weighted loss)
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path

import fasttext
import fasttext.util
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.training.labeling import LABEL_NAMES, load_labeled_data, _POSITIVE_KEYWORDS, _NEGATIVE_KEYWORDS
from src.training.balancing import balance_with_class_weight, get_distribution
from src.training.experiment import save_experiment
from src.preprocessing.processor import preprocess
from src.analysis.absa import run_absa

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FASTTEXT_MODEL_PATH = MODELS_DIR / "cc.vi.300.bin"


def _evaluate(y_true, y_pred) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=[LABEL_NAMES[l] for l in labels], output_dict=True, zero_division=0,
        ),
    }


def _data_fingerprint(df: pd.DataFrame) -> str:
    content = f"{len(df)}_{df['sentiment'].value_counts().to_dict()}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


# ── FastText frozen embeddings ──────────────────────────────────

def _ensure_fasttext_model() -> fasttext.FastText._FastText:
    """Download pretrained Vietnamese FastText if not exists, then load."""
    if not FASTTEXT_MODEL_PATH.exists():
        logger.info("Downloading pretrained FastText Vietnamese (cc.vi.300.bin)...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import os
        orig_dir = os.getcwd()
        os.chdir(str(MODELS_DIR))
        fasttext.util.download_model("vi", if_exists="ignore")
        os.chdir(orig_dir)

    logger.info(f"Loading frozen FastText model from {FASTTEXT_MODEL_PATH}...")
    return fasttext.load_model(str(FASTTEXT_MODEL_PATH))


def _texts_to_embeddings(ft_model, texts) -> np.ndarray:
    """Batch convert texts to sentence embeddings (n_samples, 300)."""
    return np.array([ft_model.get_sentence_vector(t) for t in texts])


# ── Extra handcrafted features ──────────────────────────────────

def _extract_extra_features(texts) -> np.ndarray:
    """Extract 5 extra features per text:
    - char_len: character length (normalized)
    - word_count: number of words
    - excl_ratio: ratio of '!' chars
    - pos_ratio: ratio of positive keywords found
    - neg_ratio: ratio of negative keywords found
    """
    features = []
    for text in texts:
        t = str(text).lower()
        words = t.split()
        word_count = len(words)
        char_len = len(t)
        excl_ratio = t.count("!") / max(char_len, 1)
        pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in t)
        neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in t)
        total_kw = pos_hits + neg_hits
        pos_ratio = pos_hits / max(total_kw, 1)
        neg_ratio = neg_hits / max(total_kw, 1)
        features.append([char_len, word_count, excl_ratio, pos_ratio, neg_ratio])
    return np.array(features, dtype=np.float32)


def _build_features(ft_model, texts) -> np.ndarray:
    """Combine FastText embeddings (300) + extra features (5) → 305-dim."""
    embeddings = _texts_to_embeddings(ft_model, texts)
    extra = _extract_extra_features(texts)
    return np.hstack([embeddings, extra])


# ── ML models ───────────────────────────────────────────────────

def _get_ml_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, C=1.0,
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, class_weight="balanced", random_state=42, C=1.0),
            cv=3,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "GaussianNB": GaussianNB(),
        "MLP_NeuralNet": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            batch_size=64,
            random_state=42,
            verbose=False,
        ),
        "MLP_NeuralNet_Tuned": MLPClassifier(
            hidden_layer_sizes=(384, 192, 96),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate="adaptive",
            learning_rate_init=0.0007,
            max_iter=350,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            batch_size=128,
            random_state=42,
            verbose=False,
        ),
    }


def _get_text_models():
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=80000,
        sublinear_tf=True,
    )
    return {
        "TFIDF_LogisticRegression": Pipeline([
            ("tfidf", vectorizer),
            ("clf", LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42,
                C=2.0,
                solver="lbfgs",
            )),
        ]),
        "TFIDF_LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=80000,
                sublinear_tf=True,
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(max_iter=3000, class_weight="balanced", random_state=42, C=0.8),
                cv=3,
            )),
        ]),
    }


def _sample_weights(y: np.ndarray, class_weights: dict[int, float]) -> np.ndarray:
    return np.array([float(class_weights.get(int(label), 1.0)) for label in y], dtype=np.float32)


def _fit_classifier(model, X_train, y_train, sample_weight: np.ndarray | None = None) -> None:
    """Fit with sample_weight when the estimator supports it."""
    fit_params = inspect.signature(model.fit).parameters
    if sample_weight is not None and "sample_weight" in fit_params:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)


def _predict_with_neutral_boost(model, X, neutral_boost: float) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = np.array(getattr(model, "classes_", np.arange(proba.shape[1])))
    boosts = np.ones(proba.shape[1], dtype=np.float32)
    neutral_idx = np.where(classes == 1)[0]
    if len(neutral_idx):
        boosts[neutral_idx[0]] = neutral_boost
    return classes[np.argmax(proba * boosts, axis=1)]


def _tune_neutral_boost(model, X_val, y_val) -> tuple[float, dict]:
    """Tune a lightweight validation-time boost for the minority neutral class."""
    best_boost = 1.0
    best_metrics = _evaluate(y_val, model.predict(X_val))
    for boost in np.round(np.arange(0.8, 2.61, 0.1), 2):
        pred = _predict_with_neutral_boost(model, X_val, float(boost))
        metrics = _evaluate(y_val, pred)
        if metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_boost = float(boost)
            best_metrics = metrics
    return best_boost, best_metrics


def _train_lstm_model(
    train_texts,
    y_train: np.ndarray,
    val_texts,
    y_val: np.ndarray,
    test_texts,
    y_test: np.ndarray,
    class_weights: dict[int, float],
    run_id: str,
    ft_model=None,
) -> dict:
    """Train a lightweight LSTM text classifier when TensorFlow/Keras is available."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
    except ImportError as exc:
        return {
            "error": (
                "tensorflow_not_installed: install TensorFlow/Keras in the venv "
                "to train LSTM_NeuralNet"
            ),
            "exception": str(exc),
        }

    try:
        tf.keras.utils.set_random_seed(42)

        max_vocab = 20000
        lengths = np.array([len(str(t).split()) for t in train_texts], dtype=np.int32)
        max_len = int(np.clip(np.percentile(lengths, 95), 40, 160)) if len(lengths) else 80

        tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>", filters="")
        tokenizer.fit_on_texts(train_texts)
        vocab_size = min(max_vocab, len(tokenizer.word_index) + 1)
        embedding_source = "random_trainable"
        embedding_dim = 128
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)

        if ft_model is not None:
            embedding_dim = ft_model.get_dimension()
            embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
            for word, idx in tokenizer.word_index.items():
                if idx >= vocab_size:
                    continue
                embedding_matrix[idx] = ft_model.get_word_vector(word)
            embedding_layer = Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                mask_zero=True,
                trainable=False,
            )
            embedding_source = "fasttext_cc.vi.300_frozen"

        def _seq(texts):
            seqs = tokenizer.texts_to_sequences(texts)
            return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

        X_train_seq = _seq(train_texts)
        X_val_seq = _seq(val_texts)
        X_test_seq = _seq(test_texts)

        model = Sequential([
            embedding_layer,
            Bidirectional(LSTM(96, dropout=0.25)),
            Dense(96, activation="relu"),
            Dropout(0.35),
            Dense(len(LABEL_NAMES), activation="softmax"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ]
        history = model.fit(
            X_train_seq,
            y_train,
            validation_data=(X_val_seq, y_val),
            epochs=12,
            batch_size=96,
            class_weight={int(k): float(v) for k, v in class_weights.items()},
            callbacks=callbacks,
            verbose=0,
        )

        y_val_pred = np.argmax(model.predict(X_val_seq, verbose=0), axis=1)
        y_test_pred = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)

        val_metrics = _evaluate(y_val, y_val_pred)
        test_metrics = _evaluate(y_test, y_test_pred)

        model_path = MODELS_DIR / f"{run_id}_lstm.keras"
        tokenizer_path = MODELS_DIR / f"{run_id}_lstm_tokenizer.pkl"
        model.save(model_path)
        joblib.dump(tokenizer, tokenizer_path)

        return {
            "result": {
                "type": "sequence_neural",
                "backend": "tensorflow_keras",
                "balance": "weighted_loss",
                "val": val_metrics,
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
                "params": {
                    "max_vocab": max_vocab,
                    "vocab_size": int(vocab_size),
                    "max_len": int(max_len),
                    "embedding_dim": int(embedding_dim),
                    "embedding_source": embedding_source,
                    "lstm_units": 96,
                    "batch_size": 96,
                    "epochs_run": len(history.history.get("loss", [])),
                },
                "history": {
                    k: [round(float(x), 4) for x in v]
                    for k, v in history.history.items()
                },
            },
            "artifacts": {
                "model_path": str(model_path),
                "tokenizer_path": str(tokenizer_path),
                "max_len": int(max_len),
                "vocab_size": int(vocab_size),
            },
        }
    except Exception as exc:
        logger.exception("LSTM_NeuralNet failed")
        return {"error": str(exc)}


# ── Training results snapshot (for final report) ────────────────

_ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "analysis"
_TRAINING_RESULTS_FILE = _ANALYSIS_DIR / "training_results.json"
_BEST_MODEL_META_FILE = MODELS_DIR / "best_model_meta.json"


def _save_training_results(record: dict) -> None:
    """Append the experiment record to analysis/training_results.json for reporting."""
    import json

    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if _TRAINING_RESULTS_FILE.exists():
        try:
            existing = json.loads(_TRAINING_RESULTS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            existing = []

    existing.append(record)
    _TRAINING_RESULTS_FILE.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info(f"Training results saved: {_TRAINING_RESULTS_FILE}")


# ── Main pipeline ──────────────────────────────────────────────

def train_pipeline(force: bool = False, csv_path: str | None = None) -> dict:
    """Run full training pipeline:
    CSV → preprocess → FastText embeddings (frozen) + extra features
    → train 5 models + ensemble → compare all 6 → save best.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 0. ABSA analysis (runs before training, outputs saved to analysis/) ──
    logger.info("── Running ABSA analysis before training ──")
    absa_result = run_absa(csv_path=csv_path)
    if absa_result["status"] != "success":
        logger.warning(f"ABSA failed: {absa_result.get('reason')} — continuing without ABSA summary")
        absa_result = {}

    # ── 1. Load & label ─────────────────────────────────────────
    logger.info("Loading labeled data from CSV...")
    df = load_labeled_data(csv_path)

    if df.empty or len(df) < 50:
        logger.warning(f"Not enough data ({len(df)} samples).")
        return {"status": "skipped", "reason": "insufficient_data", "sample_count": len(df)}

    fingerprint = _data_fingerprint(df)
    fingerprint_file = MODELS_DIR / ".last_fingerprint"
    if not force and fingerprint_file.exists():
        if fingerprint_file.read_text().strip() == fingerprint:
            logger.info("Data unchanged. Use --force to retrain.")
            return {"status": "skipped", "reason": "data_unchanged", "fingerprint": fingerprint}

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Training run: {run_id} ({len(df)} samples)")

    # ── 2. Preprocess ───────────────────────────────────────────
    logger.info("Preprocessing text...")
    df["text_clean"] = df["text"].apply(lambda t: preprocess(t, use_tokenizer=True, remove_sw=True))
    df = df[df["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
    logger.info(f"After preprocessing: {len(df)} samples")

    dist_before = get_distribution(df["sentiment"].values, LABEL_NAMES)
    class_weights = balance_with_class_weight(df["sentiment"].values)

    # ── 3. FastText embeddings + extra features → 305-dim ──────
    ft_model = _ensure_fasttext_model()

    y = df["sentiment"].values

    # ── 4. Split (70/15/15) ─────────────────────────────────────
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=y,
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=42, stratify=df_temp["sentiment"].values,
    )
    y_train = df_train["sentiment"].values
    y_val = df_val["sentiment"].values
    y_test = df_test["sentiment"].values
    logger.info(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    logger.info("Generating features (FastText 300-dim + 5 extra features)...")
    X_train = _build_features(ft_model, df_train["text_clean"].values)
    X_val = _build_features(ft_model, df_val["text_clean"].values)
    X_test = _build_features(ft_model, df_test["text_clean"].values)
    logger.info(f"Feature matrix shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ── 5. Normalize features ───────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── 6. Train 5 individual models ────────────────────────────
    logger.info("── Training 5 models on FastText embeddings + extra features ──")

    ml_models = _get_ml_models()
    results = {}
    trained_models = {}
    model_inputs = {}
    lstm_artifacts = None
    train_sample_weight = _sample_weights(y_train, class_weights)

    for name, model in ml_models.items():
        logger.info(f"Training {name}...")
        try:
            _fit_classifier(model, X_train, y_train, sample_weight=train_sample_weight)
            trained_models[name] = model
            model_inputs[name] = "features"

            val_metrics = _evaluate(y_val, model.predict(X_val))
            test_metrics = _evaluate(y_test, model.predict(X_test))

            results[name] = {
                "type": "individual",
                "balance": "weighted_loss",
                "val": val_metrics,
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
            }
            logger.info(f"  {name}: acc={test_metrics['accuracy']}, f1={test_metrics['f1_macro']}")

            if hasattr(model, "predict_proba"):
                boost, boosted_val_metrics = _tune_neutral_boost(model, X_val, y_val)
                if boost != 1.0 and boosted_val_metrics["f1_macro"] >= val_metrics["f1_macro"] + 0.002:
                    boosted_test_pred = _predict_with_neutral_boost(model, X_test, boost)
                    boosted_test_metrics = _evaluate(y_test, boosted_test_pred)
                    boosted_name = f"{name}_NeutralBoost"
                    results[boosted_name] = {
                        "type": "calibrated",
                        "base_model": name,
                        "balance": "weighted_loss+neutral_boost",
                        "neutral_boost": boost,
                        "val": boosted_val_metrics,
                        "test": boosted_test_metrics,
                        "f1_macro": boosted_test_metrics["f1_macro"],
                        "accuracy": boosted_test_metrics["accuracy"],
                    }
                    trained_models[boosted_name] = {
                        "_neutral_calibrated": model,
                        "neutral_boost": boost,
                    }
                    model_inputs[boosted_name] = "features"
                    logger.info(
                        f"  {boosted_name}: boost={boost}, "
                        f"acc={boosted_test_metrics['accuracy']}, f1={boosted_test_metrics['f1_macro']}"
                    )
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {"type": "individual", "error": str(e)}

    logger.info("Training TF-IDF text models...")
    text_models = _get_text_models()
    train_texts = df_train["text_clean"].values
    val_texts = df_val["text_clean"].values
    test_texts = df_test["text_clean"].values
    for name, model in text_models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(train_texts, y_train)
            trained_models[name] = model
            model_inputs[name] = "text_clean"

            val_metrics = _evaluate(y_val, model.predict(val_texts))
            test_metrics = _evaluate(y_test, model.predict(test_texts))
            results[name] = {
                "type": "text_ngram",
                "features": "tfidf_word_1_2gram",
                "balance": "class_weight",
                "val": val_metrics,
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
            }
            logger.info(f"  {name}: acc={test_metrics['accuracy']}, f1={test_metrics['f1_macro']}")

            if hasattr(model, "predict_proba"):
                boost, boosted_val_metrics = _tune_neutral_boost(model, val_texts, y_val)
                if boost != 1.0 and boosted_val_metrics["f1_macro"] >= val_metrics["f1_macro"] + 0.002:
                    boosted_test_pred = _predict_with_neutral_boost(model, test_texts, boost)
                    boosted_test_metrics = _evaluate(y_test, boosted_test_pred)
                    boosted_name = f"{name}_NeutralBoost"
                    results[boosted_name] = {
                        "type": "calibrated_text_ngram",
                        "base_model": name,
                        "features": "tfidf_word_1_2gram",
                        "balance": "class_weight+neutral_boost",
                        "neutral_boost": boost,
                        "val": boosted_val_metrics,
                        "test": boosted_test_metrics,
                        "f1_macro": boosted_test_metrics["f1_macro"],
                        "accuracy": boosted_test_metrics["accuracy"],
                    }
                    trained_models[boosted_name] = {
                        "_neutral_calibrated": model,
                        "neutral_boost": boost,
                    }
                    model_inputs[boosted_name] = "text_clean"
                    logger.info(
                        f"  {boosted_name}: boost={boost}, "
                        f"acc={boosted_test_metrics['accuracy']}, f1={boosted_test_metrics['f1_macro']}"
                    )
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {"type": "text_ngram", "error": str(e)}

    logger.info("Training LSTM_NeuralNet (sequence model)...")
    lstm_payload = _train_lstm_model(
        df_train["text_clean"].values,
        y_train,
        df_val["text_clean"].values,
        y_val,
        df_test["text_clean"].values,
        y_test,
        class_weights,
        run_id,
        ft_model=ft_model,
    )
    if "error" in lstm_payload:
        results["LSTM_NeuralNet"] = {
            "type": "sequence_neural",
            "backend": "tensorflow_keras",
            "error": lstm_payload["error"],
        }
        if "exception" in lstm_payload:
            results["LSTM_NeuralNet"]["exception"] = lstm_payload["exception"]
        logger.warning(f"  LSTM_NeuralNet skipped/failed: {lstm_payload['error']}")
    else:
        results["LSTM_NeuralNet"] = lstm_payload["result"]
        lstm_artifacts = lstm_payload["artifacts"]
        logger.info(
            "  LSTM_NeuralNet: "
            f"acc={results['LSTM_NeuralNet']['accuracy']}, "
            f"f1={results['LSTM_NeuralNet']['f1_macro']}"
        )

    # ── 7. Ensemble (soft voting from all successful models) ────
    logger.info("── Building Ensemble (soft voting) ──")
    ensemble_models = {
        n: m for n, m in trained_models.items()
        if hasattr(m, "predict_proba") and model_inputs.get(n) == "features"
    }

    if len(ensemble_models) >= 2:
        try:
            classes = np.unique(y_train)

            def ensemble_predict_proba(X):
                aligned = []
                for model in ensemble_models.values():
                    proba = model.predict_proba(X)
                    model_classes = np.array(getattr(model, "classes_", classes))
                    aligned_proba = np.zeros((len(X), len(classes)), dtype=np.float32)
                    for i, cls in enumerate(classes):
                        src = np.where(model_classes == cls)[0]
                        if len(src):
                            aligned_proba[:, i] = proba[:, src[0]]
                    aligned.append(aligned_proba)
                return np.mean(aligned, axis=0)

            def ensemble_predict(X, neutral_boost: float = 1.0):
                avg_proba = ensemble_predict_proba(X)
                boosts = np.ones(avg_proba.shape[1], dtype=np.float32)
                neutral_idx = np.where(classes == 1)[0]
                if len(neutral_idx):
                    boosts[neutral_idx[0]] = neutral_boost
                return classes[np.argmax(avg_proba * boosts, axis=1)]

            y_val_ens = ensemble_predict(X_val)
            y_test_ens = ensemble_predict(X_test)

            val_metrics = _evaluate(y_val, y_val_ens)
            test_metrics = _evaluate(y_test, y_test_ens)

            results["Ensemble_SoftVote"] = {
                "type": "ensemble",
                "balance": "weighted_loss",
                "n_models": len(ensemble_models),
                "members": list(ensemble_models.keys()),
                "val": val_metrics,
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
            }
            trained_models["Ensemble_SoftVote"] = {
                "_ensemble_models": ensemble_models,
                "classes": classes,
            }
            model_inputs["Ensemble_SoftVote"] = "features"
            logger.info(f"  Ensemble: acc={test_metrics['accuracy']}, f1={test_metrics['f1_macro']}")

            best_boost = 1.0
            best_val_metrics = val_metrics
            for boost in np.round(np.arange(0.8, 2.61, 0.1), 2):
                boosted_val = _evaluate(y_val, ensemble_predict(X_val, neutral_boost=float(boost)))
                if boosted_val["f1_macro"] > best_val_metrics["f1_macro"]:
                    best_boost = float(boost)
                    best_val_metrics = boosted_val
            if best_boost != 1.0 and best_val_metrics["f1_macro"] >= val_metrics["f1_macro"] + 0.002:
                boosted_test = _evaluate(y_test, ensemble_predict(X_test, neutral_boost=best_boost))
                results["Ensemble_SoftVote_NeutralBoost"] = {
                    "type": "calibrated_ensemble",
                    "base_model": "Ensemble_SoftVote",
                    "balance": "weighted_loss+neutral_boost",
                    "neutral_boost": best_boost,
                    "n_models": len(ensemble_models),
                    "members": list(ensemble_models.keys()),
                    "val": best_val_metrics,
                    "test": boosted_test,
                    "f1_macro": boosted_test["f1_macro"],
                    "accuracy": boosted_test["accuracy"],
                }
                trained_models["Ensemble_SoftVote_NeutralBoost"] = {
                    "_ensemble_models": ensemble_models,
                    "classes": classes,
                    "neutral_boost": best_boost,
                }
                model_inputs["Ensemble_SoftVote_NeutralBoost"] = "features"
                logger.info(
                    "  Ensemble_SoftVote_NeutralBoost: "
                    f"boost={best_boost}, acc={boosted_test['accuracy']}, f1={boosted_test['f1_macro']}"
                )
        except Exception as e:
            logger.error(f"  Ensemble failed: {e}")
            results["Ensemble_SoftVote"] = {"type": "ensemble", "error": str(e)}

    # ── 8. Save best model ──────────────────────────────────────
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return {"status": "failed", "reason": "all_models_failed"}

    best_name = max(valid, key=lambda k: valid[k]["f1_macro"])
    logger.info(f"Best model: {best_name} F1={valid[best_name]['f1_macro']}")

    if best_name == "LSTM_NeuralNet":
        best_meta = {
            "name": best_name,
            "backend": "tensorflow_keras_lstm",
            "input_type": "text_clean",
            **(lstm_artifacts or {}),
        }
    else:
        joblib.dump(trained_models[best_name], MODELS_DIR / "best_model.pkl")
        input_type = model_inputs.get(best_name, "features")
        best_meta = {
            "name": best_name,
            "backend": "sklearn_text" if input_type == "text_clean" else "sklearn_fasttext",
            "input_type": input_type,
            "model_path": str(MODELS_DIR / "best_model.pkl"),
            "scaler_path": str(MODELS_DIR / "scaler.pkl"),
        }
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    _BEST_MODEL_META_FILE.write_text(
        json.dumps(best_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fingerprint_file.write_text(fingerprint)

    experiment_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "sample_count": len(df),
        "embedding": "fasttext_cc.vi.300_frozen",
        "feature_dim": X_train.shape[1],
        "extra_features": ["char_len", "word_count", "excl_ratio", "pos_ratio", "neg_ratio"],
        "split": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "balance_method": "weighted_loss (class_weight='balanced')",
        "class_weights": {str(k): round(v, 3) for k, v in class_weights.items()},
        "distribution_before": dist_before,
        "models": results,
        "best_model": {
            "name": best_name,
            "type": valid[best_name]["type"],
            "f1_macro": valid[best_name]["f1_macro"],
            "accuracy": valid[best_name]["accuracy"],
        },
        "best_model_artifact": best_meta,
        "absa_summary": absa_result.get("summary", {}),
        "absa_aspect_mentions": absa_result.get("aspect_mentions", 0),
        "absa_drilldowns": absa_result.get("drilldown_csvs", {}),
        "absa_insights": absa_result.get("insights", {}),
    }
    save_experiment(experiment_record)
    _save_training_results(experiment_record)

    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "generate_report",
            Path(__file__).resolve().parents[2] / "analysis" / "generate_report.py",
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        report_path = _mod.generate()
        logger.info(f"Report generated: {report_path}")
    except Exception as e:
        logger.warning(f"Report generation failed (non-critical): {e}")

    return {"status": "success", **experiment_record}


def predict(texts: list[str]) -> list[dict]:
    """Load frozen FastText + scaler + best model → predict."""
    model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta = None
    if _BEST_MODEL_META_FILE.exists():
        try:
            meta = json.loads(_BEST_MODEL_META_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = None

    if meta and meta.get("backend") == "tensorflow_keras_lstm":
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError as exc:
            raise ImportError("Best model is LSTM, but TensorFlow/Keras is not installed.") from exc

        keras_model = tf.keras.models.load_model(meta["model_path"])
        tokenizer = joblib.load(meta["tokenizer_path"])
        max_len = int(meta["max_len"])

        results = []
        for text in texts:
            clean = preprocess(text, use_tokenizer=True, remove_sw=True)
            seq = tokenizer.texts_to_sequences([clean])
            padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
            proba = keras_model.predict(padded, verbose=0)[0]
            pred = int(np.argmax(proba))
            results.append({
                "text": text[:100],
                "sentiment": pred,
                "sentiment_name": LABEL_NAMES[pred],
                "confidence": round(float(proba[pred]), 4),
            })
        return results

    if not model_path.exists():
        raise FileNotFoundError("No trained model. Run `python run.py train` first.")

    ml_model = joblib.load(model_path)
    input_type = (meta or {}).get("input_type", "features")
    scaler = joblib.load(scaler_path) if scaler_path.exists() and input_type == "features" else None
    ft_model = _ensure_fasttext_model() if input_type == "features" else None

    results = []
    for text in texts:
        clean = preprocess(text, use_tokenizer=True, remove_sw=True)
        if input_type == "text_clean":
            model_input = [clean]
        else:
            model_input = _build_features(ft_model, [clean])
            if scaler:
                model_input = scaler.transform(model_input)
        if isinstance(ml_model, dict) and "_neutral_calibrated" in ml_model:
            pred = _predict_with_neutral_boost(
                ml_model["_neutral_calibrated"],
                model_input,
                float(ml_model.get("neutral_boost", 1.0)),
            )[0]
        elif isinstance(ml_model, dict) and "_ensemble_models" in ml_model:
            classes = np.array(ml_model.get("classes", [0, 1, 2]))
            proba_list = []
            for model in ml_model["_ensemble_models"].values():
                proba = model.predict_proba(model_input)
                model_classes = np.array(getattr(model, "classes_", classes))
                n_rows = len(model_input) if input_type == "text_clean" else model_input.shape[0]
                aligned = np.zeros((n_rows, len(classes)), dtype=np.float32)
                for i, cls in enumerate(classes):
                    src = np.where(model_classes == cls)[0]
                    if len(src):
                        aligned[:, i] = proba[:, src[0]]
                proba_list.append(aligned)
            avg_proba = np.mean(proba_list, axis=0)
            neutral_boost = float(ml_model.get("neutral_boost", 1.0))
            boosts = np.ones(avg_proba.shape[1], dtype=np.float32)
            neutral_idx = np.where(classes == 1)[0]
            if len(neutral_idx):
                boosts[neutral_idx[0]] = neutral_boost
            pred = classes[np.argmax(avg_proba * boosts, axis=1)][0]
        else:
            pred = ml_model.predict(model_input)[0]
        results.append({
            "text": text[:100],
            "sentiment": int(pred),
            "sentiment_name": LABEL_NAMES[pred],
        })
    return results
