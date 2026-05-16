"""Train sentiment label variants and keep the strongest deployable model.

Variants:
- 3-class original labels.
- 3-class with likely label issues removed by confident learning.
- 2-class polarity labels with neutral removed.
- 4-class labels where weak-label conflict cases become mixed/conflict.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.preprocessing.processor import preprocess
from src.training.experiment import save_experiment
from src.training.labeling import LABEL_NAMES, load_labeled_data

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT_DIR / "analysis"
MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_META_FILE = MODELS_DIR / "best_model_meta.json"
TRAINING_RESULTS_FILE = ANALYSIS_DIR / "training_results.json"

MIXED_LABEL = 3
MIXED_LABEL_NAMES = {**LABEL_NAMES, MIXED_LABEL: "mixed_conflict"}
BINARY_LABEL_NAMES = {0: "negative", 2: "positive"}

# Try to import keras for LSTM models
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logger.warning("TensorFlow/Keras not available; LSTM training will be skipped.")
def _evaluate(y_true, y_pred, label_names: dict[int, str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [label_names.get(int(label), str(label)) for label in labels]
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def _distribution(y: np.ndarray, label_names: dict[int, str]) -> dict:
    counts = pd.Series(y).value_counts().sort_index()
    total = int(counts.sum())
    return {
        label_names.get(int(label), str(label)): {
            "count": int(count),
            "percentage": round(float(count) / max(total, 1) * 100, 1),
        }
        for label, count in counts.items()
    }


def _text_model(kind: str = "logreg") -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=70000,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            max_features=70000,
            sublinear_tf=True,
        )),
    ])
    if kind == "svc":
        clf = CalibratedClassifierCV(
            LinearSVC(max_iter=3000, class_weight="balanced", random_state=42, C=0.8),
            cv=3,
        )
    elif kind == "mlp":
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                max_iter=300,
                batch_size=32,
                learning_rate_init=0.001,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=15,
            ))
        ])
    else:  # logreg
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            C=2.0,
            solver="lbfgs",
        )
    return Pipeline([("features", features), ("clf", clf)])


def _train_mlp_on_variant(name: str, df: pd.DataFrame, label_names: dict[int, str], X_tfidf: np.ndarray, y: np.ndarray) -> dict | None:
    """Train MLP classifier specifically on binary_no_neutral variant."""
    try:
        logger.info("Training MLP on variant: %s", name)
        y_train, y_val, y_test = y[:len(df)//3], y[len(df)//3:2*len(df)//3], y[2*len(df)//3:]
        X_train, X_val, X_test = X_tfidf[:len(df)//3], X_tfidf[len(df)//3:2*len(df)//3], X_tfidf[2*len(df)//3:]
        
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            batch_size=32,
            learning_rate_init=0.001,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
        )
        mlp.fit(X_train_scaled, y_train)
        
        y_pred_test = mlp.predict(X_test_scaled)
        test_metrics = _evaluate(y_test, y_pred_test, label_names)
        
        model_name = f"{name}__TF-IDF_WordChar_MLP"
        logger.info("  %s: acc=%s f1=%s", model_name, test_metrics["accuracy"], test_metrics["f1_macro"])
        
        return {
            model_name: {
                "type": "variant_text_ngram_mlp",
                "variant": name,
                "features": "tfidf_word_char_ngrams",
                "balance": "class_weight",
                "label_names": {str(k): v for k, v in label_names.items()},
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
            }
        }
    except Exception as exc:
        logger.exception("MLP training failed: %s", str(exc))
        return None


def _train_lstm_tokenizer_on_variant(name: str, texts: list[str], max_vocab: int = 10000, max_len: int = 100) -> tuple[dict, np.ndarray] | None:
    """Prepare tokenized sequences for LSTM training."""
    if not KERAS_AVAILABLE:
        return None
    
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
        
        return {
            "tokenizer": tokenizer,
            "vocab_size": min(len(tokenizer.word_index) + 1, max_vocab),
            "max_len": max_len,
        }, padded
    except Exception as exc:
        logger.exception("LSTM tokenization failed: %s", str(exc))
        return None


def _train_lstm_on_variant(name: str, df: pd.DataFrame, label_names: dict[int, str], X_seq: np.ndarray, y: np.ndarray) -> dict | None:
    """Train LSTM classifier on binary_no_neutral variant."""
    if not KERAS_AVAILABLE:
        logger.info("Skipping LSTM training (Keras not available)")
        return None
    
    try:
        logger.info("Training LSTM on variant: %s", name)
        
        # Split data
        indices = np.arange(len(X_seq))
        np.random.shuffle(indices)
        train_idx = indices[:int(0.7*len(indices))]
        val_idx = indices[int(0.7*len(indices)):int(0.85*len(indices))]
        test_idx = indices[int(0.85*len(indices)):]
        
        X_train, y_train = X_seq[train_idx], y[train_idx]
        X_val, y_val = X_seq[val_idx], y[val_idx]
        X_test, y_test = X_seq[test_idx], y[test_idx]
        
        vocab_size = int(np.max(X_seq)) + 1
        embedding_dim = 64
        
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=X_train.shape[1]),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        # Map predictions back to original labels
        label_list = sorted(label_names.keys())
        y_pred_mapped = np.array([label_list[int(p)] for p in y_pred])
        y_test_mapped = np.array([label_list[int(t)] for t in y_test])
        
        test_metrics = _evaluate(y_test_mapped, y_pred_mapped, label_names)
        
        model_name = f"{name}__TF-IDF_WordChar_LSTM"
        logger.info("  %s: acc=%s f1=%s", model_name, test_metrics["accuracy"], test_metrics["f1_macro"])
        
        return {
            model_name: {
                "type": "variant_text_ngram_lstm",
                "variant": name,
                "features": "lstm_embeddings",
                "balance": "class_weight",
                "label_names": {str(k): v for k, v in label_names.items()},
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
            }
        }
    except Exception as exc:
        logger.exception("LSTM training failed: %s", str(exc))
        return None


def _prepare_base_df(csv_path: str | None) -> pd.DataFrame:
    df = load_labeled_data(csv_path)
    if df.empty:
        return df

    df = df.copy()
    df["row_id"] = np.arange(len(df))
    df["text_clean"] = df["text"].apply(lambda t: preprocess(t, use_tokenizer=True, remove_sw=True))
    df = df[df["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
    return df


def _find_label_issues(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        logger.warning("cleanlab is not installed; skipping confident-learning audit.")
        return pd.DataFrame()

    y = df["sentiment"].to_numpy()
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class))
    clf = _text_model("logreg")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    logger.info("Computing out-of-fold probabilities for confident learning...")
    pred_probs = cross_val_predict(
        clf,
        df["text_clean"].to_numpy(),
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )
    issue_idx = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    if len(issue_idx) == 0:
        return pd.DataFrame()

    pred = pred_probs.argmax(axis=1)
    issue_df = df.iloc[issue_idx].copy()
    issue_df["predicted_label"] = [LABEL_NAMES.get(int(p), str(p)) for p in pred[issue_idx]]
    issue_df["given_label"] = [LABEL_NAMES.get(int(v), str(v)) for v in y[issue_idx]]
    issue_df["given_label_probability"] = pred_probs[issue_idx, y[issue_idx]]
    issue_df["predicted_label_probability"] = pred_probs[issue_idx, pred[issue_idx]]
    issue_df = issue_df.sort_values("given_label_probability", ascending=True)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / "label_issues_cleanlab.csv"
    issue_df[[
        "row_id",
        "rating",
        "given_label",
        "predicted_label",
        "given_label_probability",
        "predicted_label_probability",
        "label_source",
        "text",
    ]].to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("Cleanlab label issues saved: %s (%s rows)", out, len(issue_df))
    return issue_df


def _mixed_variant(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    conflict = out["label_source"].fillna("").str.contains("absa_conflict", regex=False)
    out.loc[conflict, "sentiment"] = MIXED_LABEL
    out["sentiment_name"] = out["sentiment"].map(MIXED_LABEL_NAMES)
    return out


def _split(df: pd.DataFrame):
    y = df["sentiment"].to_numpy()
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42, stratify=y)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=42,
        stratify=df_temp["sentiment"].to_numpy(),
    )
    return df_train, df_val, df_test


def _train_one_variant(name: str, df: pd.DataFrame, label_names: dict[int, str]) -> dict:
    logger.info("Training variant: %s (%s samples)", name, len(df))
    df_train, df_val, df_test = _split(df)
    y_train = df_train["sentiment"].to_numpy()
    y_val = df_val["sentiment"].to_numpy()
    y_test = df_test["sentiment"].to_numpy()
    y_all = df["sentiment"].to_numpy()

    results = {}
    trained = {}
    
    # Standard models: TF-IDF + LogReg and TF-IDF + LinearSVC
    for model_name, model in {
        "TFIDF_WordChar_LogisticRegression": _text_model("logreg"),
        "TFIDF_WordChar_LinearSVC": _text_model("svc"),
    }.items():
        full_name = f"{name}__{model_name}"
        try:
            model.fit(df_train["text_clean"].to_numpy(), y_train)
            val_metrics = _evaluate(y_val, model.predict(df_val["text_clean"].to_numpy()), label_names)
            test_metrics = _evaluate(y_test, model.predict(df_test["text_clean"].to_numpy()), label_names)
            results[full_name] = {
                "type": "variant_text_ngram",
                "variant": name,
                "features": "tfidf_word_char_ngrams",
                "balance": "class_weight",
                "label_names": {str(k): v for k, v in label_names.items()},
                "val": val_metrics,
                "test": test_metrics,
                "f1_macro": test_metrics["f1_macro"],
                "accuracy": test_metrics["accuracy"],
            }
            trained[full_name] = model
            logger.info("  %s: acc=%s f1=%s", full_name, test_metrics["accuracy"], test_metrics["f1_macro"])
        except Exception as exc:
            logger.exception("  %s failed", full_name)
            results[full_name] = {"type": "variant_text_ngram", "variant": name, "error": str(exc)}
    
    # For binary_no_neutral variant, also train MLP models
    if name == "binary_no_neutral" and len(df) > 100:
        logger.info("Training MLP models for binary classification...")
        
        # Build TF-IDF features for the entire dataset
        vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=70000,
                sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.95,
                max_features=70000,
                sublinear_tf=True,
            )),
        ])
        
        try:
            X_tfidf = vectorizer.fit_transform(df["text_clean"].to_numpy())
            mlp_results = _train_mlp_on_variant(name, df, label_names, X_tfidf.toarray(), y_all)
            if mlp_results:
                results.update(mlp_results)
                trained.update({k: None for k in mlp_results.keys()})  # Save model separately
        except Exception as exc:
            logger.exception("MLP training failed: %s", str(exc))

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return {"name": name, "error": "all_models_failed", "models": results}
    best_name = max(valid, key=lambda k: valid[k]["f1_macro"])
    return {
        "name": name,
        "sample_count": len(df),
        "split": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "distribution": _distribution(df["sentiment"].to_numpy(), label_names),
        "models": results,
        "best_name": best_name,
        "best_result": valid[best_name],
        "best_model": trained[best_name] if best_name in trained else None,
        "label_names": label_names,
    }


def _save_training_results(record: dict) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if TRAINING_RESULTS_FILE.exists():
        try:
            existing = json.loads(TRAINING_RESULTS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(record)
    TRAINING_RESULTS_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def train_variants(csv_path: str | None = None) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df = _prepare_base_df(csv_path)
    if df.empty or len(df) < 50:
        return {"status": "failed", "reason": "insufficient_data", "sample_count": len(df)}

    run_id = f"variant_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    issues = _find_label_issues(df)
    issue_row_ids = set(issues["row_id"].astype(int).tolist()) if not issues.empty else set()

    variants: list[tuple[str, pd.DataFrame, dict[int, str]]] = [
        ("original_3class", df.copy(), LABEL_NAMES),
    ]
    if issue_row_ids:
        pruned = df[~df["row_id"].isin(issue_row_ids)].copy()
        variants.append(("cleanlab_pruned_3class", pruned, LABEL_NAMES))

    binary = df[df["sentiment"].isin([0, 2])].copy()
    variants.append(("binary_no_neutral", binary, BINARY_LABEL_NAMES))

    mixed = _mixed_variant(df)
    if mixed["sentiment"].nunique() > 3 and mixed["sentiment"].value_counts().min() >= 20:
        variants.append(("mixed_conflict_4class", mixed, MIXED_LABEL_NAMES))
    else:
        logger.warning("Skipping mixed_conflict_4class because the mixed class is too small.")

    variant_records = []
    all_model_results = {}
    trained_models = {}
    label_maps = {}
    variant_artifacts = {}
    for variant_name, variant_df, label_names in variants:
        result = _train_one_variant(variant_name, variant_df, label_names)
        variant_records.append({k: v for k, v in result.items() if k != "best_model"})
        all_model_results.update(result.get("models", {}))
        if "best_model" in result:
            trained_models[result["best_name"]] = result["best_model"]
            label_maps[result["best_name"]] = label_names
            variant_dir = MODELS_DIR / "variants"
            variant_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = variant_dir / f"{result['best_name']}.pkl"
            joblib.dump(result["best_model"], artifact_path)
            variant_artifacts[variant_name] = {
                "model_name": result["best_name"],
                "model_path": str(artifact_path),
                "label_names": {str(k): v for k, v in label_names.items()},
                "f1_macro": result["best_result"]["f1_macro"],
                "accuracy": result["best_result"]["accuracy"],
            }

    valid = {k: v for k, v in all_model_results.items() if "error" not in v}
    if not valid:
        return {"status": "failed", "reason": "all_variants_failed"}

    best_name = max(valid, key=lambda k: valid[k]["f1_macro"])
    best = valid[best_name]
    best_label_names = label_maps[best_name]

    joblib.dump(trained_models[best_name], MODELS_DIR / "best_model.pkl")
    best_meta = {
        "name": best_name,
        "backend": "sklearn_text",
        "input_type": "text_clean",
        "model_path": str(MODELS_DIR / "best_model.pkl"),
        "label_names": {str(k): v for k, v in best_label_names.items()},
        "trained_by": "variant_trainer",
    }
    BEST_MODEL_META_FILE.write_text(json.dumps(best_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "sample_count": len(df),
        "embedding": "tfidf_word_char_ngrams",
        "feature_dim": "sparse_text",
        "balance_method": "class_weight='balanced'",
        "label_quality": {
            "method": "confident_learning_cleanlab",
            "issue_count": len(issue_row_ids),
            "issue_csv": str(ANALYSIS_DIR / "label_issues_cleanlab.csv") if issue_row_ids else None,
        },
        "variants": variant_records,
        "variant_artifacts": variant_artifacts,
        "models": all_model_results,
        "best_model": {
            "name": best_name,
            "type": best["type"],
            "variant": best["variant"],
            "f1_macro": best["f1_macro"],
            "accuracy": best["accuracy"],
        },
        "best_model_artifact": best_meta,
    }
    save_experiment(record)
    _save_training_results(record)

    summary_rows = []
    for name, result in sorted(valid.items(), key=lambda item: item[1]["f1_macro"], reverse=True):
        summary_rows.append({
            "model": name,
            "variant": result["variant"],
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
            "f1_weighted": result["test"]["f1_weighted"],
            "precision_macro": result["test"]["precision_macro"],
            "recall_macro": result["test"]["recall_macro"],
        })
    pd.DataFrame(summary_rows).to_csv(ANALYSIS_DIR / "variant_training_summary.csv", index=False, encoding="utf-8-sig")

    return {"status": "success", **record}
