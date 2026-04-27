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
from sklearn.naive_bayes import GaussianNB
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
    }


# ── Training results snapshot (for final report) ────────────────

_ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "analysis"
_TRAINING_RESULTS_FILE = _ANALYSIS_DIR / "training_results.json"


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

    logger.info("Generating features (FastText 300-dim + 5 extra features)...")
    X = _build_features(ft_model, df["text_clean"].values)
    y = df["sentiment"].values
    logger.info(f"Feature matrix shape: {X.shape}")  # (n_samples, 305)

    # ── 4. Split (70/15/15) ─────────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,
    )
    logger.info(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

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

    for name, model in ml_models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model

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
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {"type": "individual", "error": str(e)}

    # ── 7. Ensemble (soft voting from all successful models) ────
    logger.info("── Building Ensemble (soft voting) ──")
    ensemble_models = {n: m for n, m in trained_models.items() if hasattr(m, "predict_proba")}

    if len(ensemble_models) >= 2:
        try:
            # Manual soft voting: average predicted probabilities
            def ensemble_predict(X):
                proba_list = [m.predict_proba(X) for m in ensemble_models.values()]
                avg_proba = np.mean(proba_list, axis=0)
                return np.argmax(avg_proba, axis=1)

            y_val_ens = ensemble_predict(X_val)
            y_test_ens = ensemble_predict(X_test)

            # Map back to original label space (classes may be 0,1,2)
            classes = np.unique(y_train)
            y_val_ens = np.array([classes[i] for i in y_val_ens])
            y_test_ens = np.array([classes[i] for i in y_test_ens])

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
            # Save ensemble as dict of models
            trained_models["Ensemble_SoftVote"] = {"_ensemble": list(ensemble_models.keys())}
            logger.info(f"  Ensemble: acc={test_metrics['accuracy']}, f1={test_metrics['f1_macro']}")
        except Exception as e:
            logger.error(f"  Ensemble failed: {e}")
            results["Ensemble_SoftVote"] = {"type": "ensemble", "error": str(e)}

    # ── 8. Save best model ──────────────────────────────────────
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return {"status": "failed", "reason": "all_models_failed"}

    best_name = max(valid, key=lambda k: valid[k]["f1_macro"])
    logger.info(f"Best model: {best_name} F1={valid[best_name]['f1_macro']}")

    joblib.dump(trained_models[best_name], MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
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
        "absa_summary": absa_result.get("summary", {}),
        "absa_aspect_mentions": absa_result.get("aspect_mentions", 0),
    }
    save_experiment(experiment_record)
    _save_training_results(experiment_record)
    return {"status": "success", **experiment_record}


def predict(texts: list[str]) -> list[dict]:
    """Load frozen FastText + scaler + best model → predict."""
    model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    if not model_path.exists():
        raise FileNotFoundError("No trained model. Run `python run.py train` first.")

    ft_model = _ensure_fasttext_model()
    ml_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    results = []
    for text in texts:
        clean = preprocess(text, use_tokenizer=True, remove_sw=True)
        features = _build_features(ft_model, [clean])
        if scaler:
            features = scaler.transform(features)
        pred = ml_model.predict(features)[0]
        results.append({
            "text": text[:100],
            "sentiment": int(pred),
            "sentiment_name": LABEL_NAMES[pred],
        })
    return results
