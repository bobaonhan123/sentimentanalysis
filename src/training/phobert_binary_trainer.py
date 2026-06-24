"""Fine-tune PhoBERT for binary negative vs non-negative company review sentiment."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from underthesea import word_tokenize

from src.artifacts.paths import phobert_paths
from src.training.labeling import (
    BINARY_LABEL_NAMES,
    DEFAULT_BINARY_FRAMING,
    apply_binary_framing,
    load_glassdoor_english_data,
    load_labeled_data,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_CONFIG = {
    "vi": {
        "model_name": "vinai/phobert-base-v2",
        "model_key": "PhoBERT",
        "model_family": "PhoBERT",
        "segment": True,
    },
    "en": {
        "model_name": "distilbert-base-uncased",
        "model_key": "DistilBERT",
        "model_family": "DistilBERT",
        "segment": False,
    },
}
LABEL_NAMES = BINARY_LABEL_NAMES[DEFAULT_BINARY_FRAMING]


class StepTimer:
    def __init__(self, message: str, *args):
        self.message = message
        self.args = args
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        logger.info("START | " + self.message, *self.args)
        return self

    def __exit__(self, exc_type, exc, traceback):
        elapsed = time.perf_counter() - self.started
        if exc_type:
            logger.exception("FAIL  | " + self.message + " | %.2fs", *self.args, elapsed)
        else:
            logger.info("DONE  | " + self.message + " | %.2fs", *self.args, elapsed)
        return False


def _attach_run_file_logger(run_id: str, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    root_logger = logging.getLogger()
    handler_name = f"phobert_binary_file:{log_path}"
    for handler in root_logger.handlers:
        if getattr(handler, "name", None) == handler_name:
            return log_path
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.name = handler_name
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)
    logger.info("LOG_FILE | %s", log_path)
    return log_path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _segment(text: str) -> str:
    if not text:
        return ""
    return word_tokenize(str(text), format="text")


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, proba_positive: np.ndarray | None = None) -> dict:
    result = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            output_dict=True,
            zero_division=0,
        ),
    }
    if proba_positive is not None:
        best_threshold = 0.5
        best_metrics = dict(result)
        best_score = float(
            fbeta_score(
                y_true,
                (proba_positive >= best_threshold).astype(int),
                beta=2,
                labels=[0],
                average="macro",
                zero_division=0,
            )
        )
        for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
            pred = (proba_positive >= threshold).astype(int)
            metrics = _evaluate(y_true, pred)
            score = float(
                fbeta_score(
                    y_true,
                    pred,
                    beta=2,
                    labels=[0],
                    average="macro",
                    zero_division=0,
                )
            )
            if score > best_score:
                best_threshold = float(threshold)
                best_metrics = metrics
                best_score = score
        result["best_threshold"] = best_threshold
        result["threshold_metrics"] = best_metrics
        result["negative_f2"] = round(best_score, 4)
        result["non_positive_f2"] = round(best_score, 4)
    return result


@dataclass
class TextDataset(Dataset):
    texts: list[str]
    labels: list[int]
    tokenizer: AutoTokenizer
    max_len: int

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_len,
        )
        encoded["labels"] = int(self.labels[index])
        return encoded


def _predict_logits(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits_list.append(outputs.logits.detach().cpu().numpy())
            labels_list.append(labels)
    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)
    proba = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred = proba.argmax(axis=1)
    return labels, pred, proba[:, 1]


def _load_binary_frame(
    *,
    csv_path: str | None = None,
    language: str = "vi",
    max_examples: int | None = None,
    seed: int = 42,
    balanced_sample: bool = True,
) -> pd.DataFrame:
    lang = (language or "vi").strip().lower()
    if lang == "en":
        df = load_glassdoor_english_data(max_rows=max_examples or None, preprocessed=True)
    else:
        df = load_labeled_data(csv_path)
    df = apply_binary_framing(df, DEFAULT_BINARY_FRAMING)
    df = df.copy()
    df["label"] = df["sentiment"].astype(int)
    if lang == "en" and "text_clean" in df.columns:
        df["text_model"] = df["text_clean"].astype(str)
    else:
        df["text_model"] = df["text"].astype(str)

    if max_examples and len(df) > max_examples:
        if balanced_sample:
            neg = df[df["label"] == 0]
            pos = df[df["label"] == 1]
            n_neg = min(len(neg), max_examples // 2)
            n_pos = min(len(pos), max_examples - n_neg)
            df = pd.concat([
                neg.sample(n=n_neg, random_state=seed),
                pos.sample(n=n_pos, random_state=seed),
            ])
        else:
            df, _ = train_test_split(
                df,
                train_size=max_examples,
                random_state=seed,
                stratify=df["label"],
            )
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def train_phobert_binary(
    *,
    csv_path: str | None = None,
    language: str = "vi",
    max_examples: int | None = 512,
    epochs: int = 1,
    batch_size: int = 8,
    max_len: int = 160,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    balanced_sample: bool = True,
    device_name: str | None = None,
) -> dict:
    """Run a binary transformer fine-tune (PhoBERT for VI, DistilBERT for EN)."""
    _seed_everything(seed)
    lang = (language or "vi").strip().lower()
    if lang not in MODEL_CONFIG:
        return {"status": "failed", "reason": f"unsupported_language:{lang}"}
    cfg = MODEL_CONFIG[lang]
    model_name = cfg["model_name"]

    paths = phobert_paths(lang)
    paths.ensure_dirs()
    paths.phobert_production_checkpoint().mkdir(parents=True, exist_ok=True)

    run_started = time.perf_counter()
    run_id = f"{'phobert' if lang == 'vi' else 'distilbert'}_binary_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = _attach_run_file_logger(run_id, paths.logs_dir)
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(device_name) if device_name else _device()
    logger.info(
        "TRANSFORMER_BINARY | run_id=%s lang=%s model=%s device=%s max_examples=%s epochs=%s batch_size=%s max_len=%s",
        run_id,
        lang,
        model_name,
        device,
        max_examples,
        epochs,
        batch_size,
        max_len,
    )

    with StepTimer("load weak-labeled data"):
        df = _load_binary_frame(
            csv_path=csv_path,
            language=lang,
            max_examples=max_examples,
            seed=seed,
            balanced_sample=balanced_sample,
        )
        logger.info("DATA | rows=%s distribution=%s", len(df), df["label"].value_counts().sort_index().to_dict())

    if len(df) < 50 or df["label"].nunique() < 2:
        return {"status": "failed", "reason": "insufficient_binary_data", "sample_count": len(df)}

    if cfg["segment"]:
        with StepTimer("segment Vietnamese text for PhoBERT"):
            df["text_model"] = df["text_model"].apply(_segment)

    with StepTimer("train/val/test split"):
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"])
        logger.info("SPLIT | train=%s val=%s test=%s", len(train_df), len(val_df), len(test_df))

    with StepTimer("load tokenizer/model"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=lang == "en")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "negative", 1: "non_negative"},
            label2id={"negative": 0, "non_negative": 1},
        ).to(device)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        TextDataset(train_df["text_model"].tolist(), train_df["label"].tolist(), tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        TextDataset(val_df["text_model"].tolist(), val_df["label"].tolist(), tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        TextDataset(test_df["text_model"].tolist(), test_df["label"].tolist(), tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].to_numpy(),
    )
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    history = []
    best_val_f1 = -1.0
    best_dir = paths.phobert_production_checkpoint()
    run_checkpoint_dir = run_dir / "checkpoint"
    with StepTimer(f"fine-tune {cfg['model_key']}"):
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            progress = tqdm(train_loader, desc=f"{cfg['model_key']} epoch {epoch}/{epochs}", leave=False)
            for batch in progress:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if device.type == "mps":
                    torch.mps.empty_cache()
                losses.append(float(loss.detach().cpu()))
                progress.set_postfix(loss=f"{np.mean(losses):.4f}")

            y_val, pred_val, proba_val = _predict_logits(model, val_loader, device)
            val_metrics = _evaluate(y_val, pred_val, proba_val)
            row = {
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)), 4),
                "val_f1_macro": val_metrics["f1_macro"],
                "val_accuracy": val_metrics["accuracy"],
                "val_threshold": val_metrics["best_threshold"],
            }
            history.append(row)
            logger.info("EPOCH | %s", row)
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(run_checkpoint_dir)
                tokenizer.save_pretrained(run_checkpoint_dir)
                logger.info("BEST_SAVED | epoch=%s val_f1=%s path=%s", epoch, best_val_f1, best_dir)

    with StepTimer(f"evaluate best {cfg['model_key']}"):
        model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
        y_val, pred_val, proba_val = _predict_logits(model, val_loader, device)
        val_metrics = _evaluate(y_val, pred_val, proba_val)
        threshold = float(val_metrics["best_threshold"])
        y_test, pred_test, proba_test = _predict_logits(model, test_loader, device)
        test_metrics = _evaluate(y_test, pred_test, proba_test)
        threshold_test_pred = (proba_test >= threshold).astype(int)
        threshold_test_metrics = _evaluate(y_test, threshold_test_pred)

    result = {
        "status": "success",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "backend": "torch_transformers",
        "model_family": cfg["model_family"],
        "model_key": cfg["model_key"],
        "language": lang,
        "device": str(device),
        "sample_count": len(df),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": max_len,
        "max_examples": max_examples,
        "balanced_sample": balanced_sample,
        "class_weights": [round(float(x), 4) for x in class_weights],
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
        "threshold": threshold,
        "threshold_test": threshold_test_metrics,
        "best_model_dir": str(best_dir),
        "run_dir": str(run_dir),
        "log_file": str(log_path),
        "duration_seconds": round(time.perf_counter() - started, 2),
    }
    (run_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    paths.production_dir.joinpath("results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    logger.info(
        "PHOBERT_BINARY_DONE | lang=%s val_f1=%s test_f1=%s threshold=%s threshold_test_f1=%s duration=%.2fs",
        lang,
        val_metrics["f1_macro"],
        test_metrics["f1_macro"],
        threshold,
        threshold_test_metrics["f1_macro"],
        time.perf_counter() - started,
    )
    return result
