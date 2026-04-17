"""Data balancing strategies for imbalanced sentiment classes."""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def get_distribution(y: np.ndarray, label_names: dict[int, str] | None = None) -> dict:
    """Get class distribution as dict."""
    counts = Counter(y)
    total = len(y)
    dist = {}
    for cls in sorted(counts.keys()):
        name = label_names.get(cls, str(cls)) if label_names else str(cls)
        dist[name] = {"count": counts[cls], "percentage": round(counts[cls] / total * 100, 1)}
    return dist


def balance_with_smote(X, y: np.ndarray, random_state: int = 42):
    """Apply SMOTE to balance classes. Falls back to class_weight if SMOTE fails.

    Returns (X_balanced, y_balanced, method_used)
    """
    try:
        from imblearn.over_sampling import SMOTE

        # SMOTE needs at least k_neighbors+1 samples in minority class
        min_class_count = min(Counter(y).values())
        k_neighbors = min(5, min_class_count - 1)

        if k_neighbors < 1:
            logger.warning(f"Minority class has only {min_class_count} samples, skipping SMOTE")
            return X, y, "none"

        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"SMOTE: {len(y)} → {len(y_res)} samples")
        return X_res, y_res, "smote"

    except ImportError:
        logger.warning("imbalanced-learn not installed, skipping SMOTE")
        return X, y, "none"
    except Exception as e:
        logger.warning(f"SMOTE failed ({e}), skipping")
        return X, y, "none"


def balance_with_class_weight(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for sklearn estimators."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    logger.info(f"Class weights: {weight_dict}")
    return weight_dict
