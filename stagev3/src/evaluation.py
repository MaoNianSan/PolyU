from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from . import config


def predict_with_scores(estimator, X) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, y_score) using v2-compatible probability thresholding.

    v2 evaluated SVC models through calibrated `predict_proba >= 0.5`. The
    previous v3 path used `predict()`, which can disagree with probability
    thresholding for SVC. This helper centralizes the corrected behavior.
    """
    if hasattr(estimator, "predict_proba"):
        try:
            y_score = np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
            y_pred = (y_score >= float(config.DECISION_THRESHOLD)).astype(int)
            return y_pred, y_score
        except Exception:
            pass
    if hasattr(estimator, "decision_function"):
        try:
            raw = np.asarray(estimator.decision_function(X), dtype=float)
            lo, hi = np.nanmin(raw), np.nanmax(raw)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                y_score = (raw - lo) / (hi - lo)
            else:
                y_score = np.full_like(raw, 0.5, dtype=float)
            y_pred = (y_score >= float(config.DECISION_THRESHOLD)).astype(int)
            return y_pred, y_score
        except Exception:
            pass
    y_pred = np.asarray(estimator.predict(X)).astype(int)
    y_score = y_pred.astype(float)
    return y_pred, y_score


def score_model(estimator, X, y_true) -> dict[str, float | int]:
    y_pred, y_score = predict_with_scores(estimator, X)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "correct_n": int(np.sum(np.asarray(y_pred) == np.asarray(y_true))),
        "total_n": int(len(y_true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
