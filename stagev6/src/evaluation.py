from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score, roc_auc_score


def safe_metric(fn, *args, default=np.nan):
    try:
        return fn(*args)
    except Exception:
        return default


def get_positive_prob(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        out = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        z = estimator.decision_function(X)
        out = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-12)
    else:
        out = estimator.predict(X).astype(float)
    return np.clip(np.asarray(out, dtype=float), 1e-6, 1 - 1e-6)


def metrics_from_hard_and_prob(y_true, y_pred, p_ad) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    p = np.clip(np.asarray(p_ad, dtype=float), 1e-6, 1 - 1e-6)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": safe_metric(roc_auc_score, y, p),
        "pr_auc": safe_metric(average_precision_score, y, p),
        "mcc": matthews_corrcoef(y, pred),
        "log_loss": safe_metric(log_loss, y, np.column_stack([1-p,p])),
        "brier": safe_metric(brier_score_loss, y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "gate_threshold": 0.50, "branch_threshold": 0.50,
    }


def metrics_binary_prob(y_true, p, threshold=0.5) -> dict:
    pred = (np.asarray(p) >= threshold).astype(int)
    return metrics_from_hard_and_prob(y_true, pred, p)
