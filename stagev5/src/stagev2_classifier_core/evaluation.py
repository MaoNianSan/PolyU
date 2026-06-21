"""Evaluation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

import config


def safe_metric(fn, *args, default=np.nan):
    try:
        return fn(*args)
    except Exception:
        return default


def metrics_from_predictions(y_true, y_prob, threshold: float = config.DECISION_THRESHOLD) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
        "pr_auc": safe_metric(average_precision_score, y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": safe_metric(log_loss, y_true, np.column_stack([1 - y_prob, y_prob])),
        "brier": safe_metric(brier_score_loss, y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
    }


def get_positive_prob(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        p = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)
    else:
        p = estimator.predict(X).astype(float)
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def repeated_cv_predict_proba(estimator, X, y, cv) -> np.ndarray:
    y = np.asarray(y).astype(int)
    total = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    for tr, va in cv.split(X, y):
        est = clone(estimator)
        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        X_va = X.iloc[va] if hasattr(X, "iloc") else X[va]
        est.fit(X_tr, y[tr])
        total[va] += get_positive_prob(est, X_va)
        counts[va] += 1
    counts[counts == 0] = 1
    return total / counts


def summarize_cv_metrics(estimator, X, y, cv) -> dict:
    p = repeated_cv_predict_proba(estimator, X, y, cv)
    metrics = metrics_from_predictions(y, p)
    return {f"cv_{k}": v for k, v in metrics.items()} | {"cv_repeated_oof_available": True}


def prediction_frame(test_df: pd.DataFrame, y_prob, model_name: str, extra: dict | None = None) -> pd.DataFrame:
    y = test_df["__y__"].astype(int).to_numpy()
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= config.DECISION_THRESHOLD).astype(int)
    df = pd.DataFrame({
        "sample_id": test_df["__sample_id__"].astype(str).values,
        "y_true": y,
        "label_disease": test_df["__y__"].astype(int).values,
        "label_early": test_df["label_mild"].astype(int).values,
        "label_middle": test_df["label_moderate"].astype(int).values,
        "label_late": test_df["label_severe"].astype(int).values,
        "label_normal": test_df["label_normal"].astype(int).values,
        "label_mild": test_df["label_mild"].astype(int).values,
        "label_moderate": test_df["label_moderate"].astype(int).values,
        "label_severe": test_df["label_severe"].astype(int).values,
        "severity_group": test_df["severity_group"].astype(str).values,
        "mmse": test_df["mmse"].values,
        "model_name": model_name,
        "y_pred": pred,
        "p_ad": p,
        "correct": (pred == y).astype(int),
    })
    df["error_type"] = np.where(df["correct"] == 1, "correct", np.where(df["y_true"] == 0, "FP_normal", "FN_" + df["severity_group"].astype(str)))
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df
