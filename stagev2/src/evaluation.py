from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .models import make_estimator


def positive_prob(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        p = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)
    else:
        p = estimator.predict(X).astype(float)
    return np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)


def binary_metrics(y_true, y_prob, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = positive_array(y_prob)
    y_pred = (y_prob >= config.DECISION_THRESHOLD).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}auc": float(auc) if np.isfinite(auc) else np.nan,
    }


def positive_array(y_prob) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1 - 1e-8)


def cv_oof_metrics(model_name: str, params: dict, X, y, seed: int) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in cv.split(X, y):
        est = make_estimator(model_name, params, seed)
        est.fit(X.iloc[tr], y[tr])
        oof[va] = positive_prob(est, X.iloc[va])
    return binary_metrics(y, oof, prefix="cv_")


def select_best_params(model_name: str, param_grid: list[dict], X, y, seed: int) -> tuple[dict, dict]:
    rows = []
    for params in param_grid:
        metrics = cv_oof_metrics(model_name, params, X, y, seed)
        rows.append({"params": params, **metrics})
    rows = sorted(rows, key=lambda r: (r["cv_accuracy"], np.nan_to_num(r["cv_auc"], nan=-1.0), r["cv_f1"]), reverse=True)
    return rows[0]["params"], {k: v for k, v in rows[0].items() if k != "params"}


def fit_and_evaluate(model_name: str, params: dict, X_train, y_train, X_test, y_test, seed: int) -> dict[str, Any]:
    est = make_estimator(model_name, params, seed)
    est.fit(X_train, y_train)
    p = positive_prob(est, X_test)
    y_pred = (p >= config.DECISION_THRESHOLD).astype(int)
    metrics = binary_metrics(y_test, p, prefix="external_")
    metrics["external_correct_n"] = int((y_pred == np.asarray(y_test).astype(int)).sum())
    metrics["external_total_n"] = int(len(y_test))
    return metrics
