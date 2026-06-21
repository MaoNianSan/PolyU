from __future__ import annotations
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score, roc_auc_score,
)

from . import config as cfg


def positive_probability(estimator, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(f"{type(estimator).__name__} must provide predict_proba for stagev6 routing.")
    p = estimator.predict_proba(X)[:, 1]
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def metrics_from_hard_and_probability(y_true: Iterable[int], y_pred: Iterable[int], y_prob: Iterable[float]) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    def safe(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return np.nan
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(safe(roc_auc_score, y, prob)),
        "pr_auc": float(safe(average_precision_score, y, prob)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "log_loss": float(safe(log_loss, y, np.column_stack([1-prob, prob]))),
        "brier": float(safe(brier_score_loss, y, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": cfg.DECISION_THRESHOLD,
    }


def stratified_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str, n_boot: int) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    records = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(idx0, size=len(idx0), replace=True),
            rng.choice(idx1, size=len(idx1), replace=True),
        ])
        records.append(metrics_from_hard_and_probability(y[idx], pred[idx], prob[idx]))
    frame = pd.DataFrame(records)
    rows = []
    for metric in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "mcc"]:
        vals = frame[metric].dropna().to_numpy(dtype=float)
        rows.append({
            "model_name": model_name, "metric": metric,
            "bootstrap_mean": float(np.mean(vals)) if len(vals) else np.nan,
            "ci_low": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
            "ci_high": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
            "n_boot": int(n_boot),
        })
    return pd.DataFrame(rows)


def component_oof_probabilities(
    fitted_gates: dict[str, object],
    fitted_branches: dict[str, object],
    X_gate: dict[str, pd.DataFrame],
    X_branch: dict[str, pd.DataFrame],
    y_late: np.ndarray,
    y_ad: np.ndarray,
    nonlate_mask: np.ndarray,
    outer_cv,
    strata: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """OOF probabilities for all reusable components on shared 3-stratum folds.

    Each validation fold is unseen by both its gate and branch. Branch fitting uses only
    *true non-late* training samples, consistent with the intended conditional task.
    Hyperparameters are fixed to full-training GridSearchCV selections, mirroring the
    stagev5 non-nested OOF reporting convention.
    """
    n = len(y_ad)
    gate_sum = {k: np.zeros(n, dtype=float) for k in fitted_gates}
    branch_sum = {k: np.zeros(n, dtype=float) for k in fitted_branches}
    counts = np.zeros(n, dtype=int)
    anchor = next(iter(X_gate.values()))
    for tr, va in outer_cv.split(anchor, strata):
        branch_tr = tr[nonlate_mask[tr]]
        if len(np.unique(y_ad[branch_tr])) != 2:
            raise RuntimeError("A stagev6 branch training fold lacks a binary AD/control class.")
        for component_id, fitted in fitted_gates.items():
            est = clone(fitted)
            est.fit(X_gate[component_id].iloc[tr], y_late[tr])
            gate_sum[component_id][va] += positive_probability(est, X_gate[component_id].iloc[va])
        for component_id, fitted in fitted_branches.items():
            est = clone(fitted)
            est.fit(X_branch[component_id].iloc[branch_tr], y_ad[branch_tr])
            branch_sum[component_id][va] += positive_probability(est, X_branch[component_id].iloc[va])
        counts[va] += 1
    if (counts == 0).any():
        raise RuntimeError("At least one sample received no OOF cascade prediction.")
    return ({k: v / counts for k, v in gate_sum.items()}, {k: v / counts for k, v in branch_sum.items()})
