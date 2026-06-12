"""Stratified bootstrap confidence intervals for external-test metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils import resample

import config
from evaluation import metrics_from_predictions


def stratified_bootstrap_ci(y_true, y_prob, model_name: str, n_boot: int = config.BOOTSTRAP_N, random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(random_state)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rows = []
    for _ in range(n_boot):
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
        idx = np.concatenate([b0, b1])
        rows.append(metrics_from_predictions(y[idx], p[idx]))
    boot = pd.DataFrame(rows)
    out = []
    for metric in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "mcc"]:
        vals = boot[metric].dropna().values
        if len(vals) == 0:
            lo = hi = mean = np.nan
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            mean = vals.mean()
        out.append({"model_name": model_name, "metric": metric, "bootstrap_mean": mean, "ci_low": lo, "ci_high": hi, "n_boot": n_boot})
    return pd.DataFrame(out)
