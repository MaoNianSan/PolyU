"""Metrics for Stagev8.5's MMSE-informed ordinal severity output."""
from __future__ import annotations
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y: Iterable[int], p: Iterable[float], threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(list(y), dtype=int)
    p = np.asarray(list(p), dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def three_strata_metrics(y: Iterable[str], pred: Iterable[str], labels: list[str]) -> dict[str, Any]:
    y = np.asarray(list(y), dtype=str)
    pred = np.asarray(list(pred), dtype=str)
    recalls = []
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
    }
    for label in labels:
        z = y == label
        rec = float(((pred == label) & z).sum() / z.sum()) if z.sum() else float("nan")
        prec_den = (pred == label).sum()
        prec = float(((pred == label) & z).sum() / prec_den) if prec_den else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if np.isfinite(rec) and (prec + rec) else 0.0
        out[f"precision_{label}"] = prec
        out[f"recall_{label}"] = rec
        out[f"f1_{label}"] = f1
        out[f"support_{label}"] = int(z.sum())
        if np.isfinite(rec):
            recalls.append(rec)
    out["balanced_accuracy"] = float(np.mean(recalls)) if recalls else float("nan")
    return out


def ordinal_continuous_metrics(mmse: Iterable[float], severity_score: Iterable[float]) -> dict[str, Any]:
    mmse = np.asarray(list(mmse), dtype=float)
    score = np.asarray(list(severity_score), dtype=float)
    valid = np.isfinite(mmse) & np.isfinite(score)
    mmse = mmse[valid]
    score = score[valid]
    if len(mmse) < 3:
        return {"n": int(len(mmse)), "spearman_rho": float("nan"), "spearman_p": float("nan"), "kendall_tau": float("nan"), "kendall_p": float("nan"), "pairwise_ordinal_accuracy": float("nan"), "n_informative_pairs": 0}
    # Greater 30-MMSE means greater cognitive severity; a higher Stagev8.5 score
    # should be concordant with it.
    severity_target = 30.0 - mmse
    sp = spearmanr(score, severity_target)
    kt = kendalltau(score, severity_target)
    credit: list[float] = []
    for i, j in combinations(range(len(mmse)), 2):
        if mmse[i] == mmse[j]:
            continue
        # lower MMSE should correspond to higher score
        expected = np.sign(mmse[j] - mmse[i])
        observed = np.sign(score[i] - score[j])
        if observed == 0:
            credit.append(0.5)
        else:
            credit.append(1.0 if observed == expected else 0.0)
    return {
        "n": int(len(mmse)),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
        "pairwise_ordinal_accuracy": float(np.mean(credit)) if credit else float("nan"),
        "n_informative_pairs": int(len(credit)),
    }


def calibration_table(y: Iterable[int], p: Iterable[float], n_bins: int = 5) -> pd.DataFrame:
    y = np.asarray(list(y), dtype=int)
    p = np.asarray(list(p), dtype=float)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["bin", "mean_predicted_probability", "fraction_positive", "n"])
    frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    bins = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    rows = []
    # calibration_curve removes empty bins, so pair values by non-empty bins.
    k = 0
    for b, count in enumerate(counts):
        if count:
            rows.append({"bin": int(b), "mean_predicted_probability": float(mean[k]), "fraction_positive": float(frac[k]), "n": int(count)})
            k += 1
    return pd.DataFrame(rows)


def confusion_long(y: Iterable[str], pred: Iterable[str], labels: list[str], true_name: str = "true_mmse_stratum", pred_name: str = "predicted_mmse_stratum") -> pd.DataFrame:
    cm = confusion_matrix(list(y), list(pred), labels=labels)
    rows = []
    for i, truth in enumerate(labels):
        for j, guess in enumerate(labels):
            rows.append({true_name: truth, pred_name: guess, "count": int(cm[i, j])})
    return pd.DataFrame(rows)
