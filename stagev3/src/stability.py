from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def summarize_stability(stability: pd.DataFrame) -> pd.DataFrame:
    keys = ["early_variant", "model_spec_id", "feature_block", "model_name", "model_variant", "is_special_model", "cv_mode"]
    rows = []
    for vals, g in stability.groupby(keys, dropna=False):
        d = dict(zip(keys, vals))
        acc = g["external_accuracy"].astype(float)
        n = len(acc)
        mean = float(acc.mean())
        std = float(acc.std(ddof=1)) if n > 1 else 0.0
        if n > 1 and std > 0:
            t = float(stats.t.ppf(0.975, df=n - 1))
            low = mean - t * std / np.sqrt(n)
            high = mean + t * std / np.sqrt(n)
        else:
            low = high = mean
        d.update({
            "n_seeds": int(n),
            "external_accuracy_mean": mean,
            "external_accuracy_std": std,
            "external_accuracy_min": float(acc.min()),
            "external_accuracy_q25": float(acc.quantile(0.25)),
            "external_accuracy_median": float(acc.median()),
            "external_accuracy_q75": float(acc.quantile(0.75)),
            "external_accuracy_max": float(acc.max()),
            "external_accuracy_ci95_low": float(low),
            "external_accuracy_ci95_high": float(high),
            "external_accuracy_95ci": f"[{low:.6f}, {high:.6f}]",
            "external_f1_mean": float(g["external_f1"].mean()),
            "external_auc_mean": float(g["external_auc"].mean()),
        })
        rows.append(d)
    return pd.DataFrame(rows).sort_values(["external_accuracy_mean", "external_f1_mean", "external_auc_mean"], ascending=False).reset_index(drop=True)


def attach_accuracy_ci(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["early_variant", "model_spec_id", "feature_block", "model_name", "model_variant", "is_special_model", "cv_mode"]
    ci_cols = ["external_accuracy_ci95_low", "external_accuracy_ci95_high", "external_accuracy_95ci"]
    out = df.drop(columns=[c for c in ci_cols if c in df.columns], errors="ignore")
    if summary.empty:
        for c in ci_cols:
            out[c] = np.nan if c != "external_accuracy_95ci" else ""
        return out
    return out.merge(summary[keys + ci_cols], on=keys, how="left")
