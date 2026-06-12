from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .evaluation import fit_and_evaluate, select_best_params
from .models import model_grid


def run_protocol_for_seed(blocks: dict, y_train, y_test, early_variant: str, seed: int, show_progress: bool = False) -> pd.DataFrame:
    rows = []
    iterator = config.FEATURE_BLOCKS
    if show_progress:
        iterator = tqdm(iterator, desc=f"seed={seed} {early_variant}")
    for feature_block in iterator:
        X_train, X_test = blocks[feature_block]
        for model_name, grid in model_grid().items():
            best_params, cv_metrics = select_best_params(model_name, grid, X_train, y_train, seed)
            ext_metrics = fit_and_evaluate(model_name, best_params, X_train, y_train, X_test, y_test, seed)
            rows.append({
                "seed": seed,
                "early_variant": early_variant,
                "model_name": model_name,
                "feature_block": feature_block,
                "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
                **cv_metrics,
                **ext_metrics,
            })
    return pd.DataFrame(rows)


def summarize_stability(stab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["early_variant", "model_name", "feature_block"]
    for group, sub in stab.groupby(keys, dropna=False):
        acc = sub["external_accuracy"].astype(float)
        rows.append({
            "early_variant": group[0],
            "model_name": group[1],
            "feature_block": group[2],
            "n_seeds": int(len(sub)),
            "external_accuracy_mean": float(acc.mean()),
            "external_accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "external_accuracy_min": float(acc.min()),
            "external_accuracy_q25": float(acc.quantile(0.25)),
            "external_accuracy_median": float(acc.median()),
            "external_accuracy_q75": float(acc.quantile(0.75)),
            "external_accuracy_max": float(acc.max()),
            "external_f1_mean": float(sub["external_f1"].astype(float).mean()),
            "external_auc_mean": float(sub["external_auc"].astype(float).mean()),
        })
    return pd.DataFrame(rows).sort_values(["external_accuracy_mean", "external_auc_mean", "external_f1_mean"], ascending=False)
