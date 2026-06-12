from __future__ import annotations

import pandas as pd

from . import config


def scale_gain_seed2026(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for early_variant in sorted(main["early_variant"].unique()):
        for model_name in sorted(main["model_name"].unique()):
            sub = main[(main["early_variant"] == early_variant) & (main["model_name"] == model_name)]
            for raw, scale in config.SCALE_PAIRS:
                a = sub[sub["feature_block"] == raw]
                b = sub[sub["feature_block"] == scale]
                if len(a) and len(b):
                    raw_acc = float(a.iloc[0]["external_accuracy"])
                    scale_acc = float(b.iloc[0]["external_accuracy"])
                    rows.append({
                        "early_variant": early_variant,
                        "model_name": model_name,
                        "raw_feature_block": raw,
                        "scale_feature_block": scale,
                        "raw_external_accuracy": raw_acc,
                        "scale_external_accuracy": scale_acc,
                        "scale_gain_delta": scale_acc - raw_acc,
                        "scale_has_gain": bool(scale_acc > raw_acc),
                    })
    return pd.DataFrame(rows)


def scale_gain_stability(stab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for early_variant in sorted(stab["early_variant"].unique()):
        for model_name in sorted(stab["model_name"].unique()):
            sub = stab[(stab["early_variant"] == early_variant) & (stab["model_name"] == model_name)]
            for raw, scale in config.SCALE_PAIRS:
                pivot = sub[sub["feature_block"].isin([raw, scale])].pivot_table(index="seed", columns="feature_block", values="external_accuracy", aggfunc="first")
                if raw in pivot.columns and scale in pivot.columns:
                    pair = pivot.dropna(subset=[raw, scale])
                    if len(pair):
                        rows.append({
                            "early_variant": early_variant,
                            "model_name": model_name,
                            "raw_feature_block": raw,
                            "scale_feature_block": scale,
                            "raw_external_accuracy_mean": float(pair[raw].mean()),
                            "scale_external_accuracy_mean": float(pair[scale].mean()),
                            "scale_gain_mean_delta": float(pair[scale].mean() - pair[raw].mean()),
                            "scale_gain_rate": float((pair[scale] > pair[raw]).mean()),
                        })
    return pd.DataFrame(rows)
