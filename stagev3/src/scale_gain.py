from __future__ import annotations

import pandas as pd

from . import config


def scale_gain_seed2026(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for early_variant in main["early_variant"].unique():
        for model_variant in sorted(main["model_variant"].unique()):
            for raw, scale in config.SCALE_PAIRS:
                r = main[(main.early_variant == early_variant) & (main.feature_block == raw) & (main.model_variant == model_variant) & (~main.is_special_model)]
                s = main[(main.early_variant == early_variant) & (main.feature_block == scale) & (main.model_variant == model_variant) & (~main.is_special_model)]
                if len(r) == 1 and len(s) == 1:
                    rv, sv = float(r.iloc[0].external_accuracy), float(s.iloc[0].external_accuracy)
                    rows.append({
                        "early_variant": early_variant,
                        "model_variant": model_variant,
                        "raw_feature_block": raw,
                        "scale_feature_block": scale,
                        "raw_external_accuracy": rv,
                        "scale_external_accuracy": sv,
                        "scale_gain_delta": sv - rv,
                        "scale_has_gain": bool(sv > rv),
                    })
    return pd.DataFrame(rows)


def scale_gain_stability(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for early_variant in stability["early_variant"].unique():
        for model_variant in sorted(stability["model_variant"].unique()):
            for raw, scale in config.SCALE_PAIRS:
                r = stability[(stability.early_variant == early_variant) & (stability.feature_block == raw) & (stability.model_variant == model_variant) & (~stability.is_special_model)]
                s = stability[(stability.early_variant == early_variant) & (stability.feature_block == scale) & (stability.model_variant == model_variant) & (~stability.is_special_model)]
                if not r.empty and not s.empty:
                    merged = r[["seed", "external_accuracy"]].merge(s[["seed", "external_accuracy"]], on="seed", suffixes=("_raw", "_scale"))
                    rows.append({
                        "early_variant": early_variant,
                        "model_variant": model_variant,
                        "raw_feature_block": raw,
                        "scale_feature_block": scale,
                        "raw_external_accuracy_mean": float(merged.external_accuracy_raw.mean()),
                        "scale_external_accuracy_mean": float(merged.external_accuracy_scale.mean()),
                        "scale_gain_mean_delta": float(merged.external_accuracy_scale.mean() - merged.external_accuracy_raw.mean()),
                        "scale_gain_rate": float((merged.external_accuracy_scale > merged.external_accuracy_raw).mean()),
                    })
    return pd.DataFrame(rows)


def earlyv1_gain_over_earlyv0(main: pd.DataFrame) -> pd.DataFrame:
    a = main[main.early_variant == "earlyv0"]
    b = main[main.early_variant == "earlyv1"]
    keys = ["model_spec_id", "feature_block", "model_name", "model_variant", "is_special_model"]
    merged = a[keys + ["external_accuracy"]].merge(b[keys + ["external_accuracy"]], on=keys, suffixes=("_earlyv0", "_earlyv1"))
    merged["external_accuracy_delta"] = merged["external_accuracy_earlyv1"] - merged["external_accuracy_earlyv0"]
    merged["earlyv1_has_gain"] = merged["external_accuracy_delta"] > 0
    return merged.rename(columns={"external_accuracy_earlyv0": "earlyv0_external_accuracy", "external_accuracy_earlyv1": "earlyv1_external_accuracy"})
