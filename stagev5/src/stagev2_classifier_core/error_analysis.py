"""Error analysis tables for stage-corrected classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def severity_accuracy_by_model(pred: pd.DataFrame) -> pd.DataFrame:
    return (
        pred.groupby(["model_name", "severity_group"], dropna=False)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"), mean_p_ad=("p_ad", "mean"), false_positive=("error_type", lambda x: int((x == "FP_normal").sum())), false_negative=("error_type", lambda x: int(x.astype(str).str.startswith("FN_").sum())))
        .reset_index()
    )


def error_type_by_model(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.groupby(["model_name", "error_type"], dropna=False).size().reset_index(name="n")


def hard_cases(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_models = pred["model_name"].nunique()
    threshold = max(1, int(np.ceil(config.HARD_CASE_WRONG_FRACTION * n_models)))
    prof = (
        pred.groupby(["sample_id", "y_true", "severity_group", "mmse"], dropna=False)
        .agg(
            n_models=("model_name", "nunique"),
            wrong_count=("correct", lambda x: int((x == 0).sum())),
            mean_p_ad=("p_ad", "mean"),
            sd_p_ad=("p_ad", "std"),
            positive_votes=("y_pred", "sum"),
        )
        .reset_index()
    )
    prof["wrong_fraction"] = prof["wrong_count"] / prof["n_models"].replace(0, np.nan)
    hard = prof[prof["wrong_count"] >= threshold].sort_values(["wrong_count", "sd_p_ad"], ascending=[False, False])
    hard_fp = hard[hard["y_true"] == 0].copy()
    hard_fn = hard[hard["y_true"] == 1].copy()
    disagreement = prof.sort_values(["sd_p_ad", "wrong_count"], ascending=[False, False]).copy()
    return hard_fp, hard_fn, disagreement


def stage_conflict_cases(pred: pd.DataFrame) -> pd.DataFrame:
    cols = ["s_E", "s_M", "s_L"]
    if not all(c in pred.columns for c in cols):
        return pd.DataFrame()
    sub = pred.dropna(subset=cols).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["middle_high_late_low"] = ((sub["s_M"] >= 0.5) & (sub["s_L"] < 0.5)).astype(int)
    sub["middle_low_late_high"] = ((sub["s_M"] < 0.5) & (sub["s_L"] >= 0.5)).astype(int)
    sub["early_middle_disagree"] = (((sub["s_E"] >= 0.5) & (sub["s_M"] < 0.5)) | ((sub["s_E"] < 0.5) & (sub["s_M"] >= 0.5))).astype(int)
    return sub[[
        "sample_id", "model_name", "y_true", "severity_group", "mmse", "y_pred", "p_ad", "correct",
        "s_E", "s_M", "s_L", "s_E_M", "s_M_L", "s_E_M_L",
        "middle_high_late_low", "middle_low_late_high", "early_middle_disagree", "error_type",
    ]]


def write_error_tables(pred: pd.DataFrame, dirs: dict) -> dict:
    sev = severity_accuracy_by_model(pred)
    err = error_type_by_model(pred)
    hard_fp, hard_fn, disagreement = hard_cases(pred)
    conflict = stage_conflict_cases(pred)
    sev.to_csv(dirs["tables"] / "severity_accuracy_by_model.csv", index=False)
    err.to_csv(dirs["tables"] / "error_type_by_model.csv", index=False)
    hard_fp.to_csv(dirs["tables"] / "hard_false_positive_cases.csv", index=False)
    hard_fn.to_csv(dirs["tables"] / "hard_false_negative_cases.csv", index=False)
    disagreement.to_csv(dirs["tables"] / "model_disagreement_cases.csv", index=False)
    conflict.to_csv(dirs["tables"] / "stage_conflict_cases.csv", index=False)
    return {
        "severity_accuracy": sev,
        "error_type": err,
        "hard_false_positive": hard_fp,
        "hard_false_negative": hard_fn,
        "model_disagreement": disagreement,
        "stage_conflict": conflict,
    }
