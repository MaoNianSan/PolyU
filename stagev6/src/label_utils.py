from __future__ import annotations
import ast
import math
from typing import Any
import numpy as np
import pandas as pd


def _parse_new_label(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [int(float(x)) for x in value[:4]]
    if pd.isna(value):
        return None
    try:
        x = ast.literal_eval(str(value).strip())
        if isinstance(x, (list, tuple)) and len(x) >= 4:
            return [int(float(v)) for v in x[:4]]
    except Exception:
        return None
    return None


def normalize_binary_label(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="raise")
    rounded = vals.round().astype(int)
    if vals.isna().any() or not set(rounded.unique()).issubset({0, 1}) or not np.allclose(vals, rounded):
        raise ValueError("Disease labels must be binary 0/1.")
    return rounded


def severity_group(y: int, mmse: Any) -> str:
    try:
        m = float(mmse)
    except Exception:
        m = float("nan")
    if int(y) == 0:
        return "control"
    if math.isnan(m):
        return "AD_unknown_MMSE"
    if m >= 25:
        return "AD_high_MMSE"
    if 21 <= m <= 24:
        return "early"
    if 13 <= m <= 20:
        return "middle"
    if m <= 12:
        return "late"
    return "AD_unknown_MMSE"


def add_standard_labels(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    if "sample_id" not in df.columns:
        raise ValueError(f"Expected sample_id in stagev5 feature CSV for {split}.")
    label_col = "disease_label" if "disease_label" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError(f"No disease label found for {split}.")
    df["__sample_id__"] = df["sample_id"].astype(str)
    df["__y__"] = normalize_binary_label(df[label_col])
    df["mmse"] = pd.to_numeric(df.get("mmse", np.nan), errors="coerce")
    parsed = df["new_label"].apply(_parse_new_label) if "new_label" in df.columns else pd.Series([None] * len(df), index=df.index)
    if parsed.notna().all():
        arr = np.asarray(parsed.tolist(), dtype=int)
        df["label_early"] = arr[:, 1]
        df["label_middle"] = arr[:, 2]
        df["label_late"] = arr[:, 3]
    else:
        y = df["__y__"].astype(int)
        m = df["mmse"]
        df["label_early"] = ((y == 1) & (m >= 21) & (m <= 24)).astype(int)
        df["label_middle"] = ((y == 1) & (m >= 13) & (m <= 20)).astype(int)
        df["label_late"] = ((y == 1) & (m <= 12)).astype(int)
    if ((df[["label_early", "label_middle", "label_late"]].sum(axis=1) > 1)).any():
        raise ValueError("Stage labels must be mutually exclusive.")
    df["label_normal"] = (df["__y__"] == 0).astype(int)
    df["severity_group"] = [severity_group(y, m) for y, m in zip(df["__y__"], df["mmse"])]
    df["__split__"] = split
    return df
