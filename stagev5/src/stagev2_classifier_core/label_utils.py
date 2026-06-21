"""Label normalization and severity grouping."""
from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pandas as pd


def normalize_binary_label(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        numeric = pd.to_numeric(s, errors="raise")
        if numeric.isna().any():
            raise ValueError("Binary label contains missing values.")
        rounded = numeric.round().astype(int)
        invalid = sorted(set(rounded.unique()) - {0, 1})
        if invalid or not np.allclose(numeric.to_numpy(dtype=float), rounded.to_numpy(dtype=float)):
            raise ValueError(f"Binary label must contain only 0/1 values; found: {sorted(numeric.unique().tolist())}")
        return rounded
    mapping = {
        "ad": 1, "alzheimers": 1, "alzheimer": 1, "dementia": 1, "patient": 1, "case": 1,
        "1": 1, "true": 1, "yes": 1,
        "control": 0, "hc": 0, "healthy": 0, "normal": 0, "non-ad": 0, "nonad": 0,
        "0": 0, "false": 0, "no": 0,
    }
    normalized = s.astype(str).str.strip().str.lower()
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        invalid = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unrecognized binary label values: {invalid}")
    return mapped.astype(int)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _parse_new_label(value: Any) -> list[int] | None:
    """Parse [disease, early, middle, late]."""
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [int(float(x)) for x in value[:4]]
    if pd.isna(value):
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        parsed = ast.literal_eval(txt)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            return [int(float(x)) for x in parsed[:4]]
    except Exception:
        return None
    return None


def add_standard_labels(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Add __sample_id__, __y__, [disease, early, middle, late], and severity_group."""
    df = df.copy()
    id_col = _find_col(df, ["sample_id", "file", "filename", "id", "subject_id"])
    if id_col is None:
        prefix = {"ad": "AD", "control": "CTRL", "test": "TEST"}.get(split, split.upper())
        df["__sample_id__"] = [f"{prefix}_{i + 1:04d}" for i in range(len(df))]
    else:
        df["__sample_id__"] = df[id_col].astype(str)

    y_col = _find_col(df, ["disease_label", "label_disease", "disease", "label", "y"])
    if y_col is None:
        raise ValueError(f"No disease/label column found in {split}. Columns: {list(df.columns)[:40]}")
    df["__y__"] = normalize_binary_label(df[y_col])

    mmse_col = _find_col(df, ["mmse", "MMSE"])
    if mmse_col is not None:
        df["mmse"] = pd.to_numeric(df[mmse_col], errors="coerce")
    elif "mmse" not in df.columns:
        df["mmse"] = np.nan

    parsed_col = _find_col(df, ["new_label"])
    parsed = df[parsed_col].apply(_parse_new_label) if parsed_col else pd.Series([None] * len(df), index=df.index)
    if parsed.notna().all():
        arr = np.array(parsed.tolist(), dtype=int)
        df["label_disease"] = arr[:, 0]
        df["label_early"] = arr[:, 1]
        df["label_middle"] = arr[:, 2]
        df["label_late"] = arr[:, 3]
        df["label_mild"] = df["label_early"]
        df["label_moderate"] = df["label_middle"]
        df["label_severe"] = df["label_late"]
        df["label_normal"] = (df["__y__"] == 0).astype(int)
    else:
        _reconstruct_project_labels(df)

    df["new_label"] = [f"[{int(y)},{int(e)},{int(m)},{int(l)}]" for y, e, m, l in zip(df["__y__"], df["label_early"], df["label_middle"], df["label_late"])]
    df["severity_group"] = [severity_group(y, m) for y, m in zip(df["__y__"], df["mmse"])]
    df["__split__"] = split
    return df


def _reconstruct_project_labels(df: pd.DataFrame) -> None:
    y = df["__y__"].astype(int)
    mmse = pd.to_numeric(df.get("mmse", np.nan), errors="coerce")
    df["label_disease"] = y
    df["label_early"] = ((y == 1) & (mmse >= 21) & (mmse <= 24)).astype(int)
    df["label_middle"] = ((y == 1) & (mmse >= 13) & (mmse <= 20)).astype(int)
    df["label_late"] = ((y == 1) & (mmse <= 12)).astype(int)
    df["label_normal"] = (y == 0).astype(int)
    df["label_mild"] = df["label_early"]
    df["label_moderate"] = df["label_middle"]
    df["label_severe"] = df["label_late"]


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
