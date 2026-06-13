from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TEXT_CANDIDATES = ["text", "Speech", "speech", "transcript", "utterance", "content", "speech_text", "sentence"]
LABEL_CANDIDATES = ["label", "disease_label", "diagnosis", "class", "target", "y"]
MMSE_CANDIDATES = ["mmse", "MMSE", "mini_mental_state", "score"]

AD_VALUES = {"ad", "dementia", "patient", "case", "1", "1.0", "true", "yes", "positive", "mild", "moderate", "severe"}
CONTROL_VALUES = {"control", "cn", "normal", "healthy", "0", "0.0", "false", "no", "negative"}


def normalize_text_value(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).replace("\r", " ").replace("\n", " ")).strip()


def find_first_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_label_value(v, source_kind: str | None = None) -> int:
    if pd.isna(v):
        if source_kind == "ad":
            return 1
        if source_kind == "control":
            return 0
        raise ValueError("Missing label and no source-based label can be inferred.")
    s = str(v).strip().lower()
    try:
        f = float(s)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
    except Exception:
        pass
    if s in AD_VALUES:
        return 1
    if s in CONTROL_VALUES:
        return 0
    raise ValueError(f"Unrecognized label value: {v!r}")


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)
