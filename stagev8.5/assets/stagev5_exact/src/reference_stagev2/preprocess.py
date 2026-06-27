from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INPUT_FILES = {
    "ad": "ad_s2t_wav2vec.csv",
    "control": "control_s2t_wav2vec.csv",
    "test": "test_s2t_wav2vec.csv",
}

SPLIT_PREFIX = {"ad": "AD", "control": "CTRL", "test": "TEST"}


def severity_flags(mmse: Any, disease_label: int) -> dict:
    """Return the project-standard labels.

    Main label order is [disease, early, middle, late]. Legacy severity aliases are
    retained for downstream compatibility:
      label_mild == label_early
      label_moderate == label_middle
      label_severe == label_late
    """
    try:
        m = float(mmse)
    except Exception:
        m = np.nan

    disease = int(disease_label)
    early = int(disease == 1 and 21 <= m <= 24)
    middle = int(disease == 1 and 13 <= m <= 20)
    late = int(disease == 1 and m <= 12)
    normal = int(disease == 0)
    label_valid = int(normal or early or middle or late)

    if disease == 0:
        subgroup = "control_normal" if (not np.isnan(m) and m >= 26) else "control_non_normal_MMSE"
    elif np.isnan(m):
        subgroup = "AD_unknown_MMSE"
    elif m >= 25:
        subgroup = "AD_high_MMSE"
    elif early:
        subgroup = "AD_early"
    elif middle:
        subgroup = "AD_middle"
    elif late:
        subgroup = "AD_late"
    else:
        subgroup = "AD_unknown_MMSE"

    return {
        "label_disease": disease,
        "label_early": early,
        "label_middle": middle,
        "label_late": late,
        "new_label": f"[{disease},{early},{middle},{late}]",
        # Backward-compatible aliases used by stage_core.
        "label_normal": normal,
        "label_mild": early,
        "label_moderate": middle,
        "label_severe": late,
        "label_valid": label_valid,
        "subgroup": subgroup,
    }


def preprocess_one(input_path: Path, source_split: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df.drop(columns=[c for c in ["Unnamed: 0", "file"] if c in df.columns], errors="ignore")
    df = df.rename(columns={"Speech": "text", "label": "disease_label"})
    required = {"text", "disease_label", "mmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} missing required columns: {sorted(missing)}")

    df["text"] = df["text"].fillna("").astype(str)
    df["disease_label"] = pd.to_numeric(df["disease_label"], errors="raise").astype(int)
    invalid_labels = sorted(set(df["disease_label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"{input_path} has non-binary label values: {invalid_labels}")
    df["mmse"] = pd.to_numeric(df["mmse"], errors="coerce")

    prefix = SPLIT_PREFIX[source_split]
    df.insert(0, "sample_id", [f"{prefix}_{i + 1:04d}" for i in range(len(df))])
    df.insert(1, "source_split", source_split)
    flags = df.apply(lambda r: severity_flags(r["mmse"], int(r["disease_label"])), axis=1, result_type="expand")
    return pd.concat([df, flags], axis=1)


def run_preprocess(input_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}
    for split, file_name in INPUT_FILES.items():
        path = input_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Cannot find input file: {path}")
        df = preprocess_one(path, split)
        df.to_csv(output_dir / f"{split}.csv", index=False, encoding="utf-8-sig")
        result[split] = df
    return result
