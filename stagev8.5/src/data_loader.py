"""Strict stagev5-feature loader for stagev6.

Only the existing stagev5 E/M/L feature CSVs are consumed.  The middle BGE-M3
window-level CSVs are aggregated by sample_id using the same mean aggregation
used by the stagev5 classifier core before any stagev6 model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as cfg


@dataclass
class LoadedData:
    train: pd.DataFrame
    external: pd.DataFrame
    early_columns: list[str]
    middle_columns: list[str]
    late_columns: list[str]
    source_paths: dict[str, str]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def resolve_late_raw_f8_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for base in cfg.RAW_F8_BASE_NAMES:
        matches = [c for c in (f"late_{base}", base) if c in df.columns]
        if len(matches) != 1:
            raise ValueError(
                f"Late raw F8 resolution failed for {base!r}; matches={matches}; "
                f"available={[c for c in df.columns if c.startswith('late_') or c == base]}"
            )
        cols.append(matches[0])
    return cols


def _feature_paths(split: str) -> dict[str, Path]:
    names = {
        "ad": {"E": "ad_BM25.csv", "M": "ad_embedding.csv", "L": "ad_LLM.csv"},
        "control": {"E": "control_BM25.csv", "M": "control_embedding.csv", "L": "control_LLM.csv"},
        "test": {"E": "test_BM25.csv", "M": "test_embedding.csv", "L": "test_LLM.csv"},
    }[split]
    return {
        "E": cfg.EARLY_DIR / names["E"],
        "M": cfg.MIDDLE_DIR / names["M"],
        "L": cfg.LATE_DIR / names["L"],
    }


def _require_unique_ids(df: pd.DataFrame, name: str) -> None:
    if "sample_id" not in df.columns:
        raise ValueError(f"{name} has no sample_id column.")
    if df["sample_id"].astype(str).duplicated().any():
        sample = df.loc[df["sample_id"].astype(str).duplicated(), "sample_id"].astype(str).head(5).tolist()
        raise ValueError(f"{name} has duplicate sample IDs: {sample}")


def _normalize_binary(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="raise")
    if x.isna().any() or not set(x.astype(int).unique()).issubset({0, 1}):
        raise ValueError("Disease labels must be binary 0/1 without missing values.")
    return x.astype(int)


def _severity_group(y: int, mmse: float) -> str:
    if int(y) == 0:
        return "control"
    if not np.isfinite(mmse):
        return "AD_unknown_MMSE"
    if mmse >= 25:
        return "AD_high_MMSE"
    if 21 <= mmse <= 24:
        return "early"
    if 13 <= mmse <= 20:
        return "middle"
    if mmse <= 12:
        return "late"
    return "AD_unknown_MMSE"


def _base_metadata(early: pd.DataFrame, split: str) -> pd.DataFrame:
    _require_unique_ids(early, f"E/{split}")
    disease_col = "disease_label" if "disease_label" in early.columns else "label"
    if disease_col not in early.columns:
        raise ValueError(f"E/{split} lacks disease_label/label.")
    if "mmse" not in early.columns:
        raise ValueError(f"E/{split} lacks mmse.")
    out = pd.DataFrame({
        "sample_id": early["sample_id"].astype(str),
        "__y__": _normalize_binary(early[disease_col]),
        "mmse": pd.to_numeric(early["mmse"], errors="coerce"),
    })
    # Existing stagev5 labels are preferred. Reconstruction is only a safeguard.
    for target, source in [("label_early", "label_early"), ("label_middle", "label_middle"), ("label_late", "label_late")]:
        if source in early.columns:
            out[target] = _normalize_binary(early[source])
        else:
            if target == "label_early":
                out[target] = ((out["__y__"] == 1) & out["mmse"].between(21, 24)).astype(int)
            elif target == "label_middle":
                out[target] = ((out["__y__"] == 1) & out["mmse"].between(13, 20)).astype(int)
            else:
                out[target] = ((out["__y__"] == 1) & (out["mmse"] <= 12)).astype(int)
    out["label_normal"] = (out["__y__"] == 0).astype(int)
    out["label_disease"] = out["__y__"].astype(int)
    out["label_mild"] = out["label_early"].astype(int)
    out["label_moderate"] = out["label_middle"].astype(int)
    out["label_severe"] = out["label_late"].astype(int)
    out["severity_group"] = [
        _severity_group(y, float(m) if pd.notna(m) else np.nan)
        for y, m in zip(out["__y__"], out["mmse"])
    ]
    out["__split__"] = split
    return out


def _aggregate_features(df: pd.DataFrame, cols: Iterable[str], prefix: str, split: str, stage: str) -> tuple[pd.DataFrame, list[str]]:
    if "sample_id" not in df.columns:
        raise ValueError(f"{stage}/{split} has no sample_id column.")
    cols = list(cols)
    if not cols:
        raise ValueError(f"{stage}/{split} has no resolved feature columns.")
    work = df[["sample_id", *cols]].copy()
    work["sample_id"] = work["sample_id"].astype(str)
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    result = work.groupby("sample_id", sort=False)[cols].mean(numeric_only=True).reset_index()
    rename = {c: c if c.startswith(prefix) else f"{prefix}{c}" for c in cols}
    return result.rename(columns=rename), list(rename.values())


def _load_split(split: str) -> tuple[pd.DataFrame, dict[str, str], list[str], list[str], list[str]]:
    paths = _feature_paths(split)
    for stage, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing inherited stagev5 {stage} feature file: {path}")
    e_raw, m_raw, l_raw = (_read_csv(paths[k]) for k in ("E", "M", "L"))
    base = _base_metadata(e_raw, split)
    e_cols = [c for c in e_raw.columns if c.startswith("early_") and pd.api.types.is_numeric_dtype(e_raw[c])]
    m_cols = [c for c in m_raw.columns if c.startswith("embedding_dim_") and pd.api.types.is_numeric_dtype(m_raw[c])]
    l_cols = resolve_late_raw_f8_columns(l_raw)
    e, e_names = _aggregate_features(e_raw, e_cols, "early_", split, "E")
    m, m_names = _aggregate_features(m_raw, m_cols, "middle_", split, "M")
    l, l_names = _aggregate_features(l_raw, l_cols, "late_", split, "L")
    expected_ids = set(base["sample_id"])
    for stage, frame in [("E", e), ("M", m), ("L", l)]:
        observed = set(frame["sample_id"])
        if observed != expected_ids:
            raise ValueError(
                f"{stage}/{split} sample_id mismatch; missing={sorted(expected_ids-observed)[:5]}, "
                f"unexpected={sorted(observed-expected_ids)[:5]}"
            )
    merged = base.merge(e, on="sample_id", how="left", validate="one_to_one")
    merged = merged.merge(m, on="sample_id", how="left", validate="one_to_one")
    merged = merged.merge(l, on="sample_id", how="left", validate="one_to_one")
    return merged, {f"{split}_{stage}": str(path) for stage, path in paths.items()}, e_names, m_names, l_names


def load_stagev5_features() -> LoadedData:
    ad, ad_paths, e_names, m_names, l_names = _load_split("ad")
    control, control_paths, e2, m2, l2 = _load_split("control")
    external, test_paths, e3, m3, l3 = _load_split("test")
    if not (e_names == e2 == e3 and m_names == m2 == m3 and l_names == l2 == l3):
        raise ValueError("Train/external feature schemas are inconsistent.")
    counts = {"E": len(e_names), "M": len(m_names), "L": len(l_names)}
    if counts != cfg.EXPECTED_FEATURE_COUNTS:
        raise ValueError(f"Strict stagev5 feature count validation failed: expected={cfg.EXPECTED_FEATURE_COUNTS}, observed={counts}")
    train = pd.concat([ad, control], ignore_index=True)
    if train["sample_id"].duplicated().any() or external["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs after split merge.")
    if not ((train["label_late"] <= train["__y__"]).all() and (external["label_late"] <= external["__y__"]).all()):
        raise ValueError("Late label must be a subset of AD labels.")
    return LoadedData(
        train=train,
        external=external,
        early_columns=e_names,
        middle_columns=m_names,
        late_columns=l_names,
        source_paths={**ad_paths, **control_paths, **test_paths},
    )


def feature_validation_summary() -> dict:
    data = load_stagev5_features()
    return {
        "n_train": int(len(data.train)),
        "n_external": int(len(data.external)),
        "n_train_ad": int(data.train["__y__"].sum()),
        "n_train_control": int((data.train["__y__"] == 0).sum()),
        "n_train_late": int(data.train["label_late"].sum()),
        "n_train_nonlate_ad": int(((data.train["__y__"] == 1) & (data.train["label_late"] == 0)).sum()),
        "n_external_ad": int(data.external["__y__"].sum()),
        "n_external_control": int((data.external["__y__"] == 0).sum()),
        "n_external_late": int(data.external["label_late"].sum()),
        "n_external_nonlate_ad": int(((data.external["__y__"] == 1) & (data.external["label_late"] == 0)).sum()),
        "n_E": len(data.early_columns),
        "n_M": len(data.middle_columns),
        "n_L": len(data.late_columns),
        "late_raw_f8_columns": data.late_columns,
        "source_paths": data.source_paths,
        "middle_aggregation": "mean by sample_id over existing stagev5 BGE-M3 window rows",
        "api_or_feature_extraction_called": False,
    }
