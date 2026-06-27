from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")


def add_stage_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stage = []
    for _, row in out.iterrows():
        if int(row["label"]) != 1 or pd.isna(row.get("mmse")):
            stage.append(pd.NA)
            continue
        mmse = float(row["mmse"])
        stage.append("late" if mmse <= 12 else "middle" if mmse <= 20 else "early")
    out["stage_label"] = stage
    out["stage_rank"] = out["stage_label"].map({"early": 1, "middle": 2, "late": 3}).astype("float")
    return out


def load_frozen_inputs() -> dict[str, pd.DataFrame]:
    for name in config.FROZEN_FILES.values():
        _require(config.FROZEN_DIR / name)
    train_meta = add_stage_labels(pd.read_csv(config.FROZEN_DIR / config.FROZEN_FILES["train_meta"]))
    external_meta = add_stage_labels(pd.read_csv(config.FROZEN_DIR / config.FROZEN_FILES["external_meta"]))
    train_early = pd.read_csv(config.FROZEN_DIR / config.FROZEN_FILES["train_early"])
    external_early = pd.read_csv(config.FROZEN_DIR / config.FROZEN_FILES["external_early"])
    train_middle = pd.read_csv(config.FROZEN_DIR / config.FROZEN_FILES["train_middle"])
    external_middle = pd.read_csv(config.FROZEN_DIR / config.FROZEN_FILES["external_middle"])
    for name, meta, early, middle in [
        ("train", train_meta, train_early, train_middle),
        ("external", external_meta, external_early, external_middle),
    ]:
        ids = meta["sample_id"].astype(str).tolist()
        if meta["sample_id"].duplicated().any() or early["sample_id"].duplicated().any() or middle["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id in frozen {name} inputs.")
        if set(ids) != set(early["sample_id"].astype(str)) or set(ids) != set(middle["sample_id"].astype(str)):
            raise ValueError(f"Frozen {name} metadata / feature sample_id mismatch.")
        if meta["text"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"Frozen {name} metadata contains empty text.")
    return {
        "train_meta": train_meta, "external_meta": external_meta,
        "train_early": train_early, "external_early": external_early,
        "train_middle": train_middle, "external_middle": external_middle,
    }


def _merge_one(meta: pd.DataFrame, early: pd.DataFrame, middle: pd.DataFrame) -> pd.DataFrame:
    base = meta[["sample_id"]].copy()
    early_cols = [c for c in early.columns if c.startswith("early_")]
    middle_cols = [c for c in middle.columns if c.startswith("middle_embedding_")]
    if not early_cols or not middle_cols:
        raise ValueError("Frozen early/middle feature columns could not be identified.")
    merged = base.merge(early[["sample_id", *early_cols]], on="sample_id", how="left", validate="one_to_one")
    merged = merged.merge(middle[["sample_id", *middle_cols]], on="sample_id", how="left", validate="one_to_one")
    X = merged.drop(columns=["sample_id"]).apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()[:10]
        raise ValueError(f"Frozen early/middle matrix contains missing values; examples={bad}")
    return X


def build_em_matrices(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        _merge_one(data["train_meta"], data["train_early"], data["train_middle"]),
        _merge_one(data["external_meta"], data["external_early"], data["external_middle"]),
    )


def data_manifest() -> dict:
    raw = {name: sha256_file(config.RAW_DIR / fn) for name, fn in config.RAW_FILES.items() if (config.RAW_DIR / fn).exists()}
    frozen = {name: sha256_file(config.FROZEN_DIR / fn) for name, fn in config.FROZEN_FILES.items()}
    return {"raw_checksums": raw, "frozen_feature_checksums": frozen}


def stage_summary(meta: pd.DataFrame) -> dict:
    sub = meta.loc[meta["label"].astype(int).eq(1)].copy()
    return {str(k): int(v) for k, v in sub["stage_label"].value_counts(dropna=False).to_dict().items()}
