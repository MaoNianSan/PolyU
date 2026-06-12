"""Stage-feature loading for stage-corrected classifiers."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config
from io_utils import read_csv_robust, save_json
from label_utils import add_standard_labels


def _match_split_file(directory: Path, split: str, stage_keywords: list[str]) -> Path | None:
    if not directory.exists():
        return None
    split = split.lower()
    files = list(directory.rglob("*.csv"))
    scored: list[tuple[int, Path]] = []
    for p in files:
        stem = p.stem.lower()
        score = 0
        if split == "ad":
            # Avoid matching 'control' and prefer stems starting with ad or containing _ad.
            if re.search(r"(^|[_\-])ad($|[_\-])", stem) or stem.startswith("ad"):
                score += 10
        elif split == "control":
            if "control" in stem or re.search(r"(^|[_\-])hc($|[_\-])", stem):
                score += 10
        elif split == "test":
            if "test" in stem or "external" in stem:
                score += 10
        for kw in stage_keywords:
            if kw.lower() in stem:
                score += 3
        if score > 0:
            scored.append((score, p))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return scored[0][1]


def _is_meta_col(col: str) -> bool:
    low = col.lower()
    if low.startswith("__"):
        return True
    if low in [x.lower() for x in config.ID_COLUMNS + config.TEXT_COLUMNS + config.LABEL_COLUMNS]:
        return True
    if low in {"severity_group"}:
        return True
    return False


def _candidate_features(df: pd.DataFrame, stage: str) -> list[str]:
    prefixes = {
        "early": config.EARLY_PREFIXES,
        "middle": config.MIDDLE_PREFIXES,
        "late": config.LATE_PREFIXES,
    }[stage]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out = []
    for c in numeric:
        low = c.lower()
        if _is_meta_col(c):
            continue
        if any(k in low for k in config.NON_FEATURE_KEYWORDS):
            continue
        if any(low.startswith(p.lower()) for p in prefixes):
            out.append(c)
        elif stage == "middle" and ("embedding" in low or low.startswith("dim_") or re.fullmatch(r"\d+", low)):
            out.append(c)
        elif stage == "late" and any(k in low for k in ["sentence", "fragmentation", "repetition", "repair_restart", "filler_vague", "grammatical", "connected_speech", "overall_expressive", "fluency", "completeness", "lexical", "discourse", "expressive", "grammar", "syntax"]):
            out.append(c)
    return out


def _ensure_sample_id(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    lower = {c.lower(): c for c in df.columns}
    for c in config.ID_COLUMNS:
        if c.lower() in lower:
            df["__sample_id__"] = df[lower[c.lower()]].astype(str)
            return df
    df["__sample_id__"] = [f"{split.upper()}_{i:04d}" for i in range(len(df))]
    return df


def _aggregate_stage_df(df: pd.DataFrame, split: str, stage: str) -> Tuple[pd.DataFrame, list[str]]:
    """Return one row per sample_id and stage-prefixed feature names."""
    df = _ensure_sample_id(df, split)
    feature_cols = _candidate_features(df, stage)
    if not feature_cols:
        raise ValueError(f"No {stage} features detected for split={split}. Columns: {list(df.columns)[:80]}")

    # Aggregate numeric features by mean if the file is window-level/sentence-level.
    agg = df.groupby("__sample_id__", sort=False)[feature_cols].mean(numeric_only=True).reset_index()

    rename = {}
    for c in feature_cols:
        low = c.lower()
        if low.startswith(f"{stage}_"):
            new = c
        elif stage == "early" and low.startswith("bm25_"):
            new = f"early_{c}"
        elif stage == "middle" and low.startswith("embedding_dim_"):
            new = f"middle_{c}"
        elif stage == "late" and low.startswith("llm_"):
            new = f"late_{c}"
        else:
            new = f"{stage}_{c}"
        rename[c] = new
    agg = agg.rename(columns=rename)
    return agg, list(rename.values())


def _load_stage_split(stage: str, split: str) -> tuple[pd.DataFrame, Path]:
    if stage == "early":
        directory = config.BM25_DIR
        keywords = ["bm25", "early"]
        exact = {
            "ad": ["ad_BM25.csv", "ad_bm25.csv", "ad.csv"],
            "control": ["control_BM25.csv", "control_bm25.csv", "control.csv"],
            "test": ["test_BM25.csv", "test_bm25.csv", "test.csv"],
        }[split]
    elif stage == "middle":
        directory = config.EMBEDDING_DIR
        keywords = ["embedding", "middle", "bge"]
        exact = {
            "ad": ["ad_embedding.csv", "ad_middle.csv", "ad_bge.csv"],
            "control": ["control_embedding.csv", "control_middle.csv", "control_bge.csv"],
            "test": ["test_embedding.csv", "test_middle.csv", "test_bge.csv"],
        }[split]
    else:
        directory = config.LLM_DIR
        keywords = ["llm", "late", "expressive"]
        exact = {
            "ad": ["ad_llm.csv", "ad_LLM.csv", "ad_late.csv", "ad_expressive.csv"],
            "control": ["control_llm.csv", "control_LLM.csv", "control_late.csv", "control_expressive.csv"],
            "test": ["test_llm.csv", "test_LLM.csv", "test_late.csv", "test_expressive.csv"],
        }[split]

    for name in exact:
        p = directory / name
        if p.exists():
            return read_csv_robust(p), p
    p = _match_split_file(directory, split, keywords)
    if p is None:
        raise FileNotFoundError(
            f"Could not find {stage} {split} CSV under {directory}. "
            f"Expected names like {exact}."
        )
    return read_csv_robust(p), p


def _merge_split(split: str) -> tuple[pd.DataFrame, dict]:
    sources: dict[str, str] = {}
    early_raw, p_e = _load_stage_split("early", split)
    middle_raw, p_m = _load_stage_split("middle", split)
    late_raw, p_l = _load_stage_split("late", split)
    sources[f"{split}_early_file"] = str(p_e)
    sources[f"{split}_middle_file"] = str(p_m)
    sources[f"{split}_late_file"] = str(p_l)

    # Labels are taken from early/BM25 files because they are sample-level and already preprocessed.
    base = add_standard_labels(early_raw, split)
    base_cols = ["__sample_id__", "__y__", "mmse", "label_normal", "label_mild", "label_moderate", "label_severe", "severity_group", "__split__"]
    base = base[base_cols].drop_duplicates("__sample_id__")

    early_df, early_cols = _aggregate_stage_df(early_raw, split, "early")
    middle_df, middle_cols = _aggregate_stage_df(middle_raw, split, "middle")
    late_df, late_cols = _aggregate_stage_df(late_raw, split, "late")

    base_ids = set(base["__sample_id__"])
    for stage, stage_df in [("early", early_df), ("middle", middle_df), ("late", late_df)]:
        stage_ids = set(stage_df["__sample_id__"])
        missing = sorted(base_ids - stage_ids)
        unexpected = sorted(stage_ids - base_ids)
        if missing or unexpected:
            raise ValueError(
                f"{stage} sample_id mismatch for split={split}: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )

    df = base.merge(early_df, on="__sample_id__", how="left")
    df = df.merge(middle_df, on="__sample_id__", how="left")
    df = df.merge(late_df, on="__sample_id__", how="left")

    sources[f"{split}_n_rows"] = str(len(df))
    sources[f"{split}_n_early_features"] = str(len(early_cols))
    sources[f"{split}_n_middle_features"] = str(len(middle_cols))
    sources[f"{split}_n_late_features"] = str(len(late_cols))
    return df, sources


def load_train_test(dirs: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load ad/control/test stage features and return train/test frames."""
    source_info = {
        "ad_project_root": str(config.AD_PROJECT_ROOT),
        "bm25_dir": str(config.BM25_DIR),
        "embedding_dir": str(config.EMBEDDING_DIR),
        "llm_dir": str(config.LLM_DIR),
    }
    try:
        ad_df, ad_info = _merge_split("ad")
        control_df, control_info = _merge_split("control")
        test_df, test_info = _merge_split("test")
        source_info.update(ad_info)
        source_info.update(control_info)
        source_info.update(test_info)
    except Exception as exc:
        source_info["load_error"] = repr(exc)
        if dirs is not None:
            save_json(source_info, dirs["reports"] / "input_data_check.json")
        raise

    train_df = pd.concat([ad_df, control_df], ignore_index=True)
    if dirs is not None:
        source_info.update({
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_train_ad": int((train_df["__y__"] == 1).sum()),
            "n_train_control": int((train_df["__y__"] == 0).sum()),
            "n_test_ad": int((test_df["__y__"] == 1).sum()),
            "n_test_control": int((test_df["__y__"] == 0).sum()),
        })
        save_json(source_info, dirs["reports"] / "input_data_check.json")
    return train_df, test_df, source_info
