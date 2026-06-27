"""Stagev8.5 feature and MMSE-severity label contracts.

The Stagev6-compatible loader is vendored verbatim in ``src/data_loader.py``.
This module adds no feature parser or transformer. It only audits the loaded
matrices and derives Stagev8.5 labels from the already loaded MMSE column.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loader import LoadedData, load_stagev5_features
from .source_integrity import verify_strict_reference_sources


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_frame(frame: pd.DataFrame, cols: list[str]) -> str:
    subset = frame[["sample_id", *cols]].copy()
    subset["sample_id"] = subset["sample_id"].astype(str)
    subset = subset.sort_values("sample_id", kind="mergesort")
    data = subset.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return sha256_text(data)


def feature_paths() -> dict[str, Path]:
    return {
        "ad_E": cfg.EARLY_DIR / "ad_BM25.csv",
        "control_E": cfg.EARLY_DIR / "control_BM25.csv",
        "test_E": cfg.EARLY_DIR / "test_BM25.csv",
        "ad_M": cfg.MIDDLE_DIR / "ad_embedding.csv",
        "control_M": cfg.MIDDLE_DIR / "control_embedding.csv",
        "test_M": cfg.MIDDLE_DIR / "test_embedding.csv",
        "ad_L": cfg.LATE_DIR / "ad_LLM.csv",
        "control_L": cfg.LATE_DIR / "control_LLM.csv",
        "test_L": cfg.LATE_DIR / "test_LLM.csv",
    }


def source_contract(root: Path) -> dict[str, Any]:
    runtime_loader = root / "src" / "data_loader.py"
    reference_loader = root / "assets" / "stagev6_contract_reference" / "data_loader.py"
    return {
        "policy": "exact_stagev6_loader_contract",
        "runtime_loader": str(runtime_loader),
        "runtime_loader_sha256": sha256_file(runtime_loader),
        "reference_loader": str(reference_loader),
        "reference_loader_sha256": sha256_file(reference_loader),
        "loader_byte_identical": sha256_file(runtime_loader) == sha256_file(reference_loader),
        "feature_mode": "fresh_stagev5_source_reconstruction_then_verbatim_stagev6_loader",
        "train_may_extract_features": False,
        "train_may_call_api": False,
        "middle_aggregation": "exact Stagev6 loader: groupby(sample_id, sort=False).mean()",
        "expected_feature_counts": cfg.EXPECTED_FEATURE_COUNTS,
        "raw_f8_base_names": cfg.RAW_F8_BASE_NAMES,
    }


def mmse_stratum(y: int, mmse: float) -> str:
    if int(y) == 0:
        return "control"
    if not np.isfinite(mmse):
        return "AD_unknown_MMSE"
    if mmse >= cfg.MMSE_HIGH_MIN:
        return "high_mmse_AD"
    if cfg.MMSE_INTERMEDIATE_MIN <= mmse <= cfg.MMSE_INTERMEDIATE_MAX:
        return "intermediate_mmse_AD"
    if mmse <= cfg.MMSE_LOW_MAX:
        return "low_mmse_AD"
    return "AD_unassigned_MMSE"


def add_mmse_severity_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["true_mmse_stratum"] = [
        mmse_stratum(int(y), float(m) if pd.notna(m) else np.nan)
        for y, m in zip(out["__y__"], out["mmse"])
    ]
    ad = out["__y__"].eq(1)
    out["target_T20_mmse_le_20"] = (ad & out["mmse"].le(cfg.MMSE_INTERMEDIATE_MAX)).astype(int)
    out["target_T14_mmse_le_14"] = (ad & out["mmse"].le(cfg.MMSE_LOW_MAX)).astype(int)
    return out


def _assert_mmse_contract(frame: pd.DataFrame, split: str) -> None:
    ad = frame[frame["__y__"].eq(1)].copy()
    if ad["mmse"].isna().any():
        raise RuntimeError(f"{split}: AD MMSE has missing values; Stagev8.5 label contract cannot be formed.")
    groups = set(ad["true_mmse_stratum"].astype(str))
    expected = set(cfg.SEVERITY_STRATA)
    if groups != expected:
        raise RuntimeError(f"{split}: MMSE strata mismatch; expected={expected}, observed={groups}")
    if not ad.loc[ad["target_T20_mmse_le_20"].eq(0), "mmse"].ge(cfg.MMSE_HIGH_MIN).all():
        raise RuntimeError(f"{split}: T20 label violates the high-MMSE boundary.")
    conditional = ad[ad["target_T20_mmse_le_20"].eq(1)]
    if not conditional.loc[conditional["target_T14_mmse_le_14"].eq(0), "mmse"].between(cfg.MMSE_INTERMEDIATE_MIN, cfg.MMSE_INTERMEDIATE_MAX).all():
        raise RuntimeError(f"{split}: T14 negative label violates intermediate-MMSE boundary.")


def build_feature_and_label_audits(root: Path, data: LoadedData | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], LoadedData]:
    strict_source = verify_strict_reference_sources()
    data = data or load_stagev5_features()
    train = add_mmse_severity_labels(data.train)
    external = add_mmse_severity_labels(data.external)
    data = LoadedData(
        train=train,
        external=external,
        early_columns=data.early_columns,
        middle_columns=data.middle_columns,
        late_columns=data.late_columns,
        source_paths=data.source_paths,
    )
    _assert_mmse_contract(train, "train")
    _assert_mmse_contract(external, "external")

    files = feature_paths()
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stagev6-contract feature file(s): {missing}")

    lock = {
        "feature_policy": "exact_stagev6_loader_contract",
        "feature_mode": "fresh_stagev5_source_reconstruction_then_verbatim_stagev6_loader",
        "strict_source_integrity": strict_source,
        "source_file_sha256": {k: sha256_file(v) for k, v in files.items()},
        "source_paths": {k: str(v.resolve()) for k, v in files.items()},
        "E_dimension": len(data.early_columns),
        "M_dimension": len(data.middle_columns),
        "L_dimension": len(data.late_columns),
        "E_columns_sha256": sha256_text("\n".join(data.early_columns)),
        "M_columns_sha256": sha256_text("\n".join(data.middle_columns)),
        "L_columns_sha256": sha256_text("\n".join(data.late_columns)),
        "train_sample_id_sha256": sha256_text("\n".join(train.sample_id.astype(str))),
        "external_sample_id_sha256": sha256_text("\n".join(external.sample_id.astype(str))),
        "train_E_matrix_sha256": sha256_frame(train, data.early_columns),
        "train_M_matrix_sha256": sha256_frame(train, data.middle_columns),
        "train_L_matrix_sha256": sha256_frame(train, data.late_columns),
        "external_E_matrix_sha256": sha256_frame(external, data.early_columns),
        "external_M_matrix_sha256": sha256_frame(external, data.middle_columns),
        "external_L_matrix_sha256": sha256_frame(external, data.late_columns),
        "middle_aggregation": "groupby(sample_id, sort=False).mean() in verbatim Stagev6 loader",
        "secondary_feature_processing": "prohibited; only model-fold median imputation and standard scaling in sklearn pipelines",
        "api_or_feature_extraction_called": False,
    }
    parity = {
        "parity_status": "pass",
        "runtime_loader_equals_stagev6_reference": source_contract(root)["loader_byte_identical"],
        "counts": {"E": len(data.early_columns), "M": len(data.middle_columns), "L": len(data.late_columns)},
        "expected_counts": cfg.EXPECTED_FEATURE_COUNTS,
        "sample_counts": {"train": int(len(train)), "external": int(len(external))},
        "train_external_schema_equal": True,
        "loaded_matrix_hashes": {k: v for k, v in lock.items() if k.endswith("matrix_sha256")},
        "raw_feature_file_hashes": lock["source_file_sha256"],
        "loader_behavior": "verbatim Stagev6 data_loader.py; Stagev8.5 adds no CSV parser or feature transformer",
    }
    if not parity["runtime_loader_equals_stagev6_reference"]:
        parity["parity_status"] = "fail"
        raise RuntimeError("Runtime loader differs from vendored Stagev6 reference.")
    if parity["counts"] != cfg.EXPECTED_FEATURE_COUNTS:
        parity["parity_status"] = "fail"
        raise RuntimeError("Feature dimensions differ from strict Stagev6 contract.")

    label_contract = {
        "status": "pass",
        "label_type": "MMSE-informed ordinal cognitive-severity strata; not clinical early/middle/late staging",
        "thresholds": {
            "high_mmse_AD": f"MMSE >= {cfg.MMSE_HIGH_MIN}",
            "intermediate_mmse_AD": f"MMSE {cfg.MMSE_INTERMEDIATE_MIN}-{cfg.MMSE_INTERMEDIATE_MAX}",
            "low_mmse_AD": f"MMSE <= {cfg.MMSE_LOW_MAX}",
            "T20": f"MMSE <= {cfg.MMSE_INTERMEDIATE_MAX} vs >= {cfg.MMSE_HIGH_MIN}",
            "T14": f"MMSE <= {cfg.MMSE_LOW_MAX} vs {cfg.MMSE_INTERMEDIATE_MIN}-{cfg.MMSE_INTERMEDIATE_MAX}, conditional on MMSE <= {cfg.MMSE_INTERMEDIATE_MAX}",
        },
        "train_mmse_stratum_counts": train["true_mmse_stratum"].value_counts().reindex(cfg.SEVERITY_STRATA_WITH_CONTROL, fill_value=0).to_dict(),
        "external_mmse_stratum_counts": external["true_mmse_stratum"].value_counts().reindex(cfg.SEVERITY_STRATA_WITH_CONTROL, fill_value=0).to_dict(),
        "head_T20_train_counts": {
            "high_mmse_negative": int(((train["__y__"] == 1) & (train["target_T20_mmse_le_20"] == 0)).sum()),
            "mmse_le_20_positive": int(((train["__y__"] == 1) & (train["target_T20_mmse_le_20"] == 1)).sum()),
        },
        "head_T14_train_counts": {
            "intermediate_mmse_negative": int(((train["__y__"] == 1) & (train["mmse"].between(cfg.MMSE_INTERMEDIATE_MIN, cfg.MMSE_INTERMEDIATE_MAX))).sum()),
            "low_mmse_positive": int(((train["__y__"] == 1) & (train["target_T14_mmse_le_14"] == 1)).sum()),
        },
        "external_selection_prohibited": True,
        "rationale": "The 21/15-20/<=14 contract is fixed before Stagev8.5 evaluation and is treated as a sensitivity-motivated MMSE severity definition, not a clinical gold-standard stage label.",
    }
    expected_train = {"control": 79, "high_mmse_AD": 21, "intermediate_mmse_AD": 41, "low_mmse_AD": 25}
    expected_external = {"control": 36, "high_mmse_AD": 14, "intermediate_mmse_AD": 13, "low_mmse_AD": 8}
    if label_contract["train_mmse_stratum_counts"] != expected_train or label_contract["external_mmse_stratum_counts"] != expected_external:
        label_contract["status"] = "fail"
        raise RuntimeError(f"Unexpected Stagev8.5 MMSE stratum counts: {label_contract}")
    return lock, parity, label_contract, data


def optional_stagev6_runtime_comparison(root: Path, stagev6_root: Path | None, data: LoadedData) -> dict[str, Any]:
    result: dict[str, Any] = {"requested": stagev6_root is not None, "status": "not_requested"}
    if stagev6_root is None:
        return result
    stagev6_root = stagev6_root.resolve()
    loader_file = stagev6_root / "src" / "data_loader.py"
    config_file = stagev6_root / "src" / "config.py"
    if not loader_file.exists() or not config_file.exists():
        raise FileNotFoundError(f"Invalid --stagev6-root; expected {loader_file} and {config_file}")
    import sys
    import types
    package = types.ModuleType("_stagev6_4_compare")
    package.__path__ = [str(stagev6_root / "src")]
    sys.modules["_stagev6_4_compare"] = package
    for name, path in [("_stagev6_4_compare.config", config_file), ("_stagev6_4_compare.data_loader", loader_file)]:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    reference = sys.modules["_stagev6_4_compare.data_loader"].load_stagev5_features()
    runtime_train = data.train.drop(columns=[c for c in data.train.columns if c.startswith("target_T") or c == "true_mmse_stratum"], errors="ignore")
    runtime_external = data.external.drop(columns=[c for c in data.external.columns if c.startswith("target_T") or c == "true_mmse_stratum"], errors="ignore")
    checks = {
        "runtime_loader_sha256": sha256_file(root / "src" / "data_loader.py"),
        "provided_stagev6_loader_sha256": sha256_file(loader_file),
        "loader_sha_equal": sha256_file(root / "src" / "data_loader.py") == sha256_file(loader_file),
        "train_ids_equal": runtime_train.sample_id.tolist() == reference.train.sample_id.tolist(),
        "external_ids_equal": runtime_external.sample_id.tolist() == reference.external.sample_id.tolist(),
        "E_columns_equal": data.early_columns == reference.early_columns,
        "M_columns_equal": data.middle_columns == reference.middle_columns,
        "L_columns_equal": data.late_columns == reference.late_columns,
        "train_E_hash_equal": sha256_frame(runtime_train, data.early_columns) == sha256_frame(reference.train, reference.early_columns),
        "train_M_hash_equal": sha256_frame(runtime_train, data.middle_columns) == sha256_frame(reference.train, reference.middle_columns),
        "train_L_hash_equal": sha256_frame(runtime_train, data.late_columns) == sha256_frame(reference.train, reference.late_columns),
        "external_E_hash_equal": sha256_frame(runtime_external, data.early_columns) == sha256_frame(reference.external, reference.early_columns),
        "external_M_hash_equal": sha256_frame(runtime_external, data.middle_columns) == sha256_frame(reference.external, reference.middle_columns),
        "external_L_hash_equal": sha256_frame(runtime_external, data.late_columns) == sha256_frame(reference.external, reference.late_columns),
    }
    result.update(checks)
    result["status"] = "pass" if all(v for k, v in checks.items() if k.endswith("_equal")) else "fail"
    if result["status"] != "pass":
        raise RuntimeError(f"Direct Stagev6 parity comparison failed: {result}")
    return result
