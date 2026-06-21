"""Read-only validation of existing stagev5 result artifacts.

This script never trains models, extracts E/M/L features, or calls an API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import stagev5_config as cfg
from src.feature_adapter import validate_feature_outputs


REQUIRED_FINAL_FILES = [
    *cfg.CANONICAL_FINAL_FILES,
    "figures/fig01_model_ranking_external_accuracy.png",
    "figures/fig02_feature_block_comparison.png",
    "figures/fig03_selected_model_confusion_matrix.png",
    "figures/fig05_stage_subgroup_accuracy.png",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _top_model_summary() -> dict[str, Any]:
    ranking = _read_csv(cfg.FINAL / "stagev5_model_ranking_by_external_accuracy.csv")
    if ranking.empty:
        return {"available": False}
    top = ranking.iloc[0].to_dict()
    return {
        "available": True,
        "selected_model": top.get("model_name"),
        "feature_block": top.get("feature_block"),
        "external_accuracy": top.get("accuracy"),
        "ranking_rows": int(len(ranking)),
    }


def main() -> int:
    """Check saved feature schemas and final-report availability."""
    missing = [name for name in REQUIRED_FINAL_FILES if not (cfg.FINAL / name).exists()]
    feature_status: dict[str, Any] = {}
    feature_error = None
    try:
        feature_status = validate_feature_outputs()
    except Exception as exc:  # pragma: no cover - CLI diagnostic path
        feature_error = repr(exc)

    report = {
        "passed": not missing and feature_error is None,
        "project_root": str(cfg.ROOT),
        "final_report_dir": str(cfg.FINAL),
        "missing_final_files": missing,
        "feature_error": feature_error,
        "feature_counts": {
            "E_model_features": feature_status.get("n_E"),
            "M_model_features": feature_status.get("n_M"),
            "L_raw_F8_model_features": feature_status.get("late_model_feature_count"),
            "L_auxiliary_diagnostic_features": feature_status.get("late_diagnostic_column_count"),
        },
        "top_model": _top_model_summary(),
        "policy": {
            "trained_models": False,
            "extracted_features": False,
            "called_api": False,
            "modified_existing_results": False,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
