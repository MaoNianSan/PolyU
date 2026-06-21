from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

from . import config as cfg
from .data_loader import feature_validation_summary


def run_self_check(require_features: bool = True) -> dict[str, Any]:
    required = [
        cfg.ROOT / "run_stagev6.py", cfg.ROOT / "README.md", cfg.ROOT / "requirements.txt",
        cfg.ROOT / "notebooks" / "stagev6_result_check.ipynb",
        cfg.ROOT / "src" / "cascade_train.py", cfg.ROOT / "src" / "data_loader.py",
        cfg.ROOT / "output" / "features" / "E_M_extraction_manifest.json",
        cfg.ROOT / "output" / "features" / "L_extraction_manifest.json",
    ]
    missing = [str(p.relative_to(cfg.ROOT)) for p in required if not p.exists()]
    status: dict[str, Any] = {}
    if require_features and not missing:
        try:
            status = feature_validation_summary()
        except Exception as exc:
            missing.append(f"feature validation: {exc}")
    result = {
        "passed": not missing,
        "missing": missing,
        "python": sys.version,
        "stagev6_protocol": {
            "cv_n_splits": cfg.CV_N_SPLITS,
            "cv_n_repeats": cfg.CV_N_REPEATS,
            "late_gate_scoring": "balanced_accuracy",
            "nonlate_branch_scoring": "accuracy",
            "selection_metric": "external_accuracy",
            "decision_threshold": cfg.DECISION_THRESHOLD,
            "model_count": len(cfg.GATE_IDS) * len(cfg.BRANCH_IDS),
        },
        "feature_status": status,
    }
    out = cfg.OUTPUT / "checks" / "self_check_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    return result
