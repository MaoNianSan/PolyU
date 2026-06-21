from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import stagev5_config as cfg


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum without loading the whole file into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_self_check(require_features: bool = False) -> dict[str, Any]:
    """Run structural checks without API calls, training, or feature extraction."""
    required = [
        cfg.ROOT / "run_stagev5.py",
        cfg.ROOT / "README.md",
        cfg.NOTEBOOKS / "stagev5_result_check.ipynb",
        cfg.ROOT / "src/reference_stagev2/stagev1_features.py",
        cfg.ROOT / "src/reference_stagev2/api_feature_extraction.py",
        cfg.ROOT / "src/reference_stagev4/late_extract.py",
        cfg.ROOT / "src/reference_stagev4/late_prompt.py",
        cfg.ROOT / "src/stagev2_classifier_core/run_stage_corrected.py",
        cfg.DOCS / "reference_source_hashes.json",
        cfg.V4_META_DIR / "train_metadata.csv",
        cfg.V4_META_DIR / "external_metadata.csv",
    ] + [cfg.INPUT_RAW / x for x in cfg.RAW_FILES.values()]
    missing = [str(p.relative_to(cfg.ROOT)) for p in required if not p.exists()]
    hashes = json.loads((cfg.DOCS / "reference_source_hashes.json").read_text(encoding="utf-8"))
    mismatches = []
    for rel, expected in hashes.get("copied_file_sha256", {}).items():
        p = cfg.ROOT / rel
        if p.exists() and sha256_file(p) != expected:
            mismatches.append(rel)
    feature_status = {}
    from .feature_adapter import expected_feature_paths, validate_feature_outputs

    feature_paths = expected_feature_paths()
    features_present = all(p.exists() for family in feature_paths.values() for p in family)
    if require_features or features_present:
        try:
            feature_status = validate_feature_outputs()
        except Exception as exc:
            missing.append(f"feature validation: {exc}")
    result = {
        "passed": not missing and not mismatches,
        "missing": missing,
        "source_hash_mismatches": mismatches,
        "python": sys.version,
        "feature_status": feature_status,
        "stagev5_cv_protocol": {
            "n_splits": cfg.CV_N_SPLITS,
            "n_repeats": cfg.CV_N_REPEATS,
            "scoring": cfg.SCORING,
            "selection_metric": "external_accuracy",
        },
    }
    out = cfg.OUTPUT / "checks" / "self_check_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if feature_status:
        print(f"E model features: {feature_status['n_E']}")
        print(f"M model features: {feature_status['n_M']}")
        print(f"L raw F8 model features: {feature_status['late_model_feature_count']}")
        print(f"L auxiliary diagnostic features: {feature_status['late_diagnostic_column_count']}")
    if not result["passed"]:
        raise SystemExit(1)
    return result


def check_api_environment() -> dict[str, Any]:
    """Report API-related environment state without making a network request."""
    key = bool(os.getenv("MAAS_API_KEY", "").strip())
    report = {
        "MAAS_API_KEY_detected": key,
        "middle_reference_cache_present": cfg.ASSET_CACHE.exists(),
        "late_P4_cache_present": (cfg.STAGEV4_EXTRACT_ROOT / "cache/late_p4_unmasked_cache.csv").exists(),
    }
    p = cfg.OUTPUT / "checks" / "api_environment_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
