from __future__ import annotations

import json
import os
from pathlib import Path

from . import config
from .models import model_specs


def raw_files_exist() -> bool:
    return all((config.INPUT_RAW_DIR / name).is_file() for name in config.RAW_FILES.values())


def _load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def feature_manifest_status() -> tuple[bool, str]:
    early_ok = True
    for variant in config.EARLY_VARIANTS:
        base = config.FEATURE_DIR / "early" / variant
        manifest = _load_manifest(base / "early_feature_manifest.json")
        early_ok &= (
            (base / "train_early_features.csv").is_file()
            and (base / "external_early_features.csv").is_file()
            and manifest.get("early_variant") == variant
            and manifest.get("token_count_model_input") is False
        )

    middle_dir = config.FEATURE_DIR / "middle"
    middle = _load_manifest(middle_dir / "middle_feature_manifest.json")
    middle_mode = str(middle.get("api_or_cache_mode", "")).lower()
    middle_ok = (
        (middle_dir / "train_middle_features.csv").is_file()
        and (middle_dir / "external_middle_features.csv").is_file()
        and middle.get("local_surrogate_allowed") is False
        and middle_mode in {"api", "cache", "mixed"}
    )

    late_dir = config.FEATURE_DIR / "late"
    late = _load_manifest(late_dir / "late_feature_manifest.json")
    counts = late.get("api_or_cache_mode_counts", {}) or {}
    late_sources = {str(k).lower() for k, value in counts.items() if int(value) > 0}
    late_ok = (
        (late_dir / "train_late_features.csv").is_file()
        and (late_dir / "external_late_features.csv").is_file()
        and late.get("local_surrogate_allowed") is False
        and bool(late_sources)
        and late_sources <= {"api", "cache"}
    )
    if early_ok and middle_ok and late_ok:
        sources = "real_api" if "api" in ({middle_mode} | late_sources) else "valid_cache"
        return True, sources
    return False, "incomplete"


def token_count_excluded() -> bool:
    return all(
        _load_manifest(
            config.FEATURE_DIR / "early" / variant / "early_feature_manifest.json"
        ).get("token_count_model_input") is False
        for variant in config.EARLY_VARIANTS
    )


def dry_run_lines() -> tuple[list[str], bool]:
    config.ensure_dirs()
    feature_ok, _ = feature_manifest_status()
    specs = len(model_specs())
    variants_ok = config.EARLY_VARIANTS == ["earlyv0", "earlyv1"]
    output_ok = all(
        path.is_dir()
        for path in [config.OUTPUT_DIR, config.FEATURE_DIR, config.FINAL_REPORT_DIR, config.FIGURES_DIR]
    )
    try:
        from .visualization import generate_figures  # noqa: F401
        visualization_ok = True
    except Exception:
        visualization_ok = False
    key_ok = bool(os.getenv(config.MAAS_API_KEY_ENV, "").strip())
    raw_ok = raw_files_exist()
    lines = [
        f"Project: {config.PROJECT_NAME}",
        f"Raw CSV files: {'OK' if raw_ok else 'missing'}",
        f"MAAS_API_KEY: {'detected' if key_ok else 'missing'}",
        f"Feature files: {'complete' if feature_ok else 'incomplete'}",
        f"Model specs per early variant: {specs}",
        f"Early variants: {', '.join(config.EARLY_VARIANTS)}",
        f"Expected main rows: {config.expected_main_rows()}",
        f"Expected stability rows under seeds 0-29: {config.expected_main_rows() * 30}",
        f"Output directories: {'OK' if output_ok else 'missing'}",
        f"Visualization module: {'OK' if visualization_ok else 'missing'}",
    ]
    structural_ok = raw_ok and specs == 102 and variants_ok and output_ok and visualization_ok
    return lines, structural_ok
