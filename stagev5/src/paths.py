"""Central filesystem paths for the stagev5 project.

All paths are resolved from this file's location so scripts work from a cloned
repository on Windows, macOS, or Linux without user-specific absolute paths.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "input"
INPUT_RAW_DIR = INPUT_DIR / "raw"
REFERENCE_STAGEV4_METADATA_DIR = INPUT_DIR / "reference_stagev4_late_metadata"

ASSET_DIR = PROJECT_ROOT / "assets"
REFERENCE_STAGEV2_CACHE = ASSET_DIR / "reference_stagev2_cache" / "huawei_bge_m3_embedding_cache.csv"

OUTPUT_DIR = PROJECT_ROOT / "output"
FEATURE_DIR = OUTPUT_DIR / "features"
EARLY_FEATURE_DIR = FEATURE_DIR / "E" / "raw_stagev2"
MIDDLE_FEATURE_DIR = FEATURE_DIR / "M" / "raw_stagev2"
LATE_FEATURE_DIR = FEATURE_DIR / "L" / "raw_stagev4"

EXTRACTION_DIR = OUTPUT_DIR / "feature_extraction"
STAGEV2_EXTRACTION_DIR = EXTRACTION_DIR / "stagev2_exact"
STAGEV4_EXTRACTION_DIR = EXTRACTION_DIR / "stagev4_unmasked"
CLASSIFIER_RUN_DIR = OUTPUT_DIR / "run_external_accuracy_selection"
FINAL_REPORT_DIR = OUTPUT_DIR / "final_report"
FIGURE_DIR = FINAL_REPORT_DIR / "figures"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
CONFIG_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
RUNLOG_DIR = PROJECT_ROOT / "runlogs"
