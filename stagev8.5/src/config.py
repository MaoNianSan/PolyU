from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_RAW_DIR = ROOT / "input" / "raw"
RAW_FILES = {"ad":"ad_s2t_wav2vec.csv", "control":"control_s2t_wav2vec.csv", "test":"test_s2t_wav2vec.csv"}
ASSETS = ROOT / "assets"
STAGEV5_EXACT_ROOT = ASSETS / "stagev5_exact"
STAGEV5_EXACT_SRC_ROOT = STAGEV5_EXACT_ROOT / "src"

OUTPUT = ROOT / "output"
FEATURE_ROOT = OUTPUT / "features"
EARLY_DIR = FEATURE_ROOT / "E" / "raw_stagev2"
MIDDLE_DIR = FEATURE_ROOT / "M" / "raw_stagev2"
LATE_DIR = FEATURE_ROOT / "L" / "raw_stagev4"
EXTRACTION_ROOT = OUTPUT / "feature_extraction"
STAGEV2_EXTRACTION_ROOT = EXTRACTION_ROOT / "stagev2_exact"
STAGEV4_EXTRACTION_ROOT = EXTRACTION_ROOT / "stagev4_unmasked"
FINAL = OUTPUT / "final_report"
FIGURES = FINAL / "figures"
MODELS = OUTPUT / "stagev8_5_models"
CHECKS = OUTPUT / "checks"

RANDOM_STATE = 2026
CV_N_SPLITS = 10
NESTED_INNER_CV_N_SPLITS = 5
BOOTSTRAP_N = 200
ANCHOR_THRESHOLD = 0.50
SEVERITY_CONFIDENCE_THRESHOLD = 0.50
SEVERITY_MARGIN_THRESHOLD = 0.10

# Stagev8.5 uses MMSE-informed severity strata, not clinical early/middle/late labels.
MMSE_HIGH_MIN = 21
MMSE_INTERMEDIATE_MIN = 15
MMSE_INTERMEDIATE_MAX = 20
MMSE_LOW_MAX = 14
SEVERITY_STRATA = ["high_mmse_AD", "intermediate_mmse_AD", "low_mmse_AD"]
SEVERITY_STRATA_WITH_CONTROL = ["control", *SEVERITY_STRATA]

RAW_F8_BASE_NAMES = [
    "sentence_structural_integrity", "phrase_continuity", "repetition_control", "repair_efficiency",
    "filler_control", "referential_clarity", "grammatical_stability", "local_coherence",
]
EXPECTED_FEATURE_COUNTS = {"E":61, "M":1024, "L":8}
COMPLETION_SENTINEL = "stagev8_5_training_complete.json"

CANONICAL_FINAL_FILES = [
    "stagev8_5_feature_lock.json",
    "stagev8_5_feature_source_audit.json",
    "stagev8_5_stagev6_source_contract.json",
    "stagev8_5_feature_parity_audit.json",
    "stagev8_5_mmse_label_contract.json",
    "stagev8_5_literature_rationale.md",
    "stagev8_5_no_external_selection_audit.json",
    "stagev8_5_anchor_parity_audit.csv",
    "stagev8_5_T20_cv_grid.csv",
    "stagev8_5_T14_cv_grid.csv",
    "stagev8_5_T20_selected_oof.csv",
    "stagev8_5_T14_selected_oof.csv",
    "stagev8_5_nested_oof_severity_scores.csv",
    "stagev8_5_nested_oof_metrics.json",
    "stagev8_5_head_seed_stability.csv",
    "stagev8_5_ordinal_seed_stability.csv",
    "stagev8_5_external_severity_scores.csv",
    "stagev8_5_external_binary_metrics.csv",
    "stagev8_5_external_ordinal_metrics.csv",
    "stagev8_5_external_threshold_metrics.csv",
    "stagev8_5_external_three_strata_metrics.csv",
    "stagev8_5_external_selective_metrics.csv",
    "stagev8_5_external_calibration_T20.csv",
    "stagev8_5_external_calibration_T14.csv",
    "stagev8_5_external_abstention_by_stratum.csv",
    "stagev8_5_confusion_matrix_three_strata.csv",
    "stagev8_5_bootstrap_ci.csv",
    "stagev8_5_model_contract.json",
    "stagev8_5_selected_model_summary.md",
    "stagev8_5_experiment_report.md",
    "stagev8_5_final_run_summary.json",
    COMPLETION_SENTINEL,
]
CANONICAL_FIGURES = [
    "figures/fig01_nested_oof_severity_vs_mmse.png",
    "figures/fig02_external_severity_vs_mmse.png",
    "figures/fig03_external_stratum_probability_distribution.png",
    "figures/fig04_threshold_calibration.png",
    "figures/fig05_selective_coverage_by_stratum.png",
    "figures/fig06_bootstrap_ci.png",
]
