from __future__ import annotations

from . import paths

ROOT = paths.PROJECT_ROOT
INPUT_RAW = paths.INPUT_RAW_DIR
V4_META_DIR = paths.REFERENCE_STAGEV4_METADATA_DIR
ASSET_CACHE = paths.REFERENCE_STAGEV2_CACHE
OUTPUT = paths.OUTPUT_DIR
FEATURE_ROOT = paths.FEATURE_DIR
EARLY_DIR = paths.EARLY_FEATURE_DIR
MIDDLE_DIR = paths.MIDDLE_FEATURE_DIR
LATE_DIR = paths.LATE_FEATURE_DIR
EXTRACT_ROOT = paths.EXTRACTION_DIR
STAGEV2_EXTRACT_ROOT = paths.STAGEV2_EXTRACTION_DIR
STAGEV4_EXTRACT_ROOT = paths.STAGEV4_EXTRACTION_DIR
CLASSIFIER_RUN = paths.CLASSIFIER_RUN_DIR
FINAL = paths.FINAL_REPORT_DIR
FIGURES = paths.FIGURE_DIR
DOCS = paths.DOCS_DIR
NOTEBOOKS = paths.NOTEBOOK_DIR
CONFIGS = paths.CONFIG_DIR
RUNLOGS = paths.RUNLOG_DIR

RANDOM_STATE = 2026
CV_N_SPLITS = 10
CV_N_REPEATS = 1
INNER_N_SPLITS = 3
BOOTSTRAP_N = 200
DECISION_THRESHOLD = 0.50
SCORING = "accuracy"

RAW_FILES = {
    "ad": "ad_s2t_wav2vec.csv",
    "control": "control_s2t_wav2vec.csv",
    "test": "test_s2t_wav2vec.csv",
}

CANONICAL_FINAL_FILES = [
    "stagev5_model_ranking_by_external_accuracy.csv",
    "stagev5_external_performance_report.csv",
    "stagev5_cv_summary.csv",
    "stagev5_bootstrap_ci.csv",
    "stagev5_generalization_gap.csv",
    "stagev5_oof_predictions_top10.csv",
    "stagev5_test_predictions_all_models.csv",
    "stagev5_selected_model_summary.md",
    "stagev5_experiment_report.md",
    "stagev5_leakage_check.json",
    "stagev5_feature_source_manifest.json",
    "stagev5_stage_subgroup_accuracy.csv",
    "stagev5_error_analysis.csv",
    "stagev5_final_run_summary.json",
]
