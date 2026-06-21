from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_RAW = ROOT / "input" / "raw"
OUTPUT = ROOT / "output"
FEATURE_ROOT = OUTPUT / "features"
EARLY_DIR = FEATURE_ROOT / "E" / "raw_stagev2"
MIDDLE_DIR = FEATURE_ROOT / "M" / "raw_stagev2"
LATE_DIR = FEATURE_ROOT / "L" / "raw_stagev4"
FINAL = OUTPUT / "final_report"
FIGURES = FINAL / "figures"
RUN_ROOT = OUTPUT / "run_cascade_external_accuracy_selection"

RANDOM_STATE = 2026
CV_N_SPLITS = 10
CV_N_REPEATS = 1
BOOTSTRAP_N = 200
GATE_THRESHOLD = 0.50
BRANCH_THRESHOLD = 0.50

# Same grids and classifier family scale as stagev5's main model registry.
CLASS_WEIGHTS = [None, "balanced"]
LR_C_GRID = [0.03, 0.1, 1.0]
SVC_C_GRID = [0.1, 1.0, 3.0]
POLY_COEF0 = [0.0, 1.0]
MLP_HIDDEN = [(16,), (32,)]
MLP_ALPHA = [0.001, 0.01]

# Stagev5 feature contract. Stagev6 reads these existing artifacts without API calls.
FEATURE_POLICY = {
    "early": "strict stagev2 BM25 feature files", 
    "middle": "strict stagev2 BGE-M3 window embeddings aggregated by sample_id mean", 
    "late": "strict stagev4 unmasked raw P4/F8 expressive-form scores",
}

CANONICAL_FINAL_FILES = [
    "stagev6_model_ranking_by_external_accuracy.csv",
    "stagev6_model_ranking_by_external_test_accuracy.csv",
    "stagev6_external_performance_report.csv",
    "stagev6_external_test_performance_report.csv",
    "stagev6_cv_summary.csv",
    "stagev6_bootstrap_ci.csv",
    "stagev6_generalization_gap.csv",
    "stagev6_oof_predictions_top10.csv",
    "stagev6_test_predictions_all_models.csv",
    "stagev6_gate_component_performance.csv",
    "stagev6_branch_component_performance.csv",
    "stagev6_component_specifications.csv",
    "stagev6_route_diagnostics.csv",
    "stagev6_stage_subgroup_accuracy.csv",
    "stagev6_error_analysis.csv",
    "stagev6_selected_model_summary.md",
    "stagev6_experiment_report.md",
    "stagev6_feature_source_manifest.json",
    "stagev6_leakage_check.json",
    "stagev6_final_run_summary.json",
]
