from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FEATURE_ROOT = OUTPUT / "features"
EARLY_DIR = FEATURE_ROOT / "E" / "raw_stagev2"
MIDDLE_DIR = FEATURE_ROOT / "M" / "raw_stagev2"
LATE_DIR = FEATURE_ROOT / "L" / "raw_stagev4"
RUN = OUTPUT / "run_stagev6_cascade"
FINAL = OUTPUT / "final_report"
FIGURES = FINAL / "figures"

RANDOM_STATE = 2026
CV_N_SPLITS = 10
CV_N_REPEATS = 1
BOOTSTRAP_N = 200
DECISION_THRESHOLD = 0.50

RAW_F8_BASE_NAMES = [
    "sentence_structural_integrity",
    "phrase_continuity",
    "repetition_control",
    "repair_efficiency",
    "filler_control",
    "referential_clarity",
    "grammatical_stability",
    "local_coherence",
]

EXPECTED_FEATURE_COUNTS = {"E": 61, "M": 1024, "L": 8}

# Fixed stagev6 structure. All six final models are the Cartesian product of these components.
GATE_IDS = ["g1_l_lr_l2", "g2_middle_late_lr_l2", "g3_middle_late_svc_poly3"]
BRANCH_IDS = ["b1_early_middle_svc_poly3", "b2_early_middle_lr_l2"]

CANONICAL_FINAL_FILES = [
    "stagev6_model_ranking_by_external_accuracy.csv",
    "stagev6_external_performance_report.csv",
    "stagev6_cv_summary.csv",
    "stagev6_bootstrap_ci.csv",
    "stagev6_generalization_gap.csv",
    "stagev6_oof_predictions_top6.csv",
    "stagev6_test_predictions_all_models.csv",
    "stagev6_late_gate_performance.csv",
    "stagev6_nonlate_branch_performance.csv",
    "stagev6_route_diagnostics.csv",
    "stagev6_component_specifications.csv",
    "stagev6_stage_subgroup_accuracy.csv",
    "stagev6_error_analysis.csv",
    "stagev6_selected_model_summary.md",
    "stagev6_experiment_report.md",
    "stagev6_feature_source_manifest.json",
    "stagev6_leakage_check.json",
    "stagev6_final_run_summary.json",
]
