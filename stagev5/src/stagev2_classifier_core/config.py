"""Configuration for the stage-corrected AD classifier package."""
from __future__ import annotations

import os
from pathlib import Path

# The package is intended to sit beside AD_predata_project:
# project_root/
#   AD_predata_project/
#   stage_corrected/
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
AD_PROJECT_ROOT = Path(os.environ.get("STAGE_AD_PROJECT_ROOT", PROJECT_ROOT / "AD_predata_project")).resolve()
OUTPUT_ROOT = Path(os.environ.get("STAGE_OUTPUT_ROOT", PACKAGE_ROOT / "output")).resolve()

BM25_DIR = Path(os.environ.get("STAGE_BM25_DIR", AD_PROJECT_ROOT / "output" / "features" / "bm25")).resolve()
EMBEDDING_DIR = Path(os.environ.get("STAGE_EMBEDDING_DIR", AD_PROJECT_ROOT / "output" / "features" / "embedding")).resolve()
LLM_DIR = Path(os.environ.get("STAGE_LLM_DIR", AD_PROJECT_ROOT / "output" / "features" / "llm")).resolve()
PREPROCESS_DIR = Path(os.environ.get("STAGE_PREPROCESS_DIR", AD_PROJECT_ROOT / "output" / "preprocess")).resolve()

RANDOM_STATE = 2026
CV_N_SPLITS = int(os.environ.get("STAGE_CV_N_SPLITS", "10"))
CV_N_REPEATS = int(os.environ.get("STAGE_CV_N_REPEATS", "1"))
INNER_N_SPLITS = int(os.environ.get("STAGE_INNER_N_SPLITS", "3"))
BOOTSTRAP_N = int(os.environ.get("STAGE_BOOTSTRAP_N", "200"))
N_JOBS = int(os.environ.get("STAGE_N_JOBS", "-1"))
SCORING = os.environ.get("STAGE_SCORING", "accuracy")
DECISION_THRESHOLD = float(os.environ.get("STAGE_DECISION_THRESHOLD", "0.5"))

# Model grids are intentionally moderate for a small-N external-validation setting.
CLASS_WEIGHTS = [None, "balanced"]
LR_C_GRID = [0.03, 0.1, 1.0]
SVC_C_GRID = [0.1, 1.0, 3.0]
POLY_DEGREES = [2, 3]
POLY_COEF0 = [0.0, 1.0]
MLP_HIDDEN = [(16,), (32,)]
MLP_ALPHA = [0.001, 0.01]

# Stage feature matching. The loader renames recognized features to early_/middle_/late_ prefixes.
EARLY_PREFIXES = ["early_", "bm25_", "info_"]
MIDDLE_PREFIXES = ["middle_", "bge_", "embedding_dim_", "semantic_", "coherence_", "drift_", "align_", "window_"]
LATE_PREFIXES = ["late_", "llm_", "sentence", "fragmentation", "repetition", "repair_restart", "filler_vague", "grammatical", "connected_speech", "overall_expressive", "fluency", "lexical", "discourse", "expressive", "grammar", "syntax", "pause", "fragment", "completeness"]

ID_COLUMNS = ["sample_id", "file", "filename", "id", "subject_id"]
TEXT_COLUMNS = ["text", "speech", "transcript", "masked_text", "pos_skeleton"]
LABEL_COLUMNS = [
    "label", "disease", "disease_label", "y", "new_label",
    "label_normal", "label_mild", "label_moderate", "label_severe",
    "normal", "mild", "moderate", "severe", "mmse", "split",
]
NON_FEATURE_KEYWORDS = ["success", "error", "model", "api", "cache", "status", "prompt", "raw", "json", "response", "reason"]

# Hard-case threshold: wrong by at least this fraction of completed models.
HARD_CASE_WRONG_FRACTION = 0.60
