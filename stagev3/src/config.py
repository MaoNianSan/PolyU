from __future__ import annotations

import os
from pathlib import Path

PROJECT_NAME = "stagev3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_RAW_DIR = PROJECT_ROOT / "input" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
FEATURE_DIR = OUTPUT_DIR / "features"
RUNS_DIR = OUTPUT_DIR / "runs"
FINAL_REPORT_DIR = OUTPUT_DIR / "final_report"
FIGURES_DIR = FINAL_REPORT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"
PREPROCESS_DIR = OUTPUT_DIR / "preprocess"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

RAW_FILES = {
    "ad": "ad_s2t_wav2vec.csv",
    "control": "control_s2t_wav2vec.csv",
    "test": "test_s2t_wav2vec.csv",
}

PRIMARY_SEED = 2026
DEFAULT_STABILITY_SEEDS = list(range(30))
N_SPLITS_EXACT = 10
N_SPLITS_FAST = 3
N_SPLITS_V2_REPEATED = 5
N_REPEATS_V2_REPEATED = 3
INNER_N_SPLITS = 3
CI_LEVEL = 0.95
DECISION_THRESHOLD = 0.5

EARLY_VARIANTS = ["earlyv0", "earlyv1"]

FEATURE_BLOCKS = [
    "early_only",
    "middle_only",
    "late_only",
    "early_middle",
    "early_middle_scale",
    "middle_late",
    "middle_late_scale",
    "all",
    "all_plus_interactions",
    "early_late",
    "stage_activation_summary",
    "sequential_interactions",
]

CLASSIFIER_VARIANTS = [
    "lr__l2",
    "lr__l1",
    "lr__elasticnet",
    "svc__linear",
    "svc__poly2",
    "svc__poly3",
    "svc__rbf",
    "svc__sigmoid",
]

SPECIAL_MODEL_SPECS = [
    "stage_score_early_middle__lr__l2",
    "stage_score_middle_late__lr__l2",
    "stage_score_early_late__lr__l2",
    "stage_score_three_stage__lr__l2",
    "early_middle__mlp__small",
    "mlp_svc_late_calibrated__lr__l2",
]

SCALE_PAIRS = [
    ("early_middle", "early_middle_scale"),
    ("middle_late", "middle_late_scale"),
    ("all", "all_plus_interactions"),
    ("early_late", "sequential_interactions"),
]

MAAS_API_KEY_ENV = "MAAS_API_KEY"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = int(value.strip())
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = float(value.strip())
    if parsed <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {parsed}")
    return parsed


HUAWEI_MAAS_BASE_URL = os.getenv(
    "HUAWEI_MAAS_BASE_URL", "https://api.modelarts-maas.com/v1"
).strip()
HUAWEI_AUTH_HEADER = "Authorization"
HUAWEI_AUTH_PREFIX = "Bearer"
HUAWEI_MAAS_TRUST_ENV = os.getenv("HUAWEI_MAAS_TRUST_ENV", "false").lower() in {"1", "true", "yes"}
HUAWEI_MAAS_SSL_VERIFY = os.getenv("HUAWEI_MAAS_SSL_VERIFY", "true").lower() in {"1", "true", "yes"}
HUAWEI_MAAS_TIMEOUT = _env_float("HUAWEI_MAAS_TIMEOUT", 120.0)
HUAWEI_MAAS_MAX_RETRIES = _env_int("HUAWEI_MAAS_MAX_RETRIES", 5)
HUAWEI_MAAS_BACKOFF_SECONDS = 2.0

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3").strip()
EMBEDDING_ENDPOINT = "/embeddings"
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1"))
# Separate embedding-stage network controls. The late LLM can keep the longer
# default timeout, while BGE-M3 cache filling should fail fast enough to avoid
# appearing frozen on a single problematic request.
EMBEDDING_TIMEOUT = _env_float("EMBEDDING_TIMEOUT", 45.0)
EMBEDDING_MAX_RETRIES = _env_int("EMBEDDING_MAX_RETRIES", 2)
MIDDLE_SHOW_PROGRESS = _env_bool("MIDDLE_SHOW_PROGRESS", True)
MIDDLE_FEATURE_MODE = os.getenv("MIDDLE_FEATURE_MODE", "v2_full_mean").strip()
MIDDLE_KEEP_DIMS = int(os.getenv("MIDDLE_KEEP_DIMS", "1024"))
MIDDLE_INCLUDE_V3_STATS = _env_bool("MIDDLE_INCLUDE_V3_STATS", False)
WINDOW_SIZE_WORDS = 15
STRIDE_WORDS = 5

LATE_LLM_MODEL = "qwen3-235b-a22b"
LATE_LLM_ENDPOINT = "/chat/completions"
LATE_LLM_TEMPERATURE = 0.0
LATE_LLM_TOP_P = 1.0
LATE_LLM_MAX_TOKENS = 256
LATE_LLM_PROMPT_VERSION = "late_form_masked_v1"
LATE_ADD_V2_DERIVED_FORM_FEATURES = _env_bool("LATE_ADD_V2_DERIVED_FORM_FEATURES", True)

LOCAL_SURROGATE_ALLOWED = False

# v2-compatible model grids. These are intentionally larger than the earlier
# stagev3 single-point grids because the v2 high-accuracy anchor depends on
# SVC poly coef0/probability and small-sample class-weight choices.
CLASS_WEIGHTS = [None, "balanced"]
LR_C_GRID = [0.03, 0.1, 1.0]
SVC_C_GRID = [0.1, 1.0, 3.0]
POLY_COEF0 = [0.0, 1.0]
MLP_HIDDEN = [(16,), (32,)]
MLP_ALPHA = [0.001, 0.01]


def ensure_dirs() -> None:
    for p in [
        INPUT_RAW_DIR, OUTPUT_DIR, FEATURE_DIR, RUNS_DIR, FINAL_REPORT_DIR, FIGURES_DIR,
        CACHE_DIR, PREPROCESS_DIR, DOCS_DIR, NOTEBOOK_DIR,
        FEATURE_DIR / "early" / "earlyv0", FEATURE_DIR / "early" / "earlyv1",
        FEATURE_DIR / "middle", FEATURE_DIR / "late",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def expected_main_rows() -> int:
    return len(EARLY_VARIANTS) * (len(FEATURE_BLOCKS) * len(CLASSIFIER_VARIANTS) + len(SPECIAL_MODEL_SPECS))


def expected_specs_per_variant() -> int:
    return len(FEATURE_BLOCKS) * len(CLASSIFIER_VARIANTS) + len(SPECIAL_MODEL_SPECS)
