from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_RAW_DIR = PROJECT_ROOT / "input" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
FEATURE_DIR = OUTPUT_DIR / "features"
RUNS_DIR = OUTPUT_DIR / "runs"
FINAL_REPORT_DIR = OUTPUT_DIR / "final_report"
CACHE_DIR = OUTPUT_DIR / "cache"
PREPROCESS_DIR = OUTPUT_DIR / "preprocess"

RAW_FILES = {
    "ad": "ad_s2t_wav2vec.csv",
    "control": "control_s2t_wav2vec.csv",
    "test": "test_s2t_wav2vec.csv",
}

PRIMARY_SEED = 2026
STABILITY_SEEDS = list(range(30))
N_SPLITS = 10
DECISION_THRESHOLD = 0.5

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
]
EARLY_VARIANTS = ["earlyv0", "earlyv1"]
SCALE_PAIRS = [
    ("early_middle", "early_middle_scale"),
    ("middle_late", "middle_late_scale"),
    ("all", "all_plus_interactions"),
]

MODEL_PARAM_GRID = {
    "Logistic Regression": [
        {"C": 1.0, "class_weight": None},
    ],
    "Linear SVM": [
        {"C": 1.0, "class_weight": None},
    ],
    "RBF SVM": [
        {"C": 1.0, "gamma": "scale", "class_weight": None},
    ],
}

HUAWEI_MAAS_API_KEY_ENV = "MAAS_API_KEY"
HUAWEI_MAAS_BASE_URL = os.getenv("HUAWEI_MAAS_BASE_URL", "https://api.modelarts-maas.com/v1")
HUAWEI_AUTH_HEADER = "Authorization"
HUAWEI_AUTH_PREFIX = "Bearer"
HUAWEI_MAAS_TIMEOUT = 120
HUAWEI_MAAS_MAX_RETRIES = 5
HUAWEI_MAAS_BACKOFF_SECONDS = 2.0
HUAWEI_MAAS_TRUST_ENV = True

EMBEDDING_MODEL = "bge-m3"
EMBEDDING_ENDPOINT = "/embeddings"
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_DIM_LOCAL_FALLBACK = 64
WINDOW_SIZE_WORDS = 15
STRIDE_WORDS = 5

LATE_LLM_MODEL = "qwen3-235b-a22b"
LATE_LLM_ENDPOINT = "/chat/completions"
LATE_LLM_TEMPERATURE = 0.0
LATE_LLM_TOP_P = 1.0
LATE_LLM_MAX_TOKENS = 256
LATE_LLM_PROMPT_VERSION = "late_form_masked_v1"

# Default is True so the project remains runnable without private API keys.
# Set STAGEV2_REQUIRE_API=1 to enforce real Huawei MaaS/LLM calls and fail if no cache/key is available.
ALLOW_LOCAL_SURROGATE_WITHOUT_API = os.getenv("STAGEV2_REQUIRE_API", "0").strip() not in {"1", "true", "TRUE", "yes"}

def ensure_dirs() -> None:
    for p in [OUTPUT_DIR, FEATURE_DIR, RUNS_DIR, FINAL_REPORT_DIR, CACHE_DIR, PREPROCESS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
