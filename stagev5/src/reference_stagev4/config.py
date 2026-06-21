from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
RAW_DIR = INPUT_DIR / "raw"
FROZEN_DIR = INPUT_DIR / "frozen_features"
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache"
FEATURE_DIR = OUTPUT_DIR / "features"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
FINAL_DIR = OUTPUT_DIR / "final_report"
LOG_DIR = OUTPUT_DIR / "logs"
SUMMARY_DIR = OUTPUT_DIR / "summary"

RAW_FILES = {
    "ad": "ad_s2t_wav2vec.csv",
    "control": "control_s2t_wav2vec.csv",
    "external": "test_s2t_wav2vec.csv",
}
FROZEN_FILES = {
    "train_meta": "train_metadata.csv",
    "external_meta": "external_metadata.csv",
    "train_early": "train_earlyv0.csv",
    "external_early": "external_earlyv0.csv",
    "train_middle": "train_middle_v2_full_mean.csv",
    "external_middle": "external_middle_v2_full_mean.csv",
}

# Scientific protocol: intentionally not environment-overridable.
PRIMARY_SEED = 2026
N_SPLITS = 5
N_REPEATS = 3
DECISION_THRESHOLD = 0.50
BOOTSTRAP_N = 1000
CLASSIFIERS = ("lr_l2", "linear_svm", "svc_poly3_history", "svc_rbf")
LATE_CLASSIFIER = "lr_l2"
LATE_PROMPT_VERSION = "P4_unmasked_form_integrity"
LATE_MASK_VERSION = "U0_unmasked_normalized_asr"
LATE_DIMENSION_VERSION = "F8_analytic_integrity_v1"
LATE_ALLOWED_VALUES = {1, 3, 5, 7, 9}
LLM_SCHEMA_MAX_ATTEMPTS = 3
LLM_STABILITY_N_PER_GROUP = 3

# API transport only: environment-overridable by design.
MAAS_API_KEY_ENV = "MAAS_API_KEY"
HUAWEI_MAAS_BASE_URL = os.getenv("HUAWEI_MAAS_BASE_URL", "https://api.modelarts-maas.com/v1").rstrip("/")
HUAWEI_MAAS_SSL_VERIFY = os.getenv("HUAWEI_MAAS_SSL_VERIFY", "true").lower() not in {"0", "false", "no"}
HUAWEI_MAAS_TRUST_ENV = os.getenv("HUAWEI_MAAS_TRUST_ENV", "false").lower() in {"1", "true", "yes"}
HUAWEI_MAAS_TIMEOUT = float(os.getenv("HUAWEI_MAAS_TIMEOUT", "120"))
HUAWEI_MAAS_MAX_RETRIES = int(os.getenv("HUAWEI_MAAS_MAX_RETRIES", "3"))
HUAWEI_MAAS_BACKOFF_SECONDS = float(os.getenv("HUAWEI_MAAS_BACKOFF_SECONDS", "1.5"))
LATE_LLM_MODEL = os.getenv("LATE_LLM_MODEL", "qwen3-235b-a22b")
LATE_LLM_TEMPERATURE = 0.0
LATE_LLM_TOP_P = 1.0
LATE_LLM_MAX_TOKENS = 700
LATE_LLM_ENDPOINT = "/chat/completions"


def ensure_output_dirs() -> None:
    for d in (CACHE_DIR, FEATURE_DIR, DIAGNOSTICS_DIR, METRICS_DIR, FIGURES_DIR, FINAL_DIR, LOG_DIR, SUMMARY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def protocol_dict() -> dict:
    return json.loads((CONFIG_DIR / "scientific_protocol.json").read_text(encoding="utf-8"))


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(protocol_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
