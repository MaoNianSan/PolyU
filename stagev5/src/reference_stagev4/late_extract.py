from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .api_client import HuaweiMaaSClient, MaaSError
from .late_mask import unmasked_transcript
from .late_prompt import JSON_KEYS, RAW8_COLUMNS, build_messages, prompt_hash

CACHE_FILE = "late_p4_unmasked_cache.csv"
CACHE_COLUMNS = [
    "cache_key", "sample_id", "split", "model", "prompt_hash", "text_hash", "scored_text_hash",
    "scores_json", "source", "api_attempts", "created_at", "error",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _feature_path(split: str) -> Path:
    return config.FEATURE_DIR / f"{split}_late_p4_unmasked_f8.csv"


def _cache_key(sample_id: str, raw_text: str, scored_text: str) -> str:
    material = "\n".join([
        str(sample_id), config.LATE_LLM_MODEL, config.LATE_PROMPT_VERSION, config.LATE_MASK_VERSION,
        config.LATE_DIMENSION_VERSION, prompt_hash(), sha256_text(raw_text), sha256_text(scored_text),
    ])
    return sha256_text(material)


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    if "cache_key" not in df:
        return {}
    return {str(r["cache_key"]): r.to_dict() for _, r in df.iterrows()}


def _append_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in CACHE_COLUMNS:
        if c not in df:
            df[c] = ""
    df[CACHE_COLUMNS].to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False, encoding="utf-8")


def parse_strict_json(content: str) -> dict[str, float]:
    raw = str(content).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        raise ValueError("Response must be one JSON object and no prose.")
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != set(JSON_KEYS):
        raise ValueError(f"JSON keys must exactly equal {JSON_KEYS}; actual={sorted(obj) if isinstance(obj, dict) else type(obj)}")
    out: dict[str, float] = {}
    for col, key in zip(RAW8_COLUMNS, JSON_KEYS):
        value = obj[key]
        if isinstance(value, bool) or not isinstance(value, int) or value not in config.LATE_ALLOWED_VALUES:
            raise ValueError(f"{key} must be one of {sorted(config.LATE_ALLOWED_VALUES)}; got {value!r}")
        out[col] = float(value)
    return out


def _cached_scores(row: dict[str, Any] | None) -> dict[str, float] | None:
    if not row or str(row.get("source", "")) not in {"api", "cache"}:
        return None
    try:
        obj = json.loads(str(row["scores_json"]))
        if set(obj) != set(RAW8_COLUMNS):
            return None
        values = {k: int(v) for k, v in obj.items()}
        if any(v not in config.LATE_ALLOWED_VALUES for v in values.values()):
            return None
        return {k: float(v) for k, v in values.items()}
    except Exception:
        return None


def request_scores(client: HuaweiMaaSClient, scored_text: str) -> tuple[dict[str, float], int]:
    last = ""
    for attempt in range(1, config.LLM_SCHEMA_MAX_ATTEMPTS + 1):
        try:
            return parse_strict_json(client.chat(build_messages(scored_text))), attempt
        except (MaaSError, ValueError, json.JSONDecodeError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"P4 unmasked late scoring failed strict schema validation after {config.LLM_SCHEMA_MAX_ATTEMPTS} attempts. {last}")


def extract_late_scores(meta: pd.DataFrame, split: str, force: bool = False) -> pd.DataFrame:
    config.ensure_output_dirs()
    path = _feature_path(split)
    wanted_ids = meta["sample_id"].astype(str).tolist()
    if path.exists() and not force:
        existing = pd.read_csv(path)
        required = {"sample_id", *RAW8_COLUMNS, "scored_text"}
        if required.issubset(existing.columns) and existing["sample_id"].astype(str).tolist() == wanted_ids:
            if existing[RAW8_COLUMNS].isna().any().any():
                raise ValueError(f"Existing late P4 unmasked feature file has missing values: {path}")
            if "late_llm_source" not in existing.columns or not set(existing["late_llm_source"].astype(str)).issubset({"api", "cache"}):
                raise ValueError(f"Existing late P4 unmasked feature file has a non-API/non-cache source and cannot be reused: {path}")
            return existing

    cache_path = config.CACHE_DIR / CACHE_FILE
    cache = _load_cache(cache_path)
    client = HuaweiMaaSClient()
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    new_cache: list[dict[str, Any]] = []
    for _, item in tqdm(meta.iterrows(), total=len(meta), desc=f"Late {split} P4 unmasked F8"):
        sample_id = str(item["sample_id"])
        raw_text = str(item["text"])
        scored = unmasked_transcript(raw_text)
        key = _cache_key(sample_id, raw_text, scored)
        scores = _cached_scores(cache.get(key))
        source = "cache" if scores is not None else "api"
        attempts = 0
        error = ""
        if scores is None:
            if not client.has_key():
                raise RuntimeError(f"Missing {config.MAAS_API_KEY_ENV}; no valid P4 unmasked cache for {sample_id}.")
            try:
                scores, attempts = request_scores(client, scored)
            except RuntimeError as exc:
                error = str(exc)
                audit.append({"sample_id": sample_id, "split": split, "status": "failed", "attempts": config.LLM_SCHEMA_MAX_ATTEMPTS, "message": error})
                pd.DataFrame(audit).to_csv(config.DIAGNOSTICS_DIR / "late_response_validation_report.csv", index=False, encoding="utf-8")
                raise
            row = {
                "cache_key": key, "sample_id": sample_id, "split": split, "model": config.LATE_LLM_MODEL,
                "prompt_hash": prompt_hash(), "text_hash": sha256_text(raw_text), "scored_text_hash": sha256_text(scored),
                "scores_json": json.dumps({k: int(v) for k, v in scores.items()}, sort_keys=True), "source": "api",
                "api_attempts": attempts, "created_at": _utc(), "error": "",
            }
            new_cache.append(row)
            cache[key] = row
        audit.append({"sample_id": sample_id, "split": split, "status": "valid", "attempts": attempts, "message": "", "source": source})
        rows.append({
            "sample_id": sample_id, "split": split, "mask_version": config.LATE_MASK_VERSION,
            "prompt_version": config.LATE_PROMPT_VERSION, "prompt_hash": prompt_hash(), "text_hash": sha256_text(raw_text),
            "scored_text_hash": sha256_text(scored), "scored_text": scored, "late_llm_source": source, "late_llm_attempts": attempts,
            **scores,
        })
    _append_cache(cache_path, new_cache)
    pd.DataFrame(audit).to_csv(config.DIAGNOSTICS_DIR / "late_response_validation_report.csv", index=False, encoding="utf-8")
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False, encoding="utf-8")
    return out


def run_llm_stability(meta: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = config.DIAGNOSTICS_DIR / "llm_stability_repeated_scores.csv"
    summary_path = config.DIAGNOSTICS_DIR / "llm_stability_summary.csv"
    if raw_path.exists() and summary_path.exists() and not force:
        return pd.read_csv(raw_path), pd.read_csv(summary_path)
    staged = meta.copy()
    staged["group"] = np.where(staged["label"].astype(int).eq(0), "control", staged["stage_label"].fillna("ad_missing_stage"))
    rng = np.random.default_rng(config.PRIMARY_SEED)
    chosen = []
    for group, sub in staged.groupby("group", sort=True):
        chosen.append(sub.sample(n=min(config.LLM_STABILITY_N_PER_GROUP, len(sub)), random_state=int(rng.integers(1, 2**31 - 1))))
    selected = pd.concat(chosen, ignore_index=True)
    client = HuaweiMaaSClient()
    if not client.has_key():
        raise RuntimeError(f"{config.MAAS_API_KEY_ENV} is required for --mode llm_stability.")
    rows = []
    for _, item in tqdm(selected.iterrows(), total=len(selected), desc="P4 unmasked LLM stability"):
        scored = unmasked_transcript(str(item["text"]))
        for replicate in (1, 2):
            scores, attempts = request_scores(client, scored)
            rows.append({"sample_id": str(item["sample_id"]), "group": item["group"], "replicate": replicate, "attempts": attempts, **scores})
    raw = pd.DataFrame(rows)
    diffs = []
    for sample_id, sub in raw.groupby("sample_id"):
        a = sub.loc[sub["replicate"].eq(1), RAW8_COLUMNS].iloc[0].to_numpy(float)
        b = sub.loc[sub["replicate"].eq(2), RAW8_COLUMNS].iloc[0].to_numpy(float)
        diffs.append({"sample_id": sample_id, "mean_abs_difference": float(np.mean(np.abs(a-b))), "exact_vector_match": int(np.array_equal(a,b))})
    pair = pd.DataFrame(diffs)
    summary_rows = [{
        "n_samples": int(pair.shape[0]),
        "mean_abs_difference": float(pair["mean_abs_difference"].mean()),
        "max_sample_mean_abs_difference": float(pair["mean_abs_difference"].max()),
        "exact_vector_match_rate": float(pair["exact_vector_match"].mean()),
    }]
    for col in RAW8_COLUMNS:
        wide = raw.pivot(index="sample_id", columns="replicate", values=col)
        summary_rows[0][f"{col}_mean_abs_difference"] = float((wide[1]-wide[2]).abs().mean())
    summary = pd.DataFrame(summary_rows)
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    return raw, summary
