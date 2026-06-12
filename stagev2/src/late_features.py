from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .middle_features import HuaweiMaaSClient, normalize_text, sha256_text, tokenize

TOKEN_OR_PUNCT_RE = re.compile(r"[A-Za-z']+|\d+|[^\w\s]")
PERSON_WORDS = {"boy", "girl", "child", "children", "kid", "kids", "mother", "woman", "lady", "son", "daughter", "sister", "brother", "he", "she", "him", "her", "they", "them", "person", "people"}
OBJECT_WORDS = {"cookie", "cookies", "jar", "stool", "chair", "sink", "water", "dish", "dishes", "cup", "plate", "curtain", "curtains", "window", "floor", "counter", "cabinet", "apron", "faucet", "towel"}
PLACE_WORDS = {"kitchen", "room", "house", "home", "inside", "outside", "yard", "garden"}
ACTION_WORDS = {"take", "taking", "reach", "reaching", "steal", "stealing", "fall", "falling", "wash", "washing", "drying", "overflow", "overflowing", "spill", "spilling", "stand", "standing", "sit", "sitting", "climb", "climbing", "look", "looking", "see", "doing", "get", "getting"}
SCORE_COLUMNS = [
    "sentence_completeness",
    "fragmentation_control",
    "repetition_control",
    "repair_restart_control",
    "filler_vague_control",
    "grammatical_stability",
    "connected_speech_quality",
    "overall_expressive_form_quality",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_mask_text(text: Any) -> str:
    tokens = TOKEN_OR_PUNCT_RE.findall(normalize_text(text))
    masked = []
    for tok in tokens:
        low = tok.lower()
        if low in PERSON_WORDS:
            masked.append("[PERSON]")
        elif low in OBJECT_WORDS:
            masked.append("[OBJECT]")
        elif low in PLACE_WORDS:
            masked.append("[PLACE]")
        elif low in ACTION_WORDS:
            masked.append("[ACTION]")
        elif re.fullmatch(r"\d+", tok):
            masked.append("[NUMBER]")
        else:
            masked.append(tok)
    out = " ".join(masked)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return re.sub(r"\s+", " ", out).strip()


def _llm_cache_key(masked_text: str) -> str:
    return sha256_text(config.LATE_LLM_MODEL + "\n" + config.LATE_LLM_PROMPT_VERSION + "\n" + masked_text)


def build_late_prompt(masked_text: str) -> list[dict[str, str]]:
    system = (
        "You are a formal connected-speech quality scorer. You are not diagnosing Alzheimer's disease. "
        "You are not estimating MMSE. You must not infer medical risk. You must not use scene-specific content. "
        "You only evaluate the formal quality of connected speech from a masked transcript. Return strict JSON only."
    )
    user = f"""
Evaluate only the expressive-form quality of the following content-masked transcript.
Rules:
- Do not diagnose any disease.
- Do not estimate MMSE.
- Do not infer medical risk.
- Do not evaluate whether Cookie Theft information units are complete.
- Do not use scene-specific content.
- Scores must range from 0 to 5, where 0 is extremely poor and 5 is very good.
- brief_form_only_reason must be at most 30 English words and must not contain medical diagnosis words.
Masked transcript:
{masked_text}
Return strict JSON with exactly these keys:
{{
  "sentence_completeness": 0.0,
  "fragmentation_control": 0.0,
  "repetition_control": 0.0,
  "repair_restart_control": 0.0,
  "filler_vague_control": 0.0,
  "grammatical_stability": 0.0,
  "connected_speech_quality": 0.0,
  "overall_expressive_form_quality": 0.0,
  "brief_form_only_reason": ""
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> dict[str, Any]:
    s = str(text).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("Could not parse JSON object")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")
    return obj


def sanitize_scores(obj: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for col in SCORE_COLUMNS:
        val = float(obj.get(col, 0.0))
        out[col] = max(0.0, min(5.0, val))
    out["brief_form_only_reason"] = str(obj.get("brief_form_only_reason", ""))[:240]
    return out


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["cache_key"]: row for row in csv.DictReader(f)}


def _append_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = ["cache_key", "model", "prompt_version", "source", "masked_text", "parsed_score_json", "created_at", "llm_parse_success", "llm_error"]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _local_form_scores(masked_text: str, raw_text: str) -> dict[str, Any]:
    toks = tokenize(raw_text)
    n = max(len(toks), 1)
    unique_ratio = len(set(toks)) / n
    filler_words = {"um", "uh", "er", "ah", "like", "you", "know"}
    filler_rate = sum(t in filler_words for t in toks) / n
    short_token_rate = sum(len(t) <= 2 for t in toks) / n
    repeat_rate = 0.0
    if len(toks) > 1:
        repeat_rate = sum(toks[i] == toks[i - 1] for i in range(1, len(toks))) / (len(toks) - 1)
    punct_count = len(re.findall(r"[.!?]", str(raw_text)))
    length_quality = min(np.log1p(n) / np.log1p(120), 1.0)
    completeness = 5 * (0.55 * length_quality + 0.45 * min(unique_ratio / 0.55, 1.0))
    fragmentation = 5 * max(0.0, 1 - 2.5 * short_token_rate)
    repetition = 5 * max(0.0, 1 - 6.0 * repeat_rate)
    repair = 5 * max(0.0, 1 - 1.8 * filler_rate)
    filler = 5 * max(0.0, 1 - 4.0 * filler_rate)
    grammar = float(np.clip((completeness + fragmentation) / 2, 0, 5))
    connected = float(np.clip((completeness + repetition + repair) / 3, 0, 5))
    overall = float(np.clip(np.mean([completeness, fragmentation, repetition, repair, filler, grammar, connected]), 0, 5))
    return {
        "sentence_completeness": float(completeness),
        "fragmentation_control": float(fragmentation),
        "repetition_control": float(repetition),
        "repair_restart_control": float(repair),
        "filler_vague_control": float(filler),
        "grammatical_stability": grammar,
        "connected_speech_quality": connected,
        "overall_expressive_form_quality": overall,
        "brief_form_only_reason": "Local surrogate form-only scores; no API key/cache was available.",
        "late_token_count": int(n),
        "late_unique_ratio": float(unique_ratio),
        "late_punctuation_count": int(punct_count),
    }


def _get_llm_scores(rows: list[dict[str, str]], cache_path: Path) -> tuple[dict[str, dict[str, Any]], dict]:
    cache = _read_cache(cache_path)
    missing = [r for r in rows if r["cache_key"] not in cache]
    summary = {"model": config.LATE_LLM_MODEL, "base_url": config.HUAWEI_MAAS_BASE_URL, "new_cache_rows": 0, "source": "cache"}
    if not missing:
        return cache, summary
    new_rows = []
    api_key = os.getenv(config.HUAWEI_MAAS_API_KEY_ENV, "").strip()
    if api_key:
        client = HuaweiMaaSClient(timeout=config.HUAWEI_MAAS_TIMEOUT)
        for item in missing:
            payload = {
                "model": config.LATE_LLM_MODEL,
                "messages": build_late_prompt(item["masked_text"]),
                "temperature": config.LATE_LLM_TEMPERATURE,
                "top_p": config.LATE_LLM_TOP_P,
                "max_tokens": config.LATE_LLM_MAX_TOKENS,
                "enable_thinking": False,
            }
            data = client.post_json(config.LATE_LLM_ENDPOINT, payload, timeout=config.HUAWEI_MAAS_TIMEOUT)
            content = data["choices"][0]["message"]["content"]
            parsed = sanitize_scores(parse_json_object(content))
            row = {"cache_key": item["cache_key"], "model": config.LATE_LLM_MODEL, "prompt_version": config.LATE_LLM_PROMPT_VERSION, "source": "huawei_maas_api", "masked_text": item["masked_text"], "parsed_score_json": json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), "created_at": now_utc(), "llm_parse_success": 1, "llm_error": ""}
            cache[item["cache_key"]] = row; new_rows.append(row)
        summary["source"] = "huawei_maas_api"
    elif config.ALLOW_LOCAL_SURROGATE_WITHOUT_API:
        for item in missing:
            parsed = _local_form_scores(item["masked_text"], item["raw_text"])
            row = {"cache_key": item["cache_key"], "model": f"local_form_surrogate_for_{config.LATE_LLM_MODEL}", "prompt_version": config.LATE_LLM_PROMPT_VERSION, "source": "local_surrogate_no_api_key", "masked_text": item["masked_text"], "parsed_score_json": json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), "created_at": now_utc(), "llm_parse_success": 1, "llm_error": ""}
            cache[item["cache_key"]] = row; new_rows.append(row)
        summary["source"] = "local_surrogate_no_api_key"
    else:
        raise RuntimeError(
            f"Missing {config.HUAWEI_MAAS_API_KEY_ENV} and no complete late LLM cache at {cache_path}. "
            "Set MAAS_API_KEY or unset STAGEV2_REQUIRE_API to allow local surrogate features."
        )
    _append_cache(cache_path, new_rows)
    summary["new_cache_rows"] = len(new_rows)
    return cache, summary


def _make_late_frame(df: pd.DataFrame, cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out_rows = []
    for _, row in df.iterrows():
        masked = content_mask_text(row["text"])
        key = _llm_cache_key(masked)
        parsed = json.loads(cache[key]["parsed_score_json"])
        out = {"sample_id": row["sample_id"], "masked_text": masked}
        for col in SCORE_COLUMNS:
            out[f"late_{col}"] = float(parsed.get(col, 0.0))
        out["late_form_quality_inverse"] = 5.0 - out["late_overall_expressive_form_quality"]
        out["late_token_count"] = float(parsed.get("late_token_count", max(len(tokenize(row["text"])), 1)))
        out["late_unique_ratio"] = float(parsed.get("late_unique_ratio", len(set(tokenize(row["text"]))) / max(len(tokenize(row["text"])), 1)))
        out["late_punctuation_count"] = float(parsed.get("late_punctuation_count", len(re.findall(r"[.!?]", str(row["text"])))) )
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def generate_late_features(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path = config.FEATURE_DIR / "late") -> tuple[dict[str, pd.DataFrame], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for df in [train, test]:
        for _, row in df.iterrows():
            masked = content_mask_text(row["text"])
            rows.append({"cache_key": _llm_cache_key(masked), "masked_text": masked, "raw_text": str(row["text"])})
    cache, summary = _get_llm_scores(rows, config.CACHE_DIR / "late_llm_cache.csv")
    out = {"train": _make_late_frame(train, cache), "test": _make_late_frame(test, cache)}
    for split, df in out.items():
        df.to_csv(output_dir / f"{split}_late.csv", index=False, encoding="utf-8")
    summary["train_rows"] = int(len(out["train"]))
    summary["test_rows"] = int(len(out["test"]))
    return out, summary
