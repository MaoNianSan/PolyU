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

import pandas as pd
import requests

# API/model settings are copied from the original project reference.
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
EMBEDDING_TIMEOUT = 120
WINDOW_SIZE_WORDS = 15
STRIDE_WORDS = 5

LATE_LLM_MODEL = "qwen3-235b-a22b"
LATE_LLM_ENDPOINT = "/chat/completions"
LATE_LLM_TEMPERATURE = 0.0
LATE_LLM_TOP_P = 1.0
LATE_LLM_MAX_TOKENS = 256
LATE_LLM_TIMEOUT = 120
LATE_LLM_ENABLE_THINKING = False
LATE_LLM_PROMPT_VERSION = "late_form_masked_v1"

TOKEN_RE = re.compile(r"[A-Za-z']+|\d+")
TOKEN_OR_PUNCT_RE = re.compile(r"[A-Za-z']+|\d+|[^\w\s]")

META_COLUMNS = [
    "sample_id", "disease_label", "mmse", "new_label",
    "label_normal", "label_mild", "label_moderate", "label_severe",
]
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
LLM_OUTPUT_COLUMNS = META_COLUMNS + [
    "masked_text",
    *SCORE_COLUMNS,
    "llm_parse_success",
    "llm_error",
    "brief_form_only_reason",
]

PERSON_WORDS = {
    "boy", "girl", "child", "children", "kid", "kids", "mother", "woman", "lady", "son", "daughter",
    "sister", "brother", "he", "she", "him", "her", "they", "them", "person", "people",
}
OBJECT_WORDS = {
    "cookie", "cookies", "jar", "stool", "chair", "sink", "water", "dish", "dishes", "cup", "plate",
    "curtain", "curtains", "window", "floor", "counter", "cabinet", "apron", "faucet", "towel",
}
PLACE_WORDS = {"kitchen", "room", "house", "home", "inside", "outside", "yard", "garden"}
ACTION_WORDS = {
    "take", "taking", "reach", "reaching", "steal", "stealing", "fall", "falling", "wash", "washing",
    "drying", "overflow", "overflowing", "spill", "spilling", "stand", "standing", "sit", "sitting",
    "climb", "climbing", "look", "looking", "see", "doing", "get", "getting",
}
MASK_LEAKAGE_WORDS = {
    "cookie", "cookies", "mother", "sink", "water", "boy", "girl", "stool", "dish", "dishes", "kitchen",
    "jar", "window", "curtain", "curtains",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text).lower())


def split_windows_words(text: str, window_size: int = WINDOW_SIZE_WORDS, stride: int = STRIDE_WORDS) -> list[dict[str, Any]]:
    tokens = tokenize(text)
    if not tokens:
        return [{"window_id": 0, "start_word_idx": 0, "end_word_idx": 0, "window_text": ""}]
    if len(tokens) <= window_size:
        return [{"window_id": 0, "start_word_idx": 0, "end_word_idx": len(tokens), "window_text": " ".join(tokens)}]
    out: list[dict[str, Any]] = []
    start = 0
    window_id = 0
    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        out.append({"window_id": window_id, "start_word_idx": start, "end_word_idx": end, "window_text": " ".join(tokens[start:end])})
        if end >= len(tokens):
            break
        start += stride
        window_id += 1
    return out


def vector_to_dim_columns(vector: list[float], prefix: str = "embedding_dim_") -> dict[str, float]:
    return {f"{prefix}{i:04d}": float(v) for i, v in enumerate(vector)}


def append_cache_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_cache_csv(path: Path, key_col: str, reuse_cache: bool) -> dict[str, dict[str, Any]]:
    if not reuse_cache or not path.exists():
        return {}
    df = pd.read_csv(path)
    if key_col not in df.columns:
        return {}
    return {str(row[key_col]): row.to_dict() for _, row in df.iterrows()}


class HuaweiMaaSClient:
    def __init__(self, timeout: int | float | None = None) -> None:
        self.api_key = os.getenv(HUAWEI_MAAS_API_KEY_ENV, "").strip()
        self.base_url = HUAWEI_MAAS_BASE_URL.rstrip("/")
        self.timeout = timeout or HUAWEI_MAAS_TIMEOUT
        self.session = requests.Session()
        self.session.trust_env = HUAWEI_MAAS_TRUST_ENV

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(f"Missing API key environment variable: {HUAWEI_MAAS_API_KEY_ENV}")
        return {HUAWEI_AUTH_HEADER: f"{HUAWEI_AUTH_PREFIX} {self.api_key}", "Content-Type": "application/json"}

    def _post_json(self, endpoint: str, payload: dict[str, Any], timeout: int | float | None = None) -> dict[str, Any]:
        url = self.base_url + endpoint
        last_error: str | None = None
        for attempt in range(1, HUAWEI_MAAS_MAX_RETRIES + 1):
            try:
                response = self.session.post(url, headers=self._headers(), json=payload, timeout=timeout or self.timeout)
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < HUAWEI_MAAS_MAX_RETRIES:
                    time.sleep(HUAWEI_MAAS_BACKOFF_SECONDS * attempt)
        raise RuntimeError(last_error or "Unknown Huawei MaaS API error")

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": EMBEDDING_MODEL, "input": texts}
        data = self._post_json(EMBEDDING_ENDPOINT, payload, timeout=EMBEDDING_TIMEOUT)
        if "data" not in data:
            raise RuntimeError(f"Embedding response missing 'data': {str(data)[:500]}")
        items = sorted(data["data"], key=lambda x: x.get("index", 0))
        embeddings = []
        for item in items:
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Embedding item missing list field 'embedding'")
            embeddings.append([float(x) for x in emb])
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding response length mismatch: expected {len(texts)}, got {len(embeddings)}")
        return embeddings

    def chat_json(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], str]:
        payload = {
            "model": LATE_LLM_MODEL,
            "messages": messages,
            "temperature": LATE_LLM_TEMPERATURE,
            "top_p": LATE_LLM_TOP_P,
            "max_tokens": LATE_LLM_MAX_TOKENS,
            "enable_thinking": LATE_LLM_ENABLE_THINKING,
        }
        data = self._post_json(LATE_LLM_ENDPOINT, payload, timeout=LATE_LLM_TIMEOUT)
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Chat response missing choices[0].message.content: {str(data)[:500]}") from exc
        return data, str(content)


def _embedding_cache_key(text: str) -> str:
    return sha256_text(EMBEDDING_MODEL + "\n" + text)


def run_middle_embeddings(preprocessed: dict[str, pd.DataFrame], embedding_dir: Path, cache_dir: Path, reuse_cache: bool = False) -> dict[str, Any]:
    embedding_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "huawei_bge_m3_embedding_cache.csv"
    cache = load_cache_csv(cache_path, "cache_key", reuse_cache=reuse_cache)

    windows_by_split: dict[str, list[dict[str, Any]]] = {}
    unique_texts: dict[str, str] = {}
    for split, df in preprocessed.items():
        rows = []
        for _, row in df.iterrows():
            for win in split_windows_words(str(row["text"])):
                key = _embedding_cache_key(win["window_text"])
                item = {
                    "sample_id": row["sample_id"],
                    "window_id": win["window_id"],
                    "start_word_idx": win["start_word_idx"],
                    "end_word_idx": win["end_word_idx"],
                    "window_text": win["window_text"],
                    "cache_key": key,
                }
                rows.append(item)
                if key not in cache and key not in unique_texts:
                    unique_texts[key] = win["window_text"]
        windows_by_split[split] = rows

    new_cache_rows: list[dict[str, Any]] = []
    if unique_texts:
        client = HuaweiMaaSClient(timeout=EMBEDDING_TIMEOUT)
        keys = list(unique_texts.keys())
        texts = [unique_texts[k] for k in keys]
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_keys = keys[start:start + EMBEDDING_BATCH_SIZE]
            batch_texts = texts[start:start + EMBEDDING_BATCH_SIZE]
            embeddings = client.embed(batch_texts)
            for key, text, embedding in zip(batch_keys, batch_texts, embeddings):
                row = {
                    "cache_key": key,
                    "model": EMBEDDING_MODEL,
                    "text_hash": sha256_text(text),
                    "window_text": text,
                    "embedding_json": json.dumps(embedding, separators=(",", ":")),
                    "created_at": now_utc(),
                    "embedding_success": 1,
                    "embedding_error": "",
                }
                cache[key] = row
                new_cache_rows.append(row)
        append_cache_rows(cache_path, new_cache_rows, [
            "cache_key", "model", "text_hash", "window_text", "embedding_json", "created_at", "embedding_success", "embedding_error",
        ])

    summary = {"model": EMBEDDING_MODEL, "base_url": HUAWEI_MAAS_BASE_URL, "new_cache_rows": len(new_cache_rows), "datasets": {}}
    for split, rows in windows_by_split.items():
        out_rows: list[dict[str, Any]] = []
        for item in rows:
            cache_row = cache.get(item["cache_key"], {})
            base = {k: item[k] for k in ["sample_id", "window_id", "start_word_idx", "end_word_idx", "window_text"]}
            raw = cache_row.get("embedding_json", "")
            if raw:
                emb = json.loads(str(raw))
                base["embedding_success"] = 1
                base["embedding_error"] = ""
                base.update(vector_to_dim_columns(emb))
            else:
                base["embedding_success"] = 0
                base["embedding_error"] = str(cache_row.get("embedding_error", "embedding unavailable"))[:500]
            out_rows.append(base)
        out_df = pd.DataFrame(out_rows)
        if not any(c.startswith("embedding_dim_") for c in out_df.columns):
            raise RuntimeError(f"No BGE-M3 embedding dimensions generated for split={split}.")
        out_path = embedding_dir / {"ad": "ad_embedding.csv", "control": "control_embedding.csv", "test": "test_embedding.csv"}[split]
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        summary["datasets"][split] = {"output_path": str(out_path), "rows": int(len(out_df))}
    return summary


def content_mask_text(text: Any) -> str:
    tokens = TOKEN_OR_PUNCT_RE.findall(normalize_text(text))
    masked: list[str] = []
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
    return sha256_text(LATE_LLM_MODEL + "\n" + LATE_LLM_PROMPT_VERSION + "\n" + masked_text)


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
- For fragmentation_control, repetition_control, repair_restart_control, and filler_vague_control, higher means better control.
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


def parse_strict_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError("Could not parse a JSON object from LLM response")


def sanitize_reason(reason: Any) -> str:
    text = normalize_text(reason)
    banned = ["ad", "alzheimer", "dementia", "mmse", "diagnosis", "diagnose", "risk", "patient"]
    words = text.split()
    if len(words) > 30:
        text = " ".join(words[:30])
    low = text.lower()
    if any(re.search(rf"\b{re.escape(w)}\b", low) for w in banned):
        return "Form-only scoring completed; reason removed because it contained restricted medical wording."
    return text


def parse_scores(content: str) -> dict[str, Any]:
    obj = parse_strict_json_object(content)
    parsed: dict[str, Any] = {}
    for col in SCORE_COLUMNS:
        val = float(obj.get(col, 0.0))
        parsed[col] = max(0.0, min(5.0, val))
    parsed["brief_form_only_reason"] = sanitize_reason(obj.get("brief_form_only_reason", ""))
    return parsed


def run_late_llm(preprocessed: dict[str, pd.DataFrame], llm_dir: Path, cache_dir: Path, reuse_cache: bool = False) -> dict[str, Any]:
    llm_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "late_llm_cache.csv"
    cache = load_cache_csv(cache_path, "cache_key", reuse_cache=reuse_cache)

    all_masked: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    for split, df in preprocessed.items():
        work = df.copy()
        work["masked_text"] = work["text"].map(content_mask_text)
        work["llm_cache_key"] = work["masked_text"].map(_llm_cache_key)
        frames[split] = work
        for _, row in work.iterrows():
            key = str(row["llm_cache_key"])
            if key not in cache:
                all_masked[key] = str(row["masked_text"])

    new_rows: list[dict[str, Any]] = []
    if all_masked:
        client = HuaweiMaaSClient(timeout=LATE_LLM_TIMEOUT)
        for key, masked_text in all_masked.items():
            raw_response, content = client.chat_json(build_late_prompt(masked_text))
            parsed = parse_scores(content)
            row = {
                "cache_key": key,
                "model": LATE_LLM_MODEL,
                "prompt_version": LATE_LLM_PROMPT_VERSION,
                "masked_text_hash": sha256_text(masked_text),
                "masked_text": masked_text,
                "raw_response_json": json.dumps(raw_response, ensure_ascii=False),
                "parsed_score_json": json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
                "created_at": now_utc(),
                "llm_parse_success": 1,
                "llm_error": "",
            }
            cache[key] = row
            new_rows.append(row)
        append_cache_rows(cache_path, new_rows, [
            "cache_key", "model", "prompt_version", "masked_text_hash", "masked_text", "raw_response_json",
            "parsed_score_json", "created_at", "llm_parse_success", "llm_error",
        ])

    summary = {"model": LATE_LLM_MODEL, "base_url": HUAWEI_MAAS_BASE_URL, "new_cache_rows": len(new_rows), "datasets": {}}
    for split, df in frames.items():
        out_rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            cache_row = cache.get(str(row["llm_cache_key"]), {})
            raw = cache_row.get("parsed_score_json", "")
            if not raw:
                raise RuntimeError(f"No late LLM score generated for sample_id={row['sample_id']}")
            parsed = json.loads(str(raw))
            out = {c: row[c] for c in META_COLUMNS}
            out["masked_text"] = row["masked_text"]
            for col in SCORE_COLUMNS:
                out[col] = parsed.get(col, pd.NA)
            out["llm_parse_success"] = 1
            out["llm_error"] = ""
            out["brief_form_only_reason"] = parsed.get("brief_form_only_reason", "")
            out_rows.append(out)
        out_df = pd.DataFrame(out_rows)[LLM_OUTPUT_COLUMNS]
        out_path = llm_dir / {"ad": "ad_LLM.csv", "control": "control_LLM.csv", "test": "test_LLM.csv"}[split]
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        leakage_count = int(out_df["masked_text"].fillna("").astype(str).str.lower().map(lambda x: any(re.search(rf"\b{re.escape(w)}\b", x) for w in MASK_LEAKAGE_WORDS)).sum())
        summary["datasets"][split] = {"output_path": str(out_path), "rows": int(len(out_df)), "masked_text_leakage_row_count": leakage_count}
    return summary
