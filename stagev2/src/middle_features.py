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
import requests
from sklearn.feature_extraction.text import HashingVectorizer

from . import config

TOKEN_RE = re.compile(r"[A-Za-z']+|\d+")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\r", " ").replace("\n", " ")).strip()


def tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text).lower())


def split_windows_words(text: str, window_size: int = config.WINDOW_SIZE_WORDS, stride: int = config.STRIDE_WORDS) -> list[str]:
    tokens = tokenize(text)
    if not tokens:
        return [""]
    if len(tokens) <= window_size:
        return [" ".join(tokens)]
    windows = []
    start = 0
    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        windows.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
        start += stride
    return windows


def _embedding_cache_key(text: str) -> str:
    return sha256_text(config.EMBEDDING_MODEL + "\n" + text)


class HuaweiMaaSClient:
    def __init__(self, timeout: int | float | None = None) -> None:
        self.api_key = os.getenv(config.HUAWEI_MAAS_API_KEY_ENV, "").strip()
        self.base_url = config.HUAWEI_MAAS_BASE_URL.rstrip("/")
        self.timeout = timeout or config.HUAWEI_MAAS_TIMEOUT
        self.session = requests.Session()
        self.session.trust_env = config.HUAWEI_MAAS_TRUST_ENV

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(f"Missing API key environment variable: {config.HUAWEI_MAAS_API_KEY_ENV}")
        return {config.HUAWEI_AUTH_HEADER: f"{config.HUAWEI_AUTH_PREFIX} {self.api_key}", "Content-Type": "application/json"}

    def post_json(self, endpoint: str, payload: dict[str, Any], timeout: int | float | None = None) -> dict[str, Any]:
        url = self.base_url + endpoint
        last_error = None
        for attempt in range(1, config.HUAWEI_MAAS_MAX_RETRIES + 1):
            try:
                response = self.session.post(url, headers=self._headers(), json=payload, timeout=timeout or self.timeout)
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                return response.json()
            except Exception as exc:
                last_error = str(exc)
                if attempt < config.HUAWEI_MAAS_MAX_RETRIES:
                    time.sleep(config.HUAWEI_MAAS_BACKOFF_SECONDS * attempt)
        raise RuntimeError(last_error or "Unknown Huawei MaaS API error")

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": config.EMBEDDING_MODEL, "input": texts}
        data = self.post_json(config.EMBEDDING_ENDPOINT, payload, timeout=config.HUAWEI_MAAS_TIMEOUT)
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        embeddings = [[float(v) for v in item["embedding"]] for item in items]
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding length mismatch: expected {len(texts)}, got {len(embeddings)}")
        return embeddings


def _read_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    out = {}
    with cache_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[str(row.get("cache_key", ""))] = row
    return out


def _append_cache(cache_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    exists = cache_path.exists() and cache_path.stat().st_size > 0
    fields = ["cache_key", "model", "source", "window_text", "embedding_json", "created_at", "embedding_success", "embedding_error"]
    with cache_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _local_surrogate_embeddings(texts: list[str]) -> list[list[float]]:
    vectorizer = HashingVectorizer(
        n_features=config.EMBEDDING_DIM_LOCAL_FALLBACK,
        alternate_sign=False,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.transform(texts).astype(float).toarray()
    return matrix.tolist()


def _get_embeddings(unique_texts: dict[str, str], cache_path: Path) -> tuple[dict[str, dict[str, Any]], dict]:
    cache = _read_cache(cache_path)
    missing = {k: v for k, v in unique_texts.items() if k not in cache}
    summary = {"model": config.EMBEDDING_MODEL, "base_url": config.HUAWEI_MAAS_BASE_URL, "new_cache_rows": 0, "source": "cache"}
    if not missing:
        return cache, summary

    api_key = os.getenv(config.HUAWEI_MAAS_API_KEY_ENV, "").strip()
    rows = []
    if api_key:
        client = HuaweiMaaSClient()
        keys = list(missing.keys())
        texts = [missing[k] for k in keys]
        for start in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
            bkeys = keys[start:start + config.EMBEDDING_BATCH_SIZE]
            btexts = texts[start:start + config.EMBEDDING_BATCH_SIZE]
            embeddings = client.embed(btexts)
            for key, text, emb in zip(bkeys, btexts, embeddings):
                row = {
                    "cache_key": key, "model": config.EMBEDDING_MODEL, "source": "huawei_maas_api",
                    "window_text": text, "embedding_json": json.dumps(emb, separators=(",", ":")),
                    "created_at": now_utc(), "embedding_success": 1, "embedding_error": "",
                }
                cache[key] = row; rows.append(row)
        summary["source"] = "huawei_maas_api"
    elif config.ALLOW_LOCAL_SURROGATE_WITHOUT_API:
        keys = list(missing.keys())
        texts = [missing[k] for k in keys]
        embeddings = _local_surrogate_embeddings(texts)
        for key, text, emb in zip(keys, texts, embeddings):
            row = {
                "cache_key": key, "model": f"local_hashing_surrogate_for_{config.EMBEDDING_MODEL}", "source": "local_surrogate_no_api_key",
                "window_text": text, "embedding_json": json.dumps(emb, separators=(",", ":")),
                "created_at": now_utc(), "embedding_success": 1, "embedding_error": "",
            }
            cache[key] = row; rows.append(row)
        summary["source"] = "local_surrogate_no_api_key"
    else:
        raise RuntimeError(
            f"Missing {config.HUAWEI_MAAS_API_KEY_ENV} and no complete embedding cache at {cache_path}. "
            "Set MAAS_API_KEY or unset STAGEV2_REQUIRE_API to allow local surrogate features."
        )
    _append_cache(cache_path, rows)
    summary["new_cache_rows"] = len(rows)
    return cache, summary


def _aggregate_sample_embeddings(sample_ids: list[str], windows_by_sample: dict[str, list[str]], cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for sid in sample_ids:
        wins = windows_by_sample[sid]
        vectors = []
        for w in wins:
            row = cache[_embedding_cache_key(w)]
            vectors.append(np.asarray(json.loads(row["embedding_json"]), dtype=float))
        mat = np.vstack(vectors)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        if len(vectors) > 1:
            diffs = np.diff(mat, axis=0)
            drift_mean = float(np.mean(np.linalg.norm(diffs, axis=1)))
            drift_max = float(np.max(np.linalg.norm(diffs, axis=1)))
        else:
            drift_mean = 0.0
            drift_max = 0.0
        out = {"sample_id": sid, "middle_n_windows": len(wins), "middle_embedding_drift_mean": drift_mean, "middle_embedding_drift_max": drift_max}
        for i, v in enumerate(mean):
            out[f"middle_embed_mean_{i:03d}"] = float(v)
        for i, v in enumerate(std):
            out[f"middle_embed_std_{i:03d}"] = float(v)
        rows.append(out)
    return pd.DataFrame(rows)


def generate_middle_features(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path = config.FEATURE_DIR / "middle") -> tuple[dict[str, pd.DataFrame], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.CACHE_DIR / "huawei_bge_m3_embedding_cache.csv"
    frames = {"train": train, "test": test}
    windows_by_split = {}
    unique = {}
    for split, df in frames.items():
        by_sample = {}
        for _, row in df.iterrows():
            wins = split_windows_words(row["text"])
            by_sample[str(row["sample_id"])] = wins
            for w in wins:
                unique[_embedding_cache_key(w)] = w
        windows_by_split[split] = by_sample
    cache, summary = _get_embeddings(unique, cache_path)
    out = {}
    for split, df in frames.items():
        feats = _aggregate_sample_embeddings(df["sample_id"].astype(str).tolist(), windows_by_split[split], cache)
        feats.to_csv(output_dir / f"{split}_middle.csv", index=False, encoding="utf-8")
        out[split] = feats
    summary["train_rows"] = int(len(out["train"]))
    summary["test_rows"] = int(len(out["test"]))
    return out, summary
