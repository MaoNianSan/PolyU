from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests import Response

from . import config

_DIRECT_CONNECTION_REQUIRED = False


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mask_secret(value: str, left: int = 4, right: int = 3) -> str:
    if not value:
        return ""
    if len(value) <= left + right:
        return "***"
    return value[:left] + "..." + value[-right:]


class MaaSAPIError(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        status_code: int | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.endpoint = endpoint


def _http_error_kind(status_code: int) -> str:
    if status_code == 401:
        return "unauthorized / 401"
    if status_code == 403:
        return "forbidden / 403"
    if status_code == 404:
        return "endpoint not found / 404"
    if status_code == 429:
        return "rate limit / 429"
    return "other HTTP error"


class HuaweiMaaSClient:
    """Huawei MaaS-compatible client, rewritten for stagev3.

    It follows the original stage2 convention: MAAS_API_KEY, /embeddings, /chat/completions,
    Bearer authorization, retries, cache-first execution.
    """
    def __init__(
        self,
        timeout: int | float | None = None,
        max_retries: int | None = None,
    ) -> None:
        global _DIRECT_CONNECTION_REQUIRED
        api_key = os.getenv(config.MAAS_API_KEY_ENV)
        if api_key is not None:
            api_key = api_key.strip()
        self.api_key = api_key or ""
        self.base_url = config.HUAWEI_MAAS_BASE_URL.rstrip("/")
        self.timeout = config.HUAWEI_MAAS_TIMEOUT if timeout is None else timeout
        self.max_retries = config.HUAWEI_MAAS_MAX_RETRIES if max_retries is None else max_retries
        self.configured_trust_env = config.HUAWEI_MAAS_TRUST_ENV
        self.direct_fallback_used = _DIRECT_CONNECTION_REQUIRED
        self.session = requests.Session()
        self.session.trust_env = self.configured_trust_env and not _DIRECT_CONNECTION_REQUIRED

    def has_key(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise MaaSAPIError("missing key", f"Missing API key environment variable: {config.MAAS_API_KEY_ENV}")
        return {config.HUAWEI_AUTH_HEADER: f"{config.HUAWEI_AUTH_PREFIX} {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _raise_for_status(response: Response, endpoint: str) -> None:
        if response.status_code < 400:
            return
        kind = _http_error_kind(response.status_code)
        response_detail = response.text.strip().replace("\r", " ").replace("\n", " ")
        if response.status_code == 403 and "ModelArts.81011" in response_detail:
            kind = "content safety / 403"
        if len(response_detail) > 500:
            response_detail = response_detail[:497] + "..."
        message = f"{kind} (HTTP {response.status_code}) at {endpoint}"
        if response_detail:
            message += f": {response_detail}"
        raise MaaSAPIError(kind, message, response.status_code, endpoint)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        global _DIRECT_CONNECTION_REQUIRED
        url = self.base_url + endpoint
        last_error: MaaSAPIError | None = None
        allow_direct_fallback = self.session.trust_env
        routes = [True, False] if allow_direct_fallback else [False]
        for use_environment in routes:
            self.session.trust_env = use_environment
            route_retries = self.max_retries
            route_network_failed = False
            for attempt in range(1, route_retries + 1):
                network_route_failed = False
                try:
                    response = self.session.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                        timeout=min(self.timeout, 30.0) if use_environment else self.timeout,
                        verify=config.HUAWEI_MAAS_SSL_VERIFY,
                    )
                    self._raise_for_status(response, endpoint)
                    data = response.json()
                    if not use_environment and allow_direct_fallback:
                        _DIRECT_CONNECTION_REQUIRED = True
                        self.direct_fallback_used = True
                        print(
                            "[Huawei MaaS] Environment proxy route failed; "
                            "using a direct connection for this process.",
                            flush=True,
                        )
                    return data
                except requests.exceptions.SSLError as exc:
                    last_error = MaaSAPIError(
                        "SSL error",
                        f"SSL error at {endpoint}: {type(exc).__name__}: {exc}",
                        endpoint=endpoint,
                    )
                    network_route_failed = True
                    route_network_failed = True
                except requests.exceptions.Timeout as exc:
                    last_error = MaaSAPIError(
                        "timeout",
                        f"timeout at {endpoint}: {type(exc).__name__}: {exc}",
                        endpoint=endpoint,
                    )
                    network_route_failed = True
                    route_network_failed = True
                except MaaSAPIError as exc:
                    last_error = exc
                    retryable_http_error = exc.status_code == 429
                    if exc.status_code is not None and exc.status_code < 500 and not retryable_http_error:
                        raise
                except requests.exceptions.RequestException as exc:
                    last_error = MaaSAPIError(
                        "other error",
                        f"network error at {endpoint}: {type(exc).__name__}: {exc}",
                        endpoint=endpoint,
                    )
                    network_route_failed = True
                    route_network_failed = True
                except (ValueError, json.JSONDecodeError) as exc:
                    raise MaaSAPIError(
                        "other error",
                        f"invalid JSON response at {endpoint}: {type(exc).__name__}",
                        endpoint=endpoint,
                    ) from exc
                if network_route_failed and use_environment and allow_direct_fallback:
                    break
                if attempt < route_retries:
                    time.sleep(config.HUAWEI_MAAS_BACKOFF_SECONDS * attempt)
            if not (use_environment and allow_direct_fallback and route_network_failed):
                break
        raise last_error or MaaSAPIError("unknown error", "Unknown Huawei MaaS API error")

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": config.EMBEDDING_MODEL, "input": texts}
        data = self._post_json(config.EMBEDDING_ENDPOINT, payload)
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        out = []
        for item in items:
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise MaaSAPIError("other error", "Embedding response item is missing an embedding vector.")
            out.append([float(x) for x in emb])
        if len(out) != len(texts):
            raise MaaSAPIError(
                "other error",
                f"Embedding response length mismatch: expected {len(texts)}, got {len(out)}",
            )
        return out

    def chat_json(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], str]:
        payload = {
            "model": config.LATE_LLM_MODEL,
            "messages": messages,
            "temperature": config.LATE_LLM_TEMPERATURE,
            "top_p": config.LATE_LLM_TOP_P,
            "max_tokens": config.LATE_LLM_MAX_TOKENS,
            "enable_thinking": False,
        }
        data = self._post_json(config.LATE_LLM_ENDPOINT, payload)
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if content is None:
            raise RuntimeError(f"Chat response missing choices[0].message.content: {str(data)[:500]}")
        return data, str(content)



def remove_surrogate_cache_rows(path: Path, source_col: str = "source") -> int:
    """Remove previously generated local-surrogate cache rows.

    stagev3 now uses only real API responses or previously cached real-API
    responses. Historical surrogate rows are not valid inputs for formal runs.
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        path.unlink(missing_ok=True)
        return -1
    if source_col not in df.columns:
        return 0
    mask = df[source_col].astype(str).str.strip().str.lower().isin({"local_surrogate", "surrogate"})
    removed = int(mask.sum())
    if removed:
        kept = df.loc[~mask].copy()
        if kept.empty:
            path.unlink(missing_ok=True)
        else:
            kept.to_csv(path, index=False, encoding="utf-8")
    return removed

def load_cache(path: Path, key_col: str) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    if key_col not in df.columns:
        return {}
    return {str(row[key_col]): row.to_dict() for _, row in df.iterrows()}


def append_cache(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    df = pd.DataFrame(rows)
    for f in fieldnames:
        if f not in df.columns:
            df[f] = ""
    df[fieldnames].to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8")
