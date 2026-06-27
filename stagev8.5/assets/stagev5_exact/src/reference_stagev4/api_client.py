from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from . import config


@dataclass
class MaaSError(RuntimeError):
    message: str
    status_code: int | None = None
    def __str__(self) -> str:
        return self.message


class HuaweiMaaSClient:
    """Minimal Huawei MaaS chat client for the P3 late scorer.

    It never changes a transcript, mask, or prompt after a provider failure.
    A 403/content-safety response is surfaced as an error rather than silently
    falling back to another mask variant.
    """
    def __init__(self) -> None:
        self.api_key = os.getenv(config.MAAS_API_KEY_ENV, "").strip()
        self.session = requests.Session()
        self.session.trust_env = config.HUAWEI_MAAS_TRUST_ENV

    def has_key(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise MaaSError(f"Missing environment variable {config.MAAS_API_KEY_ENV}.")
        url = config.HUAWEI_MAAS_BASE_URL + config.LATE_LLM_ENDPOINT
        payload: dict[str, Any] = {
            "model": config.LATE_LLM_MODEL,
            "messages": messages,
            "temperature": config.LATE_LLM_TEMPERATURE,
            "top_p": config.LATE_LLM_TOP_P,
            "max_tokens": config.LATE_LLM_MAX_TOKENS,
            "enable_thinking": False,
        }
        last: Exception | None = None
        for attempt in range(1, config.HUAWEI_MAAS_MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    url, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload, timeout=config.HUAWEI_MAAS_TIMEOUT, verify=config.HUAWEI_MAAS_SSL_VERIFY,
                )
                if response.status_code >= 400:
                    detail = response.text.replace("\n", " ")[:600]
                    raise MaaSError(f"Huawei MaaS HTTP {response.status_code}: {detail}", response.status_code)
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if content is None:
                    raise MaaSError(f"Chat response lacks choices[0].message.content: {str(data)[:500]}")
                return str(content)
            except (requests.RequestException, ValueError, MaaSError) as exc:
                last = exc
                # 4xx except 429 is not transient.
                if isinstance(exc, MaaSError) and exc.status_code is not None and exc.status_code < 500 and exc.status_code != 429:
                    raise
                if attempt < config.HUAWEI_MAAS_MAX_RETRIES:
                    time.sleep(config.HUAWEI_MAAS_BACKOFF_SECONDS * attempt)
        raise MaaSError(f"Huawei MaaS request failed after {config.HUAWEI_MAAS_MAX_RETRIES} attempts: {last}")
