from __future__ import annotations

"""Windows Schannel transport shim for fresh MaaS extraction.

This module intentionally does not alter any copied Stagev5/Stagev4 source file.
It replaces only the runtime HTTPS transport when CPython requests/OpenSSL cannot
complete the provider's TLS exchange on Windows. The original feature functions,
text handling, windowing, cache keys, payload dictionaries, parsing, scoring and
feature writing remain the copied reference implementation.

The shim prepares the same POST body that ``requests`` would prepare for
``json=payload`` and sends those bytes via ``curl.exe`` (Windows Schannel).
No request payloads, transcripts, API keys, or response bodies are written to
Stagev8 logs. The temporary curl configuration contains the Authorization header
and is removed immediately after each request.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


_TRANSPORT_AUDIT: dict[str, Any] = {
    "transport_mode": "requests",
    "installed": False,
    "curl_executable": None,
    "request_count": 0,
    "direct_connection": None,
    "tls_backend": None,
    "reference_sources_modified": False,
    "request_body_policy": "not_applicable",
}


def _quote_config(value: str) -> str:
    """Quote a curl config-file value without exposing it in the process argv."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "") + '"'


@dataclass
class CurlResponse:
    status_code: int
    text: str
    headers: dict[str, str]

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class CurlSchannelSession:
    """Small Session-compatible object used only by copied source clients."""

    def __init__(self, curl_executable: str) -> None:
        self.curl_executable = curl_executable
        self.trust_env = False
        # A plain requests Session is used only to reproduce requests' JSON body
        # preparation. It never performs network I/O.
        self._preparer = requests.Session()

    @staticmethod
    def _body_bytes(prepared: requests.PreparedRequest) -> bytes:
        body = prepared.body
        if body is None:
            return b""
        if isinstance(body, bytes):
            return body
        if isinstance(body, str):
            return body.encode("utf-8")
        raise TypeError(f"Unsupported prepared request body type: {type(body)!r}")

    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: int | float | None = None,
        verify: bool | str = True,
        **_: Any,
    ) -> CurlResponse:
        request = requests.Request(method="POST", url=url, headers=headers or {}, json=json)
        prepared = self._preparer.prepare_request(request)
        body = self._body_bytes(prepared)
        effective_timeout = float(timeout or 120.0)
        connect_timeout = min(20.0, effective_timeout)

        with tempfile.TemporaryDirectory(prefix="stagev8_curl_") as tmp:
            tmp_path = Path(tmp)
            response_path = tmp_path / "response.json"
            config_path = tmp_path / "curl.conf"
            config_lines = [
                "silent",
                "show-error",
                "noproxy = \"*\"",
                "request = \"POST\"",
                f"url = {_quote_config(str(url))}",
                f"output = {_quote_config(response_path.as_posix())}",
                "write-out = \"%{http_code}\"",
                f"connect-timeout = {_quote_config(str(int(connect_timeout)))}",
                f"max-time = {_quote_config(str(max(1, int(effective_timeout))))}",
            ]
            # Preserve the copied source's explicit caller headers. This includes
            # the Authorization and Content-Type values. Do not pass them via argv.
            for key, value in (headers or {}).items():
                config_lines.append(f"header = {_quote_config(f'{key}: {value}')}")
            # Keep certificate verification enabled by default. The only existing
            # caller setting is True; support False transparently for completeness.
            if verify is False:
                config_lines.append("insecure")
            config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
            try:
                completed = subprocess.run(
                    [self.curl_executable, "--config", str(config_path), "--data-binary", "@-"],
                    input=body,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=effective_timeout + 30.0,
                )
            except subprocess.TimeoutExpired as exc:
                raise requests.Timeout(f"curl.exe transport timed out after {effective_timeout}s") from exc
            except OSError as exc:
                raise requests.ConnectionError(f"curl.exe transport could not start: {exc}") from exc

            stdout = completed.stdout.decode("utf-8", errors="replace").strip()
            match = re.search(r"(\d{3})\s*$", stdout)
            body_text = response_path.read_text(encoding="utf-8", errors="replace") if response_path.exists() else ""
            if completed.returncode != 0 or match is None:
                stderr = completed.stderr.decode("utf-8", errors="replace").strip()
                detail = stderr or stdout or "curl.exe did not return an HTTP status"
                raise requests.ConnectionError(f"curl.exe Schannel transport failed: {detail[:800]}")
            _TRANSPORT_AUDIT["request_count"] = int(_TRANSPORT_AUDIT.get("request_count", 0)) + 1
            return CurlResponse(status_code=int(match.group(1)), text=body_text, headers={})


def install_curl_schannel_transport() -> dict[str, Any]:
    """Install an external Windows-only transport override without editing reference code."""
    if _TRANSPORT_AUDIT.get("installed"):
        return dict(_TRANSPORT_AUDIT)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError(
            "Fresh Stagev5 extraction requires curl.exe for the Windows Schannel transport, "
            "but curl.exe was not found on PATH."
        )

    # Import here to avoid affecting regular self-check work.
    from stagev5_exact.src.reference_stagev2 import api_feature_extraction as stagev2_api
    from stagev5_exact.src.reference_stagev4 import api_client as stagev4_api
    from stagev5_exact.src.reference_stagev4 import late_extract as stagev4_late

    stagev2_original = stagev2_api.HuaweiMaaSClient
    stagev4_original = stagev4_api.HuaweiMaaSClient

    class Stagev2CurlMaaSClient(stagev2_original):
        def __init__(self, timeout: int | float | None = None) -> None:
            super().__init__(timeout=timeout)
            self.session = CurlSchannelSession(curl)

    class Stagev4CurlMaaSClient(stagev4_original):
        def __init__(self) -> None:
            super().__init__()
            self.session = CurlSchannelSession(curl)

    # Patch only the runtime class names resolved by the copied entrypoints.
    stagev2_api.HuaweiMaaSClient = Stagev2CurlMaaSClient
    stagev4_api.HuaweiMaaSClient = Stagev4CurlMaaSClient
    stagev4_late.HuaweiMaaSClient = Stagev4CurlMaaSClient

    # Fail closed if the module identity patched here is not the same namespace
    # used by the copied Stagev5/Stagev4 entrypoints.  The previous package
    # accidentally patched an ``assets.stagev5_exact`` namespace while the
    # extraction entrypoint used ``stagev5_exact``; this guard makes a silent
    # fallback to requests/OpenSSL impossible.
    stagev2_global = stagev2_api.run_middle_embeddings.__globals__.get("HuaweiMaaSClient")
    stagev4_global = stagev4_late.extract_late_scores.__globals__.get("HuaweiMaaSClient")
    if stagev2_global is not Stagev2CurlMaaSClient or stagev4_global is not Stagev4CurlMaaSClient:
        raise RuntimeError(
            "Windows Schannel transport shim was not installed into the exact module namespace "
            "used by copied Stagev5/Stagev4 extraction; refusing to fall back to requests/OpenSSL."
        )
    probe_middle = Stagev2CurlMaaSClient()
    probe_late = Stagev4CurlMaaSClient()
    if not isinstance(probe_middle.session, CurlSchannelSession) or not isinstance(probe_late.session, CurlSchannelSession):
        raise RuntimeError("Windows Schannel transport shim session verification failed.")

    _TRANSPORT_AUDIT.update(
        {
            "transport_mode": "curl_schannel_runtime_shim",
            "installed": True,
            "curl_executable": str(Path(curl).resolve()),
            "direct_connection": True,
            "tls_backend": "Windows Schannel via curl.exe",
            "reference_sources_modified": False,
            "request_body_policy": "bytes prepared from the copied request json=payload path; transport only",
            "patched_module_namespace": "stagev5_exact.src.reference_stagev2 / stagev5_exact.src.reference_stagev4",
            "stagev2_runtime_class": f"{Stagev2CurlMaaSClient.__module__}.{Stagev2CurlMaaSClient.__name__}",
            "stagev4_runtime_class": f"{Stagev4CurlMaaSClient.__module__}.{Stagev4CurlMaaSClient.__name__}",
            "runtime_patch_verified": True,
        }
    )
    return dict(_TRANSPORT_AUDIT)


def transport_audit() -> dict[str, Any]:
    return dict(_TRANSPORT_AUDIT)
