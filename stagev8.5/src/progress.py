from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enabled() -> bool:
    value = os.getenv("STAGEV8_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


@dataclass
class StageProgress:
    """Small dependency-free console/file progress reporter.

    It is intentionally not a rich UI dependency. It works in PowerShell, CMD,
    VS Code terminals, and redirected logs. No output directory is created until
    a command actually runs.
    """

    task: str
    total: int | None = None
    root: Path | None = None
    jsonl_name: str = "stagev8_runtime_progress.jsonl"
    start_time: float = field(default_factory=time.time)
    current: int = 0
    enabled: bool = field(default_factory=_enabled)

    def __post_init__(self) -> None:
        self.log_path: Path | None = None
        if self.root is not None:
            out = self.root / "output"
            out.mkdir(parents=True, exist_ok=True)
            self.log_path = out / self.jsonl_name
        self.event("start", f"{self.task} started")

    def event(self, phase: str, message: str, **extra: Any) -> None:
        elapsed = time.time() - self.start_time
        payload: dict[str, Any] = {
            "time": utc(),
            "pid": os.getpid(),
            "task": self.task,
            "phase": phase,
            "step": self.current,
            "total": self.total,
            "elapsed_sec": round(elapsed, 1),
            "message": message,
        }
        payload.update(extra)
        if self.enabled:
            prefix = f"[{self.task}]"
            if self.total:
                prefix += f" {self.current}/{self.total}"
            print(f"{prefix} {phase}: {message} (elapsed={elapsed:.1f}s)", flush=True)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def step(self, message: str, **extra: Any) -> None:
        self.current += 1
        self.event("step", message, **extra)

    def done(self, message: str = "completed", **extra: Any) -> None:
        self.event("done", message, **extra)

    def fail(self, message: str, **extra: Any) -> None:
        self.event("fail", message, **extra)


class progress_section:
    """Context manager that logs failure while preserving the original exception."""

    def __init__(self, progress: StageProgress, message: str, **extra: Any) -> None:
        self.progress = progress
        self.message = message
        self.extra = extra

    def __enter__(self) -> "progress_section":
        self.progress.step(self.message, **self.extra)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if exc is not None:
            self.progress.fail(f"failed during: {self.message}", error=repr(exc))
        return False
