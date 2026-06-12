"""Path helpers."""
from __future__ import annotations

from pathlib import Path
import config


def ensure_output_dirs() -> dict[str, Path]:
    dirs = {
        "root": config.OUTPUT_ROOT,
        "tables": config.OUTPUT_ROOT / "tables",
        "predictions": config.OUTPUT_ROOT / "predictions",
        "reports": config.OUTPUT_ROOT / "reports",
        "models": config.OUTPUT_ROOT / "models",
        "models_all": config.OUTPUT_ROOT / "models" / "all_models",
        "models_selected": config.OUTPUT_ROOT / "models" / "selected",
        "logs": config.OUTPUT_ROOT / "logs",
        "figures": config.OUTPUT_ROOT / "figures",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs
