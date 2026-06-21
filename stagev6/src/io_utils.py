from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd


def read_csv_robust(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
