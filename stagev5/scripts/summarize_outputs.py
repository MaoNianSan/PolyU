"""Print a compact read-only inventory of GitHub-facing stagev5 outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import stagev5_config as cfg


def _file_info(path: Path) -> dict[str, object]:
    rel = path.relative_to(cfg.ROOT).as_posix()
    info: dict[str, object] = {"path": rel, "bytes": path.stat().st_size}
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
            info.update({"rows": int(len(frame)), "columns": int(len(frame.columns))})
        except Exception as exc:  # pragma: no cover - CLI diagnostic path
            info["read_error"] = repr(exc)
    return info


def main() -> int:
    patterns = ["*.md", "*.csv", "*.json", "figures/*.png"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(cfg.FINAL.glob(pattern)))
    summary = {
        "project_root": str(cfg.ROOT),
        "final_report_dir": str(cfg.FINAL),
        "file_count": len(files),
        "files": [_file_info(path) for path in files],
        "policy": {
            "trained_models": False,
            "extracted_features": False,
            "called_api": False,
            "modified_existing_results": False,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
