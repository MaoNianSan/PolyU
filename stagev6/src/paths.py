from __future__ import annotations
from pathlib import Path
from . import stagev6_config as cfg

def ensure_output_dirs() -> dict[str, Path]:
    out={"root":cfg.RUN_ROOT,"tables":cfg.RUN_ROOT/"tables","predictions":cfg.RUN_ROOT/"predictions","reports":cfg.RUN_ROOT/"reports","models":cfg.RUN_ROOT/"models","models_gate":cfg.RUN_ROOT/"models"/"gate","models_branch":cfg.RUN_ROOT/"models"/"branch","models_selected":cfg.RUN_ROOT/"models"/"selected","logs":cfg.RUN_ROOT/"logs","figures":cfg.RUN_ROOT/"figures"}
    for p in out.values(): p.mkdir(parents=True,exist_ok=True)
    cfg.FINAL.mkdir(parents=True,exist_ok=True); cfg.FIGURES.mkdir(parents=True,exist_ok=True)
    return out
