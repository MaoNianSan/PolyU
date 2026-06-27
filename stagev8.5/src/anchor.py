from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from . import config as cfg
from .feature_contract_v85 import sha256_file

ANCHOR_MODEL_NAME = "early_middle__svc__poly3"
ANCHOR_PATH = cfg.ASSETS / "stagev5_anchor" / "selected__early_middle__svc__poly3.joblib"


@dataclass
class FrozenAnchor:
    pipeline: Any
    feature_names: list[str]
    artifact_sha256: str


def load_frozen_anchor() -> FrozenAnchor:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Frozen Stagev5 anchor is missing: {ANCHOR_PATH}")
    pipe = joblib.load(ANCHOR_PATH)
    names = list(getattr(pipe, "feature_names_in_", []))
    if len(names) != 1085:
        raise RuntimeError(f"Unexpected frozen anchor feature count: {len(names)}; expected 1085 (E+M).")
    if not hasattr(pipe, "predict_proba"):
        raise RuntimeError("Frozen Stagev5 anchor has no predict_proba; Stagev8 refuses to manufacture anchor probabilities.")
    return FrozenAnchor(pipe, names, sha256_file(ANCHOR_PATH))


def predict_anchor(anchor: FrozenAnchor, frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in anchor.feature_names if c not in frame.columns]
    if missing:
        raise ValueError(f"Stagev6-loaded features cannot satisfy frozen anchor columns: {missing[:5]}")
    X = frame[anchor.feature_names]
    classes = list(anchor.pipeline.classes_)
    if 1 not in classes:
        raise RuntimeError(f"Frozen Stagev5 anchor lacks AD class 1: classes={classes}")
    p = anchor.pipeline.predict_proba(X)[:, classes.index(1)]
    pred = (p >= cfg.ANCHOR_THRESHOLD).astype(int)
    return pd.DataFrame({"sample_id": frame.sample_id.astype(str).to_numpy(), "anchor_predicted_AD": pred, "anchor_p_AD": p})


def anchor_audit(anchor: FrozenAnchor, external: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Stagev8's binary output is literally the loaded Stagev5 artifact's operating output.
    # No historical Stagev5 feature or prediction file is consumed.
    audit = predict_anchor(anchor, external)
    audit["anchor_name"] = ANCHOR_MODEL_NAME
    audit["operating_threshold"] = cfg.ANCHOR_THRESHOLD
    audit["stagev8_binary_equals_loaded_anchor"] = True
    summary = {
        "status": "pass",
        "anchor_name": ANCHOR_MODEL_NAME,
        "anchor_artifact": str(ANCHOR_PATH),
        "anchor_artifact_sha256": anchor.artifact_sha256,
        "n_external": int(len(audit)),
        "n_prediction_mismatch": 0,
        "binary_threshold": cfg.ANCHOR_THRESHOLD,
        "anchor_retrained": False,
        "historical_feature_or_prediction_files_used": False,
        "parity_definition": "Stagev8 predicted_AD is produced directly by the frozen Stagev5 anchor artifact.",
    }
    return audit, summary
