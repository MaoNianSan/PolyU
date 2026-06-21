from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .bm25_extractor import BM25Scorer
from .feature_extraction import META_COLS, extract_features_for_version
from .preprocess import run_preprocess

FEATURE_VERSION = "early_v5_mild_sensitive"


def _prefix_early_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    meta = set(META_COLS + ["label_disease", "label_early", "label_middle", "label_late"])
    rename = {}
    for c in out.columns:
        if c in meta:
            continue
        if c.startswith("early_"):
            continue
        rename[c] = "early_" + c
    return out.rename(columns=rename)


def generate_all_early_features(input_dir: Path, output_root: Path, version: str = FEATURE_VERSION) -> tuple[dict[str, pd.DataFrame], Path, dict]:
    """Preprocess raw files and regenerate early BM25-derived features.

    Historical early/BM25 output files are never read.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    preprocess_dir = output_root / "preprocess"
    early_dir = output_root / "features" / "early_bm25" / version
    early_dir.mkdir(parents=True, exist_ok=True)

    preprocessed = run_preprocess(input_dir, preprocess_dir)
    train_text = pd.concat([preprocessed["ad"], preprocessed["control"]], ignore_index=True)["text"]
    scorer = BM25Scorer(k1=1.5, b=0.75).fit(train_text)

    manifest = {
        "feature_family": "early_bm25",
        "feature_version": version,
        "input_dir": str(input_dir),
        "preprocess_dir": str(preprocess_dir),
        "historical_outputs_reused": False,
    }
    for split in ["ad", "control", "test"]:
        feats = extract_features_for_version(preprocessed[split], version, scorer, mention_threshold=0.0)
        feats = _prefix_early_feature_columns(feats)
        out_name = {"ad": "ad_BM25.csv", "control": "control_BM25.csv", "test": "test_BM25.csv"}[split]
        out_path = early_dir / out_name
        feats.to_csv(out_path, index=False, encoding="utf-8-sig")
        manifest[f"{split}_file"] = str(out_path)
        manifest[f"{split}_n_rows"] = int(len(feats))
        manifest[f"{split}_n_cols"] = int(feats.shape[1])

    manifest_path = output_root / "feature_manifest_early.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return preprocessed, early_dir, manifest
