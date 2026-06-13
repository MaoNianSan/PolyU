from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .normalization import find_first_column, normalize_label_value, normalize_text_value, TEXT_CANDIDATES, LABEL_CANDIDATES, MMSE_CANDIDATES


def _zip_candidates(source_zip: Path | None = None) -> list[Path]:
    if source_zip:
        return [source_zip]
    roots = [config.PROJECT_ROOT.parent, Path("/mnt/data"), Path.cwd(), config.PROJECT_ROOT]
    names = ["stagev2(2).zip", "stagev2(3).zip", "stagev3.zip", "stagev2.zip"]
    found: list[Path] = []
    for root in roots:
        for name in names:
            p = root / name
            if p.exists() and p.is_file() and p.resolve() != (config.PROJECT_ROOT.parent / "stagev3.zip").resolve():
                found.append(p.resolve())
        found.extend([p.resolve() for p in root.glob("stagev2*.zip") if p.is_file()])
    unique = []
    seen = set()
    for p in found:
        if p not in seen:
            unique.append(p); seen.add(p)
    return sorted(unique, key=lambda p: p.stat().st_mtime, reverse=True)


def find_and_copy_raw_files(source_zip: Path | None, raw_dir: Path) -> str:
    raw_dir.mkdir(parents=True, exist_ok=True)
    required = list(config.RAW_FILES.values())
    if all((raw_dir / f).exists() for f in required):
        return str(raw_dir)
    for zpath in _zip_candidates(source_zip):
        if not zpath.exists():
            continue
        try:
            with zipfile.ZipFile(zpath) as zf:
                names = zf.namelist()
                mapping = {}
                for req in required:
                    matches = [n for n in names if Path(n).name == req]
                    if matches:
                        mapping[req] = matches[0]
                if len(mapping) == len(required):
                    for req, member in mapping.items():
                        with zf.open(member) as src, (raw_dir / req).open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                    return str(zpath)
        except zipfile.BadZipFile:
            continue
    missing = [f for f in required if not (raw_dir / f).exists()]
    raise FileNotFoundError(
        "Cannot find required raw CSV files or original zip. "
        f"Missing={missing}. Place them under {raw_dir}."
    )


def _normalize_one(path: Path, source_kind: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    text_col = find_first_column(df.columns, TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError(f"Cannot identify text column in {path}. Expected one of {TEXT_CANDIDATES}; actual={list(df.columns)}")
    label_col = find_first_column(df.columns, LABEL_CANDIDATES)
    mmse_col = find_first_column(df.columns, MMSE_CANDIDATES)
    out = pd.DataFrame()
    if "sample_id" in df.columns:
        out["sample_id"] = df["sample_id"].astype(str)
    else:
        # v2-compatible IDs make feature/prediction comparison direct across
        # versions: AD_0001 / CTRL_0001 / TEST_0001.
        prefix = {"ad": "AD", "control": "CTRL", "test": "TEST"}.get(source_kind, split.upper())
        out["sample_id"] = [f"{prefix}_{i+1:04d}" for i in range(len(df))]
    out["text"] = df[text_col].map(normalize_text_value)
    if label_col is None:
        if source_kind in {"ad", "control"}:
            out["label"] = 1 if source_kind == "ad" else 0
        else:
            raise ValueError(f"Cannot identify label column in {path}; external test labels are required.")
    else:
        bad = []
        labels = []
        for v in df[label_col].tolist():
            try:
                labels.append(normalize_label_value(v, source_kind=source_kind))
            except Exception:
                bad.append(v)
                labels.append(None)
        if bad:
            raise ValueError(f"Unrecognized label values in {path}: {bad[:20]}")
        out["label"] = labels
    out["split"] = split
    out["source_file"] = path.name
    out["source_kind"] = source_kind
    out["mmse"] = pd.to_numeric(df[mmse_col], errors="coerce") if mmse_col else pd.NA
    out["token_count"] = out["text"].map(lambda s: max(len(str(s).split()), 1))
    audit = out[["sample_id", "source_file", "source_kind", "split", "label", "token_count"]].copy()
    audit["empty_text"] = out["text"].eq("")
    audit["text_column_used"] = text_col
    audit["label_column_used"] = label_col or "source_inferred"
    audit["mmse_column_used"] = mmse_col or "missing"
    return out, audit


def run_preprocess(raw_dir: Path, preprocess_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    ad, audit_ad = _normalize_one(raw_dir / config.RAW_FILES["ad"], "ad", "train")
    control, audit_control = _normalize_one(raw_dir / config.RAW_FILES["control"], "control", "train")
    test, audit_test = _normalize_one(raw_dir / config.RAW_FILES["test"], "test", "external")
    train = pd.concat([ad, control], ignore_index=True)
    audit = pd.concat([audit_ad, audit_control, audit_test], ignore_index=True)
    train.to_csv(preprocess_dir / "train_preprocessed.csv", index=False, encoding="utf-8")
    test.to_csv(preprocess_dir / "external_test_preprocessed.csv", index=False, encoding="utf-8")
    audit.to_csv(config.FINAL_REPORT_DIR / "preprocessing_audit.csv", index=False, encoding="utf-8")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_n": int(len(train)),
        "external_n": int(len(test)),
        "train_label_counts": {str(k): int(v) for k, v in train["label"].value_counts().to_dict().items()},
        "external_label_counts": {str(k): int(v) for k, v in test["label"].value_counts().to_dict().items()},
        "empty_text_n": int(audit["empty_text"].sum()),
        "raw_dir": str(raw_dir),
        "historical_feature_outputs_reused": False,
    }
    (preprocess_dir / "preprocess_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return train, test, summary
