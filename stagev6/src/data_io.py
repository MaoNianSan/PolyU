"""Read only the generated stagev5 E/M/L feature artifacts."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import stagev6_config as cfg
from .io_utils import read_csv_robust, save_json
from .label_utils import add_standard_labels

LATE_F8 = [
    "late_sentence_structural_integrity",
    "late_phrase_continuity",
    "late_repetition_control",
    "late_repair_efficiency",
    "late_filler_control",
    "late_referential_clarity",
    "late_grammatical_stability",
    "late_local_coherence",
]


def _path(stage: str, split: str) -> Path:
    dirs = {"early": cfg.EARLY_DIR, "middle": cfg.MIDDLE_DIR, "late": cfg.LATE_DIR}
    names = {
        "early": {"ad": "ad_BM25.csv", "control": "control_BM25.csv", "test": "test_BM25.csv"},
        "middle": {"ad": "ad_embedding.csv", "control": "control_embedding.csv", "test": "test_embedding.csv"},
        "late": {"ad": "ad_LLM.csv", "control": "control_LLM.csv", "test": "test_LLM.csv"},
    }
    p = dirs[stage] / names[stage][split]
    if not p.exists():
        raise FileNotFoundError(f"Missing stagev5 generated feature file: {p}")
    return p


def _aggregate(df: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, list[str]]:
    if "sample_id" not in df.columns:
        raise ValueError(f"{stage} feature file lacks sample_id.")
    if stage == "early":
        cols = [c for c in df.columns if c.startswith("early_") and pd.api.types.is_numeric_dtype(df[c])]
        # Stagev5 early source uses early_ columns exactly; if an older cache uses bm25_, fail rather than redefine.
    elif stage == "middle":
        cols = [c for c in df.columns if c.startswith("embedding_dim_") and pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols = [c for c in LATE_F8 if c in df.columns]
        if cols != LATE_F8:
            raise ValueError(f"Late raw F8 mismatch. Expected {LATE_F8}; found {cols}")
    if not cols:
        raise ValueError(f"No usable {stage} feature columns.")
    out = df.groupby("sample_id", sort=False)[cols].mean(numeric_only=True).reset_index()
    if stage == "middle":
        out = out.rename(columns={c: f"middle_{c}" for c in cols})
        cols = [f"middle_{c}" for c in cols]
    elif stage == "early":
        # Names are already stagev5 canonical early_ names.
        pass
    return out, cols


def _load_split(split: str) -> tuple[pd.DataFrame, dict]:
    e_raw, m_raw, l_raw = (read_csv_robust(_path(stage, split)) for stage in ["early", "middle", "late"])
    base = add_standard_labels(e_raw, split)
    base = base[["__sample_id__", "__y__", "mmse", "label_normal", "label_early", "label_middle", "label_late", "severity_group", "__split__"]].drop_duplicates("__sample_id__")
    e, e_cols = _aggregate(e_raw, "early")
    m, m_cols = _aggregate(m_raw, "middle")
    l, l_cols = _aggregate(l_raw, "late")
    ids = set(base["__sample_id__"])
    for name, d in [("E", e), ("M", m), ("L", l)]:
        cur = set(d["sample_id"])
        if cur != ids:
            missing = sorted(ids - cur)[:10]
            unexpected = sorted(cur - ids)[:10]
            raise ValueError(f"{name} sample IDs differ in {split}: missing={missing}; unexpected={unexpected}")
    d = base.merge(e.rename(columns={"sample_id": "__sample_id__"}), on="__sample_id__", how="left")
    d = d.merge(m.rename(columns={"sample_id": "__sample_id__"}), on="__sample_id__", how="left")
    d = d.merge(l.rename(columns={"sample_id": "__sample_id__"}), on="__sample_id__", how="left")
    info = {
        f"{split}_early_file": str(_path("early", split)), f"{split}_middle_file": str(_path("middle", split)), f"{split}_late_file": str(_path("late", split)),
        f"{split}_n_rows": len(d), f"{split}_n_early_features": len(e_cols), f"{split}_n_middle_features": len(m_cols), f"{split}_n_late_features": len(l_cols),
    }
    return d, info


def load_train_test(report_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ad, ai = _load_split("ad")
    control, ci = _load_split("control")
    test, ti = _load_split("test")
    train = pd.concat([ad, control], ignore_index=True)
    info = {**ai, **ci, **ti, "n_train": len(train), "n_test": len(test),
            "n_train_ad": int(train["__y__"].sum()), "n_train_control": int((train["__y__"] == 0).sum()),
            "n_train_late": int(train["label_late"].sum()), "n_train_nonlate": int((train["label_late"] == 0).sum()),
            "n_test_ad": int(test["__y__"].sum()), "n_test_control": int((test["__y__"] == 0).sum()),
            "n_test_late": int(test["label_late"].sum()), "n_test_nonlate": int((test["label_late"] == 0).sum())}
    if report_dir is not None:
        save_json(info, report_dir / "input_data_check.json")
    return train, test, info


def infer_schema(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    common = [c for c in train.columns if c in test.columns]
    early = [c for c in common if c.startswith("early_")]
    middle = [c for c in common if c.startswith("middle_embedding_dim_")]
    late = [c for c in common if c in LATE_F8]
    if not early or len(middle) != 1024 or late != LATE_F8:
        raise ValueError({"n_early": len(early), "n_middle": len(middle), "late": late})
    return {"early_columns": early, "middle_columns": middle, "late_columns": late,
            "n_early": len(early), "n_middle": len(middle), "n_late": len(late),
            "late_raw_f8_columns": late, "late_auxiliary_used_in_model": False,
            "middle_aggregation": "mean by sample_id from existing stagev5 window-level BGE-M3 rows"}
