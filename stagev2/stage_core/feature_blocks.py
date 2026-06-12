"""Feature-block construction, including stagev2 sequential interaction blocks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def infer_stage_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    common = [c for c in train_df.columns if c in test_df.columns]
    early = [c for c in common if c.startswith("early_") and pd.api.types.is_numeric_dtype(train_df[c])]
    middle = [c for c in common if c.startswith("middle_") and pd.api.types.is_numeric_dtype(train_df[c])]
    late = [c for c in common if c.startswith("late_") and pd.api.types.is_numeric_dtype(train_df[c])]
    if not early or not middle or not late:
        raise ValueError(
            "Stage feature inference failed: "
            f"n_early={len(early)}, n_middle={len(middle)}, n_late={len(late)}. "
            "Check early_v5_mild_sensitive, BGE-M3, and LLM feature files."
        )
    return {
        "early_columns": early,
        "middle_columns": middle,
        "late_columns": late,
        "n_early": len(early),
        "n_middle": len(middle),
        "n_late": len(late),
    }


def save_feature_manifest(schema: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _fit_stage_signal(train_block: pd.DataFrame, test_block: pd.DataFrame, name: str):
    """Return train/test unsupervised stage activation signals in [0, 1].

    This is intentionally label-free. It supports the mechanism idea:
    later-stage feature activation scales earlier-stage raw features.
    It uses train medians/stds only, then computes mean absolute z activation.
    """
    tr = _safe_numeric(train_block)
    te = _safe_numeric(test_block)
    med = tr.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = tr.std(axis=0).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    std = std.replace(0.0, 1.0)
    ztr = (tr.fillna(med) - med) / std
    zte = (te.fillna(med) - med) / std
    atr = ztr.abs().mean(axis=1).to_numpy(dtype=float)
    ate = zte.abs().mean(axis=1).to_numpy(dtype=float)
    # Robust min-max from train distribution only.
    lo, hi = np.nanpercentile(atr, [5, 95]) if len(atr) else (0.0, 1.0)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    strn = np.clip((atr - lo) / (hi - lo), 0.0, 1.0)
    stst = np.clip((ate - lo) / (hi - lo), 0.0, 1.0)
    return pd.Series(strn, index=train_block.index, name=name), pd.Series(stst, index=test_block.index, name=name)


def _scaled_block(block: pd.DataFrame, signal: pd.Series, prefix: str) -> pd.DataFrame:
    scaled = block.mul(signal.to_numpy(), axis=0)
    scaled.columns = [f"{prefix}__{c}" for c in block.columns]
    return scaled


def _summary_frame(signals: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(signals)


def build_feature_blocks(train_df: pd.DataFrame, test_df: pd.DataFrame, schema: dict) -> dict:
    e = schema["early_columns"]
    m = schema["middle_columns"]
    l = schema["late_columns"]

    Xtr_e, Xte_e = train_df[e].copy(), test_df[e].copy()
    Xtr_m, Xte_m = train_df[m].copy(), test_df[m].copy()
    Xtr_l, Xte_l = train_df[l].copy(), test_df[l].copy()

    # Unsupervised later-stage activation signals.
    tr_mid_sig, te_mid_sig = _fit_stage_signal(Xtr_m, Xte_m, "middle_activation")
    tr_late_sig, te_late_sig = _fit_stage_signal(Xtr_l, Xte_l, "late_activation")
    tr_ml_sig = (tr_mid_sig * tr_late_sig).rename("middle_late_activation")
    te_ml_sig = (te_mid_sig * te_late_sig).rename("middle_late_activation")

    tr_signal = _summary_frame({
        "middle_activation": tr_mid_sig,
        "late_activation": tr_late_sig,
        "middle_late_activation": tr_ml_sig,
    })
    te_signal = _summary_frame({
        "middle_activation": te_mid_sig,
        "late_activation": te_late_sig,
        "middle_late_activation": te_ml_sig,
    })

    # Sequential scaling blocks: later-stage summaries scale earlier-stage raw features.
    tr_e_by_m = _scaled_block(Xtr_e, tr_mid_sig, "early_scaled_by_middle")
    te_e_by_m = _scaled_block(Xte_e, te_mid_sig, "early_scaled_by_middle")
    tr_e_by_l = _scaled_block(Xtr_e, tr_late_sig, "early_scaled_by_late")
    te_e_by_l = _scaled_block(Xte_e, te_late_sig, "early_scaled_by_late")
    tr_e_by_ml = _scaled_block(Xtr_e, tr_ml_sig, "early_scaled_by_middle_late")
    te_e_by_ml = _scaled_block(Xte_e, te_ml_sig, "early_scaled_by_middle_late")
    tr_m_by_l = _scaled_block(Xtr_m, tr_late_sig, "middle_scaled_by_late")
    te_m_by_l = _scaled_block(Xte_m, te_late_sig, "middle_scaled_by_late")

    tr_early_middle_scale = pd.concat([Xtr_e, Xtr_m, tr_signal[["middle_activation"]], tr_e_by_m], axis=1)
    te_early_middle_scale = pd.concat([Xte_e, Xte_m, te_signal[["middle_activation"]], te_e_by_m], axis=1)

    tr_middle_late_scale = pd.concat([Xtr_m, Xtr_l, tr_signal[["late_activation"]], tr_m_by_l], axis=1)
    te_middle_late_scale = pd.concat([Xte_m, Xte_l, te_signal[["late_activation"]], te_m_by_l], axis=1)

    tr_sequential_interactions = pd.concat([
        tr_signal, tr_e_by_m, tr_e_by_l, tr_e_by_ml, tr_m_by_l
    ], axis=1)
    te_sequential_interactions = pd.concat([
        te_signal, te_e_by_m, te_e_by_l, te_e_by_ml, te_m_by_l
    ], axis=1)

    tr_all_plus_interactions = pd.concat([
        Xtr_e, Xtr_m, Xtr_l, tr_signal, tr_e_by_m, tr_e_by_l, tr_e_by_ml, tr_m_by_l
    ], axis=1)
    te_all_plus_interactions = pd.concat([
        Xte_e, Xte_m, Xte_l, te_signal, te_e_by_m, te_e_by_l, te_e_by_ml, te_m_by_l
    ], axis=1)

    out = {
        "X_train_early": Xtr_e,
        "X_test_early": Xte_e,
        "X_train_middle": Xtr_m,
        "X_test_middle": Xte_m,
        "X_train_late": Xtr_l,
        "X_test_late": Xte_l,
        "X_train_early_middle": pd.concat([Xtr_e, Xtr_m], axis=1),
        "X_test_early_middle": pd.concat([Xte_e, Xte_m], axis=1),
        "X_train_middle_late": pd.concat([Xtr_m, Xtr_l], axis=1),
        "X_test_middle_late": pd.concat([Xte_m, Xte_l], axis=1),
        "X_train_early_late": pd.concat([Xtr_e, Xtr_l], axis=1),
        "X_test_early_late": pd.concat([Xte_e, Xte_l], axis=1),
        "X_train_all": pd.concat([Xtr_e, Xtr_m, Xtr_l], axis=1),
        "X_test_all": pd.concat([Xte_e, Xte_m, Xte_l], axis=1),
        "X_train_stage_activation_summary": tr_signal,
        "X_test_stage_activation_summary": te_signal,
        "X_train_early_middle_scale": tr_early_middle_scale,
        "X_test_early_middle_scale": te_early_middle_scale,
        "X_train_middle_late_scale": tr_middle_late_scale,
        "X_test_middle_late_scale": te_middle_late_scale,
        "X_train_sequential_interactions": tr_sequential_interactions,
        "X_test_sequential_interactions": te_sequential_interactions,
        "X_train_all_plus_interactions": tr_all_plus_interactions,
        "X_test_all_plus_interactions": te_all_plus_interactions,
        "y_train": train_df["__y__"].astype(int).to_numpy(),
        "y_test": test_df["__y__"].astype(int).to_numpy(),
        "n_early": len(e),
        "n_middle": len(m),
        "n_late": len(l),
    }
    return out
