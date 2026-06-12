from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .early_features import early_feature_columns

META_COLS = {
    "sample_id", "split", "source_split", "label", "mmse", "new_label",
    "label_disease", "label_early", "label_middle", "label_late",
    "label_normal", "label_mild", "label_moderate", "label_severe", "label_valid", "subgroup",
    "text", "file", "masked_text", "token_count",
}


def _middle_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("middle_") and pd.api.types.is_numeric_dtype(df[c])]


def _late_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("late_") and pd.api.types.is_numeric_dtype(df[c])]


def merge_stage_features(base: pd.DataFrame, early: pd.DataFrame, middle: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    keep_base = [c for c in base.columns if c in ["sample_id", "label", "split", "mmse", "subgroup", "new_label", "label_early", "label_middle", "label_late"]]
    out = base[keep_base].copy()
    for part in [early, middle, late]:
        cols = [c for c in part.columns if c != "sample_id" and c not in META_COLS]
        out = out.merge(part[["sample_id"] + cols], on="sample_id", how="left")
    return out


def infer_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    common = [c for c in train_df.columns if c in test_df.columns]
    e = [c for c in common if c.startswith("early_") and pd.api.types.is_numeric_dtype(train_df[c])]
    m = [c for c in common if c.startswith("middle_") and pd.api.types.is_numeric_dtype(train_df[c])]
    l = [c for c in common if c.startswith("late_") and pd.api.types.is_numeric_dtype(train_df[c])]
    # token_count audit is not prefixed and is not selected. Keep an explicit guard anyway.
    e = [c for c in e if c != "early_token_count"]
    if not e or not m or not l:
        raise ValueError(f"Feature inference failed: n_early={len(e)}, n_middle={len(m)}, n_late={len(l)}")
    return {"early": e, "middle": m, "late": l}


def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_stage_signal(train_block: pd.DataFrame, test_block: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series]:
    tr = _safe_numeric(train_block)
    te = _safe_numeric(test_block)
    med = tr.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = tr.std(axis=0).replace([np.inf, -np.inf], np.nan).fillna(1.0).replace(0.0, 1.0)
    ztr = (tr.fillna(med) - med) / std
    zte = (te.fillna(med) - med) / std
    atr = ztr.abs().mean(axis=1).to_numpy(float)
    ate = zte.abs().mean(axis=1).to_numpy(float)
    lo, hi = np.nanpercentile(atr, [5, 95]) if len(atr) else (0.0, 1.0)
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
    return (
        pd.Series(np.clip((atr - lo) / (hi - lo), 0, 1), index=train_block.index, name=name),
        pd.Series(np.clip((ate - lo) / (hi - lo), 0, 1), index=test_block.index, name=name),
    )


def _scaled(block: pd.DataFrame, signal: pd.Series, prefix: str) -> pd.DataFrame:
    out = block.mul(signal.to_numpy(), axis=0)
    out.columns = [f"{prefix}__{c}" for c in block.columns]
    return out


def build_feature_blocks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    cols = infer_columns(train_df, test_df)
    e, m, l = cols["early"], cols["middle"], cols["late"]
    Xtr_e, Xte_e = train_df[e].copy(), test_df[e].copy()
    Xtr_m, Xte_m = train_df[m].copy(), test_df[m].copy()
    Xtr_l, Xte_l = train_df[l].copy(), test_df[l].copy()
    tr_mid_sig, te_mid_sig = _fit_stage_signal(Xtr_m, Xte_m, "middle_activation")
    tr_late_sig, te_late_sig = _fit_stage_signal(Xtr_l, Xte_l, "late_activation")
    tr_ml_sig = (tr_mid_sig * tr_late_sig).rename("middle_late_activation")
    te_ml_sig = (te_mid_sig * te_late_sig).rename("middle_late_activation")
    tr_signal = pd.DataFrame({"middle_activation": tr_mid_sig, "late_activation": tr_late_sig, "middle_late_activation": tr_ml_sig})
    te_signal = pd.DataFrame({"middle_activation": te_mid_sig, "late_activation": te_late_sig, "middle_late_activation": te_ml_sig})
    tr_e_by_m = _scaled(Xtr_e, tr_mid_sig, "early_scaled_by_middle")
    te_e_by_m = _scaled(Xte_e, te_mid_sig, "early_scaled_by_middle")
    tr_m_by_l = _scaled(Xtr_m, tr_late_sig, "middle_scaled_by_late")
    te_m_by_l = _scaled(Xte_m, te_late_sig, "middle_scaled_by_late")
    tr_e_by_l = _scaled(Xtr_e, tr_late_sig, "early_scaled_by_late")
    te_e_by_l = _scaled(Xte_e, te_late_sig, "early_scaled_by_late")
    tr_e_by_ml = _scaled(Xtr_e, tr_ml_sig, "early_scaled_by_middle_late")
    te_e_by_ml = _scaled(Xte_e, te_ml_sig, "early_scaled_by_middle_late")
    blocks = {
        "early_only": (Xtr_e, Xte_e),
        "middle_only": (Xtr_m, Xte_m),
        "late_only": (Xtr_l, Xte_l),
        "early_middle": (pd.concat([Xtr_e, Xtr_m], axis=1), pd.concat([Xte_e, Xte_m], axis=1)),
        "early_middle_scale": (pd.concat([Xtr_e, Xtr_m, tr_signal[["middle_activation"]], tr_e_by_m], axis=1), pd.concat([Xte_e, Xte_m, te_signal[["middle_activation"]], te_e_by_m], axis=1)),
        "middle_late": (pd.concat([Xtr_m, Xtr_l], axis=1), pd.concat([Xte_m, Xte_l], axis=1)),
        "middle_late_scale": (pd.concat([Xtr_m, Xtr_l, tr_signal[["late_activation"]], tr_m_by_l], axis=1), pd.concat([Xte_m, Xte_l, te_signal[["late_activation"]], te_m_by_l], axis=1)),
        "all": (pd.concat([Xtr_e, Xtr_m, Xtr_l], axis=1), pd.concat([Xte_e, Xte_m, Xte_l], axis=1)),
        "all_plus_interactions": (pd.concat([Xtr_e, Xtr_m, Xtr_l, tr_signal, tr_e_by_m, tr_e_by_l, tr_e_by_ml, tr_m_by_l], axis=1), pd.concat([Xte_e, Xte_m, Xte_l, te_signal, te_e_by_m, te_e_by_l, te_e_by_ml, te_m_by_l], axis=1)),
    }
    return {name: blocks[name] for name in config.FEATURE_BLOCKS}
