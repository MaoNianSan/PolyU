from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .normalization import safe_numeric_frame

META_COLS = {"sample_id", "label", "split", "source_file", "source_kind", "mmse", "text", "masked_text", "token_count", "late_llm_mode", "late_llm_error"}


def merge_stage_features(base: pd.DataFrame, early: pd.DataFrame, middle: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["sample_id", "label", "split", "mmse", "token_count"] if c in base.columns]
    out = base[keep].copy()
    for part in [early, middle, late]:
        cols = [c for c in part.columns if c != "sample_id" and c not in META_COLS]
        out = out.merge(part[["sample_id"] + cols], on="sample_id", how="left")
    return out


def infer_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, list[str]]:
    common = [c for c in train_df.columns if c in test_df.columns]
    e = [c for c in common if c.startswith("early_") and pd.api.types.is_numeric_dtype(train_df[c])]
    m = [c for c in common if c.startswith("middle_") and pd.api.types.is_numeric_dtype(train_df[c])]
    l = [c for c in common if c.startswith("late_") and pd.api.types.is_numeric_dtype(train_df[c])]
    # Hard guard: audit token_count must never enter the classifier.
    e = [c for c in e if c != "early_token_count" and c != "token_count"]
    if "token_count" in e + m + l:
        raise AssertionError("token_count leaked into model feature columns")
    if not e or not m or not l:
        raise ValueError(f"Feature inference failed: early={len(e)}, middle={len(m)}, late={len(l)}")
    return {"early": e, "middle": m, "late": l}


def _fit_stage_signal(train_block: pd.DataFrame, test_block: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series]:
    tr = safe_numeric_frame(train_block)
    te = safe_numeric_frame(test_block)
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


def build_feature_blocks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    cols = infer_columns(train_df, test_df)
    e, m, l = cols["early"], cols["middle"], cols["late"]
    Xtr_e, Xte_e = train_df[e].copy(), test_df[e].copy()
    Xtr_m, Xte_m = train_df[m].copy(), test_df[m].copy()
    Xtr_l, Xte_l = train_df[l].copy(), test_df[l].copy()
    tr_e_sig, te_e_sig = _fit_stage_signal(Xtr_e, Xte_e, "early_activation")
    tr_m_sig, te_m_sig = _fit_stage_signal(Xtr_m, Xte_m, "middle_activation")
    tr_l_sig, te_l_sig = _fit_stage_signal(Xtr_l, Xte_l, "late_activation")
    tr_em = (tr_e_sig * tr_m_sig).rename("early_middle_activation")
    te_em = (te_e_sig * te_m_sig).rename("early_middle_activation")
    tr_ml = (tr_m_sig * tr_l_sig).rename("middle_late_activation")
    te_ml = (te_m_sig * te_l_sig).rename("middle_late_activation")
    tr_el = (tr_e_sig * tr_l_sig).rename("early_late_activation")
    te_el = (te_e_sig * te_l_sig).rename("early_late_activation")
    tr_stage = pd.DataFrame({"early_activation": tr_e_sig, "middle_activation": tr_m_sig, "late_activation": tr_l_sig, "early_middle_activation": tr_em, "middle_late_activation": tr_ml, "early_late_activation": tr_el})
    te_stage = pd.DataFrame({"early_activation": te_e_sig, "middle_activation": te_m_sig, "late_activation": te_l_sig, "early_middle_activation": te_em, "middle_late_activation": te_ml, "early_late_activation": te_el})
    tr_e_by_m = _scaled(Xtr_e, tr_m_sig, "early_scaled_by_middle")
    te_e_by_m = _scaled(Xte_e, te_m_sig, "early_scaled_by_middle")
    tr_m_by_l = _scaled(Xtr_m, tr_l_sig, "middle_scaled_by_late")
    te_m_by_l = _scaled(Xte_m, te_l_sig, "middle_scaled_by_late")
    tr_e_by_l = _scaled(Xtr_e, tr_l_sig, "early_scaled_by_late")
    te_e_by_l = _scaled(Xte_e, te_l_sig, "early_scaled_by_late")
    tr_seq = pd.concat([tr_stage, tr_e_by_m, tr_m_by_l, tr_e_by_l], axis=1)
    te_seq = pd.concat([te_stage, te_e_by_m, te_m_by_l, te_e_by_l], axis=1)
    blocks = {
        "__stage_dims__": (len(e), len(m), len(l)),
        "early_only": (Xtr_e, Xte_e),
        "middle_only": (Xtr_m, Xte_m),
        "late_only": (Xtr_l, Xte_l),
        "early_middle": (pd.concat([Xtr_e, Xtr_m], axis=1), pd.concat([Xte_e, Xte_m], axis=1)),
        "early_middle_scale": (pd.concat([Xtr_e, Xtr_m, tr_stage[["middle_activation"]], tr_e_by_m], axis=1), pd.concat([Xte_e, Xte_m, te_stage[["middle_activation"]], te_e_by_m], axis=1)),
        "middle_late": (pd.concat([Xtr_m, Xtr_l], axis=1), pd.concat([Xte_m, Xte_l], axis=1)),
        "middle_late_scale": (pd.concat([Xtr_m, Xtr_l, tr_stage[["late_activation"]], tr_m_by_l], axis=1), pd.concat([Xte_m, Xte_l, te_stage[["late_activation"]], te_m_by_l], axis=1)),
        "all": (pd.concat([Xtr_e, Xtr_m, Xtr_l], axis=1), pd.concat([Xte_e, Xte_m, Xte_l], axis=1)),
        "all_plus_interactions": (pd.concat([Xtr_e, Xtr_m, Xtr_l, tr_seq], axis=1), pd.concat([Xte_e, Xte_m, Xte_l, te_seq], axis=1)),
        "early_late": (pd.concat([Xtr_e, Xtr_l], axis=1), pd.concat([Xte_e, Xte_l], axis=1)),
        "stage_activation_summary": (tr_stage, te_stage),
        "sequential_interactions": (tr_seq, te_seq),
        "stage_score_early_middle": (tr_stage[["early_activation", "middle_activation", "early_middle_activation"]], te_stage[["early_activation", "middle_activation", "early_middle_activation"]]),
        "stage_score_middle_late": (tr_stage[["middle_activation", "late_activation", "middle_late_activation"]], te_stage[["middle_activation", "late_activation", "middle_late_activation"]]),
        "stage_score_early_late": (tr_stage[["early_activation", "late_activation", "early_late_activation"]], te_stage[["early_activation", "late_activation", "early_late_activation"]]),
        "stage_score_three_stage": (tr_stage, te_stage),
    }
    return blocks
