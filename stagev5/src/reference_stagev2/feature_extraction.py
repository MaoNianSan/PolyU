from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .bm25_extractor import BM25Scorer
from .bm25_units import Unit, get_units, relation_units

GROUPS = ["object", "action", "relation", "hazard", "scene"]
CRITICAL_MISSING = {
    "missing_water_overflow": ["water_overflowing", "water_spilling", "sink_overflow_risk", "sink_water_overflowing", "sink_water_overflow"],
    "missing_stool_risk": ["stool_falling", "stool_tipping", "stool_fall_risk", "stool_chair_falling"],
    "missing_mother_context": ["mother", "mother_woman", "mother_at_sink", "mother_at_sink_relation", "mother_washing_dishes"],
    "missing_child_cookie_action": ["boy_taking_cookie", "boy_reaching_cookie_jar", "taking_reaching_stealing", "cookie_jar"],
    "missing_global_scene": ["kitchen", "kitchen_scene", "cookie_theft_scene", "two_children"],
}
META_COLS = ["sample_id", "source_split", "disease_label", "mmse", "new_label", "label_disease", "label_early", "label_middle", "label_late", "label_normal", "label_mild", "label_moderate", "label_severe", "label_valid", "subgroup"]


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def score_units(df: pd.DataFrame, units: list[Unit], scorer: BM25Scorer) -> pd.DataFrame:
    rows = []
    for text in df["text"].fillna("").astype(str):
        row = {}
        for u in units:
            phrase_scores = [scorer.score_phrase(text, phrase) for phrase in u.phrases]
            row[f"unit_{safe_name(u.name)}"] = max(phrase_scores) if phrase_scores else 0.0
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def add_coverage_omission(features: pd.DataFrame, units: list[Unit], mention_threshold: float = 0.0) -> pd.DataFrame:
    out = features.copy()
    unit_cols = [f"unit_{safe_name(u.name)}" for u in units if f"unit_{safe_name(u.name)}" in out.columns]
    mentioned = out[unit_cols] > mention_threshold if unit_cols else pd.DataFrame(index=out.index)
    out["n_units_mentioned"] = mentioned.sum(axis=1) if unit_cols else 0

    for group in GROUPS:
        group_units = [u for u in units if u.group == group]
        group_cols = [f"unit_{safe_name(u.name)}" for u in group_units if f"unit_{safe_name(u.name)}" in out.columns]
        if group_cols:
            out[f"{group}_coverage"] = mentioned[group_cols].mean(axis=1)
            out[f"missing_{group}_count"] = len(group_cols) - mentioned[group_cols].sum(axis=1)
        else:
            out[f"{group}_coverage"] = 0.0
            out[f"missing_{group}_count"] = 0

    for missing_name, unit_names in CRITICAL_MISSING.items():
        cols = [f"unit_{safe_name(n)}" for n in unit_names if f"unit_{safe_name(n)}" in out.columns]
        out[missing_name] = 1 if not cols else (~mentioned[cols].any(axis=1)).astype(int)

    coverage_cols = [f"{g}_coverage" for g in GROUPS]
    critical_cols = list(CRITICAL_MISSING.keys())
    weights = np.array([0.20, 0.25, 0.25, 0.20, 0.10])
    out["early_integrity_score"] = (out[coverage_cols].to_numpy() * weights).sum(axis=1)
    out["early_omission_risk_score"] = 0.55 * (1 - out["early_integrity_score"]) + 0.45 * out[critical_cols].mean(axis=1)
    return out


def add_relation_gate_scores(df: pd.DataFrame, scorer: BM25Scorer, mention_threshold: float = 0.0) -> pd.DataFrame:
    """Return compact relation summaries without exposing raw relation units to the main classifier."""
    rel_units = relation_units()
    rel_scored = score_units(df, rel_units, scorer)
    rel_enriched = add_coverage_omission(rel_scored, rel_units, mention_threshold)
    rel_unit_cols = [c for c in rel_scored.columns if c.startswith("unit_")]
    out = pd.DataFrame(index=df.index)
    out["relation_risk_score"] = rel_scored[rel_unit_cols].max(axis=1) if rel_unit_cols else 0.0
    out["relation_mean_score"] = rel_scored[rel_unit_cols].mean(axis=1) if rel_unit_cols else 0.0
    out["relation_coverage_score"] = rel_enriched["relation_coverage"]
    out["relation_missing_count"] = rel_enriched["missing_relation_count"]
    return out


def select_refined_columns(enriched: pd.DataFrame, version: str) -> pd.DataFrame:
    """Drop/keep columns for refined V5 variants after full feature extraction."""
    meta_cols = [c for c in META_COLS if c in enriched.columns]
    unit_cols = [c for c in enriched.columns if c.startswith("unit_")]
    relation_unit_cols = [c for c in unit_cols if any(key in c for key in [
        "boy_on_stool", "boy_reaching_cookie_jar", "girl_waiting_for_cookie", "mother_at_sink_relation",
        "mother_unaware_of_children", "sink_water_overflowing", "stool_fall_risk", "child_accident_risk"
    ])]
    non_relation_unit_cols = [c for c in unit_cols if c not in relation_unit_cols]
    coverage_cols = [c for c in enriched.columns if c.endswith("_coverage") or c == "n_units_mentioned"]
    missing_cols = [c for c in enriched.columns if c.startswith("missing_")]
    score_cols = ["early_integrity_score", "early_omission_risk_score"]
    relation_summary_cols = [c for c in ["relation_risk_score", "relation_mean_score", "relation_coverage_score", "relation_missing_count"] if c in enriched.columns]

    if version == "early_v5_coverage_omission_only":
        keep = meta_cols + coverage_cols + missing_cols + score_cols
    elif version == "early_v5_balanced_refined":
        keep = meta_cols + non_relation_unit_cols + coverage_cols + missing_cols + score_cols + relation_summary_cols
    elif version == "early_v5_no_relation":
        keep = meta_cols + non_relation_unit_cols + coverage_cols + missing_cols + score_cols
    elif version == "early_v5_relation_only":
        keep = meta_cols + relation_unit_cols + coverage_cols + missing_cols + score_cols
    else:
        keep = list(enriched.columns)

    # Deduplicate while preserving order.
    seen = set(); ordered = []
    for c in keep:
        if c in enriched.columns and c not in seen:
            ordered.append(c); seen.add(c)
    return enriched[ordered]


def extract_features_for_version(df: pd.DataFrame, version: str, scorer: BM25Scorer, mention_threshold: float = 0.0) -> pd.DataFrame:
    units = get_units(version)
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    scored = score_units(df, units, scorer)
    enriched = add_coverage_omission(scored, units, mention_threshold)

    # For the balanced refined model, relation is not raw model input; it is summarized as gate scores.
    if version == "early_v5_balanced_refined":
        rel_summary = add_relation_gate_scores(df, scorer, mention_threshold)
        enriched = pd.concat([enriched.reset_index(drop=True), rel_summary.reset_index(drop=True)], axis=1)

    out = pd.concat([meta.reset_index(drop=True), enriched.reset_index(drop=True)], axis=1)
    return select_refined_columns(out, version)


def save_feature_sets(preprocessed: dict[str, pd.DataFrame], versions: list[str], scorer: BM25Scorer, output_dir: Path, mention_threshold: float = 0.0) -> dict[str, dict[str, pd.DataFrame]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_features = {}
    for version in versions:
        all_features[version] = {}
        for split, df in preprocessed.items():
            feats = extract_features_for_version(df, version, scorer, mention_threshold)
            feats.to_csv(output_dir / f"{split}_{version}.csv", index=False)
            all_features[version][split] = feats
    return all_features
