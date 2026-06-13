from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[a-zA-Z]+")
META_COLS = [
    "sample_id", "split", "source_split", "label", "mmse", "new_label",
    "label_disease", "label_early", "label_middle", "label_late",
    "label_normal", "label_mild", "label_moderate", "label_severe", "label_valid", "subgroup",
    "source_file", "source_kind",
]
FEATURE_FAMILY = "early_v5_mild_sensitive_v2compat"


@dataclass(frozen=True)
class Unit:
    name: str
    group: str
    phrases: tuple[str, ...]
    critical: bool = False


OLD10 = [
    Unit("boy_child", "object", ("boy", "child", "little boy", "son"), True),
    Unit("girl_daughter", "object", ("girl", "daughter", "sister", "little girl"), True),
    Unit("mother_woman", "object", ("mother", "woman", "lady", "mom"), True),
    Unit("cookie_jar", "object", ("cookie jar", "cookies", "cookie", "jar"), True),
    Unit("stool_chair_falling", "hazard", ("stool", "chair", "fall", "falling", "tip over"), True),
    Unit("sink_water_overflow", "hazard", ("sink", "water", "overflow", "running water", "spilling"), True),
    Unit("dish_washing", "action", ("dish", "dishes", "washing", "drying dish", "plate")),
    Unit("kitchen", "scene", ("kitchen",)),
    Unit("window_curtain", "scene", ("window", "curtain", "curtains")),
    Unit("taking_reaching_stealing", "action", ("take", "taking", "reach", "reaching", "steal", "stealing"), True),
]

EXPANDED = [
    Unit("boy", "object", ("boy", "little boy", "son", "child"), True),
    Unit("girl", "object", ("girl", "little girl", "daughter", "sister"), True),
    Unit("mother", "object", ("mother", "mom", "woman", "lady"), True),
    Unit("cookie", "object", ("cookie", "cookies"), True),
    Unit("cookie_jar", "object", ("cookie jar", "jar", "cookies in jar"), True),
    Unit("stool", "object", ("stool", "chair", "step stool"), True),
    Unit("sink", "object", ("sink", "basin"), True),
    Unit("water", "object", ("water", "faucet", "tap"), True),
    Unit("dish", "object", ("dish", "dishes", "plate")),
    Unit("window", "object", ("window", "windows")),
    Unit("curtain", "object", ("curtain", "curtains")),
    Unit("boy_taking_cookie", "action", ("boy taking cookie", "boy take cookie", "steal cookie", "taking cookies", "get cookie", "reaching cookie"), True),
    Unit("girl_reaching_cookie", "action", ("girl reaching", "girl wants cookie", "girl waiting", "sister waiting", "hand up")),
    Unit("mother_washing_dishes", "action", ("mother washing dishes", "woman washing dishes", "drying dish", "washing dish", "doing dishes"), True),
    Unit("water_overflowing", "action", ("water overflowing", "water running", "sink overflowing", "tap on", "faucet on", "water spilling"), True),
    Unit("stool_falling", "action", ("stool falling", "chair falling", "fall off", "tip over", "tipping"), True),
    Unit("child_climbing", "action", ("boy climbing", "child climbing", "standing on stool", "on stool"), True),
    Unit("stool_tipping", "hazard", ("stool tipping", "tip over", "falling stool", "lose balance"), True),
    Unit("water_spilling", "hazard", ("water spilling", "water overflowing", "spill", "overflow"), True),
    Unit("child_fall_risk", "hazard", ("fall", "falling", "hurt himself", "accident", "danger"), True),
    Unit("mother_unaware", "hazard", ("mother unaware", "not looking", "oblivious", "doesn't see", "not paying attention"), True),
    Unit("sink_overflow_risk", "hazard", ("sink overflow", "water over the sink", "water on floor"), True),
    Unit("kitchen_scene", "scene", ("kitchen", "kitchen scene")),
    Unit("two_children", "scene", ("two children", "boy and girl", "brother and sister"), True),
    Unit("mother_at_sink", "scene", ("mother at sink", "woman at sink", "standing by sink"), True),
    Unit("cookie_theft_scene", "scene", ("cookie theft", "taking cookies", "stealing cookies", "cookie jar"), True),
]

RELATION = [
    Unit("boy_on_stool", "relation", ("boy on stool", "standing on stool", "boy standing on chair", "child on stool"), True),
    Unit("boy_reaching_cookie_jar", "relation", ("boy reaching cookie jar", "boy getting cookies", "boy taking cookies", "reaching into cookie jar"), True),
    Unit("girl_waiting_for_cookie", "relation", ("girl waiting for cookie", "girl wants cookie", "girl asking for cookie", "hand out")),
    Unit("mother_at_sink_relation", "relation", ("mother at sink", "woman by sink", "mother washing dishes"), True),
    Unit("mother_unaware_of_children", "relation", ("mother not looking", "mother unaware", "doesn't see children", "oblivious to children"), True),
    Unit("sink_water_overflowing", "relation", ("sink overflowing", "water running over", "water spilling from sink", "faucet running"), True),
    Unit("stool_fall_risk", "relation", ("stool tipping", "boy may fall", "fall off stool", "chair tipping"), True),
    Unit("child_accident_risk", "relation", ("child could fall", "boy could get hurt", "dangerous", "accident"), True),
]

GROUPS = ["object", "action", "relation", "hazard", "scene"]
CRITICAL_MISSING = {
    "missing_water_overflow": ["water_overflowing", "water_spilling", "sink_overflow_risk", "sink_water_overflowing", "sink_water_overflow"],
    "missing_stool_risk": ["stool_falling", "stool_tipping", "stool_fall_risk", "stool_chair_falling"],
    "missing_mother_context": ["mother", "mother_woman", "mother_at_sink", "mother_at_sink_relation", "mother_washing_dishes"],
    "missing_child_cookie_action": ["boy_taking_cookie", "boy_reaching_cookie_jar", "taking_reaching_stealing", "cookie_jar"],
    "missing_global_scene": ["kitchen", "kitchen_scene", "cookie_theft_scene", "two_children"],
}


def unique_units(units: list[Unit]) -> list[Unit]:
    seen, out = set(), []
    for u in units:
        if u.name not in seen:
            out.append(u)
            seen.add(u.name)
    return out


def original_early_units() -> list[Unit]:
    return unique_units(OLD10 + EXPANDED + RELATION)


class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf: dict[str, float] = {}
        self.avgdl = 1.0
        self.n_docs = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(str(text).lower())

    def fit(self, texts: Iterable[str]) -> "BM25Scorer":
        docs = [self.tokenize(t) for t in texts]
        self.n_docs = max(len(docs), 1)
        lengths = [len(d) for d in docs] or [1]
        self.avgdl = max(float(np.mean(lengths)), 1.0)
        df = Counter()
        for doc in docs:
            for tok in set(doc):
                df[tok] += 1
        self.idf = {tok: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for tok, n in df.items()}
        return self

    def score_phrase(self, text: str, phrase: str) -> float:
        doc = self.tokenize(text)
        if not doc:
            return 0.0
        counts = Counter(doc)
        dl = len(doc)
        toks = self.tokenize(phrase)
        if not toks:
            return 0.0
        score = 0.0
        for tok in toks:
            tf = counts.get(tok, 0)
            if tf <= 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf.get(tok, math.log(1 + (self.n_docs + 0.5) / 0.5)) * (tf * (self.k1 + 1) / denom)
        return float(score / max(len(toks), 1))


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def score_units(df: pd.DataFrame, units: list[Unit], scorer: BM25Scorer) -> pd.DataFrame:
    rows = []
    for text in df["text"].fillna("").astype(str):
        row = {}
        for u in units:
            scores = [scorer.score_phrase(text, phrase) for phrase in u.phrases]
            row[f"early_unit_{safe_name(u.name)}"] = max(scores) if scores else 0.0
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def add_coverage_omission(features: pd.DataFrame, units: list[Unit], mention_threshold: float = 0.0) -> pd.DataFrame:
    out = features.copy()
    unit_cols = [f"early_unit_{safe_name(u.name)}" for u in units if f"early_unit_{safe_name(u.name)}" in out.columns]
    mentioned = out[unit_cols] > mention_threshold if unit_cols else pd.DataFrame(index=out.index)
    out["early_n_units_mentioned"] = mentioned.sum(axis=1) if unit_cols else 0
    for group in GROUPS:
        group_units = [u for u in units if u.group == group]
        group_cols = [f"early_unit_{safe_name(u.name)}" for u in group_units if f"early_unit_{safe_name(u.name)}" in out.columns]
        if group_cols:
            out[f"early_{group}_coverage"] = mentioned[group_cols].mean(axis=1)
            out[f"early_missing_{group}_count"] = len(group_cols) - mentioned[group_cols].sum(axis=1)
        else:
            out[f"early_{group}_coverage"] = 0.0
            out[f"early_missing_{group}_count"] = 0
    for missing_name, unit_names in CRITICAL_MISSING.items():
        cols = [f"early_unit_{safe_name(n)}" for n in unit_names if f"early_unit_{safe_name(n)}" in out.columns]
        out[f"early_{missing_name}"] = 1 if not cols else (~mentioned[cols].any(axis=1)).astype(int)
    coverage_cols = [f"early_{g}_coverage" for g in GROUPS]
    critical_cols = [f"early_{c}" for c in CRITICAL_MISSING]
    weights = np.array([0.20, 0.25, 0.25, 0.20, 0.10])
    out["early_integrity_score"] = (out[coverage_cols].to_numpy() * weights).sum(axis=1)
    out["early_omission_risk_score"] = 0.55 * (1 - out["early_integrity_score"]) + 0.45 * out[critical_cols].mean(axis=1)
    return out


def extract_earlyv0(df: pd.DataFrame, scorer: BM25Scorer) -> pd.DataFrame:
    units = original_early_units()
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    scored = score_units(df, units, scorer)
    enriched = add_coverage_omission(scored, units, mention_threshold=0.0)
    return pd.concat([meta.reset_index(drop=True), enriched.reset_index(drop=True)], axis=1)


def add_earlyv1_efficiency(earlyv0: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    out = earlyv0.copy()
    token_count = source_df["text"].fillna("").astype(str).map(lambda x: max(len(x.split()), 1)).to_numpy()
    out["token_count"] = token_count
    out["early_content_efficiency"] = out["early_integrity_score"].astype(float) / np.log1p(token_count)
    return out


def _sample_ids_match(path: Path, expected_ids: pd.Series) -> bool:
    try:
        df = pd.read_csv(path, usecols=["sample_id"])
    except Exception:
        return False
    return df["sample_id"].astype(str).tolist() == expected_ids.astype(str).tolist()


def _variant_manifest_valid(path: Path, variant: str) -> bool:
    if not path.exists():
        return False
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        m.get("feature_family") == FEATURE_FAMILY
        and m.get("early_variant") == variant
        and m.get("token_count_model_input") is False
        and m.get("regenerated_from_raw_text") is True
        and int(m.get("n_model_input_features", -1)) in {61, 62}
    )


def _load_existing_early(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> dict[str, dict[str, pd.DataFrame]] | None:
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for variant in ["earlyv0", "earlyv1"]:
        vdir = output_dir / variant
        tr_path = vdir / "train_early_features.csv"
        te_path = vdir / "external_early_features.csv"
        mf_path = vdir / "early_feature_manifest.json"
        if not (tr_path.exists() and te_path.exists() and _variant_manifest_valid(mf_path, variant)):
            return None
        if not (_sample_ids_match(tr_path, train["sample_id"]) and _sample_ids_match(te_path, test["sample_id"])):
            return None
        tr = pd.read_csv(tr_path)
        te = pd.read_csv(te_path)
        if variant == "earlyv0" and "early_content_efficiency" in tr.columns:
            return None
        if variant == "earlyv1" and "early_content_efficiency" not in tr.columns:
            return None
        result[variant] = {"train": tr, "test": te}
    return result


def _delete_existing_early(output_dir: Path) -> None:
    for variant in ["earlyv0", "earlyv1"]:
        vdir = output_dir / variant
        for name in ["train_early_features.csv", "external_early_features.csv", "early_feature_manifest.json"]:
            (vdir / name).unlink(missing_ok=True)


def early_feature_columns(df: pd.DataFrame) -> list[str]:
    audit_exclude = {"token_count"}
    meta = set(META_COLS)
    return [
        c for c in df.columns
        if c not in meta and c not in audit_exclude and c.startswith("early_") and pd.api.types.is_numeric_dtype(df[c])
    ]


def generate_early_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    force_extract: bool = False,
    reuse_features: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if force_extract:
        _delete_existing_early(output_dir)
    if reuse_features:
        existing = _load_existing_early(train, test, output_dir)
        if existing is not None:
            return existing
        _delete_existing_early(output_dir)

    scorer = BM25Scorer(k1=1.5, b=0.75).fit(train["text"].fillna("").astype(str))
    v0_train = extract_earlyv0(train, scorer)
    v0_test = extract_earlyv0(test, scorer)
    v1_train = add_earlyv1_efficiency(v0_train, train)
    v1_test = add_earlyv1_efficiency(v0_test, test)
    result = {
        "earlyv0": {"train": v0_train, "test": v0_test},
        "earlyv1": {"train": v1_train, "test": v1_test},
    }
    for variant, parts in result.items():
        vdir = output_dir / variant
        vdir.mkdir(parents=True, exist_ok=True)
        tr, te = parts["train"], parts["test"]
        tr.to_csv(vdir / "train_early_features.csv", index=False, encoding="utf-8")
        te.to_csv(vdir / "external_early_features.csv", index=False, encoding="utf-8")
        n_model_input = len(early_feature_columns(tr))
        manifest = {
            "feature_family": FEATURE_FAMILY,
            "early_variant": variant,
            "regenerated_from_raw_text": True,
            "historical_feature_outputs_reused": False,
            "feature_load_mode": "extracted_this_run",
            "feature_extracted_this_run": True,
            "token_count_model_input": False,
            "early_content_efficiency_added": bool(variant == "earlyv1"),
            "n_train_rows": int(len(tr)),
            "n_external_rows": int(len(te)),
            "n_features_total_including_audit": int(tr.shape[1]),
            "n_model_input_features": int(n_model_input),
            "unit_source": "stagev2 OLD10 + EXPANDED + RELATION",
        }
        (vdir / "early_feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
