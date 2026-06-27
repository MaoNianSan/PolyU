"""Stagev8.5: MMSE-informed ordinal severity analysis.

Feature reconstruction is intentionally separate. This module reads only the
validated E/M/L feature files through the verbatim Stagev6 loader, retains the
frozen Stagev5 AD/control anchor, and fits two prespecified conditional
threshold heads.
"""
from __future__ import annotations

import json
import math
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config as cfg
from .anchor import anchor_audit, load_frozen_anchor, predict_anchor
from .feature_contract_v85 import (
    add_mmse_severity_labels,
    build_feature_and_label_audits,
    optional_stagev6_runtime_comparison,
    source_contract,
)
from .progress import StageProgress, progress_section
from .severity_metrics import (
    binary_metrics,
    calibration_table,
    confusion_long,
    ordinal_continuous_metrics,
    three_strata_metrics,
)
from .severity_model_specs import SeverityHeadSpec, severity_head_specs
from .severity_reporting import (
    experiment_report,
    literature_rationale,
    make_figures,
    selected_summary,
    write_json,
)
from .source_integrity import verify_strict_reference_sources


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def blocks(data) -> dict[str, list[str]]:
    return {
        "E": data.early_columns,
        "M": data.middle_columns,
        "L": data.late_columns,
        "EM": [*data.early_columns, *data.middle_columns],
    }


def _ad_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[frame["__y__"].eq(1)].copy()
    if out.empty:
        raise RuntimeError("No AD samples available for Stagev8.5 severity analysis.")
    if out["mmse"].isna().any():
        raise RuntimeError("Stagev8.5 requires non-missing AD MMSE values.")
    return out


def _head_data(ad: pd.DataFrame, head: str) -> tuple[pd.DataFrame, np.ndarray, str]:
    if head == "T20":
        y = ad["target_T20_mmse_le_20"].astype(int).to_numpy()
        return ad.copy(), y, f"MMSE <= {cfg.MMSE_INTERMEDIATE_MAX} (1) vs MMSE >= {cfg.MMSE_HIGH_MIN} (0), true AD only"
    if head == "T14":
        d = ad[ad["target_T20_mmse_le_20"].eq(1)].copy()
        y = d["target_T14_mmse_le_14"].astype(int).to_numpy()
        return d, y, f"MMSE <= {cfg.MMSE_LOW_MAX} (1) vs MMSE {cfg.MMSE_INTERMEDIATE_MIN}-{cfg.MMSE_INTERMEDIATE_MAX} (0), conditional on MMSE <= {cfg.MMSE_INTERMEDIATE_MAX} true AD only"
    raise ValueError(head)


def _matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Materialise the locked feature order as a contiguous numeric matrix.

    This does not alter values, aggregation, or columns; it prevents pandas block
    fragmentation from dominating repeated cross-validation fits.
    """
    return frame.loc[:, cols].to_numpy(dtype=float, copy=True)


def _positive_probability(estimator: Any, X: Any) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise RuntimeError("Stagev8.5 estimator must expose predict_proba.")
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise RuntimeError(f"Estimator lacks positive class 1: {classes}")
    return estimator.predict_proba(X)[:, classes.index(1)]


def _grid_cv(y: np.ndarray, seed: int, n_splits: int) -> StratifiedKFold:
    counts = pd.Series(y).value_counts()
    if int(counts.min()) < n_splits:
        raise RuntimeError(f"Cannot form {n_splits}-fold stratified CV; minority count={int(counts.min())}.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _fit_final_head(ad: pd.DataFrame, all_blocks: dict[str, list[str]], spec: SeverityHeadSpec, n_jobs: int, model_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    d, y, definition = _head_data(ad, spec.head)
    cv = _grid_cv(y, cfg.RANDOM_STATE, cfg.CV_N_SPLITS)
    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch="2*n_jobs",
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn\\.linear_model")
        search.fit(_matrix(d, all_blocks[spec.feature_block]), y)
    keep = [c for c in search.cv_results_ if c.startswith("param_") or c in {"mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time", "std_fit_time"}]
    grid = pd.DataFrame(search.cv_results_)[keep].copy()
    grid.insert(0, "head", spec.head)
    grid.insert(1, "model_key", spec.key)
    grid.insert(2, "family", spec.family)
    grid.insert(3, "feature_block", spec.feature_block)
    grid.insert(4, "target_definition", definition)
    grid = grid.sort_values(["rank_test_score", "mean_test_score"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    fitted = {"spec": spec, "estimator": search.best_estimator_, "best_params": search.best_params_, "cv_best_balanced_accuracy": float(search.best_score_), "definition": definition, "n_train": int(len(d)), "n_positive": int(y.sum()), "n_negative": int((1-y).sum())}
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, model_dir / f"stagev8_5_head_{spec.head}.joblib")
    return fitted, grid, {k: fitted[k] for k in ["definition", "n_train", "n_positive", "n_negative", "cv_best_balanced_accuracy", "best_params"]}


def _fit_fold_models(train_ad: pd.DataFrame, all_blocks: dict[str, list[str]], t20_spec: SeverityHeadSpec, t14_spec: SeverityHeadSpec, *, tune: bool, inner_seed: int, fixed_t20: Any | None = None, fixed_t14: Any | None = None, n_jobs: int = 1) -> tuple[Any, Any, dict[str, Any]]:
    """Fit both heads in one outer fold, preserving the conditional T14 training set."""
    d20, y20, _ = _head_data(train_ad, "T20")
    if tune:
        cv20 = _grid_cv(y20, inner_seed, cfg.NESTED_INNER_CV_N_SPLITS)
        s20 = GridSearchCV(clone(t20_spec.estimator), t20_spec.grid, scoring="balanced_accuracy", cv=cv20, n_jobs=n_jobs, pre_dispatch="2*n_jobs", refit=True, error_score="raise")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn\\.linear_model")
            s20.fit(_matrix(d20, all_blocks[t20_spec.feature_block]), y20)
        est20 = s20.best_estimator_
        p20 = s20.best_params_
    else:
        est20 = clone(fixed_t20)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn\\.linear_model")
            est20.fit(_matrix(d20, all_blocks[t20_spec.feature_block]), y20)
        p20 = None

    d14, y14, _ = _head_data(train_ad, "T14")
    if tune:
        cv14 = _grid_cv(y14, inner_seed, cfg.NESTED_INNER_CV_N_SPLITS)
        s14 = GridSearchCV(clone(t14_spec.estimator), t14_spec.grid, scoring="balanced_accuracy", cv=cv14, n_jobs=n_jobs, pre_dispatch="2*n_jobs", refit=True, error_score="raise")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn\\.linear_model")
            s14.fit(_matrix(d14, all_blocks[t14_spec.feature_block]), y14)
        est14 = s14.best_estimator_
        p14 = s14.best_params_
    else:
        est14 = clone(fixed_t14)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn\\.linear_model")
            est14.fit(_matrix(d14, all_blocks[t14_spec.feature_block]), y14)
        p14 = None
    return est20, est14, {"T20_params": p20, "T14_params": p14}


def _compose_scores(frame: pd.DataFrame, s20: np.ndarray, s14: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["s20_mmse_le_20_given_AD"] = np.asarray(s20, dtype=float)
    out["s14_mmse_le_14_given_mmse_le_20_AD"] = np.asarray(s14, dtype=float)
    out["q_high_mmse_given_AD"] = 1.0 - out["s20_mmse_le_20_given_AD"]
    out["q_intermediate_mmse_given_AD"] = out["s20_mmse_le_20_given_AD"] * (1.0 - out["s14_mmse_le_14_given_mmse_le_20_AD"])
    out["q_low_mmse_given_AD"] = out["s20_mmse_le_20_given_AD"] * out["s14_mmse_le_14_given_mmse_le_20_AD"]
    q_cols = ["q_high_mmse_given_AD", "q_intermediate_mmse_given_AD", "q_low_mmse_given_AD"]
    q = out[q_cols].to_numpy(dtype=float)
    labels = np.asarray(cfg.SEVERITY_STRATA)
    order = np.sort(q, axis=1)
    out["conditional_assigned_mmse_stratum"] = labels[np.argmax(q, axis=1)]
    out["severity_score"] = out["q_intermediate_mmse_given_AD"] + 2.0 * out["q_low_mmse_given_AD"]
    out["severity_confidence"] = order[:, -1]
    out["severity_margin"] = order[:, -1] - order[:, -2]
    safe_q = np.clip(q, 1e-12, 1.0)
    out["severity_entropy"] = -(safe_q * np.log(safe_q)).sum(axis=1)
    out["conditional_probability_sum"] = q.sum(axis=1)
    return out


def _nested_oof(ad: pd.DataFrame, all_blocks: dict[str, list[str]], t20_spec: SeverityHeadSpec, t14_spec: SeverityHeadSpec, n_jobs: int, progress: StageProgress | None = None) -> pd.DataFrame:
    outer = _grid_cv(ad["true_mmse_stratum"].astype(str).to_numpy(), cfg.RANDOM_STATE, cfg.CV_N_SPLITS)
    rows: list[pd.DataFrame] = []
    for fold, (tr, te) in enumerate(outer.split(ad, ad["true_mmse_stratum"].astype(str))):
        train_fold = ad.iloc[tr].copy()
        test_fold = ad.iloc[te].copy()
        est20, est14, params = _fit_fold_models(
            train_fold, all_blocks, t20_spec, t14_spec,
            tune=True, inner_seed=cfg.RANDOM_STATE + fold + 1,
            n_jobs=n_jobs,
        )
        p20 = _positive_probability(est20, _matrix(test_fold, all_blocks[t20_spec.feature_block]))
        p14 = _positive_probability(est14, _matrix(test_fold, all_blocks[t14_spec.feature_block]))
        z = _compose_scores(test_fold, p20, p14)
        z["outer_fold"] = int(fold)
        z["nested_T20_best_params"] = json.dumps(params["T20_params"], sort_keys=True, default=str)
        z["nested_T14_best_params"] = json.dumps(params["T14_params"], sort_keys=True, default=str)
        rows.append(z)
        if progress is not None:
            progress.event("progress", "nested OOF outer fold completed", completed_outer_fold=int(fold + 1), n_outer_folds=int(cfg.CV_N_SPLITS))
    out = pd.concat(rows, ignore_index=True).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if out["sample_id"].duplicated().any() or len(out) != len(ad):
        raise RuntimeError("Nested OOF construction failed to yield exactly one score per AD training sample.")
    return out


def _selected_head_oof(ad: pd.DataFrame, all_blocks: dict[str, list[str]], fitted: dict[str, Any], head: str, n_jobs: int) -> pd.DataFrame:
    d, y, definition = _head_data(ad, head)
    cv = _grid_cv(y, cfg.RANDOM_STATE, cfg.CV_N_SPLITS)
    p = np.full(len(d), np.nan)
    for tr, te in cv.split(d, y):
        est = clone(fitted["estimator"])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn\\.linear_model")
            est.fit(_matrix(d.iloc[tr], all_blocks[fitted["spec"].feature_block]), y[tr])
        p[te] = _positive_probability(est, _matrix(d.iloc[te], all_blocks[fitted["spec"].feature_block]))
    out = d[["sample_id", "mmse", "true_mmse_stratum", "target_T20_mmse_le_20", "target_T14_mmse_le_14"]].copy()
    out["head"] = head
    out["y_true"] = y
    out["oof_positive_score"] = p
    out["target_definition"] = definition
    return out


def _seed_stability(ad: pd.DataFrame, all_blocks: dict[str, list[str]], fitted20: dict[str, Any], fitted14: dict[str, Any], seeds: list[int], progress: StageProgress | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    head_rows: list[dict[str, Any]] = []
    ordinal_rows: list[dict[str, Any]] = []
    for seed in seeds:
        outer = _grid_cv(ad["true_mmse_stratum"].astype(str).to_numpy(), int(seed), cfg.CV_N_SPLITS)
        p20 = np.full(len(ad), np.nan)
        p14 = np.full(len(ad), np.nan)
        for tr, te in outer.split(ad, ad["true_mmse_stratum"].astype(str)):
            train_fold = ad.iloc[tr].copy()
            test_fold = ad.iloc[te].copy()
            est20, est14, _ = _fit_fold_models(
                train_fold,
                all_blocks,
                fitted20["spec"],
                fitted14["spec"],
                tune=False,
                inner_seed=int(seed),
                fixed_t20=fitted20["estimator"],
                fixed_t14=fitted14["estimator"],
                n_jobs=1,
            )
            p20[te] = _positive_probability(est20, _matrix(test_fold, all_blocks[fitted20["spec"].feature_block]))
            p14[te] = _positive_probability(est14, _matrix(test_fold, all_blocks[fitted14["spec"].feature_block]))
        z = _compose_scores(ad, p20, p14)
        m20 = binary_metrics(z["target_T20_mmse_le_20"], z["s20_mmse_le_20_given_AD"])
        cond = z[z["target_T20_mmse_le_20"].eq(1)]
        m14 = binary_metrics(cond["target_T14_mmse_le_14"], cond["s14_mmse_le_14_given_mmse_le_20_AD"])
        for head, meta, met in [("T20", fitted20, m20), ("T14", fitted14, m14)]:
            head_rows.append({"seed": int(seed), "head": head, "model_key": meta["spec"].key, "feature_block": meta["spec"].feature_block, "family": meta["spec"].family, **met})
        ord_met = ordinal_continuous_metrics(z["mmse"], z["severity_score"])
        tri = three_strata_metrics(z["true_mmse_stratum"], z["conditional_assigned_mmse_stratum"], cfg.SEVERITY_STRATA)
        ordinal_rows.append({"seed": int(seed), **ord_met, **{f"three_strata_{k}": v for k, v in tri.items()}})
        if progress is not None:
            progress.event("progress", "fixed-hyperparameter stability seed completed", completed_seed=int(seed), completed_seed_index=int(len(ordinal_rows)), n_seeds=int(len(seeds)))
    return pd.DataFrame(head_rows), pd.DataFrame(ordinal_rows)


def _attach_anchor_and_reports(external: pd.DataFrame, anchor_pred: pd.DataFrame, fitted20: dict[str, Any], fitted14: dict[str, Any], all_blocks: dict[str, list[str]]) -> pd.DataFrame:
    p20 = _positive_probability(fitted20["estimator"], _matrix(external, all_blocks[fitted20["spec"].feature_block]))
    p14 = _positive_probability(fitted14["estimator"], _matrix(external, all_blocks[fitted14["spec"].feature_block]))
    out = _compose_scores(external, p20, p14)
    out = out.merge(anchor_pred, on="sample_id", how="left", validate="one_to_one")
    out["predicted_AD"] = out["anchor_predicted_AD"].astype(int)
    out["severity_assigned_with_anchor"] = np.where(out["predicted_AD"].eq(1), out["conditional_assigned_mmse_stratum"], "control")
    can_report = out["predicted_AD"].eq(1) & out["severity_confidence"].ge(cfg.SEVERITY_CONFIDENCE_THRESHOLD) & out["severity_margin"].ge(cfg.SEVERITY_MARGIN_THRESHOLD)
    out["reported_severity_stratum"] = np.where(
        out["predicted_AD"].eq(0),
        "control",
        np.where(can_report, out["conditional_assigned_mmse_stratum"], "AD_severity_indeterminate"),
    )
    out["severity_reported"] = can_report.astype(int)
    out["anchor_admitted_true_AD"] = (out["__y__"].eq(1) & out["predicted_AD"].eq(1)).astype(int)
    out["model_based_score_control"] = 1.0 - out["anchor_p_AD"]
    out["model_based_score_high_mmse"] = out["anchor_p_AD"] * out["q_high_mmse_given_AD"]
    out["model_based_score_intermediate_mmse"] = out["anchor_p_AD"] * out["q_intermediate_mmse_given_AD"]
    out["model_based_score_low_mmse"] = out["anchor_p_AD"] * out["q_low_mmse_given_AD"]
    out["model_based_score_sum"] = out[["model_based_score_control", "model_based_score_high_mmse", "model_based_score_intermediate_mmse", "model_based_score_low_mmse"]].sum(axis=1)
    return out


def _external_metrics(pred: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    binary = binary_metrics(pred["__y__"], pred["anchor_p_AD"], cfg.ANCHOR_THRESHOLD)
    ad = pred[pred["__y__"].eq(1)].copy()
    admitted = ad[ad["predicted_AD"].eq(1)].copy()
    ordinal_all = ordinal_continuous_metrics(ad["mmse"], ad["severity_score"])
    ordinal_admitted = ordinal_continuous_metrics(admitted["mmse"], admitted["severity_score"])
    ordinal = {"scope_all_true_AD": ordinal_all, "scope_anchor_admitted_true_AD": ordinal_admitted}

    t20_all = binary_metrics(ad["target_T20_mmse_le_20"], ad["s20_mmse_le_20_given_AD"])
    t20_adm = binary_metrics(admitted["target_T20_mmse_le_20"], admitted["s20_mmse_le_20_given_AD"]) if len(admitted) else {}
    c14 = ad[ad["target_T20_mmse_le_20"].eq(1)].copy()
    c14_adm = admitted[admitted["target_T20_mmse_le_20"].eq(1)].copy()
    t14_all = binary_metrics(c14["target_T14_mmse_le_14"], c14["s14_mmse_le_14_given_mmse_le_20_AD"])
    t14_adm = binary_metrics(c14_adm["target_T14_mmse_le_14"], c14_adm["s14_mmse_le_14_given_mmse_le_20_AD"]) if len(c14_adm) else {}
    threshold = {"T20_all_true_AD": t20_all, "T20_anchor_admitted_true_AD": t20_adm, "T14_conditional_all_true_AD": t14_all, "T14_conditional_anchor_admitted_true_AD": t14_adm}

    tri_cond = three_strata_metrics(ad["true_mmse_stratum"], ad["conditional_assigned_mmse_stratum"], cfg.SEVERITY_STRATA)
    tri_anchored = three_strata_metrics(ad["true_mmse_stratum"], ad["severity_assigned_with_anchor"], cfg.SEVERITY_STRATA)
    tri_adm = three_strata_metrics(admitted["true_mmse_stratum"], admitted["conditional_assigned_mmse_stratum"], cfg.SEVERITY_STRATA) if len(admitted) else {}
    reported = admitted[admitted["reported_severity_stratum"].ne("AD_severity_indeterminate")].copy()
    tri_selective = three_strata_metrics(reported["true_mmse_stratum"], reported["reported_severity_stratum"], cfg.SEVERITY_STRATA) if len(reported) else {}
    selective = {
        **tri_selective,
        "n_true_AD_anchor_admitted": int(len(admitted)),
        "n_reported": int(len(reported)),
        "coverage": float(len(reported) / len(admitted)) if len(admitted) else float("nan"),
        "abstention_rate": float(1.0 - len(reported) / len(admitted)) if len(admitted) else float("nan"),
    }
    three = {"conditional_all_true_AD": tri_cond, "anchored_all_true_AD": tri_anchored, "anchor_admitted_true_AD": tri_adm}

    rows = []
    for g in cfg.SEVERITY_STRATA:
        x = admitted[admitted["true_mmse_stratum"].eq(g)]
        n = len(x)
        reported_n = int(x["reported_severity_stratum"].ne("AD_severity_indeterminate").sum()) if n else 0
        rows.append({"true_mmse_stratum": g, "n_anchor_admitted": int(n), "n_reported": reported_n, "coverage": reported_n / n if n else float("nan"), "abstention_rate": 1.0 - reported_n / n if n else float("nan")})
    return binary, ordinal, threshold, three, selective, pd.DataFrame(rows)


def _stratified_resample(pred: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pieces = []
    for _, group in pred.groupby("true_mmse_stratum", sort=False):
        idx = rng.integers(0, len(group), len(group))
        pieces.append(group.iloc[idx])
    return pd.concat(pieces, ignore_index=True)


def _bootstrap(pred: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    reps: list[dict[str, float]] = []
    for _ in range(n):
        sample = _stratified_resample(pred, rng)
        binary, ordinal, threshold, three, selective, _ = _external_metrics(sample)
        reps.append({
            "binary_accuracy": binary["accuracy"],
            "binary_balanced_accuracy": binary["balanced_accuracy"],
            "severity_spearman_rho": ordinal["scope_all_true_AD"]["spearman_rho"],
            "severity_kendall_tau": ordinal["scope_all_true_AD"]["kendall_tau"],
            "severity_pairwise_ordinal_accuracy": ordinal["scope_all_true_AD"]["pairwise_ordinal_accuracy"],
            "T20_balanced_accuracy": threshold["T20_all_true_AD"]["balanced_accuracy"],
            "T14_balanced_accuracy": threshold["T14_conditional_all_true_AD"]["balanced_accuracy"],
            "three_strata_macro_f1_anchor_admitted": three["anchor_admitted_true_AD"].get("macro_f1", np.nan),
            "selective_coverage_anchor_admitted": selective["coverage"],
        })
    reps_df = pd.DataFrame(reps)
    binary, ordinal, threshold, three, selective, _ = _external_metrics(pred)
    point = {
        "binary_accuracy": binary["accuracy"],
        "binary_balanced_accuracy": binary["balanced_accuracy"],
        "severity_spearman_rho": ordinal["scope_all_true_AD"]["spearman_rho"],
        "severity_kendall_tau": ordinal["scope_all_true_AD"]["kendall_tau"],
        "severity_pairwise_ordinal_accuracy": ordinal["scope_all_true_AD"]["pairwise_ordinal_accuracy"],
        "T20_balanced_accuracy": threshold["T20_all_true_AD"]["balanced_accuracy"],
        "T14_balanced_accuracy": threshold["T14_conditional_all_true_AD"]["balanced_accuracy"],
        "three_strata_macro_f1_anchor_admitted": three["anchor_admitted_true_AD"].get("macro_f1", np.nan),
        "selective_coverage_anchor_admitted": selective["coverage"],
    }
    return pd.DataFrame([
        {"metric": metric, "estimate": float(value), "ci_low": float(reps_df[metric].quantile(0.025)), "ci_high": float(reps_df[metric].quantile(0.975)), "bootstrap_n": int(n), "bootstrap_scheme": "stratified_by_true_mmse_stratum"}
        for metric, value in point.items()
    ])


def _fresh_feature_audit(root: Path) -> dict[str, Any]:
    path = cfg.CHECKS / "stagev8_5_fresh_stagev5_feature_rebuild_audit.json"
    if not path.exists():
        raise FileNotFoundError(f"Stagev8.5 training requires its fresh feature reconstruction audit: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Stagev8.5 fresh feature audit is not successful.")
    if int(audit.get("middle_new_cache_rows", 0)) <= 0 or int(audit.get("late_api_scored_samples", 0)) <= 0:
        raise RuntimeError("Stagev8.5 requires fresh BGE-M3 and late-LLM API extraction; audit reports no new calls.")
    if audit.get("raw_input_audit", {}).get("historical_feature_or_cache_input") is not False:
        raise RuntimeError("Stagev8.5 fresh feature audit indicates historical cache/feature input.")
    return {"path": str(path.resolve()), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(), "middle_new_cache_rows": int(audit["middle_new_cache_rows"]), "late_api_scored_samples": int(audit["late_api_scored_samples"]), "transport_audit": audit.get("transport_audit", {})}


def _write_common_audits(root: Path, final: Path, stagev6_root: Path | None = None):
    lock, parity, labels, data = build_feature_and_label_audits(root)
    direct = optional_stagev6_runtime_comparison(root, stagev6_root, data)
    fresh = _fresh_feature_audit(root)
    write_json(final / "stagev8_5_feature_lock.json", lock)
    write_json(final / "stagev8_5_stagev6_source_contract.json", {**source_contract(root), "optional_direct_runtime_comparison": direct})
    write_json(final / "stagev8_5_feature_parity_audit.json", parity)
    write_json(final / "stagev8_5_mmse_label_contract.json", labels)
    write_json(final / "stagev8_5_feature_source_audit.json", fresh)
    (final / "stagev8_5_literature_rationale.md").write_text(literature_rationale(), encoding="utf-8")
    no_external = {
        "status": "pass",
        "external_selection_prohibited": True,
        "predeclared_mmse_contract": {"high": f">={cfg.MMSE_HIGH_MIN}", "intermediate": f"{cfg.MMSE_INTERMEDIATE_MIN}-{cfg.MMSE_INTERMEDIATE_MAX}", "low": f"<={cfg.MMSE_LOW_MAX}"},
        "predeclared_model_families": {"T20": "E+M LR-ElasticNet", "T14": "L raw F8 LR-L2"},
        "predeclared_abstention_rule": {"confidence_min": cfg.SEVERITY_CONFIDENCE_THRESHOLD, "margin_min": cfg.SEVERITY_MARGIN_THRESHOLD},
        "permitted_internal_selection": "10-fold training-CV hyperparameter tuning within each fixed model family only",
        "external_role": "reference evaluation only",
    }
    write_json(final / "stagev8_5_no_external_selection_audit.json", no_external)
    return data, lock, parity, labels, direct, fresh


def self_check(root: Path, stagev6_root: Path | None = None) -> dict[str, Any]:
    from .raw_feature_rebuild import raw_input_audit
    source_integrity = verify_strict_reference_sources()
    raw = raw_input_audit(root)
    feature_files = list(cfg.FEATURE_ROOT.rglob("*.csv")) if cfg.FEATURE_ROOT.exists() else []
    payload: dict[str, Any] = {
        "status": "pass",
        "created_at": utc(),
        "input_mode": "raw_only",
        "strict_source_integrity": source_integrity,
        "raw_input_audit": raw,
        "generated_features_present": bool(feature_files),
        "next_step": "python .\\run_stagev8.py --mode check_api; python .\\run_stagev8.py --mode extract_features" if not feature_files else "python .\\run_stagev8.py --mode train_preflight",
        "no_api_or_feature_extraction": True,
        "stagev8_5_goal": "MMSE-informed ordinal severity output, not clinical early/middle/late staging",
    }
    if feature_files:
        _, _, labels, data = build_feature_and_label_audits(root)
        payload.update({"feature_counts": {"E": len(data.early_columns), "M": len(data.middle_columns), "L": len(data.late_columns)}, "mmse_label_contract": labels})
    write_json(cfg.CHECKS / "stagev8_5_self_check.json", payload)
    print(json.dumps(payload, indent=2, default=str))
    return payload


def anchor_check(root: Path) -> dict[str, Any]:
    _, _, _, data = build_feature_and_label_audits(root)
    anchor = load_frozen_anchor()
    audit, summary = anchor_audit(anchor, data.external)
    audit.to_csv(cfg.CHECKS / "stagev8_5_anchor_parity_audit.csv", index=False)
    write_json(cfg.CHECKS / "stagev8_5_anchor_check.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def train_preflight(root: Path, stagev6_root: Path | None = None) -> dict[str, Any]:
    _, _, labels, data = build_feature_and_label_audits(root)
    direct = optional_stagev6_runtime_comparison(root, stagev6_root, data)
    fresh = _fresh_feature_audit(root)
    anchor = load_frozen_anchor()
    _, anchor_summary = anchor_audit(anchor, data.external)
    specs = severity_head_specs()
    sentinel = cfg.FINAL / cfg.COMPLETION_SENTINEL
    ad = _ad_frame(data.train)
    counts = {head: _head_data(ad, head)[1] for head in specs}
    payload = {
        "status": "pass",
        "would_allow_training": not sentinel.exists(),
        "completed_run_detected": sentinel.exists(),
        "completion_sentinel": str(sentinel.resolve()),
        "feature_counts": {"E": len(data.early_columns), "M": len(data.middle_columns), "L": len(data.late_columns)},
        "mmse_label_counts": labels["train_mmse_stratum_counts"],
        "head_counts": {head: {"n": int(len(y)), "n_positive": int(y.sum()), "n_negative": int((1-y).sum())} for head, y in counts.items()},
        "anchor_parity": anchor_summary,
        "fresh_feature_source": fresh,
        "fixed_heads": {head: {"feature_block": spec.feature_block, "family": spec.family, "target_definition": spec.target_definition} for head, spec in specs.items()},
        "no_models_fitted": True,
        "no_api_or_feature_extraction": True,
        "optional_direct_stagev6_runtime_comparison": direct,
    }
    write_json(cfg.CHECKS / "stagev8_5_train_preflight.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def _clean_known_outputs(final: Path) -> None:
    for name in cfg.CANONICAL_FINAL_FILES + cfg.CANONICAL_FIGURES:
        path = final / name
        if path.exists():
            path.unlink()


def run_training(root: Path, n_jobs: int, bootstrap_n: int, seeds: list[int], force: bool = False, stagev6_root: Path | None = None) -> None:
    progress = StageProgress("train", total=13, root=root)
    try:
        final = cfg.FINAL
        final.mkdir(parents=True, exist_ok=True)
        cfg.MODELS.mkdir(parents=True, exist_ok=True)
        sentinel = final / cfg.COMPLETION_SENTINEL
        with progress_section(progress, "check Stagev8.5 completion sentinel and overwrite policy"):
            if sentinel.exists() and not force:
                raise FileExistsError(f"Completed Stagev8.5 run detected at {sentinel.resolve()}. Use --force only to replace known Stagev8.5 reports.")
            if force:
                _clean_known_outputs(final)
        with progress_section(progress, "write fresh-feature, Stagev6-loader, and MMSE-label audits"):
            data, lock, parity, labels, direct, fresh = _write_common_audits(root, final, stagev6_root)
            train = data.train.copy()
            external = data.external.copy()
            ad_train = _ad_frame(train)
            all_blocks = blocks(data)
        with progress_section(progress, "load frozen Stagev5 anchor and write parity audit"):
            anchor = load_frozen_anchor()
            anchor_df, anchor_summary = anchor_audit(anchor, external)
            anchor_df.to_csv(final / "stagev8_5_anchor_parity_audit.csv", index=False)
        specs = severity_head_specs()
        with progress_section(progress, "fit T20 E+M elastic-net head and rank internal hyperparameters"):
            fitted20, grid20, meta20 = _fit_final_head(ad_train, all_blocks, specs["T20"], n_jobs, cfg.MODELS)
            grid20.to_csv(final / "stagev8_5_T20_cv_grid.csv", index=False)
        with progress_section(progress, "fit T14 raw-F8 L2 head and rank internal hyperparameters"):
            fitted14, grid14, meta14 = _fit_final_head(ad_train, all_blocks, specs["T14"], n_jobs, cfg.MODELS)
            grid14.to_csv(final / "stagev8_5_T14_cv_grid.csv", index=False)
        with progress_section(progress, "write selected-head ten-fold OOF threshold scores"):
            _selected_head_oof(ad_train, all_blocks, fitted20, "T20", n_jobs).to_csv(final / "stagev8_5_T20_selected_oof.csv", index=False)
            _selected_head_oof(ad_train, all_blocks, fitted14, "T14", n_jobs).to_csv(final / "stagev8_5_T14_selected_oof.csv", index=False)
        with progress_section(progress, "compute nested ten-fold OOF ordinal severity scores"):
            nested = _nested_oof(ad_train, all_blocks, specs["T20"], specs["T14"], n_jobs, progress)
            nested.to_csv(final / "stagev8_5_nested_oof_severity_scores.csv", index=False)
            nested_ordinal = ordinal_continuous_metrics(nested["mmse"], nested["severity_score"])
            nested_t20 = binary_metrics(nested["target_T20_mmse_le_20"], nested["s20_mmse_le_20_given_AD"])
            nested_t14 = binary_metrics(nested.loc[nested["target_T20_mmse_le_20"].eq(1), "target_T14_mmse_le_14"], nested.loc[nested["target_T20_mmse_le_20"].eq(1), "s14_mmse_le_14_given_mmse_le_20_AD"])
            nested_tri = three_strata_metrics(nested["true_mmse_stratum"], nested["conditional_assigned_mmse_stratum"], cfg.SEVERITY_STRATA)
            write_json(final / "stagev8_5_nested_oof_metrics.json", {"ordinal": nested_ordinal, "T20": nested_t20, "T14_conditional": nested_t14, "three_strata": nested_tri, "outer_cv": 10, "inner_cv": 5})
        with progress_section(progress, "run 30-seed fixed-hyperparameter OOF stability audits", n_seeds=len(seeds)):
            head_stability, ordinal_stability = _seed_stability(ad_train, all_blocks, fitted20, fitted14, seeds, progress)
            head_stability.to_csv(final / "stagev8_5_head_seed_stability.csv", index=False)
            ordinal_stability.to_csv(final / "stagev8_5_ordinal_seed_stability.csv", index=False)
        with progress_section(progress, "compose external AD severity probabilities with frozen-anchor outputs"):
            pred = _attach_anchor_and_reports(external, anchor_df[["sample_id", "anchor_predicted_AD", "anchor_p_AD"]], fitted20, fitted14, all_blocks)
            pred.to_csv(final / "stagev8_5_external_severity_scores.csv", index=False)
            if float((pred["conditional_probability_sum"] - 1.0).abs().max()) > 1e-10:
                raise RuntimeError("Stagev8.5 conditional severity probabilities do not sum to one.")
            if float((pred["model_based_score_sum"] - 1.0).abs().max()) > 1e-10:
                raise RuntimeError("Stagev8.5 model-based score probabilities do not sum to one.")
        with progress_section(progress, "evaluate external ordinal, threshold, selective, and calibration metrics"):
            binary, ordinal, threshold, three, selective, abstention = _external_metrics(pred)
            pd.DataFrame([{"scope": scope, **metrics} for scope, metrics in ordinal.items()]).to_csv(final / "stagev8_5_external_ordinal_metrics.csv", index=False)
            pd.DataFrame([{"scope": scope, **metrics} for scope, metrics in threshold.items()]).to_csv(final / "stagev8_5_external_threshold_metrics.csv", index=False)
            pd.DataFrame([{"scope": scope, **metrics} for scope, metrics in three.items()]).to_csv(final / "stagev8_5_external_three_strata_metrics.csv", index=False)
            pd.DataFrame([binary]).to_csv(final / "stagev8_5_external_binary_metrics.csv", index=False)
            pd.DataFrame([selective]).to_csv(final / "stagev8_5_external_selective_metrics.csv", index=False)
            abstention.to_csv(final / "stagev8_5_external_abstention_by_stratum.csv", index=False)
            ad = pred[pred["__y__"].eq(1)]
            cal20 = calibration_table(ad["target_T20_mmse_le_20"], ad["s20_mmse_le_20_given_AD"])
            cond14 = ad[ad["target_T20_mmse_le_20"].eq(1)]
            cal14 = calibration_table(cond14["target_T14_mmse_le_14"], cond14["s14_mmse_le_14_given_mmse_le_20_AD"])
            cal20.to_csv(final / "stagev8_5_external_calibration_T20.csv", index=False)
            cal14.to_csv(final / "stagev8_5_external_calibration_T14.csv", index=False)
            admitted = ad[ad["predicted_AD"].eq(1)]
            confusion_long(admitted["true_mmse_stratum"], admitted["conditional_assigned_mmse_stratum"], cfg.SEVERITY_STRATA).to_csv(final / "stagev8_5_confusion_matrix_three_strata.csv", index=False)
        with progress_section(progress, "run stratified bootstrap confidence intervals", bootstrap_n=bootstrap_n):
            boot = _bootstrap(pred, bootstrap_n)
            boot.to_csv(final / "stagev8_5_bootstrap_ci.csv", index=False)
        selected = {
            "T20": {"model_key": fitted20["spec"].key, "feature_block": fitted20["spec"].feature_block, "family": fitted20["spec"].family, "best_params": fitted20["best_params"], "cv_best_balanced_accuracy": fitted20["cv_best_balanced_accuracy"]},
            "T14": {"model_key": fitted14["spec"].key, "feature_block": fitted14["spec"].feature_block, "family": fitted14["spec"].family, "best_params": fitted14["best_params"], "cv_best_balanced_accuracy": fitted14["cv_best_balanced_accuracy"]},
        }
        with progress_section(progress, "write Stagev8.5 model contract, reports, and figures"):
            model_contract = {"T20": {"feature_block": specs["T20"].feature_block, "family": specs["T20"].family, "grid": specs["T20"].grid, "target_definition": specs["T20"].target_definition, "rationale": specs["T20"].rationale}, "T14": {"feature_block": specs["T14"].feature_block, "family": specs["T14"].family, "grid": specs["T14"].grid, "target_definition": specs["T14"].target_definition, "rationale": specs["T14"].rationale}, "external_selection_prohibited": True}
            write_json(final / "stagev8_5_model_contract.json", model_contract)
            (final / "stagev8_5_selected_model_summary.md").write_text(selected_summary(binary, ordinal["scope_all_true_AD"], threshold["T20_all_true_AD"], threshold["T14_conditional_all_true_AD"], selected), encoding="utf-8")
            (final / "stagev8_5_experiment_report.md").write_text(experiment_report(selected), encoding="utf-8")
            make_figures(final, nested, pred, cal20, cal14, abstention, boot)
        run_summary = {
            "status": "complete",
            "created_at": utc(),
            "project": "Stagev8.5",
            "objective": "MMSE-informed ordinal cognitive-severity output; not clinical early/middle/late staging",
            "selected_components": selected,
            "n_train": int(len(train)),
            "n_train_AD": int(len(ad_train)),
            "n_external": int(len(external)),
            "n_external_AD": int((external["__y__"] == 1).sum()),
            "feature_counts": {"E": len(data.early_columns), "M": len(data.middle_columns), "L": len(data.late_columns)},
            "mmse_strata": {"high": f">={cfg.MMSE_HIGH_MIN}", "intermediate": f"{cfg.MMSE_INTERMEDIATE_MIN}-{cfg.MMSE_INTERMEDIATE_MAX}", "low": f"<={cfg.MMSE_LOW_MAX}"},
            "bootstrap_n": int(bootstrap_n),
            "stability_seeds": seeds,
            "external_set_role": "reference evaluation; prohibited from model/threshold/configuration selection",
            "binary_anchor_retrained": False,
            "feature_extraction_called": True,
            "api_called": True,
            "feature_source": fresh,
        }
        write_json(final / "stagev8_5_final_run_summary.json", run_summary)
        with progress_section(progress, "write final Stagev8.5 completion sentinel"):
            write_json(sentinel, run_summary)
        progress.done("Stagev8.5 training completed", final_report=str(final.resolve()))
        print(json.dumps({"status": "complete", "project": "Stagev8.5", "final_report": str(final.resolve()), "progress_log": str((root / "output" / "stagev8_runtime_progress.jsonl").resolve())}, indent=2))
    except Exception as exc:
        progress.fail("Stagev8.5 training failed", error=repr(exc))
        raise
