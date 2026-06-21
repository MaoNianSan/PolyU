"""Training and reporting for the fixed stagev6 late-first cascade panel."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from . import config as cfg
from .data_loader import LoadedData, feature_validation_summary, load_stagev5_features
from .metrics import (
    component_oof_probabilities,
    metrics_from_hard_and_probability,
    positive_probability,
    stratified_bootstrap_ci,
)
from .model_specs import ComponentSpec, branch_specs, cascade_name, gate_specs
from .reporting import copy_final, save_figures, write_markdown_reports

# sklearn >= 1.8 emits a deprecation warning for an explicit l2 penalty.
# Stagev6 retains the stagev5-compatible estimator declaration; suppress only this non-actionable warning.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*penalty.*")


def _jsonable_params(params: dict[str, Any]) -> str:
    def convert(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v
    return json.dumps({k: convert(v) for k, v in params.items()}, ensure_ascii=False, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_dirs(overwrite: bool) -> dict[str, Path]:
    if overwrite:
        for path in [cfg.RUN, cfg.FINAL]:
            if path.exists():
                shutil.rmtree(path)
    dirs = {
        "run": cfg.RUN,
        "tables": cfg.RUN / "tables",
        "predictions": cfg.RUN / "predictions",
        "reports": cfg.RUN / "reports",
        "models": cfg.RUN / "models",
        "components": cfg.RUN / "models" / "components",
        "selected": cfg.RUN / "models" / "selected",
        "logs": cfg.RUN / "logs",
    }
    for path in [*dirs.values(), cfg.FINAL, cfg.FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _feature_blocks(data: LoadedData) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    tr, ex = data.train, data.external
    blocks_train = {
        "late": tr[data.late_columns].copy(),
        "middle_late": tr[[*data.middle_columns, *data.late_columns]].copy(),
        "early_middle": tr[[*data.early_columns, *data.middle_columns]].copy(),
    }
    blocks_external = {
        "late": ex[data.late_columns].copy(),
        "middle_late": ex[[*data.middle_columns, *data.late_columns]].copy(),
        "early_middle": ex[[*data.early_columns, *data.middle_columns]].copy(),
    }
    return blocks_train, blocks_external


def _fit_component(
    spec: ComponentSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: RepeatedStratifiedKFold,
    n_jobs: int,
) -> tuple[object, dict[str, Any]]:
    started = time.time()
    grid = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        scoring=spec.scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    grid.fit(X, y)
    row = {
        "component_id": spec.component_id,
        "role": spec.role,
        "feature_block": spec.feature_block,
        "classifier": spec.classifier,
        "scoring": spec.scoring,
        "n_training_samples": int(len(y)),
        "n_positive_training_samples": int(np.sum(y)),
        "best_cv_score_grid": float(grid.best_score_),
        "best_params": _jsonable_params(grid.best_params_),
        "fit_seconds": float(time.time() - started),
    }
    return grid.best_estimator_, row


def _cascade_prediction_frame(
    metadata: pd.DataFrame,
    model_name: str,
    gate_id: str,
    branch_id: str,
    p_late: np.ndarray,
    p_branch: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    y = metadata["__y__"].to_numpy(dtype=int)
    is_late = metadata["label_late"].to_numpy(dtype=int)
    p_late = np.clip(np.asarray(p_late, dtype=float), 1e-6, 1 - 1e-6)
    p_branch = np.clip(np.asarray(p_branch, dtype=float), 1e-6, 1 - 1e-6)
    late_route = (p_late >= cfg.DECISION_THRESHOLD).astype(int)
    branch_pred = (p_branch >= cfg.DECISION_THRESHOLD).astype(int)
    y_pred = np.where(late_route == 1, 1, branch_pred).astype(int)
    p_mixture = p_late + (1.0 - p_late) * p_branch
    final_route = np.where(late_route == 1, "late_direct_AD", np.where(branch_pred == 1, "nonlate_branch_AD", "nonlate_branch_control"))

    route_type: list[str] = []
    for true_y, true_late, direct_late, final in zip(y, is_late, late_route, y_pred):
        if true_y == 0:
            route_type.append("control_false_late_route" if direct_late else ("control_branch_FP" if final else "correct_control"))
        elif true_late == 1:
            route_type.append("correct_late_direct_AD" if direct_late else ("late_missed_but_branch_recovered" if final else "late_missed_final_FN"))
        else:
            route_type.append("nonlate_AD_overrouted_late" if direct_late else ("correct_nonlate_branch_AD" if final else "nonlate_AD_branch_FN"))

    out = pd.DataFrame({
        "sample_id": metadata["sample_id"].astype(str).to_numpy(),
        "split": split_name,
        "y_true": y,
        "label_disease": metadata["label_disease"].astype(int).to_numpy(),
        "label_early": metadata["label_early"].astype(int).to_numpy(),
        "label_middle": metadata["label_middle"].astype(int).to_numpy(),
        "label_late": is_late,
        "label_normal": metadata["label_normal"].astype(int).to_numpy(),
        "label_mild": metadata["label_mild"].astype(int).to_numpy(),
        "label_moderate": metadata["label_moderate"].astype(int).to_numpy(),
        "label_severe": metadata["label_severe"].astype(int).to_numpy(),
        "severity_group": metadata["severity_group"].astype(str).to_numpy(),
        "mmse": metadata["mmse"].to_numpy(),
        "model_name": model_name,
        "gate_id": gate_id,
        "branch_id": branch_id,
        "p_late": p_late,
        "late_route": late_route,
        "p_ad_given_nonlate": p_branch,
        "branch_y_pred": branch_pred,
        "p_ad_mixture": p_mixture,
        "p_ad": p_mixture,
        "final_route": final_route,
        "y_pred": y_pred,
        "correct": (y_pred == y).astype(int),
        "route_error_type": route_type,
    })
    out["error_type"] = np.where(
        out["correct"].eq(1),
        "correct",
        np.where(out["y_true"].eq(0), "FP_normal", "FN_" + out["severity_group"].astype(str)),
    )
    return out


def _component_metric_rows(
    role: str,
    component_id: str,
    y: np.ndarray,
    p: np.ndarray,
    split: str,
    evaluation_subset: str,
) -> dict[str, Any]:
    pred = (np.asarray(p) >= cfg.DECISION_THRESHOLD).astype(int)
    row = {
        "component_id": component_id,
        "role": role,
        "split": split,
        "evaluation_subset": evaluation_subset,
        "n": int(len(y)),
        "threshold": cfg.DECISION_THRESHOLD,
    }
    row.update(metrics_from_hard_and_probability(y, pred, p))
    return row


def _write_table_both(frame: pd.DataFrame, dirs: dict[str, Path], filename: str, location: str = "tables") -> None:
    run_path = dirs[location] / filename
    frame.to_csv(run_path, index=False)
    copy_final(run_path, filename)


def _route_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    total = (
        predictions.groupby(["model_name", "gate_id", "branch_id", "route_error_type"], dropna=False)
        .size().reset_index(name="n")
    )
    total["proportion"] = total["n"] / total.groupby("model_name")["n"].transform("sum")
    route_counts = (
        predictions.groupby(["model_name", "gate_id", "branch_id", "final_route"], dropna=False)
        .size().reset_index(name="n")
    )
    route_counts["proportion"] = route_counts["n"] / route_counts.groupby("model_name")["n"].transform("sum")
    total["diagnostic_type"] = "route_error_type"
    route_counts = route_counts.rename(columns={"final_route": "route_error_type"})
    route_counts["diagnostic_type"] = "final_route"
    return pd.concat([total, route_counts], ignore_index=True)


def _stage_subgroup(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["model_name", "severity_group"], dropna=False)
        .agg(
            n=("sample_id", "size"),
            accuracy=("correct", "mean"),
            mean_p_ad=("p_ad", "mean"),
            mean_p_late=("p_late", "mean"),
            late_route_rate=("late_route", "mean"),
            false_negatives=("error_type", lambda x: int(x.astype(str).str.startswith("FN_").sum())),
            false_positives=("error_type", lambda x: int(x.astype(str).eq("FP_normal").sum())),
        ).reset_index()
    )


def _component_specification_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["role", "component_id"]).reset_index(drop=True)


def train_stagev6(n_jobs: str | int = "-1", bootstrap_n: int = cfg.BOOTSTRAP_N, overwrite: bool = False) -> dict[str, Any]:
    n_jobs_int = int(n_jobs)
    dirs = _ensure_dirs(overwrite=overwrite)
    started = time.time()
    validation = feature_validation_summary()
    data = load_stagev5_features()
    X_train_blocks, X_external_blocks = _feature_blocks(data)

    y_ad = data.train["__y__"].to_numpy(dtype=int)
    y_late = data.train["label_late"].to_numpy(dtype=int)
    nonlate_mask = (y_late == 0)
    y_branch = y_ad[nonlate_mask]
    route_strata = np.where(y_late == 1, "late_AD", np.where(y_ad == 1, "nonlate_AD", "control"))
    if len(np.unique(route_strata)) != 3:
        raise RuntimeError("Stagev6 requires control, nonlate AD, and late AD samples in training data.")

    gate_cv = RepeatedStratifiedKFold(n_splits=cfg.CV_N_SPLITS, n_repeats=cfg.CV_N_REPEATS, random_state=cfg.RANDOM_STATE)
    branch_cv = RepeatedStratifiedKFold(n_splits=cfg.CV_N_SPLITS, n_repeats=cfg.CV_N_REPEATS, random_state=cfg.RANDOM_STATE)
    outer_cv = RepeatedStratifiedKFold(n_splits=cfg.CV_N_SPLITS, n_repeats=cfg.CV_N_REPEATS, random_state=cfg.RANDOM_STATE)

    all_gate_specs, all_branch_specs = gate_specs(), branch_specs()
    fitted_gates: dict[str, object] = {}
    fitted_branches: dict[str, object] = {}
    component_rows: list[dict[str, Any]] = []

    print("[stagev6] Fitting late-gate components ...", flush=True)
    for spec in all_gate_specs:
        print(f"[stagev6] gate {spec.component_id}", flush=True)
        model, row = _fit_component(spec, X_train_blocks[spec.feature_block], y_late, gate_cv, n_jobs_int)
        fitted_gates[spec.component_id] = model
        component_rows.append(row)
        joblib.dump(model, dirs["components"] / f"{spec.component_id}.joblib")

    print("[stagev6] Fitting non-late branch components ...", flush=True)
    for spec in all_branch_specs:
        print(f"[stagev6] branch {spec.component_id}", flush=True)
        model, row = _fit_component(spec, X_train_blocks[spec.feature_block].iloc[nonlate_mask], y_branch, branch_cv, n_jobs_int)
        fitted_branches[spec.component_id] = model
        component_rows.append(row)
        joblib.dump(model, dirs["components"] / f"{spec.component_id}.joblib")

    component_specs = _component_specification_frame(component_rows)
    _write_table_both(component_specs, dirs, "stagev6_component_specifications.csv")

    gate_train_X = {s.component_id: X_train_blocks[s.feature_block] for s in all_gate_specs}
    branch_train_X = {s.component_id: X_train_blocks[s.feature_block] for s in all_branch_specs}
    gate_oof, branch_oof = component_oof_probabilities(
        fitted_gates, fitted_branches, gate_train_X, branch_train_X,
        y_late=y_late, y_ad=y_ad, nonlate_mask=nonlate_mask, outer_cv=outer_cv, strata=route_strata,
    )
    gate_external = {s.component_id: positive_probability(fitted_gates[s.component_id], X_external_blocks[s.feature_block]) for s in all_gate_specs}
    branch_external = {s.component_id: positive_probability(fitted_branches[s.component_id], X_external_blocks[s.feature_block]) for s in all_branch_specs}

    # Component diagnostics preserve the distinction between late gating and non-late disease classification.
    gate_component_rows: list[dict[str, Any]] = []
    branch_component_rows: list[dict[str, Any]] = []
    y_late_external = data.external["label_late"].to_numpy(dtype=int)
    y_ad_external = data.external["__y__"].to_numpy(dtype=int)
    ext_nonlate_mask = (y_late_external == 0)
    for spec in all_gate_specs:
        gate_component_rows.extend([
            _component_metric_rows("late_gate", spec.component_id, y_late, gate_oof[spec.component_id], "oof", "all_train"),
            _component_metric_rows("late_gate", spec.component_id, y_late_external, gate_external[spec.component_id], "external", "all_external"),
        ])
    for spec in all_branch_specs:
        branch_component_rows.extend([
            _component_metric_rows("nonlate_branch", spec.component_id, y_ad[nonlate_mask], branch_oof[spec.component_id][nonlate_mask], "oof", "true_nonlate_train"),
            _component_metric_rows("nonlate_branch", spec.component_id, y_ad_external[ext_nonlate_mask], branch_external[spec.component_id][ext_nonlate_mask], "external", "true_nonlate_external"),
        ])
    gate_component_df = pd.DataFrame(gate_component_rows)
    branch_component_df = pd.DataFrame(branch_component_rows)
    _write_table_both(gate_component_df, dirs, "stagev6_late_gate_performance.csv")
    _write_table_both(branch_component_df, dirs, "stagev6_nonlate_branch_performance.csv")

    cv_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    external_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []

    for gate_spec in all_gate_specs:
        for branch_spec in all_branch_specs:
            name = cascade_name(gate_spec.component_id, branch_spec.component_id)
            oof_pred = _cascade_prediction_frame(
                data.train, name, gate_spec.component_id, branch_spec.component_id,
                gate_oof[gate_spec.component_id], branch_oof[branch_spec.component_id], "train_oof",
            )
            external_pred = _cascade_prediction_frame(
                data.external, name, gate_spec.component_id, branch_spec.component_id,
                gate_external[gate_spec.component_id], branch_external[branch_spec.component_id], "external",
            )
            cv = metrics_from_hard_and_probability(oof_pred["y_true"], oof_pred["y_pred"], oof_pred["p_ad_mixture"])
            ext = metrics_from_hard_and_probability(external_pred["y_true"], external_pred["y_pred"], external_pred["p_ad_mixture"])
            component_lookup = component_specs.set_index("component_id")
            cv_row = {
                "model_name": name,
                "group": "late_first_cascade",
                "feature_block": f"{gate_spec.feature_block}__to__{branch_spec.feature_block}",
                "gate_id": gate_spec.component_id,
                "branch_id": branch_spec.component_id,
                "gate_best_params": component_lookup.loc[gate_spec.component_id, "best_params"],
                "branch_best_params": component_lookup.loc[branch_spec.component_id, "best_params"],
                "gate_best_cv_score_grid": component_lookup.loc[gate_spec.component_id, "best_cv_score_grid"],
                "branch_best_cv_score_grid": component_lookup.loc[branch_spec.component_id, "best_cv_score_grid"],
                "cv_repeated_oof_available": True,
            }
            cv_row.update({f"cv_{k}": v for k, v in cv.items()})
            ext_row = {
                "model_name": name,
                "group": "late_first_cascade",
                "feature_block": f"{gate_spec.feature_block}__to__{branch_spec.feature_block}",
                "gate_id": gate_spec.component_id,
                "branch_id": branch_spec.component_id,
                "gate_best_params": component_lookup.loc[gate_spec.component_id, "best_params"],
                "branch_best_params": component_lookup.loc[branch_spec.component_id, "best_params"],
            }
            ext_row.update(ext)
            cv_rows.append(cv_row)
            external_rows.append(ext_row)
            oof_frames.append(oof_pred)
            external_frames.append(external_pred)
            bootstrap_frames.append(stratified_bootstrap_ci(
                external_pred["y_true"].to_numpy(), external_pred["y_pred"].to_numpy(),
                external_pred["p_ad_mixture"].to_numpy(), name, bootstrap_n,
            ))

            joblib.dump(
                {
                    "stage_version": "stagev6",
                    "model_name": name,
                    "gate_id": gate_spec.component_id,
                    "branch_id": branch_spec.component_id,
                    "gate_estimator": fitted_gates[gate_spec.component_id],
                    "branch_estimator": fitted_branches[branch_spec.component_id],
                    "hard_route_threshold": cfg.DECISION_THRESHOLD,
                    "p_ad_mixture": "p_late + (1-p_late)*p_ad_given_nonlate",
                },
                dirs["models"] / f"{name}.joblib",
            )

    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_accuracy", "cv_balanced_accuracy"], ascending=False).reset_index(drop=True)
    external_df = pd.DataFrame(external_rows).sort_values(
        ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"],
        ascending=False,
    ).reset_index(drop=True)
    oof_df = pd.concat(oof_frames, ignore_index=True)
    external_df_predictions = pd.concat(external_frames, ignore_index=True)
    bootstrap_df = pd.concat(bootstrap_frames, ignore_index=True)

    gap = cv_df[["model_name", "cv_accuracy", "cv_balanced_accuracy", "cv_f1", "cv_roc_auc", "cv_pr_auc"]].merge(
        external_df[["model_name", "accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]], on="model_name", how="inner"
    ).rename(columns={
        "accuracy": "external_accuracy", "balanced_accuracy": "external_balanced_accuracy",
        "f1": "external_f1", "roc_auc": "external_roc_auc", "pr_auc": "external_pr_auc",
    })
    gap["accuracy_gap"] = gap["cv_accuracy"] - gap["external_accuracy"]
    gap["balanced_accuracy_gap"] = gap["cv_balanced_accuracy"] - gap["external_balanced_accuracy"]
    gap["roc_auc_gap"] = gap["cv_roc_auc"] - gap["external_roc_auc"]
    gap = gap.sort_values(["accuracy_gap", "balanced_accuracy_gap"], ascending=False)

    subgroup = _stage_subgroup(external_df_predictions)
    route = _route_diagnostics(external_df_predictions)
    selected = str(external_df.iloc[0]["model_name"])
    errors = external_df_predictions.loc[
        external_df_predictions.model_name.eq(selected) & external_df_predictions.correct.eq(0)
    ].sort_values(["p_ad_mixture", "p_late"])

    _write_table_both(external_df, dirs, "stagev6_model_ranking_by_external_accuracy.csv")
    _write_table_both(external_df, dirs, "stagev6_external_performance_report.csv")
    _write_table_both(cv_df, dirs, "stagev6_cv_summary.csv")
    _write_table_both(bootstrap_df, dirs, "stagev6_bootstrap_ci.csv")
    _write_table_both(gap, dirs, "stagev6_generalization_gap.csv")
    _write_table_both(oof_df, dirs, "stagev6_oof_predictions_top6.csv", location="predictions")
    _write_table_both(external_df_predictions, dirs, "stagev6_test_predictions_all_models.csv", location="predictions")
    _write_table_both(subgroup, dirs, "stagev6_stage_subgroup_accuracy.csv")
    _write_table_both(route, dirs, "stagev6_route_diagnostics.csv")
    _write_table_both(errors, dirs, "stagev6_error_analysis.csv")

    source_feature_manifests: dict[str, Any] = {}
    for file_name in ["E_M_extraction_manifest.json", "L_extraction_manifest.json"]:
        manifest_path = cfg.FEATURE_ROOT / file_name
        source_feature_manifests[file_name] = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_manifest = {
        "stage_version": "stagev6",
        "inherits_feature_files_from": "stagev5",
        "feature_validation": validation,
        "feature_file_sha256": {key: _sha256(Path(value)) for key, value in validation["source_paths"].items()},
        "source_feature_manifests": source_feature_manifests,
        "late_auxiliary_features_used_in_model": False,
        "mmse_used_as_input_feature": False,
        "middle_sample_aggregation": "mean by sample_id over stagev5 BGE-M3 window-level rows",
    }
    (dirs["reports"] / "stagev6_feature_source_manifest.json").write_text(json.dumps(feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_final(dirs["reports"] / "stagev6_feature_source_manifest.json", "stagev6_feature_source_manifest.json")

    leakage = {
        "standard_scaler_inside_each_pipeline": True,
        "median_imputer_inside_each_pipeline": True,
        "late_gate_grid_scoring": "balanced_accuracy",
        "nonlate_branch_grid_scoring": "accuracy",
        "late_gate_training_label": "late vs non-late",
        "nonlate_branch_training_data": "true non-late samples only",
        "cascade_oof_split_strata": ["control", "nonlate_AD", "late_AD"],
        "component_hyperparameters_selected_on_training_data_only": True,
        "external_set_used_for_fit_or_preprocessing": False,
        "external_accuracy_used_for_final_model_selection": True,
        "external_set_role": "held-out external validation, not unbiased final test after external ranking",
        "feature_extraction_or_api_called_in_train": False,
        "mmse_used_as_input_feature": False,
        "hard_decision_rule": "late_route if p_late >= 0.50, otherwise nonlate branch at 0.50",
        "run_seconds": float(time.time() - started),
    }
    (dirs["reports"] / "stagev6_leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_final(dirs["reports"] / "stagev6_leakage_check.json", "stagev6_leakage_check.json")

    figures = save_figures(external_df, gap, external_df_predictions, subgroup, bootstrap_df)
    summary = {
        "selected_model": selected,
        "gate_id": str(external_df.iloc[0]["gate_id"]),
        "branch_id": str(external_df.iloc[0]["branch_id"]),
        "external_accuracy": float(external_df.iloc[0]["accuracy"]),
        "external_balanced_accuracy": float(external_df.iloc[0]["balanced_accuracy"]),
        "n_cascade_models_completed": int(len(external_df)),
        "selection_metric": "external_accuracy",
        "bootstrap_n": int(bootstrap_n),
        "figures": figures,
        "feature_validation": validation,
    }
    (dirs["reports"] / "stagev6_final_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_final(dirs["reports"] / "stagev6_final_run_summary.json", "stagev6_final_run_summary.json")
    write_markdown_reports(external_df, component_specs, route, feature_manifest, figures)

    # Selected composite artifact is saved after final ranking, exactly as the report indicates.
    selected_row = external_df.iloc[0]
    joblib.dump(
        {
            "stage_version": "stagev6",
            "model_name": selected,
            "gate_id": str(selected_row["gate_id"]),
            "branch_id": str(selected_row["branch_id"]),
            "gate_estimator": fitted_gates[str(selected_row["gate_id"])],
            "branch_estimator": fitted_branches[str(selected_row["branch_id"])],
            "decision_threshold": cfg.DECISION_THRESHOLD,
            "feature_blocks": {
                "gate": next(s.feature_block for s in all_gate_specs if s.component_id == str(selected_row["gate_id"])),
                "branch": next(s.feature_block for s in all_branch_specs if s.component_id == str(selected_row["branch_id"])),
            },
        },
        dirs["selected"] / f"selected__{selected}.joblib",
    )

    print("[stagev6] TRAINING AND REPORTING COMPLETED", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
