"""Run all stagev2 classifiers under external-accuracy selection."""
from __future__ import annotations

import json
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

import config
from bootstrap_ci import stratified_bootstrap_ci
from data_io import load_train_test
from error_analysis import write_error_tables
from evaluation import get_positive_prob, metrics_from_predictions, prediction_frame, repeated_cv_predict_proba
from feature_blocks import build_feature_blocks, infer_stage_columns, save_feature_manifest
from io_utils import save_json
from model_registry import build_model_specs
from paths import ensure_output_dirs
from report import choose_recommendations, write_report

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TOP_N_OOF = 10


def _select_block(blocks: dict, feature_block: str, split: str):
    return blocks[f"X_{split}_{feature_block}"]


def _jsonable_params(params: dict) -> str:
    def conv(v):
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v
    return json.dumps({k: conv(v) for k, v in params.items()}, ensure_ascii=False, sort_keys=True)


def _safe_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_").replace(" ", "_") + ".joblib"


def _stage_extra(estimator, X) -> dict:
    extra = {}
    if hasattr(estimator, "stage_scores"):
        scores = estimator.stage_scores(X)
        if len(scores) == 3:
            a, b, c = scores
            cls_name = estimator.__class__.__name__
            if cls_name == "MLPSVCLateCalibratedLR":
                extra.update({
                    "p_MLP": a,
                    "p_SVC": b,
                    "s_L": c,
                    "p_MLP_L": a * c,
                    "p_SVC_L": b * c,
                    "s_E": np.nan,
                    "s_M": np.nan,
                    "s_E_M": np.nan,
                    "s_M_L": np.nan,
                    "s_E_M_L": np.nan,
                })
            else:
                s_e, s_m, s_l = a, b, c
                extra.update({
                    "s_E": s_e,
                    "s_M": s_m,
                    "s_L": s_l,
                    "s_E_M": s_e * s_m,
                    "s_M_L": s_m * s_l,
                    "s_E_M_L": s_e * s_m * s_l,
                    "p_MLP": np.nan,
                    "p_SVC": np.nan,
                    "p_MLP_L": np.nan,
                    "p_SVC_L": np.nan,
                })
    else:
        for k in ["s_E", "s_M", "s_L", "s_E_M", "s_M_L", "s_E_M_L", "p_MLP", "p_SVC", "p_MLP_L", "p_SVC_L"]:
            extra[k] = np.nan
    return extra


def _prefixed_cv_metrics(y_train, oof_prob: np.ndarray) -> dict:
    out = {f"cv_{k}": v for k, v in metrics_from_predictions(y_train, oof_prob).items()}
    out["cv_repeated_oof_available"] = True
    return out


def run_one_model(spec, blocks, train_df, test_df, cv):
    X_train = _select_block(blocks, spec.feature_block, "train")
    X_test = _select_block(blocks, spec.feature_block, "test")
    y_train = blocks["y_train"]
    y_test = blocks["y_test"]

    start = time.time()
    grid = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        scoring=config.SCORING,
        cv=cv,
        n_jobs=config.N_JOBS,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    oof_prob = repeated_cv_predict_proba(best, X_train, y_train, cv)
    cv_metrics = _prefixed_cv_metrics(y_train, oof_prob)
    cv_row = {
        "model_name": spec.name,
        "group": spec.group,
        "feature_block": spec.feature_block,
        "mechanism_consistent": spec.mechanism_consistent,
        "best_cv_score_grid": grid.best_score_,
        "best_params": _jsonable_params(grid.best_params_),
        "fit_seconds": time.time() - start,
    }
    cv_row.update(cv_metrics)

    y_prob_test = get_positive_prob(best, X_test)
    ext = metrics_from_predictions(y_test, y_prob_test)
    external_row = {
        "model_name": spec.name,
        "group": spec.group,
        "feature_block": spec.feature_block,
        "mechanism_consistent": spec.mechanism_consistent,
        "best_params": _jsonable_params(grid.best_params_),
    }
    external_row.update(ext)

    test_pred = prediction_frame(test_df, y_prob_test, spec.name, extra=_stage_extra(best, X_test))
    train_pred = prediction_frame(train_df, oof_prob, spec.name)
    return cv_row, external_row, test_pred, train_pred, best


def make_generalization_gap(cv_results: pd.DataFrame, external_results: pd.DataFrame) -> pd.DataFrame:
    cv_cols = ["model_name", "cv_accuracy", "cv_balanced_accuracy", "cv_f1", "cv_roc_auc", "cv_pr_auc", "best_cv_score_grid"]
    ex_cols = ["model_name", "accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]
    gap = cv_results[cv_cols].merge(external_results[ex_cols], on="model_name", how="inner")
    gap = gap.rename(columns={
        "accuracy": "external_accuracy",
        "balanced_accuracy": "external_balanced_accuracy",
        "f1": "external_f1",
        "roc_auc": "external_roc_auc",
        "pr_auc": "external_pr_auc",
    })
    gap["accuracy_gap"] = gap["cv_accuracy"] - gap["external_accuracy"]
    gap["balanced_accuracy_gap"] = gap["cv_balanced_accuracy"] - gap["external_balanced_accuracy"]
    gap["roc_auc_gap"] = gap["cv_roc_auc"] - gap["external_roc_auc"]
    return gap.sort_values(["accuracy_gap", "balanced_accuracy_gap"], ascending=False)


def main() -> None:
    dirs = ensure_output_dirs()
    run_start = time.time()

    train_df, test_df, source_info = load_train_test(dirs)
    schema = infer_stage_columns(train_df, test_df)
    save_feature_manifest(schema, dirs["reports"] / "feature_manifest_used.json")
    blocks = build_feature_blocks(train_df, test_df, schema)

    run_config = {
        "random_state": config.RANDOM_STATE,
        "cv_n_splits": config.CV_N_SPLITS,
        "cv_n_repeats": config.CV_N_REPEATS,
        "inner_n_splits": config.INNER_N_SPLITS,
        "bootstrap_n": config.BOOTSTRAP_N,
        "scoring": config.SCORING,
        "selection_metric": "external_accuracy",
        "decision_threshold": config.DECISION_THRESHOLD,
        "n_early": schema["n_early"],
        "n_middle": schema["n_middle"],
        "n_late": schema["n_late"],
        "data_sources": source_info,
    }
    save_json(run_config, dirs["reports"] / "run_config.json")

    cv = RepeatedStratifiedKFold(n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS, random_state=config.RANDOM_STATE)
    specs = build_model_specs(blocks["n_early"], blocks["n_middle"], blocks["n_late"])

    cv_rows, ext_rows, pred_frames, train_frames, boot_frames = [], [], [], [], []
    fitted, errors = {}, []

    for spec in specs:
        print(f"[stagev2] Running {spec.name} ...", flush=True)
        try:
            cv_row, ext_row, pred, train_pred, best = run_one_model(spec, blocks, train_df, test_df, cv)
            cv_rows.append(cv_row)
            ext_rows.append(ext_row)
            pred_frames.append(pred)
            train_frames.append(train_pred)
            boot_frames.append(stratified_bootstrap_ci(blocks["y_test"], pred["p_ad"].values, spec.name))
            fitted[spec.name] = best
        except Exception as exc:
            errors.append({"model_name": spec.name, "error": repr(exc), "traceback": traceback.format_exc()})
            print(f"[stagev2] ERROR in {spec.name}: {exc}", flush=True)

    if not ext_rows:
        raise RuntimeError(f"No model completed successfully. Errors: {errors}")

    cv_results = pd.DataFrame(cv_rows).sort_values(["cv_accuracy", "cv_balanced_accuracy"], ascending=False)
    external_results = pd.DataFrame(ext_rows).sort_values(
        ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"],
        ascending=False,
    )
    predictions = pd.concat(pred_frames, ignore_index=True)
    train_predictions_all = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    top10_models = external_results["model_name"].head(TOP_N_OOF).tolist()
    train_predictions_top10 = train_predictions_all[train_predictions_all["model_name"].isin(top10_models)].copy()
    boot = pd.concat(boot_frames, ignore_index=True) if boot_frames else pd.DataFrame()
    gap = make_generalization_gap(cv_results, external_results)

    # Canonical output names.
    cv_results.to_csv(dirs["tables"] / "stagev2_cv_summary.csv", index=False)
    external_results.to_csv(dirs["tables"] / "stagev2_external_performance_report.csv", index=False)
    external_results.to_csv(dirs["tables"] / "stagev2_model_ranking_by_external_accuracy.csv", index=False)
    boot.to_csv(dirs["tables"] / "stagev2_bootstrap_ci.csv", index=False)
    gap.to_csv(dirs["tables"] / "stagev2_generalization_gap.csv", index=False)
    predictions.to_csv(dirs["predictions"] / "stagev2_test_predictions_all_models.csv", index=False)
    train_predictions_top10.to_csv(dirs["predictions"] / "stagev2_oof_predictions_top10.csv", index=False)
    cv_results[["model_name", "group", "feature_block", "best_cv_score_grid", "best_params"]].to_csv(dirs["tables"] / "stagev2_best_params.csv", index=False)

    # Save every fitted model locally; output/models is excluded from GitHub by .gitignore.
    for model_name, model in fitted.items():
        joblib.dump(model, dirs["models_all"] / _safe_model_filename(model_name))

    error_tables = write_error_tables(predictions, dirs)
    if errors:
        pd.DataFrame(errors).to_csv(dirs["logs"] / "stagev2_model_errors.csv", index=False)

    rec = choose_recommendations(external_results)
    write_report(dirs, external_results, cv_results, gap, error_tables, rec, schema, source_info)

    selected = rec["final_recommended_model"]
    if selected in fitted:
        joblib.dump(fitted[selected], dirs["models_selected"] / ("selected__" + _safe_model_filename(selected)))

    leakage = {
        "standard_scaler_inside_pipeline": True,
        "imputer_inside_pipeline": True,
        "stage_scores_inner_oof": True,
        "mlp_svc_late_inner_oof": True,
        "external_accuracy_used_for_model_selection": True,
        "external_set_role": "held-out external validation, not unbiased final test",
        "mmse_used_as_input_feature": False,
        "tfidf_used": False,
        "rbf_svm_used": True,
        "historical_outputs_reused": False,
        "run_seconds": time.time() - run_start,
    }
    save_json(leakage, dirs["reports"] / "stagev2_leakage_check.json")
    with (dirs["reports"] / "stagev2_leakage_check.md").open("w", encoding="utf-8") as f:
        for k, v in leakage.items():
            f.write(f"- {k}: {v}\n")

    print("[stagev2] Done.")
    print(f"[stagev2] Results: {dirs['root']}")


if __name__ == "__main__":
    main()
