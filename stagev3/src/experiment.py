from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from . import config
from .evaluation import predict_with_scores, score_model
from .models import ModelSpec, make_estimator, model_specs, param_grid
from .normalization import safe_numeric_frame


def _n_splits(cv_mode: str) -> int:
    if cv_mode == "v2_repeated":
        return config.N_SPLITS_V2_REPEATED
    return config.N_SPLITS_EXACT if cv_mode == "exact" else config.N_SPLITS_FAST


def _cv_iterator(cv_mode: str, y_train, seed: int):
    if cv_mode == "v2_repeated":
        return RepeatedStratifiedKFold(
            n_splits=config.N_SPLITS_V2_REPEATED,
            n_repeats=config.N_REPEATS_V2_REPEATED,
            random_state=seed,
        )
    return StratifiedKFold(n_splits=_n_splits(cv_mode), shuffle=True, random_state=seed)


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def _select_best(candidates: list[dict]) -> dict:
    return sorted(candidates, key=lambda r: (r["cv_accuracy"], np.nan_to_num(r["cv_auc"], nan=-1), r["cv_f1"]), reverse=True)[0]


def _fast_proxy_metrics(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test) -> tuple[dict, dict]:
    Xtr = X_train.copy().apply(pd.to_numeric, errors="coerce")
    Xte = X_test.copy().apply(pd.to_numeric, errors="coerce")
    med = Xtr.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = Xtr.std(axis=0).replace([np.inf, -np.inf], np.nan).fillna(1.0).replace(0.0, 1.0)
    Ztr = ((Xtr.fillna(med) - med) / std).to_numpy(float)
    Zte = ((Xte.fillna(med) - med) / std).to_numpy(float)
    y = np.asarray(y_train).astype(int)
    if len(np.unique(y)) < 2:
        w = np.zeros(Ztr.shape[1])
    else:
        w = Ztr[y == 1].mean(axis=0) - Ztr[y == 0].mean(axis=0)
    s_tr = Ztr @ w
    s_te = Zte @ w
    thr = float(np.median(s_tr)) if len(s_tr) else 0.0
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
    def met(y_true, scores):
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = float('nan')
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "auc": float(auc),
            "correct_n": int(np.sum(pred == np.asarray(y_true))),
            "total_n": int(len(y_true)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    return met(y_train, s_tr), met(y_test, s_te)


def _append_external_predictions(
    sink: list[dict] | None,
    estimator,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    external_meta: pd.DataFrame | None,
    seed: int,
    cv_mode: str,
    early_variant: str,
    spec,
) -> None:
    if sink is None or estimator is None:
        return
    y_pred, y_score = predict_with_scores(estimator, X_test)
    meta = external_meta.reset_index(drop=True).copy() if external_meta is not None else pd.DataFrame(index=range(len(y_test)))
    for i in range(len(y_test)):
        sink.append({
            "seed": int(seed),
            "cv_mode": cv_mode,
            "early_variant": early_variant,
            "model_spec_id": spec.model_spec_id,
            "feature_block": spec.feature_block,
            "model_name": spec.model_name,
            "model_variant": spec.model_variant,
            "is_special_model": bool(spec.is_special),
            "row_index": int(i),
            "sample_id": str(meta.loc[i, "sample_id"]) if "sample_id" in meta.columns else str(i),
            "mmse": meta.loc[i, "mmse"] if "mmse" in meta.columns else np.nan,
            "y_true": int(y_test[i]),
            "y_pred": int(y_pred[i]),
            "y_score": float(y_score[i]),
            "correct": int(y_pred[i] == y_test[i]),
        })


def run_protocol_for_seed(
    blocks: dict,
    y_train,
    y_test,
    early_variant: str,
    seed: int,
    cv_mode: str = "exact",
    progress_callback: Callable[[dict], None] | None = None,
    external_meta: pd.DataFrame | None = None,
    prediction_sink: list[dict] | None = None,
    specs: list[ModelSpec] | None = None,
) -> pd.DataFrame:
    rows = []
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    specs = model_specs() if specs is None else list(specs)
    stage_dims = blocks.get("__stage_dims__")
    for index, spec in enumerate(specs, start=1):
        if progress_callback:
            progress_callback({
                "event": "started",
                "model_spec_progress": index,
                "model_spec_total": len(specs),
                "current_seed": int(seed),
                "current_early_variant": early_variant,
                "current_feature_block": spec.feature_block,
                "current_model_name": spec.model_name,
                "current_model_spec_id": spec.model_spec_id,
                "current_cv_mode": cv_mode,
            })
        if spec.feature_block not in blocks:
            raise KeyError(f"Feature block missing for model spec {spec.model_spec_id}: {spec.feature_block}")
        X_train, X_test = blocks[spec.feature_block]
        X_train = safe_numeric_frame(X_train)
        X_test = safe_numeric_frame(X_test)
        if "token_count" in X_train.columns:
            raise AssertionError(f"token_count leaked into model input for {spec.model_spec_id}")
        final_est = None
        if cv_mode == "fast":
            train_m, ext = _fast_proxy_metrics(X_train, y_train, X_test, y_test)
            best = {
                "params": {"fast_proxy_estimator": "closed_form_stage_score"},
                "cv_accuracy": train_m["accuracy"],
                "cv_precision": train_m["precision"],
                "cv_recall": train_m["recall"],
                "cv_f1": train_m["f1"],
                "cv_auc": train_m["auc"],
            }
        else:
            candidates = []
            cv = _cv_iterator(cv_mode, y_train, seed)
            for params in param_grid(spec.model_variant, cv_mode):
                fold_metrics = []
                for tr_idx, va_idx in cv.split(X_train, y_train):
                    est = make_estimator(spec.model_variant, params, seed, stage_dims=stage_dims)
                    est.fit(X_train.iloc[tr_idx], y_train[tr_idx])
                    fold_metrics.append(score_model(est, X_train.iloc[va_idx], y_train[va_idx]))
                mean = _mean_metrics(fold_metrics)
                candidates.append({
                    "params": params,
                    "cv_accuracy": mean["accuracy"],
                    "cv_precision": mean["precision"],
                    "cv_recall": mean["recall"],
                    "cv_f1": mean["f1"],
                    "cv_auc": mean["auc"],
                })
            best = _select_best(candidates)
            final_est = make_estimator(spec.model_variant, best["params"], seed, stage_dims=stage_dims)
            final_est.fit(X_train, y_train)
            ext = score_model(final_est, X_test, y_test)
            _append_external_predictions(
                prediction_sink, final_est, X_test, y_test, external_meta,
                seed, cv_mode, early_variant, spec,
            )
        rows.append({
            "seed": int(seed),
            "cv_mode": cv_mode,
            "early_variant": early_variant,
            "model_spec_id": spec.model_spec_id,
            "feature_block": spec.feature_block,
            "model_name": spec.model_name,
            "model_variant": spec.model_variant,
            "is_special_model": bool(spec.is_special),
            "best_params": json.dumps(best["params"], ensure_ascii=False, sort_keys=True),
            "cv_accuracy": best["cv_accuracy"],
            "cv_precision": best["cv_precision"],
            "cv_recall": best["cv_recall"],
            "cv_f1": best["cv_f1"],
            "cv_auc": best["cv_auc"],
            "external_accuracy": ext["accuracy"],
            "external_precision": ext["precision"],
            "external_recall": ext["recall"],
            "external_f1": ext["f1"],
            "external_auc": ext["auc"],
            "external_correct_n": ext["correct_n"],
            "external_total_n": ext["total_n"],
            "external_tn": ext["tn"],
            "external_fp": ext["fp"],
            "external_fn": ext["fn"],
            "external_tp": ext["tp"],
            "n_features": int(X_train.shape[1]),
        })
        if progress_callback:
            progress_callback({"event": "completed"})
    return pd.DataFrame(rows)
