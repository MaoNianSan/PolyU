from __future__ import annotations

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .stage_models import MLPSVCLateCalibratedLR, StageScoreCalibratedLR


@dataclass(frozen=True)
class ModelSpec:
    model_spec_id: str
    feature_block: str
    model_name: str
    model_variant: str
    is_special: bool = False


def model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for block in config.FEATURE_BLOCKS:
        for variant in config.CLASSIFIER_VARIANTS:
            specs.append(ModelSpec(f"{block}__{variant}", block, variant, variant, False))
    # v2-compatible special models: they operate on raw all-stage feature order
    # [early, middle, late], not on v3 activation summary placeholders.
    specs.extend([
        ModelSpec("stage_score_early_middle__lr__l2", "all", "stage_score_early_middle__lr__l2", "stage_score_early_middle__lr__l2", True),
        ModelSpec("stage_score_middle_late__lr__l2", "all", "stage_score_middle_late__lr__l2", "stage_score_middle_late__lr__l2", True),
        ModelSpec("stage_score_early_late__lr__l2", "all", "stage_score_early_late__lr__l2", "stage_score_early_late__lr__l2", True),
        ModelSpec("stage_score_three_stage__lr__l2", "all", "stage_score_three_stage__lr__l2", "stage_score_three_stage__lr__l2", True),
        ModelSpec("early_middle__mlp__small", "early_middle", "early_middle__mlp__small", "mlp__small", True),
        ModelSpec("mlp_svc_late_calibrated__lr__l2", "all", "mlp_svc_late_calibrated__lr__l2", "mlp_svc_late_calibrated__lr__l2", True),
    ])
    return specs


def param_grid(model_variant: str, cv_mode: str) -> list[dict]:
    if cv_mode == "fast":
        return [{"fast_proxy_estimator": "closed_form_stage_score"}]
    if model_variant == "lr__l2":
        return [{"C": C, "penalty": "l2", "solver": "liblinear", "class_weight": cw} for C in config.LR_C_GRID for cw in config.CLASS_WEIGHTS]
    if model_variant == "lr__l1":
        return [{"C": C, "penalty": "l1", "solver": "liblinear", "class_weight": cw} for C in config.LR_C_GRID for cw in config.CLASS_WEIGHTS]
    if model_variant == "lr__elasticnet":
        return [
            {"C": C, "penalty": "elasticnet", "solver": "saga", "l1_ratio": l1, "class_weight": cw}
            for C in config.LR_C_GRID for l1 in [0.25, 0.5, 0.75] for cw in config.CLASS_WEIGHTS
        ]
    if model_variant == "svc__linear":
        return [{"C": C, "kernel": "linear", "class_weight": cw} for C in config.SVC_C_GRID for cw in config.CLASS_WEIGHTS]
    if model_variant == "svc__poly2":
        return [
            {"C": C, "kernel": "poly", "degree": 2, "gamma": "scale", "coef0": coef0, "class_weight": cw}
            for C in config.SVC_C_GRID for coef0 in config.POLY_COEF0 for cw in config.CLASS_WEIGHTS
        ]
    if model_variant == "svc__poly3":
        return [
            {"C": C, "kernel": "poly", "degree": 3, "gamma": "scale", "coef0": coef0, "class_weight": cw}
            for C in config.SVC_C_GRID for coef0 in config.POLY_COEF0 for cw in config.CLASS_WEIGHTS
        ]
    if model_variant == "svc__rbf":
        return [
            {"C": C, "kernel": "rbf", "gamma": gamma, "class_weight": cw}
            for C in config.SVC_C_GRID for gamma in ["scale", "auto"] for cw in config.CLASS_WEIGHTS
        ]
    if model_variant == "svc__sigmoid":
        return [
            {"C": C, "kernel": "sigmoid", "gamma": "scale", "coef0": coef0, "class_weight": cw}
            for C in config.SVC_C_GRID for coef0 in config.POLY_COEF0 for cw in config.CLASS_WEIGHTS
        ]
    if model_variant == "mlp__small":
        return [
            {"hidden_layer_sizes": h, "alpha": alpha, "learning_rate_init": 0.001}
            for h in config.MLP_HIDDEN for alpha in config.MLP_ALPHA
        ]
    if model_variant.startswith("stage_score_"):
        return [
            {"base_C": base_C, "meta_C": meta_C, "class_weight": cw}
            for base_C in [0.03, 0.1, 1.0] for meta_C in [0.1, 1.0] for cw in config.CLASS_WEIGHTS
        ]
    if model_variant == "mlp_svc_late_calibrated__lr__l2":
        return [
            {"meta_C": meta_C, "late_C": late_C, "class_weight": cw}
            for meta_C in [0.1, 1.0] for late_C in [0.03, 0.1, 1.0] for cw in config.CLASS_WEIGHTS
        ]
    raise ValueError(f"Unknown model_variant={model_variant}")


def _stage_dims_or_error(stage_dims: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if stage_dims is None:
        raise ValueError("stage_dims=(n_early,n_middle,n_late) is required for v2 special stage models")
    return tuple(int(x) for x in stage_dims)  # type: ignore[return-value]


def make_estimator(model_variant: str, params: dict, seed: int, stage_dims: tuple[int, int, int] | None = None):
    if model_variant.startswith("stage_score_"):
        n_early, n_middle, n_late = _stage_dims_or_error(stage_dims)
        mode_map = {
            "stage_score_early_middle__lr__l2": "early_middle",
            "stage_score_middle_late__lr__l2": "middle_late",
            "stage_score_early_late__lr__l2": "early_late",
            "stage_score_three_stage__lr__l2": "stage_score",
        }
        return StageScoreCalibratedLR(
            n_early=n_early,
            n_middle=n_middle,
            n_late=n_late,
            mode=mode_map[model_variant],
            base_C=float(params.get("base_C", 1.0)),
            meta_C=float(params.get("meta_C", 1.0)),
            class_weight=params.get("class_weight", None),
            inner_splits=config.INNER_N_SPLITS,
            random_state=seed,
        )
    if model_variant == "mlp_svc_late_calibrated__lr__l2":
        n_early, n_middle, n_late = _stage_dims_or_error(stage_dims)
        return MLPSVCLateCalibratedLR(
            n_early=n_early,
            n_middle=n_middle,
            n_late=n_late,
            meta_C=float(params.get("meta_C", 1.0)),
            late_C=float(params.get("late_C", 1.0)),
            class_weight=params.get("class_weight", None),
            inner_splits=config.INNER_N_SPLITS,
            random_state=seed,
        )
    if model_variant.startswith("lr__"):
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=params.get("penalty", "l2"),
            solver=params.get("solver", "liblinear"),
            l1_ratio=params.get("l1_ratio", None),
            class_weight=params.get("class_weight", None),
            max_iter=8000,
            random_state=seed,
        )
    elif model_variant.startswith("svc__"):
        clf = SVC(
            kernel=params.get("kernel", "rbf"),
            C=float(params.get("C", 1.0)),
            degree=int(params.get("degree", 3)),
            gamma=params.get("gamma", "scale"),
            coef0=float(params.get("coef0", 0.0)),
            class_weight=params.get("class_weight", None),
            probability=True,
            random_state=seed,
        )
    elif model_variant == "mlp__small":
        clf = MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (16,)),
            alpha=float(params.get("alpha", 0.001)),
            learning_rate_init=float(params.get("learning_rate_init", 0.001)),
            max_iter=1500,
            early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_variant={model_variant}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])


def make_fast_estimator(model_variant: str, seed: int) -> Pipeline:
    clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=1000, random_state=seed)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])
