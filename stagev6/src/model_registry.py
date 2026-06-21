from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from . import stagev6_config as cfg

@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    group: str
    feature_block: str
    estimator: Any
    param_grid: dict


def _pipe(clf):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])


def _base_specs(role: str, block: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    specs.append(ModelSpec(f"{role}__{block}__lr__l2", role, "lr", block,
        _pipe(LogisticRegression(penalty="l2", solver="liblinear", max_iter=5000, random_state=cfg.RANDOM_STATE)),
        {"clf__C": cfg.LR_C_GRID, "clf__class_weight": cfg.CLASS_WEIGHTS}))
    specs.append(ModelSpec(f"{role}__{block}__lr__l1", role, "lr", block,
        _pipe(LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=cfg.RANDOM_STATE)),
        {"clf__C": cfg.LR_C_GRID, "clf__class_weight": cfg.CLASS_WEIGHTS}))
    specs.append(ModelSpec(f"{role}__{block}__lr__elasticnet", role, "lr", block,
        _pipe(LogisticRegression(penalty="elasticnet", solver="saga", max_iter=8000, random_state=cfg.RANDOM_STATE)),
        {"clf__C": cfg.LR_C_GRID, "clf__l1_ratio": [0.25, 0.5, 0.75], "clf__class_weight": cfg.CLASS_WEIGHTS}))
    for kernel, suffix, degree, extra in [
        ("linear", "linear", None, {}),
        ("poly", "poly2", 2, {"clf__gamma": ["scale"], "clf__coef0": cfg.POLY_COEF0}),
        ("poly", "poly3", 3, {"clf__gamma": ["scale"], "clf__coef0": cfg.POLY_COEF0}),
        ("rbf", "rbf", None, {"clf__gamma": ["scale", "auto"]}),
        ("sigmoid", "sigmoid", None, {"clf__gamma": ["scale"], "clf__coef0": cfg.POLY_COEF0}),
    ]:
        kwargs = {"kernel": kernel, "probability": True, "random_state": cfg.RANDOM_STATE}
        if degree is not None:
            kwargs["degree"] = degree
        grid = {"clf__C": cfg.SVC_C_GRID, "clf__class_weight": cfg.CLASS_WEIGHTS, **extra}
        specs.append(ModelSpec(f"{role}__{block}__svc__{suffix}", role, "svc", block, _pipe(SVC(**kwargs)), grid))
    return specs


def build_gate_specs() -> list[ModelSpec]:
    # Full stagev5 LR/SVC classifier family on each scientifically valid late gate feature block.
    specs = _base_specs("gate", "late") + _base_specs("gate", "middle_late")
    return specs


def build_branch_specs() -> list[ModelSpec]:
    # Full stagev5 LR/SVC family plus its small MLP anchor on the fixed E+M non-late branch.
    specs = _base_specs("branch", "early_middle")
    specs.append(ModelSpec(
        "branch__early_middle__mlp__small", "branch", "mlp", "early_middle",
        _pipe(MLPClassifier(max_iter=1500, early_stopping=True, random_state=cfg.RANDOM_STATE)),
        {"clf__hidden_layer_sizes": cfg.MLP_HIDDEN, "clf__alpha": cfg.MLP_ALPHA, "clf__learning_rate_init": [0.001]},
    ))
    return specs
