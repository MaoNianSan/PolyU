from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as cfg


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    role: str
    feature_block: str
    classifier: str
    estimator: Any
    param_grid: dict
    scoring: str


def _pipe(clf: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def _lr_l2() -> Pipeline:
    return _pipe(LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=5000, random_state=cfg.RANDOM_STATE,
    ))


def _svc_poly3() -> Pipeline:
    return _pipe(SVC(
        kernel="poly", degree=3, gamma="scale", probability=True, random_state=cfg.RANDOM_STATE,
    ))


def gate_specs() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            component_id="g1_l_lr_l2", role="late_gate", feature_block="late", classifier="lr_l2",
            estimator=_lr_l2(),
            param_grid={"clf__C": [0.03, 0.1, 1.0], "clf__class_weight": [None, "balanced"]},
            scoring="balanced_accuracy",
        ),
        ComponentSpec(
            component_id="g2_middle_late_lr_l2", role="late_gate", feature_block="middle_late", classifier="lr_l2",
            estimator=_lr_l2(),
            param_grid={"clf__C": [0.03, 0.1, 1.0], "clf__class_weight": [None, "balanced"]},
            scoring="balanced_accuracy",
        ),
        ComponentSpec(
            component_id="g3_middle_late_svc_poly3", role="late_gate", feature_block="middle_late", classifier="svc_poly3",
            estimator=_svc_poly3(),
            param_grid={
                "clf__C": [0.1, 1.0, 3.0], "clf__gamma": ["scale"],
                "clf__coef0": [0.0, 1.0], "clf__class_weight": [None, "balanced"],
            },
            scoring="balanced_accuracy",
        ),
    ]


def branch_specs() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            component_id="b1_early_middle_svc_poly3", role="nonlate_branch", feature_block="early_middle", classifier="svc_poly3",
            estimator=_svc_poly3(),
            param_grid={
                "clf__C": [0.1, 1.0, 3.0], "clf__gamma": ["scale"],
                "clf__coef0": [0.0, 1.0], "clf__class_weight": [None, "balanced"],
            },
            scoring="accuracy",
        ),
        ComponentSpec(
            component_id="b2_early_middle_lr_l2", role="nonlate_branch", feature_block="early_middle", classifier="lr_l2",
            estimator=_lr_l2(),
            param_grid={"clf__C": [0.03, 0.1, 1.0], "clf__class_weight": [None, "balanced"]},
            scoring="accuracy",
        ),
    ]


def cascade_name(gate_id: str, branch_id: str) -> str:
    return f"cascade__{gate_id}__to__{branch_id}"
