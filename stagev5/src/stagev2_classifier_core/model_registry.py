"""Model registry for stagev2 full classifier + interaction experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from stage_models import MLPSVCLateCalibratedLR, StageScoreCalibratedLR


@dataclass
class ModelSpec:
    name: str
    group: str
    feature_block: str
    estimator: Any
    param_grid: dict
    mechanism_consistent: bool = False


def _pipe(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def _lr_l2():
    return _pipe(LogisticRegression(penalty="l2", solver="liblinear", max_iter=5000, random_state=config.RANDOM_STATE))


def _lr_l1():
    return _pipe(LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=config.RANDOM_STATE))


def _lr_elastic():
    return _pipe(LogisticRegression(penalty="elasticnet", solver="saga", max_iter=8000, random_state=config.RANDOM_STATE))


def _svc(kernel: str, degree: int | None = None):
    kwargs = {"kernel": kernel, "probability": True, "random_state": config.RANDOM_STATE}
    if degree is not None:
        kwargs["degree"] = degree
    return _pipe(SVC(**kwargs))


def _mlp():
    return _pipe(MLPClassifier(max_iter=1500, early_stopping=True, random_state=config.RANDOM_STATE))


def _add_classifier_family(specs: list[ModelSpec], block: str, group: str, include_heavy: bool = True) -> None:
    """Add LR and SVM families for a feature block."""
    specs.append(ModelSpec(
        name=f"{block}__lr__l2",
        group=group,
        feature_block=block,
        estimator=_lr_l2(),
        param_grid={"clf__C": config.LR_C_GRID, "clf__class_weight": config.CLASS_WEIGHTS},
    ))
    specs.append(ModelSpec(
        name=f"{block}__lr__l1",
        group=group,
        feature_block=block,
        estimator=_lr_l1(),
        param_grid={"clf__C": config.LR_C_GRID, "clf__class_weight": config.CLASS_WEIGHTS},
    ))
    specs.append(ModelSpec(
        name=f"{block}__lr__elasticnet",
        group=group,
        feature_block=block,
        estimator=_lr_elastic(),
        param_grid={"clf__C": config.LR_C_GRID, "clf__l1_ratio": [0.25, 0.5, 0.75], "clf__class_weight": config.CLASS_WEIGHTS},
    ))
    specs.append(ModelSpec(
        name=f"{block}__svc__linear",
        group=group,
        feature_block=block,
        estimator=_svc("linear"),
        param_grid={"clf__C": config.SVC_C_GRID, "clf__class_weight": config.CLASS_WEIGHTS},
    ))
    specs.append(ModelSpec(
        name=f"{block}__svc__poly2",
        group=group,
        feature_block=block,
        estimator=_svc("poly", degree=2),
        param_grid={"clf__C": config.SVC_C_GRID, "clf__gamma": ["scale"], "clf__coef0": config.POLY_COEF0, "clf__class_weight": config.CLASS_WEIGHTS},
    ))
    specs.append(ModelSpec(
        name=f"{block}__svc__poly3",
        group=group,
        feature_block=block,
        estimator=_svc("poly", degree=3),
        param_grid={"clf__C": config.SVC_C_GRID, "clf__gamma": ["scale"], "clf__coef0": config.POLY_COEF0, "clf__class_weight": config.CLASS_WEIGHTS},
    ))
    if include_heavy:
        specs.append(ModelSpec(
            name=f"{block}__svc__rbf",
            group=group,
            feature_block=block,
            estimator=_svc("rbf"),
            param_grid={"clf__C": config.SVC_C_GRID, "clf__gamma": ["scale", "auto"], "clf__class_weight": config.CLASS_WEIGHTS},
        ))
        specs.append(ModelSpec(
            name=f"{block}__svc__sigmoid",
            group=group,
            feature_block=block,
            estimator=_svc("sigmoid"),
            param_grid={"clf__C": config.SVC_C_GRID, "clf__gamma": ["scale"], "clf__coef0": config.POLY_COEF0, "clf__class_weight": config.CLASS_WEIGHTS},
        ))


def build_model_specs(n_early: int, n_middle: int, n_late: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    # A. Diagnostic single-stage and raw-fusion baselines.
    base_blocks = [
        ("early", "single_stage_early_mild"),
        ("middle", "single_stage_middle_moderate"),
        ("late", "single_stage_late_severe"),
        ("early_middle", "two_stage_raw"),
        ("middle_late", "two_stage_raw"),
        ("early_late", "two_stage_raw"),
        ("all", "three_stage_raw"),
    ]
    for block, group in base_blocks:
        _add_classifier_family(specs, block, group, include_heavy=True)

    # B. Sequential interaction blocks.
    # later-stage feature activation scales earlier-stage features.
    interaction_blocks = [
        ("stage_activation_summary", "interaction_summary_only"),
        ("early_middle_scale", "sequential_scale"),
        ("middle_late_scale", "sequential_scale"),
        ("sequential_interactions", "sequential_scale"),
        ("all_plus_interactions", "three_stage_full_interaction"),
    ]
    for block, group in interaction_blocks:
        _add_classifier_family(specs, block, group, include_heavy=True)

    # C. Mechanism-prior stage-score models using supervised OOF stage scores.
    # These directly test score-level interactions: sE*sM, sM*sL, sE*sM*sL.
    for name, mode in [
        ("stage_score_early_middle__lr__l2", "early_middle"),
        ("stage_score_middle_late__lr__l2", "middle_late"),
        ("stage_score_early_late__lr__l2", "early_late"),
        ("stage_score_three_stage__lr__l2", "stage_score"),
    ]:
        specs.append(ModelSpec(
            name=name,
            group="stage_score_interaction",
            feature_block="all",
            estimator=StageScoreCalibratedLR(n_early, n_middle, n_late, mode=mode, random_state=config.RANDOM_STATE, inner_splits=config.INNER_N_SPLITS),
            param_grid={
                "base_C": [0.03, 0.1, 1.0],
                "meta_C": [0.1, 1.0],
                "class_weight": config.CLASS_WEIGHTS,
            },
            mechanism_consistent=True,
        ))

    # D. Prior high-performance anchor with late correction.
    specs.append(ModelSpec(
        name="early_middle__mlp__small",
        group="performance_anchor",
        feature_block="early_middle",
        estimator=_mlp(),
        param_grid={
            "clf__hidden_layer_sizes": config.MLP_HIDDEN,
            "clf__alpha": config.MLP_ALPHA,
            "clf__learning_rate_init": [0.001],
        },
    ))
    specs.append(ModelSpec(
        name="mlp_svc_late_calibrated__lr__l2",
        group="performance_corrected",
        feature_block="all",
        estimator=MLPSVCLateCalibratedLR(n_early, n_middle, n_late, random_state=config.RANDOM_STATE, inner_splits=config.INNER_N_SPLITS),
        param_grid={
            "meta_C": [0.1, 1.0],
            "late_C": [0.03, 0.1, 1.0],
            "class_weight": config.CLASS_WEIGHTS,
        },
        mechanism_consistent=True,
    ))

    return specs
