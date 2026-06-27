"""Fixed Stagev8.5 model families for two ordinal MMSE thresholds."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg


@dataclass(frozen=True)
class SeverityHeadSpec:
    key: str
    head: str
    feature_block: str
    family: str
    estimator: Any
    grid: dict[str, list[Any]]
    target_definition: str
    rationale: str


def pipeline(clf: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def severity_head_specs() -> dict[str, SeverityHeadSpec]:
    # Families are fixed before external evaluation. Grid search tunes only each
    # fixed family's regularisation parameters using internal training CV.
    return {
        "T20": SeverityHeadSpec(
            key="T20",
            head="T20",
            feature_block="EM",
            family="LR-ElasticNet",
            estimator=pipeline(LogisticRegression(
                solver="saga", penalty="elasticnet", max_iter=5000,
                tol=1e-3, random_state=cfg.RANDOM_STATE,
            )),
            grid={
                "clf__C": [0.1, 1.0],
                "clf__l1_ratio": [0.1, 0.5],
                "clf__class_weight": ["balanced"],
            },
            target_definition=f"MMSE <= {cfg.MMSE_INTERMEDIATE_MAX} (1) vs MMSE >= {cfg.MMSE_HIGH_MIN} (0), true AD training samples only",
            rationale="Pre-specified E+M elastic-net transition model for the high-to-intermediate/low MMSE boundary.",
        ),
        "T14": SeverityHeadSpec(
            key="T14",
            head="T14",
            feature_block="L",
            family="LR-L2",
            estimator=pipeline(LogisticRegression(
                solver="liblinear", penalty="l2", max_iter=10000,
                random_state=cfg.RANDOM_STATE,
            )),
            grid={
                "clf__C": [0.1, 1.0],
                "clf__class_weight": ["balanced"],
            },
            target_definition=f"MMSE <= {cfg.MMSE_LOW_MAX} (1) vs MMSE {cfg.MMSE_INTERMEDIATE_MIN}-{cfg.MMSE_INTERMEDIATE_MAX} (0), conditional on MMSE <= {cfg.MMSE_INTERMEDIATE_MAX} true AD training samples",
            rationale="Fixed raw-F8-only L2 logistic regression for the intermediate-to-low MMSE transition; no new late features or fusion models are introduced.",
        ),
    }
