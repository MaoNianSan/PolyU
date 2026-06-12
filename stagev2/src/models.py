from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def make_estimator(model_name: str, params: dict, seed: int) -> Pipeline:
    if model_name == "Logistic Regression":
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight"),
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )
    elif model_name == "Linear SVM":
        clf = SVC(
            kernel="linear",
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight"),
            probability=False,
            random_state=seed,
        )
    elif model_name == "RBF SVM":
        clf = SVC(
            kernel="rbf",
            C=float(params.get("C", 1.0)),
            gamma=params.get("gamma", "scale"),
            class_weight=params.get("class_weight"),
            probability=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def model_grid() -> dict[str, list[dict]]:
    return config.MODEL_PARAM_GRID
