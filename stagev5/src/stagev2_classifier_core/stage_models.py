"""Custom stage-score calibrated classifiers."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

import config


def _clip(p, eps: float = 1e-5):
    return np.clip(np.asarray(p, dtype=float), eps, 1 - eps)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _make_lr(C=1.0, class_weight=None, random_state=2026):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", max_iter=5000, random_state=random_state)),
    ])


def _make_mlp(random_state=2026):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.001, learning_rate_init=0.001, max_iter=1500, early_stopping=True, random_state=random_state)),
    ])


def _make_svc(random_state=2026):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="poly", degree=2, C=1.0, gamma="scale", coef0=1.0, probability=True, random_state=random_state)),
    ])


class StageScoreCalibratedLR(BaseEstimator, ClassifierMixin):
    """OOF stage-score LR using [s_E, s_M, s_L] interactions.

    mode:
    - early_middle: [s_E, s_M, s_E*s_M]
    - middle_late: [s_M, s_L, s_M*s_L]
    - early_late: [s_E, s_L, s_E*s_L]
    - stage_score: [s_E, s_M, s_L, s_E*s_M, s_M*s_L, s_E*s_M*s_L]
    """

    def __init__(self, n_early: int, n_middle: int, n_late: int, mode: str = "stage_score", base_C: float = 1.0, meta_C: float = 1.0, class_weight=None, inner_splits: int = 5, random_state: int = 2026):
        self.n_early = n_early
        self.n_middle = n_middle
        self.n_late = n_late
        self.mode = mode
        self.base_C = base_C
        self.meta_C = meta_C
        self.class_weight = class_weight
        self.inner_splits = inner_splits
        self.random_state = random_state

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        e0 = 0
        e1 = self.n_early
        m1 = e1 + self.n_middle
        l1 = m1 + self.n_late
        return X[:, e0:e1], X[:, e1:m1], X[:, m1:l1]

    def _build_meta(self, s_e, s_m, s_l):
        s_e, s_m, s_l = _clip(s_e), _clip(s_m), _clip(s_l)
        if self.mode == "early_middle":
            return np.column_stack([s_e, s_m, s_e * s_m])
        if self.mode == "middle_late":
            return np.column_stack([s_m, s_l, s_m * s_l])
        if self.mode == "early_late":
            return np.column_stack([s_e, s_l, s_e * s_l])
        if self.mode == "stage_score":
            return np.column_stack([s_e, s_m, s_l, s_e * s_m, s_m * s_l, s_e * s_m * s_l])
        raise ValueError(f"Unknown mode: {self.mode}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        X_e, X_m, X_l = self._split(X)
        min_class = np.bincount(y).min()
        n_splits = int(min(self.inner_splits, min_class))
        if n_splits < 2:
            raise ValueError("Not enough samples per class for inner OOF scoring.")
        inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        e_model = _make_lr(self.base_C, self.class_weight, self.random_state)
        m_model = _make_lr(self.base_C, self.class_weight, self.random_state)
        l_model = _make_lr(self.base_C, self.class_weight, self.random_state)
        s_e = cross_val_predict(e_model, X_e, y, cv=inner, method="predict_proba")[:, 1]
        s_m = cross_val_predict(m_model, X_m, y, cv=inner, method="predict_proba")[:, 1]
        s_l = cross_val_predict(l_model, X_l, y, cv=inner, method="predict_proba")[:, 1]
        Z = self._build_meta(s_e, s_m, s_l)
        self.meta_model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self.meta_C, class_weight=self.class_weight, solver="liblinear", max_iter=5000, random_state=self.random_state)),
        ])
        self.meta_model_.fit(Z, y)

        self.early_model_ = _make_lr(self.base_C, self.class_weight, self.random_state).fit(X_e, y)
        self.middle_model_ = _make_lr(self.base_C, self.class_weight, self.random_state).fit(X_m, y)
        self.late_model_ = _make_lr(self.base_C, self.class_weight, self.random_state).fit(X_l, y)
        return self

    def stage_scores(self, X):
        check_is_fitted(self, ["early_model_", "middle_model_", "late_model_", "meta_model_"])
        X_e, X_m, X_l = self._split(X)
        s_e = self.early_model_.predict_proba(X_e)[:, 1]
        s_m = self.middle_model_.predict_proba(X_m)[:, 1]
        s_l = self.late_model_.predict_proba(X_l)[:, 1]
        return s_e, s_m, s_l

    def predict_proba(self, X):
        s_e, s_m, s_l = self.stage_scores(X)
        Z = self._build_meta(s_e, s_m, s_l)
        return self.meta_model_.predict_proba(Z)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= config.DECISION_THRESHOLD).astype(int)


class MLPSVCLateCalibratedLR(BaseEstimator, ClassifierMixin):
    """Meta LR over [p_MLP(early+middle), p_SVC(early+middle), s_L, interactions]."""

    def __init__(self, n_early: int, n_middle: int, n_late: int, meta_C: float = 1.0, late_C: float = 1.0, class_weight=None, inner_splits: int = 5, random_state: int = 2026):
        self.n_early = n_early
        self.n_middle = n_middle
        self.n_late = n_late
        self.meta_C = meta_C
        self.late_C = late_C
        self.class_weight = class_weight
        self.inner_splits = inner_splits
        self.random_state = random_state

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        e1 = self.n_early
        m1 = e1 + self.n_middle
        return X[:, :m1], X[:, m1:m1 + self.n_late]

    def _build_meta(self, p_mlp, p_svc, s_l):
        p_mlp, p_svc, s_l = _clip(p_mlp), _clip(p_svc), _clip(s_l)
        return np.column_stack([p_mlp, p_svc, s_l, p_mlp * s_l, p_svc * s_l])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        X_em, X_l = self._split(X)
        min_class = np.bincount(y).min()
        n_splits = int(min(self.inner_splits, min_class))
        if n_splits < 2:
            raise ValueError("Not enough samples per class for inner OOF scoring.")
        inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        mlp = _make_mlp(self.random_state)
        svc = _make_svc(self.random_state)
        late = _make_lr(self.late_C, self.class_weight, self.random_state)
        p_mlp = cross_val_predict(mlp, X_em, y, cv=inner, method="predict_proba")[:, 1]
        p_svc = cross_val_predict(svc, X_em, y, cv=inner, method="predict_proba")[:, 1]
        s_l = cross_val_predict(late, X_l, y, cv=inner, method="predict_proba")[:, 1]
        Z = self._build_meta(p_mlp, p_svc, s_l)
        self.meta_model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self.meta_C, class_weight=self.class_weight, solver="liblinear", max_iter=5000, random_state=self.random_state)),
        ]).fit(Z, y)

        self.mlp_model_ = _make_mlp(self.random_state).fit(X_em, y)
        self.svc_model_ = _make_svc(self.random_state).fit(X_em, y)
        self.late_model_ = _make_lr(self.late_C, self.class_weight, self.random_state).fit(X_l, y)
        return self

    def stage_scores(self, X):
        check_is_fitted(self, ["mlp_model_", "svc_model_", "late_model_", "meta_model_"])
        X_em, X_l = self._split(X)
        p_mlp = self.mlp_model_.predict_proba(X_em)[:, 1]
        p_svc = self.svc_model_.predict_proba(X_em)[:, 1]
        s_l = self.late_model_.predict_proba(X_l)[:, 1]
        return p_mlp, p_svc, s_l

    def predict_proba(self, X):
        p_mlp, p_svc, s_l = self.stage_scores(X)
        Z = self._build_meta(p_mlp, p_svc, s_l)
        return self.meta_model_.predict_proba(Z)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= config.DECISION_THRESHOLD).astype(int)
