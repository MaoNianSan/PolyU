from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_merge import build_feature_blocks, merge_stage_features

OUT = ROOT / "output" / "final_report"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(ROOT / "output" / "preprocess" / "train_preprocessed.csv")
test = pd.read_csv(ROOT / "output" / "preprocess" / "external_test_preprocessed.csv")
mid_tr = pd.read_csv(ROOT / "output" / "features" / "middle" / "train_middle_features.csv")
mid_te = pd.read_csv(ROOT / "output" / "features" / "middle" / "external_middle_features.csv")
late_tr = pd.read_csv(ROOT / "output" / "features" / "late" / "train_late_features.csv")
late_te = pd.read_csv(ROOT / "output" / "features" / "late" / "external_late_features.csv")

y_train = train["label"].astype(int).to_numpy()
y_test = test["label"].astype(int).to_numpy()
rows = []

for early_variant in ["earlyv0", "earlyv1"]:
    e_tr = pd.read_csv(ROOT / "output" / "features" / "early" / early_variant / "train_early_features.csv")
    e_te = pd.read_csv(ROOT / "output" / "features" / "early" / early_variant / "external_early_features.csv")
    merged_train = merge_stage_features(train, e_tr, mid_tr, late_tr)
    merged_test = merge_stage_features(test, e_te, mid_te, late_te)
    blocks = build_feature_blocks(merged_train, merged_test)
    X_train, X_test = blocks["early_middle"]
    est = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="poly", degree=3, C=1.0, gamma="scale", coef0=1.0, probability=True, random_state=2026)),
    ])
    est.fit(X_train, y_train)
    prob = est.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({
        "early_variant": early_variant,
        "anchor_model": "early_middle__svc__poly3",
        "fixed_params": "C=1.0,gamma=scale,coef0=1.0,probability=True,threshold=0.5",
        "n_features": int(X_train.shape[1]),
        "external_accuracy": float(accuracy_score(y_test, pred)),
        "external_sensitivity": float(recall_score(y_test, pred, zero_division=0)),
        "external_auc": float(roc_auc_score(y_test, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / "v2_compat_anchor_check.csv", index=False)
md = [
    "# v2-compatible anchor check",
    "",
    "This smoke check verifies the main repair target without running the full 204-model grid.",
    "",
    result.to_markdown(index=False),
    "",
    "Target anchor: v2 high-performance `early_middle__svc__poly3` with full BGE-M3 mean embedding, `coef0=1.0`, `probability=True`, and probability threshold 0.5.",
]
(OUT / "v2_compat_anchor_check.md").write_text("\n".join(md), encoding="utf-8")
print(result.to_string(index=False))
