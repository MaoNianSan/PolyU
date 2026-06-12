from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def early_distribution_summary(early_features: dict[str, dict[str, pd.DataFrame]], output_path: Path) -> pd.DataFrame:
    rows = []
    requested = ["early_integrity_score", "early_omission_risk_score", "early_content_efficiency"]
    for variant, parts in early_features.items():
        for split, df in parts.items():
            for label in sorted(df["label"].dropna().unique()):
                sub = df[df["label"] == label]
                for feat in requested:
                    if feat not in sub.columns:
                        continue
                    x = pd.to_numeric(sub[feat], errors="coerce").dropna()
                    if len(x) == 0:
                        continue
                    rows.append({
                        "early_variant": variant,
                        "split": split,
                        "label": int(label),
                        "feature": feat,
                        "mean": float(x.mean()),
                        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                        "median": float(x.median()),
                        "q25": float(x.quantile(0.25)),
                        "q75": float(x.quantile(0.75)),
                        "min": float(x.min()),
                        "max": float(x.max()),
                    })
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def earlyv1_gain_over_earlyv0(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in sorted(main["model_name"].unique()):
        for feature_block in sorted(main["feature_block"].unique()):
            sub = main[(main["model_name"] == model_name) & (main["feature_block"] == feature_block)]
            v0 = sub[sub["early_variant"] == "earlyv0"]
            v1 = sub[sub["early_variant"] == "earlyv1"]
            if len(v0) and len(v1):
                a = float(v0.iloc[0]["external_accuracy"])
                b = float(v1.iloc[0]["external_accuracy"])
                rows.append({
                    "model_name": model_name,
                    "feature_block": feature_block,
                    "earlyv0_external_accuracy": a,
                    "earlyv1_external_accuracy": b,
                    "external_accuracy_delta": b - a,
                    "earlyv1_has_gain": bool(b > a),
                })
    return pd.DataFrame(rows)


def write_summary_md(
    path: Path,
    preprocess_summary: dict,
    feature_summary: dict,
    main: pd.DataFrame,
    stability_summary: pd.DataFrame,
    early_gain: pd.DataFrame,
    scale_seed: pd.DataFrame,
    scale_stability: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top_main = main.sort_values(["external_accuracy", "external_auc", "external_f1"], ascending=False).head(10)
    top_stab = stability_summary.sort_values(["external_accuracy_mean", "external_auc_mean", "external_f1_mean"], ascending=False).head(10)
    best = top_main.iloc[0]
    api_note = []
    for name, summary in feature_summary.items():
        if isinstance(summary, dict) and "source" in summary:
            api_note.append(f"- {name}: {summary.get('source')}")
    md = f"""# stagev2 Summary

## 1. Project overview

`stagev2` is a clean Python project for evaluating early, middle, and late stage transcript features for AD-vs-control classification. It implements both `earlyv0` and `earlyv1` in the same run and uses a fixed protocol: 10-fold internal cross-validation plus held-out external test evaluation.

## 2. Data extraction from original zip

Raw CSV files are stored in `input/raw/`. The project code can also search a source zip for the three required files and copy them into this folder without modifying the original zip.

## 3. Data setting

- Train set: `ad_s2t_wav2vec.csv` + `control_s2t_wav2vec.csv`
- External test set: `test_s2t_wav2vec.csv`
- Train samples: {preprocess_summary.get('train_n')}
- External test samples: {preprocess_summary.get('external_test_n')}
- Positive label: AD = 1; Control = 0

## 4. Feature setting

Feature blocks retained in this version:

`early_only`, `middle_only`, `late_only`, `early_middle`, `early_middle_scale`, `middle_late`, `middle_late_scale`, `all`, `all_plus_interactions`.

API/cache source status:

{chr(10).join(api_note) if api_note else '- not recorded'}

## 5. earlyv0 and earlyv1 definition

- `earlyv0`: original BM25-based early feature set.
- `earlyv1`: `earlyv0 + early_content_efficiency`.

The added feature is:

```python
token_count = max(len(str(text).split()), 1)
early_content_efficiency = early_integrity_score / np.log1p(token_count)
```

`token_count` is retained as an audit column but excluded from model input.

## 6. Classifier setting

Only three model families are used:

- Logistic Regression
- Linear SVM
- RBF SVM

No Random Forest, XGBoost, MLP, bootstrap CI, or extra diagnostic metrics are included.

## 7. 10-fold CV + external test protocol

For each feature block, early variant, model family, and seed:

1. Run stratified 10-fold CV on the train set.
2. Select hyperparameters by `cv_accuracy`.
3. Break ties by `cv_auc`, then `cv_f1`.
4. Refit the selected model on the full train set.
5. Evaluate once on the held-out external test set.

External test results are not used for model selection.

## 8. Main results under seed=2026

Best seed=2026 row:

- early_variant: `{best['early_variant']}`
- model_name: `{best['model_name']}`
- feature_block: `{best['feature_block']}`
- external_accuracy: {best['external_accuracy']:.6f}
- external_f1: {best['external_f1']:.6f}
- external_auc: {best['external_auc']:.6f}

Top seed=2026 rows:

{top_main[['early_variant','model_name','feature_block','cv_accuracy','external_accuracy','external_f1','external_auc']].to_markdown(index=False)}

## 9. Stability results under seeds 0–29

Top stability rows:

{top_stab[['early_variant','model_name','feature_block','n_seeds','external_accuracy_mean','external_accuracy_std','external_f1_mean','external_auc_mean']].to_markdown(index=False)}

## 10. earlyv1 vs earlyv0 external accuracy gain

`earlyv1_has_gain` is defined as seed=2026 `earlyv1_external_accuracy > earlyv0_external_accuracy` for the same model family and feature block.

Rows with gain: {int(early_gain['earlyv1_has_gain'].sum()) if len(early_gain) else 0} / {len(early_gain)}.

## 11. Scale gain based on external accuracy

Scale gain is defined only as external accuracy improvement over the matching raw block:

```python
scale_gain_delta = scale_external_accuracy - raw_external_accuracy
scale_has_gain = scale_gain_delta > 0
```

Seed=2026 scale-gain rows with gain: {int(scale_seed['scale_has_gain'].sum()) if len(scale_seed) else 0} / {len(scale_seed)}.

Stability scale gain uses mean external accuracy over seeds 0–29 and additionally reports `scale_gain_rate`.

## 12. Final recommendation

Use the stability summary rather than a single external-test score as the primary decision aid. Prefer a feature block only when it has competitive seed=2026 accuracy and stable external accuracy mean across seeds 0–29. Treat scale features as useful only if they show positive external-accuracy gain under the defined raw-vs-scale comparison.
"""
    path.write_text(md, encoding="utf-8")
