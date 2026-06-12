# stagev2 Summary

## 1. Project overview

`stagev2` is a clean Python project for evaluating early, middle, and late stage transcript features for AD-vs-control classification. It implements both `earlyv0` and `earlyv1` in the same run and uses a fixed protocol: 10-fold internal cross-validation plus held-out external test evaluation.

## 2. Data extraction from original zip

Raw CSV files are stored in `input/raw/`. The project code can also search a source zip for the three required files and copy them into this folder without modifying the original zip.

## 3. Data setting

- Train set: `ad_s2t_wav2vec.csv` + `control_s2t_wav2vec.csv`
- External test set: `test_s2t_wav2vec.csv`
- Train samples: 166
- External test samples: 71
- Positive label: AD = 1; Control = 0

## 4. Feature setting

Feature blocks retained in this version:

`early_only`, `middle_only`, `late_only`, `early_middle`, `early_middle_scale`, `middle_late`, `middle_late_scale`, `all`, `all_plus_interactions`.

API/cache source status:

- middle: cache
- late: cache

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

- early_variant: `earlyv0`
- model_name: `Logistic Regression`
- feature_block: `early_only`
- external_accuracy: 0.718310
- external_f1: 0.722222
- external_auc: 0.735714

Top seed=2026 rows:

| early_variant   | model_name          | feature_block   |   cv_accuracy |   external_accuracy |   external_f1 |   external_auc |
|:----------------|:--------------------|:----------------|--------------:|--------------------:|--------------:|---------------:|
| earlyv0         | Logistic Regression | early_only      |      0.698795 |            0.71831  |      0.722222 |       0.735714 |
| earlyv1         | RBF SVM             | all             |      0.76506  |            0.704225 |      0.695652 |       0.760317 |
| earlyv0         | RBF SVM             | all             |      0.76506  |            0.704225 |      0.695652 |       0.759524 |
| earlyv1         | Logistic Regression | early_only      |      0.704819 |            0.704225 |      0.712329 |       0.735714 |
| earlyv0         | Linear SVM          | middle_late     |      0.662651 |            0.704225 |      0.644068 |       0.719841 |
| earlyv1         | Linear SVM          | middle_late     |      0.662651 |            0.704225 |      0.644068 |       0.719841 |
| earlyv0         | Logistic Regression | middle_late     |      0.674699 |            0.704225 |      0.695652 |       0.715873 |
| earlyv1         | Logistic Regression | middle_late     |      0.674699 |            0.704225 |      0.695652 |       0.715873 |
| earlyv0         | RBF SVM             | early_only      |      0.76506  |            0.690141 |      0.717949 |       0.761905 |
| earlyv1         | RBF SVM             | early_only      |      0.76506  |            0.690141 |      0.717949 |       0.760317 |

## 9. Stability results under seeds 0–29

Top stability rows:

| early_variant   | model_name          | feature_block   |   n_seeds |   external_accuracy_mean |   external_accuracy_std |   external_f1_mean |   external_auc_mean |
|:----------------|:--------------------|:----------------|----------:|-------------------------:|------------------------:|-------------------:|--------------------:|
| earlyv0         | Logistic Regression | early_only      |        30 |                 0.71831  |             2.25841e-16 |           0.722222 |            0.735714 |
| earlyv1         | RBF SVM             | all             |        30 |                 0.704225 |             0           |           0.695652 |            0.760317 |
| earlyv0         | RBF SVM             | all             |        30 |                 0.704225 |             0           |           0.695652 |            0.759524 |
| earlyv1         | Logistic Regression | early_only      |        30 |                 0.704225 |             0           |           0.712329 |            0.735714 |
| earlyv0         | Linear SVM          | middle_late     |        30 |                 0.704225 |             0           |           0.644068 |            0.719841 |
| earlyv1         | Linear SVM          | middle_late     |        30 |                 0.704225 |             0           |           0.644068 |            0.719841 |
| earlyv0         | Logistic Regression | middle_late     |        30 |                 0.704225 |             0           |           0.695652 |            0.715873 |
| earlyv1         | Logistic Regression | middle_late     |        30 |                 0.704225 |             0           |           0.695652 |            0.715873 |
| earlyv0         | RBF SVM             | early_only      |        30 |                 0.690141 |             2.25841e-16 |           0.717949 |            0.761905 |
| earlyv1         | RBF SVM             | early_only      |        30 |                 0.690141 |             2.25841e-16 |           0.717949 |            0.760317 |

## 10. earlyv1 vs earlyv0 external accuracy gain

`earlyv1_has_gain` is defined as seed=2026 `earlyv1_external_accuracy > earlyv0_external_accuracy` for the same model family and feature block.

Rows with gain: 3 / 27.

## 11. Scale gain based on external accuracy

Scale gain is defined only as external accuracy improvement over the matching raw block:

```python
scale_gain_delta = scale_external_accuracy - raw_external_accuracy
scale_has_gain = scale_gain_delta > 0
```

Seed=2026 scale-gain rows with gain: 5 / 18.

Stability scale gain uses mean external accuracy over seeds 0–29 and additionally reports `scale_gain_rate`.

## 12. Final recommendation

Use the stability summary rather than a single external-test score as the primary decision aid. Prefer a feature block only when it has competitive seed=2026 accuracy and stable external accuracy mean across seeds 0–29. Treat scale features as useful only if they show positive external-accuracy gain under the defined raw-vs-scale comparison.
