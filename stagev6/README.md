# stagev6 — late-first cascade classifier

## Objective

Stagev6 is a conditional two-level AD classifier using **only the already-generated strict stagev5 feature files**:

1. **Late gate:** classify `late` versus `non-late`.
2. **Hard route:** samples predicted late are directly assigned AD.
3. **Non-late branch:** samples not predicted late receive an `AD versus control` decision.

The project is not a parallel early/middle/late/control multiclass model. Each component is binary.

## Fixed model panel

The final model panel contains the Cartesian product of three gates and two branches: **6 cascade models**.

| ID | Component | Feature block | Classifier | Grid score |
|---|---|---|---|---|
| G1 | late gate | L | L2 logistic regression | balanced accuracy |
| G2 | late gate | M+L | L2 logistic regression | balanced accuracy |
| G3 | late gate | M+L | polynomial SVC, degree 3 | balanced accuracy |
| B1 | non-late branch | E+M | polynomial SVC, degree 3 | accuracy |
| B2 | non-late branch | E+M | L2 logistic regression | accuracy |

The six specifications are complete cascades. Shared gate/branch components are fitted once and then recombined, preventing redundant training without altering any final prediction.

## Inherited feature policy

- **E:** stagev2 early BM25 features, 61 dimensions.
- **M:** stagev2 BGE-M3 embedding features, 1,024 dimensions after mean aggregation by `sample_id`.
- **L:** stagev4 unmasked P4/F8 features, 8 raw dimensions only.
- No API request, feature extraction, cache update, or MMSE input is performed in `train` mode.

The exact inherited source manifests are retained under `output/features/`.

## Training and evaluation

- Component hyperparameters: `GridSearchCV`.
- Gate selection scoring: balanced accuracy.
- Branch selection scoring: accuracy.
- Cascade OOF diagnostics: shared 10-fold stratified splits over `control`, `nonlate_AD`, and `late_AD`.
- External metrics and bootstrap confidence intervals: hard late-first routing; continuous probability metrics use `p_ad_mixture`.
- Final ranking: external accuracy, matching the stagev5 reporting convention.

The external set remains excluded from feature construction, imputation fitting, scaling fitting, and classifier fitting. It is not an unbiased final test once it is used for model ranking.

## Run on Windows PowerShell

```powershell
cd D:\research\H.L.Liang-Lab\Code\expore\stagev6
python .\run_stagev6.py --mode self_check
python .\run_stagev6.py --mode train --n-jobs 12 --bootstrap-n 200
```

`--mode train` reuses the included stagev5 feature CSVs directly. To replace prior Stagev6 outputs explicitly:

```powershell
python .\run_stagev6.py --mode train --n-jobs 12 --bootstrap-n 200 --overwrite
```

To execute the read-only audit notebook after training:

```powershell
python .\run_stagev6.py --mode render_notebook
```

## Main output files

All canonical result files are written to `output/final_report/`:

- `stagev6_model_ranking_by_external_accuracy.csv`
- `stagev6_external_performance_report.csv`
- `stagev6_cv_summary.csv`
- `stagev6_bootstrap_ci.csv`
- `stagev6_test_predictions_all_models.csv`
- `stagev6_late_gate_performance.csv`
- `stagev6_nonlate_branch_performance.csv`
- `stagev6_route_diagnostics.csv`
- `stagev6_component_specifications.csv`
- `stagev6_selected_model_summary.md`
- `stagev6_experiment_report.md`

`stagev6_test_predictions_all_models.csv` contains `p_late`, `late_route`, `p_ad_given_nonlate`, `p_ad_mixture`, `final_route`, `y_pred`, and `route_error_type` for every external sample and every cascade model.
