# stagev5

`stagev5` is a locked three-stage Cookie Theft AD-vs-control benchmark. The repository is organized so GitHub readers can inspect the completed experiment, rerun safety checks, and reproduce classifier training from already extracted E/M/L feature CSV files.

## Do-Not-Rerun Warning

Do not run `--mode all` or `--mode extract_features` unless you intentionally want to regenerate features and potentially call external APIs. The committed result files are treated as completed experiment artifacts and should not be edited to improve metrics.

Safe display/check commands:

```powershell
python .\run_stagev5.py --mode self_check
python .\scripts\check_existing_results.py
python .\run_stagev5.py --mode render_notebook
```

Training from existing E/M/L CSV files only:

```powershell
python .\run_stagev5.py --mode train --n-jobs 12 --bootstrap-n 200
```

Full regeneration, for controlled reruns only:

```powershell
python .\run_stagev5.py --mode all --n-jobs 12 --bootstrap-n 200
```

## Feature Policy

| Feature family | Locked source | Model input policy |
|---|---|---|
| E | `stagev2.zip` | Strictly follows the stagev2 early BM25 feature extraction. |
| M | `stagev2.zip` | Strictly follows the stagev2 BGE-M3 middle embedding extraction. |
| L | `stagev4_unmasked_form_comparator.zip` | Strictly follows the stagev4 unmasked late P4/F8 extraction. |

E strictly follows `stagev2.zip` early feature extraction. M strictly follows `stagev2.zip` middle feature extraction. L strictly follows `stagev4_unmasked_form_comparator.zip` late P4 unmasked F8 extraction. The adapter only aligns IDs, columns, manifests, and model blocks; it does not redefine E, M, or L.

Expected model feature counts:

- E model features: 61
- M model features: 1024
- L raw F8 model features: 8
- L auxiliary diagnostic features: reported when present, but not used as model inputs

See [docs/STAGEV5_FEATURE_POLICY.md](docs/STAGEV5_FEATURE_POLICY.md) for the detailed provenance boundary.

## Data Layout

```text
input/raw/                                  Raw transcript CSV files
input/reference_stagev4_late_metadata/      Frozen train/external metadata for L
assets/reference_stagev2_cache/             Optional copied BGE-M3 cache
output/features/E/raw_stagev2/              Existing E CSV files
output/features/M/raw_stagev2/              Existing M CSV files
output/features/L/raw_stagev4/              Existing L CSV files
output/final_report/                        GitHub-facing completed reports
notebooks/stagev5_result_check.ipynb        Display-only result audit notebook
docs/                                       Reproducibility and output guides
```

## Environment Setup

```powershell
cd stagev5
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`MAAS_API_KEY` is only needed for late-feature extraction. It is not needed for `self_check`, `train` from existing feature CSV files, `render_notebook`, or the read-only scripts.

## Run Modes

| Mode | Reads existing outputs | Trains models | Extracts E/M/L | May call API |
|---|---:|---:|---:|---:|
| `self_check` | yes | no | no | no |
| `check_api` | yes | no | no | no |
| `render_notebook` | yes | no | no | no |
| `train` | yes, existing E/M/L CSV | yes | no | no |
| `extract_features` | reads raw/cache | no | yes | yes |
| `all` | reads raw/cache | yes | yes | yes |

`--mode train` reads existing E/M/L CSV files and does not call APIs. `--mode render_notebook` reads existing CSV/JSON/MD/PNG files and does not call APIs. `--mode all` or `--mode extract_features` can trigger feature extraction and API usage.

## Reproduce From Existing Features

1. Run `python .\run_stagev5.py --mode self_check` to validate source layout and feature schemas.
2. Run `python .\run_stagev5.py --mode train --n-jobs 12 --bootstrap-n 200` to rerun the stagev2 classifier panel from the existing E/M/L files.
3. Run `python .\run_stagev5.py --mode render_notebook` to refresh the GitHub notebook from saved results.

The scientific contract is fixed: 10-fold CV, stagev2 classifier panel, external accuracy ranking, and bootstrap `n=200`.

## Main Output Files

The GitHub-facing report directory is `output/final_report/`:

- `stagev5_model_ranking_by_external_accuracy.csv`
- `stagev5_external_performance_report.csv`
- `stagev5_cv_summary.csv`
- `stagev5_bootstrap_ci.csv`
- `stagev5_generalization_gap.csv`
- `stagev5_oof_predictions_top10.csv`
- `stagev5_test_predictions_all_models.csv`
- `stagev5_stage_subgroup_accuracy.csv`
- `stagev5_error_analysis.csv`
- `stagev5_selected_model_summary.md`
- `stagev5_experiment_report.md`
- `stagev5_feature_source_manifest.json`
- `stagev5_leakage_check.json`
- `figures/*.png`

Use `python .\scripts\summarize_outputs.py` for a compact read-only inventory.

## Notebook Guide

`notebooks/stagev5_result_check.ipynb` is display-only. It reads saved CSV, JSON, Markdown, and PNG files; it never trains, extracts features, or calls APIs. It shows experiment identity, feature-source audit, model ranking by external accuracy, selected-model performance, bootstrap CIs, feature-block comparison, confusion matrix, subgroup accuracy, error analysis, leakage checks, and the final conclusion.

## Leakage-Control Notes

- E and M come from the locked stagev2 implementations.
- L comes from the locked stagev4 unmasked P4/F8 implementation.
- BM25 fitting, imputers, and scalers are kept inside the existing training logic.
- The external set is used for final ranking/reporting, not for model fitting or preprocessing fit.
- Stage/severity outputs are subgroup diagnostics for the binary disease decision, not a supervised four-class stage classifier.

## GitHub Upload Notes

Recommended to include:

- Source code under `src/`, `run_stagev5.py`, `scripts/`, `configs/`, and `tests/`
- `README.md`, `.gitignore`, `requirements.txt`, and `docs/`
- `notebooks/stagev5_result_check.ipynb`
- `output/final_report/*.md`, `*.csv`, `*.json`
- `output/final_report/figures/*.png`

Usually exclude local caches, model pickles, temporary files, virtual environments, logs, and `output/run_external_accuracy_selection/models/`.
