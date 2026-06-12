# Global repair summary

This version applies the confirmed stagev2 repair policy:

## Runtime instructions

1. Use the current system Python or active Conda Python. Do not create a `.venv`.
2. Run commands from the `stagev2` project root containing `run_stagev2.py`, `requirements.txt`, `src/`, and `stage_core/`.
3. Install dependencies with `python -m pip install -r requirements.txt`.
4. Put raw CSV files under `stagev2\input\raw\`.
5. Start the clean global pipeline with `python .\run_stagev2.py --data-root .\input\raw`.

## Core logic changes

1. Raw data are the only runtime input.
2. Historical outputs, old features, old caches, and old model files are removed from the project package and not reused by code.
3. Feature extraction is regenerated in the current project:
   - early: BM25 information-unit features;
   - middle: Huawei MaaS BGE-M3 window embeddings;
   - late: qwen3-235b-a22b content-masked expressive-form LLM scores.
4. MMSE thresholds are unified:
   - early / mild: 21-24;
   - middle / moderate: 13-20;
   - late / severe: <=12.
5. Main label order is `[disease, early, middle, late]`.
6. GridSearchCV scoring is `accuracy`.
7. Classification threshold is fixed at 0.5.
8. Models are ranked and selected by held-out external accuracy.
9. CV and OOF predictions are internal diagnostics only.
10. External set is described as held-out external validation, not unbiased final test after selection.
11. RBF SVM is retained and reported as a nonlinear baseline.
12. All fitted models are saved locally under `output/models/all_models/`.
13. The selected model is saved under `output/models/selected/`.

## Canonical output names

- `stagev2_model_ranking_by_external_accuracy.csv`
- `stagev2_external_performance_report.csv`
- `stagev2_cv_summary.csv`
- `stagev2_oof_predictions_top10.csv`
- `stagev2_test_predictions_all_models.csv`
- `stagev2_selected_model_summary.md`
- `stagev2_experiment_report.md`
- `stagev2_leakage_check.json`

## GitHub maintenance

The cleaned repository excludes local data, generated output, cache files, zip files, joblib models, API keys, and Python cache files through `.gitignore`.
