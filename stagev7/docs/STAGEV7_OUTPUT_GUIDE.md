# Stagev7 output guide

## Primary files

- `stagev7_final_cascade_predictions.csv`: C06 primary system, sample-level probabilities, gate decisions, final stage, AD status, and path.
- `stagev7_final_primary_performance.csv`: primary external binary and stage metrics.
- `stagev7_bootstrap_ci.csv`: bootstrap confidence intervals for C06.
- `stagev7_gate_model_ranking_all.csv`: internal CV gate panels.
- `stagev7_cascade_ranking_external_exploratory.csv`: all pre-specified cascades, clearly exploratory.
- `stagev7_flat_multiclass_baseline_performance.csv`: parallel multiclass LR/SVC comparison.
- `stagev7_final_stage_strict_audit.csv`: preserves `AD_high_MMSE`; do not interpret as a four-class primary metric.

## Selection discipline

C06 is the primary cascade before external evaluation. CV identifies reasonable fitted components within each bounded gate panel. The held-out external set is not used to fit, preprocess, tune, select thresholds, or choose C06.


## Read-only IPY result audit

Use `notebooks/stagev7_result_check.ipynb` only after `--mode train` has finished, or execute it safely with:

```powershell
python run_stagev7.py --mode render_notebook
```

The notebook is display-only. It never fits a classifier, regenerates E/M/L features, calls an API, or changes saved reports. It searches for the project root from the root directory, `notebooks/`, or `STAGEV7_ROOT`; absent artifacts are shown as availability notices rather than exceptions.
