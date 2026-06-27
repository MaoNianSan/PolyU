# Stagev8.5 — Global Fresh Rerun Package

## Full-rerun status

This distribution is intentionally clean: it contains the raw CSV inputs, frozen anchor artifact, hash-locked Stagev5/Stagev4 feature source, Stagev8.5 severity code, and audit notebook, but **no generated E/M/L feature CSVs, embedding cache, late-score cache, trained models, reports, figures, or completion sentinel**.

Use the single global command below to execute one auditable run from raw data to rendered IPython notebook:

```powershell
python .\run_stagev8.py --mode global_rerun --n-jobs 12 --bootstrap-n 200 --stability-seeds 0-29
```

The mode is deliberately destructive only for generated Stagev8.5 outputs. It preserves `input/raw/` and all hash-locked reference assets. It performs:

1. source and raw-input audits;
2. fresh Stagev5 E/M/L reconstruction through live BGE-M3 and qwen3-235b-a22b requests;
3. frozen Stagev5 AD/control anchor parity audit;
4. Stagev8.5 preflight and ordinal-severity training;
5. 30-seed stability and 200-bootstrap external evaluation;
6. execution of `notebooks/stagev8_5_result_audit.ipynb` against the newly generated outputs.

No cache replay is accepted in `global_rerun`. The final audit file is `output/checks/stagev8_5_global_rerun_manifest.json`.

## Scientific scope

Stagev8.5 separates two claims:

1. **Disease decision:** the frozen Stagev5 `early_middle__svc__poly3(E+M)` classifier remains the AD/control anchor and is never retrained.
2. **Severity output:** two prespecified conditional probability heads produce an **MMSE-informed ordinal cognitive-severity tendency**, not a clinical early/middle/late diagnosis.

The fixed MMSE strata are:

```text
high-MMSE AD:          MMSE >=21
intermediate-MMSE AD:  MMSE 15–20
low-MMSE AD:           MMSE <=14
```

The two threshold heads are:

```text
T20: P(MMSE <=20 | AD)                         = E+M elastic-net logistic regression
T14: P(MMSE <=14 | MMSE <=20, AD)              = raw F8 L-only L2 logistic regression

q_high         = 1 - s20
q_intermediate = s20 * (1 - s14)
q_low          = s20 * s14
severity_score = q_intermediate + 2*q_low
```

`severity_score` ranges from 0 to 2. Larger values indicate a stronger low-MMSE / higher cognitive-severity tendency. `reported_severity_stratum` is withheld as `AD_severity_indeterminate` unless:

```text
max(q) >= 0.50 and top1(q)-top2(q) >= 0.10
```

## Feature provenance

The completed numerical run used fresh API feature reconstruction:

```text
raw CSV
→ copied Stagev5 Stagev2 preprocessing
→ copied Stagev5 E BM25
→ copied Stagev5 M BGE-M3
→ copied Stagev5/Stagev4 P4/F8 late scoring
→ copied Stagev5 feature adapter/validation
→ E=61, M=1024, L=8
```

The copied Stagev5/Stagev4 feature source remains hash-locked. Windows `curl.exe` / Schannel was used only as a runtime transport shim because Python `requests`/OpenSSL could not complete the provider TLS exchange; it did not alter the feature inputs, model names, endpoints, request payload dictionaries, parsers, cache keys, feature aggregation, or score rules.

## Read-only result audit notebook

The notebook is intentionally read-only. It loads existing CSV/JSON/Markdown/PNG artifacts and never:

- calls an external API;
- extracts features;
- fits or selects models;
- changes model outputs;
- overwrites reports.

From the project root:

```powershell
python .\run_stagev8.py --mode render_notebook
```

Or open `notebooks/stagev8_5_result_audit.ipynb` in VS Code/Jupyter and run all cells.

## Primary result files

All completed outputs live under `output/final_report/` and use the `stagev8_5_` prefix. Key files include:

- `stagev8_5_final_run_summary.json`
- `stagev8_5_feature_source_audit.json`
- `stagev8_5_external_severity_scores.csv`
- `stagev8_5_external_ordinal_metrics.csv`
- `stagev8_5_external_threshold_metrics.csv`
- `stagev8_5_external_three_strata_metrics.csv`
- `stagev8_5_external_selective_metrics.csv`
- `stagev8_5_bootstrap_ci.csv`
- `stagev8_5_experiment_report.md`
- `stagev8_5_version_normalization_audit.json`

The external set is a reference evaluation only. It was not used to choose MMSE boundaries, feature blocks, model families, hyperparameters, confidence thresholds, or reporting margins.
