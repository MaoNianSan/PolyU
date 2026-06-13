# stagev3

Clean regenerated project for AD connected-speech stage-feature classification.

Important design points:

1. Code remains a standalone `stagev3` package; the middle/late API-call logic has been rewritten to be stagev2-compatible without requiring any external stagev2 files at runtime.
2. The only copied artifacts are the raw CSV files under `input/raw/`.
3. Early, middle, and late features are extracted from raw normalized text when valid stagev3 feature CSVs are absent.
4. Feature extraction only needs to be done once. After valid feature CSVs are produced under `output/features/`, later `seed2026`, `stability`, or `all` runs reuse them automatically.
5. Huawei BGE-M3 embedding and LLM late-feature extraction follow a stage2-compatible API/cache convention: `MAAS_API_KEY`, `https://api.modelarts-maas.com/v1`, `/embeddings`, `/chat/completions`, cache-first behavior. The canonical environment variable is `MAAS_API_KEY`. Middle cache accepts both v3 `source=api` rows and v2 `source=huawei_maas_api` rows.
6. Real API or complete real-API cache is required by default. Local surrogate middle/late feature generation is disabled. Any historical `local_surrogate` cache rows or feature manifests are treated as invalid and removed/regenerated.

## v2-logic rewrite note

This version is intended to be a standalone replacement for the previous `stagev3` folder. It does not import or read code from `stagev2`, but `src/middle_features.py` now uses the stagev2-equivalent word-window/cache-key logic inside the stagev3 output wrapper. Details are in `docs/stagev3_v2logic_patch.md`.

## Install

```bash
pip install -r requirements.txt
```

## API key

Linux/macOS:

```bash
export MAAS_API_KEY=your_key_here
```

Windows PowerShell:

```powershell
$env:MAAS_API_KEY="your_key_here"

# Lightweight structure check, no API call, no model training
python run_stagev3.py --dry-run

# Optional API check
python run_stagev3.py --check-api

# Run seed2026 only
python run_stagev3.py --mode seed2026 --cv-mode exact

# Run stability after features are extracted
python run_stagev3.py --mode stability --seeds 0-29 --cv-mode exact
```

The code reads only `MAAS_API_KEY` for Huawei MaaS authentication.
If `MAAS_API_KEY` is set in Windows Environment Variables GUI, close and reopen
PowerShell before running the project. An already-open PowerShell process does
not automatically receive GUI environment-variable changes.

In the tested Windows PowerShell environment, `curl.exe` can access Huawei MaaS
normally, while Python `requests` may fail when routed through system proxy
settings. Therefore, the project defaults to
`HUAWEI_MAAS_TRUST_ENV=false`.

## API and environment checks

```powershell
# Check whether Python can see the key
python -c "import os; print(bool(os.getenv('MAAS_API_KEY')))"

# Check API connection only
python run_stagev3.py --check-api

# Run seed2026 exact experiment
python run_stagev3.py --mode seed2026 --cv-mode exact

# Run stability after features have been extracted
python run_stagev3.py --mode stability --seeds 0-29 --cv-mode exact
```

`--dry-run` checks project structure, raw files, feature completeness, model
registry size, output directories, and visualization imports. It does not call
an API, extract features, or train models.

`--check-api` sends one minimal request to the Huawei MaaS embedding endpoint.
It does not extract project features or train models. The
startup diagnostic prints only a masked key, never the complete key.

When `HUAWEI_MAAS_TRUST_ENV=true`, the client first honors environment/system
proxy settings. If that route fails with SSL, timeout, or connection errors, it
automatically retries with a direct connection and reuses that route for the
rest of the process. The tested Windows configuration recommends leaving this
setting at its default value of `false`. Certificate verification remains
enabled unless `HUAWEI_MAAS_SSL_VERIFY=false` is explicitly set.

Supported network settings and defaults:

```text
HUAWEI_MAAS_SSL_VERIFY=true
HUAWEI_MAAS_TRUST_ENV=false
HUAWEI_MAAS_TIMEOUT=120
HUAWEI_MAAS_MAX_RETRIES=5
EMBEDDING_BATCH_SIZE=1
HUAWEI_MAAS_BASE_URL=https://api.modelarts-maas.com/v1
EMBEDDING_MODEL=bge-m3
```

For a temporary SSL EOF diagnostic only:

```powershell
$env:HUAWEI_MAAS_SSL_VERIFY="false"
$env:EMBEDDING_BATCH_SIZE="1"
python run_stagev3.py --check-api
```

## Run modes

```bash
# Only run primary seed=2026
python run_stagev3.py --mode seed2026 --cv-mode exact

# Run stability under seeds 0-29
python run_stagev3.py --mode stability --seeds 0-29 --cv-mode exact

# Run both primary and stability
python run_stagev3.py --mode all --seeds 0-29 --cv-mode exact

# Fast structural/debug run
python run_stagev3.py --mode all --seeds 0-29 --cv-mode fast

# Explicit heavy validation
python run_stagev3.py --validate strict
```

Default command:

```bash
python run_stagev3.py
```

is equivalent to:

```bash
python run_stagev3.py --mode seed2026 --cv-mode exact
```

## Feature extraction policy

Default behavior:

```text
1. If valid stagev3 feature CSVs already exist, reuse them.
2. If feature CSVs are missing or invalid, extract from raw text.
3. Extraction uses real API or complete real-API cache only.
4. If MAAS_API_KEY is absent and cache is incomplete, stop with a clear error.
5. Local surrogate fallback is disabled.
```

Force re-extraction:

```bash
python run_stagev3.py --mode seed2026 --cv-mode exact --force-features
```

Use this only when you intentionally want to delete generated feature CSVs and re-extract from raw text using API/cache.

## Selected-after-seed2026 mode

```bash
# After full seed2026 has already been completed
python run_stagev3.py --mode selected_after_seed2026 --seeds 0-29 --cv-mode exact --min-external-accuracy 0.75
```

This command reads `output/final_report/seed2026_main_results.csv`, selects
models with seed2026 `external_accuracy >= 0.75`, runs multi-seed stability
only for those selected model specs, and reports external accuracy
mean/std/95% CI. Selection happens once and the fixed model set is identified
by `early_variant + model_spec_id`; models are not reselected within individual
seeds. Valid files under `output/features/` are reused under the normal feature
extraction and API/cache policy.

The selected mode defaults to `--n-jobs 8 --resume`. Each completed seed is
saved under:

```text
output/final_report/selected_stability_checkpoints_external075/seed_000.csv
```

Re-running the same command skips valid completed seed checkpoints. Use
`--force-rerun` to recompute all requested seeds, or `--no-resume` to ignore
existing checkpoints for that run. Selected mode requires complete valid
features and does not call the feature APIs or re-extract features.

Selected stability CI reflects random-seed / CV-split variability among models
selected by seed2026 external accuracy. Because external accuracy is used for
model selection, this is not an unbiased final external-test confidence
interval.

## Output scale

The target model registry is:

```text
12 normal feature blocks × 8 classifier variants + 6 special models = 102 specs
102 specs × earlyv0/earlyv1 = 204 main rows
204 × 30 seeds = 6120 stability rows
```

## Main output files

```text
output/final_report/seed2026_main_results.csv
output/final_report/seed_stability_results.csv
output/final_report/seed_stability_summary.csv
output/final_report/earlyv1_gain_over_earlyv0.csv
output/final_report/scale_gain_seed2026.csv
output/final_report/scale_gain_stability.csv
output/final_report/early_distribution_summary.csv
output/final_report/stagev3_summary.md
output/final_report/run_manifest.json
output/final_report/run_progress.json
output/final_report/run_progress.md
output/final_report/validation_report.md
output/final_report/preprocessing_audit.csv
output/final_report/figures/*.png
notebooks/stagev3_result_check.ipynb
```

Feature extraction is performed once and reused by `seed2026`, `stability`,
and `all` modes. Process visualization is written to terminal progress bars,
`run_progress.json`, and `run_progress.md`. Result visualization is written to
`output/final_report/figures/`.

Validation defaults to `standard`. Use `--validate light` for basic structure
checks or `--validate strict` for full column, uniqueness, CI, gain, and
notebook checks.

## Metrics

Only these five metrics are retained:

```text
Accuracy, Precision, Recall, F1-score, AUC
```

CI is the 95% t-interval of external accuracy across stability seeds. It reflects seed/CV-split randomness, not binomial test-case uncertainty.
