# stagev3 v2logic progress patch

This package keeps the stagev3 pipeline and stagev2-compatible middle feature logic, but improves observability and network failure behavior during BGE-M3 cache filling.

## Changes

1. Added `--diagnose-middle-cache` to report middle cache coverage without API calls.
2. Added `--middle-only` to run preprocessing, early features, and middle feature/cache extraction only.
3. Added an inner tqdm progress bar for pending middle API windows.
4. Added batch-level progress updates to `output/final_report/run_progress.json`.
5. Added embedding-specific timeout/retry environment variables:
   - `EMBEDDING_TIMEOUT`, default `45`
   - `EMBEDDING_MAX_RETRIES`, default `2`
   These only affect the middle embedding client. The late LLM keeps the normal MaaS timeout/retry defaults.

## Recommended commands

```powershell
cd D:\research\H.L.Liang-Lab\Code\expore\stagev3_v2logic_progress

$env:HUAWEI_MAAS_TRUST_ENV="false"
$env:HUAWEI_MAAS_SSL_VERIFY="true"
$env:EMBEDDING_BATCH_SIZE="1"
$env:EMBEDDING_TIMEOUT="45"
$env:EMBEDDING_MAX_RETRIES="2"

python run_stagev3.py --diagnose-middle-cache
python run_stagev3.py --middle-only --mode seed2026 --cv-mode exact
python run_stagev3.py --mode seed2026 --cv-mode exact
```

If the middle cache is complete, the full run should not call the BGE-M3 embedding API at stage 3.
