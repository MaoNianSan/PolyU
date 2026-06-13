# stagev3 v2-logic patch

This package keeps the stagev3 experiment/report/classifier framework, but rewrites the model-call-sensitive feature extraction logic to be stagev2-compatible and self-contained.

## What changed

### 1. Middle BGE-M3 embedding

`src/middle_features.py` was rewritten as a v3 wrapper around the stable stagev2 middle logic:

- uses stagev2 regex word tokenization: `[A-Za-z']+|\d+`;
- uses the same word-window design: `WINDOW_SIZE_WORDS=15`, `STRIDE_WORDS=5`;
- keeps the same cache key convention: `sha256(EMBEDDING_MODEL + "\n" + window_text)`;
- reads and writes `output/cache/huawei_bge_m3_embedding_cache.csv`;
- accepts both stagev3 cache source `api` and stagev2 cache source `huawei_maas_api`;
- preserves either v2 or v3 cache column order when appending new rows, avoiding cache corruption;
- flushes cache after every successful batch;
- escalates safety fallback as: raw window -> safety mask -> aggressive mask -> neutral length-preserving window -> minimal neutral window;
- writes failed safety windows to `output/cache/embedding_safety_failures.csv`.

The package no longer needs any file from stagev2 at runtime. A complete v2 cache may still be copied into `output/cache/` and will be accepted.

### 2. Late LLM expressive-form scoring

`src/late_features.py` keeps the stagev3 output schema but adopts a stronger stagev2-style content mask:

- `PERSON`, `OBJECT`, `PLACE`, `ACTION`, `NUMBER` typed placeholders;
- no bracketed placeholders in the main mask;
- expanded Cookie Theft scene/action vocabulary;
- cache is flushed after each successful LLM request.

### 3. What did not change

The following remain stagev3-native:

- `run_stagev3.py` entry point;
- output file names expected by v3;
- early feature variants;
- feature merge and scale/interactions;
- model registry and validation;
- final report and notebook generation.

## Recommended command

```powershell
cd D:\research\H.L.Liang-Lab\Code\expore\stagev3_v2logic

$env:HUAWEI_MAAS_TRUST_ENV="false"
$env:HUAWEI_MAAS_SSL_VERIFY="true"
$env:EMBEDDING_BATCH_SIZE="1"

python run_stagev3.py --check-api
python run_stagev3.py --mode seed2026 --cv-mode exact
```

## Expected behavior

If the existing cache is incomplete, the middle stage will continue from the cached rows and only call the API for missing windows. If a content-safety 403 occurs, the rewritten middle module should retry with masked/neutral text and continue, while storing the resulting vector under the original window cache key.
