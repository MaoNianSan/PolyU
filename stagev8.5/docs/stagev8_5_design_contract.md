# Stagev8.5 design contract

## Version status

Stagev8.5 is the formal name for the completed Stagev8.4 MMSE-informed ordinal severity analysis. Numerical outputs are retained without feature extraction, model fitting, resampling, or threshold changes. The version-normalization audit records the original artifact hashes.

## Immutable feature source

- Fresh raw-only Stagev5/Stagev4 reconstruction completed in the source run.
- Feature dimensions are locked: E=61, M=1024, L=8.
- Copied Stagev5/Stagev4 source is hash-locked.
- Raw MMSE is used only as target/evaluation metadata, never as a model feature.

## Frozen disease anchor

- Frozen Stagev5 `early_middle__svc__poly3(E+M)` artifact.
- No retraining, threshold tuning, or external selection.
- `predicted_AD` is generated directly at threshold 0.50.

## Fixed ordinal severity heads

- T20: E+M elastic-net logistic regression; target `MMSE <=20` vs `MMSE >=21` within true AD training samples.
- T14: raw F8 L-only L2 logistic regression; target `MMSE <=14` vs `MMSE 15–20` among true AD samples with MMSE <=20.
- `q_high = 1-s20`, `q_intermediate = s20(1-s14)`, `q_low = s20*s14`.
- `severity_score = q_intermediate + 2*q_low`.

## Evaluation hierarchy

Primary: Spearman rho, Kendall tau, and pairwise ordinal accuracy between `severity_score` and `30-MMSE` among true AD samples.

Secondary: T20/T14 threshold metrics, nested OOF severity metrics, 30-seed stability, selective coverage, uncertainty distribution, and auxiliary three-strata metrics.

External data are reference evaluation only.

## Read-only notebook

`notebooks/stagev8_5_result_audit.ipynb` only reads saved artifacts. It cannot trigger API calls, feature extraction, model fitting, or output overwrites.
