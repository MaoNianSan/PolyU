# stagev3 v2-compatible repair summary

Implemented repairs:

- Restored v2-compatible sample IDs: `AD_0001`, `CTRL_0001`, `TEST_0001`.
- Replaced simplified early features with v2 `early_v5_mild_sensitive` logic.
- Restored middle features from 44 compressed dimensions to 1025 v2-compatible dimensions.
- Restored SVC/LR grids, including `coef0=[0.0,1.0]`, class-weight search, and `probability=True`.
- Replaced `predict()`-based evaluation with probability-threshold evaluation.
- Added v2 special models: `StageScoreCalibratedLR` and `MLPSVCLateCalibratedLR`.
- Added v2-derived late form features while preserving the existing complete late cache.
- Added per-model external prediction output for full seed2026 runs.
- Added `scripts/run_v2_compat_anchor_check.py` as a quick regression test.
- Added optional `--cv-mode v2_repeated` for 5-fold × 3 repeated CV.

Smoke-check result:

```text
earlyv0 + early_middle__svc__poly3:
external_accuracy = 0.845070
external_sensitivity = 0.942857
external_auc = 0.853968
confusion matrix = tn=27, fp=9, fn=2, tp=33
n_features = 1086
```

This confirms that the major v2 high-performance anchor has been restored. The full 204-model exact grid was not completed during packaging because the restored v2-compatible parameter grids and special OOF models are much heavier than the previous simplified v3 protocol.
