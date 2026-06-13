# stagev3 v2-compatible full repair programming prompt

## Goal

Repair stagev3 so it can reproduce the stagev2 high-performance logic before adding new stagev3 mechanisms. The first regression target is the v2 anchor model:

```text
early_middle__svc__poly3
external_accuracy ~= 0.845070
sensitivity ~= 0.942857
specificity ~= 0.750000
```

This target is a compatibility check, not an unbiased final-test claim.

## Required repairs

1. Restore v2-compatible sample IDs:
   - AD: `AD_0001`, `AD_0002`, ...
   - Control: `CTRL_0001`, `CTRL_0002`, ...
   - External test: `TEST_0001`, `TEST_0002`, ...

2. Restore early features to `early_v5_mild_sensitive`:
   - Use `OLD10 + EXPANDED + RELATION` units.
   - Keep raw BM25 unit scores.
   - Add coverage, missing counts, critical missing indicators, integrity score, and omission risk score.
   - Exclude `token_count` from classifier input.
   - Expected model-input dimensions:
     - `earlyv0`: 61
     - `earlyv1`: 62, adding `early_content_efficiency` only.

3. Restore middle features to full v2 BGE-M3 mean embedding:
   - Use v2-compatible word windows: window size 15, stride 5.
   - Use real Huawei BGE-M3 API cache or API; no surrogate rows.
   - Aggregate window embeddings by sample mean.
   - Keep full 1024 embedding dimensions.
   - Retain v2-compatible `middle_window_id` mean feature.
   - Expected middle dimension: 1025.

4. Restore SVC and LR grids:
   - `LR_C_GRID = [0.03, 0.1, 1.0]`
   - `SVC_C_GRID = [0.1, 1.0, 3.0]`
   - `CLASS_WEIGHTS = [None, "balanced"]`
   - `POLY_COEF0 = [0.0, 1.0]`
   - All SVC models must use `probability=True`.

5. Restore v2-compatible probability evaluation:
   - Prefer `predict_proba(X)[:, 1]`.
   - Use `y_pred = (prob >= 0.5)`.
   - Do not use raw `SVC.predict()` as the main external prediction rule.

6. Restore special models:
   - Implement `StageScoreCalibratedLR` using OOF scores `s_E`, `s_M`, `s_L` and interactions.
   - Implement `MLPSVCLateCalibratedLR` using MLP/SVC early-middle scores plus late LR score.
   - These models must operate on raw all-stage feature order `[early, middle, late]`, not on v3 activation-summary placeholders.

7. Restore late-stage compatibility enough for feature-block comparability:
   - Keep current cache-compatible late API logic to avoid unnecessary re-calls.
   - Add v2-derived form variables: `late_form_quality_inverse`, `late_token_count`, `late_unique_ratio`, `late_punctuation_count`.

8. Restore diagnostic output:
   - Output per-model external predictions for the seed2026 full run as `seed2026_external_predictions_all_models.csv`.
   - Include `sample_id`, `mmse`, `y_true`, `y_pred`, `y_score`, and `correct`.

9. Add quick anchor check:
   - Provide `scripts/run_v2_compat_anchor_check.py`.
   - It should run without API if feature/cache files are complete.
   - It should verify `early_middle__svc__poly3` external accuracy.

## Verification commands

```powershell
python run_stagev3.py --diagnose-middle-cache
python scripts/run_v2_compat_anchor_check.py
python run_stagev3.py --mode seed2026 --cv-mode exact --validate standard
```

Optional v2 CV protocol:

```powershell
python run_stagev3.py --mode seed2026 --cv-mode v2_repeated --validate standard
```

## Acceptance checks

Minimum expected after repair:

```text
earlyv0 early_middle__svc__poly3 fixed anchor:
- n_features = 1086
- external_accuracy = 0.845070
- external_sensitivity = 0.942857
- confusion matrix = tn=27, fp=9, fn=2, tp=33
```

The full 204-model grid may take substantially longer than the anchor check because the v2-compatible grids restore multiple C/class_weight/coef0 combinations and real special-model inner OOF training.
