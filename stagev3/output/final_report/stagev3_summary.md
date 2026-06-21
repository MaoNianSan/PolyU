# stagev3 summary

## 1. Run setting
- run_mode: `seed2026`
- cv_mode: `exact`
- main_seed: `2026`
- stability_seeds: `[]`
- n_model_specs_per_early_variant: `102`
- n_expected_main_rows: `204`
- n_expected_stability_rows: `0`
- raw_data_source: `D:\research\H.L.Liang-Lab\Code\Github\stagev3\input\raw`
- api_mode: `real_api`
- validation_status: `PASS`

## 2. Data setting
- train_n: 166
- external_n: 71
- train_label_counts: {'1': 87, '0': 79}
- external_label_counts: {'0': 36, '1': 35}

## 3. Feature extraction
- early/middle/late features are extracted from normalized raw text when valid stagev3 feature CSVs are absent.
- Once generated, valid feature CSVs are reused across seed2026/stability/all runs; use --force-features to re-extract.
- Historical stage2 feature CSV outputs are not reused.
- Huawei BGE-M3 and LLM API logic is rewritten using the stage2-compatible MAAS_API_KEY, endpoint, retry, and cache convention.
- Real API or complete real-API cache is required by default; local surrogate feature generation is disabled.
- feature_summary: `{"early": {"earlyv0": "completed", "earlyv1": "completed"}, "middle": {"feature_family": "middle_huawei_bge_m3_window_embedding", "feature_schema": "v2_full_mean", "middle_keep_dims": 1024, "middle_include_v3_stats": false, "regenerated_from_raw_text": true, "historical_feature_outputs_reused": false, "api_logic_source": "stagev2_cache_first_logic_v3_wrapper", "windowing_source": "stagev2_regex_word_windows", "api_or_cache_mode": "api", "feature_load_mode": "existing_feature_csv", "feature_extracted_this_run": false, "local_surrogate_allowed": false, "cache_path": "D:\\research\\H.L.Liang-Lab\\Code\\Github\\stagev3\\output\\cache\\huawei_bge_m3_embedding_cache.csv", "new_cache_rows": 4635, "unique_texts": 4839, "ignored_surrogate_cache_rows": 0, "accepted_cache_sources": ["api", "huawei_maas_api", "huawei_maas_api_v2"], "safety_masked_api_calls": 11, "safety_level_counts": {"safety_mask": 8, "aggressive_mask": 3}, "stagev2_cache_compatible": true, "n_train_rows": 166, "n_external_rows"`

## 4. Main seed=2026 top results
| early_variant   | model_spec_id                   |   external_accuracy |   external_f1 |   external_auc | external_accuracy_95ci   |
|:----------------|:--------------------------------|--------------------:|--------------:|---------------:|:-------------------------|
| earlyv1         | early_middle_scale__svc__poly3  |            0.816901 |      0.831169 |       0.849206 |                          |
| earlyv0         | early_middle_scale__svc__poly3  |            0.816901 |      0.831169 |       0.848413 |                          |
| earlyv0         | all__svc__poly3                 |            0.816901 |      0.831169 |       0.846032 |                          |
| earlyv1         | all__svc__poly3                 |            0.816901 |      0.831169 |       0.846032 |                          |
| earlyv0         | early_middle__svc__poly3        |            0.816901 |      0.826667 |       0.853968 |                          |
| earlyv1         | early_middle__svc__poly3        |            0.816901 |      0.826667 |       0.853175 |                          |
| earlyv0         | middle_only__svc__poly3         |            0.802817 |      0.815789 |       0.84127  |                          |
| earlyv1         | middle_only__svc__poly3         |            0.802817 |      0.815789 |       0.84127  |                          |
| earlyv1         | mlp_svc_late_calibrated__lr__l2 |            0.788732 |      0.810127 |       0.81746  |                          |
| earlyv0         | middle_only__svc__rbf           |            0.788732 |      0.805195 |       0.845238 |                          |

## 6. earlyv1 vs earlyv0
| model_spec_id                         |   earlyv0_external_accuracy |   earlyv1_external_accuracy |   external_accuracy_delta | earlyv1_has_gain   |
|:--------------------------------------|----------------------------:|----------------------------:|--------------------------:|:-------------------|
| mlp_svc_late_calibrated__lr__l2       |                    0.760563 |                    0.788732 |                 0.028169  | True               |
| sequential_interactions__lr__l1       |                    0.71831  |                    0.746479 |                 0.028169  | True               |
| early_late__svc__poly3                |                    0.704225 |                    0.71831  |                 0.0140845 | True               |
| early_late__svc__linear               |                    0.704225 |                    0.71831  |                 0.0140845 | True               |
| stage_activation_summary__svc__linear |                    0.619718 |                    0.633803 |                 0.0140845 | True               |
| stage_activation_summary__svc__poly3  |                    0.619718 |                    0.633803 |                 0.0140845 | True               |
| all_plus_interactions__svc__sigmoid   |                    0.690141 |                    0.704225 |                 0.0140845 | True               |
| stage_score_early_middle__lr__l2      |                    0.760563 |                    0.774648 |                 0.0140845 | True               |
| early_middle__svc__sigmoid            |                    0.746479 |                    0.760563 |                 0.0140845 | True               |
| early_only__svc__rbf                  |                    0.690141 |                    0.704225 |                 0.0140845 | True               |

## 7. Scale gain under seed=2026
| early_variant   | model_variant   | raw_feature_block   | scale_feature_block     |   raw_external_accuracy |   scale_external_accuracy |   scale_gain_delta | scale_has_gain   |
|:----------------|:----------------|:--------------------|:------------------------|------------------------:|--------------------------:|-------------------:|:-----------------|
| earlyv1         | lr__l1          | early_late          | sequential_interactions |                0.56338  |                  0.746479 |          0.183099  | True             |
| earlyv0         | lr__l1          | early_late          | sequential_interactions |                0.56338  |                  0.71831  |          0.15493   | True             |
| earlyv1         | lr__l2          | early_late          | sequential_interactions |                0.690141 |                  0.774648 |          0.084507  | True             |
| earlyv0         | lr__l2          | early_late          | sequential_interactions |                0.690141 |                  0.774648 |          0.084507  | True             |
| earlyv0         | lr__elasticnet  | early_late          | sequential_interactions |                0.661972 |                  0.732394 |          0.0704225 | True             |
| earlyv1         | lr__elasticnet  | early_late          | sequential_interactions |                0.661972 |                  0.71831  |          0.056338  | True             |
| earlyv0         | svc__linear     | early_late          | sequential_interactions |                0.704225 |                  0.732394 |          0.028169  | True             |
| earlyv0         | lr__l1          | middle_late         | middle_late_scale       |                0.661972 |                  0.676056 |          0.0140845 | True             |
| earlyv1         | lr__l1          | middle_late         | middle_late_scale       |                0.661972 |                  0.676056 |          0.0140845 | True             |
| earlyv1         | lr__elasticnet  | middle_late         | middle_late_scale       |                0.676056 |                  0.690141 |          0.0140845 | True             |

## 9. Final recommendation
Use `--mode seed2026 --cv-mode exact` for primary model selection evidence, and `--mode stability --seeds 0-29 --cv-mode exact` for seed-level robustness. Treat `--cv-mode fast` as a structural/debug output only.