# stagev3 summary

## 1. Run setting
- run_mode: `selected_after_seed2026`
- cv_mode: `fast`
- main_seed: `2026`
- stability_seeds: `[0, 1]`
- n_model_specs_per_early_variant: `102`
- n_expected_main_rows: `204`
- n_expected_stability_rows: `408`
- raw_data_source: `D:\research\H.L.Liang-Lab\Code\expore\stagev3_fixed\input\raw`
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
- feature_summary: `{"early": {"earlyv0": "completed", "earlyv1": "completed"}, "middle": {"feature_family": "middle_huawei_bge_m3_window_embedding", "feature_schema": "v2_full_mean", "middle_keep_dims": 1024, "middle_include_v3_stats": false, "regenerated_from_raw_text": true, "historical_feature_outputs_reused": false, "api_logic_source": "stagev2_cache_first_logic_v3_wrapper", "windowing_source": "stagev2_regex_word_windows", "api_or_cache_mode": "api", "feature_load_mode": "existing_feature_csv", "feature_extracted_this_run": false, "local_surrogate_allowed": false, "cache_path": "D:\\research\\H.L.Liang-Lab\\Code\\expore\\stagev3_fixed\\output\\cache\\huawei_bge_m3_embedding_cache.csv", "new_cache_rows": 4839, "unique_texts": 4839, "ignored_surrogate_cache_rows": 0, "accepted_cache_sources": ["api", "huawei_maas_api", "huawei_maas_api_v2"], "safety_masked_api_calls": 19, "safety_level_counts": {"safety_mask": 9, "aggressive_mask": 8, "neutral_length": 2}, "stagev2_cache_compatible": true, "n_train_ro`

## 4. Main seed=2026 top results
| early_variant   | model_spec_id                  |   external_accuracy |   external_f1 |   external_auc |   external_accuracy_95ci |
|:----------------|:-------------------------------|--------------------:|--------------:|---------------:|-------------------------:|
| earlyv0         | early_middle__svc__poly3       |            0.84507  |      0.857143 |       0.855556 |                      nan |
| earlyv1         | all__svc__poly3                |            0.830986 |      0.846154 |       0.851587 |                      nan |
| earlyv0         | all__svc__poly3                |            0.830986 |      0.846154 |       0.85     |                      nan |
| earlyv1         | early_middle__svc__poly3       |            0.830986 |      0.842105 |       0.856349 |                      nan |
| earlyv0         | early_middle_scale__svc__poly3 |            0.802817 |      0.820513 |       0.853968 |                      nan |
| earlyv1         | early_middle_scale__svc__poly3 |            0.802817 |      0.820513 |       0.853175 |                      nan |
| earlyv0         | middle_only__svc__poly3        |            0.802817 |      0.815789 |       0.846825 |                      nan |
| earlyv1         | middle_only__svc__poly3        |            0.802817 |      0.815789 |       0.846825 |                      nan |
| earlyv0         | middle_late__svc__poly3        |            0.788732 |      0.805195 |       0.838889 |                      nan |
| earlyv1         | middle_late__svc__poly3        |            0.788732 |      0.805195 |       0.838889 |                      nan |

## 6. earlyv1 vs earlyv0
| model_spec_id                          |   earlyv0_external_accuracy |   earlyv1_external_accuracy |   external_accuracy_delta | earlyv1_has_gain   |
|:---------------------------------------|----------------------------:|----------------------------:|--------------------------:|:-------------------|
| mlp_svc_late_calibrated__lr__l2        |                    0.746479 |                    0.774648 |                 0.028169  | True               |
| stage_activation_summary__svc__sigmoid |                    0.577465 |                    0.605634 |                 0.028169  | True               |
| all_plus_interactions__svc__rbf        |                    0.676056 |                    0.690141 |                 0.0140845 | True               |
| early_late__svc__linear                |                    0.661972 |                    0.676056 |                 0.0140845 | True               |
| early_middle_scale__svc__linear        |                    0.71831  |                    0.732394 |                 0.0140845 | True               |
| early_only__svc__rbf                   |                    0.690141 |                    0.704225 |                 0.0140845 | True               |
| stage_activation_summary__svc__poly2   |                    0.605634 |                    0.619718 |                 0.0140845 | True               |
| sequential_interactions__svc__rbf      |                    0.71831  |                    0.732394 |                 0.0140845 | True               |
| early_middle_scale__lr__l1             |                    0.746479 |                    0.746479 |                 0         | False              |
| late_only__svc__rbf                    |                    0.591549 |                    0.591549 |                 0         | False              |

## 7. Scale gain under seed=2026
| early_variant   | model_variant   | raw_feature_block   | scale_feature_block     |   raw_external_accuracy |   scale_external_accuracy |   scale_gain_delta | scale_has_gain   |
|:----------------|:----------------|:--------------------|:------------------------|------------------------:|--------------------------:|-------------------:|:-----------------|
| earlyv0         | lr__l1          | early_late          | sequential_interactions |                0.591549 |                  0.732394 |          0.140845  | True             |
| earlyv1         | lr__l1          | early_late          | sequential_interactions |                0.591549 |                  0.732394 |          0.140845  | True             |
| earlyv1         | lr__elasticnet  | early_late          | sequential_interactions |                0.647887 |                  0.71831  |          0.0704225 | True             |
| earlyv0         | lr__elasticnet  | early_late          | sequential_interactions |                0.647887 |                  0.71831  |          0.0704225 | True             |
| earlyv0         | lr__l2          | early_late          | sequential_interactions |                0.661972 |                  0.71831  |          0.056338  | True             |
| earlyv1         | lr__l2          | early_late          | sequential_interactions |                0.661972 |                  0.71831  |          0.056338  | True             |
| earlyv1         | svc__sigmoid    | middle_late         | middle_late_scale       |                0.690141 |                  0.71831  |          0.028169  | True             |
| earlyv0         | svc__sigmoid    | middle_late         | middle_late_scale       |                0.690141 |                  0.71831  |          0.028169  | True             |
| earlyv1         | svc__rbf        | early_late          | sequential_interactions |                0.704225 |                  0.732394 |          0.028169  | True             |
| earlyv0         | svc__rbf        | early_late          | sequential_interactions |                0.704225 |                  0.71831  |          0.0140845 | True             |

## 9. Final recommendation
Use `--mode seed2026 --cv-mode exact` for primary model selection evidence, and `--mode stability --seeds 0-29 --cv-mode exact` for seed-level robustness. Treat `--cv-mode fast` as a structural/debug output only.