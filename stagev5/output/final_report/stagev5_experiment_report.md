# stagev5 experiment report

## Objective

Binary AD-versus-control classification using the stagev2 classifier panel, with severity/subgroup diagnostic outputs.

## Fixed feature provenance

- E: strict stagev2 early BM25 implementation.
- M: strict stagev2 BGE-M3 window embedding implementation.
- L: strict stagev4_unmasked P4/F8 expressive-form implementation.

## Feature-source audit

- L raw model features: 8
- L auxiliary diagnostic features: 0
- L auxiliary features used in model: false
- L interaction activation: raw_F8_mean

## Top external models

| model_name                          | group                        | feature_block           |   accuracy |   balanced_accuracy |   sensitivity |   specificity |       f1 |   roc_auc |   pr_auc |      mcc |   tn |   fp |   fn |   tp |
|:------------------------------------|:-----------------------------|:------------------------|-----------:|--------------------:|--------------:|--------------:|---------:|----------:|---------:|---------:|-----:|-----:|-----:|-----:|
| early_middle__svc__poly3            | two_stage_raw                | early_middle            |   0.84507  |            0.846429 |      0.942857 |      0.75     | 0.857143 |  0.855556 | 0.849592 | 0.704702 |   27 |    9 |    2 |   33 |
| early_middle_scale__svc__poly3      | sequential_scale             | early_middle_scale      |   0.830986 |            0.832143 |      0.914286 |      0.75     | 0.842105 |  0.848413 | 0.833528 | 0.672338 |   27 |    9 |    3 |   32 |
| all__svc__poly3                     | three_stage_raw              | all                     |   0.816901 |            0.818254 |      0.914286 |      0.722222 | 0.831169 |  0.851587 | 0.845819 | 0.647389 |   26 |   10 |    3 |   32 |
| middle_late_scale__svc__poly3       | sequential_scale             | middle_late_scale       |   0.802817 |            0.804365 |      0.914286 |      0.694444 | 0.820513 |  0.825397 | 0.823629 | 0.622726 |   25 |   11 |    3 |   32 |
| middle__svc__poly3                  | single_stage_middle_moderate | middle                  |   0.802817 |            0.803968 |      0.885714 |      0.722222 | 0.815789 |  0.846825 | 0.855089 | 0.615306 |   26 |   10 |    4 |   31 |
| all__svc__poly2                     | three_stage_raw              | all                     |   0.788732 |            0.790476 |      0.914286 |      0.666667 | 0.810127 |  0.842857 | 0.837472 | 0.598298 |   24 |   12 |    3 |   32 |
| early_middle__lr__l2                | two_stage_raw                | early_middle            |   0.788732 |            0.790476 |      0.914286 |      0.666667 | 0.810127 |  0.824603 | 0.822862 | 0.598298 |   24 |   12 |    3 |   32 |
| middle__svc__rbf                    | single_stage_middle_moderate | middle                  |   0.788732 |            0.790079 |      0.885714 |      0.694444 | 0.805195 |  0.847619 | 0.861811 | 0.590077 |   25 |   11 |    4 |   31 |
| middle_late__svc__poly3             | two_stage_raw                | middle_late             |   0.788732 |            0.790079 |      0.885714 |      0.694444 | 0.805195 |  0.843651 | 0.851695 | 0.590077 |   25 |   11 |    4 |   31 |
| all_plus_interactions__svc__poly2   | three_stage_full_interaction | all_plus_interactions   |   0.788732 |            0.790079 |      0.885714 |      0.694444 | 0.805195 |  0.815079 | 0.792431 | 0.590077 |   25 |   11 |    4 |   31 |
| mlp_svc_late_calibrated__lr__l2     | performance_corrected        | all                     |   0.774648 |            0.776587 |      0.914286 |      0.638889 | 0.8      |  0.838889 | 0.784523 | 0.574056 |   23 |   13 |    3 |   32 |
| middle__svc__poly2                  | single_stage_middle_moderate | middle                  |   0.774648 |            0.776587 |      0.914286 |      0.638889 | 0.8      |  0.838889 | 0.8439   | 0.574056 |   23 |   13 |    3 |   32 |
| early_middle_scale__svc__poly2      | sequential_scale             | early_middle_scale      |   0.774648 |            0.776587 |      0.914286 |      0.638889 | 0.8      |  0.838095 | 0.824834 | 0.574056 |   23 |   13 |    3 |   32 |
| middle_late__svc__poly2             | two_stage_raw                | middle_late             |   0.774648 |            0.776587 |      0.914286 |      0.638889 | 0.8      |  0.833333 | 0.836274 | 0.574056 |   23 |   13 |    3 |   32 |
| middle_late__svc__rbf               | two_stage_raw                | middle_late             |   0.774648 |            0.77619  |      0.885714 |      0.666667 | 0.794872 |  0.848413 | 0.862423 | 0.565081 |   24 |   12 |    4 |   31 |
| early_middle_scale__lr__l2          | sequential_scale             | early_middle_scale      |   0.774648 |            0.77619  |      0.885714 |      0.666667 | 0.794872 |  0.83254  | 0.83075  | 0.565081 |   24 |   12 |    4 |   31 |
| all_plus_interactions__svc__poly3   | three_stage_full_interaction | all_plus_interactions   |   0.774648 |            0.775794 |      0.857143 |      0.694444 | 0.789474 |  0.827778 | 0.801109 | 0.558273 |   25 |   11 |    5 |   30 |
| sequential_interactions__svc__poly3 | sequential_scale             | sequential_interactions |   0.774648 |            0.775794 |      0.857143 |      0.694444 | 0.789474 |  0.809524 | 0.781716 | 0.558273 |   25 |   11 |    5 |   30 |
| early__svc__poly2                   | single_stage_early_mild      | early                   |   0.774648 |            0.775794 |      0.857143 |      0.694444 | 0.789474 |  0.755556 | 0.703828 | 0.558273 |   25 |   11 |    5 |   30 |
| early_middle__svc__poly2            | two_stage_raw                | early_middle            |   0.760563 |            0.762698 |      0.914286 |      0.611111 | 0.790123 |  0.846032 | 0.841961 | 0.549951 |   22 |   14 |    3 |   32 |
| early_middle__svc__rbf              | two_stage_raw                | early_middle            |   0.760563 |            0.762302 |      0.885714 |      0.638889 | 0.78481  |  0.847619 | 0.85912  | 0.540266 |   23 |   13 |    4 |   31 |
| all__svc__rbf                       | three_stage_raw              | all                     |   0.760563 |            0.762302 |      0.885714 |      0.638889 | 0.78481  |  0.846825 | 0.857109 | 0.540266 |   23 |   13 |    4 |   31 |
| middle_late_scale__svc__rbf         | sequential_scale             | middle_late_scale       |   0.760563 |            0.762302 |      0.885714 |      0.638889 | 0.78481  |  0.846032 | 0.853202 | 0.540266 |   23 |   13 |    4 |   31 |
| stage_score_three_stage__lr__l2     | stage_score_interaction      | all                     |   0.760563 |            0.762302 |      0.885714 |      0.638889 | 0.78481  |  0.810317 | 0.82391  | 0.540266 |   23 |   13 |    4 |   31 |
| middle_late_scale__lr__l2           | sequential_scale             | middle_late_scale       |   0.760563 |            0.762302 |      0.885714 |      0.638889 | 0.78481  |  0.803175 | 0.811346 | 0.540266 |   23 |   13 |    4 |   31 |
| all_plus_interactions__svc__rbf     | three_stage_full_interaction | all_plus_interactions   |   0.760563 |            0.761905 |      0.857143 |      0.666667 | 0.779221 |  0.845238 | 0.855601 | 0.532764 |   24 |   12 |    5 |   30 |
| all__svc__sigmoid                   | three_stage_raw              | all                     |   0.760563 |            0.761905 |      0.857143 |      0.666667 | 0.779221 |  0.833333 | 0.83084  | 0.532764 |   24 |   12 |    5 |   30 |
| all__lr__l2                         | three_stage_raw              | all                     |   0.760563 |            0.761905 |      0.857143 |      0.666667 | 0.779221 |  0.830952 | 0.827342 | 0.532764 |   24 |   12 |    5 |   30 |
| all_plus_interactions__svc__sigmoid | three_stage_full_interaction | all_plus_interactions   |   0.760563 |            0.761905 |      0.857143 |      0.666667 | 0.779221 |  0.829365 | 0.818463 | 0.532764 |   24 |   12 |    5 |   30 |
| early_middle_scale__svc__sigmoid    | sequential_scale             | early_middle_scale      |   0.760563 |            0.761905 |      0.857143 |      0.666667 | 0.779221 |  0.828571 | 0.823475 | 0.532764 |   24 |   12 |    5 |   30 |

## Selected model stage/subgroup diagnostic

| model_name               | severity_group   |   n |   accuracy |   mean_p_ad |   false_negatives |   false_positives |
|:-------------------------|:-----------------|----:|-----------:|------------:|------------------:|------------------:|
| early_middle__svc__poly3 | AD_high_MMSE     |   8 |   1        |    0.825218 |                 0 |                 0 |
| early_middle__svc__poly3 | control          |  36 |   0.75     |    0.394966 |                 0 |                 9 |
| early_middle__svc__poly3 | early            |   6 |   0.833333 |    0.762936 |                 1 |                 0 |
| early_middle__svc__poly3 | late             |   7 |   1        |    0.930051 |                 0 |                 0 |
| early_middle__svc__poly3 | middle           |  14 |   0.928571 |    0.818436 |                 1 |                 0 |

## Interpretation boundary

Stagev5 reports severity-specific disease-decision performance. It does not present the binary classifier as a supervised four-class stage classifier.
