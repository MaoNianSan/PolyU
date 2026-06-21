# stagev5 selected model summary

## Selection protocol

- Feature sources: E/M from stagev2; L from stagev4_unmasked P4/F8.
- L raw model features: 8
- L auxiliary diagnostic features: 0
- L auxiliary features used in model: false
- L activation for interaction blocks: raw_F8_mean
- Training: stagev2 classifier panel, GridSearchCV, repeated stratified 10-fold CV (10×1).
- Ranking: held-out external accuracy. The external set was not used for fit, scaling, imputation, or BM25 fitting.

## Selected model

| model_name               | group         | feature_block   |   accuracy |   balanced_accuracy |   sensitivity |   specificity |       f1 |   roc_auc |   pr_auc |      mcc |   tn |   fp |   fn |   tp |
|:-------------------------|:--------------|:----------------|-----------:|--------------------:|--------------:|--------------:|---------:|----------:|---------:|---------:|-----:|-----:|-----:|-----:|
| early_middle__svc__poly3 | two_stage_raw | early_middle    |    0.84507 |            0.846429 |      0.942857 |          0.75 | 0.857143 |  0.855556 | 0.849592 | 0.704702 |   27 |    9 |    2 |   33 |

## Stage/subgroup diagnostic

| model_name               | severity_group   |   n |   accuracy |   mean_p_ad |   false_negatives |   false_positives |
|:-------------------------|:-----------------|----:|-----------:|------------:|------------------:|------------------:|
| early_middle__svc__poly3 | AD_high_MMSE     |   8 |   1        |    0.825218 |                 0 |                 0 |
| early_middle__svc__poly3 | control          |  36 |   0.75     |    0.394966 |                 0 |                 9 |
| early_middle__svc__poly3 | early            |   6 |   0.833333 |    0.762936 |                 1 |                 0 |
| early_middle__svc__poly3 | late             |   7 |   1        |    0.930051 |                 0 |                 0 |
| early_middle__svc__poly3 | middle           |  14 |   0.928571 |    0.818436 |                 1 |                 0 |
