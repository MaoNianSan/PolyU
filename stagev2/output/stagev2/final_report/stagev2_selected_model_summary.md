# stagev2 selected model summary

- **best_external_accuracy_model**: early_middle__svc__poly3
- **best_mechanism_consistent_model**: mlp_svc_late_calibrated__lr__l2
- **final_recommended_model**: early_middle__svc__poly3
- **selection_metric**: external_accuracy
- **recommendation_rule**: Selected the model with highest held-out external accuracy; ties are resolved by balanced accuracy, sensitivity, specificity, F1, ROC-AUC, and PR-AUC.

## External validation metrics

| model_name               |   accuracy |   balanced_accuracy |   sensitivity |   specificity |       f1 |   roc_auc |   pr_auc |      mcc |   tn |   fp |   fn |   tp |
|:-------------------------|-----------:|--------------------:|--------------:|--------------:|---------:|----------:|---------:|---------:|-----:|-----:|-----:|-----:|
| early_middle__svc__poly3 |    0.84507 |            0.846429 |      0.942857 |          0.75 | 0.857143 |  0.856349 | 0.849876 | 0.704702 |   27 |    9 |    2 |   33 |
