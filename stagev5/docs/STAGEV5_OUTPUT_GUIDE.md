# Stagev5 Output Guide

The primary GitHub-facing result directory is `output/final_report/`.

## Core Tables

- `stagev5_model_ranking_by_external_accuracy.csv`: primary ranking table by held-out external accuracy.
- `stagev5_external_performance_report.csv`: external-set metrics for completed models.
- `stagev5_cv_summary.csv`: repeated 10-fold CV summaries.
- `stagev5_bootstrap_ci.csv`: bootstrap confidence intervals using bootstrap `n=200`.
- `stagev5_generalization_gap.csv`: CV-to-external comparison.
- `stagev5_stage_subgroup_accuracy.csv`: severity/stage subgroup diagnostic accuracy.
- `stagev5_error_analysis.csv`: selected-model external errors and hard cases.

## Reports And Manifests

- `stagev5_selected_model_summary.md`: selected model, protocol, and subgroup summary.
- `stagev5_experiment_report.md`: concise experiment report.
- `stagev5_feature_source_manifest.json`: E/M/L provenance and feature-count audit.
- `stagev5_leakage_check.json`: leakage-control check output.
- `stagev5_final_run_summary.json`: compact selected-model summary.

## Figures

The `figures/` subdirectory contains PNGs for model ranking, feature-block comparison, confusion matrix, CV/external gap, subgroup accuracy, error distribution, and bootstrap CI.

## GitHub Recommendation

Commit `output/final_report/*.md`, `output/final_report/*.csv`, `output/final_report/*.json`, and `output/final_report/figures/*.png`. Avoid committing local caches, temporary folders, virtual environments, logs, or model pickle directories unless a release specifically requires them.
