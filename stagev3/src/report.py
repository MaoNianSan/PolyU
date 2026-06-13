from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def early_distribution_summary(early: dict, out_path: Path) -> pd.DataFrame:
    rows = []
    for variant, parts in early.items():
        for split, df in parts.items():
            label_col = "label" if "label" in df.columns else None
            features = [c for c in ["early_integrity_score", "early_omission_risk_score", "early_content_efficiency"] if c in df.columns]
            if not label_col:
                continue
            for label, g in df.groupby(label_col):
                for f in features:
                    x = pd.to_numeric(g[f], errors="coerce")
                    rows.append({
                        "early_variant": variant,
                        "split": split,
                        "label": int(label),
                        "feature": f,
                        "mean": float(x.mean()),
                        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                        "median": float(x.median()),
                        "q25": float(x.quantile(0.25)),
                        "q75": float(x.quantile(0.75)),
                        "min": float(x.min()),
                        "max": float(x.max()),
                    })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out


def write_summary_md(path: Path, manifest: dict, preprocess_summary: dict, feature_summary: dict, main: pd.DataFrame | None, stability_summary: pd.DataFrame | None, early_gain: pd.DataFrame | None, scale_seed: pd.DataFrame | None, scale_stability: pd.DataFrame | None, validation_status: str) -> None:
    lines = []
    lines.append("# stagev3 summary")
    lines.append("")
    lines.append("## 1. Run setting")
    for k in ["run_mode", "cv_mode", "main_seed", "stability_seeds", "n_model_specs_per_early_variant", "n_expected_main_rows", "n_expected_stability_rows", "raw_data_source", "api_mode"]:
        lines.append(f"- {k}: `{manifest.get(k)}`")
    lines.append(f"- validation_status: `{validation_status}`")
    lines.append("")
    lines.append("## 2. Data setting")
    lines.append(f"- train_n: {preprocess_summary.get('train_n')}")
    lines.append(f"- external_n: {preprocess_summary.get('external_n')}")
    lines.append(f"- train_label_counts: {preprocess_summary.get('train_label_counts')}")
    lines.append(f"- external_label_counts: {preprocess_summary.get('external_label_counts')}")
    lines.append("")
    lines.append("## 3. Feature extraction")
    lines.append("- early/middle/late features are extracted from normalized raw text when valid stagev3 feature CSVs are absent.")
    lines.append("- Once generated, valid feature CSVs are reused across seed2026/stability/all runs; use --force-features to re-extract.")
    lines.append("- Historical stage2 feature CSV outputs are not reused.")
    lines.append("- Huawei BGE-M3 and LLM API logic is rewritten using the stage2-compatible MAAS_API_KEY, endpoint, retry, and cache convention.")
    lines.append("- Real API or complete real-API cache is required by default; local surrogate feature generation is disabled.")
    lines.append(f"- feature_summary: `{json.dumps(feature_summary, ensure_ascii=False)[:1000]}`")
    lines.append("")
    if main is not None and not main.empty:
        lines.append("## 4. Main seed=2026 top results")
        top = main.sort_values(["external_accuracy", "external_f1", "external_auc"], ascending=False).head(10)
        lines.append(top[["early_variant", "model_spec_id", "external_accuracy", "external_f1", "external_auc", "external_accuracy_95ci"]].to_markdown(index=False))
        lines.append("")
    if stability_summary is not None and not stability_summary.empty:
        lines.append("## 5. Stability top results")
        top = stability_summary.head(10)
        lines.append(top[["early_variant", "model_spec_id", "n_seeds", "external_accuracy_mean", "external_accuracy_95ci", "external_f1_mean", "external_auc_mean"]].to_markdown(index=False))
        lines.append("")
    if early_gain is not None and not early_gain.empty:
        lines.append("## 6. earlyv1 vs earlyv0")
        lines.append(early_gain.sort_values("external_accuracy_delta", ascending=False).head(10)[["model_spec_id", "earlyv0_external_accuracy", "earlyv1_external_accuracy", "external_accuracy_delta", "earlyv1_has_gain"]].to_markdown(index=False))
        lines.append("")
    if scale_seed is not None and not scale_seed.empty:
        lines.append("## 7. Scale gain under seed=2026")
        lines.append(scale_seed.sort_values("scale_gain_delta", ascending=False).head(10).to_markdown(index=False))
        lines.append("")
    if scale_stability is not None and not scale_stability.empty:
        lines.append("## 8. Scale gain under stability seeds")
        lines.append(scale_stability.sort_values("scale_gain_mean_delta", ascending=False).head(10).to_markdown(index=False))
        lines.append("")
    lines.append("## 9. Final recommendation")
    lines.append("Use `--mode seed2026 --cv-mode exact` for primary model selection evidence, and `--mode stability --seeds 0-29 --cv-mode exact` for seed-level robustness. Treat `--cv-mode fast` as a structural/debug output only.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_notebook(path: Path) -> None:
    cell_index = 0

    def markdown(text: str) -> dict:
        nonlocal cell_index
        cell_index += 1
        return {
            "cell_type": "markdown",
            "id": f"stagev3-md-{cell_index:02d}",
            "metadata": {},
            "source": [text.rstrip() + "\n"],
        }

    def code(text: str) -> dict:
        nonlocal cell_index
        cell_index += 1
        return {
            "cell_type": "code",
            "id": f"stagev3-code-{cell_index:02d}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [text.rstrip() + "\n"],
        }

    run_command = (
        "python run_stagev3.py --mode selected_after_seed2026 --seeds 0-29 "
        "--cv-mode exact --min-external-accuracy 0.75 --n-jobs 8 --resume"
    )
    nb = {
        "cells": [
            markdown("# stagev3 selected stability result check"),
            markdown("## Section 0. File audit and run status"),
            code(
                f"""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path("..")
FINAL = ROOT / "output" / "final_report"
THRESHOLD = 0.75
RUN_COMMAND = "{run_command}"

def read_csv_safe(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: cannot read {{path}}: {{exc}}")
        return pd.DataFrame()

def read_json_safe(path):
    if not path.exists():
        return {{}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: cannot read {{path}}: {{exc}}")
        return {{}}

def available(frame, columns):
    return [column for column in columns if column in frame.columns]

paths = {{
    "seed2026_main": FINAL / "seed2026_main_results.csv",
    "selected_models": FINAL / "selected_model_specs_external075.csv",
    "selected_models_md": FINAL / "selected_model_specs_external075.md",
    "selected_results": FINAL / "selected_seed_stability_results_external075.csv",
    "selected_summary": FINAL / "selected_seed_stability_summary_external075.csv",
    "selected_summary_md": FINAL / "selected_seed_stability_summary_external075.md",
    "selected_manifest": FINAL / "selected_run_manifest_external075.json",
    "predictions": FINAL / "seed2026_external_predictions_all_models.csv",
    "run_progress": FINAL / "run_progress.json",
    "validation_report": FINAL / "validation_report.md",
}}

main = read_csv_safe(paths["seed2026_main"])
selected = read_csv_safe(paths["selected_models"])
results = read_csv_safe(paths["selected_results"])
summary = read_csv_safe(paths["selected_summary"])
predictions = read_csv_safe(paths["predictions"])
manifest = read_json_safe(paths["selected_manifest"])
run_progress = read_json_safe(paths["run_progress"])

audit_rows = []
for name, file_path in paths.items():
    row = {{"name": name, "path": str(file_path), "exists": file_path.exists()}}
    if file_path.suffix.lower() == ".csv":
        frame = {{"seed2026_main": main, "selected_models": selected, "selected_results": results,
                  "selected_summary": summary, "predictions": predictions}}.get(name, pd.DataFrame())
        row["rows"] = len(frame) if file_path.exists() else np.nan
    audit_rows.append(row)
display(pd.DataFrame(audit_rows))

old_079 = sorted(FINAL.glob("*external079*"))
if old_079:
    print("NOTE: old external079 files detected; they are not used by this notebook:")
    for old_path in old_079:
        print(" -", old_path.name)

selected_count = int(manifest.get("selected_model_count", len(selected)))
n_seeds = int(manifest.get("n_seeds", results["seed"].nunique() if "seed" in results else 0))
expected_rows = selected_count * n_seeds
stable_candidate_count = int(
    (pd.to_numeric(summary["external_accuracy_mean"], errors="coerce") >= THRESHOLD).sum()
) if "external_accuracy_mean" in summary else 0
status = pd.DataFrame([{{
    "selected_model_count": selected_count,
    "stable_candidate_count": stable_candidate_count,
    "n_seeds": n_seeds,
    "actual_selected_stability_rows": len(results),
    "expected_selected_stability_rows": expected_rows,
    "row_count_matches": len(results) == expected_rows if expected_rows else False,
    "completed_seed_count": manifest.get("completed_seed_count"),
    "missing_seed_ids": manifest.get("missing_seed_ids"),
    "n_jobs": manifest.get("n_jobs"),
}}])
display(status)
if summary.empty or results.empty:
    display(Markdown("**WARNING:** selected stability outputs are missing or empty. Run:"))
    print(RUN_COMMAND)"""
            ),
            markdown("## Section 1. Seed2026 all-model ranking"),
            code(
                """if main.empty:
    print("WARNING: seed2026_main_results.csv is unavailable.")
else:
    main_view = main.copy()
    if "classifier_name" not in main_view:
        main_view["classifier_name"] = main_view.get("model_name", main_view.get("model_variant", ""))
    if "model_family" not in main_view:
        main_view["model_family"] = main_view.get("model_variant", main_view["classifier_name"]).astype(str).str.split("__").str[0]
    sort_cols = available(main_view, ["external_accuracy", "external_f1", "external_auc"])
    ranked_main = main_view.sort_values(sort_cols, ascending=False) if sort_cols else main_view
    main_columns = available(ranked_main, [
        "early_variant", "model_spec_id", "feature_block", "classifier_name",
        "model_family", "external_accuracy", "external_precision",
        "external_recall", "external_f1", "external_auc", "n_features",
    ])
    display(ranked_main[main_columns].head(20))
    if {"external_accuracy", "model_spec_id"} <= set(ranked_main.columns):
        plot_data = ranked_main.head(20).copy().iloc[::-1]
        labels = plot_data.get("early_variant", "").astype(str) + " | " + plot_data["model_spec_id"].astype(str)
        plt.figure(figsize=(10, 8))
        plt.barh(labels, plot_data["external_accuracy"])
        plt.xlabel("External accuracy")
        plt.title("Top 20 seed2026 models by external accuracy")
        plt.tight_layout()
        plt.show()"""
            ),
            markdown("## Section 2. Selected model audit"),
            code(
                """if selected.empty:
    print("WARNING: selected_model_specs_external075.csv is unavailable.")
else:
    threshold_values = selected["selection_threshold"].dropna().unique().tolist() if "selection_threshold" in selected else []
    print("selection_threshold:", threshold_values)
    print("selected_model_count:", len(selected))
    print("selected proportion:", len(selected) / len(main) if len(main) else np.nan)
    for column in ["early_variant", "feature_block", "classifier_name"]:
        if column in selected:
            display(selected[column].value_counts(dropna=False).rename("count").to_frame())
    if "external_accuracy" in selected:
        invalid_selected = selected[pd.to_numeric(selected["external_accuracy"], errors="coerce") < THRESHOLD]
        if invalid_selected.empty:
            print("PASS: all selected models satisfy external_accuracy >= 0.75")
        else:
            print("WARNING: selected models below threshold:")
            display(invalid_selected)"""
            ),
            markdown("## Section 3. Selected stability summary ranking"),
            code(
                """if summary.empty:
    print("WARNING: selected_seed_stability_summary_external075.csv is unavailable.")
    ranked_summary = pd.DataFrame()
    filtered_summary = pd.DataFrame()
else:
    filtered_summary = summary[
        pd.to_numeric(summary["external_accuracy_mean"], errors="coerce") >= THRESHOLD
    ].copy() if "external_accuracy_mean" in summary else pd.DataFrame()
    ranking = available(filtered_summary, [
        "external_accuracy_mean", "external_f1_mean", "external_auc_mean",
    ])
    ranked_summary = filtered_summary.sort_values(ranking, ascending=False) if ranking else filtered_summary.copy()
    print("selected_model_count:", len(selected))
    print("stable_candidate_count:", len(filtered_summary))
    if filtered_summary.empty:
        print("WARNING: no models satisfy multi-seed external_accuracy_mean >= 0.75.")
    summary_columns = available(ranked_summary, [
        "early_variant", "model_spec_id", "feature_block", "classifier_name",
        "model_family", "n_seeds", "seed2026_external_accuracy",
        "external_accuracy_mean", "external_accuracy_std",
        "external_accuracy_min", "external_accuracy_max",
        "external_precision_mean", "external_recall_mean",
        "external_f1_mean", "external_auc_mean",
    ])
    display(ranked_summary[summary_columns].head(30))"""
            ),
            markdown("## Section 4. External accuracy mean plot"),
            code(
                """mean_required = {"external_accuracy_mean"}
if filtered_summary.empty:
    print("WARNING: no models satisfy external_accuracy_mean >= 0.75; no mean plot is shown.")
elif not mean_required <= set(filtered_summary.columns):
    print("WARNING: external_accuracy_mean is unavailable.")
else:
    mean_ranking = available(filtered_summary, [
        "external_accuracy_mean", "external_f1_mean", "external_auc_mean",
    ])
    mean_plot = filtered_summary.sort_values(mean_ranking, ascending=False).dropna(
        subset=list(mean_required)
    ).head(30).copy().iloc[::-1]
    if mean_plot.empty:
        print("WARNING: no complete mean accuracy rows are available.")
    else:
        labels = (
            mean_plot.get("early_variant", "").astype(str) + " | " +
            mean_plot.get("feature_block", "").astype(str) + " | " +
            mean_plot.get("classifier_name", mean_plot.get("model_spec_id", "")).astype(str)
        )
        plt.figure(figsize=(11, 10))
        plt.barh(labels, mean_plot["external_accuracy_mean"])
        plt.axvline(THRESHOLD, linestyle="--")
        plt.xlabel("Mean external accuracy")
        plt.title("Selected models mean external accuracy")
        plt.tight_layout()
        plt.show()"""
            ),
            markdown("## Section 5. Seed2026 accuracy vs multi-seed stability mean"),
            code(
                """gap_required = {"seed2026_external_accuracy", "external_accuracy_mean"}
if summary.empty or not gap_required <= set(summary.columns):
    print("WARNING: seed2026/stability comparison columns are unavailable.")
else:
    gap_table = summary.copy()
    gap_table["stability_gap"] = gap_table["external_accuracy_mean"] - gap_table["seed2026_external_accuracy"]
    gap_columns = available(gap_table, [
        "early_variant", "model_spec_id", "seed2026_external_accuracy",
        "external_accuracy_mean", "stability_gap",
    ])
    display(gap_table.sort_values("stability_gap")[gap_columns])
    plot_gap = gap_table.dropna(subset=list(gap_required))
    if not plot_gap.empty:
        low = min(plot_gap["seed2026_external_accuracy"].min(), plot_gap["external_accuracy_mean"].min())
        high = max(plot_gap["seed2026_external_accuracy"].max(), plot_gap["external_accuracy_mean"].max())
        plt.figure(figsize=(8, 7))
        plt.scatter(plot_gap["seed2026_external_accuracy"], plot_gap["external_accuracy_mean"])
        plt.plot([low, high], [low, high], linestyle="--")
        plt.xlabel("Seed2026 external accuracy")
        plt.ylabel("Multi-seed mean external accuracy")
        plt.title("Seed2026 external accuracy vs multi-seed mean accuracy")
        plt.tight_layout()
        plt.show()"""
            ),
            markdown("## Section 6. Across-seed distribution"),
            code(
                """if results.empty or "seed" not in results or "external_accuracy" not in results:
    print("WARNING: selected per-seed results are unavailable.")
else:
    seed_distribution = results.groupby("seed")["external_accuracy"].agg(
        mean_external_accuracy="mean",
        max_external_accuracy="max",
        min_external_accuracy="min",
        selected_model_runs="size",
    ).reset_index()
    display(seed_distribution)
    if not ranked_summary.empty and "model_spec_id" in ranked_summary:
        top_keys = ranked_summary.head(10)[available(ranked_summary, ["early_variant", "model_spec_id"])]
        top_results = results.merge(top_keys, on=available(top_keys, ["early_variant", "model_spec_id"]), how="inner")
        groups, labels = [], []
        for key, group in top_results.groupby(available(top_results, ["early_variant", "model_spec_id"]), sort=False):
            groups.append(group["external_accuracy"].dropna().to_numpy())
            labels.append(" | ".join(map(str, key if isinstance(key, tuple) else (key,))))
        if groups:
            plt.figure(figsize=(11, 7))
            plt.boxplot(groups, labels=labels, vert=False)
            plt.xlabel("External accuracy across seeds")
            plt.title("Top 10 selected models across-seed accuracy")
            plt.tight_layout()
            plt.show()"""
            ),
            markdown("## Section 7. Stability labels"),
            code(
                """if summary.empty or "external_accuracy_mean" not in summary.columns:
    print("WARNING: stability mean is unavailable.")
    robust_summary = pd.DataFrame()
else:
    robust_summary = summary.copy()
    robust_summary["stability_label"] = np.where(
        pd.to_numeric(robust_summary["external_accuracy_mean"], errors="coerce") >= THRESHOLD,
        "stable_candidate",
        "below_threshold",
    )
    display(robust_summary["stability_label"].value_counts().rename("count").to_frame())
    stable_robustness = robust_summary[
        pd.to_numeric(robust_summary["external_accuracy_mean"], errors="coerce") >= THRESHOLD
    ].copy()
    print("Stable recommendation candidates (external_accuracy_mean >= 0.75):", len(stable_robustness))
    robust_columns = available(robust_summary, [
        "stability_label", "early_variant", "model_spec_id", "feature_block",
        "classifier_name", "external_accuracy_mean",
        "external_recall_mean", "external_auc_mean",
    ])
    for label in ["stable_candidate", "below_threshold"]:
        display(Markdown(f"### {label}"))
        display(robust_summary.loc[robust_summary["stability_label"] == label, robust_columns])"""
            ),
            markdown("## Section 8. Best stable model prediction analysis"),
            code(
                """best_model = pd.Series(dtype=object)
best_predictions = pd.DataFrame()
if ranked_summary.empty:
    print("WARNING: no stable model ranking is available.")
elif predictions.empty:
    print("WARNING: seed2026_external_predictions_all_models.csv is unavailable.")
else:
    best_model = ranked_summary.iloc[0]
    if {"early_variant", "model_spec_id"} <= set(predictions.columns):
        best_predictions = predictions[
            (predictions["early_variant"].astype(str) == str(best_model.get("early_variant"))) &
            (predictions["model_spec_id"].astype(str) == str(best_model.get("model_spec_id")))
        ].copy()
    else:
        print("WARNING: prediction model_spec_id missing; using early_variant + feature_block + classifier fallback.")
        classifier_column = "classifier_name" if "classifier_name" in predictions else "model_name"
        required_fallback = {"early_variant", "feature_block", classifier_column}
        if required_fallback <= set(predictions.columns):
            best_predictions = predictions[
                (predictions["early_variant"].astype(str) == str(best_model.get("early_variant"))) &
                (predictions["feature_block"].astype(str) == str(best_model.get("feature_block"))) &
                (predictions[classifier_column].astype(str) == str(best_model.get("classifier_name")))
            ].copy()
    print("Selected stable model:", best_model.get("early_variant"), best_model.get("model_spec_id"))
    if best_predictions.empty:
        print("WARNING: no prediction rows matched the selected stable model.")
    elif {"y_true", "y_pred"} <= set(best_predictions.columns):
        display(pd.crosstab(best_predictions["y_true"], best_predictions["y_pred"], rownames=["Actual"], colnames=["Predicted"], dropna=False))
        score_column = "p_ad" if "p_ad" in best_predictions else ("y_score" if "y_score" in best_predictions else None)
        if score_column:
            plt.figure(figsize=(8, 5))
            plt.hist(best_predictions[score_column].dropna(), bins=15)
            plt.xlabel("Predicted AD probability")
            plt.ylabel("Count")
            plt.title("Best stable model predicted AD probability")
            plt.tight_layout()
            plt.show()
        if "correct" not in best_predictions:
            best_predictions["correct"] = (best_predictions["y_true"] == best_predictions["y_pred"]).astype(int)
        if "error_type" not in best_predictions:
            best_predictions["error_type"] = np.select(
                [
                    (best_predictions["y_true"] == 1) & (best_predictions["y_pred"] == 0),
                    (best_predictions["y_true"] == 0) & (best_predictions["y_pred"] == 1),
                ],
                ["false_negative", "false_positive"],
                default="correct",
            )
        if "severity_group" not in best_predictions:
            best_predictions["severity_group"] = np.nan
        error_columns = available(best_predictions, [
            "sample_id", "y_true", "y_pred", "p_ad", "y_score", "correct",
            "mmse", "severity_group", "error_type",
        ])
        display(best_predictions.loc[best_predictions["correct"] == 0, error_columns])"""
            ),
            markdown("## Section 9. Severity / MMSE subgroup analysis"),
            code(
                """if best_predictions.empty:
    print("WARNING: no matched best-model predictions are available.")
elif not {"severity_group", "mmse"} <= set(best_predictions.columns) or best_predictions["severity_group"].isna().all():
    print("WARNING: severity_group and mmse are not both available for subgroup analysis.")
else:
    score_column = "p_ad" if "p_ad" in best_predictions else ("y_score" if "y_score" in best_predictions else None)
    subgroup_rows = []
    for severity, group in best_predictions.groupby("severity_group", dropna=False):
        subgroup_rows.append({
            "severity_group": severity,
            "n": len(group),
            "accuracy": float((group["y_true"] == group["y_pred"]).mean()),
            "mean_p_ad": float(group[score_column].mean()) if score_column else np.nan,
            "fn_count": int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum()),
            "fp_count": int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum()),
            "mean_mmse": float(pd.to_numeric(group["mmse"], errors="coerce").mean()),
        })
    subgroup = pd.DataFrame(subgroup_rows)
    display(subgroup)
    if not subgroup.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(subgroup["severity_group"].astype(str), subgroup["accuracy"])
        plt.ylabel("Accuracy")
        plt.title("Best stable model accuracy by severity group")
        plt.tight_layout()
        plt.show()"""
            ),
            markdown("## Section 10. Final recommended models"),
            code(
                """if robust_summary.empty:
    print("WARNING: no selected stability summary is available for recommendations.")
else:
    candidates = robust_summary[
        pd.to_numeric(robust_summary["external_accuracy_mean"], errors="coerce") >= THRESHOLD
    ].copy()
    if candidates.empty:
        print("WARNING: no models satisfy multi-seed external_accuracy_mean >= 0.75.")
    order = available(candidates, [
        "external_accuracy_mean", "external_recall_mean", "external_auc_mean",
    ])
    recommended = candidates.sort_values(order, ascending=False).head(10) if order else candidates.head(10)
    recommendation_columns = available(recommended, [
        "stability_label", "early_variant", "model_spec_id", "feature_block",
        "classifier_name", "n_seeds", "seed2026_external_accuracy",
        "external_accuracy_mean", "external_recall_mean",
        "external_f1_mean", "external_auc_mean",
    ])
    display(recommended[recommendation_columns])

display(Markdown(
    "**Selection is not stability:** selected_model_count uses seed2026 external accuracy; "
    "stable_candidate_count requires multi-seed external_accuracy_mean >= 0.75."
))"""
            ),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
