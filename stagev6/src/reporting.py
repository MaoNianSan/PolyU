from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg


def copy_final(src: Path, name: str) -> None:
    cfg.FINAL.mkdir(parents=True, exist_ok=True)
    target = cfg.FINAL / name
    shutil.copy2(src, target)


def save_figures(ranking: pd.DataFrame, gap: pd.DataFrame, predictions: pd.DataFrame, subgroup: pd.DataFrame, boot: pd.DataFrame) -> list[str]:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    if not ranking.empty:
        top = ranking.iloc[::-1].copy()
        plt.figure(figsize=(12, max(4, 0.5 * len(top))))
        plt.barh(top["model_name"], top["accuracy"])
        plt.xlabel("External accuracy")
        plt.tight_layout()
        p = cfg.FIGURES / "fig01_model_ranking_external_accuracy.png"; plt.savefig(p, dpi=220); plt.close(); made.append(p.name)
        arch = ranking.groupby("gate_id", as_index=False)["accuracy"].max().sort_values("accuracy", ascending=False)
        plt.figure(figsize=(9, 4.5))
        plt.bar(arch["gate_id"], arch["accuracy"])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Best external accuracy across branches")
        plt.tight_layout()
        p = cfg.FIGURES / "fig02_late_gate_architecture_comparison.png"; plt.savefig(p, dpi=220); plt.close(); made.append(p.name)
    if not predictions.empty and not ranking.empty:
        selected = str(ranking.iloc[0]["model_name"])
        sel = predictions.loc[predictions.model_name.eq(selected)]
        cm = pd.crosstab(sel.y_true, sel.y_pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm.to_numpy())
        plt.xticks([0, 1], ["Pred control", "Pred AD"]); plt.yticks([0, 1], ["True control", "True AD"])
        for i in range(2):
            for j in range(2): plt.text(j, i, str(int(cm.iloc[i, j])), ha="center", va="center")
        plt.title("Selected cascade: external confusion matrix")
        plt.tight_layout()
        p = cfg.FIGURES / "fig03_selected_model_confusion_matrix.png"; plt.savefig(p, dpi=220); plt.close(); made.append(p.name)
    if not gap.empty:
        plt.figure(figsize=(7.5, 5))
        plt.scatter(gap["cv_accuracy"], gap["external_accuracy"])
        lo = float(np.nanmin(np.r_[gap["cv_accuracy"], gap["external_accuracy"]])); hi = float(np.nanmax(np.r_[gap["cv_accuracy"], gap["external_accuracy"]]))
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel("10-fold cascade OOF accuracy")
        plt.ylabel("External accuracy")
        plt.tight_layout()
        p = cfg.FIGURES / "fig04_cv_external_gap.png"; plt.savefig(p, dpi=220); plt.close(); made.append(p.name)
    if not subgroup.empty and not ranking.empty:
        selected = str(ranking.iloc[0]["model_name"])
        sub = subgroup.loc[subgroup.model_name.eq(selected)]
        plt.figure(figsize=(8, 4.5))
        plt.bar(sub["severity_group"], sub["accuracy"])
        plt.ylim(0, 1); plt.ylabel("External disease-decision accuracy")
        plt.tight_layout()
        p = cfg.FIGURES / "fig05_stage_subgroup_accuracy.png"; plt.savefig(p, dpi=220); plt.close(); made.append(p.name)
    if not predictions.empty and not ranking.empty:
        selected = str(ranking.iloc[0]["model_name"])
        counts = predictions.loc[predictions.model_name.eq(selected), "route_error_type"].value_counts()
        plt.figure(figsize=(10, 4.5))
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=25, ha="right"); plt.ylabel("External samples")
        plt.tight_layout()
        p = cfg.FIGURES / "fig06_route_type_distribution.png"; plt.savefig(p, dpi=220); plt.close(); made.append(p.name)
    if not boot.empty and not ranking.empty:
        selected = str(ranking.iloc[0]["model_name"])
        row = boot.loc[(boot.model_name.eq(selected)) & (boot.metric.eq("accuracy"))]
        if not row.empty:
            r = row.iloc[0]
            plt.figure(figsize=(6, 2.4))
            plt.errorbar([0], [r.bootstrap_mean], yerr=[[r.bootstrap_mean-r.ci_low], [r.ci_high-r.bootstrap_mean]], fmt="o")
            plt.xlim(-1, 1); plt.xticks([]); plt.ylabel("External accuracy")
            plt.tight_layout()
            p = cfg.FIGURES / "fig07_bootstrap_external_accuracy_ci.png"; plt.savefig(p, dpi=220); plt.close(); made.append(p.name)
    return made


def write_markdown_reports(
    ranking: pd.DataFrame,
    component_specs: pd.DataFrame,
    route: pd.DataFrame,
    feature_manifest: dict[str, Any],
    figures: list[str],
) -> None:
    selected = ranking.iloc[0]
    selection = [
        "# stagev6 selected model summary", "",
        "## Selected cascade", "",
        f"- Model: `{selected['model_name']}`",
        f"- Late gate: `{selected['gate_id']}`",
        f"- Non-late AD/control branch: `{selected['branch_id']}`",
        f"- External accuracy: {float(selected['accuracy']):.6f}",
        f"- External balanced accuracy: {float(selected['balanced_accuracy']):.6f}",
        f"- External sensitivity: {float(selected['sensitivity']):.6f}",
        f"- External specificity: {float(selected['specificity']):.6f}", "",
        "## Fixed decision rule", "",
        "A sample is directly assigned AD when the late gate probability is at least 0.50. Otherwise, the E+M non-late branch determines AD versus control at 0.50.",
        "The continuous `p_ad_mixture = p_late + (1-p_late)*p_ad_given_nonlate` is stored for ranking diagnostics and probability metrics, but it is not used to override hard routing.", "",
        "## Source and evaluation constraints", "",
        "- E and M are inherited from the stagev5 stagev2-compatible files; M is mean-aggregated by sample_id from existing window rows.",
        "- L is inherited from stagev5 stagev4 unmasked raw P4/F8 files; only the strict 8 raw F8 columns are model inputs.",
        "- No API call or feature extraction occurs in stagev6 training.",
        "- The external set is held-out external validation and is used for final ranking by external accuracy, consistent with the stagev5 reporting convention.",
    ]
    (cfg.FINAL / "stagev6_selected_model_summary.md").write_text("\n".join(selection) + "\n", encoding="utf-8")
    route_sel = route.loc[route.model_name.eq(selected['model_name'])].copy()
    route_table = route_sel.to_markdown(index=False) if not route_sel.empty else "No route diagnostics available."
    ranks = ranking[["model_name", "gate_id", "branch_id", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]].to_markdown(index=False)
    comps = component_specs[["component_id", "role", "feature_block", "classifier", "scoring", "best_cv_score_grid", "best_params"]].to_markdown(index=False)
    report = [
        "# stagev6 late-first cascade experiment report", "",
        "## Design", "",
        "Stagev6 is a two-level conditional classifier: late vs non-late first, then AD vs control only for samples not directly routed as late. It is not a parallel four-class model.", "",
        "## Fixed component panel", "", comps, "",
        "## Final cascade ranking", "", ranks, "",
        "## Selected-model route diagnostics", "", route_table, "",
        "## Feature audit", "", "```json", json.dumps(feature_manifest, ensure_ascii=False, indent=2), "```", "",
        "## Pre-rendered figures", "", *[f"- `{name}`" for name in figures], "",
    ]
    (cfg.FINAL / "stagev6_experiment_report.md").write_text("\n".join(report), encoding="utf-8")
