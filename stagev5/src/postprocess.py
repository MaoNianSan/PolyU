from __future__ import annotations
import json, shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stagev5_config as cfg
from .feature_adapter import validate_feature_outputs


def _copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _save_figures(ranking: pd.DataFrame, gap: pd.DataFrame, preds: pd.DataFrame, subgroup: pd.DataFrame, boot: pd.DataFrame) -> list[str]:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    made=[]
    if not ranking.empty:
        top=ranking.head(15).copy().iloc[::-1]
        plt.figure(figsize=(10, max(4, 0.34*len(top))))
        plt.barh(top["model_name"], top["accuracy"])
        plt.xlabel("External accuracy")
        plt.tight_layout(); p=cfg.FIGURES/"fig01_model_ranking_external_accuracy.png"; plt.savefig(p,dpi=220); plt.close(); made.append(p.name)
        block=ranking.groupby("feature_block",as_index=False)["accuracy"].max().sort_values("accuracy",ascending=False)
        if not block.empty:
            plt.figure(figsize=(10,max(4,0.36*len(block))))
            plt.barh(block.iloc[::-1]["feature_block"],block.iloc[::-1]["accuracy"])
            plt.xlabel("Best external accuracy within feature block")
            plt.tight_layout(); p=cfg.FIGURES/"fig02_feature_block_comparison.png"; plt.savefig(p,dpi=220); plt.close(); made.append(p.name)
    if not gap.empty:
        top=gap.head(20).copy()
        plt.figure(figsize=(8,5))
        plt.scatter(top["cv_accuracy"],top["external_accuracy"])
        lo=float(np.nanmin(np.r_[top["cv_accuracy"],top["external_accuracy"]])); hi=float(np.nanmax(np.r_[top["cv_accuracy"],top["external_accuracy"]]))
        plt.plot([lo,hi],[lo,hi])
        plt.xlabel("Repeated 10-fold CV accuracy")
        plt.ylabel("External accuracy")
        plt.tight_layout(); p=cfg.FIGURES/"fig04_cv_external_gap.png"; plt.savefig(p,dpi=220); plt.close(); made.append(p.name)
    if not preds.empty and not ranking.empty:
        selected=str(ranking.iloc[0]["model_name"])
        sel=preds.loc[preds.model_name.eq(selected)].copy()
        if not sel.empty:
            cm=pd.crosstab(sel.y_true,sel.y_pred).reindex(index=[0,1],columns=[0,1],fill_value=0)
            plt.figure(figsize=(4.5,4))
            plt.imshow(cm.to_numpy())
            plt.xticks([0,1],["Pred control","Pred AD"]);plt.yticks([0,1],["True control","True AD"])
            for i in range(2):
                for j in range(2): plt.text(j,i,str(int(cm.iloc[i,j])),ha="center",va="center")
            plt.title("Selected model: external confusion matrix")
            plt.tight_layout(); p=cfg.FIGURES/"fig03_selected_model_confusion_matrix.png"; plt.savefig(p,dpi=220); plt.close(); made.append(p.name)
    if not subgroup.empty:
        selected=str(ranking.iloc[0]["model_name"]) if not ranking.empty else subgroup.model_name.iloc[0]
        sub=subgroup.loc[subgroup.model_name.eq(selected)].copy()
        if not sub.empty:
            plt.figure(figsize=(8,4.5)); plt.bar(sub["severity_group"],sub["accuracy"]); plt.ylim(0,1); plt.ylabel("External disease-decision accuracy")
            plt.tight_layout(); p=cfg.FIGURES/"fig05_stage_subgroup_accuracy.png"; plt.savefig(p,dpi=220); plt.close(); made.append(p.name)
    if not preds.empty and not ranking.empty:
        selected=str(ranking.iloc[0]["model_name"]); e=preds.loc[preds.model_name.eq(selected)].query("correct == 0")
        if not e.empty:
            c=e.error_type.value_counts(); plt.figure(figsize=(8,4.5));plt.bar(c.index,c.values);plt.ylabel("External errors");plt.xticks(rotation=25,ha="right")
            plt.tight_layout();p=cfg.FIGURES/"fig06_error_type_distribution.png";plt.savefig(p,dpi=220);plt.close();made.append(p.name)
    if not boot.empty and not ranking.empty:
        selected=str(ranking.iloc[0]["model_name"]); b=boot.loc[(boot.model_name.eq(selected))&(boot.metric.eq("accuracy"))]
        if not b.empty:
            r=b.iloc[0];plt.figure(figsize=(6,2.4));plt.errorbar([0],[r.bootstrap_mean],yerr=[[r.bootstrap_mean-r.ci_low],[r.ci_high-r.bootstrap_mean]],fmt='o');plt.xlim(-1,1);plt.xticks([]);plt.ylabel("External accuracy")
            plt.tight_layout();p=cfg.FIGURES/"fig07_bootstrap_external_accuracy_ci.png";plt.savefig(p,dpi=220);plt.close();made.append(p.name)
    return made


def postprocess_stagev5() -> dict[str, Any]:
    status=validate_feature_outputs()
    run=cfg.CLASSIFIER_RUN
    table=run/"tables"; pred_dir=run/"predictions"; report=run/"reports"
    cfg.FINAL.mkdir(parents=True,exist_ok=True)
    mapping={
        table/"stagev2_model_ranking_by_external_accuracy.csv": cfg.FINAL/"stagev5_model_ranking_by_external_accuracy.csv",
        table/"stagev2_external_performance_report.csv": cfg.FINAL/"stagev5_external_performance_report.csv",
        table/"stagev2_cv_summary.csv": cfg.FINAL/"stagev5_cv_summary.csv",
        table/"stagev2_bootstrap_ci.csv": cfg.FINAL/"stagev5_bootstrap_ci.csv",
        table/"stagev2_generalization_gap.csv": cfg.FINAL/"stagev5_generalization_gap.csv",
        pred_dir/"stagev2_oof_predictions_top10.csv": cfg.FINAL/"stagev5_oof_predictions_top10.csv",
        pred_dir/"stagev2_test_predictions_all_models.csv": cfg.FINAL/"stagev5_test_predictions_all_models.csv",
        report/"stagev2_leakage_check.json": cfg.FINAL/"stagev5_leakage_check.json",
    }
    for s,d in mapping.items(): _copy(s,d)
    # Preserve source core reports for audit without changing their contents.
    for name in ["stagev2_selected_model_summary.md","stagev2_experiment_report.md","feature_manifest_used.json","run_config.json"]:
        _copy(report/name,cfg.FINAL/f"source_core_{name}")
    ranking=_read(cfg.FINAL/"stagev5_model_ranking_by_external_accuracy.csv")
    ext=_read(cfg.FINAL/"stagev5_external_performance_report.csv")
    cv=_read(cfg.FINAL/"stagev5_cv_summary.csv")
    gap=_read(cfg.FINAL/"stagev5_generalization_gap.csv")
    preds=_read(cfg.FINAL/"stagev5_test_predictions_all_models.csv")
    boot=_read(cfg.FINAL/"stagev5_bootstrap_ci.csv")
    if ranking.empty or preds.empty:
        raise RuntimeError("Expected classifier outputs are unavailable after stagev2 core run.")
    subgroup=(preds.groupby(["model_name","severity_group"],dropna=False)
              .agg(n=("sample_id","size"),accuracy=("correct","mean"),mean_p_ad=("p_ad","mean"),false_negatives=("error_type",lambda x:int((x.astype(str).str.startswith("FN_")).sum())),false_positives=("error_type",lambda x:int((x.astype(str).eq("FP_normal")).sum())))
              .reset_index())
    subgroup.to_csv(cfg.FINAL/"stagev5_stage_subgroup_accuracy.csv",index=False)
    selected=str(ranking.iloc[0]["model_name"])
    selected_pred=preds.loc[preds.model_name.eq(selected)].copy()
    errors=selected_pred.loc[selected_pred.correct.eq(0)].sort_values("p_ad")
    errors.to_csv(cfg.FINAL/"stagev5_error_analysis.csv",index=False)
    source_manifest={
        "stage_version":"stagev5",
        "E":{"source":"stagev2.zip","implementation":"reference_stagev2/stagev1_features.py","feature_status":status},
        "M":{"source":"stagev2.zip","implementation":"reference_stagev2/api_feature_extraction.py","feature_status":status},
        "L":{
            "source":"stagev4_unmasked_form_comparator.zip",
            "implementation":"reference_stagev4/late_extract.py",
            "feature_status":status,
            "late_raw_f8_columns":status["late_raw_F8_columns"],
            "late_auxiliary_columns":status["late_auxiliary_columns"],
            "late_model_feature_count":status["late_model_feature_count"],
            "late_diagnostic_column_count":status["late_diagnostic_column_count"],
            "late_auxiliary_used_in_model":False,
            "late_activation_policy":status["late_activation_policy"],
        },
        "classification_core":"stagev2/stage_core with CV count configured to 10x1 through environment variables",
        "selection_metric":"external_accuracy",
        "external_set_role":"held-out external validation; not used for model fitting or preprocessing fit",
    }
    (cfg.FINAL/"stagev5_feature_source_manifest.json").write_text(json.dumps(source_manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    figures=_save_figures(ranking,gap,preds,subgroup,boot)
    r=ranking.iloc[0]
    summary={"selected_model":selected,"feature_block":str(r.get("feature_block","")),"external_accuracy":float(r["accuracy"]),"external_balanced_accuracy":float(r.get("balanced_accuracy",np.nan)),"n_models_completed":int(len(ranking)),"selection_metric":"external_accuracy","feature_dimensions":status,"figures":figures}
    (cfg.FINAL/"stagev5_final_run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    metric_cols=[c for c in ["model_name","group","feature_block","accuracy","balanced_accuracy","sensitivity","specificity","f1","roc_auc","pr_auc","mcc","tn","fp","fn","tp"] if c in ranking]
    (cfg.FINAL/"stagev5_selected_model_summary.md").write_text(
        "# stagev5 selected model summary\n\n"
        "## Selection protocol\n\n"
        "- Feature sources: E/M from stagev2; L from stagev4_unmasked P4/F8.\n"
        f"- L raw model features: {status['late_model_feature_count']}\n"
        f"- L auxiliary diagnostic features: {status['late_diagnostic_column_count']}\n"
        f"- L auxiliary features used in model: {str(status['late_auxiliary_used_in_model']).lower()}\n"
        f"- L activation for interaction blocks: {status['late_activation_policy']}\n"
        "- Training: stagev2 classifier panel, GridSearchCV, repeated stratified 10-fold CV (10×1).\n"
        "- Ranking: held-out external accuracy. The external set was not used for fit, scaling, imputation, or BM25 fitting.\n\n"
        "## Selected model\n\n" + ranking.loc[[ranking.index[0]],metric_cols].to_markdown(index=False) + "\n\n"
        "## Stage/subgroup diagnostic\n\n" + subgroup.loc[subgroup.model_name.eq(selected)].to_markdown(index=False) + "\n",
        encoding="utf-8")
    top=ranking.head(30)
    report_text=("# stagev5 experiment report\n\n"
                 "## Objective\n\n"
                 "Binary AD-versus-control classification using the stagev2 classifier panel, with severity/subgroup diagnostic outputs.\n\n"
                 "## Fixed feature provenance\n\n"
                 "- E: strict stagev2 early BM25 implementation.\n"
                 "- M: strict stagev2 BGE-M3 window embedding implementation.\n"
                 "- L: strict stagev4_unmasked P4/F8 expressive-form implementation.\n\n"
                 "## Feature-source audit\n\n"
                 f"- L raw model features: {status['late_model_feature_count']}\n"
                 f"- L auxiliary diagnostic features: {status['late_diagnostic_column_count']}\n"
                 f"- L auxiliary features used in model: {str(status['late_auxiliary_used_in_model']).lower()}\n"
                 f"- L interaction activation: {status['late_activation_policy']}\n\n"
                 "## Top external models\n\n"+top[metric_cols].to_markdown(index=False)+"\n\n"
                 "## Selected model stage/subgroup diagnostic\n\n"+subgroup.loc[subgroup.model_name.eq(selected)].to_markdown(index=False)+"\n\n"
                 "## Interpretation boundary\n\n"
                 "Stagev5 reports severity-specific disease-decision performance. It does not present the binary classifier as a supervised four-class stage classifier.\n")
    (cfg.FINAL/"stagev5_experiment_report.md").write_text(report_text,encoding="utf-8")
    return summary
