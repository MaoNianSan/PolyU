from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import stagev6_config as cfg


def _fmt(x):
    return "NA" if pd.isna(x) else f"{float(x):.6f}"


def make_figures(ranking: pd.DataFrame, gap: pd.DataFrame, pred: pd.DataFrame, subgroup: pd.DataFrame, boot: pd.DataFrame) -> list[str]:
    cfg.FIGURES.mkdir(parents=True,exist_ok=True)
    files=[]
    # 1 ranking
    top=ranking.head(20).iloc[::-1]
    fig,ax=plt.subplots(figsize=(12,8)); ax.barh(top.model_name,top.external_test_accuracy); ax.set_xlabel("External test accuracy"); ax.set_title("Stagev6 cascade models ranked by external test accuracy"); fig.tight_layout(); p=cfg.FIGURES/'fig01_cascade_model_ranking_external_test_accuracy.png'; fig.savefig(p,dpi=180); plt.close(fig); files.append(p.name)
    # 2 gate diagnostics
    gate=ranking[["model_name","gate_model_name","gate_external_test_balanced_accuracy","gate_external_test_sensitivity","gate_external_test_specificity"]].drop_duplicates().sort_values("gate_external_test_balanced_accuracy",ascending=False).head(20)
    fig,ax=plt.subplots(figsize=(12,7)); ax.barh(gate.gate_model_name,gate.gate_external_test_balanced_accuracy); ax.set_xlabel("Late-gate external test balanced accuracy"); ax.set_title("Late-gate external test diagnostic across cascade candidates"); fig.tight_layout(); p=cfg.FIGURES/'fig02_late_gate_external_test_diagnostics.png'; fig.savefig(p,dpi=180); plt.close(fig); files.append(p.name)
    # 3 selected confusion
    sel=ranking.iloc[0].model_name; s=pred[pred.model_name.eq(sel)]; tn=((s.y_true==0)&(s.y_pred==0)).sum(); fp=((s.y_true==0)&(s.y_pred==1)).sum(); fn=((s.y_true==1)&(s.y_pred==0)).sum(); tp=((s.y_true==1)&(s.y_pred==1)).sum()
    fig,ax=plt.subplots(figsize=(5.5,5)); im=ax.imshow([[tn,fp],[fn,tp]]); ax.set_xticks([0,1],['Pred Control','Pred AD']); ax.set_yticks([0,1],['True Control','True AD']); ax.set_title(f'Selected cascade confusion matrix\n{sel}');
    for i,row in enumerate([[tn,fp],[fn,tp]]):
        for j,v in enumerate(row): ax.text(j,i,str(v),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); p=cfg.FIGURES/'fig03_selected_cascade_confusion_matrix.png'; fig.savefig(p,dpi=180); plt.close(fig); files.append(p.name)
    # 4 CV ext gap
    g=gap.head(30).copy(); fig,ax=plt.subplots(figsize=(11,6)); ax.scatter(g.cv_accuracy,g.external_test_accuracy); ax.plot([0,1],[0,1]); ax.set_xlabel('OOF CV accuracy'); ax.set_ylabel('External test accuracy'); ax.set_title('CV versus external test performance'); fig.tight_layout(); p=cfg.FIGURES/'fig04_cv_external_test_gap.png'; fig.savefig(p,dpi=180); plt.close(fig); files.append(p.name)
    # 5 subgroup selected
    sg=subgroup[subgroup.model_name.eq(sel)].copy(); fig,ax=plt.subplots(figsize=(7,5)); ax.bar(sg.severity_group,sg.accuracy); ax.set_ylim(0,1); ax.set_ylabel('Accuracy'); ax.set_title('Selected cascade subgroup disease-decision accuracy'); fig.tight_layout(); p=cfg.FIGURES/'fig05_stage_subgroup_accuracy.png'; fig.savefig(p,dpi=180); plt.close(fig); files.append(p.name)
    # 6 route types
    dist=s.route_error_type.value_counts().sort_values(); fig,ax=plt.subplots(figsize=(10,6)); ax.barh(dist.index,dist.values); ax.set_title('Selected cascade external test route diagnostics'); ax.set_xlabel('External test samples'); fig.tight_layout(); p=cfg.FIGURES/'fig06_route_error_distribution.png'; fig.savefig(p,dpi=180); plt.close(fig); files.append(p.name)
    # 7 bootstrap accuracy
    b=boot[(boot.model_name.eq(sel))&(boot.metric.eq('accuracy'))]; fig,ax=plt.subplots(figsize=(7,4)); ax.errorbar([0],[b.bootstrap_mean.iloc[0]],yerr=[[b.bootstrap_mean.iloc[0]-b.ci_low.iloc[0]],[b.ci_high.iloc[0]-b.bootstrap_mean.iloc[0]]],fmt='o'); ax.set_xlim(-1,1); ax.set_ylim(0,1); ax.set_xticks([0],['Selected cascade']); ax.set_ylabel('External test accuracy'); ax.set_title('External test bootstrap 95% CI'); fig.tight_layout(); p=cfg.FIGURES/'fig07_bootstrap_external_test_accuracy_ci.png'; fig.savefig(p,dpi=180); plt.close(fig); files.append(p.name)
    return files


def write_reports(ranking: pd.DataFrame, cv: pd.DataFrame, gap: pd.DataFrame, pred: pd.DataFrame, gate: pd.DataFrame, branch: pd.DataFrame, schema: dict, source: dict, boot: pd.DataFrame, figures: list[str]) -> tuple[dict,pd.DataFrame,pd.DataFrame]:
    sel=ranking.iloc[0]; name=sel.model_name
    subgroup=(pred.groupby(['model_name','severity_group'],dropna=False).agg(n=('sample_id','size'),accuracy=('correct','mean'),mean_p_ad=('p_ad_mixture','mean'),false_negatives=('error_type',lambda x:int(x.astype(str).str.startswith('FN_').sum())),false_positives=('error_type',lambda x:int(x.astype(str).eq('FP_normal').sum()))).reset_index())
    routes=(pred.groupby(['model_name','route_error_type'],dropna=False).size().reset_index(name='n'))
    selected_errors=pred[(pred.model_name.eq(name))&(pred.correct.eq(0))].sort_values('p_ad_mixture')
    subgroup.to_csv(cfg.FINAL/'stagev6_stage_subgroup_accuracy.csv',index=False)
    routes.to_csv(cfg.FINAL/'stagev6_route_diagnostics.csv',index=False)
    selected_errors.to_csv(cfg.FINAL/'stagev6_error_analysis.csv',index=False)
    tablecols=[c for c in ['model_name','gate_model_name','branch_model_name','external_test_accuracy','external_test_balanced_accuracy','external_test_sensitivity','external_test_specificity','external_test_f1','external_test_roc_auc','external_test_pr_auc','external_test_mcc','external_test_tn','external_test_fp','external_test_fn','external_test_tp','gate_external_test_balanced_accuracy','branch_external_test_accuracy'] if c in ranking]
    (cfg.FINAL/'stagev6_selected_model_summary.md').write_text(
        '# stagev6 selected cascade summary\n\n## Selection protocol\n\n'
        '- Structure: late gate first; gate-positive samples are directly classified as AD; gate-negative samples enter the non-late AD/control branch.\n'
        '- Features: strict stagev5 E/M/L outputs only; no API calls and no feature re-extraction.\n'
        '- Component model family/grid: stagev5 LR/SVC family; branch additionally retains stagev5 small MLP anchor.\n'
        '- Component tuning: GridSearchCV with 10×1 repeated stratified CV. Gate selection metric: balanced accuracy; branch selection metric: accuracy.\n'
        '- Cascade ranking: held-out external test accuracy (`external_test_accuracy`).\n\n## Selected cascade\n\n'+ranking.loc[[ranking.index[0]],tablecols].to_markdown(index=False)+'\n\n## Selected external test route diagnostics\n\n'+routes[routes.model_name.eq(name)].to_markdown(index=False)+'\n',encoding='utf-8')
    top=ranking.head(30)[tablecols]
    text=("# stagev6 experiment report\n\n## Objective\n\nLate-first cascade for binary AD-versus-control classification. The gate targets `late vs non-late`; gate-positive samples are directly classified as AD. Gate-negative samples are classified by an E+M non-late AD/control branch.\n\n"
          "## Fixed feature provenance\n\n- E: strict stagev2 early BM25 outputs already generated in stagev5.\n- M: strict stagev2 BGE-M3 window embeddings, aggregated by `sample_id` mean exactly as in stagev5.\n- L: strict stagev4 unmasked raw P4/F8 outputs already generated in stagev5.\n\n"
          f"## Data\n\n- Train: {source['n_train']} samples; late={source['n_train_late']}; non-late={source['n_train_nonlate']}.\n- External test: {source['n_test']} samples; late={source['n_test_late']}; non-late={source['n_test_nonlate']}.\n\n"
          "## Ranking Protocol\n\nAll complete cascade candidates below are sorted by held-out external test accuracy (`external_test_accuracy`), with external test balanced accuracy and other external test metrics used only as tie-breakers.\n\n"
          "## Top cascade models by external test accuracy\n\n"+top.to_markdown(index=False)+"\n\n## Interpretation boundary\n\nStagev6 outputs `control`, `non-late AD`, or `late-direct AD` as a routing result. It is not a supervised early/middle/late multiclass classifier.\n")
    (cfg.FINAL/'stagev6_experiment_report.md').write_text(text,encoding='utf-8')
    summary={"selected_model":name,"gate_model_name":sel.gate_model_name,"branch_model_name":sel.branch_model_name,"external_test_accuracy":float(sel.external_test_accuracy),"external_test_balanced_accuracy":float(sel.external_test_balanced_accuracy),"external_accuracy":float(sel.accuracy),"external_balanced_accuracy":float(sel.balanced_accuracy),"n_cascade_models_completed":int(len(ranking)),"selection_metric":"external_test_accuracy","selection_data_split":"held-out external test","feature_dimensions":schema,"figures":figures}
    with (cfg.FINAL/'stagev6_final_run_summary.json').open('w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2,default=str)
    return summary,subgroup,routes
