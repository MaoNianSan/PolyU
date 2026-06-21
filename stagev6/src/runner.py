from __future__ import annotations
import json, time, traceback, shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from . import stagev6_config as cfg
from .bootstrap_ci import stratified_bootstrap_ci
from .cascade import block_matrix, hard_route, oof_component_predictions, route_fields
from .data_io import infer_schema, load_train_test
from .evaluation import get_positive_prob, metrics_binary_prob, metrics_from_hard_and_prob
from .io_utils import save_json
from .model_registry import build_branch_specs, build_gate_specs
from .paths import ensure_output_dirs
from .reporting import make_figures, write_reports

TOP_N_OOF=10
METRIC_COLUMNS=["accuracy","balanced_accuracy","sensitivity","specificity","precision","f1","roc_auc","pr_auc","mcc","log_loss","brier","tn","fp","fn","tp","gate_threshold","branch_threshold"]

def _json_params(d):
    out={}
    for k,v in d.items(): out[k]=v.item() if isinstance(v,(np.integer,np.floating)) else v
    return json.dumps(out,ensure_ascii=False,sort_keys=True)

def _safe(n): return n.replace('/','_').replace('\\','_').replace(' ','_')+'.joblib'

def _fit_component(spec, X, y, cv, scoring, n_jobs):
    start=time.time()
    grid=GridSearchCV(spec.estimator,spec.param_grid,scoring=scoring,cv=cv,n_jobs=n_jobs,refit=True,error_score='raise',return_train_score=False)
    grid.fit(X,y)
    return {"spec":spec,"best":grid.best_estimator_,"best_score":float(grid.best_score_),"best_params":grid.best_params_,"fit_seconds":time.time()-start}

def _component_oof(estimator, X, y, cv):
    p=np.zeros(len(y)); count=np.zeros(len(y),int)
    for tr,va in cv.split(X,y):
        m=clone(estimator); m.fit(X.iloc[tr],y[tr]); p[va]=get_positive_prob(m,X.iloc[va]); count[va]+=1
    if not np.all(count==1): raise RuntimeError('component OOF coverage failure')
    return p

def _branch_oof(estimator,X,y_ad,y_late,cv):
    p=np.zeros(len(y_ad)); count=np.zeros(len(y_ad),int)
    for tr,va in cv.split(X,y_late):
        tr_nonlate=tr[np.asarray(y_late)[tr]==0]
        m=clone(estimator); m.fit(X.iloc[tr_nonlate],y_ad[tr_nonlate]); p[va]=get_positive_prob(m,X.iloc[va]); count[va]+=1
    if not np.all(count==1): raise RuntimeError('branch OOF coverage failure')
    return p

def _copy_final(src:Path,dst:Path):
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)

def _add_external_test_aliases(df:pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    for metric in METRIC_COLUMNS:
        legacy=f'external_{metric}'
        alias=f'external_test_{metric}'
        if metric in out.columns and alias not in out.columns:
            out[alias]=out[metric]
        if legacy in out.columns and alias not in out.columns:
            out[alias]=out[legacy]
    return out

def run(n_jobs:int, bootstrap_n:int, max_gate_models:int|None=None, max_branch_models:int|None=None):
    dirs=ensure_output_dirs(); t0=time.time()
    train,test,source=load_train_test(dirs['reports']); schema=infer_schema(train,test)
    save_json(schema,dirs['reports']/'feature_manifest_used.json')
    y_ad=train['__y__'].to_numpy(int); y_late=train['label_late'].to_numpy(int)
    yt_ad=test['__y__'].to_numpy(int); yt_late=test['label_late'].to_numpy(int)
    joint=np.where(y_late==1,2,np.where(y_ad==1,1,0))
    cv_gate=RepeatedStratifiedKFold(n_splits=cfg.CV_N_SPLITS,n_repeats=cfg.CV_N_REPEATS,random_state=cfg.RANDOM_STATE)
    cv_branch=RepeatedStratifiedKFold(n_splits=cfg.CV_N_SPLITS,n_repeats=cfg.CV_N_REPEATS,random_state=cfg.RANDOM_STATE)
    cv_route=RepeatedStratifiedKFold(n_splits=cfg.CV_N_SPLITS,n_repeats=cfg.CV_N_REPEATS,random_state=cfg.RANDOM_STATE)
    gate_specs=build_gate_specs(); branch_specs=build_branch_specs()
    if max_gate_models: gate_specs=gate_specs[:max_gate_models]
    if max_branch_models: branch_specs=branch_specs[:max_branch_models]
    run_config={"stage_version":"stagev6","random_state":cfg.RANDOM_STATE,"cv_n_splits":cfg.CV_N_SPLITS,"cv_n_repeats":cfg.CV_N_REPEATS,"gate_scoring":"balanced_accuracy","branch_scoring":"accuracy","selection_metric":"external_test_accuracy","selection_data_split":"held-out external test","gate_threshold":cfg.GATE_THRESHOLD,"branch_threshold":cfg.BRANCH_THRESHOLD,"bootstrap_n":bootstrap_n,"n_jobs":n_jobs,"n_gate_components":len(gate_specs),"n_branch_components":len(branch_specs),"n_cascade_pairs":len(gate_specs)*len(branch_specs),"data_sources":source,"feature_schema":schema,"component_reuse":"Each gate and branch component is GridSearchCV-fitted once; cascade pairs combine independently fitted components and are not refit redundantly."}
    save_json(run_config,dirs['reports']/'run_config.json')
    # Fit components.
    gate_fits=[]; branch_fits=[]; errors=[]
    for spec in gate_specs:
        print(f'[stagev6] gate component {spec.name}',flush=True)
        try:
            Xtr=block_matrix(train,schema,spec.feature_block); Xte=block_matrix(test,schema,spec.feature_block)
            fit=_fit_component(spec,Xtr,y_late,cv_gate,'balanced_accuracy',n_jobs)
            fit['oof']=_component_oof(fit['best'],Xtr,y_late,cv_gate)
            fit['test']=get_positive_prob(fit['best'],Xte)
            m=metrics_binary_prob(y_late,fit['oof']); ex=metrics_binary_prob(yt_late,fit['test'])
            fit['cv_metrics']=m; fit['external_metrics']=ex
            gate_fits.append(fit); joblib.dump(fit['best'],dirs['models_gate']/_safe(spec.name))
        except Exception as exc:
            errors.append({'component':spec.name,'role':'gate','error':repr(exc),'traceback':traceback.format_exc()})
    for spec in branch_specs:
        print(f'[stagev6] branch component {spec.name}',flush=True)
        try:
            Xtr=block_matrix(train,schema,spec.feature_block); Xte=block_matrix(test,schema,spec.feature_block)
            tr_nonlate=y_late==0; te_nonlate=yt_late==0
            fit=_fit_component(spec,Xtr.loc[tr_nonlate].reset_index(drop=True),y_ad[tr_nonlate],cv_branch,'accuracy',n_jobs)
            fit['oof']=_branch_oof(fit['best'],Xtr,y_ad,y_late,cv_route)
            fit['test']=get_positive_prob(fit['best'],Xte)
            fit['cv_metrics']=metrics_binary_prob(y_ad[tr_nonlate],fit['oof'][tr_nonlate])
            fit['external_metrics']=metrics_binary_prob(yt_ad[te_nonlate],fit['test'][te_nonlate])
            branch_fits.append(fit); joblib.dump(fit['best'],dirs['models_branch']/_safe(spec.name))
        except Exception as exc:
            errors.append({'component':spec.name,'role':'branch','error':repr(exc),'traceback':traceback.format_exc()})
    if not gate_fits or not branch_fits: raise RuntimeError(f'No complete components. errors={errors}')
    # Component reports.
    comp_rows=[]; gate_rows=[]; branch_rows=[]
    for f in gate_fits:
        s=f['spec']; row={'component_name':s.name,'role':'gate','group':s.group,'feature_block':s.feature_block,'best_cv_score_grid':f['best_score'],'best_params':_json_params(f['best_params']),'fit_seconds':f['fit_seconds']}
        row.update({f'cv_{k}':v for k,v in f['cv_metrics'].items()}); row.update({f'external_{k}':v for k,v in f['external_metrics'].items()}); row=_add_external_test_aliases(pd.DataFrame([row])).iloc[0].to_dict(); gate_rows.append(row); comp_rows.append(row)
    for f in branch_fits:
        s=f['spec']; row={'component_name':s.name,'role':'branch','group':s.group,'feature_block':s.feature_block,'best_cv_score_grid':f['best_score'],'best_params':_json_params(f['best_params']),'fit_seconds':f['fit_seconds']}
        row.update({f'cv_{k}':v for k,v in f['cv_metrics'].items()}); row.update({f'external_{k}':v for k,v in f['external_metrics'].items()}); row=_add_external_test_aliases(pd.DataFrame([row])).iloc[0].to_dict(); branch_rows.append(row); comp_rows.append(row)
    gate_df=pd.DataFrame(gate_rows); branch_df=pd.DataFrame(branch_rows); comps=pd.DataFrame(comp_rows)
    # Construct every cascade pair. OOF values are created by matching fold structure for each component pair (hard routing), not by thresholding the mixture probability.
    cv_rows=[]; ext_rows=[]; preds=[]; oof_all=[]; boot=[]
    for g in gate_fits:
        for b in branch_fits:
            name=f'cascade__{g["spec"].name}__TO__{b["spec"].name}'
            print(f'[stagev6] cascade {name}',flush=True)
            # Reuse component OOF outputs produced under the common route CV split. The branch OOF has predicted late held-out rows from non-late fitted folds.
            ypred_oof,pmix_oof,route_oof=hard_route(g['oof'],b['oof'])
            cvm=metrics_from_hard_and_prob(y_ad,ypred_oof,pmix_oof)
            ypred,pmix,route=hard_route(g['test'],b['test'])
            extm=metrics_from_hard_and_prob(yt_ad,ypred,pmix)
            gext=g['external_metrics']; bext=b['external_metrics']
            common={'model_name':name,'gate_model_name':g['spec'].name,'branch_model_name':b['spec'].name,'gate_feature_block':g['spec'].feature_block,'branch_feature_block':b['spec'].feature_block,'gate_best_params':_json_params(g['best_params']),'branch_best_params':_json_params(b['best_params']),'gate_best_cv_score_grid':g['best_score'],'branch_best_cv_score_grid':b['best_score'],'gate_external_balanced_accuracy':gext['balanced_accuracy'],'gate_external_sensitivity':gext['sensitivity'],'gate_external_specificity':gext['specificity'],'gate_external_test_balanced_accuracy':gext['balanced_accuracy'],'gate_external_test_sensitivity':gext['sensitivity'],'gate_external_test_specificity':gext['specificity'],'branch_external_accuracy':bext['accuracy'],'branch_external_balanced_accuracy':bext['balanced_accuracy'],'branch_external_test_accuracy':bext['accuracy'],'branch_external_test_balanced_accuracy':bext['balanced_accuracy']}
            cv_rows.append({**common,**{f'cv_{k}':v for k,v in cvm.items()}})
            ext_rows.append({**common,**extm})
            pp=route_fields(test,g['test'],b['test'],pmix,ypred,route,name,g['spec'].name,b['spec'].name); preds.append(pp)
            oo=route_fields(train,g['oof'],b['oof'],pmix_oof,ypred_oof,route_oof,name,g['spec'].name,b['spec'].name); oof_all.append(oo)
            boot.append(stratified_bootstrap_ci(yt_ad,ypred,pmix,name,bootstrap_n,cfg.RANDOM_STATE))
    rank=_add_external_test_aliases(pd.DataFrame(ext_rows)).sort_values(['external_test_accuracy','external_test_balanced_accuracy','external_test_sensitivity','external_test_specificity','external_test_f1','external_test_roc_auc','external_test_pr_auc'],ascending=False).reset_index(drop=True)
    cvdf=pd.DataFrame(cv_rows).sort_values(['cv_accuracy','cv_balanced_accuracy'],ascending=False).reset_index(drop=True)
    pred=pd.concat(preds,ignore_index=True); oof=pd.concat(oof_all,ignore_index=True); bootdf=pd.concat(boot,ignore_index=True)
    gap=cvdf[['model_name','cv_accuracy','cv_balanced_accuracy','cv_f1','cv_roc_auc','cv_pr_auc']].merge(rank[['model_name','accuracy','balanced_accuracy','f1','roc_auc','pr_auc','external_test_accuracy','external_test_balanced_accuracy','external_test_f1','external_test_roc_auc','external_test_pr_auc']],on='model_name')
    gap=gap.rename(columns={'accuracy':'external_accuracy','balanced_accuracy':'external_balanced_accuracy','f1':'external_f1','roc_auc':'external_roc_auc','pr_auc':'external_pr_auc'})
    gap['accuracy_gap']=gap.cv_accuracy-gap.external_test_accuracy; gap['balanced_accuracy_gap']=gap.cv_balanced_accuracy-gap.external_test_balanced_accuracy; gap['roc_auc_gap']=gap.cv_roc_auc-gap.external_test_roc_auc
    # Raw reports.
    rank.to_csv(dirs['tables']/'stagev6_model_ranking_by_external_accuracy.csv',index=False); rank.to_csv(dirs['tables']/'stagev6_model_ranking_by_external_test_accuracy.csv',index=False); rank.to_csv(dirs['tables']/'stagev6_external_performance_report.csv',index=False); rank.to_csv(dirs['tables']/'stagev6_external_test_performance_report.csv',index=False); cvdf.to_csv(dirs['tables']/'stagev6_cv_summary.csv',index=False); bootdf.to_csv(dirs['tables']/'stagev6_bootstrap_ci.csv',index=False); gap.to_csv(dirs['tables']/'stagev6_generalization_gap.csv',index=False); pred.to_csv(dirs['predictions']/'stagev6_test_predictions_all_models.csv',index=False); oof[oof.model_name.isin(rank.head(TOP_N_OOF).model_name)].to_csv(dirs['predictions']/'stagev6_oof_predictions_top10.csv',index=False); gate_df.to_csv(dirs['tables']/'stagev6_gate_component_performance.csv',index=False); branch_df.to_csv(dirs['tables']/'stagev6_branch_component_performance.csv',index=False); comps.to_csv(dirs['tables']/'stagev6_component_specifications.csv',index=False)
    # Store selected component pair models.
    chosen=rank.iloc[0]; gf=next(f for f in gate_fits if f['spec'].name==chosen.gate_model_name); bf=next(f for f in branch_fits if f['spec'].name==chosen.branch_model_name)
    joblib.dump(gf['best'],dirs['models_selected']/('selected_gate__'+_safe(gf['spec'].name))); joblib.dump(bf['best'],dirs['models_selected']/('selected_branch__'+_safe(bf['spec'].name)))
    if errors: pd.DataFrame(errors).to_csv(dirs['logs']/'stagev6_component_errors.csv',index=False)
    # Copy raw outputs to canonical final report before report generation.
    for nm in ['stagev6_model_ranking_by_external_accuracy.csv','stagev6_model_ranking_by_external_test_accuracy.csv','stagev6_external_performance_report.csv','stagev6_external_test_performance_report.csv','stagev6_cv_summary.csv','stagev6_bootstrap_ci.csv','stagev6_generalization_gap.csv','stagev6_gate_component_performance.csv','stagev6_branch_component_performance.csv','stagev6_component_specifications.csv']:
        _copy_final(dirs['tables']/nm,cfg.FINAL/nm)
    for nm in ['stagev6_test_predictions_all_models.csv','stagev6_oof_predictions_top10.csv']:
        _copy_final(dirs['predictions']/nm,cfg.FINAL/nm)
    manifest={'stage_version':'stagev6','feature_policy':cfg.FEATURE_POLICY,'feature_schema':schema,'feature_reextraction_performed':False,'api_called':False,'middle_aggregation':'sample_id mean on stagev5 existing BGE-M3 windows','late_model_inputs':'raw stagev4 P4/F8 only','classification_structure':'late gate (L or M+L) -> direct AD; non-late branch (E+M) -> AD/control'}
    save_json(manifest,cfg.FINAL/'stagev6_feature_source_manifest.json')
    leakage={'standard_scaler_inside_pipeline':True,'imputer_inside_pipeline':True,'mmse_used_as_input_feature':False,'external_test_accuracy_used_for_cascade_selection':True,'external_set_role':'held-out external test; used for Stagev6 model ranking/selection, so it is not an unbiased final test after selection','late_gate_training_target':'late vs non-late','branch_training_population':'true non-late training samples only','gate_threshold':cfg.GATE_THRESHOLD,'branch_threshold':cfg.BRANCH_THRESHOLD,'hard_route_used_for_accuracy_metrics':True,'mixture_probability_used_for_probability_metrics':True,'feature_reextraction_performed':False,'api_called':False,'run_seconds':time.time()-t0}
    save_json(leakage,cfg.FINAL/'stagev6_leakage_check.json')
    # Reports/figures use final tables.
    figures=make_figures(rank,gap,pred,pd.DataFrame(),bootdf) if False else []
    # Figure function requires subgroup; compute once by temporary call from write_reports then figures.
    subgroup=(pred.groupby(['model_name','severity_group'],dropna=False).agg(n=('sample_id','size'),accuracy=('correct','mean'),mean_p_ad=('p_ad_mixture','mean'),false_negatives=('error_type',lambda x:int(x.astype(str).str.startswith('FN_').sum())),false_positives=('error_type',lambda x:int(x.astype(str).eq('FP_normal').sum()))).reset_index())
    figures=make_figures(rank,gap,pred,subgroup,bootdf)
    summary,subgroup,routes=write_reports(rank,cvdf,gap,pred,gate_df,branch_df,schema,source,bootdf,figures)
    # all final output auditing
    missing=[nm for nm in cfg.CANONICAL_FINAL_FILES if not (cfg.FINAL/nm).exists()]
    if missing: raise RuntimeError(f'Missing canonical final outputs: {missing}')
    print('[stagev6] Done.')
    print(f'[stagev6] Results: {cfg.FINAL}')
    return summary
