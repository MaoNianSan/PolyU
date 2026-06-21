from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE=2026
CV_SPLITS=10
THRESHOLD=0.50
STAGE_ORDER=["control","early_spectrum_AD","middle","late"]
COMPLETION_SENTINEL_NAME="stagev7_training_complete.json"
KNOWN_STAGEV7_FINAL_OUTPUTS=[
    "stagev7_early_gate_model_ranking.csv",
    "stagev7_middle_gate_model_ranking.csv",
    "stagev7_late_gate_model_ranking.csv",
    "stagev7_gate_model_ranking_all.csv",
    "stagev7_cascade_ranking_external_exploratory.csv",
    "stagev7_final_cascade_predictions_all.csv",
    "stagev7_final_cascade_predictions.csv",
    "stagev7_final_primary_performance.csv",
    "stagev7_final_confusion_matrix_stage_collapsed.csv",
    "stagev7_final_confusion_matrix_binary.csv",
    "stagev7_final_stage_strict_audit.csv",
    "stagev7_gate_path_audit.csv",
    "stagev7_cascade_error_analysis.csv",
    "stagev7_bootstrap_ci.csv",
    "stagev7_flat_multiclass_baseline_performance.csv",
    "stagev7_flat_multiclass_baseline_predictions.csv",
    "stagev7_leakage_check.json",
    "stagev7_feature_source_manifest.json",
    "stagev7_final_run_summary.json",
    "stagev7_selected_model_summary.md",
    "stagev7_experiment_report.md",
    COMPLETION_SENTINEL_NAME,
]
KNOWN_STAGEV7_FIGURES=[
    "figures/fig01_gate_best_cv_balanced_accuracy.png",
    "figures/fig02_cascade_external_comparison_exploratory.png",
    "figures/fig03_primary_stage_confusion_matrix.png",
    "figures/fig04_primary_stage_subgroup_accuracy.png",
    "figures/fig05_primary_bootstrap_ci.png",
]

@dataclass(frozen=True)
class ModelDef:
    key: str
    family: str
    feature_block: str
    estimator: Any
    grid: dict[str, list[Any]]


def _utc() -> str: return datetime.now(timezone.utc).isoformat()
def _json(x: Any) -> Any:
    if isinstance(x, (np.integer,np.floating)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)

def _pipe(clf: Any) -> Pipeline:
    return Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("clf",clf)])

def _multiclass_lr() -> LogisticRegression:
    kwargs={'solver':'lbfgs','max_iter':5000,'random_state':RANDOM_STATE}
    try:
        return LogisticRegression(**kwargs,multi_class='multinomial')
    except TypeError:
        return LogisticRegression(**kwargs)

def _model_defs(blocks: list[str], include_mlp: bool=False) -> list[ModelDef]:
    out=[]
    for b in blocks:
        out += [
        ModelDef(f"{b}__lr__l2","LR-L2",b,_pipe(LogisticRegression(solver="liblinear",penalty="l2",max_iter=5000,random_state=RANDOM_STATE)),{"clf__C":[0.1,1.0],"clf__class_weight":["balanced"]}),
        ModelDef(f"{b}__lr__elasticnet","LR-elasticnet",b,_pipe(LogisticRegression(solver="saga",penalty="elasticnet",max_iter=4000,random_state=RANDOM_STATE)),{"clf__C":[0.1,1.0],"clf__l1_ratio":[0.5],"clf__class_weight":["balanced"]}),
        ModelDef(f"{b}__svc__linear","Linear-SVC",b,_pipe(SVC(kernel="linear",probability=False,random_state=RANDOM_STATE)),{"clf__C":[0.1,1.0],"clf__class_weight":["balanced"]}),
        ModelDef(f"{b}__svc__poly2","SVC-poly2",b,_pipe(SVC(kernel="poly",degree=2,probability=False,random_state=RANDOM_STATE)),{"clf__C":[0.1,1.0],"clf__gamma":["scale"],"clf__coef0":[1.0],"clf__class_weight":["balanced"]}),
        ModelDef(f"{b}__svc__poly3","SVC-poly3",b,_pipe(SVC(kernel="poly",degree=3,probability=False,random_state=RANDOM_STATE)),{"clf__C":[0.1,1.0],"clf__gamma":["scale"],"clf__coef0":[1.0],"clf__class_weight":["balanced"]}),
        ModelDef(f"{b}__svc__rbf","SVC-RBF",b,_pipe(SVC(kernel="rbf",probability=False,random_state=RANDOM_STATE)),{"clf__C":[0.1,1.0],"clf__gamma":["scale"],"clf__class_weight":["balanced"]}),
        ]
        if include_mlp:
            out.append(ModelDef(f"{b}__mlp__small","MLP-small",b,_pipe(MLPClassifier(max_iter=300,early_stopping=True,validation_fraction=0.2,n_iter_no_change=15,random_state=RANDOM_STATE)),{"clf__hidden_layer_sizes":[(8,),(16,),(8,4)],"clf__alpha":[1e-4,1e-3],"clf__learning_rate_init":[0.001]}))
    return out

def _load_features(root: Path) -> tuple[pd.DataFrame,dict[str,list[str]],dict[str,Any]]:
    dirs={"E":root/'output/features/E/raw_stagev2',"M":root/'output/features/M/raw_stagev2',"L":root/'output/features/L/raw_stagev4'}
    names={"ad":{"E":"ad_BM25.csv","M":"ad_embedding.csv","L":"ad_LLM.csv"},"control":{"E":"control_BM25.csv","M":"control_embedding.csv","L":"control_LLM.csv"},"test":{"E":"test_BM25.csv","M":"test_embedding.csv","L":"test_LLM.csv"}}
    split_rows=[]; manifests={}
    for split in ["ad","control","test"]:
        e=pd.read_csv(dirs['E']/names[split]['E']); m=pd.read_csv(dirs['M']/names[split]['M']); l=pd.read_csv(dirs['L']/names[split]['L'])
        if 'sample_id' not in e or 'sample_id' not in m or 'sample_id' not in l: raise ValueError(f"sample_id missing in {split}")
        ecols=[c for c in e if c.startswith('early_') and pd.api.types.is_numeric_dtype(e[c])]
        mcols=[c for c in m if c.startswith('embedding_dim_') and pd.api.types.is_numeric_dtype(m[c])]
        lcols=[c for c in l if c.startswith('late_') and pd.api.types.is_numeric_dtype(l[c]) and not c.startswith('late_llm_')]
        if len(ecols)!=61: raise ValueError(f"Expected 61 E features; found {len(ecols)}")
        if len(mcols)!=1024: raise ValueError(f"Expected 1024 M features; found {len(mcols)}")
        if len(lcols)!=8: raise ValueError(f"Expected 8 L features; found {len(lcols)}")
        eb=e[['sample_id','disease_label','mmse','label_early','label_middle','label_late']+ecols].rename(columns={c:'E_'+c for c in ecols}).copy()
        mavg=m.groupby('sample_id',sort=False)[mcols].mean()
        mb=pd.concat(
            [
                mavg.index.to_frame(index=False),
                pd.DataFrame(mavg.to_numpy(), columns=[f'M_{c}' for c in mavg.columns]),
            ],
            axis=1,
        )
        lb=l[['sample_id']+lcols].rename(columns={c:'L_'+c for c in lcols}).copy()
        z=eb.merge(mb,on='sample_id',validate='one_to_one').merge(lb,on='sample_id',validate='one_to_one')
        z=pd.concat(
            [
                z.reset_index(drop=True),
                pd.DataFrame({'dataset_split':['external' if split=='test' else 'train']*len(z)}),
            ],
            axis=1,
        ).copy()
        split_rows.append(z)
        manifests[split]={"E_file":str(dirs['E']/names[split]['E']),"M_file":str(dirs['M']/names[split]['M']),"L_file":str(dirs['L']/names[split]['L']),"n":len(z)}
    df=pd.concat(split_rows,ignore_index=True).copy()
    for c in ['disease_label','label_early','label_middle','label_late']:
        df[c]=pd.to_numeric(df[c],errors='raise').astype(int)
    # Preserve high-MMSE AD as a distinct audit stratum; it is folded with early at Gate E only.
    def stage(r):
        if r.disease_label==0: return 'control'
        if r.label_late==1: return 'late'
        if r.label_middle==1: return 'middle'
        if r.label_early==1: return 'early'
        return 'AD_high_MMSE'
    df=df.assign(true_original_stage_label=df.apply(stage,axis=1))
    df=df.assign(true_collapsed_stage_label=df['true_original_stage_label'].replace({'early':'early_spectrum_AD','AD_high_MMSE':'early_spectrum_AD'})).copy()
    cols={'E':[c for c in df if c.startswith('E_')], 'M':[c for c in df if c.startswith('M_')], 'L':[c for c in df if c.startswith('L_')]}
    return df,cols,manifests

def _block_cols(cols:dict[str,list[str]], block:str)->list[str]:
    mapping={'E':['E'],'M':['M'],'L':['L'],'EM':['E','M'],'ML':['M','L'],'EML':['E','M','L']}
    return sum((cols[k] for k in mapping[block]),[])

def _binary_metrics(y:np.ndarray, p:np.ndarray, threshold:float=THRESHOLD)->dict[str,float]:
    pred=(p>=threshold).astype(int)
    out={'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'sensitivity':recall_score(y,pred,zero_division=0),'specificity':recall_score(y,pred,pos_label=0,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'precision':precision_score(y,pred,zero_division=0),'mcc':matthews_corrcoef(y,pred) if len(np.unique(pred))>1 else 0.0}
    out['roc_auc']=roc_auc_score(y,p) if len(np.unique(y))==2 else float('nan')
    out['pr_auc']=average_precision_score(y,p) if len(np.unique(y))==2 else float('nan')
    out['brier']=brier_score_loss(y,p)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); out.update({'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
    return out

def _gate_data(train: pd.DataFrame, gate: str)->tuple[pd.DataFrame,np.ndarray]:
    if gate=='late': d=train.copy(); y=(d.true_original_stage_label=='late').astype(int).to_numpy()
    elif gate=='middle': d=train.loc[train.true_original_stage_label!='late'].copy(); y=(d.true_original_stage_label=='middle').astype(int).to_numpy()
    elif gate=='early': d=train.loc[~train.true_original_stage_label.isin(['late','middle'])].copy(); y=(d.true_original_stage_label!='control').astype(int).to_numpy()
    else: raise ValueError(gate)
    return d,y

def _gate_defs(gate:str)->list[ModelDef]:
    # deliberately bounded panel, preserving the requested linear/non-linear alternatives
    if gate=='late':
        ds=_model_defs(['L','ML','EML']); return [d for d in ds if d.key in {'L__lr__l2','ML__lr__l2','ML__svc__linear','EML__lr__elasticnet','ML__svc__rbf'}]
    if gate=='middle':
        ds=_model_defs(['M','EM','EML']); return [d for d in ds if d.key in {'M__lr__l2','EM__lr__elasticnet','EM__svc__poly2','EM__svc__poly3','EM__svc__rbf','EML__svc__poly3'}]
    if gate=='early':
        ds=_model_defs(['E','EM'],include_mlp=True); return [d for d in ds if d.key in {'E__lr__l2','EM__lr__elasticnet','EM__svc__linear','EM__svc__poly2','EM__svc__poly3','EM__svc__rbf','EM__mlp__small'}]
    raise ValueError(gate)

def _fit_gate(train:pd.DataFrame, cols:dict[str,list[str]], gate:str, n_jobs:int, model_dir:Path)->tuple[dict[str,Any],pd.DataFrame,dict[str,Any]]:
    d,y=_gate_data(train,gate); cv=StratifiedKFold(n_splits=CV_SPLITS,shuffle=True,random_state=RANDOM_STATE)
    rows=[]; fitted={}
    for spec in _gate_defs(gate):
        X=d[_block_cols(cols,spec.feature_block)]
        search=GridSearchCV(spec.estimator,spec.grid,scoring='balanced_accuracy',cv=cv,n_jobs=n_jobs,refit=True,return_train_score=False,error_score='raise')
        search.fit(X,y)
        # Ranking uses the pre-specified 10-fold CV scoring result. We avoid a second full OOF pass,
        # which would duplicate the computational burden without changing component selection.
        row={'gate':gate,'model_key':spec.key,'family':spec.family,'feature_block':spec.feature_block,'n_train':len(d),'n_positive':int(y.sum()),'cv_best_balanced_accuracy':float(search.best_score_),'best_params':json.dumps(search.best_params_,default=_json)}
        rows.append(row); fitted[spec.key]={'estimator':search.best_estimator_,'feature_block':spec.feature_block,'family':spec.family,'best_params':search.best_params_}
        joblib.dump(fitted[spec.key],model_dir/f'{gate}__{spec.key}.joblib')
    rank=pd.DataFrame(rows).sort_values(['cv_best_balanced_accuracy'],ascending=False,na_position='last').reset_index(drop=True)
    return fitted,rank,{'n_train':len(d),'n_positive':int(y.sum()),'positive_definition': {'late':'Late','middle':'Middle among non-late','early':'Early + AD_high_MMSE among non-late/non-middle'}[gate]}

def _predict_gate(model:dict[str,Any], df:pd.DataFrame, cols:dict[str,list[str]])->np.ndarray:
    """Return a monotone gate score in [0,1]. LR yields probabilities; SVC decision values use a sigmoid score."""
    est=model['estimator']; X=df[_block_cols(cols,model['feature_block'])]
    if hasattr(est, 'predict_proba'):
        return est.predict_proba(X)[:,1]
    z=est.decision_function(X)
    return 1.0/(1.0+np.exp(-np.clip(z,-30,30)))

def _cascade_predict(models:dict[str,dict[str,Any]], df:pd.DataFrame, cols:dict[str,list[str]])->pd.DataFrame:
    pL=_predict_gate(models['late'],df,cols); pM=_predict_gate(models['middle'],df,cols); pE=_predict_gate(models['early'],df,cols)
    late=pL>=THRESHOLD; middle=(~late)&(pM>=THRESHOLD); early=(~late)&(~middle)&(pE>=THRESHOLD)
    stage=np.where(late,'late',np.where(middle,'middle',np.where(early,'early_spectrum_AD','control')))
    path=np.where(late,'late_positive',np.where(middle,'late_negative__middle_positive',np.where(early,'late_negative__middle_negative__early_spectrum_positive','late_negative__middle_negative__control')))
    o=df[['sample_id','dataset_split','disease_label','mmse','true_original_stage_label','true_collapsed_stage_label']].copy()
    o['late_gate_score']=pL; o['middle_gate_score_given_nonlate']=pM; o['early_spectrum_gate_score_given_nonlate_nonmiddle']=pE; o['late_gate_prediction']=late.astype(int); o['middle_gate_prediction']=middle.astype(int); o['early_spectrum_gate_prediction']=early.astype(int); o['predicted_stage']=stage; o['predicted_AD']=(stage!='control').astype(int); o['decision_path']=path
    o['binary_correct']=(o.predicted_AD==o.disease_label).astype(int); o['stage_correct_collapsed']=(o.predicted_stage==o.true_collapsed_stage_label).astype(int); o['stage_correct_strict']=(o.predicted_stage==o.true_original_stage_label).astype(int)
    o['ad_path_score']=pL+(1-pL)*pM+(1-pL)*(1-pM)*pE
    return o

def _multi_metrics(y:Iterable[str], pred:Iterable[str])->dict[str,float]:
    y=np.asarray(list(y)); pred=np.asarray(list(pred));
    return {'stage_accuracy':accuracy_score(y,pred),'stage_balanced_accuracy':balanced_accuracy_score(y,pred),'stage_macro_f1':f1_score(y,pred,average='macro',zero_division=0),'stage_weighted_f1':f1_score(y,pred,average='weighted',zero_division=0)}

def _evaluate_cascade(name:str, pred:pd.DataFrame)->dict[str,Any]:
    b=_binary_metrics(pred.disease_label.to_numpy(),pred.predicted_AD.to_numpy(dtype=float))
    s=_multi_metrics(pred.true_collapsed_stage_label,pred.predicted_stage)
    return {'cascade_id':name,**{f'binary_{k}':v for k,v in b.items()},**s}

def _bootstrap(pred:pd.DataFrame, n:int)->pd.DataFrame:
    rng=np.random.default_rng(RANDOM_STATE); rows=[]; nrow=len(pred)
    for i in range(n):
        x=pred.iloc[rng.integers(0,nrow,nrow)]
        r={'replicate':i,**_binary_metrics(x.disease_label.to_numpy(),x.predicted_AD.to_numpy(dtype=float)),**_multi_metrics(x.true_collapsed_stage_label,x.predicted_stage)}; rows.append(r)
    d=pd.DataFrame(rows); out=[]
    for c in d.columns:
        if c=='replicate': continue
        out.append({'metric':c,'estimate':float((_binary_metrics(pred.disease_label.to_numpy(),pred.predicted_AD.to_numpy(dtype=float)).get(c,_multi_metrics(pred.true_collapsed_stage_label,pred.predicted_stage).get(c,float('nan'))))),'ci_low':float(d[c].quantile(.025)),'ci_high':float(d[c].quantile(.975)),'bootstrap_n':n})
    return pd.DataFrame(out)

def _make_cascades(gate_models:dict[str,dict[str,Any]])->dict[str,dict[str,str]]:
    """Return 20 pre-specified, distinct cascade systems.

    The list is fixed before external evaluation.  C06 is the primary cascade.
    Component keys are verified at construction time so missing gate candidates fail
    early with a readable error rather than producing an opaque KeyError later.
    """
    specs = {
        'C01': {'late':'L__lr__l2',        'middle':'M__lr__l2',            'early':'E__lr__l2'},
        'C02': {'late':'L__lr__l2',        'middle':'EM__lr__elasticnet',  'early':'EM__lr__elasticnet'},
        'C03': {'late':'L__lr__l2',        'middle':'EM__svc__poly2',      'early':'EM__svc__poly2'},
        'C04': {'late':'L__lr__l2',        'middle':'EM__svc__poly3',      'early':'EM__svc__poly3'},
        'C05': {'late':'ML__lr__l2',       'middle':'EM__lr__elasticnet',  'early':'EM__lr__elasticnet'},
        'C06': {'late':'ML__lr__l2',       'middle':'EM__svc__poly3',      'early':'EM__svc__poly3'},
        'C07': {'late':'ML__svc__linear',  'middle':'EM__lr__elasticnet',  'early':'EM__svc__linear'},
        'C08': {'late':'L__lr__l2',        'middle':'M__lr__l2',           'early':'EM__lr__elasticnet'},
        'C09': {'late':'ML__lr__l2',       'middle':'EM__lr__elasticnet',  'early':'EM__svc__linear'},
        'C10': {'late':'EML__lr__elasticnet','middle':'EM__lr__elasticnet', 'early':'EM__svc__linear'},
        'C11': {'late':'ML__lr__l2',       'middle':'EM__svc__poly2',      'early':'EM__svc__poly2'},
        'C12': {'late':'ML__svc__linear',  'middle':'EM__svc__poly3',      'early':'EM__svc__poly3'},
        'C13': {'late':'ML__svc__linear',  'middle':'EM__svc__poly2',      'early':'EM__svc__poly3'},
        'C14': {'late':'EML__lr__elasticnet','middle':'EM__svc__poly3',     'early':'EM__svc__poly3'},
        'C15': {'late':'ML__lr__l2',       'middle':'EM__svc__rbf',        'early':'EM__svc__rbf'},
        'C16': {'late':'ML__svc__rbf',     'middle':'EM__svc__rbf',        'early':'EM__svc__rbf'},
        'C17': {'late':'ML__svc__rbf',     'middle':'EM__svc__poly3',      'early':'EM__svc__poly3'},
        'C18': {'late':'EML__lr__elasticnet','middle':'EML__svc__poly3',    'early':'EM__svc__poly3'},
        'C19': {'late':'ML__lr__l2',       'middle':'EML__svc__poly3',     'early':'EM__lr__elasticnet'},
        'C20': {'late':'ML__lr__l2',       'middle':'EM__svc__poly3',      'early':'EM__mlp__small'},
    }
    missing = {
        cascade_id: {gate: key for gate, key in components.items() if key not in gate_models.get(gate, {})}
        for cascade_id, components in specs.items()
    }
    missing = {cascade_id: values for cascade_id, values in missing.items() if values}
    if missing:
        raise KeyError({'missing_predefined_cascade_components': missing})
    return specs

def _flat_baseline_defs() -> dict[str,tuple[str,Any,dict[str,list[Any]]]]:
    return {
'B1_multinomial_lr_EML':('EML',_pipe(_multiclass_lr()),{'clf__C':[0.1,1.0],'clf__class_weight':['balanced']}),
'B2_multinomial_lr_EM':('EM',_pipe(_multiclass_lr()),{'clf__C':[0.1,1.0],'clf__class_weight':['balanced']}),
'B3_svc_rbf_EML':('EML',_pipe(SVC(kernel='rbf',probability=False,decision_function_shape='ovo',random_state=RANDOM_STATE)),{'clf__C':[0.1,1.0],'clf__gamma':['scale'],'clf__class_weight':['balanced']}),
'B4_svc_poly3_EM':('EM',_pipe(SVC(kernel='poly',degree=3,probability=False,decision_function_shape='ovo',random_state=RANDOM_STATE)),{'clf__C':[0.1,1.0],'clf__gamma':['scale'],'clf__coef0':[1.0],'clf__class_weight':['balanced']}),
}

def _fit_multiclass(train:pd.DataFrame,test:pd.DataFrame,cols:dict[str,list[str]],n_jobs:int,model_dir:Path)->tuple[pd.DataFrame,dict[str,pd.DataFrame]]:
    y=train.true_collapsed_stage_label.to_numpy(); out=[]; preds={}
    bases=_flat_baseline_defs()
    cv=StratifiedKFold(n_splits=CV_SPLITS,shuffle=True,random_state=RANDOM_STATE)
    for name,(block,est,grid) in bases.items():
        X=train[_block_cols(cols,block)]; search=GridSearchCV(est,grid,scoring='balanced_accuracy',cv=cv,n_jobs=n_jobs,refit=True,error_score='raise'); search.fit(X,y)
        pp=search.predict(test[_block_cols(cols,block)]); pa=(pp!='control').astype(int)
        p=test[['sample_id','dataset_split','disease_label','mmse','true_original_stage_label','true_collapsed_stage_label']].copy();p['predicted_stage']=pp;p['predicted_AD']=pa
        met={**_binary_metrics(p.disease_label.to_numpy(),pa.astype(float)),**_multi_metrics(p.true_collapsed_stage_label,pp)}
        out.append({'baseline_id':name,'feature_block':block,'cv_best_balanced_accuracy':float(search.best_score_),'best_params':json.dumps(search.best_params_,default=_json),**met})
        preds[name]=p; joblib.dump({'estimator':search.best_estimator_,'feature_block':block,'best_params':search.best_params_},model_dir/f'{name}.joblib')
    return pd.DataFrame(out),preds

def _write_md(path:Path, title:str, lines:list[str])->None:
    path.write_text('# '+title+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')

def _write_figures(final: Path, gate_rank: pd.DataFrame, cascade_rank: pd.DataFrame, primary: pd.DataFrame, ci: pd.DataFrame) -> None:
    """Create stagev5-style result figures from saved stagev7 tables only."""
    import matplotlib.pyplot as plt

    figdir=final/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    # Fig 1: best CV model per gate.
    best=gate_rank.sort_values('cv_best_balanced_accuracy',ascending=False).groupby('gate',as_index=False).first()
    plt.figure(figsize=(7,4))
    plt.bar(best['gate'],best['cv_best_balanced_accuracy'])
    plt.ylim(0,1); plt.ylabel('10-fold CV balanced accuracy'); plt.title('Best internal model by cascade gate')
    for i,v in enumerate(best['cv_best_balanced_accuracy']): plt.text(i,v+0.02,f'{v:.3f}',ha='center',fontsize=9)
    plt.tight_layout(); plt.savefig(figdir/'fig01_gate_best_cv_balanced_accuracy.png',dpi=180); plt.close()

    # Fig 2: predefined external cascade comparison; explicitly exploratory.
    top=cascade_rank.head(12).sort_values('binary_balanced_accuracy',ascending=True)
    plt.figure(figsize=(8,5))
    plt.barh(top['cascade_id'],top['binary_balanced_accuracy'])
    plt.xlim(0,1); plt.xlabel('External binary balanced accuracy'); plt.title('Predefined cascade comparison (exploratory)')
    plt.tight_layout(); plt.savefig(figdir/'fig02_cascade_external_comparison_exploratory.png',dpi=180); plt.close()

    # Fig 3: collapsed stage confusion matrix.
    labels=STAGE_ORDER
    cm=confusion_matrix(primary['true_collapsed_stage_label'],primary['predicted_stage'],labels=labels)
    plt.figure(figsize=(6,5)); plt.imshow(cm)
    plt.xticks(range(len(labels)),labels,rotation=30,ha='right'); plt.yticks(range(len(labels)),labels)
    plt.xlabel('Predicted stage'); plt.ylabel('True collapsed stage'); plt.title('Primary C06 collapsed-stage confusion matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j,i,str(cm[i,j]),ha='center',va='center')
    plt.tight_layout(); plt.savefig(figdir/'fig03_primary_stage_confusion_matrix.png',dpi=180); plt.close()

    # Fig 4: subgroup correctness.
    grp=primary.groupby('true_original_stage_label',dropna=False).agg(binary_accuracy=('binary_correct','mean'),collapsed_stage_accuracy=('stage_correct_collapsed','mean')).reset_index()
    x=np.arange(len(grp)); width=.36
    plt.figure(figsize=(8,4.5)); plt.bar(x-width/2,grp['binary_accuracy'],width,label='Binary accuracy'); plt.bar(x+width/2,grp['collapsed_stage_accuracy'],width,label='Collapsed-stage accuracy')
    plt.ylim(0,1); plt.xticks(x,grp['true_original_stage_label'],rotation=25,ha='right'); plt.ylabel('Accuracy'); plt.title('Primary C06 subgroup performance'); plt.legend()
    plt.tight_layout(); plt.savefig(figdir/'fig04_primary_stage_subgroup_accuracy.png',dpi=180); plt.close()

    # Fig 5: primary bootstrap CIs, selected principal metrics.
    wanted=['accuracy','balanced_accuracy','f1','stage_accuracy','stage_macro_f1']
    q=ci.loc[ci.metric.isin(wanted)].copy()
    if not q.empty:
        q=q.set_index('metric').reindex([m for m in wanted if m in set(q.metric)]).reset_index()
        y=np.arange(len(q)); err=np.vstack([q['estimate']-q['ci_low'],q['ci_high']-q['estimate']])
        plt.figure(figsize=(7,4)); plt.errorbar(q['estimate'],y,xerr=err,fmt='o',capsize=3)
        plt.xlim(0,1); plt.yticks(y,q['metric']); plt.xlabel('Metric estimate with bootstrap 95% CI'); plt.title('Primary C06 external uncertainty')
        plt.tight_layout(); plt.savefig(figdir/'fig05_primary_bootstrap_ci.png',dpi=180); plt.close()

def _completion_sentinel(final: Path) -> Path:
    return final / COMPLETION_SENTINEL_NAME

def _known_final_paths(final: Path) -> list[Path]:
    return [final / name for name in KNOWN_STAGEV7_FINAL_OUTPUTS] + [final / name for name in KNOWN_STAGEV7_FIGURES]

def _cleanup_known_final_outputs(final: Path) -> None:
    for path in _known_final_paths(final):
        if path.exists():
            if path.is_dir():
                raise IsADirectoryError(f'Refusing to remove directory listed as a file: {path}')
            path.unlink()

def _assert_output_dir_writable(final: Path) -> None:
    final.mkdir(parents=True,exist_ok=True)
    probe=final/'.stagev7_write_probe.tmp'
    probe.write_text('ok',encoding='utf-8')
    probe.unlink()

def self_check(root:Path)->None:
    req=[root/'input/raw/ad_s2t_wav2vec.csv',root/'input/raw/control_s2t_wav2vec.csv',root/'input/raw/test_s2t_wav2vec.csv',root/'output/features/E/raw_stagev2/ad_BM25.csv',root/'output/features/M/raw_stagev2/ad_embedding.csv',root/'output/features/L/raw_stagev4/ad_LLM.csv']
    missing=[str(p.relative_to(root)) for p in req if not p.exists()]
    if missing: raise FileNotFoundError({'missing_required_files':missing})
    df,cols,manifest=_load_features(root)
    train=df.query("dataset_split=='train'"); test=df.query("dataset_split=='external'")
    report={'status':'pass','created_at':_utc(),'n_train':len(train),'n_external':len(test),'feature_counts':{k:len(v) for k,v in cols.items()},'train_original_stage_counts':train.true_original_stage_label.value_counts().to_dict(),'external_original_stage_counts':test.true_original_stage_label.value_counts().to_dict(),'no_api_or_feature_extraction':True,'feature_files':manifest}
    out=root/'output/checks';out.mkdir(parents=True,exist_ok=True);(out/'stagev7_self_check.json').write_text(json.dumps(report,indent=2,default=_json),encoding='utf-8');print(json.dumps(report,indent=2,default=_json))

def train_preflight(root:Path)->None:
    final=root/'output/final_report'
    sentinel=_completion_sentinel(final).resolve()
    df,cols,manifest=_load_features(root)
    train=df.query("dataset_split=='train'")
    external=df.query("dataset_split=='external'")
    if len(train)!=166 or len(external)!=71:
        raise ValueError({'unexpected_sample_counts':{'n_train':len(train),'n_external':len(external)},'expected':{'n_train':166,'n_external':71}})
    feature_counts={k:len(v) for k,v in cols.items()}
    if feature_counts!={'E':61,'M':1024,'L':8}:
        raise ValueError({'unexpected_feature_counts':feature_counts,'expected':{'E':61,'M':1024,'L':8}})
    available={gate:{spec.key:{} for spec in _gate_defs(gate)} for gate in ['late','middle','early']}
    cascades=_make_cascades(available)
    baselines=_flat_baseline_defs()
    _assert_output_dir_writable(final)
    completed=sentinel.exists()
    report={
        'status':'pass',
        'created_at':_utc(),
        'would_allow_training':not completed,
        'completed_run_detected':completed,
        'completion_sentinel':str(sentinel),
        'n_train':len(train),
        'n_external':len(external),
        'feature_counts':feature_counts,
        'train_original_stage_counts':train.true_original_stage_label.value_counts().to_dict(),
        'external_original_stage_counts':external.true_original_stage_label.value_counts().to_dict(),
        'n_cascades':len(cascades),
        'n_flat_baselines':len(baselines),
        'output_dir_writable':True,
        'default_train_block_reason':'completed sentinel exists' if completed else None,
        'feature_files':manifest,
    }
    print(json.dumps(report,indent=2,default=_json))

def run_training(root:Path,n_jobs:int,bootstrap_n:int,force:bool=False)->None:
    final=root/'output/final_report'; models=root/'output/stagev7_models';
    sentinel=_completion_sentinel(final).resolve()
    if sentinel.exists() and not force:
        raise FileExistsError(f'Completed stagev7 training run detected at {sentinel}. Re-run with --force to overwrite known stagev7 final report outputs.')
    final.mkdir(parents=True,exist_ok=True);models.mkdir(parents=True,exist_ok=True)
    _cleanup_known_final_outputs(final)
    df,cols,manifest=_load_features(root); train=df.query("dataset_split=='train'").reset_index(drop=True); external=df.query("dataset_split=='external'").reset_index(drop=True)
    gate_models={}; ranks=[]; gate_meta={}
    for gate in ['late','middle','early']:
        gm,rank,meta=_fit_gate(train,cols,gate,n_jobs,models);gate_models[gate]=gm;ranks.append(rank);gate_meta[gate]=meta;rank.to_csv(final/f'stagev7_{gate}_gate_model_ranking.csv',index=False)
    pd.concat(ranks,ignore_index=True).to_csv(final/'stagev7_gate_model_ranking_all.csv',index=False)
    specs=_make_cascades(gate_models); cascade_rows=[]; all_pred=[]; cascade_models={}
    for cid,keys in specs.items():
        chosen={g:gate_models[g][k] for g,k in keys.items()}; pred=_cascade_predict(chosen,external,cols);pred.insert(0,'cascade_id',cid);met=_evaluate_cascade(cid,pred);met.update({f'{g}_component':k for g,k in keys.items()});cascade_rows.append(met);all_pred.append(pred);cascade_models[cid]={'components':keys}
        joblib.dump({'components':chosen,'component_keys':keys,'threshold':THRESHOLD},models/f'{cid}.joblib')
    rank=pd.DataFrame(cascade_rows).sort_values(['binary_balanced_accuracy','stage_macro_f1','binary_accuracy'],ascending=False).reset_index(drop=True)
    rank.to_csv(final/'stagev7_cascade_ranking_external_exploratory.csv',index=False)
    pd.concat(all_pred,ignore_index=True).to_csv(final/'stagev7_final_cascade_predictions_all.csv',index=False)
    # primary pre-specified cascade C06: main report; external ranking is clearly exploratory only
    primary=pd.concat(all_pred,ignore_index=True).query("cascade_id=='C06'").copy();primary.to_csv(final/'stagev7_final_cascade_predictions.csv',index=False)
    bmet=_binary_metrics(primary.disease_label.to_numpy(),primary.predicted_AD.to_numpy(dtype=float)); smet=_multi_metrics(primary.true_collapsed_stage_label,primary.predicted_stage)
    pd.DataFrame([{**bmet,**smet}]).to_csv(final/'stagev7_final_primary_performance.csv',index=False)
    pd.crosstab(primary.true_collapsed_stage_label,primary.predicted_stage,dropna=False).reindex(index=STAGE_ORDER,columns=STAGE_ORDER,fill_value=0).to_csv(final/'stagev7_final_confusion_matrix_stage_collapsed.csv')
    pd.crosstab(primary.disease_label,primary.predicted_AD,dropna=False).reindex(index=[0,1],columns=[0,1],fill_value=0).to_csv(final/'stagev7_final_confusion_matrix_binary.csv')
    primary.groupby(['true_original_stage_label','predicted_stage'],dropna=False).size().reset_index(name='n').to_csv(final/'stagev7_final_stage_strict_audit.csv',index=False)
    primary.groupby('decision_path',dropna=False).agg(n=('sample_id','size'),binary_accuracy=('binary_correct','mean'),stage_accuracy=('stage_correct_collapsed','mean')).reset_index().to_csv(final/'stagev7_gate_path_audit.csv',index=False)
    errors=primary.loc[(primary.binary_correct==0)|(primary.stage_correct_collapsed==0)].copy();errors.to_csv(final/'stagev7_cascade_error_analysis.csv',index=False)
    ci=_bootstrap(primary,bootstrap_n);ci.to_csv(final/'stagev7_bootstrap_ci.csv',index=False)
    base_rank,base_preds=_fit_multiclass(train,external,cols,n_jobs,models);base_rank.to_csv(final/'stagev7_flat_multiclass_baseline_performance.csv',index=False)
    pd.concat([p.assign(baseline_id=k) for k,p in base_preds.items()],ignore_index=True).to_csv(final/'stagev7_flat_multiclass_baseline_predictions.csv',index=False)
    _write_figures(final, pd.concat(ranks,ignore_index=True), rank, primary, ci)
    leak={'external_used_for_training_or_hyperparameter_selection':False,'external_used_for_exploratory_evaluation_of_predefined_cascades':True,'primary_cascade':'C06','primary_cascade_pre_specified':True,'feature_extraction_re_run':False,'api_calls':False,'mmse_used_as_model_feature':False,'feature_source':'strictly stagev5 generated E/M/L CSVs'}
    (final/'stagev7_leakage_check.json').write_text(json.dumps(leak,indent=2),encoding='utf-8')
    feature_manifest={'created_at':_utc(),'feature_counts':{k:len(v) for k,v in cols.items()},'feature_files':manifest,'E_policy':'stagev5 copied stagev2 BM25 features','M_policy':'stagev5 copied stagev2 BGE-M3 window embeddings aggregated by sample mean','L_policy':'stagev5 copied stagev4 unmasked raw F8 only','no_new_extraction':True}
    (final/'stagev7_feature_source_manifest.json').write_text(json.dumps(feature_manifest,indent=2),encoding='utf-8')
    summary={'created_at':_utc(),'primary_cascade':'C06','primary_components':specs['C06'],'primary_external_metrics':{**bmet,**smet},'gate_metadata':gate_meta,'n_predefined_cascades':len(specs),'bootstrap_n':bootstrap_n,'n_jobs':n_jobs,'flat_baseline_count':len(base_rank),'external_ranking_is_exploratory':True}
    (final/'stagev7_final_run_summary.json').write_text(json.dumps(summary,indent=2,default=_json),encoding='utf-8')
    _write_md(final/'stagev7_selected_model_summary.md','Stagev7 selected primary cascade',[
'**Primary cascade (pre-specified):** C06.',
'**Late gate:** `ML__lr__l2`; **middle gate:** `EM__svc__poly3`; **early-spectrum gate:** `EM__svc__poly3`.',
'Late, middle, and early-spectrum predictions all map to AD=1; control maps to AD=0.',
'`AD_high_MMSE` is retained for strict audit but joins early at the final early-spectrum gate, not relabelled as strict early.',
'External ranking of all pre-specified cascades is exploratory and must not be reported as an external-test model-selection procedure.'
])
    _write_md(final/'stagev7_experiment_report.md','Stagev7 experiment report',[
'## Design',
'Stagev7 is an ordered three-gate cascade: late vs non-late; middle vs remaining non-late; early-spectrum AD vs control among remaining samples.',
'All model inputs are copied E/M/L feature CSVs generated in stagev5. The run does not call an API or regenerate features.',
'## Selection',
'Each gate searches its bounded model panel using 10-fold stratified CV and balanced accuracy. C06 is the primary pre-specified cascade. Twenty fixed cascades plus four flat multiclass baselines are evaluated on the held-out external data for comparison.',
'## Interpretation restriction',
'External results are evaluation outcomes. Rankings across predefined cascades are labelled exploratory because several fixed candidate systems are compared on the same held-out set.'
])
    completion={'status':'complete','created_at':_utc(),'n_train':len(train),'n_external':len(external),'feature_counts':{k:len(v) for k,v in cols.items()},'primary_cascade':'C06'}
    _completion_sentinel(final).write_text(json.dumps(completion,indent=2,default=_json),encoding='utf-8')
    print('STAGEV7 TRAINING AND REPORTING COMPLETED')
    print(json.dumps(summary,indent=2,default=_json))
