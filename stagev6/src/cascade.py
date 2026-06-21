from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import clone
from .evaluation import get_positive_prob, metrics_from_hard_and_prob
from . import stagev6_config as cfg

@dataclass
class ComponentFit:
    spec: object
    best_estimator: object
    best_score: float
    best_params: dict
    fit_seconds: float
    oof_prob: np.ndarray | None = None
    component_cv_metrics: dict | None = None


def block_matrix(df: pd.DataFrame, schema: dict, block: str) -> pd.DataFrame:
    if block == "late": cols=schema["late_columns"]
    elif block == "middle_late": cols=schema["middle_columns"]+schema["late_columns"]
    elif block == "early_middle": cols=schema["early_columns"]+schema["middle_columns"]
    else: raise KeyError(block)
    return df[cols].copy()


def hard_route(p_late: np.ndarray, p_branch: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    p_late=np.asarray(p_late,float); p_branch=np.asarray(p_branch,float)
    routed=(p_late >= cfg.GATE_THRESHOLD)
    y_pred=np.where(routed, 1, (p_branch >= cfg.BRANCH_THRESHOLD).astype(int)).astype(int)
    p_mix=np.clip(p_late+(1-p_late)*p_branch,1e-6,1-1e-6)
    return y_pred, p_mix, routed.astype(int)


def oof_component_predictions(gate_estimator, branch_estimator, X_gate: pd.DataFrame, X_branch: pd.DataFrame, y_ad: np.ndarray, y_late: np.ndarray, cv):
    n=len(y_ad); pg=np.zeros(n); pb=np.zeros(n); counts=np.zeros(n,int)
    for tr,va in cv.split(X_gate, np.asarray(y_late,dtype=int)):
        g=clone(gate_estimator); b=clone(branch_estimator)
        g.fit(X_gate.iloc[tr], y_late[tr])
        nonlate_tr=tr[np.asarray(y_late)[tr] == 0]
        b.fit(X_branch.iloc[nonlate_tr], y_ad[nonlate_tr])
        pg[va]=get_positive_prob(g,X_gate.iloc[va])
        pb[va]=get_positive_prob(b,X_branch.iloc[va])
        counts[va]+=1
    if not np.all(counts==1):
        raise RuntimeError("OOF cascade coverage failure.")
    pred,pmix,route=hard_route(pg,pb)
    return {"p_late":pg,"p_ad_given_nonlate":pb,"p_ad_mixture":pmix,"y_pred":pred,"late_route":route,"metrics":metrics_from_hard_and_prob(y_ad,pred,pmix)}


def route_fields(df: pd.DataFrame, p_late, p_branch, p_mix, y_pred, late_route, model_name: str, gate_name: str, branch_name: str) -> pd.DataFrame:
    out=pd.DataFrame({
        "sample_id":df["__sample_id__"].astype(str).values,
        "y_true":df["__y__"].astype(int).values,
        "label_late":df["label_late"].astype(int).values,
        "label_early":df["label_early"].astype(int).values,
        "label_middle":df["label_middle"].astype(int).values,
        "severity_group":df["severity_group"].astype(str).values,
        "mmse":df["mmse"].values,
        "model_name":model_name,"gate_model_name":gate_name,"branch_model_name":branch_name,
        "p_late":np.asarray(p_late,float),"p_ad_given_nonlate":np.asarray(p_branch,float),"p_ad_mixture":np.asarray(p_mix,float),
        "late_route":np.asarray(late_route,int),"y_pred":np.asarray(y_pred,int),
    })
    out["final_route"]=np.where(out["late_route"].eq(1),"late_direct_AD",np.where(out["y_pred"].eq(1),"nonlate_branch_AD","nonlate_branch_control"))
    out["correct"]=(out.y_pred==out.y_true).astype(int)
    def _err(r):
        if r.y_true==1 and r.label_late==1:
            return "correct_late_direct_AD" if r.late_route==1 else ("late_missed_branch_recovered" if r.y_pred==1 else "late_missed_final_FN")
        if r.y_true==0:
            return "control_false_late_route" if r.late_route==1 else ("control_branch_FP" if r.y_pred==1 else "correct_control_branch")
        return "nonlate_AD_direct_late_route" if r.late_route==1 else ("correct_nonlate_AD_branch" if r.y_pred==1 else "nonlate_AD_branch_FN")
    out["route_error_type"]=out.apply(_err,axis=1)
    out["error_type"]=np.where(out.correct.eq(1),"correct",np.where(out.y_true.eq(0),"FP_normal","FN_"+out.severity_group.astype(str)))
    return out
