from __future__ import annotations
import numpy as np
import pandas as pd
from .evaluation import metrics_from_hard_and_prob


def stratified_bootstrap_ci(y_true, y_pred, p_ad, model_name: str, n_boot: int, random_state: int) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int); pred=np.asarray(y_pred, dtype=int); p=np.asarray(p_ad, dtype=float)
    rng=np.random.default_rng(random_state)
    i0=np.where(y==0)[0]; i1=np.where(y==1)[0]
    rows=[]
    for _ in range(n_boot):
        idx=np.concatenate([rng.choice(i0, len(i0), replace=True), rng.choice(i1, len(i1), replace=True)])
        rows.append(metrics_from_hard_and_prob(y[idx], pred[idx], p[idx]))
    frame=pd.DataFrame(rows); out=[]
    for metric in ["accuracy","balanced_accuracy","sensitivity","specificity","f1","roc_auc","pr_auc","mcc"]:
        vals=frame[metric].dropna().to_numpy()
        out.append({"model_name":model_name,"metric":metric,"bootstrap_mean":float(np.mean(vals)) if len(vals) else np.nan,
                    "ci_low":float(np.percentile(vals,2.5)) if len(vals) else np.nan,
                    "ci_high":float(np.percentile(vals,97.5)) if len(vals) else np.nan,"n_boot":n_boot})
    return pd.DataFrame(out)
