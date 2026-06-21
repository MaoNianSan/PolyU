# stagev6

`stagev6` is a late-first cascade for Cookie Theft AD-versus-control classification.

## Locked inputs

Stagev6 **does not extract any features and does not call an API**. It reads the existing Stagev5 artifacts directly:

| Family | Existing source artifact | Stagev6 use |
|---|---|---|
| E | `output/features/E/raw_stagev2/` | Non-late AD/control branch only |
| M | `output/features/M/raw_stagev2/` | Gate and branch; BGE-M3 window rows are aggregated by `sample_id` mean |
| L | `output/features/L/raw_stagev4/` | Late gate only; strict raw P4/F8 columns |

The loader verifies feature dimensions, sample IDs, fixed mutually exclusive stage labels, and expected Stagev5 sample counts before training.

## Model structure

\[
\text{Late gate} \rightarrow
\begin{cases}
\text{gate-positive} \Rightarrow \text{AD}\\
\text{gate-negative} \Rightarrow \text{non-late AD/control branch}
\end{cases}
\]

- Gate target: `late` versus `non-late`.
- Gate feature blocks: `L` and `M+L`.
- Branch population: only true non-late training samples (`control`, `early`, `middle`, and `AD_high_MMSE`).
- Branch feature block: `E+M`.
- A gate-positive sample is directly labelled AD.
- Gate/branch thresholds are both fixed at 0.50.
- Hard routing determines accuracy, sensitivity, specificity, F1, MCC, and the confusion matrix.
- \(p_{AD}=p_{late}+(1-p_{late})p_{AD\mid nonlate}\) is retained only for ROC-AUC, PR-AUC, Brier score, and log-loss.

## Stagev5-scale classifier family

The gate uses the Stagev5 LR/SVC family on both valid gate blocks:

- LR-L2, LR-L1, LR-elastic-net;
- linear SVC, polynomial SVC degree 2, polynomial SVC degree 3, RBF SVC, sigmoid SVC.

The branch uses the same LR/SVC family on `E+M` plus the Stagev5 small MLP anchor. This creates 16 gate components × 9 branch components = **144 cascade candidates**, without redundant refitting of the same components for every pair.

Component tuning uses the Stagev5 grids and `GridSearchCV` with repeated stratified 10-fold CV configured as 10×1. Gate tuning uses balanced accuracy because its positive class has only 16 training samples; branch tuning uses accuracy. Complete cascades are ranked by held-out **external test accuracy** (`external_test_accuracy`), matching Stagev5’s selection convention while making the split explicit in the outputs.

## Install

```powershell
cd stagev6
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python .\run_stagev6.py --mode self_check
python .\run_stagev6.py --mode train --n-jobs 12 --bootstrap-n 200
```

`--mode train` uses the prepackaged Stagev5 feature CSVs. It does not perform feature extraction, use an API key, or alter the original E/M/L files.

## Canonical outputs

After training, `output/final_report/` contains the Stagev6 model ranking by external test accuracy, external test performance report, CV summary, bootstrap CIs, route-aware OOF/external-test predictions, gate and branch component reports, route diagnostics, selected-model summary/report, feature-source audit, leakage audit, and seven pre-rendered PNG figures.

The notebook only reads saved outputs:

```powershell
python .\run_stagev6.py --mode render_notebook
```

## Interpretation boundary

The routing output is `control`, `non-late AD`, or `late-direct AD`. Stagev6 is not a supervised early/middle/late multiclass classifier.
