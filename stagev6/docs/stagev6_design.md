# Stagev6 fixed late-first cascade design

## Decision rule

For every sample, the late gate first estimates \(p_{late}\).  When \(p_{late}\ge0.50\), the sample is directly classified as AD.  Otherwise the non-late branch estimates \(p_{AD\mid nonlate}\) and classifies AD at 0.50.

\[
\hat y_{AD}=\begin{cases}
1,&p_{late}\ge0.50\\
\mathbb{1}\{p_{AD\mid nonlate}\ge0.50\},&p_{late}<0.50.
\end{cases}
\]

The stored continuous quantity

\[
p_{AD,mixture}=p_{late}+(1-p_{late})p_{AD\mid nonlate}
\]

is used only for probability-based metrics. It never replaces the hard routing rule.

## Fixed component panel

- Gates: L + L2 LR; M+L + L2 LR; M+L + poly-3 SVC.
- Branches: E+M + poly-3 SVC; E+M + L2 LR.
- Complete cascades: 3 × 2 = 6.

## Training protocol

Every component uses `GridSearchCV` and 10-fold `RepeatedStratifiedKFold` with one repeat, mirroring Stagev5. Gate model selection uses balanced accuracy because late is the minority class. The branch is fitted only on true non-late training samples and selected by accuracy. Cascade OOF predictions use shared folds stratified by `control`, `nonlate_AD`, and `late_AD`.

## Scope

No Stagev5 flat baseline is rerun or copied into the ranking. Stagev6 output contains only the six cascade specifications.
