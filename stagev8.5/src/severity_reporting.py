from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg


def write_json(path: Path, obj: Any) -> None:
    def convert(x: Any):
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_figures(final: Path, nested_oof: pd.DataFrame, external: pd.DataFrame, calibration_t20: pd.DataFrame, calibration_t14: pd.DataFrame, abstention: pd.DataFrame, boot: pd.DataFrame) -> None:
    # 1. Nested OOF evidence: no external data used.
    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(nested_oof["mmse"], nested_oof["severity_score"], alpha=0.8)
    plt.xlabel("MMSE")
    plt.ylabel("Stagev8.5 severity score (0–2)")
    plt.title("Nested OOF severity score versus MMSE (AD training samples)")
    _savefig(final / "figures/fig01_nested_oof_severity_vs_mmse.png")

    # 2. External AD severity curve.
    ad = external[external["__y__"].eq(1)].copy()
    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(ad["mmse"], ad["severity_score"], alpha=0.85)
    plt.xlabel("MMSE")
    plt.ylabel("Stagev8.5 severity score (0–2)")
    plt.title("External AD severity score versus MMSE")
    _savefig(final / "figures/fig02_external_severity_vs_mmse.png")

    # 3. q distribution by predefined MMSE stratum.
    groups = cfg.SEVERITY_STRATA
    q_cols = ["q_high_mmse_given_AD", "q_intermediate_mmse_given_AD", "q_low_mmse_given_AD"]
    means = np.array([
        [ad.loc[ad["true_mmse_stratum"].eq(g), c].mean() for c in q_cols]
        for g in groups
    ])
    plt.figure(figsize=(8.2, 4.8))
    x = np.arange(len(groups)); bottom = np.zeros(len(groups))
    for i, name in enumerate(["q_high", "q_intermediate", "q_low"]):
        plt.bar(x, means[:, i], bottom=bottom, label=name)
        bottom += means[:, i]
    plt.xticks(x, groups, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Mean conditional probability")
    plt.title("External conditional severity probabilities by MMSE stratum")
    plt.legend()
    _savefig(final / "figures/fig03_external_stratum_probability_distribution.png")

    # 4. Calibration diagnostic for fixed threshold heads.
    plt.figure(figsize=(7.4, 4.8))
    for name, table in [("T20", calibration_t20), ("T14", calibration_t14)]:
        if not table.empty:
            plt.plot(table["mean_predicted_probability"], table["fraction_positive"], marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("External calibration diagnostics")
    plt.legend()
    _savefig(final / "figures/fig04_threshold_calibration.png")

    # 5. Abstention by true stratum for anchor-admitted AD samples.
    plt.figure(figsize=(7.8, 4.8))
    plt.bar(abstention["true_mmse_stratum"], abstention["abstention_rate"])
    plt.ylim(0, 1)
    plt.ylabel("Abstention rate")
    plt.title("Selective severity reporting by true MMSE stratum")
    plt.xticks(rotation=15, ha="right")
    _savefig(final / "figures/fig05_selective_coverage_by_stratum.png")

    # 6. Bootstrap CIs for selected metrics.
    plt.figure(figsize=(8.5, 4.8))
    keep = boot[boot["metric"].isin([
        "binary_accuracy", "severity_spearman_rho", "severity_pairwise_ordinal_accuracy",
        "T20_balanced_accuracy", "T14_balanced_accuracy", "three_strata_macro_f1_anchor_admitted",
    ])].copy().reset_index(drop=True)
    plotted_metrics: list[str] = []
    plotted_y: list[int] = []
    for row in keep.itertuples(index=False):
        estimate = float(row.estimate)
        low = float(row.ci_low)
        high = float(row.ci_high)
        if not (np.isfinite(estimate) and np.isfinite(low) and np.isfinite(high)):
            continue
        # Percentile bootstrap endpoints can fall on the other side of the point
        # estimate in tiny samples. Matplotlib requires non-negative magnitudes.
        lower = max(0.0, estimate - low)
        upper = max(0.0, high - estimate)
        y = len(plotted_metrics)
        plt.errorbar(estimate, y, xerr=[[lower], [upper]], fmt="o")
        plotted_metrics.append(str(row.metric))
        plotted_y.append(y)
    plt.yticks(plotted_y, plotted_metrics)
    plt.xlim(-1, 1)
    plt.xlabel("Estimate and 95% stratified bootstrap CI")
    plt.title("Stagev8.5 external reference metrics")
    _savefig(final / "figures/fig06_bootstrap_ci.png")


def literature_rationale() -> str:
    return """# Stagev8.5 MMSE label rationale

## Fixed MMSE-informed strata

Stagev8.5 uses a prespecified MMSE-informed ordinal severity contract:

- **high-MMSE AD:** MMSE >=21;
- **intermediate-MMSE AD:** MMSE 15–20;
- **low-MMSE AD:** MMSE <=14.

The contract is used for ordinal cognitive-severity analysis only. It is not asserted to be a clinical early/middle/late Alzheimer disease staging gold standard.

## Literature basis

1. Henneges C, Reed C, Chen Y-F, Dell'Agnello G, Lebrec J. *Describing the Sequence of Cognitive Decline in Alzheimer's Disease Patients: Results from an Observational Study.* **Journal of Alzheimer's Disease.** 2016;52(3):1065–1080. doi:10.3233/JAD-150852. The observational GERAS analysis recruited three AD severity strata using MMSE 21–26, 15–20, and 0–14. Stagev8.5 takes the 21 and 15/14 boundaries from that published three-band framework, with the top group extended to >=21 because this dataset contains AD cases above 26 and the goal is a complete AD severity partition.

2. Perneczky R, Wagenpfeil S, Komossa K, Grimmer T, Diehl J, Kurz A. *Mapping Scores Onto Stages: Mini-Mental State Examination and Clinical Dementia Rating.* **American Journal of Geriatric Psychiatry.** 2006;14(2):139–144. doi:10.1097/01.JGP.0000192478.82189.a8. The study reported MMSE ranges of 21–25 for mild, 11–20 for moderate, and 0–10 for severe dementia when mapped to CDR categories. It supports interpreting the MMSE boundaries as a cognitive-severity ordering, while also motivating the conservative terminology used here.

## Interpretation boundary

MMSE is an overall cognitive screening score. Therefore Stagev8.5 reports **MMSE-informed ordinal cognitive-severity tendency**, not clinical-stage diagnosis. The frozen Stagev5 AD/control anchor remains the only disease classification decision.
"""


def selected_summary(binary: dict[str, Any], ordinal: dict[str, Any], t20: dict[str, Any], t14: dict[str, Any], selected: dict[str, Any]) -> str:
    return f"""# Stagev8.5 selected model summary

## Fixed design

- **AD/control anchor:** frozen Stagev5 `early_middle__svc__poly3(E+M)`; no retraining.
- **T20 head:** `{selected['T20']['family']}` on E+M, estimating `P(MMSE <=20 | AD)`.
- **T14 head:** `{selected['T14']['family']}` on raw F8 L only, estimating `P(MMSE <=14 | MMSE <=20, AD)`.
- **Feature contract:** fresh Stagev5 E/M/L reconstruction followed by the byte-identical Stagev6 loader; E=61, M=1024, L=8.
- **External role:** reference evaluation only; no external model, cutoff, or threshold selection.

## External reference results

- Binary anchor accuracy: {binary['accuracy']:.4f}
- Binary anchor balanced accuracy: {binary['balanced_accuracy']:.4f}
- Severity Spearman rho with `30-MMSE` among true AD: {ordinal['spearman_rho']:.4f}
- Severity Kendall tau with `30-MMSE` among true AD: {ordinal['kendall_tau']:.4f}
- Pairwise ordinal accuracy among true AD: {ordinal['pairwise_ordinal_accuracy']:.4f}
- T20 balanced accuracy among true AD: {t20['balanced_accuracy']:.4f}
- T14 balanced accuracy among true AD with MMSE <=20: {t14['balanced_accuracy']:.4f}

## Interpretation boundary

Stagev8.5 estimates an MMSE-informed ordinal cognitive-severity tendency. It does not claim validated clinical early/middle/late AD staging.
"""


def experiment_report(selected: dict[str, Any]) -> str:
    return f"""# Stagev8.5 experiment report

## Objective

Stagev8.5 retains the frozen Stagev5 AD/control anchor and replaces the primary three-class stage claim with an MMSE-informed ordinal severity output. The prespecified thresholds are `MMSE <=20` and `MMSE <=14`, producing conditional probabilities:

`q_high = 1 - s20`

`q_intermediate = s20 × (1 - s14)`

`q_low = s20 × s14`

The continuous severity score is `0*q_high + 1*q_intermediate + 2*q_low`.

## Fixed feature and model boundary

- E: exact Stagev5 BM25, 61 dimensions.
- M: exact Stagev5 BGE-M3 window embeddings, 1024 dimensions after verbatim Stagev6 sample aggregation.
- L: fixed raw F8 late features, 8 dimensions.
- T20: E+M elastic-net logistic regression.
- T14: raw-F8-only L2 logistic regression.

No new late features, fusion features, text manipulation, feature compression, external selection, or anchor retraining is permitted.

## Internal validation

Hyperparameters are selected by 10-fold stratified training CV within each fixed family. Nested 10-fold OOF severity scores are also generated for a less optimistic training-only ordinal diagnostic. Thirty seed-specific 10-fold OOF audits assess stability after hyperparameters are fixed.

## External evaluation

The external set is reported as a reference evaluation because it has been inspected in prior stages. It is never used to choose thresholds, model family, feature blocks, or abstention settings.

## Output semantics

The frozen anchor produces `predicted_AD`. Severity scores are conditional on AD. A displayed stratum is withheld as `AD_severity_indeterminate` unless the prespecified confidence and margin rules are met.

## Interpretation

The output represents MMSE-informed ordinal cognitive-severity tendency, not clinical dementia-stage diagnosis.
"""
