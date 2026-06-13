from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config


def _save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, top_n: int = 15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = df.sort_values(y, ascending=True).tail(top_n)
    plt.figure(figsize=(10, max(5, 0.35 * len(d))))
    plt.barh(d[x].astype(str), d[y].astype(float))
    plt.title(title)
    plt.xlabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_figures(final_dir: Path = config.FINAL_REPORT_DIR) -> dict[str, str]:
    figs = {}
    figdir = final_dir / "figures"
    main_p = final_dir / "seed2026_main_results.csv"
    stab_p = final_dir / "seed_stability_summary.csv"
    egain_p = final_dir / "earlyv1_gain_over_earlyv0.csv"
    sgain_seed_p = final_dir / "scale_gain_seed2026.csv"
    sgain_p = final_dir / "scale_gain_stability.csv"
    if main_p.exists():
        main = pd.read_csv(main_p)
        main["label"] = main["early_variant"] + " | " + main["model_spec_id"]
        p = figdir / "top_seed2026_external_accuracy.png"
        _save_bar(main, "label", "external_accuracy", "Top seed=2026 external accuracy", p)
        figs["top_seed2026_external_accuracy"] = str(p)
    if stab_p.exists():
        stab = pd.read_csv(stab_p)
        stab["label"] = stab["early_variant"] + " | " + stab["model_spec_id"]
        p = figdir / "top_stability_external_accuracy_ci.png"
        _save_bar(stab, "label", "external_accuracy_mean", "Top stability external accuracy mean", p)
        figs["top_stability_external_accuracy_ci"] = str(p)
    if egain_p.exists():
        eg = pd.read_csv(egain_p)
        eg["label"] = eg["model_spec_id"]
        p = figdir / "earlyv1_gain_over_earlyv0.png"
        _save_bar(eg, "label", "external_accuracy_delta", "earlyv1 gain over earlyv0", p)
        figs["earlyv1_gain_over_earlyv0"] = str(p)
    if sgain_seed_p.exists():
        sg = pd.read_csv(sgain_seed_p)
        if not sg.empty:
            sg["label"] = sg["early_variant"] + " | " + sg["raw_feature_block"] + "->" + sg["scale_feature_block"] + " | " + sg["model_variant"]
            p = figdir / "scale_gain_seed2026.png"
            _save_bar(sg, "label", "scale_gain_delta", "Scale gain under seed=2026", p)
            figs["scale_gain_seed2026"] = str(p)
    if sgain_p.exists():
        sg = pd.read_csv(sgain_p)
        if not sg.empty:
            sg["label"] = sg["early_variant"] + " | " + sg["raw_feature_block"] + "->" + sg["scale_feature_block"] + " | " + sg["model_variant"]
            p = figdir / "scale_gain_stability.png"
            _save_bar(sg, "label", "scale_gain_mean_delta", "Scale gain stability mean delta", p)
            figs["scale_gain_stability"] = str(p)
    return figs
