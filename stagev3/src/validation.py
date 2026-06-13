from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import config
from .models import model_specs
from .self_check import feature_manifest_status, raw_files_exist, token_count_excluded


def validate_outputs(
    mode: str,
    seeds: list[int],
    final_dir: Path = config.FINAL_REPORT_DIR,
    level: str = "standard",
    selected_model_count: int = 0,
    selected_paths: dict[str, Path] | None = None,
) -> tuple[bool, str]:
    lines = [f"# stagev3 validation report ({level})", ""]
    ok = True

    def check(condition: bool, message: str) -> None:
        nonlocal ok
        if condition:
            lines.append(f"- PASS: {message}")
        else:
            ok = False
            lines.append(f"- FAIL: {message}")

    feature_ok, feature_source = feature_manifest_status()
    check(raw_files_exist(), "raw files exist")
    check(bool(os.getenv(config.MAAS_API_KEY_ENV, "").strip()), "MAAS_API_KEY exists")
    check(feature_ok, f"feature manifest valid ({feature_source})")
    check(len(model_specs()) == 102, "model registry count = 102")
    check(config.EARLY_VARIANTS == ["earlyv0", "earlyv1"], "early variants = 2")
    check(token_count_excluded(), "token_count is excluded from model input")

    expected_main = config.expected_main_rows()
    main_p = final_dir / "seed2026_main_results.csv"
    stab_p = final_dir / "seed_stability_results.csv"
    summ_p = final_dir / "seed_stability_summary.csv"

    if level in {"standard", "strict"}:
        if mode in {"seed2026", "all"}:
            check(main_p.exists(), "seed2026_main_results.csv exists")
            if main_p.exists():
                check(len(pd.read_csv(main_p)) == expected_main, f"main results rows = {expected_main}")
        if mode in {"stability", "all"}:
            check(stab_p.exists(), "seed_stability_results.csv exists")
            if stab_p.exists():
                expected = expected_main * len(seeds)
                check(len(pd.read_csv(stab_p)) == expected, f"stability rows = {expected}")
            if seeds == config.DEFAULT_STABILITY_SEEDS:
                check(summ_p.exists(), "seed_stability_summary.csv exists")
                if summ_p.exists():
                    check(len(pd.read_csv(summ_p)) == expected_main, f"stability summary rows = {expected_main}")
        if mode == "selected_after_seed2026":
            paths = selected_paths or {}
            for key in ["models_csv", "models_md", "results_csv", "summary_csv", "summary_md", "manifest"]:
                check(bool(paths.get(key)) and paths[key].exists(), f"selected output {key} exists")
            if paths.get("results_csv") and paths["results_csv"].exists():
                selected_results = pd.read_csv(paths["results_csv"])
                expected = selected_model_count * len(seeds)
                check(len(selected_results) == expected, f"selected stability rows = {expected}")
                check(
                    not selected_results.duplicated(["seed", "early_variant", "model_spec_id"]).any(),
                    "selected stability model combinations unique",
                )
            if paths.get("summary_csv") and paths["summary_csv"].exists():
                selected_summary = pd.read_csv(paths["summary_csv"])
                check(len(selected_summary) == selected_model_count, f"selected summary rows = {selected_model_count}")
                ci = ["external_accuracy_ci95_low", "external_accuracy_ci95_high"]
                check(all(column in selected_summary.columns for column in ci), "selected stability CI fields")
        for filename in [
            "stagev3_summary.md", "run_manifest.json", "run_progress.json", "run_progress.md"
        ]:
            check((final_dir / filename).exists(), f"{filename} exists")
        figure_expectations = {
            "seed2026_main_results.csv": "top_seed2026_external_accuracy.png",
            "earlyv1_gain_over_earlyv0.csv": "earlyv1_gain_over_earlyv0.png",
            "scale_gain_seed2026.csv": "scale_gain_seed2026.png",
            "seed_stability_summary.csv": "top_stability_external_accuracy_ci.png",
            "scale_gain_stability.csv": "scale_gain_stability.png",
        }
        for csv_name, figure_name in figure_expectations.items():
            if (final_dir / csv_name).exists():
                check((final_dir / "figures" / figure_name).exists(), f"{figure_name} generated")

    if level == "strict":
        required = [
            "early_variant", "model_spec_id", "feature_block", "model_name",
            "external_accuracy", "external_f1", "external_auc", "cv_mode",
        ]
        if main_p.exists():
            main = pd.read_csv(main_p)
            check(all(column in main.columns for column in required), "main required columns")
            check(not main.duplicated(["early_variant", "model_spec_id"]).any(), "main model combinations unique")
        if stab_p.exists():
            stability = pd.read_csv(stab_p)
            check(
                not stability.duplicated(["seed", "early_variant", "model_spec_id"]).any(),
                "stability model combinations unique",
            )
        if summ_p.exists():
            summary = pd.read_csv(summ_p)
            ci = ["external_accuracy_ci95_low", "external_accuracy_ci95_high", "external_accuracy_95ci"]
            check(all(column in summary.columns for column in ci), "stability CI fields")
        for filename in [
            "earlyv1_gain_over_earlyv0.csv",
            "scale_gain_seed2026.csv",
            "scale_gain_stability.csv",
            "early_distribution_summary.csv",
            "stagev3_summary.md",
        ]:
            check((final_dir / filename).exists(), f"strict output {filename}")
        check((config.NOTEBOOK_DIR / "stagev3_result_check.ipynb").exists(), "notebook reference exists")

    lines.extend(["", f"Overall status: {'PASS' if ok else 'FAIL'}"])
    text = "\n".join(lines)
    (final_dir / "validation_report.md").write_text(text, encoding="utf-8")
    return ok, text
