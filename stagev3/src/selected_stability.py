from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .experiment import run_protocol_for_seed
from .models import ModelSpec, model_specs


def threshold_tag(threshold: float) -> str:
    return f"{int(round(float(threshold) * 100)):03d}"


def default_selected_path(final_dir: Path, threshold: float) -> Path:
    return final_dir / f"selected_model_specs_external{threshold_tag(threshold)}.csv"


def selected_output_paths(final_dir: Path, threshold: float) -> dict[str, Path]:
    tag = threshold_tag(threshold)
    return {
        "models_csv": final_dir / f"selected_model_specs_external{tag}.csv",
        "models_md": final_dir / f"selected_model_specs_external{tag}.md",
        "results_csv": final_dir / f"selected_seed_stability_results_external{tag}.csv",
        "summary_csv": final_dir / f"selected_seed_stability_summary_external{tag}.csv",
        "summary_md": final_dir / f"selected_seed_stability_summary_external{tag}.md",
        "manifest": final_dir / f"selected_run_manifest_external{tag}.json",
        "checkpoint_dir": final_dir / f"selected_stability_checkpoints_external{tag}",
    }


def _model_family(row: pd.Series) -> str:
    special = str(row.get("is_special_model", False)).strip().lower() in {"1", "true", "yes"}
    if special:
        return "special"
    return str(row.get("model_variant", row.get("model_name", ""))).split("__", 1)[0]


def _add_model_spec_ids(main: pd.DataFrame) -> pd.DataFrame:
    if "model_spec_id" in main.columns:
        return main
    registry = model_specs()
    lookup: dict[tuple[str, str], list[str]] = {}
    for spec in registry:
        for name in {spec.model_name, spec.model_variant}:
            lookup.setdefault((spec.feature_block, name), []).append(spec.model_spec_id)
    ids = []
    failures = []
    for index, row in main.iterrows():
        feature_block = str(row.get("feature_block", ""))
        model_name = str(row.get("model_name", row.get("model_variant", "")))
        candidates = sorted(set(lookup.get((feature_block, model_name), [])))
        if len(candidates) != 1:
            failures.append({
                "index": int(index),
                "early_variant": str(row.get("early_variant", "")),
                "feature_block": feature_block,
                "model_name": model_name,
                "registry_matches": candidates,
            })
            ids.append("")
        else:
            ids.append(candidates[0])
    if failures:
        raise RuntimeError(
            "Cannot reconstruct stable model_spec_id values from the current model registry: "
            f"{failures}"
        )
    out = main.copy()
    out["model_spec_id"] = ids
    return out


def select_models(
    source_path: Path,
    threshold: float,
    csv_path: Path,
    md_path: Path,
) -> pd.DataFrame:
    if not source_path.exists():
        raise RuntimeError(
            "seed2026_main_results.csv not found. Please run:\n"
            "python run_stagev3.py --mode seed2026 --cv-mode exact"
        )
    main = _add_model_spec_ids(pd.read_csv(source_path))
    required = {
        "early_variant", "model_spec_id", "feature_block", "model_name",
        "external_accuracy", "external_precision", "external_recall",
        "external_f1", "external_auc", "n_features",
    }
    missing = sorted(required - set(main.columns))
    if missing:
        raise RuntimeError(f"seed2026 main results missing columns: {missing}")
    selected = main.loc[pd.to_numeric(main["external_accuracy"], errors="coerce") >= float(threshold)].copy()
    selected["classifier_name"] = selected["model_name"].astype(str)
    selected["model_family"] = selected.apply(_model_family, axis=1)
    selected["selected_by"] = "seed2026_external_accuracy"
    selected["selection_threshold"] = float(threshold)
    selected["selection_source"] = str(source_path)
    columns = [
        "early_variant", "model_spec_id", "feature_block", "classifier_name",
        "model_family", "external_accuracy", "external_precision",
        "external_recall", "external_f1", "external_auc", "n_features",
        "selected_by", "selection_threshold", "selection_source",
    ]
    selected = selected[columns].sort_values(
        ["external_accuracy", "early_variant", "model_spec_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(csv_path, index=False, encoding="utf-8")
    lines = [
        "# Selected model specs",
        "",
        f"- Selection source: `{source_path}`",
        f"- Selection rule: `external_accuracy >= {threshold}`",
        f"- Selected model count: `{len(selected)}`",
        "",
    ]
    if selected.empty:
        lines.append("No model specs met the selection threshold.")
    else:
        lines.append(selected.to_markdown(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected


def load_selected_specs(path: Path) -> tuple[pd.DataFrame, dict[str, list[ModelSpec]]]:
    if not path.exists():
        raise RuntimeError(
            "Selected model file was not created from seed2026_main_results.csv."
        )
    selected = pd.read_csv(path)
    required = {"early_variant", "model_spec_id", "selection_threshold", "selection_source"}
    missing_columns = sorted(required - set(selected.columns))
    if missing_columns:
        raise RuntimeError(f"Selected model file missing columns: {missing_columns}")
    duplicate_mask = selected.duplicated(["early_variant", "model_spec_id"], keep=False)
    if duplicate_mask.any():
        duplicates = selected.loc[duplicate_mask, ["early_variant", "model_spec_id"]].to_dict("records")
        raise RuntimeError(f"Selected model file contains duplicate stable keys: {duplicates}")
    registry = {spec.model_spec_id: spec for spec in model_specs()}
    missing_items = []
    by_variant: dict[str, list[ModelSpec]] = {}
    for row in selected.itertuples(index=False):
        spec = registry.get(str(row.model_spec_id))
        if spec is None or str(row.early_variant) not in {"earlyv0", "earlyv1"}:
            missing_items.append({
                "early_variant": str(row.early_variant),
                "model_spec_id": str(row.model_spec_id),
            })
            continue
        by_variant.setdefault(str(row.early_variant), []).append(spec)
    if missing_items:
        raise RuntimeError(f"Selected model specs not found in current registry: {missing_items}")
    return selected, by_variant


def format_selected_results(results: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    metadata = selected[
        ["early_variant", "model_spec_id", "classifier_name", "model_family",
         "selection_threshold", "selection_source"]
    ]
    out = results.merge(metadata, on=["early_variant", "model_spec_id"], how="left", validate="many_to_one")
    columns = [
        "seed", "early_variant", "model_spec_id", "feature_block",
        "classifier_name", "model_family", "external_accuracy",
        "external_precision", "external_recall", "external_f1", "external_auc",
        "external_tn", "external_fp", "external_fn", "external_tp", "n_features",
        "selection_threshold", "selection_source",
    ]
    return out[columns]


def seed_checkpoint_path(checkpoint_dir: Path, seed: int) -> Path:
    return checkpoint_dir / f"seed_{int(seed):03d}.csv"


def _expected_selected_keys(selected: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            selected["early_variant"].astype(str),
            selected["model_spec_id"].astype(str),
        )
    )


def read_valid_seed_checkpoint(
    checkpoint_path: Path,
    seed: int,
    selected: pd.DataFrame,
    cv_mode: str,
) -> pd.DataFrame | None:
    if not checkpoint_path.exists():
        return None
    try:
        frame = pd.read_csv(checkpoint_path)
    except Exception:
        return None
    required = {
        "seed", "early_variant", "model_spec_id", "external_accuracy",
        "selection_threshold", "selection_source", "checkpoint_cv_mode",
    }
    if not required <= set(frame.columns) or len(frame) != len(selected):
        return None
    if set(pd.to_numeric(frame["seed"], errors="coerce").dropna().astype(int)) != {int(seed)}:
        return None
    if set(frame["checkpoint_cv_mode"].astype(str)) != {str(cv_mode)}:
        return None
    actual_keys = set(
        zip(frame["early_variant"].astype(str), frame["model_spec_id"].astype(str))
    )
    if actual_keys != _expected_selected_keys(selected):
        return None
    if frame.duplicated(["seed", "early_variant", "model_spec_id"]).any():
        return None
    return frame


def checkpoint_seed_status(
    checkpoint_dir: Path,
    seeds: list[int],
    selected: pd.DataFrame,
    cv_mode: str,
) -> tuple[list[int], list[int]]:
    completed = []
    missing = []
    for seed in seeds:
        path = seed_checkpoint_path(checkpoint_dir, seed)
        if read_valid_seed_checkpoint(path, seed, selected, cv_mode) is None:
            missing.append(int(seed))
        else:
            completed.append(int(seed))
    return completed, missing


def run_selected_seed_checkpoint(
    seed: int,
    blocks_by_variant: dict,
    y_train,
    y_test,
    cv_mode: str,
    selected: pd.DataFrame,
    specs_by_variant: dict[str, list[ModelSpec]],
    checkpoint_path: Path,
) -> int:
    parts = []
    for early_variant in ["earlyv0", "earlyv1"]:
        specs = specs_by_variant.get(early_variant, [])
        if not specs:
            continue
        parts.append(
            run_protocol_for_seed(
                blocks_by_variant[early_variant],
                y_train,
                y_test,
                early_variant,
                int(seed),
                cv_mode,
                specs=specs,
            )
        )
    if not parts:
        raise RuntimeError("No selected model specs are available for checkpoint execution.")
    result = format_selected_results(pd.concat(parts, ignore_index=True), selected)
    result["checkpoint_cv_mode"] = str(cv_mode)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}.{os.getpid()}.tmp"
    )
    result.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(checkpoint_path)
    return int(seed)


def merge_seed_checkpoints(
    checkpoint_dir: Path,
    seeds: list[int],
    selected: pd.DataFrame,
    cv_mode: str,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    frames = []
    completed = []
    missing = []
    for seed in seeds:
        frame = read_valid_seed_checkpoint(
            seed_checkpoint_path(checkpoint_dir, seed),
            seed,
            selected,
            cv_mode,
        )
        if frame is None:
            missing.append(int(seed))
        else:
            completed.append(int(seed))
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), completed, missing
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop(columns=["checkpoint_cv_mode"], errors="ignore")
    merged = merged.sort_values(
        ["seed", "early_variant", "model_spec_id"]
    ).reset_index(drop=True)
    return merged, completed, missing


def summarize_selected_stability(results: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    seed2026 = selected.rename(columns={"external_accuracy": "seed2026_external_accuracy"})
    seed2026 = seed2026[
        ["early_variant", "model_spec_id", "seed2026_external_accuracy",
         "selection_threshold", "selection_source"]
    ]
    keys = ["early_variant", "model_spec_id", "feature_block", "classifier_name", "model_family"]
    rows = []
    for values, group in results.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        acc = pd.to_numeric(group["external_accuracy"], errors="coerce").dropna()
        n = len(acc)
        mean = float(acc.mean()) if n else np.nan
        std = float(acc.std(ddof=1)) if n > 1 else np.nan
        if n > 1:
            margin = float(stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
            low, high = mean - margin, mean + margin
        else:
            low = high = np.nan
        row.update({
            "n_seeds": int(n),
            "external_accuracy_mean": mean,
            "external_accuracy_std": std,
            "external_accuracy_ci95_low": low,
            "external_accuracy_ci95_high": high,
            "external_precision_mean": float(group["external_precision"].mean()),
            "external_recall_mean": float(group["external_recall"].mean()),
            "external_f1_mean": float(group["external_f1"].mean()),
            "external_auc_mean": float(group["external_auc"].mean()),
            "external_accuracy_min": float(acc.min()) if n else np.nan,
            "external_accuracy_max": float(acc.max()) if n else np.nan,
        })
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.merge(seed2026, on=["early_variant", "model_spec_id"], how="left", validate="one_to_one")
    columns = [
        "early_variant", "model_spec_id", "feature_block", "classifier_name",
        "model_family", "n_seeds", "external_accuracy_mean",
        "external_accuracy_std", "external_accuracy_ci95_low",
        "external_accuracy_ci95_high", "external_precision_mean",
        "external_recall_mean", "external_f1_mean", "external_auc_mean",
        "external_accuracy_min", "external_accuracy_max",
        "seed2026_external_accuracy", "selection_threshold", "selection_source",
    ]
    return summary[columns].sort_values(
        ["external_accuracy_mean", "seed2026_external_accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def write_selected_summary_md(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Selected-model stability summary",
        "",
        (
            "Selected stability CI reflects random-seed / CV-split variability among "
            "models selected by seed2026 external accuracy. Because external accuracy "
            "is used for model selection, this is not an unbiased final external-test "
            "confidence interval."
        ),
        "",
    ]
    if summary.empty:
        lines.append("No selected stability results were produced.")
    else:
        insufficient = int((summary["n_seeds"] < 2).sum())
        lines.append(f"- Model specs: `{len(summary)}`")
        lines.append(f"- Specs with fewer than two seeds: `{insufficient}`; their 95% CI is NaN.")
        lines.append("")
        lines.append(summary.to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
