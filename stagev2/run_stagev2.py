from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.api_feature_extraction import run_late_llm, run_middle_embeddings
from src.data_source import prepare_raw_input, validate_raw_source
from src.stagev1_features import FEATURE_VERSION, generate_all_early_features

OPERATING_THRESHOLD = 0.5


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run stagev2 from raw Cookie Theft transcript CSVs. The pipeline regenerates early BM25, "
            "middle BGE-M3, and late qwen3-235b-a22b features before training classifiers. "
            "Run this command from the stagev2 project root."
        ),
        epilog=(
            r"PowerShell: python .\run_stagev2.py --data-root .\input\raw"
            "\n"
            r"Zip input: python .\run_stagev2.py --data-zip .\input\raw_data.zip"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root", type=str, default="input/raw", help="Folder containing raw ad/control/test CSV files.")
    p.add_argument("--data-zip", type=str, default=None, help="Optional zip containing the raw CSV files. Only raw input CSVs are extracted.")
    p.add_argument("--output-dir", type=str, default="output/stagev2", help="Output folder. It is regenerated unless --reuse-cache is set.")
    p.add_argument("--bootstrap-n", type=int, default=200, help="Bootstrap iterations for external validation CI.")
    p.add_argument("--n-jobs", type=str, default="-1", help="n_jobs for GridSearchCV, e.g. -1 or 1.")
    p.add_argument("--reuse-cache", action="store_true", help="Reuse API cache inside output/cache. Default is a clean feature regeneration.")
    p.add_argument("--skip-stage-run", action="store_true", help="Only regenerate features; do not train classifiers.")
    return p.parse_args()


def _safe_clean_output(output_dir: Path, reuse_cache: bool) -> None:
    if output_dir.exists() and not reuse_cache:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_project_path(value: str | None, package_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (package_dir / path).resolve()


def _validate_output_dir(output_dir: Path, package_dir: Path, data_root: Path | None, data_zip: Path | None) -> None:
    if output_dir == package_dir or output_dir in package_dir.parents:
        raise ValueError(f"Refusing unsafe --output-dir that contains the project: {output_dir}")
    if data_root is not None and (output_dir == data_root or output_dir in data_root.parents):
        raise ValueError(f"Refusing --output-dir that contains the raw input directory: {output_dir}")
    if data_zip is not None and (output_dir == data_zip.parent or output_dir in data_zip.parents):
        raise ValueError(f"Refusing --output-dir that contains the zip input: {output_dir}")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _run_stage_core(stage_core_dir: Path, output_dir: Path, early_dir: Path, embedding_dir: Path, llm_dir: Path, preprocess_dir: Path, run_dir: Path, bootstrap_n: int, n_jobs: str) -> None:
    env = os.environ.copy()
    env.update({
        "STAGE_AD_PROJECT_ROOT": str(output_dir),
        "STAGE_BM25_DIR": str(early_dir),
        "STAGE_EMBEDDING_DIR": str(embedding_dir),
        "STAGE_LLM_DIR": str(llm_dir),
        "STAGE_PREPROCESS_DIR": str(preprocess_dir),
        "STAGE_OUTPUT_ROOT": str(run_dir),
        "STAGE_BOOTSTRAP_N": str(bootstrap_n),
        "STAGE_N_JOBS": str(n_jobs),
        "STAGE_DECISION_THRESHOLD": str(OPERATING_THRESHOLD),
        "STAGE_SCORING": "accuracy",
        "STAGEV2_ENABLE_INTERACTIONS": "1",
    })
    print("\n[stagev2] Running classifier training/evaluation")
    print(f"[stagev2] Selection metric: held-out external accuracy")
    print(f"[stagev2] Output dir: {run_dir}")
    subprocess.run([sys.executable, "run_stage_corrected.py"], cwd=str(stage_core_dir), env=env, check=True)


def _write_final_comparison(run_dir: Path, output_dir: Path, feature_summaries: dict) -> None:
    reports_dir = output_dir / "final_report"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ext_path = run_dir / "tables" / "stagev2_external_performance_report.csv"
    rank_path = run_dir / "tables" / "stagev2_model_ranking_by_external_accuracy.csv"
    cv_path = run_dir / "tables" / "stagev2_cv_summary.csv"
    pred_path = run_dir / "predictions" / "stagev2_test_predictions_all_models.csv"
    oof_path = run_dir / "predictions" / "stagev2_oof_predictions_top10.csv"
    selected_path = run_dir / "reports" / "stagev2_selected_model_summary.md"
    report_path = run_dir / "reports" / "stagev2_experiment_report.md"
    leakage_path = run_dir / "reports" / "stagev2_leakage_check.json"

    if not ext_path.exists():
        raise FileNotFoundError(f"Missing external performance report: {ext_path}")

    ext = pd.read_csv(ext_path)
    rank = ext.sort_values(
        ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"],
        ascending=False,
    )
    rank.to_csv(reports_dir / "stagev2_model_ranking_by_external_accuracy.csv", index=False)
    ext.to_csv(reports_dir / "stagev2_external_performance_report.csv", index=False)

    if cv_path.exists():
        pd.read_csv(cv_path).to_csv(reports_dir / "stagev2_cv_summary.csv", index=False)
    if pred_path.exists():
        pd.read_csv(pred_path).to_csv(reports_dir / "stagev2_test_predictions_all_models.csv", index=False)
    if oof_path.exists():
        pd.read_csv(oof_path).to_csv(reports_dir / "stagev2_oof_predictions_top10.csv", index=False)
    if selected_path.exists():
        shutil.copy2(selected_path, reports_dir / "stagev2_selected_model_summary.md")
    if report_path.exists():
        shutil.copy2(report_path, reports_dir / "stagev2_experiment_report.md")
    if leakage_path.exists():
        shutil.copy2(leakage_path, reports_dir / "stagev2_leakage_check.json")

    best = rank.iloc[0]
    summary = {
        "selection_metric": "external_accuracy",
        "external_set_role": "held-out external validation, not unbiased final test",
        "selected_model": best.get("model_name"),
        "external_accuracy": float(best.get("accuracy")),
        "external_balanced_accuracy": float(best.get("balanced_accuracy")),
        "feature_summaries": feature_summaries,
    }
    _write_json(reports_dir / "stagev2_final_run_summary.json", summary)


def main() -> None:
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != package_dir:
        raise RuntimeError(
            f"Run stagev2 from the project root: {package_dir}\n"
            r"PowerShell: python .\run_stagev2.py --data-root .\input\raw"
        )

    output_dir = _resolve_project_path(args.output_dir, package_dir)
    data_root = _resolve_project_path(args.data_root, package_dir)
    data_zip = _resolve_project_path(args.data_zip, package_dir)
    assert output_dir is not None

    _validate_output_dir(output_dir, package_dir, data_root, data_zip)
    validate_raw_source(data_root, data_zip)
    if not args.reuse_cache and not os.getenv("MAAS_API_KEY", "").strip():
        raise RuntimeError("Missing required environment variable MAAS_API_KEY. Existing output was not removed.")

    _safe_clean_output(output_dir, reuse_cache=args.reuse_cache)

    print("[stagev2] Preparing raw input")
    input_dir, source_info = prepare_raw_input(data_root, output_dir, data_zip=data_zip, force_extract=not args.reuse_cache)
    print(f"[stagev2] Raw input dir: {input_dir}")

    print("[stagev2] Regenerating early BM25 features")
    preprocessed, early_dir, early_summary = generate_all_early_features(input_dir, output_dir, FEATURE_VERSION)

    feature_root = output_dir / "features"
    cache_dir = output_dir / "cache"

    print("[stagev2] Regenerating middle BGE-M3 embedding features")
    middle_summary = run_middle_embeddings(preprocessed, feature_root / "embedding", cache_dir, reuse_cache=args.reuse_cache)

    print("[stagev2] Regenerating late qwen3-235b-a22b expressive-form features")
    late_summary = run_late_llm(preprocessed, feature_root / "llm", cache_dir, reuse_cache=args.reuse_cache)

    feature_summaries = {
        "source_info": source_info,
        "early": early_summary,
        "middle": middle_summary,
        "late": late_summary,
        "historical_outputs_reused": False,
    }
    _write_json(output_dir / "stagev2_feature_generation_summary.json", feature_summaries)

    if args.skip_stage_run:
        print("[stagev2] --skip-stage-run used; feature regeneration complete.")
        return

    stage_core_dir = package_dir / "stage_core"
    run_dir = output_dir / "run_external_accuracy_selection"
    _run_stage_core(
        stage_core_dir=stage_core_dir,
        output_dir=output_dir,
        early_dir=early_dir,
        embedding_dir=feature_root / "embedding",
        llm_dir=feature_root / "llm",
        preprocess_dir=output_dir / "preprocess",
        run_dir=run_dir,
        bootstrap_n=args.bootstrap_n,
        n_jobs=args.n_jobs,
    )

    print("[stagev2] Writing final canonical report outputs")
    _write_final_comparison(run_dir, output_dir, feature_summaries)
    print("\n[stagev2] Done.")
    print(f"[stagev2] Final report dir: {output_dir / 'final_report'}")


if __name__ == "__main__":
    main()
