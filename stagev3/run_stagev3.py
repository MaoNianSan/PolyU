from __future__ import annotations

import argparse
import json
import os
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from joblib import Parallel, cpu_count, delayed, parallel_config
from tqdm.auto import tqdm

from src import config
from src.api_clients import HuaweiMaaSClient, MaaSAPIError, mask_secret
from src.early_features import generate_early_features
from src.experiment import run_protocol_for_seed
from src.feature_merge import build_feature_blocks, merge_stage_features
from src.late_features import generate_late_features, has_complete_real_features_or_cache as has_complete_late
from src.middle_features import (
    CACHE_FILE as MIDDLE_CACHE_FILE,
    generate_middle_features,
    has_complete_real_features_or_cache as has_complete_middle,
    _load_real_vectors_from_cache as _load_middle_vectors_from_cache,
    _required_window_keys as _required_middle_window_keys,
)
from src.preprocess import find_and_copy_raw_files, run_preprocess
from src.progress import RunProgress
from src.report import early_distribution_summary, write_notebook, write_summary_md
from src.scale_gain import earlyv1_gain_over_earlyv0, scale_gain_seed2026, scale_gain_stability
from src.selected_stability import (
    checkpoint_seed_status,
    load_selected_specs,
    merge_seed_checkpoints,
    run_selected_seed_checkpoint,
    select_models,
    seed_checkpoint_path,
    selected_output_paths,
    summarize_selected_stability,
    write_selected_manifest,
    write_selected_summary_md,
)
from src.self_check import dry_run_lines, feature_manifest_status, token_count_excluded
from src.stability import attach_accuracy_ci, summarize_stability
from src.validation import validate_outputs
from src.visualization import generate_figures


def parse_seeds(text: str | None, default: list[int]) -> list[int]:
    if text is None or not str(text).strip():
        return default
    text = str(text).strip()
    if "-" in text and "," not in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def automatic_worker_count(task_count: int) -> int:
    if task_count < 1:
        return 0
    try:
        available_cores = cpu_count(only_physical_cores=True)
    except TypeError:
        available_cores = cpu_count()
    return min(task_count, max(1, available_cores - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stagev3 pipeline.")
    parser.add_argument(
        "--mode",
        choices=["seed2026", "stability", "all", "selected_after_seed2026"],
        default="seed2026",
    )
    parser.add_argument("--seeds", default=None, help="Stability seeds, e.g. 0-29 or 0,1,2.")
    parser.add_argument("--cv-mode", choices=["exact", "fast", "v2_repeated"], default="exact")
    parser.add_argument("--validate", choices=["light", "standard", "strict"], default="standard")
    parser.add_argument("--source-zip", default=None)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--require-api", action="store_true", help="Deprecated; real API/cache is always required.")
    parser.add_argument("--check-api", action="store_true", help="Send one minimal embedding request and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Check structure without API calls or model training.")
    parser.add_argument("--diagnose-middle-cache", action="store_true", help="Report middle cache coverage and exit without API calls.")
    parser.add_argument("--middle-only", action="store_true", help="Run preprocessing, early features, and middle feature/cache extraction only.")
    parser.add_argument("--min-external-accuracy", type=float, default=0.75)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed selected-stability seed checkpoints (default: enabled).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun requested selected-stability seeds even when checkpoints exist.",
    )
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def _zip_project(project_root: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in project_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(project_root.parent))


def _read_api_key() -> str:
    return (os.getenv(config.MAAS_API_KEY_ENV) or "").strip()


def _print_environment_check() -> str:
    api_key = _read_api_key()
    print("Environment check:")
    print(f"- MAAS_API_KEY: {'detected' if api_key else 'missing'}")
    print(f"- masked key: {mask_secret(api_key)}")
    print(f"- HUAWEI_MAAS_BASE_URL: {config.HUAWEI_MAAS_BASE_URL}")
    print(f"- EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")
    print(f"- SSL verify: {str(config.HUAWEI_MAAS_SSL_VERIFY).lower()}")
    print(f"- trust_env: {str(config.HUAWEI_MAAS_TRUST_ENV).lower()}")
    print(f"- embedding batch size: {config.EMBEDDING_BATCH_SIZE}")
    print(f"- embedding timeout: {config.EMBEDDING_TIMEOUT}")
    print(f"- embedding max retries: {config.EMBEDDING_MAX_RETRIES}")
    return api_key


def _missing_key_message() -> str:
    return (
        "MAAS_API_KEY is missing and no complete real API cache is available.\n"
        '$env:MAAS_API_KEY="your_key_here"\n'
        "python run_stagev3.py --mode seed2026 --cv-mode exact"
    )


def _run_api_check(api_key: str) -> int:
    if not api_key:
        print("API connection FAILED: missing key")
        print(_missing_key_message())
        return 1
    try:
        client = HuaweiMaaSClient(timeout=min(config.HUAWEI_MAAS_TIMEOUT, 30.0), max_retries=1)
        embeddings = client.embed(["API connectivity test."])
        if len(embeddings) != 1 or not embeddings[0]:
            raise MaaSAPIError("other error", "Embedding endpoint returned no vector.")
    except MaaSAPIError as exc:
        print(f"API connection FAILED: {exc.kind}")
        print(str(exc))
        return 1
    print("Embedding API connection OK")
    return 0


def _write_manifest(manifest: dict) -> None:
    path = config.FINAL_REPORT_DIR / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _purge_non_api_feature_artifacts() -> dict:
    removed = {"cache_files": [], "feature_files": []}
    for cache_path in [
        config.CACHE_DIR / "huawei_bge_m3_embedding_cache.csv",
        config.CACHE_DIR / "huawei_llm_late_feature_cache.csv",
    ]:
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(cache_path)
            mask = frame.get("source", pd.Series("", index=frame.index)).astype(str).str.lower().isin(
                {"local_surrogate", "surrogate"}
            )
            if mask.any():
                kept = frame.loc[~mask]
                if kept.empty:
                    cache_path.unlink(missing_ok=True)
                else:
                    kept.to_csv(cache_path, index=False, encoding="utf-8")
                removed["cache_files"].append(str(cache_path))
        except Exception:
            cache_path.unlink(missing_ok=True)
            removed["cache_files"].append(str(cache_path))
    return removed


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _diagnose_middle_cache(args: argparse.Namespace) -> int:
    source_zip = Path(args.source_zip).resolve() if args.source_zip else None
    find_and_copy_raw_files(source_zip, config.INPUT_RAW_DIR)
    train, test, _ = run_preprocess(config.INPUT_RAW_DIR, config.PREPROCESS_DIR)
    required = _required_middle_window_keys(train, test)
    vectors, ignored = _load_middle_vectors_from_cache(config.CACHE_DIR / MIDDLE_CACHE_FILE, reuse_cache=True)
    cached = sum(1 for key in required if key in vectors)
    pending = len(required) - cached
    print("Middle cache diagnosis:")
    print(f"- cache path: {config.CACHE_DIR / MIDDLE_CACHE_FILE}")
    print(f"- required unique windows: {len(required)}")
    print(f"- cached real windows: {cached}")
    print(f"- pending API windows: {pending}")
    print(f"- ignored surrogate rows: {ignored}")
    if pending:
        print("- status: incomplete; API calls are still required unless you provide a complete real cache")
    else:
        print("- status: complete; middle stage should not call the API")
    return 0


def _run_pipeline(args: argparse.Namespace, api_key: str) -> None:
    config.ensure_dirs()
    stability_seeds = parse_seeds(
        args.seeds,
        config.DEFAULT_STABILITY_SEEDS
        if args.mode in {"stability", "all", "selected_after_seed2026"}
        else [],
    )
    selected_paths = selected_output_paths(config.FINAL_REPORT_DIR, args.min_external_accuracy)
    selected_file = selected_paths["models_csv"]
    selected_df = pd.DataFrame()
    selected_specs_by_variant: dict = {}
    checkpoint_dir = selected_paths["checkpoint_dir"]
    completed_seed_ids: list[int] = []
    missing_seed_ids: list[int] = list(stability_seeds)
    worker_count = 0
    main_results_path = config.FINAL_REPORT_DIR / "seed2026_main_results.csv"
    if args.mode == "selected_after_seed2026":
        if args.force_features:
            raise RuntimeError(
                "selected_after_seed2026 does not re-extract features. "
                "Remove --force-features and use the existing output/features files."
            )
        feature_ok, _ = feature_manifest_status()
        if not feature_ok:
            raise RuntimeError(
                "Selected stability requires complete valid features under output/features. "
                "Run seed2026 first:\n"
                "python run_stagev3.py --mode seed2026 --cv-mode exact"
            )
        select_models(
            main_results_path,
            args.min_external_accuracy,
            selected_paths["models_csv"],
            selected_paths["models_md"],
        )
        selected_file = selected_paths["models_csv"]
        selected_df, selected_specs_by_variant = load_selected_specs(selected_file)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and not args.force_rerun:
            completed_seed_ids, missing_seed_ids = checkpoint_seed_status(
                checkpoint_dir, stability_seeds, selected_df, args.cv_mode
            )
        print("Resume enabled." if args.resume else "Resume disabled.")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Completed seeds: {len(completed_seed_ids)} / {len(stability_seeds)}")
        print(f"Missing seeds: {missing_seed_ids}")
        for seed in completed_seed_ids:
            print(f"Skipping completed seed {seed}")
    progress = RunProgress(args.mode, args.cv_mode, stability_seeds, len(selected_df))
    if args.mode == "selected_after_seed2026":
        progress.update(
            worker_count=worker_count,
            resume=args.resume,
            force_rerun=args.force_rerun,
            checkpoint_dir=str(checkpoint_dir),
            completed_seed_count=len(completed_seed_ids),
            expected_seed_count=len(stability_seeds),
            completed_seed_ids=completed_seed_ids,
            missing_seed_ids=missing_seed_ids,
            model_progress={
                "completed_model_runs": len(completed_seed_ids) * len(selected_df),
            },
        )
    stage = "initializing"

    try:
        source_zip = Path(args.source_zip).resolve() if args.source_zip else None
        feature_bar = tqdm(total=5, desc="Feature extraction", unit="stage")

        stage = "[1/5] Preprocessing raw data"
        progress.update(current_stage=stage)
        feature_bar.set_description(stage)
        raw_source = find_and_copy_raw_files(source_zip, config.INPUT_RAW_DIR)
        purged = _purge_non_api_feature_artifacts()
        train, test, preprocess_summary = run_preprocess(config.INPUT_RAW_DIR, config.PREPROCESS_DIR)
        feature_bar.update(1)

        if not api_key:
            middle_complete = has_complete_middle(
                train, test, config.FEATURE_DIR / "middle", allow_existing_features=not args.force_features
            )
            late_complete = has_complete_late(
                train, test, config.FEATURE_DIR / "late", allow_existing_features=not args.force_features
            )
            if not (middle_complete and late_complete):
                raise RuntimeError(_missing_key_message())

        stage = "[2/5] Generating early features"
        progress.update(current_stage=stage)
        feature_bar.set_description(stage)
        early = generate_early_features(
            train, test, config.FEATURE_DIR / "early",
            force_extract=args.force_features, reuse_features=True,
        )
        progress.feature("early", "completed")
        feature_bar.update(1)

        stage = "[3/5] Generating middle BGE-M3 embeddings"
        progress.update(current_stage=stage)
        feature_bar.set_description(stage)

        def middle_update(values: dict) -> None:
            progress.update(middle_embedding=values)
            current = progress.data["middle_embedding"]
            feature_bar.set_postfix(
                total_windows=current["total_windows"],
                cached_windows=current["cached_windows"],
                pending_api_windows=current["pending_api_windows"],
                api_batch_size=current["api_batch_size"],
                completed_batches=current["completed_batches"],
                refresh=False,
            )

        middle, middle_manifest = generate_middle_features(
            train, test, config.FEATURE_DIR / "middle",
            force_extract=args.force_features, reuse_features=True,
            progress_callback=middle_update,
        )
        progress.feature("middle", "completed")
        feature_bar.update(1)
        if args.middle_only:
            feature_bar.close()
            progress.update(status="completed", current_stage="middle-only completed")
            progress.write_markdown()
            print("Middle-only run completed. The middle cache/features are now ready for the full run.")
            return

        stage = "[4/5] Generating late LLM features"
        progress.update(current_stage=stage)
        feature_bar.set_description(stage)

        def late_update(values: dict) -> None:
            progress.update(late_features=values)
            current = progress.data["late_features"]
            feature_bar.set_postfix(
                total_transcripts=current["total_transcripts"],
                cached_transcripts=current["cached_transcripts"],
                pending_api_transcripts=current["pending_api_transcripts"],
                completed_requests=current["completed_requests"],
                refresh=False,
            )

        late, late_manifest = generate_late_features(
            train, test, config.FEATURE_DIR / "late",
            force_extract=args.force_features, reuse_features=True,
            progress_callback=late_update,
        )
        progress.feature("late", "completed")
        feature_bar.update(1)

        stage = "[5/5] Merging feature blocks"
        progress.update(current_stage=stage)
        feature_bar.set_description(stage)
        blocks_by_variant = {}
        for early_variant in config.EARLY_VARIANTS:
            merged_train = merge_stage_features(
                train, early[early_variant]["train"], middle["train"], late["train"]
            )
            merged_test = merge_stage_features(
                test, early[early_variant]["test"], middle["test"], late["test"]
            )
            blocks_by_variant[early_variant] = build_feature_blocks(merged_train, merged_test)
        feature_bar.update(1)
        feature_bar.close()

        feature_ok, feature_source = feature_manifest_status()
        if not feature_ok:
            raise RuntimeError("Feature manifest is not real_api or valid_cache.")
        if not token_count_excluded():
            raise RuntimeError("token_count is not excluded from model features.")
        if config.expected_specs_per_variant() != 102 or config.EARLY_VARIANTS != ["earlyv0", "earlyv1"]:
            raise RuntimeError("Model registry or early variants do not match the required structure.")

        y_train = train["label"].astype(int).to_numpy()
        y_test = test["label"].astype(int).to_numpy()
        completed_runs = (
            len(completed_seed_ids) * len(selected_df)
            if args.mode == "selected_after_seed2026"
            else 0
        )
        main_prediction_rows: list[dict] = []

        def run_for_seed(
            seed: int,
            seed_index: int,
            seed_total: int,
            prediction_sink: list[dict] | None = None,
            specs_by_variant: dict | None = None,
        ) -> pd.DataFrame:
            nonlocal completed_runs
            parts = []
            variants = (
                [variant for variant in config.EARLY_VARIANTS if specs_by_variant.get(variant)]
                if specs_by_variant is not None
                else config.EARLY_VARIANTS
            )
            early_bar = tqdm(variants, desc=f"seed progress: {seed_index}/{seed_total}", leave=False)
            selected_index = 0
            for early_index, early_variant in enumerate(early_bar, start=1):
                current_specs = specs_by_variant.get(early_variant, []) if specs_by_variant is not None else None
                model_total = len(current_specs) if current_specs is not None else config.expected_specs_per_variant()
                model_bar = tqdm(
                    total=model_total,
                    desc=f"early_variant progress: {early_index}/{len(variants)}",
                    leave=False,
                    unit="model",
                )

                def model_update(values: dict) -> None:
                    nonlocal completed_runs, selected_index
                    if values["event"] == "completed":
                        completed_runs += 1
                        selected_index += 1
                        model_bar.update(1)
                        progress.update(
                            current_seed=seed,
                            current_selected_model_index=selected_index if specs_by_variant is not None else 0,
                            model_progress={"completed_model_runs": completed_runs},
                        )
                        return
                    model_bar.set_postfix(
                        current_feature_block=values["current_feature_block"],
                        current_model_name=values["current_model_name"],
                        current_model_spec_id=values["current_model_spec_id"],
                        current_cv_mode=values["current_cv_mode"],
                        current_seed=values["current_seed"],
                        refresh=False,
                    )
                    progress.update(
                        current_stage="model_training",
                        model_progress={
                            "completed_model_runs": completed_runs,
                            "seed_progress": f"{seed_index}/{seed_total}",
                            "early_variant_progress": f"{early_index}/{len(variants)}",
                            "model_spec_progress": (
                                f"{values['model_spec_progress']}/{values['model_spec_total']}"
                            ),
                            "current_seed": values["current_seed"],
                            "current_early_variant": values["current_early_variant"],
                            "current_feature_block": values["current_feature_block"],
                            "current_model_name": values["current_model_name"],
                            "current_model_spec_id": values["current_model_spec_id"],
                            "current_cv_mode": values["current_cv_mode"],
                        },
                    )

                parts.append(run_protocol_for_seed(
                    blocks_by_variant[early_variant], y_train, y_test,
                    early_variant, seed, args.cv_mode, model_update,
                    external_meta=test[[c for c in ["sample_id", "mmse"] if c in test.columns]],
                    prediction_sink=prediction_sink,
                    specs=current_specs,
                ))
                model_bar.close()
            return pd.concat(parts, ignore_index=True)

        main_df = pd.DataFrame()
        stability_df = pd.DataFrame()
        stability_summary = pd.DataFrame()

        if args.mode in {"seed2026", "all"}:
            main_df = run_for_seed(config.PRIMARY_SEED, 1, 1, main_prediction_rows)

        if args.mode in {"stability", "all"}:
            stability_parts = []
            if args.cv_mode == "fast" and stability_seeds:
                template = run_for_seed(stability_seeds[0], 1, len(stability_seeds))
                stability_parts.append(template)
                for index, seed in enumerate(stability_seeds[1:], start=2):
                    copy = template.copy()
                    copy["seed"] = seed
                    stability_parts.append(copy)
                    completed_runs += len(copy)
                    progress.update(
                        current_stage="model_training",
                        model_progress={
                            "completed_model_runs": completed_runs,
                            "seed_progress": f"{index}/{len(stability_seeds)}",
                            "current_seed": seed,
                        },
                    )
            else:
                seed_bar = tqdm(stability_seeds, desc="seed progress", unit="seed")
                for index, seed in enumerate(seed_bar, start=1):
                    seed_bar.set_postfix(current_seed=seed)
                    stability_parts.append(run_for_seed(seed, index, len(stability_seeds)))
                seed_bar.close()
            stability_df = pd.concat(stability_parts, ignore_index=True)
            stability_summary = summarize_stability(stability_df)
        else:
            stability_summary = _load_csv(config.FINAL_REPORT_DIR / "seed_stability_summary.csv")

        if not main_df.empty:
            main_df = attach_accuracy_ci(main_df, stability_summary)
            main_df.to_csv(config.FINAL_REPORT_DIR / "seed2026_main_results.csv", index=False)
            if main_prediction_rows:
                pd.DataFrame(main_prediction_rows).to_csv(
                    config.FINAL_REPORT_DIR / "seed2026_external_predictions_all_models.csv",
                    index=False,
                    encoding="utf-8",
                )
        selected_results = pd.DataFrame()
        selected_summary = pd.DataFrame()
        if args.mode == "selected_after_seed2026":
            selected_created_at = datetime.now(timezone.utc).isoformat()

            def update_selected_manifest(
                completed: list[int],
                missing: list[int],
            ) -> None:
                selected_manifest = {
                    "mode": args.mode,
                    "min_external_accuracy": args.min_external_accuracy,
                    "seeds": stability_seeds,
                    "n_seeds": len(stability_seeds),
                    "worker_count": worker_count,
                    "worker_policy": "automatic_physical_cores_minus_one",
                    "resume": args.resume,
                    "force_rerun": args.force_rerun,
                    "selected_model_count": len(selected_df),
                    "selected_model_file": str(selected_file),
                    "selection_source": (
                        str(selected_df["selection_source"].iloc[0])
                        if not selected_df.empty
                        else str(main_results_path)
                    ),
                    "feature_policy": (
                        "reuse complete valid output/features only; "
                        "no extraction and no local surrogate"
                    ),
                    "checkpoint_dir": str(checkpoint_dir),
                    "completed_seed_count": len(completed),
                    "expected_seed_count": len(stability_seeds),
                    "completed_seed_ids": completed,
                    "missing_seed_ids": missing,
                    "created_at": selected_created_at,
                    "cv_mode": args.cv_mode,
                }
                write_selected_manifest(selected_paths["manifest"], selected_manifest)

            update_selected_manifest(completed_seed_ids, missing_seed_ids)
            try:
                seeds_to_run = (
                    list(stability_seeds)
                    if args.force_rerun or not args.resume
                    else list(missing_seed_ids)
                )
                if seeds_to_run:
                    worker_count = automatic_worker_count(len(seeds_to_run))
                    print(f"Automatic worker count: {worker_count}")
                    progress.update(worker_count=worker_count)
                    update_selected_manifest(completed_seed_ids, missing_seed_ids)
                    jobs = (
                        delayed(run_selected_seed_checkpoint)(
                            seed,
                            blocks_by_variant,
                            y_train,
                            y_test,
                            args.cv_mode,
                            selected_df,
                            selected_specs_by_variant,
                            seed_checkpoint_path(checkpoint_dir, seed),
                        )
                        for seed in seeds_to_run
                    )
                    parallel = Parallel(
                        n_jobs=worker_count,
                        backend="loky",
                        return_as="generator_unordered",
                    )
                    with parallel_config(backend="loky", inner_max_num_threads=1):
                        completed_generator = parallel(jobs)
                        seed_bar = tqdm(
                            completed_generator,
                            total=len(seeds_to_run),
                            desc="selected seed progress",
                            unit="seed",
                        )
                        for seed in seed_bar:
                            completed_runs += len(selected_df)
                            completed_seed_ids, missing_seed_ids = checkpoint_seed_status(
                                checkpoint_dir, stability_seeds, selected_df, args.cv_mode
                            )
                            seed_bar.set_postfix(
                                current_seed=seed,
                                completed_seeds=f"{len(completed_seed_ids)}/{len(stability_seeds)}",
                            )
                            progress.update(
                                current_seed=int(seed),
                                current_selected_model_index=len(selected_df),
                                completed_seed_count=len(completed_seed_ids),
                                completed_seed_ids=completed_seed_ids,
                                missing_seed_ids=missing_seed_ids,
                                model_progress={"completed_model_runs": completed_runs},
                            )
                            update_selected_manifest(completed_seed_ids, missing_seed_ids)
                        seed_bar.close()
            finally:
                selected_results, completed_seed_ids, missing_seed_ids = merge_seed_checkpoints(
                    checkpoint_dir, stability_seeds, selected_df, args.cv_mode
                )
                progress.update(
                    completed_seed_count=len(completed_seed_ids),
                    completed_seed_ids=completed_seed_ids,
                    missing_seed_ids=missing_seed_ids,
                    model_progress={
                        "completed_model_runs": len(selected_results),
                    },
                )
                update_selected_manifest(completed_seed_ids, missing_seed_ids)
            if missing_seed_ids:
                raise RuntimeError(
                    "Selected stability is incomplete. "
                    f"Missing seed checkpoints: {missing_seed_ids}"
                )
            if not selected_results.empty:
                selected_summary = summarize_selected_stability(selected_results, selected_df)
            selected_results.to_csv(selected_paths["results_csv"], index=False, encoding="utf-8")
            selected_summary.to_csv(selected_paths["summary_csv"], index=False, encoding="utf-8")
            write_selected_summary_md(selected_summary, selected_paths["summary_md"])
        if not stability_df.empty:
            stability_df = attach_accuracy_ci(stability_df, stability_summary)
            stability_df.to_csv(config.FINAL_REPORT_DIR / "seed_stability_results.csv", index=False)
            stability_summary.to_csv(config.FINAL_REPORT_DIR / "seed_stability_summary.csv", index=False)

        report_main = main_df if not main_df.empty else _load_csv(config.FINAL_REPORT_DIR / "seed2026_main_results.csv")
        report_stability = (
            stability_df if not stability_df.empty
            else _load_csv(config.FINAL_REPORT_DIR / "seed_stability_results.csv")
        )
        report_summary = (
            stability_summary if not stability_summary.empty
            else _load_csv(config.FINAL_REPORT_DIR / "seed_stability_summary.csv")
        )
        if args.mode != "selected_after_seed2026" and not report_main.empty and not report_summary.empty:
            report_main = attach_accuracy_ci(report_main, report_summary)
            report_main.to_csv(config.FINAL_REPORT_DIR / "seed2026_main_results.csv", index=False)

        early_gain = earlyv1_gain_over_earlyv0(report_main) if not report_main.empty else pd.DataFrame()
        scale_seed = scale_gain_seed2026(report_main) if not report_main.empty else pd.DataFrame()
        scale_stability_df = (
            scale_gain_stability(report_stability) if not report_stability.empty else pd.DataFrame()
        )
        for frame, filename in [
            (early_gain, "earlyv1_gain_over_earlyv0.csv"),
            (scale_seed, "scale_gain_seed2026.csv"),
            (scale_stability_df, "scale_gain_stability.csv"),
        ]:
            if not frame.empty:
                frame.to_csv(config.FINAL_REPORT_DIR / filename, index=False)
        early_distribution_summary(early, config.FINAL_REPORT_DIR / "early_distribution_summary.csv")

        feature_summary = {
            "early": {"earlyv0": "completed", "earlyv1": "completed"},
            "middle": middle_manifest,
            "late": late_manifest,
        }
        manifest = {
            "project": config.PROJECT_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_mode": args.mode,
            "cv_mode": args.cv_mode,
            "validation_level": args.validate,
            "main_seed": config.PRIMARY_SEED,
            "stability_seeds": stability_seeds if args.mode in {"stability", "all", "selected_after_seed2026"} else [],
            "n_model_specs_per_early_variant": config.expected_specs_per_variant(),
            "n_expected_main_rows": config.expected_main_rows(),
            "n_expected_stability_rows": config.expected_main_rows() * len(stability_seeds),
            "raw_data_source": raw_source,
            "api_mode": feature_source,
            "local_surrogate_allowed": False,
            "v2_compatible_fixes": {
                "sample_id_schema": "AD_0001/CTRL_0001/TEST_0001",
                "early_feature_family": "early_v5_mild_sensitive_v2compat",
                "middle_feature_mode": config.MIDDLE_FEATURE_MODE,
                "middle_keep_dims": int(config.MIDDLE_KEEP_DIMS),
                "svc_probability": True,
                "poly_coef0_grid": config.POLY_COEF0,
                "prediction_output": "seed2026_external_predictions_all_models.csv",
            },
            "purged_non_api_artifacts": purged,
        }
        _write_manifest(manifest)
        write_notebook(config.NOTEBOOK_DIR / "stagev3_result_check.ipynb")
        write_summary_md(
            config.FINAL_REPORT_DIR / "stagev3_summary.md",
            manifest, preprocess_summary, feature_summary,
            report_main, report_summary, early_gain, scale_seed, scale_stability_df, "PENDING",
        )
        generate_figures(config.FINAL_REPORT_DIR)
        progress.write_markdown()
        validation_ok, _ = validate_outputs(
            args.mode,
            stability_seeds,
            config.FINAL_REPORT_DIR,
            args.validate,
            selected_model_count=len(selected_df),
            selected_paths=selected_paths,
        )
        write_summary_md(
            config.FINAL_REPORT_DIR / "stagev3_summary.md",
            manifest, preprocess_summary, feature_summary,
            report_main, report_summary, early_gain, scale_seed, scale_stability_df,
            "PASS" if validation_ok else "FAIL",
        )
        progress.complete({
            "seed2026_main_results": len(main_df),
            "seed_stability_results": len(stability_df),
            "seed_stability_summary": len(stability_summary),
            "selected_seed_stability_results": len(selected_results),
            "selected_seed_stability_summary": len(selected_summary),
        })
        print(f"[stagev3] validation={'PASS' if validation_ok else 'FAIL'}")
        print(f"[stagev3] final report: {config.FINAL_REPORT_DIR}")
        if not args.no_zip:
            out_zip = config.PROJECT_ROOT.parent / "stagev3.zip"
            _zip_project(config.PROJECT_ROOT, out_zip)
            print(f"[stagev3] wrote zip: {out_zip}")
    except BaseException as exc:
        progress.fail(stage, exc)
        raise


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()

    if args.dry_run:
        lines, ok = dry_run_lines()
        print("\n".join(lines))
        raise SystemExit(0 if ok else 1)

    if args.diagnose_middle_cache:
        raise SystemExit(_diagnose_middle_cache(args))

    api_key = _print_environment_check()
    if args.check_api:
        raise SystemExit(_run_api_check(api_key))

    try:
        _run_pipeline(args, api_key)
    except MaaSAPIError as exc:
        raise SystemExit(f"Huawei MaaS API failed: {exc.kind}\n{exc}") from None
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
