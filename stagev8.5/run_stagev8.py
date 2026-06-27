from __future__ import annotations
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


def _event_loop_policy() -> None:
    if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _seeds(value: str) -> list[int]:
    value = value.strip()
    if "-" in value:
        lo, hi = value.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def _write_global_rerun_manifest(root: Path, payload: dict) -> Path:
    path = root / "output" / "checks" / "stagev8_5_global_rerun_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def global_rerun(root: Path, args, self_check, rebuild_all_features_from_raw, anchor_check, train_preflight, run_training) -> None:
    """Destructive, auditable full Stagev8.5 rerun from raw CSV through rendered notebook.

    The mode always discards generated Stagev8.5 feature/runtime/model/report outputs and
    rebuilds them from the bundled raw CSV files. It never alters input/raw or hash-locked
    reference assets. Fresh API extraction is mandatory; an API/cache replay cannot satisfy
    this mode.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    self_check(root, args.stagev6_root)
    feature_audit = rebuild_all_features_from_raw(root, force=True)
    anchor_check(root)
    preflight = train_preflight(root, args.stagev6_root)
    if not bool(preflight.get("would_allow_training", False)):
        raise RuntimeError("Global rerun preflight refused training after fresh extraction.")
    run_training(root, args.n_jobs, args.bootstrap_n, _seeds(args.stability_seeds), True, args.stagev6_root)
    render_notebook(root)
    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "complete",
        "mode": "global_rerun",
        "started_at": started_at,
        "completed_at": completed_at,
        "feature_rebuild": "fresh_api_from_raw",
        "feature_extraction_called": True,
        "api_called": True,
        "cache_replay_allowed": False,
        "training_called": True,
        "notebook_rendered": True,
        "bootstrap_n": int(args.bootstrap_n),
        "stability_seeds": _seeds(args.stability_seeds),
        "n_jobs": int(args.n_jobs),
        "fresh_feature_audit": feature_audit,
        "preflight": preflight,
        "output_policy": "generated Stagev8.5 output was invalidated before rerun; input/raw and reference assets were retained",
    }
    path = _write_global_rerun_manifest(root, manifest)
    print(json.dumps({"status": "complete", "global_rerun_manifest": str(path), "fresh_feature_rebuild": True, "notebook_rendered": True}, indent=2))


def render_notebook(root: Path) -> None:
    _event_loop_policy()
    notebook = root / "notebooks" / "stagev8_5_result_audit.ipynb"
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(notebook)], cwd=str(root), check=True)
    print(f"Notebook rendered: {notebook}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stagev8.5: fresh Stagev5 E/M/L reconstruction + MMSE-informed ordinal severity analysis")
    parser.add_argument("--mode", choices=["self_check", "check_api", "extract_features", "anchor_check", "train_preflight", "train", "all", "global_rerun", "render_notebook"], default="self_check")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-n", type=int, default=200)
    parser.add_argument("--stability-seeds", default="0-29")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stagev6-root", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"Run from the project root: {root}")
    from src.raw_feature_rebuild import check_api_environment, rebuild_all_features_from_raw
    from src.stagev8_5_train import anchor_check, run_training, self_check, train_preflight

    if args.mode == "self_check":
        self_check(root, args.stagev6_root); return
    if args.mode == "check_api":
        print(json.dumps(check_api_environment(), indent=2)); return
    if args.mode == "extract_features":
        print(json.dumps(rebuild_all_features_from_raw(root, args.force), indent=2, default=str)); return
    if args.mode == "anchor_check":
        anchor_check(root); return
    if args.mode == "train_preflight":
        train_preflight(root, args.stagev6_root); return
    if args.mode == "train":
        train_preflight(root, args.stagev6_root)
        run_training(root, args.n_jobs, args.bootstrap_n, _seeds(args.stability_seeds), args.force, args.stagev6_root)
        return
    if args.mode == "global_rerun":
        global_rerun(root, args, self_check, rebuild_all_features_from_raw, anchor_check, train_preflight, run_training)
        return
    if args.mode == "all":
        self_check(root, args.stagev6_root)
        rebuild_all_features_from_raw(root, args.force)
        train_preflight(root, args.stagev6_root)
        run_training(root, args.n_jobs, args.bootstrap_n, _seeds(args.stability_seeds), args.force, args.stagev6_root)
        return
    if args.mode == "render_notebook":
        render_notebook(root); return


if __name__ == "__main__":
    main()
