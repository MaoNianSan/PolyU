from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src import stagev5_config as cfg
from src.feature_adapter import extract_E_M, extract_L, validate_feature_outputs
from src.postprocess import postprocess_stagev5
from src.self_check import check_api_environment, run_self_check


def parse_args() -> argparse.Namespace:
    """Parse CLI options without triggering feature extraction or training."""
    parser = argparse.ArgumentParser(
        description="stagev5: strict E/M(stagev2) + L(stagev4_unmasked), stagev2 classifier benchmark."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["self_check", "check_api", "extract_features", "train", "all", "render_notebook"],
    )
    parser.add_argument("--n-jobs", default="-1", help="GridSearchCV worker count; e.g. -1 or 12.")
    parser.add_argument("--bootstrap-n", type=int, default=cfg.BOOTSTRAP_N)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate E/M/L feature files rather than reusing valid outputs/cache.",
    )
    parser.add_argument(
        "--no-reference-middle-cache",
        action="store_true",
        help="Do not stage the supplied stagev2 BGE-M3 cache before E/M extraction.",
    )
    return parser.parse_args()


def extract(args: argparse.Namespace) -> None:
    """Run the locked E/M/L feature extractors."""
    # E and M must run through stagev2 reference code. L must run through stagev4 reference code.
    extract_E_M(force=args.force, reuse_reference_cache=not args.no_reference_middle_cache)
    extract_L(force=args.force)
    print("FEATURE EXTRACTION COMPLETED")


def train(args: argparse.Namespace) -> None:
    """Train the stagev2 classifier panel from existing E/M/L CSV files only."""
    validate_feature_outputs()
    env = os.environ.copy()
    env.update(
        {
            "STAGE_AD_PROJECT_ROOT": str(cfg.STAGEV2_EXTRACT_ROOT),
            "STAGE_BM25_DIR": str(cfg.EARLY_DIR),
            "STAGE_EMBEDDING_DIR": str(cfg.MIDDLE_DIR),
            "STAGE_LLM_DIR": str(cfg.LATE_DIR),
            "STAGE_PREPROCESS_DIR": str(cfg.STAGEV2_EXTRACT_ROOT / "preprocess"),
            "STAGE_OUTPUT_ROOT": str(cfg.CLASSIFIER_RUN),
            "STAGE_BOOTSTRAP_N": str(args.bootstrap_n),
            "STAGE_N_JOBS": str(args.n_jobs),
            "STAGE_DECISION_THRESHOLD": str(cfg.DECISION_THRESHOLD),
            "STAGE_SCORING": cfg.SCORING,
            "STAGEV2_ENABLE_INTERACTIONS": "1",
            "STAGE_CV_N_SPLITS": str(cfg.CV_N_SPLITS),
            "STAGE_CV_N_REPEATS": str(cfg.CV_N_REPEATS),
            "STAGE_INNER_N_SPLITS": str(cfg.INNER_N_SPLITS),
        }
    )
    cfg.CLASSIFIER_RUN.mkdir(parents=True, exist_ok=True)
    core = cfg.ROOT / "src" / "stagev2_classifier_core"
    subprocess.run([sys.executable, "run_stage_corrected.py"], cwd=str(core), env=env, check=True)
    result = postprocess_stagev5()
    print("TRAINING AND REPORTING COMPLETED")
    print(result)


def render_notebook() -> None:
    """Execute the display-only notebook against saved result files."""
    nb = cfg.NOTEBOOKS / "stagev5_result_check.ipynb"
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(nb)],
        cwd=str(cfg.ROOT),
        check=True,
    )
    print(f"Notebook rendered: {nb}")


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != cfg.ROOT:
        raise RuntimeError(f"Run from stagev5 project root: {cfg.ROOT}")
    if args.mode == "self_check":
        run_self_check(require_features=True)
        return
    if args.mode == "check_api":
        check_api_environment()
        return
    if args.mode == "extract_features":
        extract(args)
        return
    if args.mode == "train":
        train(args)
        return
    if args.mode == "all":
        run_self_check(require_features=False)
        extract(args)
        train(args)
        return
    if args.mode == "render_notebook":
        render_notebook()
        return


if __name__ == "__main__":
    main()
