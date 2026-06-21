from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

from src import config as cfg
from src.cascade_train import train_stagev6
from src.self_check import run_self_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "stagev6: late-first cascade using strict stagev5 E/M/L feature files. "
            "No API calls and no feature extraction are performed during training."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["self_check", "train", "all", "render_notebook"],
        help="self_check validates inherited features; train fits the 6 fixed cascade specifications.",
    )
    parser.add_argument("--n-jobs", default="-1", help="GridSearchCV worker count, e.g. 12.")
    parser.add_argument("--bootstrap-n", type=int, default=cfg.BOOTSTRAP_N)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove prior stagev6 run/final outputs before writing fresh results.",
    )
    return parser.parse_args()


def _jupyter_command(*args: str) -> list[str]:
    if sys.platform.startswith("win"):
        bootstrap = (
            "import asyncio, runpy;"
            "asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy());"
            "runpy.run_module('jupyter', run_name='__main__')"
        )
        return [sys.executable, "-c", bootstrap, *args]
    return [sys.executable, "-m", "jupyter", *args]


def _jupyter_env() -> dict[str, str]:
    env = os.environ.copy()
    warning = "ignore:Proactor event loop does not implement add_reader:RuntimeWarning:zmq._future"
    existing = env.get("PYTHONWARNINGS")
    env["PYTHONWARNINGS"] = f"{existing},{warning}" if existing else warning
    return env


def render_notebook() -> None:
    notebook = cfg.ROOT / "notebooks" / "stagev6_result_check.ipynb"
    subprocess.run(
        _jupyter_command(
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(notebook),
        ),
        cwd=str(cfg.ROOT),
        env=_jupyter_env(),
        check=True,
    )
    print(f"Notebook rendered: {notebook}")


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != cfg.ROOT:
        raise RuntimeError(f"Run from the stagev6 project root: {cfg.ROOT}")
    if args.mode == "self_check":
        run_self_check(require_features=True)
        return
    if args.mode == "train":
        train_stagev6(n_jobs=args.n_jobs, bootstrap_n=args.bootstrap_n, overwrite=args.overwrite)
        return
    if args.mode == "all":
        run_self_check(require_features=True)
        train_stagev6(n_jobs=args.n_jobs, bootstrap_n=args.bootstrap_n, overwrite=args.overwrite)
        return
    if args.mode == "render_notebook":
        render_notebook()
        return


if __name__ == "__main__":
    main()
