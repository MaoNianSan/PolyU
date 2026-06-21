from __future__ import annotations
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from src.stagev7_core import run_training, self_check, train_preflight


def set_windows_notebook_event_loop_policy() -> None:
    if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def render_notebook(root: Path) -> None:
    """Execute the read-only result notebook against saved output artifacts only."""
    set_windows_notebook_event_loop_policy()
    nb = root / 'notebooks' / 'stagev7_result_check.ipynb'
    if not nb.exists():
        raise FileNotFoundError(f'Result notebook not found: {nb}')
    bootstrap = (
        "import asyncio, sys\n"
        "if sys.platform.startswith('win') and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):\n"
        "    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())\n"
        "from nbconvert.nbconvertapp import main\n"
        "main()\n"
    )
    subprocess.run(
        [sys.executable, '-c', bootstrap, '--to', 'notebook', '--execute', '--inplace', str(nb)],
        cwd=str(root),
        check=True,
    )
    print(f'Notebook rendered: {nb}')


def main() -> None:
    p=argparse.ArgumentParser(description='stagev7 strict stagev5-feature cascade classifier')
    p.add_argument('--mode', default='self_check', choices=['self_check','train_preflight','train','all','render_notebook'])
    p.add_argument('--n-jobs', default='-1')
    p.add_argument('--bootstrap-n', type=int, default=200)
    p.add_argument('--force', action='store_true', help='Overwrite stagev7 run outputs.')
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    if Path.cwd().resolve()!=root:
        raise RuntimeError(f'Run from project root: {root}')
    if args.mode=='self_check':
        self_check(root)
    elif args.mode=='train_preflight':
        train_preflight(root)
    elif args.mode=='render_notebook':
        render_notebook(root)
    elif args.mode in {'train','all'}:
        self_check(root)
        run_training(root, n_jobs=int(args.n_jobs), bootstrap_n=args.bootstrap_n, force=args.force)


if __name__=='__main__':
    main()
