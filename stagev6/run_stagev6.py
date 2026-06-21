from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
from src import stagev6_config as cfg
from src.runner import run
from src.self_check import run_self_check


def parse_args():
    p=argparse.ArgumentParser(description='stagev6: late-first cascade using existing stagev5 E/M/L features only.')
    p.add_argument('--mode',required=True,choices=['self_check','train','all','render_notebook'])
    p.add_argument('--n-jobs',type=int,default=-1,help='GridSearchCV worker count, e.g. 12.')
    p.add_argument('--bootstrap-n',type=int,default=cfg.BOOTSTRAP_N)
    # Internal debugging cap. Omit for the full stagev5-scale component family.
    p.add_argument('--max-gate-models',type=int,default=None,help=argparse.SUPPRESS)
    p.add_argument('--max-branch-models',type=int,default=None,help=argparse.SUPPRESS)
    return p.parse_args()


def _missing_final_outputs():
    return [name for name in cfg.CANONICAL_FINAL_FILES if not (cfg.FINAL / name).exists()]


def _jupyter_command(*args: str):
    if sys.platform.startswith("win"):
        bootstrap = (
            "import asyncio, runpy;"
            "asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy());"
            "runpy.run_module('jupyter', run_name='__main__')"
        )
        return [sys.executable, "-c", bootstrap, *args]
    return [sys.executable, "-m", "jupyter", *args]


def _jupyter_env():
    env = os.environ.copy()
    warning = "ignore:Proactor event loop does not implement add_reader:RuntimeWarning:zmq._future"
    existing = env.get("PYTHONWARNINGS")
    env["PYTHONWARNINGS"] = f"{existing},{warning}" if existing else warning
    return env


def render_notebook():
    nb=cfg.ROOT/'notebooks'/'stagev6_result_check.ipynb'
    missing = _missing_final_outputs()
    if missing:
        preview = ", ".join(missing[:3])
        more = "" if len(missing) <= 3 else f", ... ({len(missing)} files missing)"
        raise RuntimeError(
            "Cannot render the notebook before Stagev6 training outputs exist. "
            f"Missing: {preview}{more}. "
            "Run: python .\\run_stagev6.py --mode train --n-jobs 12 --bootstrap-n 200"
        )
    subprocess.run(_jupyter_command('nbconvert','--to','notebook','--execute','--inplace',str(nb)),cwd=str(cfg.ROOT),env=_jupyter_env(),check=True)
    print(f'Notebook rendered: {nb}')


def main():
    args=parse_args()
    if Path.cwd().resolve()!=cfg.ROOT:
        raise RuntimeError(f'Run from stagev6 project root: {cfg.ROOT}')
    if args.mode=='self_check': run_self_check(); return
    if args.mode=='train': run(args.n_jobs,args.bootstrap_n,args.max_gate_models,args.max_branch_models); return
    if args.mode=='all': run_self_check(); run(args.n_jobs,args.bootstrap_n,args.max_gate_models,args.max_branch_models); return
    if args.mode=='render_notebook': render_notebook(); return

if __name__=='__main__': main()
