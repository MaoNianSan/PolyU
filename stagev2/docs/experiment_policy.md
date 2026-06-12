# stagev2 experiment policy

- Runtime: use the current system Python or active Conda Python; do not create a project virtual environment.
- Working directory: run installation and pipeline commands from the `stagev2` root containing `run_stagev2.py`, `requirements.txt`, `src/`, and `stage_core/`.
- Dependency installation: `python -m pip install -r requirements.txt`.
- Standard entry point: `python .\run_stagev2.py --data-root .\input\raw`.
- Input: raw transcript CSV files only.
- Regenerated features: early BM25, middle BGE-M3, late qwen3-235b-a22b LLM scores.
- Historical outputs/features/caches/models: not reused.
- Main label order: `[disease, early, middle, late]`.
- MMSE thresholds: early = 21-24, middle = 13-20, late = <=12.
- Model selection: held-out external accuracy.
- CV and OOF: internal diagnostics only.
- External set role: held-out external validation, not unbiased final testing after model selection.
- Classifier threshold: 0.5.
- GridSearchCV scoring: accuracy.
- RBF SVM: retained as a nonlinear baseline.
