# Stagev5 Reproducibility

This project is designed to support two reproducibility levels: read-only result verification and classifier reruns from existing E/M/L feature files.

## Read-Only Verification

```powershell
python .\run_stagev5.py --mode self_check
python .\scripts\check_existing_results.py
python .\scripts\summarize_outputs.py
python .\run_stagev5.py --mode render_notebook
```

These commands do not train models, extract E/M/L features, or call APIs.

## Classifier Rerun From Existing Features

```powershell
python .\run_stagev5.py --mode train --n-jobs 12 --bootstrap-n 200
```

This reruns the stagev2 classifier panel using existing feature CSV files under `output/features/`. It preserves the fixed scientific contract: 10-fold CV, stagev2 classifier panel, external accuracy ranking, and bootstrap `n=200`.

## Full Regeneration Boundary

```powershell
python .\run_stagev5.py --mode extract_features
python .\run_stagev5.py --mode all --n-jobs 12 --bootstrap-n 200
```

These modes can regenerate E/M/L features and may call APIs. Use them only when the intention is a controlled full rerun.

## Path Policy

Project paths are resolved from `src/paths.py` and `src/stagev5_config.py`. The repository should run from a cloned checkout without user-specific absolute paths in source code.
