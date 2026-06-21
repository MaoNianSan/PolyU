# stagev7

Strict **stagev5-feature** hierarchical classifier for Cookie Theft AD analysis.

## Decision path

1. Late vs non-late.
2. For samples not predicted late: middle vs remaining non-late.
3. For samples not predicted middle: early-spectrum AD vs control.

Output stage labels are `control`, `early_spectrum_AD`, `middle`, and `late`. All non-control stages map to `predicted_AD=1`.

`AD_high_MMSE` is never used as a model feature. It is retained as a strict audit label and grouped with early only in the final early-spectrum gate.

## Non-negotiable feature policy

The project reads only copied stagev5 generated features:

- E: 61 stagev2 BM25 features;
- M: 1024 stagev2 BGE-M3 embedding dimensions, averaged over windows per sample;
- L: 8 stagev4 unmasked raw F8 scores.

No APIs, LLM scoring, embeddings, BM25 extraction, or stagev5 feature regeneration occur in `train` mode.

## Commands

```powershell
python run_stagev7.py --mode self_check
python run_stagev7.py --mode train_preflight
python run_stagev7.py --mode train --n-jobs 12 --bootstrap-n 200
python run_stagev7.py --mode render_notebook
```

If re-running after output generation:

```powershell
python run_stagev7.py --mode train --n-jobs 12 --bootstrap-n 200 --force
```

`self_check`, `train_preflight`, and `render_notebook` do not train models, regenerate features, run bootstrap, or call APIs. `train_preflight` verifies the copied E/M/L feature dimensions, sample counts, cascade definitions, flat baselines, output writability, and completion-sentinel status.

Default training is blocked only when the explicit completion sentinel exists:

```text
output/final_report/stagev7_training_complete.json
```

Legacy `stagev7_*` CSV/JSON/Markdown/PNG files in `output/final_report/` do not by themselves mark training as complete. `--force` overwrites only known stagev7 final-report outputs and the completion sentinel; it does not remove `output/features/`, `input/`, cache files, or notebooks.

## Main outputs

`notebooks/stagev7_result_check.ipynb` is a display-only audit notebook. It reads only saved CSV/JSON/Markdown/PNG artifacts. The safe renderer is:

```powershell
python run_stagev7.py --mode render_notebook
```

The notebook is resilient to missing outputs: it displays an availability notice rather than starting training or raising a missing-file error.

`output/final_report/` contains:

- gate-level CV rankings;
- all predefined cascade predictions and exploratory external ranking;
- the pre-specified primary C06 cascade prediction file;
- binary and collapsed-stage confusion matrices;
- strict stage audit retaining `AD_high_MMSE`;
- bootstrap confidence intervals;
- four flat multiclass comparison baselines;
- a rendered read-only result audit notebook;
- feature provenance and leakage-check JSON files.

The primary cascade is pre-specified as:

```text
C06 = Late: LR-L2(M+L) -> Middle: SVC-poly3(E+M) -> Early-spectrum: SVC-poly3(E+M)
```

The external ranking across all 20 predefined cascade systems is exploratory and must not be used as a confirmatory model-selection claim.
