# stagev2 AD classifier

This project runs the stagev2 Alzheimer's disease connected-speech classifier from raw Cookie Theft transcript CSVs. The pipeline regenerates all features before model training:

```text
raw transcripts
-> preprocessing with [disease, early, middle, late] labels
-> early BM25 information-unit features
-> middle Huawei MaaS BGE-M3 window embeddings
-> late qwen3-235b-a22b expressive-form LLM scores
-> stage/fusion/interaction classifiers
-> external-accuracy ranking, CV diagnostics, OOF diagnostics, reports, and notebook visualization
```

The project uses the current system Python or the currently active Conda Python. Do not create a `.venv` for this project.

## Project root

Run every installation and pipeline command from the `stagev2` project root. This is the directory containing:

```text
run_stagev2.py
requirements.txt
src/
stage_core/
```

Windows PowerShell example:

```powershell
cd D:\research\H.L.Liang-Lab\Code\expore\stagev2
```

If the project is stored elsewhere, replace that path with the actual `stagev2` project root.

## Python environment check

Python 3.10 or 3.11 is recommended. Check the interpreter and its associated pip before installing dependencies:

```powershell
python --version
where.exe python
python -m pip --version
```

In PowerShell, use `where.exe python` because `where` is also a PowerShell alias. If it reports multiple interpreters, confirm that the first interpreter is the system Python or Conda Python you intend to use. `python -m pip` installs packages into the interpreter selected by `python`.

## Install dependencies

From the `stagev2` project root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data input

Place the raw data under:

```text
stagev2\input\raw\
```

Expected directory layout:

```text
input/
  raw/
    ad_s2t_wav2vec.csv
    control_s2t_wav2vec.csv
    test_s2t_wav2vec.csv
```

The required raw columns are:

```text
Speech, label, mmse
```

You may instead pass a zip file containing those three raw CSV files. Only the raw CSVs are extracted; generated files in the zip are ignored.

The pipeline does not read old output, embeddings, BM25 features, LLM features, caches, or model files as input. A normal run removes the old output and extracts every feature again from the raw CSV files.

## API configuration

Both middle embeddings and late LLM features use the same Huawei MaaS client and the same required environment variable:

```text
Huawei MaaS base URL: https://api.modelarts-maas.com/v1
Required API key env var: MAAS_API_KEY
Optional base URL env var: HUAWEI_MAAS_BASE_URL
Middle embedding model: bge-m3
Late LLM model: qwen3-235b-a22b
Late prompt version: late_form_masked_v1
```

There is no `DASHSCOPE_API_KEY` setting in this project. Set the MaaS key in the current PowerShell session before running:

```powershell
$env:MAAS_API_KEY="your_real_MAAS_API_KEY"
```

Set `HUAWEI_MAAS_BASE_URL` only when the default endpoint must be overridden:

```powershell
$env:HUAWEI_MAAS_BASE_URL="https://api.modelarts-maas.com/v1"
```

Do not commit real API keys.

## Complete Windows PowerShell run

Run this complete flow from the `stagev2` project root:

```powershell
cd D:\research\H.L.Liang-Lab\Code\expore\stagev2

python --version
where.exe python
python -m pip --version

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:MAAS_API_KEY="your_real_MAAS_API_KEY"

Remove-Item -Recurse -Force .\output -ErrorAction SilentlyContinue

python .\run_stagev2.py --data-root .\input\raw
```

For zip input, remain in the same `stagev2` project root and run:

```powershell
python .\run_stagev2.py --data-zip .\input\raw_data.zip
```

The standard global workflow intentionally performs a clean run. `--reuse-cache` is available only for deliberate debugging or resumption and is not part of the clean full-run procedure.

## Experimental policy

The current project policy is:

```text
Main label order: [disease, early, middle, late]
MMSE thresholds: early = 21-24, middle = 13-20, late = <=12
GridSearchCV scoring: accuracy
Classifier threshold: 0.5
Model ranking/selection: held-out external accuracy
CV and OOF predictions: internal diagnostics only
External set role: held-out external validation, not an unbiased final test after selection
RBF SVM: retained as nonlinear baseline
All fitted models: saved locally under output/models/all_models/
Selected model: saved under output/models/selected/
```

## Canonical outputs

After a full run, the main results are copied to:

```text
output/stagev2/final_report/stagev2_model_ranking_by_external_accuracy.csv
output/stagev2/final_report/stagev2_external_performance_report.csv
output/stagev2/final_report/stagev2_cv_summary.csv
output/stagev2/final_report/stagev2_oof_predictions_top10.csv
output/stagev2/final_report/stagev2_test_predictions_all_models.csv
output/stagev2/final_report/stagev2_selected_model_summary.md
output/stagev2/final_report/stagev2_experiment_report.md
output/stagev2/final_report/stagev2_leakage_check.json
```

The full run directory is:

```text
output/stagev2/run_external_accuracy_selection/
```

## Notebook visualization

Open:

```text
notebooks/stagev2_visualization.ipynb
```

The notebook reads generated CSV files only. It does not retrain models or call APIs.

## GitHub maintenance

The repository should track code, notebooks, docs, and configuration examples only. Local data, output, caches, and models are excluded by `.gitignore`.
