$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent)
python .\run_stagev5.py --mode self_check
python .\scripts\check_existing_results.py
python .\scripts\summarize_outputs.py
# Environment-only API readiness check; makes no API request:
# python .\run_stagev5.py --mode check_api
# Controlled full rerun only; may regenerate features and call APIs:
# python .\run_stagev5.py --mode all --n-jobs 12 --bootstrap-n 200
# Re-run only training from existing E/M/L feature CSV files:
# python .\run_stagev5.py --mode train --n-jobs 12 --bootstrap-n 200
# Execute display-only notebook after results exist:
# python .\run_stagev5.py --mode render_notebook
