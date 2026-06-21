param(
    [int]$Jobs = 12,
    [int]$BootstrapN = 200
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python .\run_stagev6.py --mode self_check
python .\run_stagev6.py --mode train --n-jobs $Jobs --bootstrap-n $BootstrapN
