param(
    [int]$Jobs = 12,
    [int]$BootstrapN = 200,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python .\run_stagev6.py --mode self_check

$arguments = @(".\run_stagev6.py", "--mode", "train", "--n-jobs", $Jobs, "--bootstrap-n", $BootstrapN)
if ($Overwrite) { $arguments += "--overwrite" }
python @arguments
