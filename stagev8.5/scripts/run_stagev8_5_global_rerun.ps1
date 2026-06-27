$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:HUAWEI_MAAS_TRUST_ENV = "false"
$env:HUAWEI_MAAS_SSL_VERIFY = "true"
Remove-Item Env:OPENSSL_CONF -ErrorAction SilentlyContinue
python .\run_stagev8.py --mode global_rerun --n-jobs 12 --bootstrap-n 200 --stability-seeds 0-29
