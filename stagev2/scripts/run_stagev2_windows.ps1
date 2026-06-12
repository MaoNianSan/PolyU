$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Using the current system or Conda Python."
Write-Host "Set MAAS_API_KEY in this PowerShell session before running this script."

python --version
if ($LASTEXITCODE -ne 0) { throw "python --version failed." }
where.exe python
if ($LASTEXITCODE -ne 0) { throw "Python was not found on PATH." }
python -m pip --version
if ($LASTEXITCODE -ne 0) { throw "python -m pip --version failed." }

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

if ([string]::IsNullOrWhiteSpace($env:MAAS_API_KEY)) {
    throw "MAAS_API_KEY is not set. Set it in this PowerShell session before running the script."
}

Remove-Item -Recurse -Force .\output -ErrorAction SilentlyContinue

python .\run_stagev2.py --data-root .\input\raw
if ($LASTEXITCODE -ne 0) { throw "stagev2 failed with exit code $LASTEXITCODE." }
