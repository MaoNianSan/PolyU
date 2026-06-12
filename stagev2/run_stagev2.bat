@echo off
setlocal
pushd "%~dp0"
echo Using the current system or Conda Python. MAAS_API_KEY must already be set.
python --version
if errorlevel 1 goto :error
where python
if errorlevel 1 goto :error
python -m pip --version
if errorlevel 1 goto :error
if "%MAAS_API_KEY%"=="" (
  echo ERROR: MAAS_API_KEY is not set.
  popd
  exit /b 1
)
python .\run_stagev2.py --data-root .\input\raw %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:error
echo ERROR: Python environment check failed.
popd
exit /b 1
