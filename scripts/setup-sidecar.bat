@echo off
REM Create a TORCH-FREE python env for tools/text_sidecar.py and install its deps.
REM torch must not be present: some frontend deps import it when available and that adds
REM 10-20 s to cold start. Usage: scripts\setup-sidecar.bat [python.exe]
setlocal
set PY=%~1
if "%PY%"=="" set PY=python
cd /d "%~dp0.."
if not exist .venv-sidecar (
  echo [sidecar] creating .venv-sidecar with %PY%
  %PY% -m venv .venv-sidecar || exit /b 1
)
.venv-sidecar\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.venv-sidecar\Scripts\python.exe -m pip install -r frontend\requirements-zh.txt || exit /b 1
.venv-sidecar\Scripts\python.exe -m pip install -r frontend\requirements-multi.txt
if errorlevel 1 echo [sidecar] warning: some multi-language deps failed (zh still works)
echo [sidecar] prefetching nltk data (cmudict / perceptron tagger)...
if not exist .nltk_data mkdir .nltk_data
set NLTK_DATA=%CD%\.nltk_data
.venv-sidecar\Scripts\python.exe -c "import nltk; nltk.download('cmudict', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True)"
if errorlevel 1 echo [sidecar] warning: nltk download failed (english G2P may not work)
echo SIDECAR_ENV_OK -^> .venv-sidecar
