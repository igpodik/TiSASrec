@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating venv...
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
)
set PYTHON=.venv\Scripts\python.exe
if "%1"=="train" (
    %PYTHON% train.py %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="predict" (
    %PYTHON% predict.py %2 %3 %4 %5 %6 %7 %8 %9
) else (
    echo Usage: run.bat train [--cpu-only]
    echo        run.bat predict [--cpu-only] [--out submission.csv]
)
