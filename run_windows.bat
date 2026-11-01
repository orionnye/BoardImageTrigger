@echo off
rem Board Capture launcher for Windows. First run sets up a private Python environment.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo First run: setting up Python environment...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist .venv\Scripts\python.exe (
        echo Python 3.11 or newer is required. Install it from python.org and try again.
        pause
        exit /b 1
    )
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python board_capture\main.py %*
if errorlevel 1 pause
