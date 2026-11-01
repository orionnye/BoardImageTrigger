#!/usr/bin/env bash
# Board Capture launcher for macOS / Linux. First run sets up a private Python environment.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "First run: setting up Python environment..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python board_capture/main.py "$@"
