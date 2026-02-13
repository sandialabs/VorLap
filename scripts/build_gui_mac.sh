#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
pyinstaller --noconfirm --clean --onefile --windowed --name vorlap_gui scripts/launch_gui.py
