$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
python -m PyInstaller --noconfirm --clean --onefile --windowed --name vorlap_gui --paths . --collect-submodules vorlap --collect-data vorlap scripts/launch_gui.py
