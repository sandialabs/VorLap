# Quick Start

## 1. Create an Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev,docs,gui]
```

## 2. Run the Test Suite

```bash
pytest
```

## 3. Build Documentation

```bash
mkdocs build --strict
```

## 4. Launch the GUI

```bash
python scripts/launch_gui.py
```

## 5. Run an Example Script

```bash
python examples/basic_synthetic_case.py
```

## 6. Run Time-Varying Inflow Reconstruction

```bash
python examples/time_varying_inflow_case.py
```

The inflow CSV must contain at least:

```text
time,inflow_speed,inflow_direction_deg
0.0,6.0,0.0
0.5,7.0,5.0
```

You can also provide direction vectors with `inflow_dir_x`, `inflow_dir_y`, and optional `inflow_dir_z`.

## 7. Generate a QBlade Loading File (Fast Path)

```bash
python examples/qblade_fastpath_loading.py --sim /path/to/case.sim --inflow data/inflow_profile.csv --output qblade_external_loading.txt
```

The exported file is compatible with QBlade `LOADINGFILE` parsing (`time Fx Fy Fz Mx My Mz` per target block).
