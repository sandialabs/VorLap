# Quick Start

## 1. Create an Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs,gui]"
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

## 8. Run The Headless Parked QBlade Sweep

The retained cluster runner lives in `run_headless/`. Install QBlade CE
2.0.9.7 separately; the historical launchers expect it at
`run_headless/QBladeCE_2.0.9.7` unless their documented override is set.
The SIL CLI accepts `QBLADE_ROOT` or `--qblade-root`.

```bash
export PYTHON_EXE=/path/to/python
sbatch run_headless/submit_vorlap_parked.sj
```

Use the same `PYTHON_EXE` for VorLap installation, QBlade SIL execution, and `scripts/build_qblade_external_linux.sh` so the compiled controller bridge links against the Python runtime QBlade will load. The default Slurm array runs 13 parked cases from 2 to 20 m/s; set `WIND_SPEEDS="2 8 14 20"` for a smaller four-case sweep.

After the run:

```bash
scripts/plot_qblade_parked_sweep.py --run-root run_headless/runs/vorlap_parked
```

For the current SIL manuscript's exact inputs, replay commands, and scientific
limits, use [paper reproducibility](reproducibility.md). The historical parked
sweep above is a smaller workflow, not the current paper's full campaign.
