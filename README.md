# VorLap

[![Tests](https://github.com/sandialabs/VorLap/actions/workflows/tests.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/tests.yml)
[![Docs](https://github.com/sandialabs/VorLap/actions/workflows/docs.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/docs.yml)
[![GUI Builds](https://github.com/sandialabs/VorLap/actions/workflows/build_gui.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/build_gui.yml)
[![Download](https://img.shields.io/github/v/release/sandialabs/VorLap?label=download)](https://github.com/sandialabs/VorLap/releases/latest)

VorLap (Vortex overLAP) predicts aerodynamic force spectra, force reconstruction, and frequency overlap risk for rotating structures using FFT-based airfoil databases.

## Features

- Load component geometry from CSV.
- Load unsteady airfoil FFT databases from HDF5.
- Compute force/moment maps over inflow and azimuth sweeps.
- Reconstruct node-level time series from spectral coefficients.
- Reconstruct node-level time histories for time-varying inflow speed and direction using a profile CSV.
- Compare shedding frequencies against parked natural frequencies.
- Use a Tkinter GUI for setup, execution, and plotting.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs,gui]"
```

Run tests:

```bash
pytest
```

Build docs:

```bash
mkdocs build --strict
```

Launch GUI:

```bash
python scripts/launch_gui.py
```

Run a scripted time-varying inflow example:

```bash
python examples/time_varying_inflow_case.py
```

Inflow profile CSV format:

```text
time,inflow_speed,inflow_direction_deg
0.0,6.0,0.0
0.5,7.0,5.0
1.0,8.0,10.0
```

Alternative direction columns are also supported: `inflow_dir_x`, `inflow_dir_y`, and optional `inflow_dir_z`.

## Project Layout

- `vorlap/`: package source
- `tests/`: automated test suite
- `examples/`: runnable example scripts
- `docs/`: MkDocs documentation
- `data/`: component, airfoil, and reference input data

## Verification

The repository includes a verification-style test in `tests/test_verification_case.py` based on the existing single-blade case and expected CL/CD force levels.

## License

See `LICENSE` and `NOTICE`.
