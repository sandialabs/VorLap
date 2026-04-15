# VorLap

<p align="center">
  <img src="docs/VorLapLogo.png" alt="VorLap Logo" width="420" />
</p>

[![Tests](https://github.com/sandialabs/VorLap/actions/workflows/tests.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/tests.yml)
[![Docs](https://github.com/sandialabs/VorLap/actions/workflows/docs.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/docs.yml)
[![Hosted Docs](https://img.shields.io/badge/docs-github_pages-blue)](https://sandialabs.github.io/VorLap/)
[![GUI Builds](https://github.com/sandialabs/VorLap/actions/workflows/build_gui.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/build_gui.yml)
[![Dev Builds (main)](https://img.shields.io/badge/dev_builds-main-blue)](https://github.com/sandialabs/VorLap/actions/workflows/build_gui.yml?query=branch%3Amain)
[![Download](https://img.shields.io/github/v/release/sandialabs/VorLap?label=download)](https://github.com/sandialabs/VorLap/releases/latest)

VorLap (Vortex overLAP) predicts aerodynamic force spectra, force reconstruction, and frequency overlap risk for rotating structures using FFT-based airfoil databases.

Hosted documentation: <https://sandialabs.github.io/VorLap/>

## Features

- Load component geometry from CSV.
- Load unsteady airfoil FFT databases from HDF5.
- Compute force/moment maps over inflow and azimuth sweeps.
- Reconstruct node-level time series from spectral coefficients.
- Reconstruct node-level time histories for time-varying inflow speed and direction using a profile CSV.
- Convert QBlade `.sim/.trb/.bld` definitions into VorLap components and export QBlade-compatible `LOADINGFILE` data.
- Compare shedding frequencies against parked natural frequencies.
- Use a Tkinter GUI for setup, execution, and plotting.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs,gui]"
```

Verify Tkinter support (required for GUI):

```bash
python -c "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.destroy(); print('tk_ok')"
```

If this fails or aborts on macOS (for example with `_tkinter` errors or Tk runtime aborts), avoid `/usr/bin/python3` for the GUI path and use one of these:

1. Dedicated Conda environment (recommended)

```bash
conda deactivate  # repeat until (base) is gone, if needed
conda create -n vorlap-gui python=3.11 -y
conda activate vorlap-gui
python -m pip install -e ".[dev,docs,gui]"
```

2. python.org framework Python + venv

```bash
deactivate  # if currently active
rm -rf .venv
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
source .venv/bin/activate
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
python3 scripts/launch_gui.py
```

If zsh reports `unknown file attribute: b`, your `python` shell alias/function is interfering with glob parsing. Use:

```bash
python3 scripts/launch_gui.py
# or explicitly:
.venv/bin/python scripts/launch_gui.py
```

Run a scripted time-varying inflow example:

```bash
python examples/time_varying_inflow_case.py
```

Run the QBlade fast-path loading export:

```bash
python examples/qblade_fastpath_loading.py --sim /path/to/case.sim --inflow data/inflow_profile.csv --output qblade_external_loading.txt
```

Build the live QBlade external-library bridge:

```bash
scripts/build_qblade_external_linux.sh
```

On Windows (PowerShell):

```powershell
scripts/build_qblade_external_windows.ps1
```

The shared library name is `libvorlap_qblade_bridge` (`.so` on Linux, `.dll` on Windows). Copy it into QBlade's `ControllerFiles` directory so QBlade can load it.

Linux bridge build + install example:

```bash
# Build using .venv python if present
scripts/build_qblade_external_linux.sh

# Or build and copy directly into a QBlade install
scripts/build_qblade_external_linux.sh \
  build/qblade_external_linux \
  /path/to/QBlade/ControllerFiles
```

If QBlade embeds a specific Python distribution at runtime, build the bridge with that exact interpreter:

```bash
PYTHON_EXE=/path/to/python3 scripts/build_qblade_external_linux.sh \
  build/qblade_external_linux \
  /path/to/QBlade/ControllerFiles
```

This is especially important for NumPy-backed embedded imports. A mismatch between the Python/NumPy used to build the bridge and the Python/NumPy seen by QBlade can surface as NumPy C-extension import failures.

Windows bridge build + install example (PowerShell):

```powershell
scripts/build_qblade_external_windows.ps1

scripts/build_qblade_external_windows.ps1 `
  -BuildDir build/qblade_external_windows `
  -InstallDir C:\path\to\QBlade\ControllerFiles
```

Prepare the `wMinSagSnubbers` QBlade case for VorLap:

```bash
python scripts/prepare_qblade_external_case.py \
  --sim ../QBlade_model_exp_9.16.25/baseline_wMinSagSnubbers-Wwnd.sim \
  --airfoils data/airfoils \
  --node-source structural \
  --n-freq-depth 20 \
  --force-scale 100 \
  --debug
```

That script updates the turbine definition with `LIBFILE_1`, `LIBFUNCTION_1`, `LIBARRAYSIZE_1`, and `LIBPARAMETERFILE_1`, appends `EXTERNAL_1_IN` / `EXTERNAL_1_OUT` tables to the structural model, and writes `Control/vorlap_qblade_external.json`.

The generated `airfoil_dir` in `vorlap_qblade_external.json` is written relative to the parameter file location when possible, so model-local layouts such as `wMinSagSnubbers/VorLapAirfoils` work across machines without hard-coded absolute paths.

The generated config also stores the original parameter-file directory as a fallback base. This allows the runtime to survive QBlade workflows that copy the JSON into a temporary run directory before calling the bridge.

The runtime config defaults to structural `BLD_*`/`STR_*` nodes so the swap mapping aligns with output locations already declared in the structural file. Use `--node-source converted` if you want all converted VorLap nodes instead.

The embedded bridge imports `vorlap.qblade_runtime` from your Python environment, so install VorLap and dependencies in the same Python used during bridge build.

QBlade-side mapping flow (general):

- `.sim`: selects turbine (`TURBFILE`) and operating conditions (`RPMPRESCRIBED`, `MEANINF`, etc.).
- `.trb`: enables external library calls via `LIBFILE_1`, `LIBFUNCTION_1`, `LIBARRAYSIZE_1`, and `LIBPARAMETERFILE_1`.
- `.str`: defines `EXTERNAL_1_IN` swap inputs (time, azimuth, node velocities) and `EXTERNAL_1_OUT` actions (`ADDFORCE`) that map returned forces to component IDs and normalized positions.
- `Control/vorlap_qblade_external.json`: high-level VorLap runtime config.
- `n_freq_depth`: number of spectral tones used per node (capped by available airfoil FFT depth).
- `force_scale`: global multiplier applied to all VorLap external forces before they are returned to QBlade.
- `debug`: enables verbose `update_message()` diagnostics, including resolved paths at init and the maximum applied force magnitude/node during updates.
- `log_file`: optional runtime log path. When `--debug` is used and no path is provided, case prep writes an absolute default path named `vorlap_qblade_debug.log` next to the generated JSON.

For lower-level bridge diagnostics, the C++ shared library also writes `vorlap_qblade_bridge_cpp.log` next to the parameter file that QBlade passes into `update_init()`. This log is written before Python runtime creation, so it is the first place to check if no Python-side log file appears.

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
