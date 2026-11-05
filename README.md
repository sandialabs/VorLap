# VorLap

<p align="center">
  <img src="docs/VorLapLogo.png" alt="VorLap Logo" width="420" />
</p>

[![Tests](https://github.com/sandialabs/VorLap/actions/workflows/tests.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/tests.yml)
[![Docs](https://github.com/sandialabs/VorLap/actions/workflows/docs.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/docs.yml)
[![Hosted Docs](https://img.shields.io/badge/docs-github_pages-blue)](https://sandialabs.github.io/VorLap/)
[![GUI Builds](https://github.com/sandialabs/VorLap/actions/workflows/build_gui.yml/badge.svg)](https://github.com/sandialabs/VorLap/actions/workflows/build_gui.yml)
[![Dev Builds (master)](https://img.shields.io/badge/dev_builds-master-blue)](https://github.com/sandialabs/VorLap/actions/workflows/build_gui.yml?query=branch%3Amaster)
[![Download](https://img.shields.io/github/v/release/sandialabs/VorLap?label=download)](https://github.com/sandialabs/VorLap/releases/latest)

VorLap (Vortex overLAP) predicts aerodynamic force spectra, force reconstruction, and frequency overlap risk for rotating structures using FFT-based airfoil databases.

Hosted documentation: <https://sandialabs.github.io/VorLap/>

Version **1.1.0** includes the code and examples accompanying the SIL paper. Start with the
[paper reproduction guide](docs/reproducibility.md),
[release notes](docs/release_notes_1.1.0.md), and [changelog](CHANGELOG.md).
The manuscript is maintained separately; its exact inputs are recorded in
`reproducibility/sil_paper_manifest.json`. QBlade is installed separately.

The [public turbine example](examples/ReferenceHVAWTQBladeModel/README.md) uses independently generated blade and torque-tube sections. The [NACA0018/Kynema example](examples/naca0018_kynema/README.md) includes the released airfoil source and CFD setup. See also the [2025 paper contracts](docs/paper_recreation.md).

Developer references:

- [Architecture](docs/developer_architecture.md)
- [Coordinate frames and sign conventions](docs/coordinate_frames.md)
- [QBlade coupling timing](docs/qblade_timing.md)

## Features

- Load component geometry from CSV.
- Load unsteady airfoil FFT databases from HDF5.
- Compute global force maps and structural moment maps from reconstructed nodal forces over inflow and azimuth sweeps.
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

If this fails or aborts on macOS (for example with `_tkinter` errors, `Connection Invalid` AppKit messages, `NSInternalInconsistencyException`, or other Tk runtime aborts), avoid `/usr/bin/python3` for the GUI path and use one of these:

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

For bridge and numerical-verification work, run the suite with numerical
RuntimeWarnings promoted to errors:

```bash
pytest -W error::RuntimeWarning
```

The native bridge ABI is also covered by `tests/test_qblade_bridge_abi.py`. This
test builds `libvorlap_qblade_bridge` in a temporary directory, launches a small
standalone C++ host process, calls `update_init`, `update`, and `update_message`,
and compares the returned float swap-array forces against the pure Python
runtime. Passing this test proves the local shared-library ABI layer; it does not
replace a QBlade run proving that QBlade itself invokes the `.trb` callback.

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

### Quick Start: QBlade Bridge

The GUI supports two bridge run modes. For routine automation, use the
SIL-coupled runner: VorLap drives QBlade through QBlade's Python SIL library,
reads the QBlade swap channels, evaluates VorLap forces, and applies them back
to QBlade with `setExternalAction()`. For a QBlade-owned timestep loop, use
`Run QBlade-driven`: QBlade runs the full simulation and calls the compiled
VorLap external-library callback. QBlade-owned quantities such as inflow
direction, mean inflow, and azimuth come from the selected QBlade model and
runtime swap data; do not duplicate them in the VorLap bridge settings.

To run your own QBlade model through the bridge:

1. Keep the QBlade model directory intact. The selected `.sim` file must still
   be able to resolve its referenced `.trb`, `.bld`, `.str`, `Aero`,
   `Structure`, and `Control` files using the same relative paths QBlade uses.
2. Provide a VorLap airfoil FFT directory. The preferred layout is a
   model-local `VorLapAirfoils` directory next to, or one directory below, the
   `.sim` file. Otherwise point the bridge at `data/airfoils` or another
   directory of compatible `.h5` airfoil databases.
3. Point VorLap at a complete QBlade CE root containing `SIL_Interface`,
   `Libraries`, the QBlade executable, and a writable `ControllerFiles`
   directory. The repo-local default is `run_headless/QBladeCE_2.0.9.7` when it
   is present.
4. Launch VorLap with `python scripts/launch_gui.py`, open the far-right
   `QBlade-Bridge` tab, and select your `.sim`, QBlade root, QBlade executable,
   airfoil directory, and output directory. The tab starts with the packaged
   reference model and repo-local QBlade paths when those files exist.
5. Use `Validate`, then `Prepare case`, then `Build bridge`, then choose one
   runner. `Run SIL bridge` is the default production/smoke automation path and
   records SIL history when enabled. `Run QBlade-driven` builds and uses the
   native callback path with QBlade's full-simulation driver; it disables
   checkpoint/resume because QBlade owns the timestep loop. The run progress and
   VorLap-side diagnostics appear in the VorLap bridge log and diagnostics
   table.
6. Use `Open QBlade GUI` when you want to inspect the selected model in QBlade.
   After a GUI-run bridge case, this opens the latest completed QBlade project
   save when present, otherwise the latest restart checkpoint project. This is
   intended for QBlade-side diagnostics and plotting without manually searching
   the output directory. The SIL run is still controlled by VorLap's Python
   runner, and the QBlade-driven button uses the QBlade library from the VorLap
   process; live progress in a separate QBlade GUI for the same automated run
   should not be assumed.

For a command-line smoke run of your own model without the GUI, use the same
inputs explicitly. This path requires a QBlade root that can run on the host
operating system; a Linux QBlade `.so` bundle will not load directly on macOS.

```bash
python examples/qblade_sil_smoke_run.py \
  --qblade-root /path/to/QBladeCE_2.0.9.7 \
  --sim /path/to/qblade_model/case.sim \
  --airfoils /path/to/qblade_model/VorLapAirfoils \
  --output-dir run_headless/runs/my_model_sil_smoke \
  --steps 20
```

This prepares a working copy under the output directory, runs a short
`sil-vorlap` case, and writes the run summary, QBlade result text file, SIL
history, source-coverage CSV, and quick plots when available. The SIL runner
applies the previously computed VorLap force block by default so its force
timing follows QBlade's native external-library table; run summaries record this
as `applied_force_lag_steps: 1`.

When using the repo-local Linux QBlade bundle from macOS, run the headless smoke
path through Docker instead:

```bash
RUN_ROOT=run_headless/runs/my_bridge_smoke \
QBLADE_STEPS=2 \
QBLADE_DURATION=60 \
WIND_SPEEDS=5 \
VORLAP_SOURCE_FREQUENCY_TRUNCATION=nyquist \
OMP_NUM_THREADS=2 \
bash run_headless/docker_smoke.sh
```

For lower-level headless control, first prepare a copy of the QBlade model, then
run the generated case through the SIL runner:

```bash
python scripts/prepare_qblade_external_case.py \
  --sim /path/to/qblade_model_copy/case.sim \
  --airfoils /path/to/qblade_model_copy/VorLapAirfoils \
  --node-source structural \
  --force-scale 1

python run_headless/run_qblade_sim.py \
  --sim /path/to/qblade_model_copy/case.sim \
  --qblade-root /path/to/QBladeCE_2.0.9.7 \
  --output-dir run_headless/runs/my_model_headless \
  --steps 20 \
  --sil-vorlap \
  --record-sil-vorlap-history
```

Use a copied model directory for `prepare_qblade_external_case.py` unless you
intend to edit the source QBlade files. For repeatable batch or production
headless work, prefer the wrappers in `run_headless/README.md`. The Linux
wrappers are named by coupling path and backend:

```bash
# SIL/Python runner, QBlade advanced through the SIL API.
./run_headless/run_qblade_vorlap_sil_10s_sample_linux_cpu.sh
./run_headless/run_qblade_vorlap_sil_10s_sample_linux_gpu.sh

# QBlade-owned runner, native external callback through runFullSimulation().
./run_headless/run_qblade_vorlap_qblade_driven_10s_sample_linux_cpu.sh
./run_headless/run_qblade_vorlap_qblade_driven_10s_sample_linux_gpu.sh
```

Keep the SIL wrappers for routine automation unless you are explicitly testing
the native QBlade external-library callback path and checking the generated
native callback evidence.

#### Running From The QBlade GUI Side

To watch the run from QBlade itself and have VorLap loads incorporated into that
same QBlade GUI run, QBlade must own the simulation loop and call the native
external-library callback declared in the prepared `.trb` and `.str` files. This
is the `qblade-external` path. It is different from the GUI-supported SIL bridge
above, where VorLap owns the run and QBlade is driven through the SIL API.

The native path is verified in this checkout for scripted QBlade
`runFullSimulation()` smoke runs with QBlade CE 2.0.9.6 and 2.0.9.7: the run
summary and bridge log show `update_init()` and sampled `update()` calls. Use
the steps below when you specifically want QBlade's GUI process to own the run
so live progress is visible inside QBlade. That live GUI-side workflow still
needs the same post-run callback evidence check before treating the result as a
QBlade-owned VorLap run.

1. Use a host-compatible QBlade GUI build. A Linux QBlade shared library bundle
   will not load directly into a macOS QBlade process; use Linux, Windows with
   the Windows bridge build, or a matching local QBlade installation.
2. Install VorLap into the same Python environment that the native bridge will
   embed or make this checkout available on `PYTHONPATH`:

   ```bash
   cd /path/to/VorLap
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev,docs,gui]"
   ```

3. Export the QBlade and Python paths before building and launching QBlade:

   ```bash
   export VORLAP_ROOT=/path/to/VorLap
   export PYTHON_EXE="$VORLAP_ROOT/.venv/bin/python"
   export QBLADE_ROOT=/path/to/QBladeCE_2.0.9.7
   export PYTHONPATH="$VORLAP_ROOT:${PYTHONPATH:-}"
   export LD_LIBRARY_PATH="$QBLADE_ROOT:$QBLADE_ROOT/Libraries:$QBLADE_ROOT/ControllerFiles:${LD_LIBRARY_PATH:-}"
   ```

   On macOS use the host-compatible QBlade build and `DYLD_LIBRARY_PATH` instead
   of `LD_LIBRARY_PATH`. On Windows, build with
   `scripts/build_qblade_external_windows.ps1` and put QBlade, Python, and the
   bridge DLL locations on `PATH`.

4. Build the native bridge against that Python and copy it into QBlade's
   `ControllerFiles` directory:

   ```bash
   PYTHON_EXE="$PYTHON_EXE" scripts/build_qblade_external_linux.sh \
     build/qblade_external_linux \
     "$QBLADE_ROOT/ControllerFiles"
   ```

5. Prepare a copy of the QBlade model. Do not patch your only source model
   unless that is intentional:

   ```bash
   "$PYTHON_EXE" scripts/prepare_qblade_external_case.py \
     --sim /path/to/qblade_model_copy/case.sim \
     --airfoils /path/to/qblade_model_copy/VorLapAirfoils \
     --node-source structural \
     --force-scale 1 \
     --debug
   ```

   The prep step updates the selected turbine with `LIBFILE_1`,
   `LIBFUNCTION_1=update`, `LIBARRAYSIZE_1`, and `LIBPARAMETERFILE_1`, appends
   the `EXTERNAL_1_IN` and `EXTERNAL_1_OUT` swap/action tables, and writes the
   VorLap runtime JSON under the model's `Control` directory.

6. Launch QBlade from the same shell so it sees the same library and Python
   paths:

   ```bash
   "$QBLADE_ROOT/run_qblade.sh"
   # or, depending on the downloaded package:
   "$QBLADE_ROOT/QBladeCE_2.0.9.7"
   ```

7. In QBlade, open the prepared model copy and run the prepared simulation from
   the QBlade GUI. QBlade should now own timestep advancement and show its usual
   GUI progress. VorLap loads are incorporated only if QBlade invokes the
   external-library callback during that run.

After the run, confirm the callback actually executed before treating the result
as a QBlade-owned VorLap run:

```bash
find /path/to/qblade_model_copy "$QBLADE_ROOT" \
  \( -name 'vorlap_qblade_bridge_cpp.log' -o -name 'vorlap_qblade_debug.log' \) \
  -print

rg -n "python_env_before_import|python_env_after_import|update_init|update" \
  /path/to/qblade_model_copy "$QBLADE_ROOT" -g 'vorlap_qblade*.log'
```

Minimum success evidence is a `vorlap_qblade_bridge_cpp.log` tied to the
prepared model, `python_env_before_import` and `python_env_after_import`
diagnostics, a nonzero `update_init` count, repeated sampled `update` counts
consistent with the completed timestep count, and a Python-side
`vorlap_qblade_debug.log` showing the VorLap runtime initialized and returned
finite force magnitudes. If the library and symbols are found but there is no
`update_init` or `update` evidence, QBlade did not enter the native callback
phase; use the SIL bridge path for supported runs and keep the native GUI path
as an unresolved QBlade-callback diagnostic.

Run a scripted time-varying inflow example:

```bash
python examples/time_varying_inflow_case.py
```

Run the QBlade fast-path loading export:

```bash
python examples/qblade_fastpath_loading.py --sim /path/to/case.sim --inflow data/inflow_profile.csv --output qblade_external_loading.txt
```

Plan QBlade bridge case preparation without launching QBlade:

```bash
python examples/qblade_bridge_prepare_dry_run.py
```

Run a short Python SIL bridge smoke case on a machine with a host-compatible
QBlade installation:

```bash
python examples/qblade_sil_smoke_run.py --qblade-root /path/to/QBladeCE
```

Build the live QBlade external-library bridge:

```bash
scripts/build_qblade_external_linux.sh
```

The build script detects and clears a stale CMake cache when the repository has
moved since a previous bridge build. If the bridge is being built for a QBlade
runtime on another machine, set `PYTHON_EXE` to the exact Python environment
that QBlade will use for embedded imports.

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

External Linux / Conda environment recipe:

Use the same Python for all three pieces:

- installing VorLap
- building `libvorlap_qblade_bridge.so`
- launching QBlade

Example setup:

```bash
cd /path/to/vorlap

export PYTHON_EXE=/path/to/conda-or-venv/bin/python3
export QBLADE_ROOT=/path/to/QBladeCE_2.0.9.6
export CONDA_LIB=/path/to/conda-or-venv/lib
export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"

# Verify the Python and NumPy you intend to use
$PYTHON_EXE -c "import sys, numpy; print(sys.executable); print(sys.version); print(numpy.__version__)"

# Install VorLap into that Python so the embedded interpreter can import it
$PYTHON_EXE -m pip install -e .
```

Build and install the bridge using that exact Python:

```bash
PYTHON_EXE=$PYTHON_EXE scripts/build_qblade_external_linux.sh \
  build/qblade_external_linux \
  "$QBLADE_ROOT/ControllerFiles"
```

Prepare a QBlade case using that same Python. If the model uses a non-default parameter-file name, pass it explicitly with `--parameter-file`; otherwise the script writes `Control/vorlap_qblade_external.json` next to the turbine file.

```bash
$PYTHON_EXE scripts/prepare_qblade_external_case.py \
  --sim /path/to/qblade_case/baseline_10m_HVAWT_reference_turbine_Wwnd.sim \
  --airfoils /path/to/qblade_case/10m_HVAWT_reference_turbine/VorLapAirfoils \
  --parameter-file Control/vorlap_10m_HVAWT_reference_external.json \
  --node-source structural \
  --force-scale 1 \
  --debug
```

Do not rely on `PYTHON_EXE=/path/to/python command` to persist after that command exits. Use `export PYTHON_EXE=...` first, or call the interpreter explicitly:

```bash
/path/to/python3 scripts/prepare_qblade_external_case.py --sim /path/to/case.sim --airfoils /path/to/VorLapAirfoils
```

If `scripts/prepare_qblade_external_case.py` is executed directly without a Python interpreter, shells may try to parse the Python file as shell script and produce errors such as `from: command not found` or `syntax error near unexpected token '('`. Run it with `$PYTHON_EXE` or `python3`.

When launching QBlade on systems where SciPy or NumPy depend on a newer Conda `libstdc++` than the system default, launch QBlade with the Conda library path active:

```bash
export PYTHON_EXE=/path/to/conda-or-venv/bin/python3
export QBLADE_ROOT=/path/to/QBladeCE_2.0.9.6
export CONDA_LIB=/path/to/conda-or-venv/lib
export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"

"$QBLADE_ROOT/run_qblade.sh"
```

If `LD_LIBRARY_PATH` is not sufficient and QBlade still loads the system `libstdc++.so.6`, force the Conda one:

```bash
export LD_PRELOAD="$CONDA_LIB/libstdc++.so.6"
"$QBLADE_ROOT/run_qblade.sh"
```

Useful verification commands on the external system:

```bash
# Confirm the import path used by the embedded Python target
$PYTHON_EXE -c "import vorlap, vorlap.qblade_runtime as q; print(vorlap.__file__); print(q.__file__)"

# Confirm the rebuilt bridge is in the QBlade ControllerFiles directory
ls -l "$QBLADE_ROOT/ControllerFiles/libvorlap_qblade_bridge.so"

# Confirm the generated JSON includes the expected debug/runtime fields
rg -n "debug|log_file|source_parameter_dir|force_scale" \
  /path/to/qblade_case/10m_HVAWT_reference_turbine/Control/vorlap_10m_HVAWT_reference_external.json

# Check for bridge and Python-side logs after a QBlade run
find /path/to/qblade_case "$QBLADE_ROOT" \
  -name 'vorlap_qblade_bridge_cpp.log' -o -name 'vorlap_qblade_debug.log'
```

Interpretation of the two runtime logs:

- `vorlap_qblade_bridge_cpp.log`: bridge-level logging from the shared library, written before Python runtime creation
- `vorlap_qblade_debug.log`: Python-side runtime logging from `vorlap.qblade_runtime`, written only after the VorLap runtime initializes successfully

Windows bridge build + install example (PowerShell):

```powershell
scripts/build_qblade_external_windows.ps1

scripts/build_qblade_external_windows.ps1 `
  -BuildDir build/qblade_external_windows `
  -InstallDir C:\path\to\QBlade\ControllerFiles
```

Prepare the public 10 m H-VAWT QBlade case for VorLap. The exact paper case
is available from the authors on request, subject to approval.

```bash
python scripts/prepare_qblade_external_case.py \
  --sim examples/ReferenceHVAWTQBladeModel/baseline_10m_HVAWT_reference_turbine_Wwnd.sim \
  --airfoils examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/VorLapAirfoils \
  --node-source structural \
  --n-freq-depth 20 \
  --force-scale 1 \
  --debug \
  --absolute-source-parameter-dir
```

Use `--force-scale` values above 1 only for explicit amplitude-sensitivity or
diagnostic stress tests, and pass `--diagnostic-force-scale` so that the run is
labeled at setup time. Do not treat exaggerated force scaling as a validated
physical case without a separate calibration basis.

That script updates the turbine definition with `LIBFILE_1`, `LIBFUNCTION_1`, `LIBARRAYSIZE_1`, and `LIBPARAMETERFILE_1`, appends `EXTERNAL_1_IN` / `EXTERNAL_1_OUT` tables to the structural model, and writes `Control/vorlap_qblade_external.json`.

The generated `airfoil_dir` in `vorlap_qblade_external.json` is written relative to the parameter file location when possible, so model-local layouts such as `10m_HVAWT_reference_turbine/VorLapAirfoils` work across machines without hard-coded absolute paths.

The runtime also supports `source_parameter_dir` as a fallback base when a config is copied away from its original case tree. Checked model-local configs store this as `.` for portability; use `--absolute-source-parameter-dir` only for a generated, untracked run artifact that must survive a JSON-only copy into a temporary QBlade directory.

The runtime config defaults to structural `BLD_*`/`STR_*` nodes so the swap mapping aligns with output locations already declared in the structural file. Use `--node-source converted` if you want all converted VorLap nodes instead.

The embedded bridge imports `vorlap.qblade_runtime` from the Python interpreter
embedded in the shared library. Generated configs include
`python_path_entries`; by default `scripts/prepare_qblade_external_case.py`
stores this checkout root there so the C++ bridge prepends it to `sys.path`
before importing VorLap. If VorLap is installed into the embedded Python
environment instead, use `--no-default-python-path` during case prep and keep the
installed environment active when QBlade loads the bridge.

QBlade-side mapping flow (general):

- `.sim`: selects turbine (`TURBFILE`) and operating conditions (`RPMPRESCRIBED`, `MEANINF`, etc.).
- `.trb`: enables external library calls via `LIBFILE_1`, `LIBFUNCTION_1`, `LIBARRAYSIZE_1`, and `LIBPARAMETERFILE_1`.
- `.str`: defines `EXTERNAL_1_IN` swap inputs (time, azimuth, structural node velocities) and `EXTERNAL_1_OUT` actions (`ADDFORCE`) that map returned forces to component IDs and normalized positions.
- `Control/vorlap_qblade_external.json`: high-level VorLap runtime config.
- `n_freq_depth`: number of spectral tones used per node (capped by available airfoil FFT depth).
- `source_frequency_cap_hz`: optional fixed upper frequency for oscillatory CL/CD tones. VorLap discards tones above the cap; it does not clamp or fold them to the cutoff. Campaigns can derive this value from the simulation timestep with `--source-frequency-truncation nyquist` or `VORLAP_SOURCE_FREQUENCY_TRUNCATION=nyquist`. The default is unfiltered.
- `force_scale`: global multiplier applied to all VorLap external forces before they are returned to QBlade. Nominal physical runs should use `1.0`; bridge-active identity runs should use `0.0`; values above `1.0` require `--diagnostic-force-scale` in the case-preparation and Headless wrapper CLIs.
- `python_path_entries`: paths prepended to the embedded interpreter's
  `sys.path` by the C++ bridge before importing VorLap. The bridge log records
  the resolved entries and the final Python import diagnostics.
- `debug`: enables verbose `update_message()` diagnostics, including resolved paths at init and the maximum applied force magnitude/node during updates.
- `log_file`: optional runtime log path. When `--debug` is used and no path is provided, case prep writes `vorlap_qblade_debug.log` next to the generated JSON and stores the path relative to that JSON when possible.

For lower-level bridge diagnostics, the C++ shared library also writes
`vorlap_qblade_bridge_cpp.log` next to the parameter file that QBlade passes into
`update_init()`. This log is written before Python runtime creation, so it is the
first place to check if no Python-side log file appears. It records
`python_env_before_import` and `python_env_after_import` JSON lines containing
the Python executable/version, `sys.path`, NumPy version, VorLap package path,
and `vorlap.qblade_runtime` path observed during `update_init()`.

For the live QBlade bridge, VorLap treats the `X_g Vel. ...` / `Y_g Vel. ...` / `Z_g Vel. ...` swap inputs as structural node velocities. The runtime converts these into aerodynamic relative velocity by subtracting structural motion from the ambient wind vector defined in the QBlade `.sim` file (`MEANINF`, `HORANGLE`, `VERTANGLE`).

Current verification status:

- The batch VorLap solver and QBlade runtime controller use the same rotated chord/normal force-basis assembly.
- Component global positions and chord/normal frames are prepared through the non-plotting `prepare_component_frames()` utility. Batch and time-varying solvers call it directly, so CSV/QBlade placeholder geometry no longer requires a plotting call before solving.
- Strouhal scaling uses the projected cross-flow length `chord * max(|sin(AOA)|, thickness_ratio)`. This preserves the `chord * |sin(AOA)|` nondimensionalization while using physical section thickness as the low-AOA floor.
- Airfoil FFT files use a validated DC-first schema: spectral index 0 is the signed mean coefficient with `St=0`, and oscillatory peaks start at index 1. The loader warns and repairs legacy index-0 Strouhal metadata when the mean coefficient is already at index 0.
- FFT phase interpolation uses unit phasors, so values near the `+pi/-pi` branch cut interpolate through 180 degrees instead of collapsing toward zero.
- Re/AOA queries outside an airfoil FFT source grid are diagnosed with `AirfoilRangeWarning` by default before flat clipping; strict audit calls can use `bounds_policy="raise"`.
- Optional source-frequency truncation records both retained `freq_max_hz` and `unfiltered_freq_max_hz`, plus the number of discarded nonzero coefficients. Alias checks use the retained forcing band. An exact Nyquist cap prevents above-Nyquist forcing but provides no anti-alias margin, so production results still require the source-coverage and PSD gates.
- CM spectra are loaded and schema-validated, but the current QBlade force path does not reconstruct them into nodal torque columns. Loading files written without an explicit torque array intentionally write zero moment columns.
- Checked QBlade example configs now use `force_scale=1.0` and model-local relative paths. The prep script and Headless wrappers reject unlabeled scales above 1.0 unless `--diagnostic-force-scale` is present.
- The Headless SIL runner verifies that any active QBlade storage filter retains every bridge input channel, rejects QBlade's all-channel missing-data sentinel response before advancing timesteps, and records structural velocity min/max/all-zero diagnostics in run summaries. SIL cases use an augmented internal storage filter while `exportResults()` keeps the compact requested result filter.
- Headless per-case run summaries now include a `reproducibility` block with command line, git state, package versions, QBlade/bridge hashes, parsed VorLap config settings, input-file hashes, loader environment, and batch case metadata.
- Generated native-bridge configs now carry explicit embedded-Python
  `python_path_entries`, and the C++ bridge logs Python/NumPy/VorLap import
  diagnostics during `update_init()`.
- Native callback run summaries now record whether the configured
  `LIBFILE_1` library was found and whether it exports the symbols QBlade
  expects from `LIBFUNCTION_1` (`update`, `update_init`, and
  `update_message` for the default bridge function).
- Headless postprocessing marks inadequate histories as `smoke_only` using sample, duration, run-status, and rainflow-cycle thresholds. Smoke-only rows are excluded from the default DEL comparison/table/plot and retained in `del_comparison_diagnostics.csv` for debugging.
- Headless production campaign setup now treats initial azimuth as an explicit case dimension, defaults production-wrapper cases to 660 s with storage beginning at 60 s and fixed-speed idling using `PRESCRIBETYPE=1` plus `RPMPRESCRIBED=1`, supports dry-run planning for a retained 10 min wind-speed matrix, requires the documented four-speed/single-initial-azimuth grid in the staged paper workflow unless explicitly disabled for smoke/debug runs, and generates explicit scenario sets for each case. Multi-azimuth grids are treated as initial-phase sensitivity studies.
- Headless postprocessing now loads the three production scenarios and compares the bridge-active zero-force identity case against nominal VorLap for DEL products, with rows paired by case, wind speed, azimuth, and station.
- Headless production campaign audits now provide plan/setup/results gates so missing manifests, failed or incomplete runs, missing exported results, mismatched scenario force scale or coupling path, mismatched QBlade case/environment settings, stale generated runtime assets, incompatible restart checkpoints, unproven TurbSim reuse, exported RPM drift, unexpected native callback activity, missing reproducibility metadata, inconsistent SIL update counts, incomplete per-config or per-timestep source coverage, incomplete ultimate-margin channel coverage, stale or non-manifest sign-probe evidence, malformed fatigue/source/ultimate postprocess artifacts, smoke-only or warning-bearing paper-facing comparison tables, nonfinite paper-facing DEL/cycle values, stale critical-case selections, critical rainflow count mismatches, and all-zero velocity diagnostics are visible before paper figures are populated.
- The runtime normalizes the QBlade inflow direction before multiplying by mean speed.
- `freq_min` and `freq_max` are honored in the overlap calculation.
- `VIV_Params` allocates fresh default arrays/lists per instance and copies caller-owned configuration arrays so runs cannot share mutable state accidentally.
- Public `QBladeController.step()` calls return stable force copies by default; the runtime explicitly uses the internal read-only view and reports exact cache hit/miss counts.
- The targeted bridge/runtime Python suite passes locally with `RuntimeWarning` promoted to an error: 186 tests, including the C++ ABI harness.
- Native QBlade external-library callback execution is verified for scripted
  QBlade CE 2.0.9.6 and 2.0.9.7 `runFullSimulation()` smoke runs when the run
  summary reports a found native library, exported callback symbols,
  `update_init`, sampled `update` calls through the completed timestep count,
  and matching `vorlap_qblade_bridge_cpp.log` import diagnostics. The Python
  SIL path remains the preferred automation path; use the QBlade-driven path
  when the native callback itself is the evidence being generated.

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
- `run_headless/`: QBlade SIL execution, campaign launchers, and retained configuration/evidence
- `qblade_external_bridge/`: buildable CMake/C++ native callback bridge
- `scripts/`: data processing, case preparation, builds, and release utilities
- `reproducibility/`: exact separate-manuscript snapshot identities

Contributor setup and checks are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Verification

The repository includes a verification-style test in `tests/test_verification_case.py` based on the existing single-blade case and expected CL/CD force levels. The force check projects the global force back onto the rotated section chord/normal basis, which is the invariant quantity after azimuth rotation.

Bridge verification is staged:

1. Python unit tests and synthetic force-frame parity.
2. Python runtime replay with a known swap buffer.
3. C++ ABI harness for `update_init`, `update`, and `update_message`.
4. Native QBlade callback proof with bridge logs.
5. SIL/native/offline force-history parity.
6. `force_scale=0` baseline identity on the final structural model.
7. Production parked-load sweep with signed fatigue channels.

## License

See `LICENSE` and `NOTICE`.
Third-party QBlade source and research input provenance are described in
[third-party inputs](docs/third_party_inputs.md); their own terms still apply.
