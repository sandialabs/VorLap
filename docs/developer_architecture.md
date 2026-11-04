# Developer Architecture

This document maps the main VorLap workflows to the modules that implement them.
It is meant as an orientation guide for future changes, not as a replacement for
the API reference.

## Offline VorLap Sweep

Offline analysis starts from component geometry, airfoil FFT databases, natural
frequencies, and `VIV_Params`. The GUI setup tab and examples call into
`vorlap.computations.compute_thrust_torque_spectrum()` or the optimized variant.

Main modules:

- `vorlap.structs`: mutable record types for components, airfoil spectra,
  inflow profiles, and analysis parameters.
- `vorlap.fileio`: CSV, HDF5, natural-frequency, inflow, and QBlade file I/O.
- `vorlap.computations`: frame preparation, local kinematics, spectrum lookup,
  force assembly, moment assembly, frequency-overlap metrics, and time-history
  reconstruction.
- `vorlap.graphics`: plotting helpers used by the GUI and scripts.
- `vorlap.gui.tabs.setup`: current GUI collector and offline run entry point.

Data flow:

1. Geometry CSV rows become `Component` instances.
2. Airfoil HDF5 files become `AirfoilFFT` instances keyed by airfoil id.
3. `prepare_component_frames()` fills or validates chord, normal, and global
   shape arrays.
4. Each inflow speed and azimuth is projected onto local chord/normal bases.
5. FFT coefficients are interpolated in Re/AOA space and scaled by dynamic
   pressure and nodal span weight.
6. Forces and moments are returned to the GUI or written by the calling script.

## Time-Varying Loading-File Export

The one-way QBlade loading-file path converts a QBlade model to VorLap
components, reconstructs forces over a prescribed inflow profile, and writes a
QBlade `LOADINGFILE`.

Main modules:

- `vorlap.fileio`: QBlade `.sim/.trb/.bld/.str` parsing, QBlade-to-VorLap
  conversion, loading-file writing, loading-file reading, and inflow-profile
  loading.
- `vorlap.computations`: nonstationary time-history reconstruction.
- `examples/qblade_fastpath_loading.py`: user-facing example.

Data flow:

1. `load_qblade_simulation_definition()` resolves the turbine file.
2. `convert_qblade_to_vorlap_inputs()` converts the QBlade geometry to
   components and requested node ids.
3. `compute_time_varying_force_history()` synthesizes nodal force histories.
4. `write_qblade_loading_file()` writes time rows with global force columns and
   zero moment columns unless a caller supplies moments.
5. `read_qblade_loading_file()` is available for round-trip checks.

## Python SIL VorLap Coupling

The proven live-coupling path drives QBlade through its Python SIL interface from
the VorLap process. VorLap reads QBlade structural velocity channels, computes
forces, and writes `ADDFORCE` actions before advancing the simulation.

Main modules:

- `vorlap.qblade_bridge_config`: typed GUI/service configuration and validation.
- `vorlap.qblade_bridge_service`: GUI-facing facade for validation, case
  materialization, preparation, run, build, sign probe, and artifact opening.
- `vorlap.qblade_case_prep`: importable case-preparation service.
- `vorlap.qblade_nodes`: shared QBlade node-id parsing and channel naming.
- `vorlap.qblade_runtime`: runtime config loading, external-library table
  generation, component selection, force update, and geometry audit writing.
- `vorlap.qblade_sil`: Python SIL couplers, timing diagnostics, sign-probe
  diagnostics, and QBlade result readers.
- `vorlap.qblade_run`: headless QBlade SIL runner, restart checkpoints, summary
  writing, and result export.
- `vorlap.gui.tabs.qblade_bridge`: QBlade-Bridge page and headless-tested GUI
  state/rendering helpers.

Data flow:

1. The GUI or CLI creates `QBladeBridgeConfig`.
2. The service validates paths, QBlade model references, airfoil availability,
   and optional QBlade install pieces.
3. The selected source model is copied to a generated working directory when
   one-off run overrides are needed.
4. `prepare_qblade_external_case()` writes the VorLap runtime config, geometry
   audit CSV, and QBlade external-action table edits for the working copy.
5. `run_qblade_case()` loads QBlade through Python SIL.
6. `SilVorLapCoupler.step()` fills the swap array, records timing, calls
   `VorLapQBladeRuntime.update()`, and applies the resulting QBlade actions.
7. The runner advances QBlade, writes restart checkpoints when requested, exports
   results, and writes a summary JSON.
8. The service derives GUI diagnostics and writes a source-coverage CSV when the
   summary includes source-coverage rows.

## Native QBlade External-Library Callback

The native callback path builds `libvorlap_qblade_bridge` for QBlade's external
controller interface. Local ABI tests prove the shared-library host contract,
and scripted QBlade `runFullSimulation()` smoke tests prove repeated native
callback invocation for the supported reference workflow.

Main modules and scripts:

- `cpp/qblade_external`: native bridge source.
- `scripts/build_qblade_external_linux.sh` and
  `scripts/build_qblade_external_windows.ps1`: platform build scripts.
- `tests/test_qblade_bridge_abi.py`: standalone host ABI proof.
- `scripts/prepare_qblade_external_case.py`: CLI wrapper around the importable
  preparation service.
- `vorlap.qblade_runtime`: Python runtime called by the embedded interpreter.

Important status:

- The ABI harness can call `update_init`, `update`, and `update_message` and
  compare native swap-array forces against Python.
- Run summaries check the configured QBlade external-library path and required
  callback symbols, so missing-library and missing-symbol failures are separated
  from QBlade not invoking a valid library.
- The `qblade_execution_mode=run-full-simulation` path calls QBlade's
  `runFullSimulation()` API for `qblade-external` cases. The 19 June 2026 Docker
  probes on QBlade CE 2.0.9.6 and 2.0.9.7 showed a found, symbol-complete bridge
  library, nonzero `update_init`, and sampled `update` counts through the
  completed timestep count.
- A QBlade `qblade-external` run is native-callback evidence only when the run
  summary and `vorlap_qblade_bridge_cpp.log` show nonzero `update_init` and
  repeated `update` counts. Do not infer success from library discovery alone.
- GUI wording should keep "Open QBlade GUI" separate from the VorLap-launched
  QBlade-driven run until a live GUI-process callback run is separately proven.

## Current Refactoring Pressure Points

- `vorlap.fileio` and `vorlap.computations` still mix many responsibilities.
- GUI application state is held in `VorLapGuiState`; legacy tab-facing app
  properties delegate to it. Numerical workers consume input snapshots and
  queue updates for the Tk thread. Replacing geometry invalidates old results.
- Several result payloads are dictionaries with implicit keys; bridge summaries
  and analysis outputs should move toward named result types while preserving
  stable JSON output.
- `Component`, `VIV_Params`, `InflowTimeSeries`, and `AirfoilFFT` are mutable
  dataclasses. Caller-owned arrays are copied at construction; analysis workers
  must also isolate mutable geometry before preparing its coordinate frames.
