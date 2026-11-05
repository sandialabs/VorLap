# QBlade Coupling Timing

This page documents the timing semantics that are implemented and tested in the
current QBlade bridge. Both Python SIL and native callbacks are exercised against
the qualified upstream solver by the [QBlade integration checks](qblade_ci.md).

## Python SIL Step Ordering

`run_qblade_case()` drives QBlade through the Python SIL interface. For each
timestep in a VorLap-coupled run:

1. `SilVorLapCoupler.step(qblade, i)` reads QBlade input channels into the swap
   array.
2. The coupler records compact timing diagnostics from the current QBlade state.
3. `VorLapQBladeRuntime.update()` computes VorLap forces from the current
   structural velocities, time, timestep, and azimuth values in the swap array.
4. The coupler writes QBlade external actions for that same step.
5. `qblade.advanceTurbineSimulation()` advances QBlade.
6. The runner updates completed-step counts, progress messages, optional restart
   checkpoints, and result exports.

The current SIL diagnostics label this as `pre_advance_explicit`. Short Docker
SIL diagnostics record this pre-advance timing with the
wrapper/applied-force time one timestep ahead of QBlade's pre-advance reported
time after step 0. Treat that as current diagnostic evidence, not as a universal
QBlade contract.

## Native Callback Timing

The native bridge ABI harness proves that a standalone host can call the exported
bridge functions and receive forces consistent with the Python runtime. That does
not prove when, or whether, QBlade invokes those callbacks during a real
simulation.

Before claiming native callback timing, an executed QBlade `qblade-external` run
must show:

- nonzero `update_init` count,
- repeated `update` calls consistent with completed timesteps,
- a callback log tied to the selected QBlade model and built bridge hash,
- force/sign behavior consistent with the Python SIL or loading-file reference
  for a controlled diagnostic case.

The generic cantilever integration tests now obtain that evidence for QBlade CE
2.0.9.7 on Linux x86_64. They compare static reaction and deflection, and harmonic
amplitude and phase, with beam theory and an independent QBlade loading file.
Deliberately corrupted exchanged loads must fail those physical assertions even
when every callback completes successfully. This qualifies the tested fixture
and runtime; it does not establish arbitrary campaign or GUI timing.

In step-loop mode the runner explicitly calls `advanceController_at_num()` before
`advanceTurbineSimulation()` for each turbine with external libraries. In
`run-full-simulation` mode QBlade owns controller scheduling. The qualified
2-second fixture exports N samples with step-loop mode and N+1 samples with the
full scheduler. The tests assert both exact time grids and compare the same
physical time interval rather than equating wrapper step counts with samples.

Native case preparation writes an absolute source parameter directory because
QBlade copies its JSON into TEMP. Imported section bases already include the
initial rotor azimuth; runtime rotation applies the change from that reference
angle, avoiding a second application of the initial azimuth.

## Timestep, Sample Step, and Output Time

Bridge configuration may set:

- `timestep_s`: requested QBlade simulation timestep override for a generated
  working copy.
- `steps`: explicit number of QBlade timesteps to run.
- `duration_s`: used to derive steps when `steps` is not set.
- `sample_step`: optional VorLap runtime source-coverage sampling interval.
- `progress_every`: message frequency in completed QBlade steps.

When `steps` is present, it is the authoritative requested run length. When only
duration and timestep are present, effective steps are derived from them by the
typed bridge configuration.

VorLap harmonic reconstruction uses physical time in seconds. For time-varying
loads, instantaneous frequencies are integrated by phase continuity. For live
SIL force updates, `VorLapQBladeRuntime.update()` uses the time and timestep
values supplied through the QBlade swap array for each step. Source-coverage
sampling records data either every update or at the configured `sample_step`
interval, depending on runtime configuration.

## Restart and Resume

The SIL runner can write periodic restart checkpoints. Checkpoint metadata stores
the selected simulation, QBlade root, output directory, result name, coupling
mode, requested steps, completed step, threads, OpenCL setting, and optional
VorLap SIL state.

Before a GUI run resumes from checkpoint metadata, validation checks that the
metadata still matches the selected model and runtime settings. A mismatch is a
pre-worker validation error, because resuming with different geometry, coupling
mode, or device settings would make timing and force histories ambiguous.

## Loader Environment

Bridge build diagnostics report both the legacy `qblade_ld_library_path` value
and a platform-specific loader pair: `qblade_loader_path_key` plus
`qblade_loader_path`. The intended variable is `LD_LIBRARY_PATH` on Linux,
`DYLD_LIBRARY_PATH` on macOS, and `PATH` on Windows.

QBlade GUI launch plans prepend the detected QBlade root, `Libraries`,
`ControllerFiles`, `Binaries`, and `Binaries/IPCBridge` directories to
`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, and `PATH`. This keeps the Linux SIL
case covered while also making the macOS and Windows dynamic-library search
paths visible and testable. The build result still performs only a local
`ctypes.CDLL` load probe; a successful probe is not a substitute for executed
QBlade callback evidence.

## GUI and QBlade GUI Visibility

The VorLap QBlade-Bridge page can launch a QBlade GUI executable as a separate
inspection action. After a GUI-run SIL bridge case, the launcher prefers the
completed QBlade project save and then the latest restart checkpoint project, so
users can inspect QBlade-side diagnostics without manually finding the `.qpr`.
The proven automated run path is still the VorLap-controlled Python SIL runner.
Progress visible in an already-running QBlade GUI for the same automated run
should not be claimed unless a future test proves that QBlade can open the
prepared case, run the native callback path inside the GUI process, and log
callback evidence there.
