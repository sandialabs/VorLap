# Grinderslev et al. 2022 single-blade bridge benchmark

This case recreates the public structural setup closest to the present
VorLap--QBlade architecture: the clamped IEA-10-MW blade and P100--I50 inflow
studied by Grinderslev et al., *Wind Energy Science* 7 (2022),
<https://doi.org/10.5194/wes-7-2201-2022>, including the published 2023
corrigendum <https://doi.org/10.5194/wes-7-2201-2022-corrigendum>.

The case is intentionally split into two evidence levels:

- Public geometry is converted from the pinned IEA-10-MW WindIO model. The
  official HAWC2 set-1 blade table is converted to 19 QBlade Timoshenko
  sections, giving the 18 beam elements reported in the paper. The public
  OpenFAST first-edgewise polynomial is retained as a traceable forcing-shape
  surrogate because the paper's fitted HAWCSTAB2 polynomial is figure-only.
- Coordinate reconciliation preserves HAWC2 `E*Ix` as QBlade `EIx` and
  `E*Iy` as `EIy`. QBlade local `Y` is chordwise and is therefore governed by
  bending about local `X`. Mass and shear-center offsets are retained. The QBlade beam
  reference remains on the WindIO blade axis, so the HAWC2 elastic-center
  offsets and principal-axis pitch are omitted; every reduction is recorded
  in `benchmark_manifest.json`.
- The VorLap input is a coherent 0.67 Hz first-edgewise-mode forcing probe.
  It is not an EllipSys3D load history and is not tuned to the published
  1.18 m limit-cycle amplitude. It verifies geometry conversion, node mapping,
  frequency reconstruction, and imposed-load transfer only. A static source
  cannot verify response-dependent wake lock-in or multiple limit cycles.

Generate the case from the public model at the recorded commit:

```bash
git clone https://github.com/IEAWindSystems/IEA-10.0-198-RWT /tmp/iea10-rwt
git -C /tmp/iea10-rwt checkout 350cf06f5db9cec12bcd73c2b174a5d9642958d8
.venv/bin/python examples/grinderslev2022_iea10/generate_case.py \
  --iea-root /tmp/iea10-rwt
```

The generator writes a hash manifest, QBlade text model, public-model geometry
table, 30 station-specific source files, external-library configuration, and
geometry audit under `generated_case/`. The public model's Apache-2.0 license
is copied beside the derived files and recorded in the manifest. The default run is 12 s at the paper's
0.006 s time step. Use `--duration 60` for the published baseline build-up
window, while retaining the evidence boundary above.

The published pitch and inclination are defined in the blade-root section
frame, not as QBlade global angles. The generator constructs that root frame,
maps P100--I50 into QBlade's horizontal/vertical settings, and records both the
target and dot-product recovery in the manifest. It also reads the public
OpenFAST `BldEdDmp1` value (0.3614 %) and maps it to a stiffness-proportional
coefficient at 0.67 Hz. This is a selected single-mode damping approximation,
not a reproduction of the paper's full HAWC2 Rayleigh damping model.

The prepared case can be run with the existing Linux/amd64 QBlade container:

```bash
docker run --rm --platform linux/amd64 \
  --volume "$PWD:/work/VorLap" --workdir /work/VorLap \
  --env QBLADE_ROOT=/work/VorLap/run_headless/QBladeCE_2.0.9.7 \
  --env PYTHON_EXE=/usr/bin/python3 --env QT_QPA_PLATFORM=offscreen \
  vorlap-qblade-headless:amd64 \
  /usr/bin/python3 run_headless/run_qblade_sim.py \
  --sim examples/grinderslev2022_iea10/generated_case/grinderslev2022_p100_i50_iea10_single_blade.sim \
  --qblade-root run_headless/QBladeCE_2.0.9.7 \
  --output-dir run_headless/runs/grinderslev2022_axis_corrected_v2_p100_i50_edge_12s \
  --threads 4 --sil-vorlap --coupling-mode sil-vorlap \
  --record-sil-vorlap-history --progress-every 250
```

Run the paired structural baseline with the same image and case, replacing the
last two lines with:

```bash
  --output-dir run_headless/runs/grinderslev2022_axis_corrected_v2_p100_i50_unloaded_12s \
  --threads 4 --coupling-mode none --progress-every 250
```

The paper comparison generator expects the coupled/unloaded result-directory
constants declared at the top of the script. After pointing those constants
to the two completed runs, regenerate the hash-bound figures and CSVs with:

```bash
MPLCONFIGDIR=/tmp/vorlap-mpl \
  python paper_drafts/NAWEA_QBladeBridge/figure_generation/generate_single_blade_benchmark.py
```

The exact high-fidelity force histories are available from the publication's
authors on request. If obtained, replace the synthetic sources and compare
aligned applied-force, modal-power, and response histories before describing
the exercise as high-fidelity code-to-code verification.

## Calibrated response-aware power benchmark

`response_aware_benchmark.py` prepares a separate response-aware experiment
without modifying `generated_case/`. The closure reads the digitized P100--I50
forced-motion aerodynamic-power curve, projects the recorded QBlade velocity
onto an independently identified QBlade first-edgewise coordinate, and
distributes the resulting generalized force in the chord direction. This is a
calibration-based reduced-order model: it tests the bridge's response feedback,
force transfer, and energy bookkeeping, but it does not predict the wake or
independently verify the IDDES solution.

The response coordinate is not an amplitude-fit to the aerodynamic result.
`identify_qblade_mode.py` uses the pre-existing 12 s
`axis_corrected_v2` coherent structural-forcing probe to fit the complex
0.67 Hz velocity shape at all 30 action nodes. Local chord directions come
from the stored coherent force vectors; the fitted shape is phase-aligned and
normalized to unit tip. The matching *unloaded* QBlade initialization decay
independently identifies 0.66985 Hz. Input histories, geometry, and structure
are SHA-256 pinned in
`reference_data/qblade_first_edge_mode_identification.json`. Reproduce the
tracked CSV/JSON from the unchanged local probe artifacts with

```bash
.venv/bin/python examples/grinderslev2022_iea10/identify_qblade_mode.py
```

Fits beginning at 0, 2, 4, and 6 s give public-shape projection overlaps of
0.9430--0.9473, modal masses of 1679.2--1694.2 kg, and maximum nodal phase
offsets below 0.026 rad. The full-record shape is frozen for the response
model. The public OpenFAST polynomial remains in the geometry table for source
provenance and visualization, but it is no longer used by the response-model
projection or force distribution.

The configuration is derived rather than amplitude-tuned. Each of the 30
modal mass weights is the `MASSD` value interpolated from
`iea10_blade_hawc2.str` times the physical nodal tributary span. Aerodynamic
weights are local chord times physical tributary span; this area quadrature is
declared in the generated manifest and is not fitted to an unpublished
spanwise CFD load distribution. The top-level external configuration retains
the exact 10-decimal QBlade action IDs. The response model uses their
one-to-one canonical six-decimal controller IDs from the shared QBlade node
parser; both ordered lists, hashes, and the explicit mapping are recorded in
the response configuration and variant manifest. The QBlade-identified
tip-normalized modal mass is 1679.207 kg (the public OpenFAST surrogate gives
1886.815 kg). With the converted structure's stiffness-proportional Rayleigh
coefficient evaluated at 0.66985 Hz, the modal damping ratio is 0.361319% and
the structural power is

```text
P_struct(A_tip) = zeta M_QBlade omega_QBlade^3 A_tip^2 = 452.343 A_tip^2 W.
```

This `C A_tip^2` relation is a nominal single-coordinate structural-model
diagnostic, not a direct measurement of QBlade's internal dissipated power.
The available SIL/QBlade histories expose external nodal force, translational
velocity, tip deflection, and root moment, but not the Chrono stiffness matrix,
rotational degrees of freedom, Rayleigh dissipation, or exact FE modal mass.
The frozen 5% comparison is therefore retained as a conservative model-form
qualification row, while `closure_claim=false` explicitly prevents a passing
or failing row from being described as direct energy closure.

The failed nominal-damping row in the 120 s result is not caused by the
aerodynamic-area lift-back. On
the same final-20-s coarse record, the actual lagged nodal work is 768.4665 W
and the counterfactual mass-coordinate work `Q[n-1] qdot_raw[n]` is
768.6558 W, a 0.0246% difference. The fine values are 769.7060 and
769.8950 W (0.0246%). The analyzer now computes and records this same-history
countercheck explicitly; changing the lift-back weights cannot explain the
6.39--6.59% difference from nominal `C A_tip^2`.

An independent damped-sine audit of the matched 120 s unloaded decays also
does not support 6% extra damping. Fitting
`c0+c1*t+exp(-lambda*t)*(a*cos(2*pi*f*t)+b*sin(2*pi*f*t))` gives
`zeta=lambda/(2*pi*f)=0.0036485` over 0--120 s and 0.0036189 over
10--120 s at `dt=0.006 s`; the `dt=0.003 s` fits are 0.0036489 and
0.0036190. These bracket the mapped `beta*omega/2=0.00361319` within about
1%. The audited result hashes are
`9cd608d6315a841c762cd2c316dbb9771622c23ba228a2c0d2137c12dad02955`
(`dt=0.006 s`) and
`92a1fafd018e79e7f556ae5ce79d30b7bf8bdcbe72d9f92b8dfcc3a4127fc794`
(`dt=0.003 s`); both fits use the complete finite edgewise-tip histories and
the stated fixed windows, with no aerodynamic-response data.

Its intersection with the digitized forced-motion curve is 1.27202 m. This is
the prediction of the converted QBlade structure and calibrated power curve,
not an additional fitted target. The paper's approximately 1.24 m
forced-motion/damping-curve intersection uses its HAWC2 structural coordinate
and damping model. The separately reported 1.18 m,
775 W FSI result is not the calibration target; neither is the approximately
1.63 m zero-damping amplitude. The digitized CSV is embedded in each JSON
configuration with SHA-256
`016ce5265b77dc0c9e577d7548e453d09f67ef732dda817618f79909546f323d`.

Prepare independent 120 s coarse/fine response cases and matching unloaded
baselines:

```bash
.venv/bin/python examples/grinderslev2022_iea10/response_aware_benchmark.py prepare \
  --output-root run_headless/runs/grinderslev2022_modal_power_inputs_qblade_mode_120s_v2
```

The prepared response cases set the legacy VorLap source `force_scale` to
zero, so every nonzero SIL force comes from the modal-power closure. The
unforced QBlade initialization transient (about 0.034 m in the prior 0.006 s
baseline) seeds the response without an imposed aerodynamic tone. Analysis
uses only the final 20 s, excluding the first 100 s build-up interval, and
subtracts the matching unloaded history when reporting the response-induced
root moment. The earlier 60 s experiment was rejected for physical settling:
its final-20-s response was still a build-up transient, so a coarse/fine pass
alone was not accepted as a limit-cycle result.

Within-run fatigue stationarity uses a centered, fixed 12-target-cycle record
inside that final 20 s. It is split into phase-aligned 6+6-cycle halves and
interpolated onto the same 500-interval-per-cycle phase grid. The gate compares
the unscaled rainflow range-power sum per target cycle at the original 5%
tolerance. Boundaries depend only on time and the pinned frequency, avoiding
the 7.5-versus-7.0 residual-cycle bias observed when arbitrary 10 s halves were
used. This remains numerical pseudo-damage, not Miner damage or fatigue life.

For example, run the 0.006 s response and unloaded pair with two threads:

```bash
INPUT_ROOT=run_headless/runs/grinderslev2022_modal_power_inputs_qblade_mode_120s_v2
.venv/bin/python run_headless/run_qblade_sim.py \
  --sim "$INPUT_ROOT/dt_0p006/response/grinderslev2022_p100_i50_iea10_single_blade.sim" \
  --qblade-root run_headless/QBladeCE_2.0.9.7 \
  --output-dir run_headless/runs/grinderslev2022_modal_power_qblade_mode_120s_dt_0p006_response \
  --result-name grinderslev2022_modal_power_qblade_mode_120s_dt_0p006_response \
  --result-filter examples/grinderslev2022_iea10/response_benchmark_results.filter \
  --threads 2 --no-use-opencl --progress-every 500 \
  --sil-vorlap --coupling-mode sil-vorlap --record-sil-vorlap-history

.venv/bin/python run_headless/run_qblade_sim.py \
  --sim "$INPUT_ROOT/dt_0p006/unloaded/grinderslev2022_p100_i50_iea10_single_blade.sim" \
  --qblade-root run_headless/QBladeCE_2.0.9.7 \
  --output-dir run_headless/runs/grinderslev2022_modal_power_qblade_mode_120s_dt_0p006_unloaded \
  --result-name grinderslev2022_modal_power_qblade_mode_120s_dt_0p006_unloaded \
  --result-filter examples/grinderslev2022_iea10/response_benchmark_results.filter \
  --threads 2 --no-use-opencl --progress-every 500 --coupling-mode none
```

Repeat those commands with `dt_0p003` for the fine pair. The generated
`run_plan.json` records all four argument lists and expected step counts. Once
the four runs complete, use the strict comparison:

```bash
.venv/bin/python examples/grinderslev2022_iea10/response_aware_benchmark.py analyze \
  --coarse-response-run run_headless/runs/grinderslev2022_modal_power_qblade_mode_120s_dt_0p006_response \
  --coarse-unloaded-run run_headless/runs/grinderslev2022_modal_power_qblade_mode_120s_dt_0p006_unloaded \
  --coarse-response-config "$INPUT_ROOT/dt_0p006/response/grinderslev2022_p100_i50_iea10_single_blade/Control/vorlap_qblade_external.json" \
  --coarse-unloaded-config "$INPUT_ROOT/dt_0p006/unloaded/grinderslev2022_p100_i50_iea10_single_blade/Control/vorlap_qblade_external.json" \
  --fine-response-run run_headless/runs/grinderslev2022_modal_power_qblade_mode_120s_dt_0p003_response \
  --fine-unloaded-run run_headless/runs/grinderslev2022_modal_power_qblade_mode_120s_dt_0p003_unloaded \
  --fine-response-config "$INPUT_ROOT/dt_0p003/response/grinderslev2022_p100_i50_iea10_single_blade/Control/vorlap_qblade_external.json" \
  --fine-unloaded-config "$INPUT_ROOT/dt_0p003/unloaded/grinderslev2022_p100_i50_iea10_single_blade/Control/vorlap_qblade_external.json" \
  --output-dir run_headless/runs/grinderslev2022_modal_power_qblade_mode_120s_comparison
```

The analyzer requires complete finite QBlade exports and version-2 SIL
histories, replays the typed response model from all 30 saved structural
velocities, checks every generated force, audits the actually applied
one-step-lag `F dot v`, and rejects any active response or source limit. It
also compares that area-lift work with `Q[n-1] qdot_raw[n]` from the same
record as a cause-isolation diagnostic; this comparison is not substituted
for the unchanged mapped-structural-model row. The
frozen coarse/fine gates are 1% for tip amplitude, root-moment DEL, and mean
applied power, and 5% for the rainflow range-power sum. The root moment, DEL,
and range-power products are numerical QBlade time-step comparisons only.
Without author-supplied high-fidelity root-load histories and blade stress/S--N
data, they are not high-fidelity root-load verification, Miner damage,
fatigue-life validation, or certification evidence.
