# Ehrmann et al. (2014) NACA 0018 lock-in benchmark

This directory defines the primary moderate-Reynolds-number response-qualification
campaign for VorLap--QBlade bridge revisions. Ehrmann et al. combines the
tracked NACA 0018 profile with measured, freely responding lock-in amplitudes
near chord Reynolds number 1.25e5. The more turbine-specific studies screened
for this campaign did not provide the same combination of moderate Reynolds
number, free response, and reproducible amplitude data.

**Current primary path:** use the static-spectrum engineering negative-damping
model and schema-v5 Flight command below. Nalu and the prescribed-motion
work-map workflow are retained as optional higher-fidelity research
infrastructure; they are not required to run or qualify this engineering
comparison.

The completed schema-v2 comparison contains three series:

1. the published flexibly mounted NACA 0018 wind-tunnel response;
2. raw QBlade with native aerodynamics and no supplemental VorLap force; and
3. the same QBlade case with the now-rejected schema-v2 airfoil-spectral
   activation plus cylinder-derived response gain.

Ehrmann et al., *Journal of Fluids and Structures* 44 (2014), 205--215,
<https://doi.org/10.1016/j.jfluidstructs.2013.10.008>, is the primary case
because its NACA 0018 profile matches the tracked VorLap source.  The published
experiment used a nominal 6 Hz elastic system at 90 degrees incidence, reported
mass-damping from 0.96 to 1.44, and observed lock-in near chord Reynolds number
1.25e5.  The repository NACA 0018 spectral grid brackets that Reynolds number
(1.0e4, 5.0e5, and 2.0e6), but those three planes are exact replicas of one
Re=5.0e5 stationary calculation. They therefore supply a frozen shedding
frequency/available-energy source only; they do not identify Reynolds-number
dependence or force--motion phase.

The executable campaign in `campaign.py` uses the reported 230 mm chord,
90-degree incidence, nominal 6 Hz natural frequency, and `m*=47.7`.  Because
the accessible source does not assign a unique NACA0018 damping value within
the reported `0.96 <= m*zeta <= 1.44` range, the campaign runs both endpoints;
it does not infer an unreported pairing.  Its `3.0 <= U* <= 6.3` grid brackets
the reported lock-in Reynolds number near 125,000, with a paired timestep at
`U*=5.8`.

The structural model is a first-bending cantilever surrogate matched in
`fn`, generalized modal `m*`, and `zeta`, not a claim of exact apparatus
geometry. Its 0.9144 m span is a declared numerical surrogate used for a
well-conditioned one-mode beam; it is not presented as a measured Ehrmann
apparatus dimension. The distributed beam mass is chosen so the tip-normalized
first-mode generalized mass reproduces the published effective mass per span.
The executed native-QBlade axis probe shows that the chord/global-Y cross-flow
bending direction uses `EIy` in this generated orientation. `EIx` is therefore
set to 100 times `EIy`, separating global-X/streamwise bending by approximately
a factor of ten in frequency. The structural-verification probe binds the
analyzed response channel to this measured global-Y direction.

`structural_verification.py` reproduces the native-aero-off verification in
`structural_verification_report.json`. The executed paired global-Y probe maps
uniquely to `Y_b Tip Trl.Def. BLD_1 [m]` (correlation 1.0; the paired `X_b`
response is below `6e-17 m`). QBlade free decays recover 5.970 Hz at both
damping endpoints and damping ratios 0.01956 and 0.02971, compared with the
targets 0.02013 and 0.03019. QBlade's result export does not expose generalized
modal mass; the report therefore binds that quantity to the generated beam
mass and tip-normalized mode quadrature, which gives 2.219914 kg and `m*=47.7`.
Campaign generation and validation require the exact tracked report SHA-256,
its passing schema/status, the verified `Y_b` channel, the 100:1 stiffness
ratio, both frequency/damping checks, and the modal-mass check. Reproduce the
four native-QBlade cases (4 CPU maximum) with:

```bash
./run_headless/docker_verify_ehrmann2014_structure.sh
```
Raw QBlade
and hybrid cases share byte-identical geometry, structure, native aerodynamic
polars, inflow, timestep, and duration. The raw controller has zero modal
gain, no startup seed, and zero residual scale. The superseded schema-v2
bridge stretches only captured NACA0018 CL spectral energy into the measured
cross-flow mode;
the residual scale is deliberately zero because the source wind-axis to
section-axis force semantics at 90 degrees are not yet qualified. Native
QBlade supplies the baseline aerodynamic load in both scenarios.

## Completed schema-v2 response validation

The Flight campaign completed all 40 matched cases under campaign fingerprint
`23903688f3c33cc23c2962dd834018f17cfe7652a8f4cf7dceec3f64df64cc94`.
Publisher Figures 6 and 7 are digitized under `reference_data/` with source
URLs, raster hashes, pixel coordinates, axis transforms, and explicit
graphical-extraction allowances. Compact promoted evidence is under
`validation_evidence/`; reproduce it from the ignored campaign archive with:

```bash
MPLCONFIGDIR=/tmp/vorlap_mplcache .venv/bin/python \
  -m examples.ehrmann2014_naca0018.build_validation_evidence
```

The completed validation fails. Here, "experiment" means the digitized
Ehrmann curve, "raw QBlade" means native QBlade with exactly zero external
response force, and "schema-v2 bridge" means the spectral-activation model
with the cylinder-derived signed gain. At `m*zeta=0.96`, the digitized
experimental peak is at least `0.012366c`, while raw QBlade and the schema-v2
bridge reach only
`1.59e-9c` and `4.19e-7c`. At `m*zeta=1.44`, the mapped experimental peak is
`0.006817c`, versus `2.17e-11c` and `4.26e-9c`. The paired-timestep amplitude
difference is at most `4.13e-10c`, so this is numerical agreement on the wrong
physical state. The active hybrid seed decays from `3.38e-3c` near 1 s to
`4.73e-9c` at 20 s; the response-dependent force does not overcome structural
damping. The result does not authorize the H-VAWT full rerun. Revise the force
law and repeat this independent benchmark before using the bridge for physical
response or fatigue claims.

## Airfoil force--motion work-map replacement

The replacement method is implemented separately as
`single_mode_airfoil_force_motion_work_map_v1`; it is not one of the curves in
the completed schema-v2 comparison. The tracked stationary NACA0018 spectra
still provide shedding frequency and available fluctuating-lift energy, but
their arbitrary FFT phase is not used to infer aerodynamic work. Signed
supplemental power instead comes from paired prescribed-motion histories:
high-fidelity NACA0018 cross-flow force minus the force from the exact native
QBlade runtime base at the same motion. QBlade retains responsibility for
Rayleigh damping; the bridge injects only positive lower-confidence
incremental aerodynamic work through an energy-conjugate generalized force.

The exact first-slice request is `work_map_data_request.json`: 48 nonzero-motion
pairs over incidence, Reynolds number, reduced frequency, and amplitude, plus
12 exact analytical zero-amplitude vertices. Build an auditable map and report
with:

```bash
.venv/bin/python -m examples.ehrmann2014_naca0018.build_airfoil_work_map \
  --manifest MANIFEST.json \
  --output-map NACA0018_work_map.json \
  --output-report NACA0018_work_map_report.json
```

Campaign generation rejects a map unless it contains positive work and the
95% lower-confidence work is nonpositive at the largest mapped amplitude.
Only after those paired histories populate the map will the 20 identity-changed
hybrid cases be rerun. The 20 completed raw-QBlade cases are eligible for exact
identity/hash reuse. No new work-map result is currently promoted to the paper,
and no H-VAWT response rerun is authorized yet.

### Native-QBlade prescribed-motion force matrix

`qblade_prescribed_motion.py` generates the exact-runtime-base half of the
paired identification set. It uses one effectively rigid, nonrotating NACA0018
blade with 0.23 m chord and 4.0 m span. Native QBlade aerodynamics remain on;
the controller, external-library callback, SIL force, and every supplemental
force are off. A `MOTIONFILE` prescribes global-Y translation, and the canonical
force is the total `Aerodynamic Force in Hub Y_h Direction [N]`. With zero yaw,
shaft tilt, and global rotation, hub Y is global cross-flow Y.

The geometry audit is explicit about the incidence convention. Chord is global
+Y, section normal is global -X, and ambient wind is global +X. QBlade reports
this as approximately +90 degrees, while VorLap evaluates the same relative
wind as `atan2(U.normal,U.chord)=-90` degrees. Pitch and twist remain zero; no
180-degree pitch transformation is used. Each completed case must also show
that the blade global-Y motion follows the prescribed platform motion, that the
panel Reynolds number matches the requested vertex, and that callback update
count is exactly zero.

Flight execution has an explicit two-stage data gate. The preflight requests
one shared-node CPU slot and executes only case 35,
`re140000_k0p45_a0p0500`: the longest-grid, largest-motion, high-Reynolds-number
representative. It must produce a complete force/motion/channel audit and a
provenance-bound scheduler report. Only then does the dependent matrix job
request 47 shared-node CPU slots and execute the other 47 independent one-core
cases. Every requested slot is assigned; unrequested cores remain available to
the scheduler. Submit both stages with:

```bash
preflight_job=$(sbatch --parsable \
  --export=ALL,VORLAP_ROOT=/pscratch/kevmoor/vorlap,CAMPAIGN_ROOT=/pscratch/kevmoor/vorlap_runs/naca0018_work_map_qblade_v1 \
  /pscratch/kevmoor/vorlap/run_headless/submit_naca0018_work_map_qblade_preflight.sj)

sbatch \
  --dependency=afterok:${preflight_job} \
  --export=ALL,VORLAP_ROOT=/pscratch/kevmoor/vorlap,CAMPAIGN_ROOT=/pscratch/kevmoor/vorlap_runs/naca0018_work_map_qblade_v1,QBLADE_STAGE=matrix \
  /pscratch/kevmoor/vorlap/run_headless/submit_naca0018_work_map_qblade.sj
```

The same exact-runtime-base matrix can instead be produced on the local Docker
instance without Slurm. This is a distinct, provenance-recorded execution
environment: the launcher binds the immutable Docker image ID, gives every
QBlade case one thread, executes the preflight first, and runs at most eight
matrix cases concurrently inside a container capped at eight CPUs. It resumes
checkpoints and exact completed audits in place:

```bash
VORLAP_QBLADE_LOCAL_CPUS=8 \
  ./run_headless/docker_run_naca0018_work_map_qblade.sh
```

The default local campaign is retained under
`run_headless/runs/naca0018_work_map_qblade_v2`; its complete
`qblade_results.json` is accepted by the same paired-history assembler. This
replaces only the exact native-QBlade producer. It does not replace the
independent high-fidelity force histories needed to identify the signed work
map.

The wrappers validate the exact QBlade 2.0.9.7 library and dependency hashes,
checkpoint-resume incomplete cases, and postprocess every history. The matrix
stage automatically writes the complete `qblade_results.json`, including the
exact preflight case-audit and scheduler-report identities, for the paired
history assembler. The default wall time is two hours. If the matrix expires,
resubmit it with eight hours; completed histories are skipped and incomplete
cases resume from their 1,000-step checkpoints:

```bash
sbatch \
  --time=08:00:00 \
  --export=ALL,VORLAP_ROOT=/pscratch/kevmoor/vorlap,CAMPAIGN_ROOT=/pscratch/kevmoor/vorlap_runs/naca0018_work_map_qblade_v1,QBLADE_STAGE=matrix \
  /pscratch/kevmoor/vorlap/run_headless/submit_naca0018_work_map_qblade.sj
```

For a submission-side launch-plan check that runs no QBlade cases, add
`QBLADE_WORK_MAP_DRY_RUN=1` to the relevant `--export`. The shared preflight
report is `qblade_preflight_scheduler_report.json`; the matrix report is
execution-specific (`qblade_matrix_flight_scheduler_report.json` or
`qblade_matrix_local_scheduler_report.json`). A dry run does not satisfy the
pre-data gate; the matrix still requires an executed, complete preflight report.

### Nalu-Wind prescribed-motion force matrix

`nalu_prescribed_motion.py` generates the high-fidelity half of the same 48
vertices. It uses the frozen Nalu-Wind 2.2.1 SST-IDDES runtime, the original
periodic span-4 m NACA0018 mesh, and a one-time `[0.23,0.23,1.0]` mesh scaling
in three zero-amplitude ALE precursors. Motion cases restart from those scaled,
developed-wake precursors and translate the complete mesh in global Y. The
canonical total cross-flow force is `Fpy+Fvy` on the combined
`[wing,wing-pp]` surface. Positive force and velocity are both global +Y, so
positive cycle work is flow-to-structure work.

This is the optional high-fidelity identification path for the force--motion
work map; it is not used by the engineering lock-in closure.  The submission
scripts do not depend on another user's private Flight environment.  The
solver identity is the official Nalu-Wind v2.2.1 release at Git commit
`ffa9de729df2a11b5241fdeb7628e7fab9f48f9b`, and every completed stage binds
the exact `naluX` SHA-256.

Before submission, run the read-only Flight audit. It checks the VorLap Python
environment, resolves `naluX` from an explicit path, `PATH`, or an active Spack
environment, and hashes the exact Exodus mesh:

```bash
export VORLAP_ROOT=/pscratch/kevmoor/vorlap
set -o pipefail
bash "${VORLAP_ROOT}/run_headless/audit_naca0018_work_map_flight_inputs.sh" \
  2>&1 | tee /pscratch/kevmoor/vorlap_runs/naca0018_work_map_flight_input_audit.txt
```

The audit exits 2 until both production inputs are available. If Nalu is
already installed outside the active environment, export its absolute path;
an accessible setup script remains supported for the submitted jobs:

```bash
export NALU_EXECUTABLE=/absolute/path/to/naluX
test -x "${NALU_EXECUTABLE}"
sha256sum "${NALU_EXECUTABLE}"
# Alternatively: export NALU_FLIGHT_ENV=/accessible/project/setup_env.sh
```

Completed logs must prove the official v2.2.1 version and clean release commit.
The workflow therefore fails closed if an arbitrary `naluX` happens to be on
`PATH`.

The exact remote Exodus path and SHA-256 are deliberately not guessed. On
Flight, first identify the `-90` mesh and bind its bytes:

```bash
export VORLAP_ROOT=/pscratch/kevmoor/vorlap
export NALU_CAMPAIGN=/pscratch/kevmoor/vorlap_runs/naca0018_prescribed_motion_nalu_v1
export NALU_MESH=/gpfs/kevmoor/airfoil_vortexshedding_mesh_run_post_nalu/mesh_gen_scripts/grids/NACA0018-90.exo
test -f "${NALU_MESH}"
export NALU_MESH_SHA256="$(sha256sum "${NALU_MESH}" | awk '{print $1}')"
```

Run the required 2/4/8-full-node scaling probes. Each submitted allocation uses
all 112 physical cores per node; the three jobs are independent and may run
concurrently:

```bash
for nodes in 2 4 8; do
  tasks=$((112 * nodes))
  sbatch \
    --time=02:00:00 \
    --nodes="${nodes}" --ntasks="${tasks}" --ntasks-per-node=112 \
    --export=ALL,VORLAP_ROOT="${VORLAP_ROOT}",CAMPAIGN_ROOT="${NALU_CAMPAIGN}",NALU_MESH="${NALU_MESH}",NALU_MESH_SHA256="${NALU_MESH_SHA256}",NALU_PREFLIGHT_STAGE=scaling \
    "${VORLAP_ROOT}/run_headless/submit_naca0018_work_map_nalu_preflight.sj"
done
```

After all three finish, the status gate selects the smallest measured node
count whose precursor and longest motion both project below two hours. If none
does, it falls back to the smallest measured node count below eight hours and
records that choice explicitly:

```bash
export PYTHON_EXE=/pscratch/kevmoor/vorlap/.venv/bin/python
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.nalu_prescribed_motion status \
  --campaign "${NALU_CAMPAIGN}" --require-mesh-audit --require-scaling
export NALU_NODES="$(jq -er '.scaling_gate.recommended_nodes | select(. == 2 or . == 4 or . == 8)' "${NALU_CAMPAIGN}/nalu_results_manifest.json")"
export NALU_WALLTIME="$(jq -er '.scaling_gate.recommended_walltime | select(. == "02:00:00" or . == "08:00:00")' "${NALU_CAMPAIGN}/nalu_results_manifest.json")"
export NALU_TASKS=$((112 * NALU_NODES))
```

Then run the three Re-specific precursors, one worst-case moving-mesh preflight,
and only after it passes the remaining 47 motions:

```bash
precursor_job=$(sbatch --parsable \
  --time="${NALU_WALLTIME}" \
  --nodes="${NALU_NODES}" --ntasks="${NALU_TASKS}" --ntasks-per-node=112 --array=0-2 \
  --export=ALL,VORLAP_ROOT="${VORLAP_ROOT}",CAMPAIGN_ROOT="${NALU_CAMPAIGN}",NALU_STAGE=precursor \
  "${VORLAP_ROOT}/run_headless/submit_naca0018_work_map_nalu.sj")

motion_preflight_job=$(sbatch --parsable --dependency=afterok:${precursor_job} \
  --time="${NALU_WALLTIME}" \
  --nodes="${NALU_NODES}" --ntasks="${NALU_TASKS}" --ntasks-per-node=112 \
  --export=ALL,VORLAP_ROOT="${VORLAP_ROOT}",CAMPAIGN_ROOT="${NALU_CAMPAIGN}",NALU_MESH="${NALU_MESH}",NALU_MESH_SHA256="${NALU_MESH_SHA256}",NALU_PREFLIGHT_STAGE=motion \
  "${VORLAP_ROOT}/run_headless/submit_naca0018_work_map_nalu_preflight.sj")

sbatch --dependency=afterok:${motion_preflight_job} \
  --time="${NALU_WALLTIME}" \
  --nodes="${NALU_NODES}" --ntasks="${NALU_TASKS}" --ntasks-per-node=112 --array=0-46 \
  --export=ALL,VORLAP_ROOT="${VORLAP_ROOT}",CAMPAIGN_ROOT="${NALU_CAMPAIGN}",NALU_STAGE=motion \
  "${VORLAP_ROOT}/run_headless/submit_naca0018_work_map_nalu.sj"
```

The production array serializes its final whole-campaign audit, so concurrent
case completions cannot overwrite a complete result with a stale incomplete
manifest. After the array ends, require all producer gates explicitly:

```bash
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.nalu_prescribed_motion status \
  --campaign "${NALU_CAMPAIGN}" --require-mesh-audit --require-scaling \
  --require-precursors --require-motion-preflight --allocated-nodes "${NALU_NODES}"
jq -e '.status == "complete" and .summary.complete_motion_cases == 48 and .summary.failed_motion_cases == 0' \
  "${NALU_CAMPAIGN}/nalu_results_manifest.json"
```

### Identification bundle and map

Only the two complete producer manifests may be assembled. The assembler
rechecks their solver, mesh, QBlade library, geometry/frame, force-channel,
preflight, scaling, callback, and per-case history identities; it then requires
byte-identical time/displacement/velocity columns before subtracting forces.

```bash
export QBLADE_CAMPAIGN=/pscratch/kevmoor/vorlap_runs/naca0018_work_map_qblade_v1
export MAP_ROOT=/pscratch/kevmoor/vorlap_runs/naca0018_work_map_v1
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.assemble_airfoil_work_map_identification \
  --nalu-results "${NALU_CAMPAIGN}/nalu_results_manifest.json" \
  --qblade-results "${QBLADE_CAMPAIGN}/qblade_results.json" \
  --static-spectrum "${VORLAP_ROOT}/examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/VorLapAirfoils/NACA0018.h5" \
  --output "${MAP_ROOT}/identification_bundle"

"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.build_airfoil_work_map \
  --manifest "${MAP_ROOT}/identification_bundle/identification_manifest.json" \
  --output-map "${MAP_ROOT}/NACA0018_work_map.json" \
  --output-report "${MAP_ROOT}/NACA0018_work_map_report.json"
```

The builder exits nonzero when the measured positive-work region is absent or
still positive at the largest mapped amplitude. In that case the written map
is diagnostic only and must not be submitted to the Ehrmann or H-VAWT runners.

## Engineering spectral negative-damping model (primary path)

Nalu and a prescribed-motion work map are no longer prerequisites for the
engineering comparison. The primary new model uses the full tracked static
NACA0018 lift spectrum to calculate the coefficient energy whose shedding
frequencies fall in the structural lock-in band. It then applies the
energy-conjugate Scanlan--Vickery feedback

\[
F_{fb}=\tfrac12\rho U c\,H C_{L,cap}\,v
\left[1-(A/(\lambda c))^2\right].
\]

The bridge therefore contributes negative aerodynamic damping below the
limiting amplitude, zero mean work at the limit, and positive damping above
it. QBlade still applies the structural Rayleigh damping and native
aerodynamics; the bridge does not subtract or duplicate them. The spectrum
supplies every harmonic's frequency and energy, while modal mass, frequency,
shape, and damping determine whether negative damping exceeds structural
damping.

`calibrate_engineering_lockin.py` freezes only two response constants. The two
outer Ehrmann Figure 7 markers identify the limiting amplitude and onset
mass-damping threshold. The three untouched interior markers have 8.8%, 10.2%,
and 14.6% amplitude error. With those constants frozen, the static spectrum
and the reported 25% tunnel blockage predict the unused Figure 6 peak at
`U*=4.713`, 2.8% above the digitized `U*=4.586`. This is an internal
calibration/holdout result, not independent cross-airfoil validation. The
tracked HDF5 Reynolds planes are replicated, so Reynolds transfer remains
unqualified.

The schema-v5 campaign compares only raw QBlade and the latest engineering
bridge. It retains the original broad `U*=3.0--6.3` grid and adds `U*=4.6`,
`4.7`, and `4.8` to resolve the predicted high-damping lock-in peak instead of
artificially widening the capture model. The paired timestep is evaluated at
the resolved peak, `U*=4.7`, where both damping cases have positive-work
branches. It reuses 18 nominal raw cases from the completed campaign and runs
8 new raw cases plus all 26 new hybrids.

The 52-case campaign has also been completed locally in the capped Linux
QBlade container under fingerprint
`496c92e1f2cd302313e6d0a588f791c07a7e5a808df9b887a93984c7850f46e6`.
At `m*zeta=0.96`, the latest bridge gives `A/c=0.012842` versus the
digitized `0.012366` (3.8% high); at `m*zeta=1.44`, it gives `0.005473`
versus `0.006817` (19.7% low). Both peaks occur at `U*=4.7`, both resolve
adjacent finite lock-in points, and both satisfy the frequency, stationarity,
and paired-timestep checks. Raw-QBlade peaks remain below `1.6e-9 c`.
Compact, repository-resident evidence is in
`engineering_validation_evidence/`: the JSON report identifies the experiment,
raw QBlade, and latest VorLap--QBlade roles; the 52-row case metrics and plot
data are CSV; the comparison is stored as PDF and PNG; and `SHA256SUMS` binds
all five artifacts. This qualifies the same-profile engineering screen only;
it is not independent cross-airfoil or H-VAWT validation.

The Flight command below reproduces the same comparison on the production
platform; it is no longer required to establish the local 52-case result:

```bash
sbatch \
  --export=ALL,VORLAP_ROOT=/pscratch/kevmoor/vorlap,CAMPAIGN_ROOT=/pscratch/kevmoor/vorlap_runs/ehrmann2014_engineering_lockin_v5,CALIBRATION=/pscratch/kevmoor/vorlap/examples/ehrmann2014_naca0018/engineering_lockin_calibration.json,RAW_REUSE_ROOT=/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_validation_v1/results \
  /pscratch/kevmoor/vorlap/run_headless/submit_ehrmann2014_engineering_flight.sj
```

The shared-node allocation uses 34 one-core QBlade steps for the 34 new cases;
all allocated cores are occupied on a fresh campaign and the remaining Flight
cores stay scheduler-visible. If two hours is insufficient, repeat the exact
submission with `--time=08:00:00`; complete results are identity-checked and
skipped, and incomplete cases use QBlade checkpoint resume.

The prescribed-motion/Nalu workflow below remains an optional higher-fidelity
future qualification route. It is not on the critical path for this model or
the schema-v5 Ehrmann run.

### Post-map Flight submission

Once the signed work map has been built, use the dedicated schema-v2 launcher;
do not reuse the legacy 40-case wrapper. `WORK_MAP` must name the exact regular
JSON map file. `RAW_REUSE_ROOT` must name the `results/` directory of the frozen
completed campaign, so that its immediate children are the 20 `*_raw` case
directories. Both inputs are mandatory and symlinks are refused. For example:

```bash
sbatch \
  --export=ALL,VORLAP_ROOT=/pscratch/kevmoor/vorlap,CAMPAIGN_ROOT=/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_work_map_validation_v2,WORK_MAP=/pscratch/kevmoor/vorlap_runs/naca0018_work_map_v1/NACA0018_work_map.json,RAW_REUSE_ROOT=/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_validation_v1/results \
  /pscratch/kevmoor/vorlap/run_headless/submit_ehrmann2014_work_map_flight.sj
```

The wrapper generates and atomically publishes the exact 40-case work-map
campaign, but schedules only the 20 identity-changed hybrids. It requests 20
physical cores on a shared Flight node and runs 20 one-core QBlade steps on a
fresh campaign, so every allocated core is occupied while the scheduler can
use the other node cores for other jobs. Exact complete results are skipped on
resubmission and incomplete hybrids resume from their 5,000-step checkpoints.
If two hours is insufficient, repeat the identical command with
`--time=08:00:00`.

After all 40 rows (20 reused raw plus 20 new hybrid) are complete, the same job
runs `campaign analyze` and builds the frozen schema-v2 acceptance evidence
under `CAMPAIGN_ROOT/validation_evidence`. It does not copy a result into the
paper; paper promotion remains conditional on that evidence gate.

## Flight execution

The primary Flight launcher requests 40 physical cores and 80 GiB on one shared
112-core, eight-NUMA-domain node. The tracked
`thread_scaling_benchmark.json` Docker smoke (hybrid case index 1, 500 steps)
took 13.323872 s with four threads and 13.737617 s with one; both summaries
reported `failed=false` and `exported_results=true`, and the exported numeric
row was identical. Four threads therefore shortened this smoke by only 3.0118%
while reserving four times as many cores. This is operational allocation
evidence, not physical validation evidence. One thread per case gives
materially better allocation throughput. On a fresh campaign with no reused
results, all 40 allocated cores are occupied by independent QBlade cases;
resume or exact-result reuse can reduce the runnable count. Because the batch
job intentionally omits `#SBATCH --exclusive`, the other 72 node cores remain
available to the scheduler.

The Python scheduler starts one
`srun --exclusive --exact -N1 -n1 -c1 --cpu-bind=cores` step per case.
Job-step `--exclusive` prevents overlap inside the 40-core allocation; it does
not request an exclusive node. `--hint=nomultithread` binds the allocation to
physical cores, and the wrapper refuses campaign-slot or node-topology
mismatches. Flight reports `SLURM_CPUS_ON_NODE=112`; that is a node-visible
topology value, not this campaign's task-slot count. The wrapper therefore
audits the campaign slots as `SLURM_NTASKS * SLURM_CPUS_PER_TASK = 40` and
audits the 112-core, eight-NUMA-domain node topology independently.
Runnable cases are ordered by decreasing `num_steps`, putting indices 36--39
first, although all 40 launch together on a fresh campaign. The scheduler
dynamically backfills, stops launching new cases after a failure, and records
per-case commands, logs, return codes, and elapsed wall time in
`flight_scheduler_report.json`. The initial allocation is two hours:

The self-contained wrapper does not require a `.venv` inside the current
uppercase checkout or a separately deployed resolver script.
It first honors an explicit `PYTHON_EXE`, then checks the current checkout's
`.venv`, then reuses the proven frozen-bridge environment at
`/pscratch/kevmoor/vorlap/.venv/bin/python`, and finally checks `python3`.
Whichever interpreter is selected must import NumPy, SciPy, h5py, and the
Ehrmann/VorLap modules from the current `VORLAP_ROOT`; otherwise submission
fails before any QBlade steps launch.

The same submission wrapper now prepares the campaign when
`CAMPAIGN_ROOT/campaign_manifest.json` is absent. It generates into a sibling
staging directory, validates all 40 cases, and only then atomically publishes
the campaign at `CAMPAIGN_ROOT`. A pre-existing root without a valid manifest
is refused rather than overwritten. `RAW_REUSE_ROOT` remains optional and is
honored during this automatic preparation with the same exact-identity checks
as the standalone preparation script. No separate prepare command is required
for a fresh campaign.

```bash
sbatch --export=ALL,VORLAP_ROOT=/pscratch/kevmoor/vorlap,CAMPAIGN_ROOT=/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_validation_v1 /pscratch/kevmoor/vorlap/run_headless/submit_ehrmann2014_flight.sj
```

If that allocation reaches its wall-time, submit the identical job with eight
hours. Per-case 5,000-step QBlade checkpoints and scheduler resume checks make
this safe: exact complete results are hash/identity checked and skipped, while
incomplete cases resume. Existing results with a mismatched identity are
refused. The wrapper generates the campaign analysis JSON/CSV only after every
scheduled case completes successfully.

```bash
sbatch --time=08:00:00 --export=ALL,VORLAP_ROOT=/pscratch/kevmoor/vorlap,CAMPAIGN_ROOT=/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_validation_v1 /pscratch/kevmoor/vorlap/run_headless/submit_ehrmann2014_flight.sj
```

The original `submit_ehrmann2014_validation.sj` `%2` array remains available
as a conservative low-concurrency fallback. It is not the primary Flight
submission path.

The companion comparison is deliberately narrow. It evaluates peak cross-flow response,
peak reduced velocity, the observed lock-in interval, dominant response
frequency, and paired-timestep repeatability.  Peak amplitude is acceptable for
the present screening purpose when the latest bridge is within
`0.30*A_experiment + sigma_digitization`. The older
`0.30*A_experiment + 0.01*c` rule is retained only with legacy evidence and is
not used to qualify the engineering closure. Raw QBlade is the sole numerical
baseline.

Benner et al. (2019), <https://doi.org/10.1016/j.jfluidstructs.2019.01.018>,
is retained only as VAWT-specific literature context: its Reynolds-number range
is below the requested validation regime, so it is not part of this campaign
or its acceptance gate. Fixed wind-turbine-blade shedding studies can support
the high-incidence mechanism, but cannot replace Ehrmann's freely responding
amplitude data.
