# Vlastos/Lupi Sc=5.18 cantilever-cylinder bridge screen

This example adds the experimental/numerical evidence rung missing from the
full-blade Grinderslev benchmark. Vlastos et al. (2026),
<https://doi.org/10.5194/wes-11-2585-2026>, model the Lupi et al. (2021)
wind-tunnel cylinder as a Timoshenko cantilever and compare hGAST predictions
against measured lock-in amplitudes. Publisher Table 1 exposes the 0.775 m
height, 0.050 m diameter, 15.38 Hz first frequency, 0.4311 kg/m equivalent
mass, 0.0028 damping ratio, and Sc=5.18 configuration. Figure 9 is available
as a CC BY vector figure and is digitized with provenance under
`reference_data/`.

The recreation represents the circular cantilever using QBlade's isolated
one-blade Timoshenko beam. VorLap supplies a coherent `CL0=0.3`, `St=0.2`
fixed-wake harmonic load. The reference hGAST calculation instead strongly
couples a response-dependent Hartlen--Currie lift oscillator to the beam and
updates correlation length. Consequently this example tests load
reconstruction and structural transfer, while its disagreement with the
measured/hGAST lock-in band is expected model-form evidence. It must not be
described as experimental VIV validation.

Generate the nine-speed case set with:

```bash
.venv/bin/python examples/vlastos2026_cantilever/generate_case.py
```

Run each generated `.sim` through `run_headless/run_qblade_sim.py` using the
same Linux/amd64 container workflow as the Grinderslev example. Use
`--sil-vorlap --coupling-mode sil-vorlap --record-sil-vorlap-history` for the
loaded cases. One `--coupling-mode none` run supplies the paired gravity and
startup baseline because native QBlade aerodynamics are disabled.

## Response-aware, three-structure benchmark

`response_benchmark.py` adds a pre-registered response-aware campaign without
changing the legacy fixed-wake example. It uses all three Table 1 structures,
requires at least 50 global QBlade steps per natural cycle, and generates both
the nominal time step and a halved-time-step case. Existing campaign
directories are never deleted or replaced.

The response-aware campaign disables the persistent `CL0=0.3` source with
`force_scale=0`. Each case instead receives the same deterministic five-cycle
Hann-tapered modal seed targeting `A_seed/D=0.005`; the seed is distributed as
mass times mode shape and is exactly zero afterward. This avoids both the
zero-motion invariant of an unseeded negative-damping model and contamination
of the steady response by a permanent harmonic source. Seed provenance and
force-limit counts are retained in runtime diagnostics. The first unforced
cycle must also realize `0.001 <= A/D <= 0.010`; final amplitudes still must
pass the declared stationarity checks.

The closure is calibrated only to the figure-digitized Sc=5.18 hGAST curve.
Its dimensionless activation, negative-damping gain, and saturation amplitude
are then fingerprinted before the Sc=2.94 and Sc=7.02 Figure 10 targets are
read. Experimental Lupi peaks are reported as a separate validation screen;
they are not tuning data. The simple closure currently **fails** the frozen
analytical held-out gate: it predicts `y_max/D=0.000` versus `0.540` for the
published Sc=2.94 hGAST peak and `0.198` versus `0.150` for Sc=7.02. This is
retained model-form evidence, not hidden by a held-out retune. A
phase-continuous wake oscillator is the next model rung before a production
claim.

The QBlade structures exactly retain the published equivalent mass, damping,
and natural frequency. With the example's `rho=1.225 kg/m^3`, however, those
values imply `Sc=2.845`, `4.953`, and `6.677`, respectively, rather than the
published `2.94`, `5.18`, and `7.02`. Both values are recorded in every case;
the generated structures are not described as exact-Sc matches. No single
air density reconciles all three rounded Table 1 values.

Generate the immutable case inputs (this writes inputs only; it does not run
the long campaign):

```bash
.venv/bin/python examples/vlastos2026_cantilever/response_benchmark.py generate \
  --output examples/vlastos2026_cantilever/generated_response_campaign
```

With no `--duration` argument, generation uses the predeclared, case-specific
stationarity caps from `stationary_campaign_protocol.py`. Every immutable case
runs to its full conservative cap, from 240 to 2520 natural cycles, before the
terminal result/runtime stationarity checks are applied. This is a fixed-cap
workflow, not adaptive stopping or continuation. The earlier uniform 8 s inputs
are transient diagnostics and are not eligible for qualification. `--duration`
remains available only for an explicitly short diagnostic campaign.
The protocol resolves and checks the parked simulation, callback, runtime swap
mapping, HAWT orientation, blade, beam, and rigid-tower semantics in addition
to checking every declared input hash; simply regenerating a manifest after a
material input change therefore does not preserve eligibility.
Qualification inputs are pinned to the repository at the standard Docker path
`/work/VorLap`; launch the container with that exact mount so the callback
cannot import a different checkout.

Generate and validate the immutable duration protocol outside the campaign
input tree:

```bash
.venv/bin/python examples/vlastos2026_cantilever/stationary_campaign_protocol.py generate \
  --campaign-manifest examples/vlastos2026_cantilever/generated_response_campaign/campaign_manifest.json \
  --output run_headless/runs/vlastos2026_response/stationary_protocol.json
```

Create a calibration-stage dry run plan:

```bash
.venv/bin/python examples/vlastos2026_cantilever/response_benchmark.py run \
  --campaign examples/vlastos2026_cantilever/generated_response_campaign/campaign_manifest.json \
  --results run_headless/runs/vlastos2026_response \
  --stage calibration \
  --plan-output run_headless/runs/vlastos2026_response/calibration_plan.json
```

`run` prints commands but executes nothing unless `--execute` is explicit.
The plan runs cases sequentially with four QBlade threads (below the eight-core
project cap), no checkpoints by default, and compact SIL history. Compact
history streams response, startup-seed, code-computed scalar work,
aggregate-force, and cryptographic one-step-lag evidence without retaining the
full per-node vector payload in memory. The scalar work is code-traceable but
is not independently reconstructible after the nodal vectors are discarded.
Held-out execution is additionally blocked by the recorded analytical failure
unless `--allow-failed-analytical-gate` is supplied for a clearly diagnostic
investigation. The live runtime is preflighted to confirm that the enabled
`modal_negative_damping` model was actually constructed; a legacy
frozen-phase bridge-comparator run is refused.

## Legacy frozen-phase bridge comparator

`matched_frozen_campaign.py` derives 36 matched fine-time-step controls from a
validated 72-case stationary response campaign. Structure, inflow, geometry,
external-library mapping, and the hash-bound `CYLINDER.h5` source remain
identical. The comparator changes `force_scale` from zero to one, disables the
modal response closure and startup seed, and runs to a predeclared 240-cycle
cap before applying the same terminal 40-cycle stationarity gate. Generation
and planning never launch QBlade; execution requires the separate explicit
`run --execute` gate and remains four-thread sequential.

This comparator is the operational legacy bridge behavior, not strict
open-loop one-way forcing. Even with the modal response closure disabled, the
base VorLap load calculation consumes QBlade structural velocity through
`V_rel = V_infinity - V_structure`; the manifest therefore records
`strict_open_loop_one_way=false`. A true open-loop control would require a
separate runtime branch that freezes or removes structural velocity from the
base load calculation. That branch is not part of this comparison.

Generate, validate, and plan without launching a solver:

```bash
.venv/bin/python examples/vlastos2026_cantilever/matched_frozen_campaign.py generate \
  --parent-campaign run_headless/runs/vlastos2026_response_stationary_v1/campaign/campaign_manifest.json \
  --output run_headless/runs/vlastos2026_response_stationary_v1/legacy_frozen_phase_campaign

.venv/bin/python examples/vlastos2026_cantilever/matched_frozen_campaign.py validate \
  --parent-campaign run_headless/runs/vlastos2026_response_stationary_v1/campaign/campaign_manifest.json \
  --campaign run_headless/runs/vlastos2026_response_stationary_v1/legacy_frozen_phase_campaign/campaign_manifest.json

.venv/bin/python examples/vlastos2026_cantilever/matched_frozen_campaign.py plan \
  --parent-campaign run_headless/runs/vlastos2026_response_stationary_v1/campaign/campaign_manifest.json \
  --campaign run_headless/runs/vlastos2026_response_stationary_v1/legacy_frozen_phase_campaign/campaign_manifest.json \
  --results run_headless/runs/vlastos2026_response_stationary_v1/legacy_frozen_phase_results \
  --output run_headless/runs/vlastos2026_response_stationary_v1/legacy_frozen_phase_plan.json
```

The plan is directly executable through the same audited module; no external
argv loop is needed. Execution remains impossible unless `--execute` is
present, runs one four-thread case at a time, preflights all disabled-response
configs before creating the results root, rejects nonempty outputs, audits each
completed runtime, and writes a hash-bound receipt:

```bash
.venv/bin/python examples/vlastos2026_cantilever/matched_frozen_campaign.py run \
  --parent-campaign run_headless/runs/vlastos2026_response_stationary_v1/campaign/campaign_manifest.json \
  --campaign run_headless/runs/vlastos2026_response_stationary_v1/legacy_frozen_phase_campaign/campaign_manifest.json \
  --results run_headless/runs/vlastos2026_response_stationary_v1/legacy_frozen_phase_results \
  --plan-output run_headless/runs/vlastos2026_response_stationary_v1/legacy_frozen_phase_execution_plan.json \
  --execute
```

Generate and execute the plan inside the standard `/work/VorLap` container
mount when using the Linux QBlade build; execution paths are resolved in that
environment while runner and filter identities remain repository-relative.

The analyzer compares exactly co-located response-aware and legacy
frozen-phase terminal amplitudes against zero VIV, the Sc=5.18 hGAST and Lupi
curves, and held-out Figure 10 peak amplitudes at the separately hash-bound
model-parameter peak locations. Its output is an amplitude-skill diagnostic
only. It does not establish local stress ranges, DEL, Miner damage, or
conservative fatigue bias.

After results exist, extract the half-range tip amplitude over the last 40
natural cycles, compare nominal and half time steps, and write the response
curves, publication-quality plot, and acceptance summary with:

```bash
.venv/bin/python examples/vlastos2026_cantilever/response_benchmark.py analyze \
  --campaign examples/vlastos2026_cantilever/generated_response_campaign/campaign_manifest.json \
  --results run_headless/runs/vlastos2026_response \
  --output run_headless/runs/vlastos2026_response/analysis
```

The overall status is an external hGAST response comparison and model-form
assessment. It is not equation-level verification because this compact
Scanlan-style closure does not solve hGAST's Hartlen--Currie equations and the
QBlade structure is reduced. The experimental differences remain explicitly
labeled as a validation screen. Verification claims are reserved for separate
same-equation or structural-transfer tests.

## Frozen Scruton-number engineering transfer

`engineering_sc_transfer.py` adds a case-level transfer layer around the
unchanged schema-v2 response model. The activation envelope is translated by
the frozen Belloli peak-location relation, `a_sat` is scaled by the frozen
Belloli peak-amplitude relation, and `H1` is unchanged. The layer is exactly
the original schema-v2 parameter set at `Sc=5.18` and refuses Scruton numbers
outside `[2.94, 7.02]`; schema-v1 and ordinary schema-v2 runtime behavior are
unchanged.

The `Sc=2.94` and `Sc=7.02` hGAST targets were available during selection from
a frozen 20-form menu. They are therefore **transfer tuning and engineering
screening evidence, not independent validation**. The analytical gate uses
`|A_pred-A_target| <= 0.30 A_target + 0.01D` and
`|U*_peak,pred-U*_peak,target| <= 0.25`. Recreate the tracked comparison JSON,
CSV, LaTeX table, and PDF with:

```bash
.venv/bin/python examples/vlastos2026_cantilever/engineering_sc_transfer.py artifacts \
  --output examples/vlastos2026_cantilever/engineering_sc_transfer_evidence
```

Generate and validate the exact eight-case, same-endpoint time-step-paired
input campaign without launching QBlade:

```bash
.venv/bin/python examples/vlastos2026_cantilever/engineering_sc_transfer.py generate \
  --output run_headless/runs/vlastos2026_belloli_sc_transfer

.venv/bin/python examples/vlastos2026_cantilever/engineering_sc_transfer.py validate \
  --campaign run_headless/runs/vlastos2026_belloli_sc_transfer/campaign_manifest.json
```

The generator's `plan` command remains an inputs-only listing. For actual
coupled execution, first finish the 24-case Sc=5.18 calibration and produce a
passing engineering calibration-readiness artifact. Then, in the same pinned
QBlade runtime used to generate the campaign, create a canonical array plan:

```bash
.venv/bin/python examples/vlastos2026_cantilever/engineering_holdout_run.py readiness \
  --analysis run_headless/runs/vlastos2026_response_stationary_v2_linux/analysis \
  --output run_headless/runs/vlastos2026_response_stationary_v2_linux/analysis/calibration_readiness.json

.venv/bin/python examples/vlastos2026_cantilever/engineering_sc_transfer_run.py plan \
  --manifest run_headless/runs/vlastos2026_belloli_sc_transfer/campaign_manifest.json \
  --calibration-readiness run_headless/runs/vlastos2026_response_stationary_v2_linux/analysis/calibration_readiness.json \
  --results /results/vlastos2026_belloli_sc_transfer \
  --qblade-root run_headless/QBladeCE_2.0.9.7 \
  --threads 4 \
  --checkpoint-every 1000 \
  --output run_headless/runs/vlastos2026_belloli_sc_transfer_execution_plan.json
```

Run one zero-based array index and analyze all eight results with:

```bash
.venv/bin/python examples/vlastos2026_cantilever/engineering_sc_transfer_run.py run \
  --manifest run_headless/runs/vlastos2026_belloli_sc_transfer/campaign_manifest.json \
  --calibration-readiness run_headless/runs/vlastos2026_response_stationary_v2_linux/analysis/calibration_readiness.json \
  --plan run_headless/runs/vlastos2026_belloli_sc_transfer_execution_plan.json \
  --case-index "${SLURM_ARRAY_TASK_ID}" \
  --acknowledge-transfer-tuning-screen

.venv/bin/python examples/vlastos2026_cantilever/engineering_sc_transfer_run.py analyze \
  --manifest run_headless/runs/vlastos2026_belloli_sc_transfer/campaign_manifest.json \
  --calibration-readiness run_headless/runs/vlastos2026_response_stationary_v2_linux/analysis/calibration_readiness.json \
  --results /results/vlastos2026_belloli_sc_transfer \
  --output /results/vlastos2026_belloli_sc_transfer_analysis
```

The runner revalidates the canonical plan and campaign at dispatch, executes
one index, and audits the shared schema-v2 runtime evidence. Analysis remains
`not_evaluated` until all eight cases are present. The completed screen passes
the Sc=2.94 amplitude gate at `0.42552D` versus `0.350D` (21.6% error), but the
Sc=7.02 result of `0.08933D` versus `0.150D` (40.4% error) misses its frozen
tolerance by `0.00567D`. All numerical and runtime gates pass. The overall
conjunctive assessment is therefore a retained negative engineering result,
not independent validation.

For native-Linux HPC execution, generate the campaign, readiness artifact, and
canonical plan at their final absolute paths, then submit
`run_headless/submit_vlastos_sc_transfer.sj`. Its `0-7%2` array runs two
four-thread cases at once, enforcing the project-wide eight-core cap. Export
`VORLAP_ROOT`, `TRANSFER_ROOT`, `CALIBRATION_READINESS`, and `PLAN` when the
site paths differ from the defaults. The plan's results root must be outside
the repository, campaign, and QBlade trees.
