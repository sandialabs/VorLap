# Headless workflows for the SIL paper

The source release retains the qualified campaign code and compact evidence.
QBlade CE 2.0.9.7 is installed separately, rather than redistributed in Git.
See [paper reproducibility](../docs/reproducibility.md) for the separate
manuscript snapshot and the distinction between paper and simulation replay.
The historical branch name and execution paths below identify original runs;
a new checkout or release commit is not a fresh qualification of those runs.

This branch contains the unified semi-empirical VIV gain model, its frozen
training/validation evidence, the restored manuscript, and the H-VAWT
application campaign.
The authoritative execution contract is
[`../unified_viv_gain_model_codex_goal.md`](../unified_viv_gain_model_codex_goal.md).

The unified workflow is integrated alongside the historical parked-demo,
changed-geometry, Vlastos, NACA0018 work-map, Fu Flight reproduction, Ehrmann,
and recovery launchers. Those older workflows remain available in this main
checkout; the manuscript and its active figure generator use only the evidence
declared in the current evidence matrix.

## Active H-VAWT inputs

- `config/hvawt_response_model_v8_complete.json`: current complete response model.
- `config/hvawt_response_campaign_spec.json`: current campaign and claim boundary.
- `config/geometry_blended_airfoil_calibration.json`: frozen NACA0018 engineering
  calibration used by v8.
- `config/hvawt_response_mode_identification.json`: reviewed modes 6 and 12.
- `config/hvawt_response_mode_shapes.csv` and
  `config/hvawt_response_mode_force_duals.csv`: response shapes and exact force
  adjoint.
- `config/parked_dlc_matrix_approved.json`: approved 34-row application matrix.
- `config/parked_fatigue_results.filter`: required QBlade result selection.

The QBlade runtime remains `QBladeCE_2.0.9.7`. Do not replace its library,
bridge, environment, or hashes without reopening runtime qualification.

## Active H-VAWT commands

Prepare the current response cases locally or on the staged Flight checkout:

```bash
bash run_headless/prepare_hvawt_response_pilot_native.sh
```

The preparation script prints the exact sealed submission commands. The
prepared two-speed 120-second package is an inactive contingency, documented in
[`../hvawt_two_speed_120s_flight_test.md`](../hvawt_two_speed_120s_flight_test.md);
the active sequence intentionally skips it after a passing preflight.
The complete staged execution sequence through the conditional 102-case screen
is recorded in
[`../unified_hvawt_flight_execution.md`](../unified_hvawt_flight_execution.md).
The full 20-case job uses one 660 s case per `0-19%20` array task, allowing all
20 independent cases to run concurrently when Flight resources permit.
Measured preflight throughput predicts about 38.4 h per case; do not combine
an H0/complete pair sequentially inside the 47:59 wall request.
The failed v4 full-duration root is retained as evidence of the retired,
uncalibrated `100 N` runtime abort. Active v8/full-DLC cases reject nonfinite
forces and record the applied force norm but do not clip or impose an arbitrary
finite ceiling. Generate a fresh v5 root; do not resume or mix v4 results.

Active response entrypoints are:

- `hvawt_response_pilot.py`;
- `submit_hvawt_two_speed_120s_test.sj`;
- `submit_hvawt_response_preflight.sj`;
- `submit_hvawt_response_pilot.sj`; and
- `submit_hvawt_response_analysis.sj`.

Active complete-DLC entrypoints are:

- `prepare_hvawt_complete_dlc_native.sh`;
- `hvawt_complete_dlc_campaign.py`;
- `submit_hvawt_complete_dlc_u50_smoke.sj`;
- `submit_hvawt_complete_dlc_flight.sj`; and
- `submit_hvawt_complete_dlc_analysis.sj`.

No Flight submission is authorized merely by this README or the Codex goal.
Submission requires an explicit user instruction.

## Retained validation evidence

The primary protected validation is the 11-condition Horcas `COMB` assessment:

```text
examples/horcas2020_iea10_tip_geometry/evidence/combined_tip_v1/assessment.json
```

The corresponding frozen model and protocol are beside that artifact. Ehrmann
schema-v5 evidence is retained as NACA0018 same-study engineering qualification.
The original failed Fu transfer and later training artifacts are retained to
prevent relabeling training reproduction as validation.

Do not rerun completed evidence campaigns unless a source identity check shows
that the tracked compact artifacts cannot be reproduced or the user explicitly
requests a rerun.

## Focused local checks

For the current unified-gain task, use the focused command in the Codex goal.
At minimum, shell changes should pass `bash -n`, Python changes should pass
`python -m py_compile`, and all changes should pass `git diff --check`. Do not
run the full repository suite by default.
