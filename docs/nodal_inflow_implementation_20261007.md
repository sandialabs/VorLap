# Instantaneous nodal inflow implementation

Status: model/runtime inputs and explicit native SIL input verified; manuscript note added and rebuilt. Automatic sampling still awaits the user's physical input choice. `use_qblade_local_inflow` requires explicit nodal samples at present; the stock runner raises rather than silently using mean wind if they are absent.

## Implemented

- Optional finite nodal XYZ velocity arrays reach the generic runtime and the H-VAWT and single-mode SIL adapters. Generic runtime permutes wind and structural velocity together.
- `use_qblade_local_inflow` is a strict JSON boolean, absent/false by default. Case preparation exposes `--use-qblade-local-inflow` and generates current global-position channels in matching node order.
- Checkpoint restore rejects switching this input mode. Response adapters reject local-inflow configurations with a different controller/modal node order.
- `SilVorLapCoupler.step(..., ambient_velocity_global_mps=wind)` accepts explicit finite nodal XYZ vectors and forwards them to the selected runtime/response adapter before applying the existing force lag.

## Physical input decision

The installed QBlade 2.0.9.7 Linux library's `GetWindspeedArray` calls `wakeInductionOpenMP`, `addBladeInductionVelocities`, and `getFreeStream`, then sums their XYZ vectors. It provides total local flow, including induction. Its C SIL interface does not expose an arbitrary-position freestream-only vector. Scalar blade inflow channels cannot supply XYZ nodal vectors.

The pending user question asks whether the flag should use turbulent ambient flow only or full QBlade local flow. The former needs a TurbSim-field sampler with QBlade-compatible spatial/time conventions; the latter can use `getWindspeedArray`. Sampling must use current global nodal positions before QBlade advances. Neither interpretation has been silently substituted for the requested input.

## Verification

Command, from the VorLap repository:

```sh
/Users/kevmoor/miniconda3/bin/python -m pytest tests/test_qblade_runtime.py tests/test_qblade_case_prep.py tests/test_qblade_sil.py tests/test_qblade_hvawt_sil.py tests/test_qblade_single_mode_work_map_sil.py tests/test_qblade_single_mode_engineering_lockin_sil.py -q
git diff --check
```

Result: 192 passed, 67 existing airfoil-repair warnings, 67.30 s; diff check clean. Tests verify independently calculated force magnitude/direction with distinct XYZ wind and reversed node order, uniform-flow equivalence, invalid-input rejection, generated position tables, restart mode identity, H-VAWT spectral speed/AoA/Re, changed lock-in feedback, and modal/nodal work closure.

## Native Docker verification and paper update

Started the existing Colima profile at the user's request and used the existing `vorlap-qblade-headless:amd64` image (`fc81d0c506c2`), with repository input mounted read-only and networking disabled. `run_headless/verify_nodal_inflow_smoke.py` stages the archived H-VAWT case, generates a new 16-s 5x5 TurbSim field, and runs four native steps at all 39 nodes. Synthetic constant coefficients CL=1 and CD=0.5 give an independent dimensional-force/direction check. Two cases passed: zero native polar coefficients and the original native aerodynamic polars. No production case files or libraries were altered.

The initial scalar/array equality check correctly revealed different API contents: the scalar getter includes freestream plus wake induction, while the CPU array getter additionally includes bound-blade induction. Disabling structural aerodynamic loads alone leaves circulation active; zero native polar coefficients give the required zero-induction limit. Scalar and array samples then agree exactly. With original polars, their largest difference is 0.11043 m/s; equality is not asserted between these different quantities. Reversing every coordinate array and undoing the output permutation gives identical array samples in both cases.

Native samples were passed explicitly through the opt-in SIL input. The maximum analytic-force errors were 7.11e-15 N (zero induction) and 8.88e-15 N (native aerodynamics). Nodes moved about 0.00174 m. The physical input times were 0, 0, 0.0075, and 0.015 s, consistent with QBlade's initial held timestep. This verifies the API/input/force path, not response calibration or fatigue convergence. The automatic sampling backend remains pending the ambient-only versus induction-inclusive choice.

Raw receipts and logs are in ignored `run_headless/runs/nodal_inflow_verification_20261007/`; the compact retained receipt is `docs/nodal_inflow_native_verification_20261007.json`. The reusable native probe runs inside the image as:

```sh
python3 run_headless/verify_nodal_inflow_smoke.py --output /evidence/zero_induction
python3 run_headless/verify_nodal_inflow_smoke.py --output /evidence/native_aero --native-aerodynamics
```

Section 2 now states that the reported cases use mean spectral inflow and instantaneous nodal inflow is an optional modeling choice. All manuscript text remains in the main TeX source. An isolated `latexmk -norc` build passed at 19 pages, `qpdf --check` passed, and rendered page 7 was inspected. The initial in-place build encountered inconsistent auxiliary/BibTeX state, which the isolated build resolved. After adding the explicit SIL input, the 137 affected runtime/preparation/SIL tests passed.

User edits in `tests/test_hvawt_paper_reproducibility.py` and the nested manuscript were preserved during implementation and verification.

## Pre-push verification

The affected runtime, case-preparation, SIL adapter, and manuscript reproducibility suite passed: 210 tests, 67 existing legacy-airfoil repair warnings. Review then found that explicit nodal inputs could bypass the SIL flag's node-order and checkpoint-mode checks when the flag was false. The SIL step now rejects that combination before reading QBlade inputs or changing force/state data; the regression test verifies the unchanged swap, counters, and external actions. CLI help states that explicitly supplied samples are required and automatic sampling is not configured.

After that correction, `python -m pytest tests/test_qblade_sil.py tests/test_qblade_hvawt_sil.py tests/test_qblade_case_prep.py -q` passed all 101 tests, with the same 67 legacy-airfoil repair warnings. `git diff --check` is clean. This correction changes invalid-input handling, not the native force path verified above.
