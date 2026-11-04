# Model-based negative damping investigation

Date: 2026-09-09. Working checkout: `VorLap`, branch
`response-aware-viv-fatigue`, initial commit `a7db335`.

**Finding:** a model-based negative-damping framework is feasible. Both the
aerodynamic gain and the saturation law need motion-dependent aerodynamic
information. Structural analysis supplies the modal dynamics and the
projection of sectional saturation. The implemented prototype demonstrates
that route for published CFD work, but present evidence does not establish
generic accuracy equal to experimentally trained VorLap.

## Question and working hypothesis

Can structural and aerodynamic models replace experimental response fitting
of the current `H` and `Lambda`, and retain comparable accuracy?

The current law is `Q=c_a0[1-(A/(Lambda L_ref))^2] qdot_f`, with
`c_a0=H B_spectral`. `Lambda L_ref` is an aerodynamic zero-work amplitude.
A linear structural model supplies modes, mass, frequency, and assumed
damping, but no finite saturation amplitude. The mode shape can project a
sectional aerodynamic saturation law into a modal saturation law. A
prescribed-motion aerodynamic model can identify both coefficients, or supply
the full signed work curve if the quadratic approximation is inadequate.

## Scope and acceptance criteria

- [x] Map the present gain, signed-power, work-map, and structural interfaces.
- [x] Verify literature and accessible data; distinguish fixed-body force
  spectra from synchronous motion-correlated work.
- [x] Implement a separate model-based coefficient and modal-projection API.
- [x] Demonstrate inference from CFD power and structural data without using
  free-response amplitudes in coefficient selection.
- [x] Check normalization, dimensions, sign, native-base subtraction,
  analytical limits, and cycle-averaged work through the existing actuator.
- [x] Quantify the quadratic reduction error and sensitivity to damping/data.
- [x] Run focused new and existing regressions; inspect generated figures.
- [x] Document feasible scope, experimental comparison limits, and a concrete
  data/model-development path for generic VorLap use.

The experiment is offline and does not replace the frozen production gain
model. No protected Horcas `COMB` target enters fitting. Generated outputs
live in the ignored `run_headless/runs/model_based_negative_damping/` tree.

Initial unrelated changes are preserved: modified
`tests/test_hvawt_paper_reproducibility.py`; untracked
`examples/benner2019_naca0021/`, `qblade_qpr_export_probe.py`, and
`scripts/NACA0018_legacy/`. The parent workspace is not a Git repository.

## Evidence distinctions

1. Analytic tests establish code/coordinate correctness.
2. Reducing published forced-motion CFD power is model-based coefficient
   identification, not a fresh CFD solution or experimental validation.
3. Comparing its structural equilibrium with published FSI is a numerical
   comparison. Published FSI targets are already known, so it is not a newly
   blinded holdout. The paper also imposed an FSI-derived mean deflection in
   its forced-motion calculation.
4. A matched comparison with experiment-trained VorLap requires independently
   generated aero data and a frozen comparison on the same withheld physical
   cases. Such paired NACA0018 force histories are absent locally.

## What the structure can determine

For the same power-conjugate modal coordinate `u=phi*q`, the structural model
gives

```text
M_m = phiᵀ M phi       K_m = phiᵀ K phi       C_m = phiᵀ C phi
omega_m = sqrt(K_m/M_m)                       zeta_m = C_m/(2 M_m omega_m)
P_struct(A) = 0.5 C_m omega_m² A².
```

The damping matrix or a damping law must be supplied: mass and stiffness
alone do not identify material, joint, or other dissipative losses. A
nonlinear conservative stiffness can shift frequency and change the motion
shape; its work over a closed cycle remains zero. Amplitude-dependent
structural damping could help limit response, but then aerodynamic and
structural saturation must remain separate terms in the energy balance.

There **is** a useful structural contribution to effective Lambda. If a
sectional aero model supplies `F_i=(b0_i-b2_i*a_i²)*v_i`, where `a_i` is the
local peak envelope, then

```text
c0 = sum(b0_i phi_i²)       c2 = sum(b2_i phi_i⁴)
A_infinity = sqrt(c0/c2)    A_equilibrium = sqrt((c0-C_m)/c2), when c0>C_m.
```

Span quadrature is already included in `b0_i,b2_i`. Thus mode shapes predict
the projection of an aerodynamic saturation law. For a uniform local law,
the illustrative shape `phi=x²` produces a modal/local saturation ratio of
`sqrt(9/5)=1.341641`; the implemented quadrature reproduces this result. This
example isolates modal weighting and is not a fitted cantilever eigenmode.

The most portable quantity is dimensional `A_infinity`, with an explicit
coordinate. `Lambda=A_infinity/L_ref` depends on the chosen reference length.
The present turbine branch actually stores `Lambda=1` and a dimensional
limiting amplitude as `L_ref`. That convention is not a structural prediction.

## What the aerodynamics must determine

Prescribe `q=A sin(omega t)` and calculate synchronized aerodynamic work:

```text
P_a(A,omega) = mean(Q_a*qdot)
c_a(A,omega) = 2 P_a / (omega² A²)
```

If `c_a=c0-c2*A²` adequately represents the data, recover
`H=c0/B_spectral` and `Lambda=A_infinity/L_ref`. If it does not, retain the
signed work map. An equivalent sectional representation uses
`C_P=p2*(a/c)²-p4*(a/c)⁴` and `k=omega*c/(2U)`:

```text
b0 = rho U c span p2/(4 k²)
b2 = rho U span p4/(4 k² c).
```

For a uniform section, `H=p2/(2 k² C_captured)` and
`lambda_section=sqrt(p2/p4)`. These equalities use this repository's captured
harmonic coefficient convention. They do not make `H` a universal Scanlan
derivative or equate it with another model's `H1`.

A stationary spectrum supplies force magnitude and shedding frequency, but
not the phase relative to motion. Identical force spectra can produce
positive, zero, or negative work. Therefore a static spectrum cannot identify
the missing coefficients by itself. Further, `H` is undefined when its
spectral basis is zero while prescribed motion still produces finite work;
a direct signed-work representation avoids that artificial restriction.

The [literature review](literature.md) documents prescribed-motion energy
maps, wake oscillators, airfoil CFD, publicly available models, and their
transfer limits. The strongest route for the present deep-stall VIV problem
is motion-conditioned CFD followed by a compact surrogate. Existing
wake-oscillator constants remain empirical and require demonstrated transfer.

Cheaper aerodynamic models have narrower uses. Static-polar quasi-steady
models can estimate galloping damping, but experiments show that VIV and
galloping interaction can invalidate that prediction near their overlapping
regimes. [Mannini et al. (2016)](https://doi.org/10.1016/j.jfluidstructs.2016.08.002)
Attached-flow unsteady theory supplies aerodynamic phase within its
potential-flow assumptions; it does not resolve the separated vortex wake
needed here. That last limitation follows from the formulation, rather than
an accuracy comparison made in this investigation.
[Theodorsen, NACA Report 496](https://ntrs.nasa.gov/citations/19930090935)

## What was implemented

The new [model module](https://github.com/sandialabs/VorLap/blob/release/v1.1.0/vorlap/model_based_negative_damping.py) provides:

- Real-mode projection from `M,K,C`, with symmetry, passivity, and eigenmode
  residual checks.
- Conversion of sectional nondimensional work into dimensional signed
  coefficients and integration with quadratic/fourth modal moments.
- Explicit evaluation of the existing spectral basis, allowing consistent
  conversion of modeled `c0` into `H` where that basis exists.
- Equal absolute-power least-squares identification of the quadratic law,
  reporting residuals and rejecting non-saturating or insufficient data.
- Bounded equilibrium evaluation and a signed `ExternalModalPowerTarget`
  adapter for the existing VorLap modal actuator.

The adapter explicitly converts the original modal coordinate to the
actuator's unit-maximum translational mode coordinate. It requires a declared
work role: either total aerodynamics is replaced, or the input is already
high-fidelity-minus-exact-runtime-base work. It cannot verify the truth of
that declaration; production qualification also needs the existing
source-bound paired-history importer. No full CFD power is automatically
added on top of native QBlade aerodynamics.

This API represents a positive-gain, saturating quadratic branch. It rejects
`c0<=0` or `c2<=0`; it is not a complete stable/subcritical/hysteretic work-map
solver. It also assumes fixed motion frequency, an isolated real mode, and
compatible supplied sectional data. Displacement-phase forces, added mass,
frequency shifts, torsion, and spanwise coupling require additional physics.

The [driver](https://github.com/sandialabs/VorLap/blob/release/v1.1.0/examples/model_based_negative_damping/investigate.py)
reuses `TabulatedAerodynamicPowerCurve` for the full-power comparison. That
existing table root finder resolves this single descending branch; it is not
an exhaustive multiple-root solver for arbitrary maps. The prototype does
not wire a new default into the H-VAWT or turbine runtime.

## Reproduced numerical results

The benchmark uses 15 existing digitized points from the Grinderslev
P100-I50 forced-motion CFD curve, along with the independently archived
QBlade structural-mode identification. The driver reprojects the nodal mass
weights to `M=1679.206980 kg`, uses the structural probe frequency
`f=0.66985 Hz`, and derives damping from the archived stiffness-proportional
coefficient `beta=0.00171697302786 s`. This gives `C_m=51.071965 kg/s` and
`P_struct=452.342536 A² W`. These are reduced structural-model properties,
not newly exported full Chrono matrices or measured modal damping.

The source CFD prescribed `0.67 Hz`; as in the existing benchmark, this
calculation treats the 0.0224% frequency difference as negligible. It does
not establish frequency transfer. The paper's alternative `540 A² W`
damping relation was fitted to FSI dissipation and is excluded from our
structural-model prediction.

| Aero representation with the same converted structure | Damped amplitude | Difference from published 1.18 m FSI | Undamped amplitude |
|---|---:|---:|---:|
| Full CFD work table | 1.272023 m | +7.80% | 1.570000 m |
| Quadratic CFD work reduction | 1.361640 m | +15.39% | 1.544742 m |

The published undamped FSI amplitude is approximately 1.63 m. The source
paper's own forced-motion/damping comparison uses different structural
damping and gives approximately 1.24 m. These are numerical comparisons;
the forced-motion setup also uses a mean deflection informed by FSI.
[Grinderslev et al. (2022)](https://wes.copernicus.org/articles/7/2201/2022/)

The quadratic fit gives `c0=229.007447 kg/s`,
`c2=95.970517 kg/(s m²)`, and `A_infinity=1.544742 m`. Its **251.36 W RMSE**
and **37.06% relative L2 power error** show that the curve is not well
represented by two constants. Leaving each interior amplitude out in turn
gives power RMSEs of 96.42 W for table interpolation and 293.30 W for the
quadratic fit. These are reduction/interpolation diagnostics on the same
CFD curve, not independent physical validation.

The global quadratic fit also predicts a smaller origin gain than the
table's quadratic first interval. Consequently, its averaged response grows
more slowly, despite having a larger equilibrium. The table origin itself
is a continuation assumption from the first 0.1 m sample; neither growth
history constitutes an independently verified onset prediction.

Deterministic shifts of every nonzero digitized amplitude by ±0.01 m and
power by ±20 W give table equilibria from 1.26036 to 1.28366 m. These are
four sensitivity cases, not statistical uncertainty bounds. Changing the
assumed structural damping by ±20% gives 1.30220 and 1.24474 m. Neither
sensitivity includes CFD turbulence/model error or missing 3D physics.

The averaged energy equation was integrated from a declared 0.01 m seed for
600 s. Both representations settled to their calculated equilibria within
`3.5e-14 m`; tightening solver tolerances changed the amplitude histories by
at most `3.65e-6 m`. This verifies the reduced energy calculation. It is not
a fresh CFD/FSI run or a validation of transient wake dynamics.

Generated `run_headless/runs/model_based_negative_damping/results.md`,
`run_headless/runs/model_based_negative_damping/results.json`,
and `run_headless/runs/model_based_negative_damping/work_and_response.pdf`
are retained locally in the ignored output tree. The figure compares power,
damping sensitivity, and averaged response growth. Its final PNG was
visually inspected by execution and review roles; the PDF passed
`qpdf --check`.

## Can this match an experimentally trained model?

**Possibly within a qualified domain, but the present comparison cannot
answer that experimentally.** A response-trained model has learned errors
and unmodeled effects from that particular experiment, including structural
damping, blockage, turbulence, supports, and measurement definitions. A
model-only prediction must supply those effects independently or quantify
their uncertainty. Replacing an experimental fit with a CFD fit does not
remove model error.

Current Ehrmann training uses two outer response markers; the three interior
same-profile markers have reported errors of 8.8%, 10.2%, and 14.6%. Those
figures are a useful future comparator, but comparing them numerically with
the 7.80% CFD-blade result would mix different geometries, data sources, and
validation tasks. No claim of equal or better experimental accuracy follows.
See the [current Ehrmann record](https://github.com/sandialabs/VorLap/blob/release/v1.1.0/examples/ehrmann2014_naca0018/README.md).

## Concrete path to a generic VorLap model

1. **Complete the existing NACA0018 prescribed-motion pilot.** The
   [frozen request](https://github.com/sandialabs/VorLap/blob/release/v1.1.0/examples/ehrmann2014_naca0018/work_map_data_request.json)
   specifies incidence −90°, `Re={64000,100000,140000}`,
   `k={0.45,0.65,0.8,1.05}`, and four nonzero `A/c` values
   `{0.005,0.0125,0.025,0.05}`. It needs 48 high-fidelity and 48 identical-motion
   native-base histories, plus 12 analytical origin vertices. The required
   data are absent locally. The request's walltime is a planning target,
   not a measured cost estimate from this investigation.
2. **Verify aerodynamic work before reducing it.** Check mesh, time step,
   span/coherence, retained cycles, mean deformation, force/motion signs,
   and base subtraction. Resolve both positive and negative work and its
   amplitude slope. Do not assume URANS predicts the correct sign because
   it converges; published blade comparisons show that it may not.
   [Grinderslev et al. (2023)](https://wes.copernicus.org/articles/8/1625/2023/)
3. **Select the smallest adequate representation on aerodynamic evidence.**
   Use `H,Lambda` only if their inferred curve reproduces work, onset, roots,
   and slopes within a tolerance chosen for the intended response accuracy.
   Otherwise retain a signed `C_P(Re,k,A/c,incidence,direction,geometry)` map.
   No numeric fit-quality gate was optimized against experimental response
   in this investigation. Existing work-map code stores signed work but its
   older actuator route applies only positive lower-confidence injection;
   a general signed map adapter needs separate implementation and review.
4. **Freeze an experimental comparison before making predictions.** Use the
   same structural model, native aerodynamic baseline, measured environment,
   and amplitude definition for model-derived and response-trained variants.
   Compare peak amplitude, onset/cessation, frequency, and the entire
   response envelope at the same cases. Retain the existing 30% amplitude
   plus 0.01 reference-length screen where applicable, and report paired
   error differences; passing that screen alone is not equality with the
   trained model. Already used Ehrmann points are historical comparisons,
   so reserve new conditions or another eligible dataset for independent
   validation. Do not refit after inspecting those responses.
5. **Extend geometry and operating regimes after that comparison.** Add
   incidence, Reynolds, profile shape, thickness, roughness/turbulence and
   motion direction. For blades, demonstrate whether sectional strips can
   represent taper, twist, tip geometry, inclined spanwise flow and coherence;
   otherwise use 3D work maps or an identified coupled wake-state model.
   Store geometry/domain/provenance with the surrogate, and reject unsupported
   extrapolation. Significant frequency changes require displacement-phase
   aerodynamic forces and a coupled frequency/amplitude solve.

The literature supplies modeling equations and reproducible CFD starting
points, but this search found no downloadable target-matched force-motion
atlas that makes the negative-damping model immediately generic. Developing
that aerodynamic dataset and validating its transfer is the remaining
physical work; the current software experiment establishes the reduction
and integration path.

## Reproduction and verification

From the `VorLap` repository root:

```bash
.venv/bin/python examples/model_based_negative_damping/investigate.py
.venv/bin/python -m pytest -q tests/test_model_based_negative_damping.py
.venv/bin/python -m pytest -q tests/test_qblade_engineering_lockin.py tests/test_unified_viv_gain.py tests/test_qblade_modal_power_model.py tests/test_qblade_airfoil_work_map.py tests/test_airfoil_work_map_identification.py tests/test_qblade_single_mode_work_map_sil.py tests/test_qblade_single_mode_engineering_lockin_sil.py
qpdf --check run_headless/runs/model_based_negative_damping/work_and_response.pdf
git diff --check
```

**101 unique focused tests passed:** 25 new physical/API checks and 76
existing affected gain, work-map, identification, actuator, and SIL checks.
The independent reviewer also ran the new tests with the existing modal-power
and identification tests (52 passing, overlapping the above total), replayed
the driver into a separate temporary directory, and inspected its figure.
Physical tests cover mass/stiffness energy, modal rescaling and physical
Lambda invariance, signed nodal work, nondimensionalization, bounded support,
the real existing actuator, and the analytic averaged-envelope solution.
No test-count total treats repeated tests as new evidence.

Investigation complete. Generic experimental parity remains unestablished;
that is a finding of this task, not a passing validation claim.
