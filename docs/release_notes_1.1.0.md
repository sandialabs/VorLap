# VorLap 1.1.0

VorLap 1.1.0 includes the solver, GUI, QBlade SIL bridge, documentation,
tests and examples accompanying the engineering VIV paper. It is prepared as
one release commit on the existing GitHub history. The default branch is
`master`.

## Public H-VAWT example

The 10 m H-VAWT example uses independently generated blade and torque-tube
sectional definitions. The blade matches the published 2024 Torque OWENS
bending stiffness targets; other properties follow documented generic
material and geometry assumptions. The example includes its generator,
provenance, actual NACA0018 source database and a measured Docker comparison.

The exact paper case is available from the authors on request, subject to
approval. Please feel free to contact the authors to discuss access.

The short Docker check completed three 270-step spectral SIL runs: original
and public at 0.0075 s (2.025 s), and public at 0.00375 s (1.0125 s). Each
completed with finite exports, 270 bridge updates, no source clipping or
nonfinite values, and zero exported RPM error. Peak supplemental force changed
by 0.185%; several sectional moments changed substantially. The half-step
spot check's largest normalized moment-channel RMS difference was 2.04% over
the common first-second interval.

These checks establish finite damped startup operation over the tested
intervals. They do not establish equivalent paper loads, long-duration
stability or fatigue life. The public example README and comparison report
describe the structural station changes and retain the original results'
model boundary.

## Airfoil and CFD setup

The NACA0018/Kynema example includes raw static-airfoil force histories,
Pointwise mesh setup, Kynema/Nalu templates, scaling calculations and source
identities. Preparation defaults to printing a submission command and refuses
existing case output. The supplied runtime HDF5 is the actual paper source;
its replicated Reynolds planes extend lookup coverage without adding new CFD
evidence. The example README distinguishes historical source data from the
current portable preparation scripts.

## Verification

The software suite covers physical and numerical invariants, source data,
bridge preparation/runtime interfaces, GUI configuration and execution,
outputs, campaign provenance, examples and packaging. External QBlade,
historical campaign and companion-paper integrations are explicit optional
checks. Exact numerical replay uses the documented pinned environment.

Run the checks from an installed Git checkout:

```bash
python -m pytest
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

The release includes separate third-party notices and excludes downloaded
QBlade runtimes, local environments, campaign output and author paper inputs.
