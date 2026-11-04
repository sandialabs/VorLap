# Geometry-blended engineering VIV closure

This directory freezes the final semi-empirical training stage. The model uses
stationary shedding frequency and fluctuating-force magnitude, but never uses
the arbitrary phase of a stationary force record. It blends two signed
cycle-work mechanisms using the declared body-fixed breadth/depth ratio:

- a reciprocal Griffin branch for separated rectangular afterbodies; and
- a velocity-correlated negative-damping branch with signed amplitude
  saturation for thin airfoil/plate sections.

The training package uses the Mannini 3:2 response curve, the three Marra 4:1
peaks, and all five Ehrmann NACA0018 damping points plus its response-peak
location. The largest fitted peak errors are 27.15% for the rectangular branch
and 14.64% for the airfoil branch. The earlier rectangular-only Ehrmann failure
is preserved separately under
`examples/ehrmann2014_naca0018/evidence/griffin_v1_ehrmann/`.

The Liu 5:1 response is excluded from fitting. Its protected assessment is in
`examples/liu2024_rectangular_5x1/evidence/geometry_blended_validation/` and
gives 5.0% peak-amplitude error and 3.6% onset error.

Regenerate and check the artifacts with:

```bash
python -m examples.viv_geometry_blended.calibrate
python -m examples.liu2024_rectangular_5x1.assess_geometry_blended_model
python -m pytest -q tests/test_qblade_geometry_blended_lockin.py
```

This is experimental validation of one protected sectional-response case. It
does not validate a turbine response or fatigue load. H-VAWT use requires a
matched native/hybrid pilot, and fatigue use additionally requires a
conservative discrepancy factor applied to the coupled load histories.
