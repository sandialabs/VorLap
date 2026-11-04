# Horcas 2020 IEA-10-MW tip-geometry validation

This package freezes the high-fidelity turbine response comparison from
Horcas et al. (2020), Figure 19.  The source study couples EllipSys3D and
HAWC2 for five IEA-10-MW blade-tip geometries at 11 inflow inclinations and
18 m/s.

The four primitive geometries (`BASE`, `EXT`, `SWEEP`, and `DIHE`) are
training evidence.  The combined sweep-plus-dihedral geometry (`COMB`) is a
complete geometry holdout: all 11 response points remain excluded until the
closure, coefficients, and acceptance protocol are frozen.  This provides a
multi-condition wind-turbine HFM validation, rather than relying on one
cross-profile peak.

The digitized amplitude is reported both as the paper's peak-to-peak value and
as peak harmonic amplitude (`0.5*peak-to-peak`) used by VorLap.  Figure 19 is
vector data; the CSV is not author-provided raw output.  The paper also notes
that some 55-degree simulations were still increasing at the end of the
45.3-second run.

This package cannot validate fatigue or certification loads because the paper
does not publish response/load histories suitable for rainflow counting.

## Frozen result

The work-equivalent negative-damping model was frozen in commit `87f23dd`
before the `COMB` targets were evaluated.  The complete 11-condition holdout
passes every preregistered gate:

- curve RMSE is 0.1088 m, or 12.9% of the HFM peak;
- peak amplitude is 0.8180 m versus 0.8419 m (2.84% error);
- both curves peak at 55 degrees;
- the half-peak resolved intervals are 52.5--57.5 degrees (model) and
  55--57.5 degrees (HFM), a one-sample onset difference; and
- maximum frequency-ratio error in the HFM resolved interval is 3.92%.

All 11 points remain in `evidence/combined_tip_v1/assessment.json` and
`validation.pdf`; the aggregate result is not a selected-peak-only claim.
The post-validation conservative screen uses
`A_design = 1.30*A_model + delta_A`, with the fitted additive discrepancy
recorded in the assessment.  That envelope covers this holdout but is not a
fatigue validation.

Validate the frozen package with:

```bash
python -m examples.horcas2020_iea10_tip_geometry.validate
python -m examples.horcas2020_iea10_tip_geometry.fit_geometry_composition
python -m examples.horcas2020_iea10_tip_geometry.assess_geometry_composition
```
