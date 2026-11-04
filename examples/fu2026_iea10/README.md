# Fu 2026 IEA-10-MW Figure 21 reference

This package freezes the 20 tip-response targets in Fu et al. (2026), Figure
21, for a provenance-bound, no-tuning comparison with a response-aware
VorLap--QBlade model. It contains reference data and validation logic only. It
does not prepare or run QBlade, and it does not modify the existing
Grinderslev example.

## Files and validation

- `targets.csv` contains four configurations at five wind speeds, including
  reduced velocity, edgewise and flapwise amplitudes in normalized and metric
  form, response frequency, digitization uncertainty, and preregistered role.
- `provenance.json` records the publication, source and artifact hashes, public
  IEA-model revision, formulas, recreation boundaries, and debug/holdout split.
- `validate.py` verifies those hashes, all 20 rows, angle mappings, unit and
  nondimensional conversions, Figure 21/Table 4 agreement, and the no-DEL
  boundary.

Run the local validation with:

```bash
python3 examples/fu2026_iea10/validate.py
```

The source PDF is intentionally not copied into the repository. To verify a
download byte-for-byte as well, use:

```bash
python3 examples/fu2026_iea10/validate.py \
  --source-pdf /path/to/bladejfs.pdf
```

Validation fails closed if the source-PDF SHA-256 is absent or changed.

## Primary source and attribution

The primary source is X. Fu, G. Vijayakumar, B. Lee, and M. Sprague,
"Numerical simulation of vortex-induced vibration response of a single IEA
10-MW wind turbine blade," *Journal of Fluids and Structures* 143 (2026)
104537, [doi:10.1016/j.jfluidstructs.2026.104537](https://doi.org/10.1016/j.jfluidstructs.2026.104537).
The article is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The points were recovered from Figure 21 in the
[author-hosted journal PDF](https://xuepengfu.github.io/assets/pdf/bladejfs.pdf),
SHA-256
`304bc4e37021f4dd64e17d0a6e3a4b0ed06eddf232a10878031269b9014485e7`,
retrieved 2026-08-09. Circle centers were read from the PDF vector paths and
calibrated against the vector axis ticks; they were not picked from a raster
image. These are therefore vector-digitized estimates, not author-provided raw
solver values.

The public turbine inputs are bound to
[IEAWindSystems/IEA-10.0-198-RWT commit `350cf06`](https://github.com/IEAWindSystems/IEA-10.0-198-RWT/tree/350cf06f5db9cec12bcd73c2b174a5d9642958d8),
which is Apache-2.0. Hashes for the WindIO YAML and the two BeamDyn files are
recorded in `provenance.json`. Fu et al. report 73 BeamDyn quadrature nodes,
but the paper does not publish their exact spacing or pin a complete ExaWind
case revision.

## Definitions and uncertainty

Fu et al. define the steady tip amplitude as

\[
A_{\max}=\sqrt{2}\,\operatorname{RMS}(x-\bar{x}),
\]

and use \(\bar c=3.72\) m, \(f_{n2}=0.6739\) Hz, and

\[
U_r=\frac{U\cos\gamma}{f_{n2}\bar c}.
\]

The common, conservative figure-digitization uncertainties are
\(\pm0.01\) in \(U_r\), \(\pm0.002\) in either normalized amplitude
(\(\pm0.00744\) m), and \(\pm0.02\) in \(f_y/f_{n2}\)
(\(\pm0.013478\) Hz). These are not CFD uncertainty bars. Fu et al. separately
report approximately 5--10% mesh variation for strongly responding cases.

The P80A310/U18 digitization gives \(A_y=1.1872\) m and \(A_x=0.1971\)
m, consistent with the paper's Table 4 values of 1.19 m and 0.20 m. The
P95A310/U18 values, 1.2250 m and 0.2140 m, likewise agree with Table 4's 1.23
m and 0.21 m within the declared digitization band.

Frequency is marked scoreable only when \(A_y/\bar c\geq0.01\). This is a
preregistered comparison rule, not a threshold claimed by Fu et al.; below it,
the dominant spectral peak is not treated as a robust physical observable.

## Recreation boundary

The intended evidence boundary is:

| Category | Count | Meaning |
|---|---:|---|
| Exact Fu recreation | 0/20 | The current QBlade structure and damping are reduced analogues, and raw Fu CFD force histories plus the cited complete ExaWind case are unavailable. |
| Existing QBlade input match | 1/20 | Existing P100--I50 at 18 m/s maps to P80A310/U18. This matches the documented input condition, not Fu's exact structure or aerodynamics. |
| Documented analogue target | 20/20 | All four angle configurations and five speeds can be represented after explicit parameterization and a coordinate re-audit. |

Fu's convention maps to the DTU/root-frame convention as
\(P_{\mathrm{DTU}}=180^\circ-\alpha\) and
\(I_{\mathrm{DTU}}=360^\circ-\gamma\). Thus P80A310, P90A310, P95A310,
and P95A320 map to P100--I50, P90--I50, P85--I50, and P85--I40,
respectively.

## Preregistered split

`P80A310_U18` is the sole debug case because it matches the current
P100--I50/U18 input and has an independent Table 4 check. It may be used to
audit coordinates, signs, units, triggering, sampling, the steady window, and
the \(\sqrt{2}\)-RMS calculation. It must not be used to tune aerodynamic
coefficients, force amplitude, response frequency, or structural properties.

The other 19 rows are locked holdouts from fitting: P80A310 at 6, 10, 14, and
22 m/s, plus all five speeds for P90A310, P95A310, and P95A320. No response
coefficient, structural property, force scale, or scoring rule may change after
the debug audit if those rows are to retain holdout status. Because the public
targets are included here, "locked holdout" is more accurate than strictly
"blind."

## Claim boundary

This package supports comparison of tip amplitudes, response frequency above
the signal threshold, and VIV/non-VIV response classification. Figure 21 does
not provide time histories or spanwise loads. Consequently, this package does
not support time-history, load-history, rainflow, DEL, or fatigue verification;
those require attributable raw histories from the authors or independently
reproduced high-fidelity simulations.
