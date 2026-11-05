# Liu 5:1 rectangular-cylinder untouched validation

This directory freezes the independent experimental response target before the
final geometry-blended closure is fitted. The response target is the bare `B/D=5`
reference tested by Liu et al.; splitter cases are used only to reconstruct the
bare peak from ratios printed in the article. The robust target is
`h_rms/D=0.0210 +/- 0.0015`.

The static shedding input is independent: Nguyen et al.'s fixed-body BARC data.
Their coefficients and Strouhal numbers use the streamwise width `B`; the CSV
also records the exact conversion to the cross-flow depth `D` used by the VIV
equation. No Liu response value is used to fit the closure. The final training
set is the Mannini 3:2 response curve, the three Marra 4:1 peaks, and the
Ehrmann NACA0018 peak-amplitude and peak-location data.

The frozen geometry-blended closure predicts `h_rms/D=0.0220059` versus the
experimental bare peak `0.0209581`, a 5.0% error without retuning. Its predicted
onset reduced velocity is `8.05` versus the reported `8.35` (3.6%). The
nonlinear SDOF implementation is stationary, matches its analytic work balance,
is frequency locked, and is timestep-repeatable. This passes the predeclared
independent engineering response gate and permits H-VAWT pilot preparation. It
does not validate H-VAWT loads or fatigue. The authoritative result is
`evidence/geometry_blended_validation/validation.json`; older Griffin evidence
is retained only as a superseded model rung.
