# Schema-v2 Sc=5.18 calibration evidence

This directory contains the compact analysis products from the completed
24-case, 12-point paired-timestep QBlade calibration.  The campaign was
generated, executed, and analyzed with the pinned Linux/amd64 image
`vorlap-qblade-headless@sha256:99864db6be24387033888d0fe4b8859879f2987b2e4042df08c501e71a71cc61`.
The campaign-manifest SHA-256 is
`b6b32cce4e20b2458c2f0fd42580966dd54f77322817a614f066f98bcc8296f8`,
the stationary-protocol SHA-256 is
`00fdb882a16c06881db2dd3f54c4985f5c1555b54c3425bceed69d4664f360dc`,
and the portable coefficient fingerprint is
`330242617d9d04d2cb771767fefbedac6414bea9a04a9b6ea9e813c0aba18c7d`.

All 24 cases and all strict calibration-stage checks pass.  The QBlade curve
peaks at `A/D=0.233082` and `U*=6.25`, compared with the sampled digitized
Lupi peak `A/D=0.235286` and `U*=6.350126`.  The 12 timestep pairs have maximum
amplitude and frequency differences `0.0011049D` and `0.000128899 f_n`.
The 16-point hGAST curve comparison passes with RMSE `0.0204821D` and maximum
error `0.0411417D`.  The denser Lupi experimental-curve RMSE is `0.0728871D`,
above its `0.05D` screen; this is therefore lock-in-peak engineering adequacy,
not complete experimental-curve validation.

Recreate the compact analysis from the immutable campaign and raw result root
with:

```bash
python examples/vlastos2026_cantilever/response_benchmark.py analyze \
  --campaign run_headless/runs/vlastos2026_response_stationary_v2_linux/campaign/campaign_manifest.json \
  --results /results \
  --output schema_v2_calibration_evidence
```

`SHA256SUMS` binds the deposited bytes.  The raw QBlade histories are retained
separately because they are much larger than this compact paper evidence.
