# Short Docker comparison — 3 November 2026

The original and public section models both completed **270 spectral SIL steps
at 0.0075 s**, integrating 2.025 s. The public half-step spot check completed
270 steps at 0.00375 s, integrating 1.0125 s. All runs had finite load exports,
270 nonzero-force bridge updates, no source nonfinite values or clipping, and
zero exported RPM error. This establishes finite startup operation over these
intervals.

Settings: 11.25 m/s wind, prescribed 1 rpm, initial azimuth 40°, six threads,
10 s structural ramp and 5 s additional startup damping. Startup damping is
active throughout; this does **not** establish undamped/long-duration stability,
fatigue equivalence or converged structural response. The bridge uses spectral
forcing, without the original turbine's fitted modal response basis.

Peak applied bridge force changed from **17.86973 N to 17.90284 N (+0.185%)**. Sectional moments changed much more, so the public example is geometrically similar and functionally runnable; it is **not a quantitatively equivalent structural model**.

Only the blade and torque-tube physical input files were replaced. Their independent section stations also change the structural/coupling grid (274 to 171 converted physical elements). This paired comparison does not separate each component's contribution.

The following are peaks of absolute QBlade local moment channels, in N m. The RMS difference is normalized by the original channel RMS over exported times 0–2.0175 s; it is not a fatigue metric.

| Blade fraction | Local moment | Original peak | Public peak | RMS difference / original RMS |
| --- | --- | ---: | ---: | ---: |
| 0.230 | X_l | 15.228 | 9.312 | 39.32% |
| 0.230 | Y_l | 148.379 | 161.743 | 9.25% |
| 0.230 | Z_l | 2.370 | 9.907 | 528.92% |
| 0.500 | X_l | 12.268 | 18.936 | 262.99% |
| 0.500 | Y_l | 9.095 | 4.383 | 145.34% |
| 0.500 | Z_l | 310.599 | 10.735 | 96.69% |
| 0.770 | X_l | 9.411 | 8.930 | 219.19% |
| 0.770 | Y_l | 150.698 | 163.572 | 9.28% |
| 0.770 | Z_l | 2.537 | 9.908 | 497.77% |
| 0.940 | X_l | 0.930 | 0.842 | 194.34% |
| 0.940 | Y_l | 11.994 | 15.905 | 32.48% |
| 0.940 | Z_l | 2.508 | 9.908 | 495.83% |

The half-step comparison interpolates the finer export onto the 135 common coarse timestamps, 0–1.005 s. Its largest normalized RMS difference across the 12 moment channels is **2.04%**. This is a short integration-sensitivity spot check, not a convergence study.

`comparison.json` supplies all 24 load-channel differences, reference scales, run/file identities and matching core-source hashes. Exact cases, full histories and summaries remain privately with the paper. Docker image SHA-256: `77c308c6cfa0a0c604c83962d9d1f72487dc253c48d60f20a87bd6ee9e2db9a4` (amd64 under arm64 host translation). The three runs took about 12.7 minutes each on this machine; no native-performance claim is made.

The 2024 reference supplies blade EI targets; this check does not rerun its historical OWENS fatigue analysis or replace either paper's original results. The exact paper case is available from the authors on request, subject to approval.
