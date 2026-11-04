# Belloli-informed Scruton transfer engineering screen

These deterministic artifacts document the frozen
`shift_1__h1_none__asat_1` analytical transfer and completed eight-case
coupled QBlade screen:

- `analytical_gate.json` records the dense-grid gate and fingerprints.
- `analytical_comparison.csv` is the source table for all plotted comparisons.
- `engineering_sc_transfer_table.tex` is the compact manuscript-ready table.
- `engineering_sc_transfer_comparison.pdf` compares the original schema-v2 and
  transferred curves with the digitized hGAST and Lupi peak targets.
- `engineering_sc_transfer_assessment.json`, `response_by_case.csv`, and
  `runtime_case_audits.json` contain the coupled assessment and compact runtime
  evidence.
- `engineering_sc_transfer_qblade_screen.pdf` compares the nominal and
  half-timestep QBlade responses with the declared experimental target bands.
- `SHA256SUMS` binds the compact evidence package.

The `Sc=2.94` and `Sc=7.02` targets were available when the form was selected
from a predeclared 20-form menu. The figure and table therefore label them as
transfer tuning and engineering screening, **not independent validation**.
Acceptance is `|error| <= 0.30 A_target + 0.01D` with peak-location error
`<= 0.25` in reduced velocity for the analytical screen. In coupled QBlade,
`Sc=2.94` produces `0.42552D` versus the `0.350D` target (21.6% error) and
passes. `Sc=7.02` produces `0.08933D` versus `0.150D` (40.4% error) and misses
the frozen tolerance by `0.00567D`. All runtime, stationarity, frequency, and
timestep gates pass, but the conjunctive two-Scruton amplitude screen fails.
The negative high-Scruton result is reported without retuning.
