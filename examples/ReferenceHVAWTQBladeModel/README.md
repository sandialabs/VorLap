# Public 10 m H-VAWT bridge example

This is a geometrically similar bridge example of the turbine studied in the
SIL paper. The exact structural input files used in the paper could not be
shared publicly. The physical changes introduced for this public replacement
are isolated to the blade and torque-tube sectional definitions, including
their lumped masses, inertia/offset assumptions and sectional damping.
Geometry, aerodynamic inputs, struts and the bridge interface are retained.
Independent section stations also change structural discretization. Sectional
moments can differ substantially even when applied bridge forces are close.
The exact paper case is available from the authors on request, subject to
approval. Please feel free to contact the authors to discuss access.

The replacement blade uses the **OWENS** reference values published by
Moore, Brownstein and Ross (2024),
[doi:10.1088/1742-6596/2767/7/072025](https://doi.org/10.1088/1742-6596/2767/7/072025):
0.5 m NACA0018 chord, flap EI = 1.21e5 N m² and lag EI = 2.87e6 N m².
An independent generic glass-fiber shell/cap model supplies the remaining
section properties. The torque tube uses explicit nominal steel annuli.
These are equivalent beam assumptions, not manufacturing data or a qualified
structural design.

`generate_public_sections.py` reads no original supplier inputs.
`section_provenance.json` records the public targets, assumed material and
geometry values, quadrature and generated file hashes. Recreate or check:

```bash
python examples/ReferenceHVAWTQBladeModel/generate_public_sections.py --check
```

Use `baseline_10m_HVAWT_reference_turbine_Wwnd.sim` with the normal VorLap
QBlade preparation/run workflow. Install QBlade separately as described in
the main README. The default bridge configuration uses the spectral force
path. Historical HVAWT modal/response contracts describe the original paper
model; do not apply that basis to this replacement as a validated modal model.

The matched Docker check and its limits are recorded in `comparison_summary.md`.
A short startup integration can establish finite bridge operation and
quantify that interval's response differences. It cannot establish equivalent
long-duration fatigue, resonant response or timestep convergence of paper
results. The published results retain their original private inputs.

The actual released NACA0018 database and Kynema source/setup are documented
in [the airfoil example](../naca0018_kynema/README.md).
