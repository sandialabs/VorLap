# Model-based negative damping feasibility example

Run from the repository root:

```bash
.venv/bin/python examples/model_based_negative_damping/investigate.py
```

The driver derives an offline negative-damping law from the existing
Grinderslev forced-motion CFD power digitization, projects archived QBlade
structural mass and damping, and compares the resulting equilibrium with
published FSI. It also compares the full work table, damping/data sensitivity,
and averaged response evolution. It uses no experimental response for fitting.

Outputs are written to the ignored
`run_headless/runs/model_based_negative_damping/` directory; use `--output-dir`
for another location. `results.json` binds source and code hashes and records
the numerical environment. `results.md` and `work_and_response.pdf` present
the results.

Read the [investigation](../../docs/model_based_negative_damping/README.md)
and [literature review](../../docs/model_based_negative_damping/literature.md)
for equations, evidence limits, and the path to an independently assessed
generic model. This example does not generate CFD or change the production
VorLap gain model.
