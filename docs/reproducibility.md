# Code examples and paper recreation

VorLap 1.1.0 supplies the solver, GUI, QBlade SIL bridge, benchmark examples,
campaign drivers and compact code-side evidence accompanying **An Engineering-Level
Vortex Induced Vibration Methodology for Parked-Load Assessment**.

## Public example and exact paper case

The public 10 m H-VAWT example uses replacement blade and torque-tube
sectional definitions and independent section stations. Its README records
the published stiffness targets, generic assumptions and measured short-run
differences. The public example demonstrates the bridge workflow; the paper's
original response model remains bound to its original structural inputs.

The exact paper case is available from the authors on request, subject to
approval. Please feel free to contact the authors to discuss access.
The actual NACA0018 runtime database, raw force histories and Kynema/Pointwise
setup are included in the public examples.

## Software verification

Install the numerical constraints used for exact replay, then run:

```bash
python -m pip install -c reproducibility/requirements-software-checks.txt -e ".[dev,docs,gui]"
python -m pytest
python -m mkdocs build --strict
python examples/basic_synthetic_case.py
python examples/single_blade_verification.py
```

The core solver, GUI, campaign, benchmark and synthetic physical checks run
without a QBlade installation or companion paper. Paper-only modules skip
explicitly when that separately maintained checkout is absent. Tests bound
to a qualified QBlade/bridge installation are marked `requires_qblade`.
To require those runtime checks:

```bash
python -m pytest --require-qblade -m requires_qblade
```

Install the qualified QBlade CE 2.0.9.7 runtime, its SIL wrapper and the VorLap
bridge at `run_headless/QBladeCE_2.0.9.7/`. The presence check does not bypass
byte identities. Historical integrations use their recorded source and run
inputs. Campaign provenance checks require an installed Git checkout; the
Python distribution itself can install without Git history.

## Companion paper replay

The manuscript, figures, compact reductions and figure generators are
maintained separately. Obtain the companion snapshot and any required exact
case inputs from the authors. Put the companion files at
`paper_drafts/NAWEA_QBladeBridge/` and follow that snapshot's top-level and
`analysis/reproducibility/README.md` instructions.

The code-side `reproducibility/sil_paper_manifest.json` verifies selected
companion file identities:

```bash
python scripts/verify_paper_snapshot.py
python -m pytest --require-paper -m paper
```

File identity verification is separate from executing the scientific replay.
Use the companion snapshot's matching verifier and rendering environment.
Historical byte comparisons name Python 3.11.5, NumPy 2.4.6, Matplotlib 3.10.9
and FreeType 2.14.1. Different BLAS, font or rendering builds can change
floating-point fingerprints and figure bytes. Preserve retained numerical
results when diagnosing those differences. Manuscript builds also require
`latexmk`, `qpdf`, `pdfinfo`, `pdftotext` and the documented Bash version.

## Full simulation replay

Compact paper replay uses retained reductions. A fresh simulation campaign
additionally requires the original case and wind records, qualified QBlade
and Python/bridge environment, and the computational resources named by the
campaign receipts. Historical launchers accept `VORLAP_PAPER_SOURCE_ARCHIVE`
for the exact source-case archive supplied by the authors.

Obtain QBlade separately from the [QBlade project](https://qblade.org/downloads/)
and retain its notices. Set `QBLADE_ROOT` for the SIL CLI and follow
`run_headless/README.md` for campaign contracts. The original modal basis and
campaign receipts do not qualify a changed structural model automatically.
