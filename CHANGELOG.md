# Changelog

## 1.1.0

- Add the Python software-in-the-loop (SIL) QBlade coupling, native bridge
  sources, case preparation and headless campaign tools used by the 2026
  engineering VIV manuscript.
- Include the public 10 m H-VAWT example with independently generated blade
  and torque-tube sections, published reference stiffness targets and a
  documented short Docker comparison.
- Include the actual NACA0018 runtime database, raw force histories and
  portable Kynema/Pointwise preparation scripts. Case preparation refuses
  existing output and requires explicit Slurm submission.
- Retain response models, independent benchmark examples, compact evidence
  and physical/interface regression tests, including the migrated 2025 paper
  recreation contracts and source fixture.
- Improve GUI execution, configuration, plotting and error recovery, with
  offline operation without a QBlade installation.
- Support explicit nodal inflow samples in the SIL runtime. Automatic sampling
  of QBlade nodal inflow remains unconfigured.
- Modernize packaging, contribution and citation metadata, documentation and
  CI verification; exclude downloaded QBlade binaries and generated outputs.

The exact paper case is available from the authors on request, subject to
approval. The public example's model assumptions and measured comparison are
recorded in its README and `comparison_summary.md`.

The previous paper tag is `paper-2025-nawea-v1.0.0`. Its Python package metadata
read `0.1.0`; this release aligns package and software versioning at `1.1.0`,
following the 1.x minor increment.

See the [release notes](docs/release_notes_1.1.0.md) for workflows and validation.
