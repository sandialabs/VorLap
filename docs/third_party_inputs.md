# Third-party inputs

The root MIT license covers VorLap's own code. Retaining an input in this
repository does not change its source's license or establish ownership.

## QBlade

QBlade CE binaries, support libraries, bundled sample projects, and controllers
are external installations. They are excluded from the 1.1.0 source snapshot
and Python distributions. Obtain the required version from the
[QBlade project](https://qblade.org/downloads/) and keep the supplied notices.

`run_headless/config/hvawt_pilot_hpc_inputs/StrModel_qblade_2.0.4.cpp` is a
verbatim third-party structural-parser reference, not VorLap implementation.
Its header credits David Marten and specifies Academic Public License v1.0.
The accompanying `LICENSE_QBlade_APL-1.0.txt` retains that license. The file is
used to check the historical structural-field convention; do not relicense it
as MIT or substitute a changed parser without reopening its identity check.
QBlade's [license information](https://qblade.org/qblade-license-information/)
and the source header identify the separate terms.

## Adapted H-VAWT model

The public `examples/ReferenceHVAWTQBladeModel/` now replaces the blade and
torque-tube sectional inputs with independently generated generic sections.
The blade targets the published 2024 OWENS reference bending stiffnesses;
other properties follow explicit example assumptions. Original paper inputs
and the U05 source-case archive are private external recreation inputs.
The exact paper case is available from the authors on request, subject to
approval. The released airfoil database and CFD setup are documented in the
NACA0018/Kynema example.

The short Docker spectral-SIL comparison checks this example's startup
integration. It does not transfer the original paper's fitted modal basis,
fatigue results or qualification to the replacement. Further modal-response
claims require new identification and appropriate simulations.

## Airfoil and benchmark data

`data/airfoils/` contains the retained NACA0018 database, the generic legacy
spectra, and the Bidadi FFA spectral bundle. The bundle manifest and its
builder record source identity and processing. `examples/` retains separately
documented Ehrmann, Horcas, Grinderslev, Fu, Vlastos, rectangular-section,
and other benchmark datasets and evidence. Consult each example's README,
reference-data notes, protocols, and receipts for attribution, digitization,
calibration, training, and holdout boundaries.

The companion paper supplies cited images, compact result reductions,
publication files and evidence records. Its manifest verifies file identity.
Use the companion paper's attribution and environment instructions for replay.
