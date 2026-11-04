# Native-x86 H-VAWT pilot inputs

The original pilot source case is now a private external paper input. Public
1.1.0 preparation is intentionally not self-contained for that historical
campaign. Set `VORLAP_PAPER_SOURCE_ARCHIVE` to the original private archive
when running the historical pilot or native nodal-inflow verification.
The private paper source snapshot retains the complete original launcher and
its historical Git/runtime requirements. The new public sections must not
replace the old archive while preserving old paper qualification receipts.

The identities are:

- `U05p00ms_source_case.tar.gz` is the reviewed 20-file, 1-RPM H-VAWT source
  case used to construct the path-bound structural variants and response pilot.
  It embeds the calibrated tracked `NACA0018.h5` bytes (SHA-256
  `64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369`).
  Archive SHA-256:
  `6f6daf91d026087bb5f5acec6b313c7cead1a70faa1deded1819e15c60d7d403`.
- `StrModel_qblade_2.0.4.cpp` is the archived QBlade 2.0.4 structural parser
  source used only to verify the documented SUBELEMENTS damping/drag field
  order. SHA-256:
  `398bde198b8a3551cdc8469256be2d4b6cd8885ed1e003d93e7464cdf7b025e2`.

`run_headless/prepare_hvawt_response_pilot_native.sh` verifies both identities,
extracts the source case, creates the documented Rayleigh selection artifact,
generates the exact 24-case response pilot on its execution filesystem, and
validates/lists the campaign. It does not launch QBlade.

The projected damping ratios used by the pilot are engineering assumptions
derived from the documented parser convention and tracked modal strain energy;
they are not experimental or active-decay measurements.
