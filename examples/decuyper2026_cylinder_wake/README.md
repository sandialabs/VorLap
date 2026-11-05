# 4TU prescribed-motion cylinder wake benchmark

This standalone scaffold pins the public 4TU.ResearchData version-1 archive
for **Unsteady Fluid Mechanics Benchmark Dataset - Imposed Cylinder Motion at
Re=100**. It is the proposed force-response benchmark for a minimal wake ROM:
prescribed cylinder motion is known, and the supplied `cy`, `cx`, surface
pressure, and flow fields can be compared without running QBlade.

The source is a Reynolds-100 **CFD simulation dataset, not a physical
experiment**. Its benchmark provenance is Decuyper, De Troyer, and Runacres
(2024), [Unsteady Fluid Mechanics](https://www.nonlinearbenchmark.org/benchmarks/unsteady-fluid-mechanics).
The source report specifies `y` in m, surface kinematic pressure `p_k=p/rho` in
m²/s², `fs=40 Hz`, and a shared developed stationary initial flow. These facts
must still agree with the acquired MAT inventory before they are used.

No reference data are committed. The archive is 6,261,659,558 bytes, and both
`raw/` and `processed/` are ignored. Acquisition requires at least 20 GiB of
free space and never extracts the ZIP automatically.

## Acquisition

From the repository root, inspect the immutable plan without network access or
writes:

```bash
python examples/decuyper2026_cylinder_wake/acquire_reference.py dry-run
```

Acquire or resume the pinned archive explicitly:

```bash
python examples/decuyper2026_cylinder_wake/acquire_reference.py acquire
```

An interrupted transfer remains as `raw/Oscillating_cylinder_dataset.zip.part`.
Resume requires an HTTP `206` response whose `Content-Range` begins at the
exact partial-file length and names the pinned total length. A server that
ignores the range is rejected rather than appended.

Verify an existing archive without using the network or requiring additional
free space:

```bash
python examples/decuyper2026_cylinder_wake/acquire_reference.py verify
```

Verification is deliberately ordered: exact byte count, official MD5, locally
computed SHA-256, then a ZIP central-directory safety audit. The archive is
published from `.part` only after all gates pass. The ZIP audit rejects unsafe
paths, duplicate normalized members, symlinks/special files, and anything
other than one unique
`oscillating_cylinder_benchmark_dataset_v4.mat` basename. It does not extract
or decompress the multi-gigabyte members.

## Frozen benchmark split

`benchmark_protocol.json` preregisters semantic selectors, not guessed MAT
indices:

- Training: the stationary record and all 12 swept-sine records (13 total).
- Development: phase realization 1 from each of six multisine amplitude groups
  (6 total).
- Locked final holdout: phase realization 2 from those groups and all 10
  monosines (16 total).

Raw identifiers remain explicitly unresolved. Before fitting, a separate
human-reviewed, hash-bound inventory must establish the source labels. The
final force histories may not be read for sign selection, preprocessing,
model fitting, architecture selection, or threshold selection.

The supplied force sign is not assumed. Training data must first pass the
pressure-integration gate, including contour orientation, outward normals,
time and displacement units, transverse-coordinate sign, `D`, `U_inf`,
pressure units/reference, coefficient normalization, and a unique transverse
sign. For the source kinematic-pressure representation `p_kin=p/rho`, the
frozen pressure-force formula is
`C_pforce = -2 integral((p_kin-p_ref)n ds)/(U_inf^2 D)`. That formula may be
used only after the metadata confirms the representation and reference.
Unresolved metadata stops the benchmark rather than selecting a convenient
convention.

The minimum ROM uses `s = omega0*t`, `eta = y/D`, and
`u = dy/dt/(omega0*D)`. The stationary training `cy` spectrum fixes `omega0`
before fitting and must place `f0 = omega0/(2*pi)` within 2% of the nominal
3 Hz wake frequency. The optimizer fits only `epsilon`, `b`, `c_q`, `c_v`,
and one bounded `q0`, `qdot0` pair shared by every record. Here `qdot0` means
`dq/ds` evaluated at `s=0`, not a derivative with respect to physical time.

## Evidence and license boundary

Passing this protocol supports code-to-code prescribed-motion wake-force
verification against the CFD reference. It is not experimental validation and
does not establish free-response lock-in, blade fatigue conservatism, QBlade
structural closure, or two-way aeroelastic validation.

The dataset is CC BY-NC 4.0 and is not relicensed by VorLap's MIT license. This
scaffold is for noncommercial research verification and does not redistribute
the archive or processed data. See `DATA_LICENSE_NOTICE.md` before using or
sharing any acquired material.
