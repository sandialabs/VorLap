# 2025 NAWEA paper recreation contracts

The paper drivers and frozen inputs live in the `2025_NAWEA_VorLap` paper archive,
under `reproduction/`. Its top-level `README.md` is the recreation recipe.
`reproduction/reproduce.py` replaces the standalone
`test/single_blade_verification_Reconstruction.py` and historical
`test/test_vorlap_Reference_Turbine.py`; exact originals are retained in that
archive's `reproduction/legacy_drivers/` for provenance.

Run the fast library contracts from this checkout:

```bash
python -m pytest -q tests/test_paper_recreation_contracts.py
```

These 12 parameterized cases run in the ordinary `tests/` CI suite, without a
paper checkout. They verify:

- FFT-to-Strouhal-to-time-history consistency after changing speed and chord,
  including negative DC, phase, even Nyquist, odd final-bin scaling and the
  low-angle thickness floor.
- Raw-history conversion of pitching moment using `Mtz/(q*chord)`, with the
  illustrative Strouhal offset applied only to non-DC entries.
- Analytic multi-component force histories, nonuniform nodal integration,
  component/node indexing, mean-load inclusion, rotated section bases and
  physical moment about an offset axis, through both public solver paths.
- Full synthetic speed/azimuth overlap grids, harmonic targets, amplitude and
  frequency cutoffs, and exclusion of DC from overlap.
- The actual nine-entry FFA-W3-211 verification waveform and its
  `0.9040041661508004` Pearson correlation, using the small provenance-tracked
  fixture in `tests/fixtures/nawea2025/`.

Existing interpolation, file-I/O, converter, computation and modal-overlap tests
continue to cover the individual components. The full paper replay additionally
compares complete historical overlap grids, histories, PSDs and processed
spectra against the earlier companion, with tolerances and hashes recorded in
the paper archive. Run that acceptance check after changes to these APIs:

```bash
VORLAP_REPO="$PWD"
PAPER_ARCHIVE=/path/to/2025_NAWEA_VorLap
PYTHONPATH="$VORLAP_REPO" python "$PAPER_ARCHIVE/reproduction/reproduce.py"
```

Use the paper's pinned environment for that full replay. It is not part of this
repository's ordinary CI because the paper inputs are housed separately.

Physical compatibility does not mean retaining old plotting mistakes. The
archived Figure 8 used elementwise `r * F`; physical torque remains `r × F`.
Archived Figure 10 selected Arm1/node2 and shifted the mean into an oscillation;
the paper driver isolates that historical plot and also emits corrected
Blade3/node2 loads. These obsolete conventions must not enter the core library.
External structural mode-shape images are archival copies, not VorLap outputs.

The fixture was extracted from companion commit
`45f827e11356586b7963fc1be8a752e9e8c3745d`, before this migration. See its
`provenance.json` for source hashes, units, selection and tolerances. Investigate
physical or numerical changes before updating any expected values.
