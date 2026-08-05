# 2025 NAWEA / WES VorLap computational companion

This directory regenerates the computational content for the manuscript at
paper commit `1b206f3f36d3a5fbfa28ea241759e3671201cd35`. It is intentionally
based on VorLap `main` commit `191d49b801c563ea088b886814e3369b26ed086b`
plus a small set of paper-specific corrections and provenance files.

The published figure files are not overwritten. The driver writes equivalent
figures, corrected inputs, and a numerical summary to an ignored build folder:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r paper/2025-nawea/requirements.txt
.venv/bin/python paper/2025-nawea/reproduce.py
.venv/bin/python -m pytest -q
```

The full run creates `paper/2025-nawea/build/summary.json`, a corrected copy of
the historical NACA 0018 spectral input, a fresh single-Reynolds-number HDF5
from `scripts/NACA18.zip`, and the figure files under `build/figures/`. The two
publication-era histories used by the detailed NACA panels are kept separately
in `inputs/naca_re5e5_paper_histories.zip` so the later raw-data export is not
mistaken for the data that created the archived plots.

## Relationship to the published figures

The closest committed publication-era code state is `3912d7d`; the historical
`paper` snapshot is `e640561`. The final PDFs also contain manual relabeling and
external exports that were not committed to VorLap. Therefore this companion
uses numerical/content equivalence rather than PDF-byte equality.

| Figures | Release behavior |
|---|---|
| 1--4 and the two 84° panels | Replotted headlessly from the publication-era Re=500,000 NACA force histories using the original 4 m CFD span, full-record PSD, and archived 10.3--102 s time window. |
| 5--6 | Replotted from the historical processed spectra. The DC Strouhal entry is corrected to zero; nonzero values, including the illustrative +0.07 shift, are unchanged. |
| 7 | Regenerated from the tracked H-VAWT component geometry. |
| 8 | Regenerated from the tracked FFA-W3-211 raw history and processed spectrum on a common 1 ms, interval-centered grid, retaining DC plus eight dominant oscillatory entries. The first incomplete interval containing the source startup impulse is excluded. The comparison reports the reviewer-requested demeaned Pearson correlation of 0.90 over the plotted 0--2 s window. |
| 9 | Regenerated with the stated 15 Hz verification frequency. |
| 10--13 | Archival ParaView mode-shape exports from the external structural solver. Their solver/export state is not in VorLap and is not falsely claimed as regenerated here. |
| 14 | `fig14_torque.pdf` is an explicitly quarantined compatibility calculation using the publication-era elementwise `r * F` plotting quantity. `fig14_torque_corrected.pdf` is also emitted using the physical `r × F` moment. The library always uses `r × F`. |
| 15 | Regenerated from the corrected executable reference-turbine case. |
| 16 | `fig16_ReconstructedForce.pdf` preserves the archived waveform, which the publication-era script obtained from global node 2 (Arm 1), despite the manuscript caption naming Blade 3, node 2. `fig16_ReconstructedForce_blade3_node2.pdf` is the corrected companion for the stated selection at 60° and 6 m/s. |

## Reviewer-sensitive conventions

The driver and tests enforce the conventions added during review:

- the NACA 0018 thickness ratio is 0.18;
- the signed mean is stored at zero Strouhal number;
- the illustrative +0.07 shift applies only to non-DC entries;
- even-record Nyquist entries are retained once, while the last positive bin of
  an odd record is doubled;
- the verification modal frequency is 15 Hz;
- the verification comparison uses 1 ms interval centers and DC plus eight
  oscillatory entries, reproducing the reviewed $r=0.90$ diagnostic;
- the reference-turbine sweep includes every integer speed from 1 through
  16 m/s;
- physical moments in the library use `r × F`;
- component-local node selection is explicit; the archived global-node Figure
  16 behavior is quarantined to the paper compatibility output;
- the low-AoA Strouhal reference length is bounded below by physical airfoil
  thickness, preventing singular dimensional frequencies.

The tracked top-level `data/airfoils/NACA0018.h5` is retained unchanged as an
archival publication input. The reproduction driver makes and records a
corrected build-local copy; it never silently changes or uses the archival file
as corrected data. The compact paper-input archive covers the two Re=500,000
detailed-history panels. Raw histories behind the multi-angle and other-
Reynolds-number summary planes are not duplicated because the tracked
historical HDF5 is sufficient for Figures 5--6; that limitation is recorded in
`provenance.json`.
