# Fixed-azimuth candidate from the 11.25 m/s idling record

Selected **147 degrees in the recorded QBlade rotor convention**, targeting
Blade 1 flapwise moment at 23% height, seed 4433456. The three-bladed rotor has
120-degree geometric symmetry: 147 degrees is 27 degrees after blade relabeling.
Thus the user's approximately 30-degree orientation is consistent with the
selected rotor geometry. The same turbulent realization need not give identical
labeled-blade histories after that permutation.

The selection maximizes the median over ten rotations of fast flapwise RMS
within a 10-degree circular window, evaluated every degree. A zero-phase,
fourth-order Butterworth high-pass at 0.5 Hz removes slow 1 rpm load variation
while retaining the 2.51/4.57 Hz response bands. Data use physical QBlade time
and verified recorded phase 40.00749588 + 6t degrees; the phase is not reset at
the start of the retained window.

At 147 degrees, median per-revolution RMS is 13.824 Nm (QBlade), 47.572 Nm
(Spectral VorLap), and 99.715 Nm (Hybrid VorLap). These are diagnostic high-pass
moment RMS values, not DEL, fatigue damage, or direct force-contribution
measurements. Hybrid's 25th–75th percentiles are 75.748–115.541 Nm. A local
linear-detrend metric that does not use the high-pass filter independently
selects 147 degrees.

Eight of nine combinations of cutoff 0.2/0.5/1 Hz and window width 5/10/20 degrees
select 146–147 degrees. The broad 20-degree / 0.2 Hz alternative selects 227
degrees, a secondary load-response lobe. The candidate is selected from one seed
and rotating response histories; response lag and wake history mean it is not
yet a proven worst stationary-rotor angle. The requested fixed-rotor matched
trio tests whether sustained loading develops at the selected orientation.

Reproduce from this repository:

```sh
MPLCONFIGDIR=/tmp/nawea_mpl XDG_CACHE_HOME=/tmp/nawea_cache \
  python examples/fixed_azimuth_selection_20260922/analyze.py
```

The compact NPZ in the adjacent paper repository is pinned by SHA256 in
`evidence/selection.json`; supply `--data` to replay another location. No raw
load histories are modified. `evidence/azimuth_metrics.csv` retains each model's
phase-conditioned statistics; `hybrid_revolution_rms.csv` contains each
revolution; the PNG/PDF visualizes the selection and competing lobe.

The fixed-azimuth submission is in `../fixed_azimuth_20260922/`. It must check
actual recorded azimuth: the approximately 40-degree initialized phase of the
old 1 rpm run accumulated during startup and must not be added to the new
147-degree zero-rpm input.
