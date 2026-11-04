# Theory Notes

## VorLap Methodology Summary (From the Paper)

The NAWEA paper describes VorLap as a workflow with two primary outputs:

- A frequency-overlap screening metric to identify likely shedding-mode alignment.
- Reconstructed time-domain nodal loads for one-way structural coupling.

At a high level, the methodology is:

1. Spectral preprocessing of source data:
- Start from lift/drag (or force/moment) time series from CFD/experiments.
- Convert frequencies to Strouhal form (`St = f c_n / V_inf`) and force levels to coefficient form (`C_L`, `C_D`).
- Compute FFT amplitude/phase and rank non-DC peaks by PSD using combined-force magnitude (`C_F = sqrt(C_L^2 + C_D^2)`), retaining only the top ranked peaks.

2. Local kinematics for each structure node:
- For each inflow speed and azimuth, resolve local angle of attack and effective speed from inflow projected on local chord/normal directions.
- Compute local Reynolds number from effective speed and local chord.

3. Interpolation in aerodynamic parameter space:
- Interpolate `St`, amplitude, and phase over Reynolds number and angle of attack grids (per ranked peak index). Phase interpolation is performed through unit phasors, not by direct linear interpolation of radians, so values on opposite sides of the `+pi/-pi` branch cut remain adjacent.
- Use this to map source spectral data onto the local conditions at each node.

4. Combinatorial overlap reduction:
- For each candidate shedding frequency, compare against each structural natural frequency and selected harmonics.
- Track the minimum absolute percent difference as the “worst-case overlap” metric for each inflow/azimuth point.
- Apply practical filters:
  - Peak-depth limit (only top `N` ranked frequencies).
  - Amplitude cutoff (ignore low-energy peaks).

5. Optional signal reconstruction:
- Reconstruct local lift/drag time signals from interpolated frequency, amplitude, and phase.
- Rotate and assemble these nodal loads into global force time histories for downstream structural models.

6. Time-varying inflow reconstruction:
- For nonstationary inflow files (`time`, `inflow_speed`, direction), VorLap evaluates local Re/AOA at each time sample.
- It interpolates Strouhal/amplitude/phase at each sample and converts Strouhal to instantaneous shedding frequency.
- Harmonics are synthesized with phase continuity (`dphi/dt = 2*pi*f(t)`), using FFT phase as the initial condition.
- Optional cycle-based smoothing can be applied to frequency/amplitude trajectories to avoid non-physical jumps under rapid inflow changes.

This is the bridge the paper emphasizes: from high-dimensional unsteady spectral datasets to practical, design-stage VIV screening and load synthesis on arbitrary beam-type multi-body structures.

## Core Load Model

At each component node, VorLap computes:

- Effective speed from local inflow projection on chord/normal directions.
- Reynolds number from effective speed and local chord.
- FFT coefficient lookup/interpolation for `CL`, `CD`, and `CF`.
- Nodal force from dynamic pressure scaling.

## Spanwise Nodal Weighting

Nodal load weighting uses half-segment averaging:

- End nodes: half of adjacent segment length.
- Interior nodes: average of left/right adjacent segment lengths.

This avoids dropping first-node contribution and keeps total nodal span weighting consistent.

## Moments

Global moments are computed as:

`M = r × F`

where `r` is the node position relative to `rotation_axis_offset`.

## Frequency Overlap

VorLap airfoil FFT files use an explicit DC-first spectral schema:

- Spectral index 0 is the mean/DC coefficient entry for `CL`, `CD`, `CM`, and `CF`.
- The index-0 Strouhal number must be exactly zero.
- The index-0 amplitude is the signed mean coefficient; DC phase is ignored but must be finite.
- Spectral indices 1 and higher are oscillatory peaks with nonnegative amplitudes.
- Any nonzero harmonic amplitude must have a positive Strouhal number.

The `AirfoilFFT` constructor validates this schema strictly. The HDF5 loader also validates after orienting dimensions. For legacy files that store the mean coefficient at index 0 but carry nonzero index-0 Strouhal metadata, the loader repairs only the index-0 Strouhal values to zero and emits a `RuntimeWarning`. The loader also converts signed harmonic coefficients to nonnegative amplitude/phase form by adding pi to the phase where needed. A ranked-peak file that omits the DC entry, or stores a zero-Strouhal DC entry away from index 0, is rejected rather than interpreted as a mean-load spectrum.

FFT phase offsets are circular variables. The scalar and batch interpolation paths therefore interpolate `sin(phase)` and `cos(phase)` and recover phase with `atan2`, with a deterministic fallback only for exactly cancelled phasors where the circular mean is undefined. This prevents a `+179 deg` / `-179 deg` neighborhood from being interpolated as `0 deg`.

Reynolds-number and AOA queries outside the source grid are not extrapolated linearly. VorLap flat-clips them to the nearest source-grid boundary. The interpolation API now diagnoses this explicitly: the default `bounds_policy="warn"` emits `AirfoilRangeWarning`, `bounds_policy="raise"` fails strict audit runs, and `bounds_policy="clip"` keeps silent clipping only when a caller has explicitly accepted it. Paper-grade campaigns should use warning-strict or raise-on-out-of-range settings and report any clipped cases.

The current QBlade bridge returns nodal forces reconstructed from CL/CD spectra. CM spectra are loaded and schema-validated as source metadata, but they are not reconstructed into QBlade nodal torque columns by the present force path. Loading files written without an explicit torque array intentionally contain zero moment columns.

Shedding frequencies are estimated from Strouhal form:

`f = St * V_eff / L_st`

with:

`L_st = chord * max(|sin(AOA)|, thickness_ratio)`

This uses the projected cross-flow chord length for the Strouhal scaling while preventing the reference length from collapsing below the physical airfoil thickness at low angles of attack. For a typical 12% thick airfoil, the thickness floor is active only at very low AOA; above that, the scaling is the requested `chord * |sin(AOA)|` convention. For round or very bluff members where `thickness_ratio` is near 1, the same expression recovers a diameter-like reference length.
