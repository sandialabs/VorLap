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
- Interpolate `St`, amplitude, and phase over Reynolds number and angle of attack grids (per ranked peak index).
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

Shedding frequencies are estimated from Strouhal form:

`f = St * V_eff / L_st`

with `L_st = chord * |sin(AOA)|` and a small epsilon floor to avoid divide-by-zero at near-zero AOA.
