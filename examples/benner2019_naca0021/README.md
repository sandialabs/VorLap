# Benner et al. (2019) VAWT-airfoil VIV benchmark

This is the primary wind-turbine-airfoil experimental comparison for the latest
VorLap--QBlade bridge.  Benner et al., *Journal of Fluids and Structures* 91
(2019), 102577, <https://doi.org/10.1016/j.jfluidstructs.2019.01.018>, tested a
flexibly mounted NACA 0021 airfoil used in VAWT designs over 0--180 degrees.
Cross-flow VIV was observed from approximately 60--130 degrees.  At 90 degrees,
the reported peak was `A/c = 1.93` at `U* = 5.7`, and the broadest reported
lock-in interval was approximately `1.7 < U* < 12`.

The benchmark contains exactly three response series:

1. Benner et al. experimental displacement and frequency data;
2. raw QBlade with native aerodynamics and no supplemental VorLap force; and
3. the same QBlade case with the latest airfoil-spectral lock-in bridge.

The bridge currently uses NACA 0018 spectral input.  The 18%-to-21% thickness
transfer is accepted as a declared high-incidence modeling approximation, not
as exact geometry identity.  Bidadi et al. (2024) found that deep-stall vortex
shedding of slender wind-turbine airfoils is much less sensitive to thickness
than the attached-flow regime, with shedding frequencies nearly independent of
thickness and VIV concentrated near 90 degrees.  This supports using Benner as
the VAWT validation case above 30 degrees; it does not prove identical force
amplitudes.  The observed raw-QBlade/hybrid/experiment amplitude discrepancy is
therefore reported rather than corrected by tuning.

The remaining source-data mismatch is Reynolds number, not thickness.  Benner
et al. report `600 <= Re_c <= 13300`, whereas the tracked NACA 0018 spectral
planes begin at `Re_c = 40000`.  The Benner recreation therefore requires a
lower-Reynolds-number spectral source before it can be called a direct hybrid
validation; silently clipping or extrapolating the present source is not
allowed.  The profile-matched Ehrmann benchmark at `Re_c ~= 125000` is already
bracketed by the tracked source and is the immediate executable experiment.

Before launching, digitize the published amplitude and frequency curves with
figure/page/checksum provenance and extract the exact mass, natural frequency,
damping, spring, blockage, and initialization values.  The two numerical cases
must use identical structural and native-aerodynamic inputs.  The declared
screening tolerance is `abs(A_model-A_experiment)/c <= 0.30*A_experiment/c +
0.01`; lock-in onset/range and frequency are separate reported diagnostics.
