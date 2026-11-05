# Marra 4:1 rectangular-cylinder training evidence

The three exact peak-response rows are transcribed from Table 5 of Gao et al.,
which republishes the Marra et al. (`B/D=4`) wind-tunnel response and gives the
structural parameters used for each Scruton number. They are training data only.
The previous Mannini `B/D=1.5` assessment was committed as a failed independent
test before these rows were permitted to alter the closure.

The fixed-body scale uses Knisely's reported `CL_rms=0.64` and `St=0.140`, both
referenced to the cross-flow depth. The runtime closure uses the
sinusoidal-equivalent coefficient `sqrt(2)*CL_rms`.

After the spectrum-scaled Rayleigh revision failed this dataset, these rows were
used to fit the two coefficients of a static-spectrum-scaled Gao--Scanlan
reciprocal-Griffin closure. The calibrated force per span is

```text
f = 0.5*rho*U^2*D*Ceff*w * (Gc*sign(v) - Gv*v/U)
```

with `Gc=0.6977212` and `Gv=18.88355`. The first term injects work at low
amplitude and the second provides restoring aerodynamic damping at high
amplitude. This is training evidence only; the untouched assessment is the Liu
5:1 case in the neighboring example directory.
