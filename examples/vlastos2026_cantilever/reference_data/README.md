# Figure 9 vector digitization

The CSV contains the `Sc=5.18` experimental markers and the principal
varying-parameter hGAST curve from Figure 9b of Vlastos et al. (2026),
<https://doi.org/10.5194/wes-11-2585-2026>. The publisher figure is CC BY 4.0.

The high-resolution publisher PDF was downloaded from
<https://wes.copernicus.org/articles/11/2585/2026/wes-11-2585-2026-f09-high-res.pdf>
and had SHA-256
`e20f72c24c431b064615883965ef0f20d0f69b91bdefddf1c1ec52ef9cf9ee91`.
The vector plot rectangle was mapped exactly from `U*=3.5..8.5` and
`y_max/D=0..0.6`; filled-square centers and the continuous path vertices were
then transformed to data coordinates. The publisher PDF itself is not
redistributed here. Values are figure-digitized evidence, not author-supplied
raw histories; plotting should retain that distinction.

`all_sc_figure10_peak_digitized.csv` adds the experimental, calibrated
Hartlen--Currie, Eurocode Approach 2, and Blevins peak envelopes from the same
publisher asset. Values are read to approximately +/-0.01 in `y_max/D` and
carry that uncertainty explicitly. The Vlastos text independently states
that the Sc=2.94 Hartlen--Currie result overpredicts the experiment by about
55%, that higher-Sc peaks agree well, that Eurocode overpredicts the
cantilever response, and that Blevins underpredicts it.

`vlastos2026_model_parameters.csv` records the three exact Table 1 structures,
the Figure 7 regressions `G'_max = -1.1922 Sc + 14.084` and
`H'_max = 0.1593 Sc + 0.972`, and figure-derived peak velocities. The
published geometry shared by the cases is `H=0.775 m`, `D=0.050 m`; the
calibration uses `CL0=0.3`. Vlastos et al. activate the calibrated oscillator
over a prescribed reduced-velocity interval and linearly decay its parameters
to zero at the interval ends. Their broad cantilever sensitivity range is
`U*=[4.5,10]`.

The publisher Figure 10 web PNG had SHA-256
`304712615814061f651960e049a41cfdefb56eb6ca52c7aead43616eae951e53`.

The geometric/structural values used by the generator come from publisher
Table 1. Its XLSX URL is
<https://wes.copernicus.org/articles/11/2585/2026/wes-11-2585-2026-t01.xlsx>
and its SHA-256 is
`d24a65f63a33783ee6d69cc239915affa0c2cb6a43eb180cd1ee9db2ee7c4a22`.
