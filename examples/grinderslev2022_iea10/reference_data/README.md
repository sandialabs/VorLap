# Grinderslev P100--I50 forced-motion power digitization

`p100_i50_figure6_power_digitized.csv` contains the black-diamond P100--I50
curve from Figure 6 of Grinderslev et al. (2022),
<https://doi.org/10.5194/wes-7-2201-2022>. The publisher web figure URL is
<https://wes.copernicus.org/articles/7/2201/2022/wes-7-2201-2022-f06-web.png>;
the downloaded 2067-by-999 PNG had SHA-256
`76dd4306bad74642fb0bbb8b3320f01c7ae240605f939fa68e37bc50d2af501b`.

The plot rectangle was mapped from tip amplitude 0--2 m and aerodynamic power
-1000--1800 W. Diamond centers were transformed to data coordinates and
rounded consistently with the visual resolution; the CSV carries estimated
digitization uncertainties. These are figure-derived points, not
author-supplied CFD histories. The paper reports the exact external targets
used to cross-check the digitization: the undamped zero crossing is about
1.6 m, the baseline-damped FSI equilibrium is about 1.18 m and 775 W, and the
forced-motion/damping intersection is about 1.24 m and 880 W. The 2023
corrigendum, <https://doi.org/10.5194/wes-7-2201-2022-corrigendum>, is
mandatory when using the amplitude-evolution equation.

This CC-BY figure-level dataset supports calibration and code-to-code
comparison. It does not replace the authors' force histories or establish
independent validation.
