# Model-based negative damping: literature and derivation

Investigation date: 2026-09-09. Scope: the amplitude-envelope law implemented
in `vorlap/qblade_single_mode_engineering_lockin_sil.py` and the two tagged
gain coordinates in `vorlap/unified_viv_gain.py`. This is a feasibility note,
not a new validation claim.

The practical route is to obtain signed aerodynamic work from prescribed
motion, combine it with modal structural losses, and predict free response.
Both the small-amplitude gain and aerodynamic saturation can then come from
an aerodynamic calculation. Structural analysis supplies modal mass, stiffness,
motion shape, and whatever damping law has actually been specified or
identified. It cannot by itself supply the aerodynamic saturation parameter.
The published demonstrations support this route within specified geometries
and flow conditions. They do not establish experimental-fit accuracy for an
arbitrary airfoil, Reynolds number, or three-dimensional blade.

## Meaning of H and Lambda in the present code

Let `u = phi*q`, with dimensionless mode shape `phi`, modal displacement `q`
in metres, and peak harmonic amplitude `A`. Use generalized force `Q` in N,
and positive aerodynamic power for work transferred from flow to structure.
The current envelope law is

\[
 M\ddot q+C_s\dot q+Kq=Q_a,\qquad
 Q_a=c_{a0}\left(1-\frac{A^2}{A_\infty^2}\right)\dot q,
 \qquad A_\infty=\Lambda L_\mathrm{ref}.
\]

Here `c_a0` and `C_s` are kg/s. Positive `c_a0` is destabilizing. For fixed
frequency `omega`, period-mean powers and the slowly varying envelope obey

\[
 P_a=\tfrac12c_{a0}\omega^2A^2
       (1-A^2/A_\infty^2),\quad
 P_s=\tfrac12C_s\omega^2A^2,\quad
 \dot A=\frac{P_a-P_s}{M\omega^2 A}.
\]

The last equation assumes a nearly harmonic, slowly evolving response at
fixed frequency. Its dimensional form also avoids division by structural
damping when `C_s=0`. The [Grinderslev corrigendum](https://wes.copernicus.org/articles/7/2201/2022/wes-7-2201-2022-corrigendum.pdf)
must be used if implementing that paper's amplitude-evolution equation.

With fixed coefficients, the nonzero stable equilibrium is

\[
 A_\mathrm{eq}=A_\infty\sqrt{1-C_s/c_{a0}},\qquad c_{a0}>C_s.
\]

Thus `Lambda` specifies the amplitude where **aerodynamic work** becomes
zero; it is not the damped equilibrium amplitude or a structural strain
limit. A perfectly linear elastic oscillator can exhibit this saturation
because its fluid forces are nonlinear. Conversely, conservative nonlinear
stiffness changes frequency and motion shape but dissipates no net energy
over a closed cycle. Its effect enters through the sampled aerodynamic map
and the structural backbone, not through a universal structural Lambda.

In the airfoil branch, `H=c_a0/B_spectral`. The code computes

\[
 B_\mathrm{spectral}=\sum_i
 \tfrac12\rho U_i c_i\Delta s_i\,
 C_{\mathrm{captured},i}\,\phi_i^2,
\]

where the captured coefficient already includes the implemented directional
projection. The branch's `H` is dimensionless, but its value depends on this
specific spectral basis. In the turbine branch the stored gain is instead
`c_a0/C_s`. Those gains are different coordinates and cannot be transferred
as though they were the same aerodynamic coefficient.

## A useful structural contribution to saturation

The following is a derivation for a scalar, independent-strip, local
amplitude-envelope model, not a statement that strips reproduce blade flow.
Let local aerodynamic force per span satisfy

\[
 f(s)=b(s)\left[1-
 \left(\frac{A|\phi(s)|}{\lambda_a(s)L(s)}\right)^2\right]
 \phi(s)\dot q.
\]

For a fixed motion direction and coefficients, modal projection gives

\[
 c_{a0}=\int b\phi^2\,ds,\qquad
 d_a=\int\frac{b\phi^4}{[\lambda_a L]^2}\,ds,\qquad
 A_\infty^2=\frac{c_{a0}}{d_a}.
\]

The positive-gain, positive-saturation form requires `c_a0>0` and `d_a>0`.
Structural modes therefore determine how a known sectional saturation is
converted to modal/tip saturation. They do not determine the unknown
sectional `lambda_a`. Signed local coefficients, cross-mode forces, torsion,
or spatially coupled aerodynamics may require a more general map.

For uniform `b`, `lambda_a`, and `L`, a rigid mode has
`A_inf=lambda_a*L`. The illustrative normalized shape `phi=x`, `0<=x<=1`,
has `A_inf=lambda_a*L*sqrt(5/3)`. This is a modal-weighting example, not a
cantilever eigenfunction approximation. Under `phi -> r*phi`, generalized
amplitudes scale by `1/|r|`, leaving physical displacement unchanged.

Do not confuse this envelope law with an instantaneous `q^2*qdot` law:
`<q^2*qdot^2>=A^4*omega^2/8`, whereas
`A^2*<qdot^2>=A^4*omega^2/2`. Their cubic coefficients differ by four.

## What an aerodynamic model must provide

For each prescribed harmonic motion `q=A*sin(omega*t)`, calculate

\[
 P_a=\langle Q_a\dot q\rangle,\qquad
 c_a(A,\omega)=\frac{2P_a}{\omega^2A^2}.
\]

Fit `c_a=c_a0-d_a*A^2` only where the data support that shape. Then
`A_inf=sqrt(c_a0/d_a)` and `H=c_a0/B_spectral` follow algebraically. These are
CFD-derived surrogate coefficients if the inputs are CFD, even though a
regression is still used. Experimental response data need not enter fitting.
Fit quality does not independently establish aerodynamic accuracy.

Retain the force component in phase with displacement as well as velocity
if the response frequency is not demonstrably fixed. A purely energy-based
map at one frequency determines amplitude balance but does not determine
the frequency shift due to aerodynamic stiffness or added mass.

For a section of chord `c` and represented span `ell`, the existing work-map
normalization is

\[
 C_P=\frac{\langle F'\dot y\rangle}{\tfrac12\rho U^3c\ell},
 \quad k=\frac{\omega c}{2U},\quad a=\frac{A}{c}.
\]

Writing `C_Fv=2*<C_F*cos(omega*t)>` for the force coefficient component in
phase with `ydot`, one obtains `C_P=k*a*C_Fv`. A fixed-body force spectrum
specifies frequency and magnitude but lacks this phase relative to body
motion. For an imposed force harmonic of amplitude `F0` and phase `theta`
relative to displacement, `P=0.5*F0*A*omega*sin(theta)`; identical spectral
magnitudes can therefore inject, remove, or transfer zero mean energy.

At very small amplitudes, estimates of `P/A^2` need a resolved linear-response
limit, enough cycles, and phase statistics. A prescribed zero-motion record
does not measure force-motion phase. The current map's quadratic origin is
a regularity assumption, not new aerodynamic data.

If native QBlade aerodynamics remain active, fit incremental power from
paired histories at identical motion and physical conditions:
`P_increment=< (F_high_fidelity-F_native)*ydot >`. Adding the full CFD work
would count native work twice. Signed power is necessary to identify zero
crossings and stable/unstable branches; a single positive gain plus quadratic
saturation cannot represent every work-map topology.

## Primary evidence and its limits

| Source | What it establishes for this task | Transfer boundary |
| --- | --- | --- |
| [Grinderslev et al. (2022), WES 7, 2201–2213](https://wes.copernicus.org/articles/7/2201/2022/) | Forced-motion CFD work curves plus structural damping predict single-blade VIV. P100–I50 has an undamped work zero near 1.6 m versus FSI about 1.63 m; the damped intersection is about 1.24 m versus FSI about 1.18 m. Other incidences exhibit multiple possible response amplitudes. | CFD-to-FSI agreement for one IEA 10-MW blade, not experimental validation or a NACA0018 sectional law. Public figures exist; underlying histories are available on request. |
| [Grinderslev et al. (2023), WES 8, 1625–1638](https://wes.copernicus.org/articles/8/1625/2023/) | Provides modal `V^T M V`, `V^T C V`, `V^T K V` dissipation calculations and independent CFD sensitivity. High-inclination power is 1280–1450 W across methods; low-inclination URANS predicts +1560–2000 W while resolved methods give about −300 W. | Numerical grid convergence of URANS does not establish correct stability; spanwise flow/coherence and turbulence closure can reverse the work sign. Histories are on request. |
| [Morse and Williamson (2009), JFM 634, 5–39](https://doi.org/10.1017/S0022112009990516) | Controlled-motion cylinder force/phase maps predict free response and distinguish stable, unstable, hysteretic, and intermittently switching states. | Experimental aerodynamic identification; establishes the method, not an experiment-free universal model. |
| [Menon and Mittal (2020), JFS 97, 103078](https://arxiv.org/abs/2003.02402) | Computed prescribed-pitch energy maps predict growth, saturation, and gust-triggered transitions in an airfoil oscillator. | Low-Reynolds-number pitching stall-flutter study; it is not a high-Reynolds-number translating NACA0018 VIV map. |
| [Skrzypinski et al. (2014), Wind Energy 17, 1495–1514](https://onlinelibrary.wiley.com/doi/abs/10.1002/we.1647) | Moving DU96-W-180 airfoil calculations produce negative aerodynamic damping near the stationary shedding frequency in both 2D and 3D CFD. | Confirms airfoil applicability of motion-dependent calculations; does not supply generic profile transfer. |
| [Facchinetti, de Langre and Biolley (2004), JFS 19, 123–140](https://yakari.polytechnique.fr/Django-pub/documents/matteo2004rp-2pp.pdf) | A van der Pol wake oscillator with acceleration coupling captures important cylinder VIV behavior. The familiar `epsilon=0.3`, `A=12` values were selected against experiments. | Useful low-order architecture and cylinder prior; those constants are not derived from structural FE and are not universal airfoil constants. |
| [Vlastos et al. (2026), WES 11, 2585–2600](https://wes.copernicus.org/articles/11/2585/2026/) | Hartlen–Currie constants are fitted to rigid-cylinder experiments then used with a cantilever beam. The lowest-Sc tip amplitude is overpredicted by about 55%; lock-in width needs cantilever-specific calibration. | Direct evidence against assuming even rigid-cylinder-to-cantilever calibration transfers perfectly. Low-Reynolds-number data, first bending mode, no ambient-turbulence assessment. |
| [Bidadi et al. (2023), Journal of Turbulence 24, 393–418](https://www.osti.gov/servlets/purl/1992015) | Documents NACA0012/0021 deep-stall grid/turbulence requirements and stationary force statistics. | Useful for generating baseline CFD and shedding spectra, not a substitute for synchronized prescribed-motion work. |
| [Fu et al. (2026), JFS 143, 104537](https://www.sciencedirect.com/science/article/pii/S088997462600037X) | Open ExaWind blade-resolved VIV approach, structural modes and grid/time-step sensitivity. The selected full-blade simulations use 1456 CPU cores for about 10 days per case. | Comparison with other simulations; experimental validation is still planned in the paper. Cost motivates offline maps and inexpensive online projection. |
| [Bourguet (2026), JFM 1038, R1](https://doi.org/10.1017/jfm.2026.11748) | Low-Reynolds-number numerical examples show some vibration branches increasing with added structural damping, explained through amplitude-frequency energy maps. | Does not transfer quantitatively to wind-turbine blades, but demonstrates that a universally monotone fixed-frequency damping/amplitude rule is too restrictive. |

## Public starting points and the remaining data gap

1. The [IEA 10-MW geometry and structure](https://github.com/IEAWindTask37/IEA-10.0-198-RWT)
   are the shared published blade definition. Their existence does not make
   the associated CFD force histories public.
2. The [Grinderslev Figure 6](https://wes.copernicus.org/articles/7/2201/2022/wes-7-2201-2022-f06-web.png)
   is already digitized locally in
   `examples/grinderslev2022_iea10/reference_data/p100_i50_figure6_power_digitized.csv`.
   Use its uncertainty-bearing points to test extraction of work coefficients
   and predict the damped response. Preserve its status as figure-derived
   CFD evidence, not author-supplied time histories.
3. The [ExaWind benchmark repository](https://github.com/Exawind/exawind-benchmarks)
   cited by Fu currently redirects to
   [kynema/kynema-benchmarks](https://github.com/kynema/kynema-benchmarks).
   It provides an open implementation route; choose exact revisions before
   building cases. This review did not execute those remote cases.
4. [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) offers 2D/3D
   incompressible moving-boundary CFD and force extraction, with
   [public examples](https://github.com/WaterLily-jl/WaterLily-Examples).
   It is a candidate for low-Reynolds-number methodological demonstrations.
   A cheap 2D calculation would not qualify a high-Reynolds-number blade map.
5. The current repository already has
   `vorlap/qblade_airfoil_work_map.py`,
   `examples/ehrmann2014_naca0018/build_airfoil_work_map.py`, and
   `examples/ehrmann2014_naca0018/work_map_data_request.json`.
   The frozen request specifies 3 Reynolds numbers, 4 reduced frequencies,
   and 4 nonzero amplitudes: **48 paired prescribed-motion points per
   fidelity**, plus 12 analytical zero-amplitude vertices. It also contains
   a Nalu-Wind ALE recipe and source identity requirements. This is a useful
   pilot at one incidence and motion direction, not a generic airfoil atlas.

The bounded literature search found no directly downloadable, target-matched
NACA0018 synchronized force-motion atlas that closes that request. This is
not a claim that none exists. The missing asset is motion-conditioned signed
work over a sufficient parameter domain, plus independent experimental
assessment. Stationary airfoil datasets and the whole-blade power curve do
not close that gap.

## Feasible development path and acceptance claim

First verify the generic calculation using analytic modal projection and
the published whole-blade power points. Next populate the existing work-map
interface from prescribed-motion CFD, freeze its domain and interpolation,
and predict changes in structural damping/mass without refitting. Assess
unseen CFD motions separately from unseen experiments. Use nonlinear work
tables when a two-coefficient law cannot reproduce the roots and slopes.

An eventual reusable map must expose at least profile geometry, incidence,
Reynolds number, reduced frequency, amplitude, and motion direction. For a
blade, spanwise inflow, taper/twist, tip geometry, and coherence need either
explicit inputs or demonstrated restrictions. Structural damping uncertainty
must remain visible; FE mass and stiffness do not independently identify
material/joint damping.

The defensible present conclusion is **feasible as a generic framework with
model-derived aerodynamic maps and structural projection; unproven as a
universal predictive coefficient model matching experimental calibration**.
Experimental data can be reserved for validation, but equivalent accuracy
must be demonstrated on held-out conditions rather than inferred from a
successful fit to CFD or an experimentally calibrated wake oscillator.
