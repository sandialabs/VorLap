# Mannini 3:2 rectangular-cylinder independent VIV screen

This example is the first cross-profile experimental assessment of the
Ehrmann-trained engineering negative-damping closure. The source is the
spring-mounted `B/D=1.5` rectangular cylinder tested by Mannini, Marra,
Massai, and Bartoli. `B=116 mm` is streamwise, `D=77 mm` is cross-flow, and
the span is `986 mm`.

The assessment deliberately starts with test case 6 (`Sc=87.7`). The source
identifies it as the case in which the VIV and galloping ranges are separated,
so its resonance peak is a clean test of the VIV saturation law rather than a
fit to the low-damping VIV--galloping interaction.

The model is not fitted here. The static-body `St=0.106` and
sinusoidal-equivalent `CL0=0.656` values are admissible model inputs, but the
response curve is used only after the Ehrmann calibration is frozen. The
first gate is analytical: the signed closure changes from negative to positive
aerodynamic damping at its limiting amplitude. Consequently, its isolated
stable negative-damping branch cannot exceed that amplitude, regardless of
the spectral-force magnitude. This bound is compared with the independently
digitized experimental peak before any rectangular-cylinder QBlade campaign
is generated.

Reproduce the assessment with:

```bash
python -m examples.mannini2016_rectangular_3x2.assess_frozen_model
```

An assessment failure is retained as training evidence; it is not relabeled
as validation. The next model revision and next untouched experimental study
are controlled by `independent_viv_validation_todo.md`.
