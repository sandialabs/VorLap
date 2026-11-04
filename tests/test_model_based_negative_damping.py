"""Independent physical checks for the opt-in prescribed-work reduction."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vorlap.model_based_negative_damping import (
    QuadraticWorkLaw,
    StructuralMode,
    fit_quadratic_work_law,
    project_sectional_work,
    project_structural_mode,
    sectional_coefficients_from_work,
    spectral_modal_basis,
)
from vorlap.qblade_modal_power_model import (
    QBladeModalPowerModel,
    QBladeModalPowerModelConfig,
)


def _structural_system():
    """Coupled physical coordinates with independently prescribed modes."""
    mass = np.diag([2.0, 3.0])
    mass_sqrt = np.diag(np.sqrt([2.0, 3.0]))
    angle = 0.4
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    frequencies = np.array([3.0, 7.0])
    stiffness = mass_sqrt @ rotation @ np.diag(frequencies**2) @ rotation.T @ mass_sqrt
    damping = 0.1 * mass + 0.02 * stiffness
    shape = np.linalg.solve(mass_sqrt, rotation[:, 0])
    return mass, stiffness, damping, shape


def test_matrix_projection_preserves_known_frequency_damping_and_energy():
    mass, stiffness, damping, shape = _structural_system()
    mode = project_structural_mode(mass, stiffness, damping, shape)
    assert mode.mass_kg == pytest.approx(1.0)
    assert mode.omega_radps == pytest.approx(3.0)
    assert mode.damping_ratio == pytest.approx(0.1 / 6.0 + 0.02 * 3.0 / 2.0)
    assert mode.eigen_residual < 1e-14

    time = np.linspace(0.0, 2.0 * np.pi / 3.0, 1001)
    amplitude = 0.7
    displacement = shape[:, None] * amplitude * np.sin(3.0 * time)
    velocity = shape[:, None] * 3.0 * amplitude * np.cos(3.0 * time)
    energy = (np.einsum("it,ij,jt->t", velocity, mass, velocity)
              + np.einsum("it,ij,jt->t", displacement, stiffness, displacement)) / 2.0
    np.testing.assert_allclose(energy, 0.5 * 9.0 * amplitude**2, rtol=1e-14)
    dissipated = np.einsum("it,ij,jt->t", velocity, damping, velocity)
    mean_power = np.trapezoid(dissipated, time) / time[-1]
    assert mean_power == pytest.approx(mode.power_coefficient_wpm2 * amplitude**2)


@pytest.mark.parametrize("scale", [-4.0, 0.25, 7.0])
def test_modal_normalization_preserves_physical_amplitude_and_engineering_gain(scale):
    mass, stiffness, damping, shape = _structural_system()
    mode = project_structural_mode(mass, stiffness, damping, shape)
    scaled_mode = project_structural_mode(mass, stiffness, damping, scale * shape)
    np.testing.assert_allclose(
        [scaled_mode.mass_kg, scaled_mode.stiffness_npm, scaled_mode.damping_kgps],
        scale**2 * np.array([mode.mass_kg, mode.stiffness_npm, mode.damping_kgps]))
    assert scaled_mode.damping_ratio == pytest.approx(mode.damping_ratio)
    common = dict(zero_gain_kgps=[10.0, 20.0],
                  saturation_coefficient_kgpspm2=[3.0, 4.0],
                  omega_radps=mode.omega_radps,
                  maximum_local_amplitude_m=[5.0, 5.0])
    law = project_sectional_work(mode_shape=shape, **common)
    scaled_law = project_sectional_work(mode_shape=scale * shape, **common)
    amplitude = law.equilibrium_amplitude_m(mode.damping_kgps)
    scaled_amplitude = scaled_law.equilibrium_amplitude_m(scaled_mode.damping_kgps)
    np.testing.assert_allclose(np.abs(shape * amplitude),
                               np.abs(scale * shape * scaled_amplitude))
    assert scaled_law.power_w(scaled_amplitude) == pytest.approx(law.power_w(amplitude))
    spectral = dict(density_kgpm3=1.2, speed_mps=[10.0, 11.0],
                    chord_m=[0.5, 0.7], span_weights_m=[0.3, 0.4],
                    captured_coefficient=[0.1, 0.2])
    basis = spectral_modal_basis(mode_shape=shape, **spectral)
    scaled_basis = spectral_modal_basis(mode_shape=scale * shape, **spectral)
    assert scaled_basis == pytest.approx(scale**2 * basis)
    parameters = law.engineering_parameters(spectral_basis_kgps=basis,
        reference_length_m=1.0, mode_shape_max_abs=float(np.max(abs(shape))))
    scaled_parameters = scaled_law.engineering_parameters(
        spectral_basis_kgps=scaled_basis,
        reference_length_m=1.0, mode_shape_max_abs=float(np.max(abs(scale * shape))))
    assert scaled_parameters == pytest.approx(parameters)


@pytest.mark.parametrize("change,match", [
    ("mass_indefinite", "positive definite"),
    ("stiffness_indefinite", "positive semidefinite"),
    ("active_damping", "positive semidefinite"),
    ("asymmetric_damping", "symmetric"),
    ("not_eigenmode", "not an eigenmode"),
    ("zero_mode", "projected mass"),
    ("nan_mode", "finite"),
])
def test_structural_projection_rejects_invalid_energy_or_eigenmode(change, match):
    mass, stiffness, damping, shape = _structural_system()
    if change == "mass_indefinite":
        mass = np.diag([2.0, -3.0])
    elif change == "stiffness_indefinite":
        stiffness = -stiffness
    elif change == "active_damping":
        damping = -damping
    elif change == "asymmetric_damping":
        damping[0, 1] += 1.0
    elif change == "not_eigenmode":
        shape = np.ones(2)
    elif change == "zero_mode":
        shape = np.zeros(2)
    else:
        shape[0] = np.nan
    with pytest.raises(ValueError, match=match):
        project_structural_mode(mass, stiffness, damping, shape)


def test_sectional_work_conversion_and_projection_match_harmonic_nodal_work():
    # Include a locally dissipative strip and a motion node: neither should
    # be clipped or counted as an actively moving area.
    rho, omega, amplitude = 1.225, 4.0, 0.45
    speed = np.array([8.0, 11.0, 9.0])
    chord = np.array([0.7, 0.5, 0.6])
    span = np.array([0.3, 0.4, 0.5])
    shape = np.array([0.5, -1.0, 0.0])
    p2, p4 = np.array([-0.02, 0.1, 0.5]), np.array([0.1, 0.2, 0.3])
    k = omega * chord / (2.0 * speed)
    b0, b2 = sectional_coefficients_from_work(
        power_quadratic_cp=p2, power_quartic_cp=p4,
        reduced_frequency_k=k, density_kgpm3=rho, speed_mps=speed,
        chord_m=chord, span_weights_m=span)
    law = project_sectional_work(zero_gain_kgps=b0,
        saturation_coefficient_kgpspm2=b2, mode_shape=shape,
        omega_radps=omega, maximum_local_amplitude_m=np.ones(3))
    time = np.linspace(0.0, 2.0 * np.pi / omega, 2001)
    velocity = shape[:, None] * amplitude * omega * np.cos(omega * time)
    local_amplitude = np.abs(shape) * amplitude
    force = (b0 - b2 * local_amplitude**2)[:, None] * velocity
    actual_section_power = np.trapezoid(force * velocity, time, axis=1) / time[-1]
    ratio = local_amplitude / chord
    prescribed_section_power = 0.5 * rho * speed**3 * chord * span * (
        p2 * ratio**2 - p4 * ratio**4)
    np.testing.assert_allclose(actual_section_power, prescribed_section_power, atol=1e-12)
    assert actual_section_power[0] < 0.0
    assert actual_section_power[2] == 0.0
    assert sum(actual_section_power) == pytest.approx(law.power_w(amplitude))


def test_sectional_dimensionalization_uses_span_integral_and_doubles_with_density():
    inputs = dict(power_quadratic_cp=[0.1], power_quartic_cp=[0.3],
                  reduced_frequency_k=[0.2], speed_mps=[10.0],
                  chord_m=[0.5], span_weights_m=[2.0])
    b0, b2 = sectional_coefficients_from_work(density_kgpm3=1.0, **inputs)
    doubled = sectional_coefficients_from_work(density_kgpm3=2.0, **inputs)
    np.testing.assert_allclose(np.array(doubled), 2.0 * np.array([b0, b2]))
    split = sectional_coefficients_from_work(density_kgpm3=1.0,
        **{key: ([value[0], value[0]] if key != "span_weights_m" else [1.0, 1.0])
           for key, value in inputs.items()})
    np.testing.assert_allclose([sum(split[0]), sum(split[1])], [b0[0], b2[0]])


def test_work_fit_recovers_known_harmonic_coefficients_across_amplitude_scales():
    omega = 3.2
    for scale in (1e-4, 1.0, 1e4):
        amplitude = scale * np.array([0.0, 0.1, 0.4, 0.8, 1.2, 1.5])
        power = 120.0 * (amplitude / scale)**2 - 60.0 * (amplitude / scale)**4
        fit = fit_quadratic_work_law(amplitude, power, omega_radps=omega)
        assert fit.law.limiting_amplitude_m == pytest.approx(np.sqrt(2.0) * scale)
        assert fit.law.zero_gain_kgps == pytest.approx(240.0 / (omega * scale)**2)
        assert fit.power_rmse_w < 1e-10
        assert fit.power_relative_l2_error < 1e-12
        assert fit.samples == amplitude.size
        assert fit.law.power_w(1.5 * scale) < 0.0
        with pytest.raises(ValueError, match="support"):
            fit.law.power_w(1.51 * scale)


def test_work_fit_reports_nonquadratic_error_without_silently_matching_targets():
    amplitude = np.linspace(0.0, 1.5, 21)
    power = 100.0 * amplitude**2 - 60.0 * amplitude**4 + 8.0 * amplitude**6
    fit = fit_quadratic_work_law(amplitude, power, omega_radps=4.0)
    fitted_power = np.array([fit.law.power_w(a) for a in amplitude])
    residual = fitted_power - power
    assert fit.power_rmse_w > 1.0
    assert fit.power_maximum_error_w == pytest.approx(max(abs(residual)))
    # A least-squares residual is orthogonal to the fitted basis; neither an
    # endpoint nor a selected equilibrium was forced to fit instead.
    assert residual @ amplitude**2 == pytest.approx(0.0, abs=1e-10)
    assert residual @ amplitude**4 == pytest.approx(0.0, abs=1e-10)
    assert abs(residual[-1]) > 1.0


@pytest.mark.parametrize("amplitudes,power", [
    ([0.0, 0.1, 0.2], [0.0, 1.0, 2.0]),
    ([0.0, 0.1, 0.1, 0.2], [0.0, 1.0, 1.0, 2.0]),
    ([0.0, 0.1, 0.2, 0.3], [1.0, 1.0, 2.0, 3.0]),
    ([0.0, 0.1, 0.2, 0.3], [0.0, -1.0, -2.0, -3.0]),
    ([0.0, 0.1, 0.2, np.nan], [0.0, 1.0, 2.0, 3.0]),
])
def test_unidentifiable_or_invalid_work_data_fail(amplitudes, power):
    with pytest.raises(ValueError):
        fit_quadratic_work_law(amplitudes, power, omega_radps=4.0)


def test_equilibrium_distinguishes_damping_from_aerodynamic_saturation_and_domain():
    law = QuadraticWorkLaw(8.0, 2.0, 3.0, 2.1)
    assert law.limiting_amplitude_m == 2.0
    assert law.equilibrium_amplitude_m(0.0) == 2.0
    assert law.equilibrium_amplitude_m(6.0) == 1.0
    assert law.equilibrium_amplitude_m(8.0) == 0.0
    assert law.equilibrium_amplitude_m(10.0) == 0.0
    truncated = replace(law, maximum_amplitude_m=1.5)
    with pytest.raises(ValueError, match="support"):
        truncated.equilibrium_amplitude_m(0.0)
    with pytest.raises(ValueError, match="support"):
        truncated.engineering_parameters(spectral_basis_kgps=1.0, reference_length_m=1.0,
                                         mode_shape_max_abs=1.0)
    with pytest.raises(ValueError):
        law.engineering_parameters(spectral_basis_kgps=0.0, reference_length_m=1.0,
                                   mode_shape_max_abs=1.0)


def test_external_target_requires_work_role_and_handles_actuator_normalization():
    law = QuadraticWorkLaw(160.0, 640.0, 2.0 * np.pi, 1.5)
    target = law.external_target(1.2, mode_shape_max_abs=2.0,
        work_role="incremental_high_fidelity_minus_runtime_base")
    assert target.target_cycle_mean_power_w == pytest.approx(law.power_w(0.6))
    assert target.target_cycle_mean_power_w < 0.0
    assert target.origin_equivalent_velocity_gain_kgps == 40.0
    replacement = law.external_target(1.2, mode_shape_max_abs=2.0,
                                      work_role="total_aerodynamics_replaced")
    assert replacement.target_cycle_mean_power_w == target.target_cycle_mean_power_w
    assert replacement.context_fingerprint != target.context_fingerprint
    with pytest.raises(ValueError, match="Declare"):
        law.external_target(0.2, mode_shape_max_abs=2.0, work_role="total_added_to_native")
    with pytest.raises(ValueError, match="support"):
        law.external_target(3.1, mode_shape_max_abs=2.0,
                            work_role="incremental_high_fidelity_minus_runtime_base")


@pytest.mark.parametrize("amplitude", [0.4, 1.3])
def test_external_target_realized_harmonic_nodal_work_preserves_both_signs(amplitude):
    omega = 2.0 * np.pi
    shape = np.array([0.2, 0.6, 1.0])
    law = QuadraticWorkLaw(160.0, 640.0, omega, 1.5)
    config = QBladeModalPowerModelConfig(
        node_ids=("a", "b", "c"), mode_shape=tuple(2.0 * shape),
        modal_mass_weights_kg=(2.0, 3.0, 4.0),
        aerodynamic_tributary_weights=(0.4, 0.8, 1.2),
        natural_frequency_hz=1.0, power_curve=None,
        rms_time_constant_s=2.0, max_abs_target_power_w=1e6,
        max_abs_generalized_force_n=1e6, max_abs_nodal_force_n=1e6,
        bandpass_damping_ratio=0.2, enabled=True,
        external_target_source_fingerprint="b" * 64)
    model = QBladeModalPowerModel(config)
    time = np.arange(0.0, 40.0 + 0.005, 0.01)
    power = []
    targets = []
    for t in time:
        velocity = np.zeros((3, 3))
        velocity[:, 0] = shape * amplitude * omega * np.cos(omega * t)
        # Orthogonal motion must not create work in the selected actuator.
        velocity[:, 1] = 0.13 * np.sin(1.7 * t)
        result = model.step(float(t), node_ids=config.node_ids,
            direction_basis=np.tile([1.0, 0.0, 0.0], (3, 1)),
            structural_velocity_mps=velocity,
            target_power_evaluator=lambda a: law.external_target(
                a, mode_shape_max_abs=2.0,
                work_role="incremental_high_fidelity_minus_runtime_base"))
        direct_work = float(np.sum(result.additive_force_n * velocity))
        assert direct_work == pytest.approx(result.diagnostics.instantaneous_structural_power_w)
        power.append(direct_work)
        targets.append(result.diagnostics.target_cycle_mean_power_w)
    retained = time >= 30.0
    mean_power = np.trapezoid(np.array(power)[retained], time[retained]) / 10.0
    expected = law.power_w(amplitude / 2.0)
    # This includes real bandpass and RMS-estimator dynamics rather than an
    # idealized force substitution; the settled finite-memory error is small.
    assert mean_power == pytest.approx(expected, rel=0.004)
    assert np.sign(mean_power) == np.sign(expected)
    assert np.sign(targets[-1]) == np.sign(expected)


def test_cycle_averaged_energy_evolution_converges_to_balance_with_correct_factor_two():
    law = QuadraticWorkLaw(8.0, 2.0, 3.0, 2.5)
    mode = StructuralMode(10.0, 90.0, 4.0, 0.0)
    initial = 0.1

    def amplitude_rate(_time, value):
        a = value[0]
        net_power = law.power_w(a) - mode.power_coefficient_wpm2 * a**2
        return [net_power / (mode.mass_kg * mode.omega_radps**2 * a)]

    time = np.linspace(0.0, 80.0, 101)
    result = solve_ivp(amplitude_rate, (0.0, time[-1]), [initial], t_eval=time,
                       rtol=1e-9, atol=1e-11)
    assert result.success
    # A² follows a logistic ODE; its exponential rate is (c0-cs)/M,
    # while small-amplitude A grows at half that rate.
    equilibrium_squared = 2.0
    analytic = np.sqrt(equilibrium_squared / (
        1.0 + (equilibrium_squared / initial**2 - 1.0) * np.exp(-0.4 * time)))
    np.testing.assert_allclose(result.y[0], analytic, rtol=2e-8, atol=1e-10)
    assert result.y[0, -1] == pytest.approx(law.equilibrium_amplitude_m(mode.damping_kgps), rel=1e-8)
