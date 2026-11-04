"""Offline reduction of structural models and prescribed-motion aerodynamic work.

This opt-in research interface does not change any frozen VorLap calibration.
Positive c_a in Q=c_a*qdot injects energy. Amplitudes are harmonic PEAK
envelopes, not instantaneous displacement or RMS. Thus
P(A)=0.5*omega**2*A**2*(c0-c2*A**2). Structural damping is separate.

The matrices and aerodynamic work must use the SAME generalized coordinate
q [m]. A supplied mode is never silently renormalized. FE rotational entries
must be power-conjugate to q as well. All aerodynamic coefficients describe
one declared flow, frequency, mean geometry, and motion pattern; no transfer
to a different operating point or extrapolation is implied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .qblade_modal_power_model import ExternalModalPowerTarget
from .qblade_engineering_lockin import semantic_sha256


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not boolean.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (not zero and result == 0.0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return result


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector.")
    return result


@dataclass(frozen=True)
class StructuralMode:
    """One real mode: modal kg, N/m, kg/s, with q measured in metres."""

    mass_kg: float
    stiffness_npm: float
    damping_kgps: float
    eigen_residual: float

    def __post_init__(self) -> None:
        _positive(self.mass_kg, "mass_kg")
        _positive(self.stiffness_npm, "stiffness_npm")
        _positive(self.damping_kgps, "damping_kgps", zero=True)
        _positive(self.eigen_residual, "eigen_residual", zero=True)

    @property
    def omega_radps(self) -> float:
        return math.sqrt(self.stiffness_npm / self.mass_kg)

    @property
    def damping_ratio(self) -> float:
        return self.damping_kgps / (2.0 * self.mass_kg * self.omega_radps)

    @property
    def power_coefficient_wpm2(self) -> float:
        return 0.5 * self.damping_kgps * self.omega_radps**2


def project_structural_mode(
    mass_matrix: Any,
    stiffness_matrix: Any,
    damping_matrix: Any,
    mode_shape: Any,
    *,
    eigen_residual_tolerance: float = 1.0e-6,
) -> StructuralMode:
    """Project M,K,C and reject non-modes or non-passive structural matrices.

    This is a real, isolated-mode reduction. Strong non-proportional damping,
    gyroscopic terms, and modal coupling need a multi-mode formulation.
    A supplied C is a model assumption unless independently characterized.
    """
    phi = _vector(mode_shape, "mode_shape")
    matrices = []
    for name, raw in (("mass", mass_matrix), ("stiffness", stiffness_matrix),
                      ("damping", damping_matrix)):
        matrix = np.asarray(raw, dtype=float)
        if matrix.shape != (phi.size, phi.size) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} matrix must be a finite square matrix matching the mode.")
        scale = max(float(np.linalg.norm(matrix, ord=2)), np.finfo(float).tiny)
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12 * scale):
            raise ValueError(f"{name} matrix must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues[0] < -1.0e-12 * scale or (name == "mass" and eigenvalues[0] <= 0.0):
            raise ValueError(f"{name} matrix must be {'positive definite' if name == 'mass' else 'positive semidefinite'}.")
        matrices.append(matrix)
    mass, stiffness, damping = matrices
    m, k, c = (float(phi @ matrix @ phi) for matrix in matrices)
    _positive(m, "projected mass")
    _positive(k, "projected stiffness")
    kphi = stiffness @ phi
    inertial = (k / m) * (mass @ phi)
    residual = float(np.linalg.norm(kphi - inertial) /
                     (np.linalg.norm(kphi) + np.linalg.norm(inertial)))
    tolerance = _positive(eigen_residual_tolerance, "eigen_residual_tolerance", zero=True)
    if residual > tolerance:
        raise ValueError(f"Supplied shape is not an eigenmode: relative residual {residual:.6g}.")
    return StructuralMode(m, k, max(c, 0.0), residual)


def spectral_modal_basis(
    *, density_kgpm3: float, speed_mps: Any, chord_m: Any,
    span_weights_m: Any, captured_coefficient: Any, mode_shape: Any,
) -> float:
    """B_spectral [kg/s] in the existing engineering closure's convention.

    captured_coefficient must already include force/motion direction and the
    existing spectral capture policy. It is harmonic RSS, not automatically
    an RMS coefficient. span_weights_m is physical quadrature, not a mean.
    """
    density = _positive(density_kgpm3, "density_kgpm3")
    speed, chord, span, coefficient, phi = (
        _vector(x, name) for x, name in (
            (speed_mps, "speed_mps"), (chord_m, "chord_m"),
            (span_weights_m, "span_weights_m"),
            (captured_coefficient, "captured_coefficient"), (mode_shape, "mode_shape")))
    if len({x.size for x in (speed, chord, span, coefficient, phi)}) != 1:
        raise ValueError("Section arrays must have equal length.")
    if np.any(speed < 0) or np.any(chord <= 0) or np.any(span < 0) or np.any(coefficient < 0):
        raise ValueError("Section speeds/weights/coefficients must be nonnegative; chords positive.")
    return float(np.sum(0.5 * density * speed * chord * span * coefficient * phi**2))


def sectional_coefficients_from_work(
    *, power_quadratic_cp: Any, power_quartic_cp: Any, reduced_frequency_k: Any,
    density_kgpm3: float, speed_mps: Any, chord_m: Any, span_weights_m: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert C_P=p2*(a/c)^2-p4*(a/c)^4 to b0,b2 in F=(b0-b2*a²)*v.

    C_P=P/(0.5*rho*U³*c*span); k=omega*c/(2U). b0 [kg/s],
    b2 [kg/(s m²)] include physical span quadrature. p2,p4 may be signed.
    Paired CFD-minus-runtime-base C_P must be supplied for supplemental use.
    """
    density = _positive(density_kgpm3, "density_kgpm3")
    p2, p4, k, speed, chord, span = (
        _vector(x, name) for x, name in (
            (power_quadratic_cp, "power_quadratic_cp"),
            (power_quartic_cp, "power_quartic_cp"),
            (reduced_frequency_k, "reduced_frequency_k"), (speed_mps, "speed_mps"),
            (chord_m, "chord_m"), (span_weights_m, "span_weights_m")))
    if len({x.size for x in (p2, p4, k, speed, chord, span)}) != 1:
        raise ValueError("Section arrays must have equal length.")
    if np.any(k <= 0) or np.any(speed <= 0) or np.any(chord <= 0) or np.any(span < 0):
        raise ValueError("k, speed, chord must be positive and span weights nonnegative.")
    scale = density * speed * chord * span / (4.0 * k**2)
    return scale * p2, scale * p4 / chord**2


@dataclass(frozen=True)
class QuadraticWorkLaw:
    """Signed c(A)=c0-c2*A²; parameters are aerodynamic, with bounded support."""

    zero_gain_kgps: float
    saturation_coefficient_kgpspm2: float
    omega_radps: float
    maximum_amplitude_m: float

    def __post_init__(self) -> None:
        for name in ("zero_gain_kgps", "saturation_coefficient_kgpspm2", "omega_radps",
                     "maximum_amplitude_m"):
            _positive(getattr(self, name), name)

    @property
    def limiting_amplitude_m(self) -> float:
        """Algebraic zero-work amplitude; may lie outside sampled support."""
        return math.sqrt(self.zero_gain_kgps / self.saturation_coefficient_kgpspm2)

    def amplitude(self, amplitude_m: float) -> float:
        amplitude = _positive(amplitude_m, "amplitude_m", zero=True)
        if amplitude > self.maximum_amplitude_m:
            raise ValueError("Requested amplitude is outside aerodynamic data support.")
        return amplitude

    def velocity_gain(self, amplitude_m: float) -> float:
        amplitude = self.amplitude(amplitude_m)
        return self.zero_gain_kgps - self.saturation_coefficient_kgpspm2 * amplitude**2

    def power_w(self, amplitude_m: float) -> float:
        amplitude = self.amplitude(amplitude_m)
        return 0.5 * self.omega_radps**2 * amplitude**2 * self.velocity_gain(amplitude)

    def equilibrium_amplitude_m(self, opposing_damping_kgps: float) -> float:
        """Stable origin or stable nonzero equilibrium; reject unsupported root.

        At c0=c_opposing the linear growth rate is zero but the negative cubic
        term attracts finite perturbations. No positive-part clipping of the
        aerodynamic force is used.
        """
        opposing = _positive(opposing_damping_kgps, "opposing_damping_kgps", zero=True)
        amplitude = math.sqrt(max(0.0, self.zero_gain_kgps - opposing) /
                              self.saturation_coefficient_kgpspm2)
        return self.amplitude(amplitude)

    def engineering_parameters(self, *, spectral_basis_kgps: float,
                               reference_length_m: float,
                               mode_shape_max_abs: float) -> dict[str, float]:
        """Return H and Lambda using the explicitly supplied legacy basis.

        B_spectral must use the same original shape as this law. L_ref is a
        physical length, and Lambda uses the actuator's unit-maximum motion
        amplitude. mode_shape_max_abs converts the generalized saturation
        amplitude to maximum physical displacement.

        These values are a research result, not a frozen-model schema artifact.
        A zero static spectral basis cannot represent nonzero prescribed work.
        """
        basis = _positive(spectral_basis_kgps, "spectral_basis_kgps")
        length = _positive(reference_length_m, "reference_length_m")
        scale = _positive(mode_shape_max_abs, "mode_shape_max_abs")
        self.amplitude(self.limiting_amplitude_m)
        return {"feedback_gain": self.zero_gain_kgps / basis,
                "limiting_amplitude_ratio": scale * self.limiting_amplitude_m / length}

    def external_target(self, amplitude_m: float, *, work_role: str,
                        mode_shape_max_abs: float) -> ExternalModalPowerTarget:
        """Feed signed work into the existing opt-in modal actuator.

        amplitude_m is in the actuator's unit-maximum translational mode
        coordinate. mode_shape_max_abs is the maximum absolute translational
        entry of the shape used to derive THIS law. The actuator normalizes
        its supplied shape, so q_actuator=s*q_original, c0_actuator=c0/s².
        The scalar actuator does not support mixed rotational DOFs.

        Caller must assert paired base subtraction, or that all aerodynamic
        work is being replaced. No automatic base-work subtraction is possible
        from structural damping or a static spectrum.
        """
        if work_role not in {"incremental_high_fidelity_minus_runtime_base",
                             "total_aerodynamics_replaced"}:
            raise ValueError("Declare paired incremental work or replacement of total aerodynamics.")
        scale = _positive(mode_shape_max_abs, "mode_shape_max_abs")
        amplitude = _positive(amplitude_m, "amplitude_m", zero=True)
        return ExternalModalPowerTarget(
            target_cycle_mean_power_w=self.power_w(amplitude / scale),
            origin_equivalent_velocity_gain_kgps=self.zero_gain_kgps / scale**2,
            context_fingerprint=semantic_sha256({"law": asdict(self), "work_role": work_role,
                                                "mode_shape_max_abs": scale,
                                                "actuator_amplitude_m": amplitude}),
        )


def project_sectional_work(
    *, zero_gain_kgps: Any, saturation_coefficient_kgpspm2: Any, mode_shape: Any,
    omega_radps: float, maximum_local_amplitude_m: Any,
) -> QuadraticWorkLaw:
    """Modalize local PEAK-envelope work; c0=Σb0*phi², c2=Σb2*phi⁴.

    All sections must describe the supplied common omega. Mode shape changes
    effective Lambda but does not determine the sectional aerodynamic b0,b2.
    Spanwise coupling is absent from this strip approximation.
    """
    b0, b2, phi, bounds = (_vector(x, name) for x, name in (
        (zero_gain_kgps, "zero_gain_kgps"),
        (saturation_coefficient_kgpspm2, "saturation_coefficient_kgpspm2"),
        (mode_shape, "mode_shape"), (maximum_local_amplitude_m, "maximum_local_amplitude_m")))
    if len({x.size for x in (b0, b2, phi, bounds)}) != 1 or np.any(bounds <= 0):
        raise ValueError("Section arrays must have equal length and positive amplitude bounds.")
    active = phi != 0.0
    if not np.any(active):
        raise ValueError("The mode must have nonzero motion.")
    return QuadraticWorkLaw(float(np.sum(b0 * phi**2)), float(np.sum(b2 * phi**4)),
                            omega_radps, float(np.min(bounds[active] / np.abs(phi[active]))))


@dataclass(frozen=True)
class WorkLawFit:
    law: QuadraticWorkLaw
    power_rmse_w: float
    power_relative_l2_error: float
    power_maximum_error_w: float
    samples: int


def fit_quadratic_work_law(
    amplitude_m: Any, power_w: Any, *, omega_radps: float,
) -> WorkLawFit:
    """Equal absolute-power least squares of P=p2*A²-p4*A⁴, without response data.

    Scale the amplitude basis for numerical conditioning. Neither endpoint nor
    onset is forced to fit. Residuals must be assessed before adopting this
    reduction; fitting success does not qualify a model. No zero-motion row
    is treated as phase information. Finite c0 is a quadratic-origin model
    assumption, not an observed A->0 derivative.
    """
    a = _vector(amplitude_m, "amplitude_m")
    p = _vector(power_w, "power_w")
    omega = _positive(omega_radps, "omega_radps")
    if a.shape != p.shape or np.any(a < 0) or np.any(np.diff(a) <= 0):
        raise ValueError("Amplitude and power must match; amplitudes must increase and be nonnegative.")
    if np.count_nonzero(a > 0) < 3:
        raise ValueError("At least three distinct nonzero amplitudes are needed to assess reduction error.")
    if a[0] == 0.0 and p[0] != 0.0:
        raise ValueError("Zero motion must have exactly zero work.")
    scale = float(a[-1])
    x = a / scale
    design = np.column_stack((x**2, -x**4))
    coefficients, _, rank, _ = np.linalg.lstsq(design, p, rcond=None)
    if rank != 2 or np.any(coefficients <= 0.0):
        raise ValueError("Power samples do not identify a positive-gain saturating quadratic law.")
    prediction = design @ coefficients
    error = prediction - p
    law = QuadraticWorkLaw(2.0 * coefficients[0] / (omega**2 * scale**2),
                           2.0 * coefficients[1] / (omega**2 * scale**4), omega, scale)
    return WorkLawFit(law, float(np.sqrt(np.mean(error**2))),
                      float(np.linalg.norm(error) / np.linalg.norm(p)),
                      float(np.max(np.abs(error))), int(a.size))
