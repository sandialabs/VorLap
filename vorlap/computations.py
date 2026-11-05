"""Backward-compatible computation API for focused VorLap physics modules."""

from .force_model import (
    compute_local_segment_length as _compute_local_segment_length,
    force_from_chord_normal as _force_from_chord_normal,
    resolve_airfoil as _resolve_airfoil,
    strouhal_reference_length as _strouhal_reference_length,
)
from .geometry_frames import (
    as_valid_frame_array as _as_valid_frame_array,
    build_fallback_frame as _build_fallback_frame,
    normalize as _normalize,
    pick_reference_axis as _pick_reference_axis,
    prepare_component_frames,
    rotate_point_about_axis as _rotate_point_about_axis,
    rotate_vector,
    rotate_vectors_about_axes as _rotate_vectors_about_axes,
    rotationMatrix,
    segment_span_vectors as _segment_span_vectors,
    validate_component_geometry_arrays as _validate_component_geometry_arrays,
)
from .spectral_reconstruction import (
    _smooth_time_series_adaptive,
    reconstruct_nonstationary_signal,
    reconstruct_signal,
)
from .stationary_sweep import (
    _compute_thrust_torque_spectrum_impl,
    _finite_1d,
    _time_vector,
    compute_thrust_torque_spectrum,
    compute_thrust_torque_spectrum_optimized,
)
from .time_varying_history import (
    _compute_time_varying_force_history_impl,
    coerce_inflow_profile as _coerce_inflow_profile,
    compute_time_varying_force_history,
    compute_time_varying_force_history_optimized,
)


_EPS = 1.0e-12


def interpolate_fft_spectrum(afft, Re_val, AOA_val, field, n_freq_depth=None):
    """
    Compatibility wrapper for the interpolation module.
    """
    from .interpolation import interpolate_fft_spectrum as _interpolate_fft_spectrum

    return _interpolate_fft_spectrum(afft, Re_val, AOA_val, field, n_freq_depth)


__all__ = [
    "compute_time_varying_force_history",
    "compute_time_varying_force_history_optimized",
    "compute_thrust_torque_spectrum",
    "compute_thrust_torque_spectrum_optimized",
    "interpolate_fft_spectrum",
    "prepare_component_frames",
    "reconstruct_nonstationary_signal",
    "reconstruct_signal",
    "rotate_vector",
    "rotationMatrix",
]
