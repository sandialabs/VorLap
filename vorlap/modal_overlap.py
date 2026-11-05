"""Frequency-and-amplitude screening of VorLap forcing against structural modes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .force_model import (
    compute_local_segment_length,
    resolve_airfoil,
    section_flow_state,
    strouhal_reference_length,
)
from .geometry_frames import normalize, prepare_component_frames, rotate_vectors_about_axes
from .interpolation import interpolate_fft_spectrum_batch
from .structs import AirfoilFFT, Component


_EPS = 1.0e-12


def _finite_vector(values: Sequence[float], name: str, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain positive values.")
    return array


def _flatten_components(
    components: Sequence[Component], node_ids: Sequence[str] | None
) -> dict[str, Any]:
    prepared = prepare_component_frames(list(components))
    count = sum(int(np.asarray(component.shape_xyz).shape[0]) for component in prepared)
    if node_ids is not None and len(node_ids) != count:
        raise ValueError(f"node_ids has {len(node_ids)} entries; expected {count}.")

    positions = np.empty((count, 3), dtype=float)
    chords = np.empty(count, dtype=float)
    thicknesses = np.empty(count, dtype=float)
    local_lengths = np.empty(count, dtype=float)
    chord_vectors = np.empty((count, 3), dtype=float)
    normal_vectors = np.empty((count, 3), dtype=float)
    component_ids: list[str] = []
    airfoil_ids: list[str] = []
    generated_node_ids: list[str] = []

    cursor = 0
    for component in prepared:
        xyz = np.asarray(component.shape_xyz, dtype=float)
        npoints = xyz.shape[0]
        slc = slice(cursor, cursor + npoints)
        positions[slc] = np.asarray(component.shape_xyz_global, dtype=float)
        chords[slc] = np.asarray(component.chord, dtype=float)
        thicknesses[slc] = np.asarray(component.thickness, dtype=float)
        chord_vectors[slc] = np.asarray(component.chord_vector, dtype=float)
        normal_vectors[slc] = np.asarray(component.normal_vector, dtype=float)
        for index in range(npoints):
            local_lengths[cursor + index] = compute_local_segment_length(xyz, index)
            component_ids.append(str(component.id))
            airfoil_ids.append(str(component.airfoil_ids[index]))
            generated_node_ids.append(f"{component.id}_{index + 1}")
        cursor += npoints

    return {
        "positions": positions,
        "chords": chords,
        "thicknesses": thicknesses,
        "local_lengths": local_lengths,
        "chord_vectors": chord_vectors,
        "normal_vectors": normal_vectors,
        "component_ids": tuple(component_ids),
        "airfoil_ids": tuple(airfoil_ids),
        "node_ids": tuple(str(value) for value in node_ids) if node_ids is not None else tuple(generated_node_ids),
    }


def _candidate_record(
    *,
    wind_speed_mps: float,
    azimuth_deg: float,
    node_index: int,
    depth_index: int,
    mode_index: int,
    node_ids: Sequence[str],
    component_ids: Sequence[str],
    forcing_frequency_hz: float,
    natural_frequency_hz: float,
    amplitude_coefficient: float,
    section_force_amplitude_n: float,
    detuning_fraction: float,
    overlap_kernel: float,
    aoa_deg: float,
    reynolds: float,
    effective_velocity_mps: float,
    source_clipped: bool,
) -> dict[str, Any]:
    return {
        "wind_speed_mps": float(wind_speed_mps),
        "azimuth_deg": float(azimuth_deg),
        "node_id": str(node_ids[node_index]),
        "component_id": str(component_ids[node_index]),
        "source_depth_index": int(depth_index),
        "natural_mode_index": int(mode_index + 1),
        "natural_frequency_hz": float(natural_frequency_hz),
        "forcing_frequency_hz": float(forcing_frequency_hz),
        "relative_detuning": float(detuning_fraction),
        "detuning_percent": float(100.0 * detuning_fraction),
        "amplitude_coefficient": float(amplitude_coefficient),
        "section_force_amplitude_n": float(section_force_amplitude_n),
        "overlap_kernel": float(overlap_kernel),
        "overlap_score_n": float(section_force_amplitude_n * overlap_kernel),
        "aoa_deg": float(aoa_deg),
        "reynolds": float(reynolds),
        "effective_velocity_mps": float(effective_velocity_mps),
        "source_clipped": bool(source_clipped),
    }


def screen_modal_overlap(
    components: Sequence[Component],
    airfoils: Mapping[str, AirfoilFFT],
    natural_frequencies_hz: Sequence[float],
    wind_speeds_mps: Sequence[float],
    azimuths_deg: Sequence[float],
    *,
    node_ids: Sequence[str] | None = None,
    rpm: float = 1.0,
    inflow_vector: Sequence[float] = (1.0, 0.0, 0.0),
    rotation_axis: Sequence[float] = (0.0, 0.0, 1.0),
    rotation_axis_offset: Sequence[float] = (0.0, 0.0, 0.0),
    fluid_density_kg_m3: float = 1.0296,
    fluid_kinematic_viscosity_m2_s: float = 1.647e-5,
    n_freq_depth: int = 20,
    amplitude_coefficient_cutoff: float = 0.002,
    source_frequency_cap_hz: float | None = None,
    overlap_band_fraction: float = 0.05,
) -> list[dict[str, Any]]:
    """Screen AmpCF/STCF tones against modes over mean wind speed and azimuth.

    The score is nodal harmonic-force amplitude multiplied by a Gaussian
    frequency-proximity kernel. It is a ranking metric, not a response or
    lock-in prediction: mode shapes, generalized modal force, damping, and
    aeroelastic feedback are not included.
    """
    if not components:
        raise ValueError("At least one component is required.")
    if not airfoils:
        raise ValueError("At least one airfoil FFT dataset is required.")
    modes = _finite_vector(natural_frequencies_hz, "natural_frequencies_hz", positive=True)
    winds = _finite_vector(wind_speeds_mps, "wind_speeds_mps", positive=True)
    azimuths = _finite_vector(azimuths_deg, "azimuths_deg")
    if int(n_freq_depth) < 2:
        raise ValueError("n_freq_depth must be at least 2 so an alternating tone is available.")
    if not math.isfinite(float(rpm)) or float(rpm) < 0.0:
        raise ValueError("rpm must be finite and nonnegative.")
    if not math.isfinite(float(amplitude_coefficient_cutoff)) or float(amplitude_coefficient_cutoff) < 0.0:
        raise ValueError("amplitude_coefficient_cutoff must be finite and nonnegative.")
    if not math.isfinite(float(overlap_band_fraction)) or float(overlap_band_fraction) <= 0.0:
        raise ValueError("overlap_band_fraction must be finite and positive.")
    if source_frequency_cap_hz is not None and (
        not math.isfinite(float(source_frequency_cap_hz)) or float(source_frequency_cap_hz) <= 0.0
    ):
        raise ValueError("source_frequency_cap_hz must be finite and positive when provided.")
    density = float(fluid_density_kg_m3)
    kinematic_viscosity = float(fluid_kinematic_viscosity_m2_s)
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("fluid_density_kg_m3 must be finite and positive.")
    if not math.isfinite(kinematic_viscosity) or kinematic_viscosity <= 0.0:
        raise ValueError("fluid_kinematic_viscosity_m2_s must be finite and positive.")

    inflow_unit = normalize(np.asarray(inflow_vector, dtype=float), "inflow_vector")
    axis_unit = normalize(np.asarray(rotation_axis, dtype=float), "rotation_axis")
    axis_offset = np.asarray(rotation_axis_offset, dtype=float).reshape(3)
    if np.any(~np.isfinite(axis_offset)):
        raise ValueError("rotation_axis_offset must contain finite values.")

    arrays = _flatten_components(components, node_ids)
    positions = arrays["positions"]
    chords = arrays["chords"]
    thicknesses = arrays["thicknesses"]
    local_lengths = arrays["local_lengths"]
    base_chord = arrays["chord_vectors"]
    base_normal = arrays["normal_vectors"]
    component_ids = arrays["component_ids"]
    flattened_node_ids = arrays["node_ids"]
    airfoil_ids = arrays["airfoil_ids"]
    node_count = len(flattened_node_ids)
    repeated_axes = np.repeat(axis_unit[None, :], node_count, axis=0)
    omega_vector = axis_unit * (float(rpm) * 2.0 * math.pi / 60.0)

    warned_missing: set[str] = set()
    grouped: list[tuple[np.ndarray, AirfoilFFT, int]] = []
    for airfoil_id in dict.fromkeys(airfoil_ids):
        indices = np.asarray([i for i, value in enumerate(airfoil_ids) if value == airfoil_id], dtype=int)
        airfoil = resolve_airfoil(dict(airfoils), airfoil_id, warned_missing_airfoils=warned_missing)
        available_depth = int(np.asarray(airfoil.CF_ST).shape[2])
        grouped.append((indices, airfoil, min(int(n_freq_depth), available_depth)))

    dynamic_viscosity = density * kinematic_viscosity
    wind_count = len(winds)
    best_overlap_scores = np.full(wind_count, -np.inf, dtype=float)
    best_detuning = np.full(wind_count, np.inf, dtype=float)
    best_overlap_records: list[dict[str, Any] | None] = [None] * wind_count
    best_match_records: list[dict[str, Any] | None] = [None] * wind_count
    overlap_candidate_counts = np.zeros(wind_count, dtype=np.int64)
    source_clip_queries = np.zeros(wind_count, dtype=np.int64)
    total_source_queries = np.zeros(wind_count, dtype=np.int64)

    relative_positions = positions - axis_offset[None, :]
    for azimuth in azimuths:
        angles = np.full(node_count, float(azimuth), dtype=float)
        chord_rotated = rotate_vectors_about_axes(base_chord, repeated_axes, angles)
        normal_rotated = rotate_vectors_about_axes(base_normal, repeated_axes, angles)
        positions_rotated = axis_offset[None, :] + rotate_vectors_about_axes(
            relative_positions, repeated_axes, angles
        )
        structural_velocity = np.cross(
            np.repeat(omega_vector[None, :], node_count, axis=0),
            positions_rotated - axis_offset[None, :],
        )
        relative_velocity = (
            winds[:, None, None] * inflow_unit[None, None, :]
            - structural_velocity[None, :, :]
        )
        flow = section_flow_state(
            relative_velocity,
            chord_rotated[None, :, :],
            normal_rotated[None, :, :],
            chords[None, :],
            local_lengths[None, :],
            density,
            dynamic_viscosity,
        )
        reference_lengths = strouhal_reference_length(
            chords[None, :], flow.aoa_deg, thicknesses[None, :]
        )

        for indices, airfoil, depth in grouped:
            re = np.asarray(flow.reynolds[:, indices], dtype=float)
            aoa = np.asarray(flow.aoa_deg[:, indices], dtype=float)
            clipped_nodes = (
                (re < float(np.min(airfoil.Re)))
                | (re > float(np.max(airfoil.Re)))
                | (aoa < float(np.min(airfoil.AOA)))
                | (aoa > float(np.max(airfoil.AOA)))
            )
            source_clip_queries += np.count_nonzero(clipped_nodes, axis=1)
            total_source_queries += int(indices.size)
            st_cf, amp_cf, _phase_cf = interpolate_fft_spectrum_batch(
                airfoil,
                re,
                aoa,
                "CF",
                n_freq_depth=depth,
                bounds_policy="clip",
            )
            group_count = len(indices)
            st_cf = st_cf.reshape(wind_count, group_count, depth)
            amp_cf = amp_cf.reshape(wind_count, group_count, depth)
            frequencies = st_cf * (
                np.asarray(flow.effective_velocity_mps[:, indices], dtype=float)
                / reference_lengths[:, indices]
            )[:, :, None]
            amplitudes = np.abs(amp_cf)
            force_amplitudes = amplitudes * np.asarray(
                flow.section_dynamic_force_n[:, indices], dtype=float
            )[:, :, None]

            frequencies = frequencies[:, :, 1:]
            amplitudes = amplitudes[:, :, 1:]
            force_amplitudes = force_amplitudes[:, :, 1:]
            depth_indices = np.arange(1, depth, dtype=int)
            valid = (
                np.isfinite(frequencies)
                & np.isfinite(amplitudes)
                & np.isfinite(force_amplitudes)
                & (frequencies > 0.0)
                & (amplitudes >= float(amplitude_coefficient_cutoff))
            )
            if source_frequency_cap_hz is not None:
                valid &= frequencies <= float(source_frequency_cap_hz)
            if not np.any(valid):
                continue

            detuning = (
                np.abs(frequencies[:, :, :, None] - modes[None, None, None, :])
                / modes[None, None, None, :]
            )
            valid_modes = valid[:, :, :, None]
            kernel = np.exp(-0.5 * (detuning / float(overlap_band_fraction)) ** 2)
            scores = np.where(valid_modes, force_amplitudes[:, :, :, None] * kernel, -np.inf)
            detuning_valid = np.where(valid_modes, detuning, np.inf)
            overlap_candidate_counts += np.count_nonzero(
                detuning_valid <= overlap_band_fraction, axis=(1, 2, 3)
            )

            score_flat = scores.reshape(wind_count, -1)
            score_argmax = np.argmax(score_flat, axis=1)
            score_max = score_flat[np.arange(wind_count), score_argmax]
            match_flat = detuning_valid.reshape(wind_count, -1)
            match_argmin = np.argmin(match_flat, axis=1)
            match_min = match_flat[np.arange(wind_count), match_argmin]

            for wind_index, wind_speed in enumerate(winds):
                if np.isfinite(score_max[wind_index]) and score_max[wind_index] > best_overlap_scores[wind_index]:
                    local_node, local_depth, mode_index = (
                        int(value)
                        for value in np.unravel_index(int(score_argmax[wind_index]), scores.shape[1:])
                    )
                    node_index = int(indices[local_node])
                    best_overlap_scores[wind_index] = float(score_max[wind_index])
                    best_overlap_records[wind_index] = _candidate_record(
                        wind_speed_mps=float(wind_speed),
                        azimuth_deg=float(azimuth),
                        node_index=node_index,
                        depth_index=int(depth_indices[local_depth]),
                        mode_index=mode_index,
                        node_ids=flattened_node_ids,
                        component_ids=component_ids,
                        forcing_frequency_hz=float(frequencies[wind_index, local_node, local_depth]),
                        natural_frequency_hz=float(modes[mode_index]),
                        amplitude_coefficient=float(amplitudes[wind_index, local_node, local_depth]),
                        section_force_amplitude_n=float(force_amplitudes[wind_index, local_node, local_depth]),
                        detuning_fraction=float(detuning[wind_index, local_node, local_depth, mode_index]),
                        overlap_kernel=float(kernel[wind_index, local_node, local_depth, mode_index]),
                        aoa_deg=float(flow.aoa_deg[wind_index, node_index]),
                        reynolds=float(flow.reynolds[wind_index, node_index]),
                        effective_velocity_mps=float(flow.effective_velocity_mps[wind_index, node_index]),
                        source_clipped=bool(clipped_nodes[wind_index, local_node]),
                    )

                if np.isfinite(match_min[wind_index]) and match_min[wind_index] < best_detuning[wind_index]:
                    local_node, local_depth, mode_index = (
                        int(value)
                        for value in np.unravel_index(int(match_argmin[wind_index]), detuning_valid.shape[1:])
                    )
                    node_index = int(indices[local_node])
                    best_detuning[wind_index] = float(match_min[wind_index])
                    best_match_records[wind_index] = _candidate_record(
                        wind_speed_mps=float(wind_speed),
                        azimuth_deg=float(azimuth),
                        node_index=node_index,
                        depth_index=int(depth_indices[local_depth]),
                        mode_index=mode_index,
                        node_ids=flattened_node_ids,
                        component_ids=component_ids,
                        forcing_frequency_hz=float(frequencies[wind_index, local_node, local_depth]),
                        natural_frequency_hz=float(modes[mode_index]),
                        amplitude_coefficient=float(amplitudes[wind_index, local_node, local_depth]),
                        section_force_amplitude_n=float(force_amplitudes[wind_index, local_node, local_depth]),
                        detuning_fraction=float(detuning[wind_index, local_node, local_depth, mode_index]),
                        overlap_kernel=float(kernel[wind_index, local_node, local_depth, mode_index]),
                        aoa_deg=float(flow.aoa_deg[wind_index, node_index]),
                        reynolds=float(flow.reynolds[wind_index, node_index]),
                        effective_velocity_mps=float(flow.effective_velocity_mps[wind_index, node_index]),
                        source_clipped=bool(clipped_nodes[wind_index, local_node]),
                    )

    rows: list[dict[str, Any]] = []
    for index, wind_speed in enumerate(winds):
        best_overlap = best_overlap_records[index]
        best_frequency_match = best_match_records[index]
        rows.append(
            {
                "wind_speed_mps": float(wind_speed),
                "overlap_score_n": 0.0 if best_overlap is None else float(best_overlap["overlap_score_n"]),
                "minimum_detuning_percent": None
                if best_frequency_match is None
                else float(best_frequency_match["detuning_percent"]),
                "within_overlap_band_candidate_count": int(overlap_candidate_counts[index]),
                "source_clip_query_fraction": 0.0
                if total_source_queries[index] == 0
                else float(source_clip_queries[index] / total_source_queries[index]),
                "best_force_overlap": best_overlap,
                "best_frequency_match": best_frequency_match,
            }
        )
    return rows


def rank_local_peaks(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    max_peaks: int = 6,
    minimum_relative_score: float = 0.2,
    minimum_spacing_mps: float = 0.75,
) -> list[dict[str, Any]]:
    """Return separated local maxima above a fraction of the global metric maximum."""
    if not rows:
        return []
    speeds = np.asarray([float(row["wind_speed_mps"]) for row in rows], dtype=float)
    values = np.asarray([float(row.get(metric, 0.0)) for row in rows], dtype=float)
    if np.any(~np.isfinite(speeds)) or np.any(~np.isfinite(values)):
        raise ValueError("Peak-screen rows must contain finite speeds and metric values.")
    if int(max_peaks) < 1:
        raise ValueError("max_peaks must be positive.")
    if not 0.0 <= float(minimum_relative_score) <= 1.0:
        raise ValueError("minimum_relative_score must be between zero and one.")
    if float(minimum_spacing_mps) < 0.0:
        raise ValueError("minimum_spacing_mps must be nonnegative.")
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return []
    candidates = []
    for index, value in enumerate(values):
        left = values[index - 1] if index > 0 else -np.inf
        right = values[index + 1] if index + 1 < len(values) else -np.inf
        if value >= left and value >= right and value >= float(minimum_relative_score) * maximum:
            candidates.append(index)
    candidates.sort(key=lambda index: (-values[index], speeds[index]))

    selected: list[int] = []
    for index in candidates:
        if any(abs(float(speeds[index] - speeds[other])) < float(minimum_spacing_mps) for other in selected):
            continue
        selected.append(index)
        if len(selected) >= int(max_peaks):
            break
    return [dict(rows[index]) for index in selected]


def assess_peak_coverage(
    peaks: Sequence[Mapping[str, Any]],
    matrix_wind_speeds_mps: Sequence[float],
    *,
    overlap_band_fraction: float = 0.05,
    minimum_tolerance_mps: float = 0.5,
) -> dict[str, Any]:
    """Check whether each ranked peak has a matrix speed within its screening bandwidth."""
    matrix_speeds = _finite_vector(matrix_wind_speeds_mps, "matrix_wind_speeds_mps", positive=True)
    checks = []
    for peak in peaks:
        speed = float(peak["wind_speed_mps"])
        nearest_index = int(np.argmin(np.abs(matrix_speeds - speed)))
        nearest = float(matrix_speeds[nearest_index])
        distance = abs(nearest - speed)
        tolerance = max(float(minimum_tolerance_mps), float(overlap_band_fraction) * speed)
        checks.append(
            {
                "peak_wind_speed_mps": speed,
                "nearest_matrix_wind_speed_mps": nearest,
                "distance_mps": distance,
                "tolerance_mps": tolerance,
                "covered": bool(distance <= tolerance + _EPS),
            }
        )
    return {
        "passed": all(bool(check["covered"]) for check in checks),
        "matrix_wind_speeds_mps": sorted(float(value) for value in matrix_speeds),
        "peak_checks": checks,
        "uncovered_peak_wind_speeds_mps": [
            float(check["peak_wind_speed_mps"]) for check in checks if not bool(check["covered"])
        ],
    }


__all__ = ["assess_peak_coverage", "rank_local_peaks", "screen_modal_overlap"]
