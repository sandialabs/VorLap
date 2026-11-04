from scipy import interpolate
from vorlap.structs import AirfoilFFT


import numpy as np
import warnings


from typing import Dict, List, Tuple, Optional, Sequence


_PHASE_VECTOR_TOL = 1.0e-12


class AirfoilRangeWarning(RuntimeWarning):
    """Warn when Re/AOA queries are outside an airfoil FFT source grid."""


def _normalize_bounds_policy(bounds_policy: str) -> str:
    """Validate range-handling policy for airfoil FFT queries."""
    policy = str(bounds_policy).lower()
    if policy not in ("warn", "raise", "clip"):
        raise ValueError("bounds_policy must be one of: 'warn', 'raise', or 'clip'.")
    return policy


def _format_range(value_min: float, value_max: float) -> str:
    """Format a query/source range compactly for diagnostics."""
    return f"[{value_min:.6g}, {value_max:.6g}]"


def _check_query_ranges(
    afft: AirfoilFFT,
    Re_q: np.ndarray,
    AOA_q: np.ndarray,
    *,
    fields: Sequence[str],
    bounds_policy: str = "warn",
) -> None:
    """
    Diagnose Re/AOA queries outside the available source-data range.

    Interpolation deliberately flat-clips outside the source grid, but final
    paper campaigns must not depend on that behavior silently. The default
    policy warns once per airfoil/field-set/dimension; strict audit paths can
    use `bounds_policy="raise"`.
    """
    policy = _normalize_bounds_policy(bounds_policy)
    re_arr = np.asarray(Re_q, dtype=float).reshape(-1)
    aoa_arr = np.asarray(AOA_q, dtype=float).reshape(-1)
    if re_arr.shape != aoa_arr.shape:
        raise ValueError("Re and AOA query arrays must have the same shape.")
    if np.any(~np.isfinite(re_arr)) or np.any(~np.isfinite(aoa_arr)):
        raise ValueError("Re and AOA query values must be finite.")

    re_min = float(np.min(afft.Re))
    re_max = float(np.max(afft.Re))
    aoa_min = float(np.min(afft.AOA))
    aoa_max = float(np.max(afft.AOA))
    re_below = re_arr < re_min
    re_above = re_arr > re_max
    reynolds_independent = (
        getattr(afft, "reynolds_model", "") == "replicated_reynolds_independent"
    )
    warning_only_re_below = re_below & reynolds_independent
    hard_re_out = re_above | (re_below & ~reynolds_independent)
    re_out = re_below | re_above
    aoa_out = (aoa_arr < aoa_min) | (aoa_arr > aoa_max)
    if not (np.any(re_out) or np.any(aoa_out)):
        return

    labels = []
    if np.any(re_out):
        labels.append(
            f"Re requested {_format_range(float(np.min(re_arr)), float(np.max(re_arr)))} "
            f"outside source {_format_range(re_min, re_max)} ({int(np.count_nonzero(re_out))}/{re_arr.size} queries)"
        )
    if np.any(aoa_out):
        labels.append(
            f"AOA requested {_format_range(float(np.min(aoa_arr)), float(np.max(aoa_arr)))} deg "
            f"outside source {_format_range(aoa_min, aoa_max)} deg ({int(np.count_nonzero(aoa_out))}/{aoa_arr.size} queries)"
        )

    field_key = tuple(str(field) for field in fields)
    message = (
        f"Airfoil FFT '{getattr(afft, 'name', 'unknown')}' query outside source grid "
        f"for fields {field_key}: " + "; ".join(labels) + ". "
        "Values will be clipped to the nearest source-grid boundary."
    )
    if policy == "raise" and (np.any(hard_re_out) or np.any(aoa_out)):
        raise ValueError(message)
    if policy == "warn" or (policy == "raise" and np.any(warning_only_re_below)):
        warned = getattr(afft, "_range_warning_keys", None)
        if warned is None:
            warned = set()
            setattr(afft, "_range_warning_keys", warned)
        if policy == "raise":
            message = (
                f"Airfoil FFT '{getattr(afft, 'name', 'unknown')}' requested Reynolds range "
                f"{_format_range(float(np.min(re_arr)), float(np.max(re_arr)))} includes values below source minimum "
                f"{re_min:.6g} ({int(np.count_nonzero(warning_only_re_below))}/{re_arr.size} queries). "
                "The source is verified as replicated_reynolds_independent, so values are clipped "
                "to the nearest Reynolds plane and this lower-bound excursion is warning-only."
            )
            range_key = ("replicated_reynolds_independent_lower_re",)
        else:
            range_key = (field_key, bool(np.any(re_out)), bool(np.any(aoa_out)))
        if range_key not in warned:
            warned.add(range_key)
            warnings.warn(message, AirfoilRangeWarning, stacklevel=3)


def _field_tensors(afft: AirfoilFFT, field: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Strouhal, amplitude, and phase tensors for an FFT field."""
    if field not in ("CL", "CD", "CM", "CF"):
        raise ValueError(f"Invalid field symbol: {field}")
    return (
        getattr(afft, f"{field}_ST"),
        getattr(afft, f"{field}_Amp"),
        getattr(afft, f"{field}_Pha"),
    )


def _wrap_phase_radians(phase: np.ndarray) -> np.ndarray:
    """Wrap phase angles to the principal [-pi, pi] interval."""
    return np.arctan2(np.sin(phase), np.cos(phase))


def _find_nearest_indices(Re_grid: np.ndarray,
                          AOA_grid: np.ndarray,
                          Re_q: np.ndarray,
                          AOA_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each query (Re_q, AOA_q), return indices of the *closest* Re and AOA.
    Ties go to the lower (left) index.

    Re_grid, AOA_grid must be 1D, sorted ascending.
    Re_q, AOA_q can be scalars or 1D arrays (nq,).
    """
    Re_q  = np.atleast_1d(Re_q).astype(float)
    AOA_q = np.atleast_1d(AOA_q).astype(float)
    if Re_grid.size == 0 or AOA_grid.size == 0:
        raise ValueError("Re and AOA grids must be non-empty.")

    # For Re
    i_right = np.searchsorted(Re_grid, Re_q, side='left')
    i_left  = np.clip(i_right - 1, 0, len(Re_grid) - 1)
    i_right = np.clip(i_right,     0, len(Re_grid) - 1)
    choose_left = np.abs(Re_q - Re_grid[i_left]) <= np.abs(Re_q - Re_grid[i_right])
    i_nn = np.where(choose_left, i_left, i_right)

    # For AOA
    j_right = np.searchsorted(AOA_grid, AOA_q, side='left')
    j_left  = np.clip(j_right - 1, 0, len(AOA_grid) - 1)
    j_right = np.clip(j_right,     0, len(AOA_grid) - 1)
    choose_left = np.abs(AOA_q - AOA_grid[j_left]) <= np.abs(AOA_q - AOA_grid[j_right])
    j_nn = np.where(choose_left, j_left, j_right)

    return i_nn, j_nn


def _lookup_apply_tensor(F: np.ndarray,
                         i_nn: np.ndarray,
                         j_nn: np.ndarray,
                         n_freq_depth: Optional[int] = None) -> np.ndarray:
    """
    F: (NR, NA, K) -> returns (nq, K_sel)
    Gathers spectrum rows at nearest (Re, AOA) for each query.
    """
    NR, NA, K = F.shape
    if n_freq_depth is None or n_freq_depth > K:
        n_freq_depth = K
    F = F[:, :, :n_freq_depth]
    # Advanced indexing over (nq,) for i and j -> (nq, K_sel)
    return F[i_nn, j_nn, :]

def lookup_fft_spectrum_nearest(afft,
                                Re_val: float,
                                AOA_val: float,
                                fields: List[str],
                                n_freq_depth: Optional[int] = None,
                                bounds_policy: str = "warn") -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Nearest-neighbor lookup (no interpolation).
    Matches your return structure: { 'CL': (ST, Amp, Pha), ... }.
    """
    Re_q  = np.atleast_1d(Re_val).astype(float)
    AOA_q = np.atleast_1d(AOA_val).astype(float)
    _check_query_ranges(afft, Re_q, AOA_q, fields=fields, bounds_policy=bounds_policy)

    # find nearest grid indices (no clamping needed beyond edges)
    i_nn, j_nn = _find_nearest_indices(afft.Re, afft.AOA, Re_q, AOA_q)

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for field in fields:
        if field == 'CL':
            STsrc, Asrc, Psrc = afft.CL_ST, afft.CL_Amp, afft.CL_Pha
        elif field == 'CD':
            STsrc, Asrc, Psrc = afft.CD_ST, afft.CD_Amp, afft.CD_Pha
        elif field == 'CF':
            STsrc, Asrc, Psrc = afft.CF_ST, afft.CF_Amp, afft.CF_Pha
        elif field == 'CM':
            STsrc, Asrc, Psrc = afft.CM_ST, afft.CM_Amp, afft.CM_Pha
        else:
            raise ValueError(f"Invalid field symbol: {field}")

        ST  = _lookup_apply_tensor(STsrc, i_nn, j_nn, n_freq_depth)[0]  # nq==1 -> [0]
        Amp = _lookup_apply_tensor(Asrc,  i_nn, j_nn, n_freq_depth)[0]
        Pha = _lookup_apply_tensor(Psrc,  i_nn, j_nn, n_freq_depth)[0]

        out[field] = (ST, Amp, Pha)
    return out

def _find_cells_and_weights(Re_grid: np.ndarray,
                            AOA_grid: np.ndarray,
                            Re_q: np.ndarray,
                            AOA_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each query (Re_q, AOA_q), clamp to flat-extrapolate and return:
      i,j: lower cell indices
      t,u: local linear coordinates in [0,1] within that cell
    """
    Re = Re_grid
    AoA = AOA_grid

    # clamp for flat extrapolation
    Re_q = np.clip(Re_q, Re[0], Re[-1])
    AOA_q = np.clip(AOA_q, AoA[0], AoA[-1])

    if len(Re) == 1:
        i = np.zeros_like(Re_q, dtype=int)
        t = np.zeros_like(Re_q, dtype=float)
    else:
        i = np.searchsorted(Re, Re_q, side='right') - 1
        i = np.clip(i, 0, len(Re) - 2)
        Re0 = Re[i]
        Re1 = Re[i + 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.divide(Re_q - Re0, Re1 - Re0, out=np.zeros_like(Re_q), where=(Re1 != Re0))

    if len(AoA) == 1:
        j = np.zeros_like(AOA_q, dtype=int)
        u = np.zeros_like(AOA_q, dtype=float)
    else:
        j = np.searchsorted(AoA, AOA_q, side='right') - 1
        j = np.clip(j, 0, len(AoA) - 2)
        AoA0 = AoA[j]
        AoA1 = AoA[j + 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.divide(AOA_q - AoA0, AoA1 - AoA0, out=np.zeros_like(AOA_q), where=(AoA1 != AoA0))

    return i, j, t, u

def _bilinear_apply_tensor(F: np.ndarray,
                           i: np.ndarray,
                           j: np.ndarray,
                           t: np.ndarray,
                           u: np.ndarray,
                           n_freq_depth: Optional[int] = None) -> np.ndarray:
    """
    F: (NR, NA, K)  -> returns (nq, K_sel)
    Applies bilinear interpolation at nq query points, for all K at once.
    """
    NR, NA, K = F.shape
    if n_freq_depth is None or n_freq_depth > K:
        n_freq_depth = K

    # slice frequencies up to depth
    F = F[:, :, :n_freq_depth]

    nq = i.shape[0]
    if NR == 1 and NA == 1:
        return np.repeat(F[0, 0, :][None, :], nq, axis=0)
    if NR == 1:
        f0 = F[0, j, :]
        f1 = F[0, j + 1, :]
        return (1.0 - u)[:, None] * f0 + u[:, None] * f1
    if NA == 1:
        f0 = F[i, 0, :]
        f1 = F[i + 1, 0, :]
        return (1.0 - t)[:, None] * f0 + t[:, None] * f1

    # gather 4 corners (nq, K_sel) via advanced indexing
    f00 = F[i,     j,     :]  # lower-left
    f10 = F[i + 1, j,     :]  # right
    f01 = F[i,     j + 1, :]  # up
    f11 = F[i + 1, j + 1, :]  # up-right

    w00 = (1 - t) * (1 - u)
    w10 = t * (1 - u)
    w01 = (1 - t) * u
    w11 = t * u

    # expand weights to (nq, 1) to broadcast along K
    return (
        w00[:, None] * f00
        + w10[:, None] * f10
        + w01[:, None] * f01
        + w11[:, None] * f11
    )


def _bilinear_apply_phase_tensor(P: np.ndarray,
                                 i: np.ndarray,
                                 j: np.ndarray,
                                 t: np.ndarray,
                                 u: np.ndarray,
                                 n_freq_depth: Optional[int] = None) -> np.ndarray:
    """
    Bilinearly interpolate phase angles through their unit phasors.

    Direct radian interpolation is wrong at the branch cut: +179 deg and
    -179 deg should interpolate near 180 deg, not 0 deg. The fallback is used
    only where the weighted phasor magnitude vanishes and the circular mean is
    mathematically undefined.
    """
    return _bilinear_apply_precomputed_phase_tensor(
        P,
        np.cos(P),
        np.sin(P),
        i,
        j,
        t,
        u,
        n_freq_depth,
    )


def _bilinear_apply_precomputed_phase_tensor(
    P: np.ndarray,
    cos_source: np.ndarray,
    sin_source: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    n_freq_depth: Optional[int] = None,
) -> np.ndarray:
    """Interpolate phase using source phasors prepared from the same tensor."""
    if cos_source.shape != P.shape or sin_source.shape != P.shape:
        raise ValueError("Cached phase phasors must match the source phase tensor.")
    cos_phase = _bilinear_apply_tensor(cos_source, i, j, t, u, n_freq_depth)
    sin_phase = _bilinear_apply_tensor(sin_source, i, j, t, u, n_freq_depth)
    phase = np.arctan2(sin_phase, cos_phase)

    magnitude = np.hypot(cos_phase, sin_phase)
    if np.any(magnitude <= _PHASE_VECTOR_TOL):
        linear_fallback = _wrap_phase_radians(_bilinear_apply_tensor(P, i, j, t, u, n_freq_depth))
        phase = np.where(magnitude > _PHASE_VECTOR_TOL, phase, linear_fallback)
    return phase


def interpolate_fft_spectra_batch_prepared(
    afft: AirfoilFFT,
    Re_vals: np.ndarray,
    AOA_vals: np.ndarray,
    fields: List[str],
    *,
    n_freq_depth: Optional[int] = None,
    bounds_policy: str = "warn",
    phase_phasors: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Interpolate several fields with shared query weights and cached phasors."""
    Re_vals = np.asarray(Re_vals, dtype=float)
    AOA_vals = np.asarray(AOA_vals, dtype=float)
    if Re_vals.shape != AOA_vals.shape:
        raise ValueError("Re_vals and AOA_vals must have the same shape")
    Re_flat = Re_vals.reshape(-1)
    AOA_flat = AOA_vals.reshape(-1)
    _check_query_ranges(
        afft,
        Re_flat,
        AOA_flat,
        fields=fields,
        bounds_policy=bounds_policy,
    )
    i, j, t, u = _find_cells_and_weights(afft.Re, afft.AOA, Re_flat, AOA_flat)
    result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for field in fields:
        STsrc, Asrc, Psrc = _field_tensors(afft, field)
        if phase_phasors is None or field not in phase_phasors:
            phase = _bilinear_apply_phase_tensor(
                Psrc, i, j, t, u, n_freq_depth
            )
        else:
            cos_source, sin_source = phase_phasors[field]
            phase = _bilinear_apply_precomputed_phase_tensor(
                Psrc,
                cos_source,
                sin_source,
                i,
                j,
                t,
                u,
                n_freq_depth,
            )
        result[field] = (
            _bilinear_apply_tensor(STsrc, i, j, t, u, n_freq_depth),
            _bilinear_apply_tensor(Asrc, i, j, t, u, n_freq_depth),
            phase,
        )
    return result


def prepare_fft_spectra_packed_tensor(
    afft: AirfoilFFT,
    fields: List[str],
    *,
    n_freq_depth: int,
) -> np.ndarray:
    """Pack ST, amplitude, and phase phasors for one-call interpolation."""

    if type(n_freq_depth) is not int or n_freq_depth < 1:
        raise ValueError("n_freq_depth must be a positive integer")
    tensors: List[np.ndarray] = []
    for field in fields:
        st_source, amplitude_source, phase_source = _field_tensors(afft, field)
        depth = min(n_freq_depth, int(st_source.shape[2]))
        if depth != n_freq_depth:
            raise ValueError("Packed interpolation depth exceeds source depth")
        tensors.extend(
            (
                st_source[:, :, :depth],
                amplitude_source[:, :, :depth],
                np.cos(phase_source[:, :, :depth]),
                np.sin(phase_source[:, :, :depth]),
            )
        )
    return np.concatenate(tensors, axis=2)


def interpolate_fft_spectra_batch_packed(
    afft: AirfoilFFT,
    Re_vals: np.ndarray,
    AOA_vals: np.ndarray,
    fields: List[str],
    *,
    n_freq_depth: int,
    bounds_policy: str,
    packed_tensor: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Interpolate several FFT fields through one packed bilinear gather."""

    Re_vals = np.asarray(Re_vals, dtype=float)
    AOA_vals = np.asarray(AOA_vals, dtype=float)
    if Re_vals.shape != AOA_vals.shape:
        raise ValueError("Re_vals and AOA_vals must have the same shape")
    Re_flat = Re_vals.reshape(-1)
    AOA_flat = AOA_vals.reshape(-1)
    _check_query_ranges(
        afft,
        Re_flat,
        AOA_flat,
        fields=fields,
        bounds_policy=bounds_policy,
    )
    expected_shape = (
        len(afft.Re),
        len(afft.AOA),
        4 * n_freq_depth * len(fields),
    )
    packed = np.asarray(packed_tensor, dtype=float)
    if packed.shape != expected_shape:
        raise ValueError("Packed FFT interpolation tensor has the wrong shape")
    i, j, t, u = _find_cells_and_weights(afft.Re, afft.AOA, Re_flat, AOA_flat)
    interpolated = _bilinear_apply_tensor(packed, i, j, t, u)
    result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    stride = 4 * n_freq_depth
    for field_index, field in enumerate(fields):
        offset = field_index * stride
        st = interpolated[:, offset : offset + n_freq_depth]
        amplitude = interpolated[
            :, offset + n_freq_depth : offset + 2 * n_freq_depth
        ]
        cos_phase = interpolated[
            :, offset + 2 * n_freq_depth : offset + 3 * n_freq_depth
        ]
        sin_phase = interpolated[
            :, offset + 3 * n_freq_depth : offset + 4 * n_freq_depth
        ]
        phase = np.arctan2(sin_phase, cos_phase)
        magnitude = np.hypot(cos_phase, sin_phase)
        if np.any(magnitude <= _PHASE_VECTOR_TOL):
            phase_source = _field_tensors(afft, field)[2]
            linear_fallback = _wrap_phase_radians(
                _bilinear_apply_tensor(
                    phase_source,
                    i,
                    j,
                    t,
                    u,
                    n_freq_depth,
                )
            )
            phase = np.where(
                magnitude > _PHASE_VECTOR_TOL,
                phase,
                linear_fallback,
            )
        result[field] = (st, amplitude, phase)
    return result

# interpolation.py (continued)
def interpolate_fft_spectrum_optimized(afft: AirfoilFFT,
                                       Re_val: float,
                                       AOA_val: float,
                                       fields: List[str],
                                       n_freq_depth: Optional[int] = None,
                                       bounds_policy: str = "warn") -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Vectorized, fast bilinear + flat extrapolation for multiple fields at once.
    Matches your return structure: { 'CL': (ST, Amp, Pha), ... } with arrays length n_freq_depth.
    `bounds_policy` controls out-of-range Re/AOA queries: "warn" clips with
    `AirfoilRangeWarning`, "raise" fails, and "clip" clips silently.
    """
    # inputs -> 1D arrays to allow easy batching later (for now 1 query)
    Re_q  = np.atleast_1d(Re_val).astype(float)
    AOA_q = np.atleast_1d(AOA_val).astype(float)
    _check_query_ranges(afft, Re_q, AOA_q, fields=fields, bounds_policy=bounds_policy)

    # compute indices/weights once
    i, j, t, u = _find_cells_and_weights(afft.Re, afft.AOA, Re_q, AOA_q)

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for field in fields:
        STsrc, Asrc, Psrc = _field_tensors(afft, field)

        # Each returns (nq, K_sel). We have nq==1, so take [0]
        ST = _bilinear_apply_tensor(STsrc, i, j, t, u, n_freq_depth)[0]
        Amp = _bilinear_apply_tensor(Asrc,  i, j, t, u, n_freq_depth)[0]
        Pha = _bilinear_apply_phase_tensor(Psrc, i, j, t, u, n_freq_depth)[0]

        out[field] = (ST, Amp, Pha)

    return out


def interpolate_fft_spectrum_batch(afft: AirfoilFFT, Re_vals: np.ndarray, AOA_vals: np.ndarray,
                                 field: str, n_freq_depth: Optional[int] = None,
                                 bounds_policy: str = "warn") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch interpolation for multiple Re/AOA points simultaneously.

    Args:
        afft: AirfoilFFT struct containing FFT results.
        Re_vals: Array of Reynolds numbers.
        AOA_vals: Array of angles of attack (degrees).
        field: Field string ('CL', 'CD', 'CM', 'CF') to interpolate.
        n_freq_depth: Optional number of frequencies to return.
        bounds_policy: "warn" clips with `AirfoilRangeWarning`, "raise" fails,
            and "clip" preserves explicit silent flat-clipping.

    Returns:
        ST_out: Array of shape [n_points, n_freq]
        amp_out: Array of shape [n_points, n_freq]
        phase_out: Array of shape [n_points, n_freq]
    """
    Re_vals = np.asarray(Re_vals, dtype=float)
    AOA_vals = np.asarray(AOA_vals, dtype=float)
    if Re_vals.shape != AOA_vals.shape:
        raise ValueError("Re_vals and AOA_vals must have the same shape")
    Re_flat = Re_vals.reshape(-1)
    AOA_flat = AOA_vals.reshape(-1)
    _check_query_ranges(afft, Re_flat, AOA_flat, fields=[field], bounds_policy=bounds_policy)
    i, j, t, u = _find_cells_and_weights(afft.Re, afft.AOA, Re_flat, AOA_flat)

    STsrc, Asrc, Psrc = _field_tensors(afft, field)
    ST_out = _bilinear_apply_tensor(STsrc, i, j, t, u, n_freq_depth)
    amp_out = _bilinear_apply_tensor(Asrc, i, j, t, u, n_freq_depth)
    phase_out = _bilinear_apply_phase_tensor(Psrc, i, j, t, u, n_freq_depth)

    return ST_out, amp_out, phase_out


def interpolate_fft_spectrum(afft: AirfoilFFT, Re_val: float, AOA_val: float, field: str,
                             n_freq_depth: Optional[int] = None,
                             bounds_policy: str = "warn") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolates the FFT amplitude and phase spectra for a given Reynolds number `Re_val`
    and angle of attack `AOA_val` using bilinear interpolation over the stored Re × AOA grid.

    Args:
        afft: AirfoilFFT struct containing FFT results.
        Re_val: Desired Reynolds number.
        AOA_val: Desired angle of attack (degrees).
        field: String ('CL', 'CD', 'CM', or 'CF') indicating which force coefficient to interpolate.
        n_freq_depth: Optional number of frequencies to return. If None, returns all frequencies.
        bounds_policy: "warn" clips with `AirfoilRangeWarning`, "raise" fails,
            and "clip" preserves explicit silent flat-clipping.

    Returns:
        freqs: Vector of frequency values.
        amp_out: Vector of interpolated amplitudes at each frequency.
        phase_out: Vector of interpolated phases at each frequency.

    Notes:
        - The interpolation is performed independently at each frequency index in the spectrum.
        - Assumes consistent frequency axis across the full 3D data structure.
        - Returns values suitable for reconstructing time-domain or frequency-domain force estimates.
    """
    result = interpolate_fft_spectrum_optimized(
        afft,
        Re_val,
        AOA_val,
        [field],
        n_freq_depth=n_freq_depth,
        bounds_policy=bounds_policy,
    )
    return result[field]


def resample_airfoil(xy: np.ndarray, npoints: int = 200) -> np.ndarray:
    """
    Resamples the given airfoil shape by:
    1. Identifying leading (min x) and trailing (max x) edges.
    2. Splitting into upper and lower surfaces.
    3. Interpolating both surfaces using `npoints` uniformly spaced x-values.
    4. Recombining to produce a smooth resampled airfoil shape.

    Args:
        xy: Nx2 matrix of (x, y) airfoil coordinates, not assumed to start at TE or LE.
        npoints: Number of points used in interpolation (default: 200).

    Returns:
        xy_resampled: Resampled and recombined airfoil shape.
    """
    x = xy[:, 0]
    y = xy[:, 1]

    # Find leading edge as the point with minimum x
    le_idx = np.argmin(x)
    x_le = x[le_idx]

    # Split into upper and lower surfaces
    upper = xy[:le_idx+1, :]
    lower = xy[le_idx:, :]

    # Sort upper surface by descending x
    upper_sort_idx = np.argsort(upper[:, 0])
    upper_sorted = upper[upper_sort_idx, :]

    # Sort lower surface by ascending x
    lower_sort_idx = np.argsort(lower[:, 0])
    lower_sorted = lower[lower_sort_idx, :]

    x_upper = upper_sorted[:, 0]
    y_upper = upper_sorted[:, 1]
    x_lower = lower_sorted[:, 0]
    y_lower = lower_sorted[:, 1]

    # Create common x-grid
    x_min = max(np.min(x_upper), np.min(x_lower))
    x_max = min(np.max(x_upper), np.max(x_lower))
    x_resample = np.linspace(x_min, x_max, int(round(npoints/2)))

    # Interpolate both surfaces
    itp_upper = interpolate.interp1d(x_upper, y_upper, bounds_error=False, fill_value='extrapolate')
    itp_lower = interpolate.interp1d(x_lower, y_lower, bounds_error=False, fill_value='extrapolate')
    y_upper_resampled = itp_upper(x_resample)
    y_lower_resampled = itp_lower(x_resample)

    # Recombine: upper reversed to preserve typical TE-to-LE-to-TE convention
    x_combined = np.concatenate([np.flip(x_resample), x_resample])
    y_combined = np.concatenate([np.flip(y_upper_resampled), y_lower_resampled])

    return np.column_stack([x_combined, y_combined])
