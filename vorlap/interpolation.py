from scipy import interpolate
from vorlap.structs import AirfoilFFT


import numpy as np


from typing import Dict, List, Tuple, Optional


def _find_cells_and_weights(Re_grid: np.ndarray,
                            AOA_grid: np.ndarray,
                            Re_q: np.ndarray,
                            AOA_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find cells and weights for bilinear interpolation.
    
    For each query (Re_q, AOA_q), clamp to flat-extrapolate and return:
        i,j: lower cell indices
        t,u: local linear coordinates in [0,1] within that cell
        
    Args:
        Re_grid: Reynolds number grid values.
        AOA_grid: Angle of attack grid values.
        Re_q: Query Reynolds number values.
        AOA_q: Query angle of attack values.
        
    Returns:
        Tuple of (i, j, t, u) arrays for interpolation.
    """
    Re = Re_grid
    AoA = AOA_grid

    # clamp for flat extrapolation
    Re_q = np.clip(Re_q, Re[0], Re[-1])
    AOA_q = np.clip(AOA_q, AoA[0], AoA[-1])

    # locate lower cell indices
    i = np.searchsorted(Re, Re_q, side='right') - 1
    j = np.searchsorted(AoA, AOA_q, side='right') - 1
    i = np.clip(i, 0, len(Re) - 2)
    j = np.clip(j, 0, len(AoA) - 2)

    Re0 = Re[i]
    Re1 = Re[i + 1]
    AoA0 = AoA[j]
    AoA1 = AoA[j + 1]

    # local barycentric coords
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(Re_q - Re0, Re1 - Re0, out=np.zeros_like(Re_q), where=(Re1 != Re0))
        u = np.divide(AOA_q - AoA0, AoA1 - AoA0, out=np.zeros_like(AOA_q), where=(AoA1 != AoA0))

    return i, j, t, u

def _bilinear_apply_tensor(F: np.ndarray,
                           i: np.ndarray,
                           j: np.ndarray,
                           t: np.ndarray,
                           u: np.ndarray,
                           n_freq_depth: Optional[int] = None) -> np.ndarray:
    """
    Apply bilinear interpolation to a tensor at query points.
    
    Args:
        F: Input tensor of shape (NR, NA, K).
        i: Lower cell indices in Reynolds direction.
        j: Lower cell indices in AOA direction.
        t: Local coordinates in Reynolds direction [0,1].
        u: Local coordinates in AOA direction [0,1].
        n_freq_depth: Optional depth limit for frequency dimension.
        
    Returns:
        Interpolated values of shape (nq, K_sel).
    """
    NR, NA, K = F.shape
    if n_freq_depth is None or n_freq_depth > K:
        n_freq_depth = K

    # slice frequencies up to depth
    F = F[:, :, :n_freq_depth]

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
    out = (w00[:, None] * f00 +
           w10[:, None] * f10 +
           w01[:, None] * f01 +
           w11[:, None] * f11)
    return out

# interpolation.py (continued)
def interpolate_fft_spectrum_optimized(afft: AirfoilFFT,
                                       Re_val: float,
                                       AOA_val: float,
                                       fields: List[str],
                                       n_freq_depth: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Vectorized, fast bilinear interpolation with flat extrapolation for multiple fields at once.
    
    Args:
        afft: AirfoilFFT struct containing FFT results.
        Re_val: Desired Reynolds number.
        AOA_val: Desired angle of attack (degrees).
        fields: List of field strings ('CL', 'CD', 'CM', 'CF') to interpolate.
        n_freq_depth: Optional number of frequencies to return.
        
    Returns:
        Dictionary mapping field names to (ST, Amp, Pha) tuples with arrays of length n_freq_depth.
    """
    # inputs -> 1D arrays to allow easy batching later (for now 1 query)
    Re_q  = np.atleast_1d(Re_val).astype(float)
    AOA_q = np.atleast_1d(AOA_val).astype(float)

    # compute indices/weights once
    i, j, t, u = _find_cells_and_weights(afft.Re, afft.AOA, Re_q, AOA_q)

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

        # Each returns (nq, K_sel). We have nq==1, so take [0]
        ST = _bilinear_apply_tensor(STsrc, i, j, t, u, n_freq_depth)[0]
        Amp = _bilinear_apply_tensor(Asrc,  i, j, t, u, n_freq_depth)[0]
        Pha = _bilinear_apply_tensor(Psrc,  i, j, t, u, n_freq_depth)[0]

        out[field] = (ST, Amp, Pha)

    return out


def interpolate_fft_spectrum_batch(afft: AirfoilFFT, Re_vals: np.ndarray, AOA_vals: np.ndarray,
                                 field: str, n_freq_depth: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch interpolation for multiple Re/AOA points simultaneously.

    Args:
        afft: AirfoilFFT struct containing FFT results with cached interpolators.
        Re_vals: Array of Reynolds numbers.
        AOA_vals: Array of angles of attack (degrees).
        field: Field string ('CL', 'CD', 'CM', 'CF') to interpolate.
        n_freq_depth: Optional number of frequencies to return.

    Returns:
        Tuple containing:
            - ST_out: Array of shape [n_points, n_freq]
            - amp_out: Array of shape [n_points, n_freq]
            - phase_out: Array of shape [n_points, n_freq]
    """
    # Ensure interpolators are cached
    afft._cache_interpolators()

    if n_freq_depth is None:
        n_freq_depth = afft.CL_ST.shape[2]
    else:
        n_freq_depth = min(n_freq_depth, afft.CL_ST.shape[2])

    n_points = len(Re_vals)
    points = np.column_stack([Re_vals, AOA_vals])

    ST_out = np.zeros((n_points, n_freq_depth))
    amp_out = np.zeros((n_points, n_freq_depth))
    phase_out = np.zeros((n_points, n_freq_depth))

    # Get the appropriate cached interpolators
    if field == 'CL':
        st_interps, amp_interps, pha_interps = afft._cl_st_interps, afft._cl_amp_interps, afft._cl_pha_interps
    elif field == 'CD':
        st_interps, amp_interps, pha_interps = afft._cd_st_interps, afft._cd_amp_interps, afft._cd_pha_interps
    elif field == 'CF':
        st_interps, amp_interps, pha_interps = afft._cf_st_interps, afft._cf_amp_interps, afft._cf_pha_interps
    else:
        raise ValueError(f"Invalid field symbol: {field}")

    # Batch interpolation
    for k in range(n_freq_depth):
        ST_out[:, k] = st_interps[k](points)
        amp_out[:, k] = amp_interps[k](points)
        phase_out[:, k] = pha_interps[k](points)

    return ST_out, amp_out, phase_out


def interpolate_fft_spectrum(afft: AirfoilFFT, Re_val: float, AOA_val: float, field: str, n_freq_depth: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate the FFT amplitude and phase spectra using bilinear interpolation over the stored Re × AOA grid.

    Args:
        afft: AirfoilFFT struct containing FFT results.
        Re_val: Desired Reynolds number.
        AOA_val: Desired angle of attack (degrees).
        field: String ('CL', 'CD', 'CM', or 'CF') indicating which force coefficient to interpolate.
        n_freq_depth: Optional number of frequencies to return. If None, returns all frequencies.

    Returns:
        Tuple containing:
            - freqs: Vector of frequency values.
            - amp_out: Vector of interpolated amplitudes at each frequency.
            - phase_out: Vector of interpolated phases at each frequency.

    Note:
        - The interpolation is performed independently at each frequency index in the spectrum.
        - Assumes consistent frequency axis across the full 3D data structure.
        - Returns values suitable for reconstructing time-domain or frequency-domain force estimates.
    """
    if field == 'CL':
        STs, amps, phases = afft.CL_ST, afft.CL_Amp, afft.CL_Pha
    elif field == 'CD':
        STs, amps, phases = afft.CD_ST, afft.CD_Amp, afft.CD_Pha
    elif field == 'CM':
        STs, amps, phases = afft.CM_ST, afft.CM_Amp, afft.CM_Pha
    elif field == 'CF':
        STs, amps, phases = afft.CF_ST, afft.CF_Amp, afft.CF_Pha
    else:
        raise ValueError(f"Invalid field symbol: {field}")

    if n_freq_depth is None:
        n_freq_depth = STs.shape[2]  # Use all frequencies

    n_freq_depth = min(n_freq_depth, STs.shape[2])  # Ensure we don't exceed available frequencies

    ST_out = np.zeros(n_freq_depth)
    amp_out = np.zeros(n_freq_depth)
    phase_out = np.zeros(n_freq_depth)

    for k in range(n_freq_depth):
        # Create interpolation functions for this frequency
        st_interp = interpolate.RegularGridInterpolator(
            (afft.Re, afft.AOA),
            STs[:, :, k],
            bounds_error=False,
            fill_value=None
        )

        amp_interp = interpolate.RegularGridInterpolator(
            (afft.Re, afft.AOA),
            amps[:, :, k],
            bounds_error=False,
            fill_value=None
        )

        pha_interp = interpolate.RegularGridInterpolator(
            (afft.Re, afft.AOA),
            phases[:, :, k],
            bounds_error=False,
            fill_value=None
        )

        # Interpolate at the requested point
        ST_out[k] = st_interp(np.array([Re_val, AOA_val]))
        amp_out[k] = amp_interp(np.array([Re_val, AOA_val]))
        phase_out[k] = pha_interp(np.array([Re_val, AOA_val]))

    return ST_out, amp_out, phase_out


# def interpolate_fft_spectrum_optimized(afft: AirfoilFFT, Re_val: float, AOA_val: float,
#                                      fields: List[str], n_freq_depth: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
#     """
#     Optimized version that interpolates multiple fields simultaneously using cached interpolators.

#     Args:
#         afft: AirfoilFFT struct containing FFT results with cached interpolators.
#         Re_val: Desired Reynolds number.
#         AOA_val: Desired angle of attack (degrees).
#         fields: List of field strings ('CL', 'CD', 'CM', 'CF') to interpolate.
#         n_freq_depth: Optional number of frequencies to return. If None, returns all frequencies.

#     Returns:
#         Dictionary mapping field names to (ST_out, amp_out, phase_out) tuples.
#     """
#     # Ensure interpolators are cached
#     afft._cache_interpolators()

#     if n_freq_depth is None:
#         n_freq_depth = afft.CL_ST.shape[2]
#     else:
#         n_freq_depth = min(n_freq_depth, afft.CL_ST.shape[2])

#     # Pre-compute the interpolation point as array to avoid repeated array creation
#     point = np.array([Re_val, AOA_val])

#     results = {}

#     for field in fields:
#         ST_out = np.zeros(n_freq_depth)
#         amp_out = np.zeros(n_freq_depth)
#         phase_out = np.zeros(n_freq_depth)

#         # Get the appropriate cached interpolators
#         if field == 'CL':
#             st_interps, amp_interps, pha_interps = afft._cl_st_interps, afft._cl_amp_interps, afft._cl_pha_interps
#         elif field == 'CD':
#             st_interps, amp_interps, pha_interps = afft._cd_st_interps, afft._cd_amp_interps, afft._cd_pha_interps
#         elif field == 'CF':
#             st_interps, amp_interps, pha_interps = afft._cf_st_interps, afft._cf_amp_interps, afft._cf_pha_interps
#         elif field == 'CM':
#             # CM not cached in current implementation, fall back to original method
#             return {field: interpolate_fft_spectrum(afft, Re_val, AOA_val, field, n_freq_depth)}
#         else:
#             raise ValueError(f"Invalid field symbol: {field}")

#         # Vectorized interpolation using cached interpolators
#         for k in range(n_freq_depth):
#             ST_out[k] = st_interps[k](point)
#             amp_out[k] = amp_interps[k](point)
#             phase_out[k] = pha_interps[k](point)

#         results[field] = (ST_out, amp_out, phase_out)

#     return results


def resample_airfoil(xy: np.ndarray, npoints: int = 200) -> np.ndarray:
    """
    Resample the given airfoil shape using uniform x-spacing.
    
    Process:
        1. Identify leading (min x) and trailing (max x) edges.
        2. Split into upper and lower surfaces.
        3. Interpolate both surfaces using `npoints` uniformly spaced x-values.
        4. Recombine to produce a smooth resampled airfoil shape.

    Args:
        xy: Nx2 matrix of (x, y) airfoil coordinates, not assumed to start at TE or LE.
        npoints: Number of points used in interpolation (default: 200).

    Returns:
        Resampled and recombined airfoil shape.
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