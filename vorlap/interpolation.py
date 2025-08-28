from scipy import interpolate
from vorlap.structs import AirfoilFFT


import numpy as np


from typing import Dict, List, Tuple, Optional


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
        ST_out: Array of shape [n_points, n_freq]
        amp_out: Array of shape [n_points, n_freq]
        phase_out: Array of shape [n_points, n_freq]
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
    Interpolates the FFT amplitude and phase spectra for a given Reynolds number `Re_val`
    and angle of attack `AOA_val` using bilinear interpolation over the stored Re × AOA grid.

    Args:
        afft: AirfoilFFT struct containing FFT results.
        Re_val: Desired Reynolds number.
        AOA_val: Desired angle of attack (degrees).
        field: String ('CL', 'CD', 'CM', or 'CF') indicating which force coefficient to interpolate.
        n_freq_depth: Optional number of frequencies to return. If None, returns all frequencies.

    Returns:
        freqs: Vector of frequency values.
        amp_out: Vector of interpolated amplitudes at each frequency.
        phase_out: Vector of interpolated phases at each frequency.

    Notes:
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


def interpolate_fft_spectrum_optimized(afft: AirfoilFFT, Re_val: float, AOA_val: float,
                                     fields: List[str], n_freq_depth: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Optimized version that interpolates multiple fields simultaneously using cached interpolators.

    Args:
        afft: AirfoilFFT struct containing FFT results with cached interpolators.
        Re_val: Desired Reynolds number.
        AOA_val: Desired angle of attack (degrees).
        fields: List of field strings ('CL', 'CD', 'CM', 'CF') to interpolate.
        n_freq_depth: Optional number of frequencies to return. If None, returns all frequencies.

    Returns:
        Dictionary mapping field names to (ST_out, amp_out, phase_out) tuples.
    """
    # Ensure interpolators are cached
    afft._cache_interpolators()

    if n_freq_depth is None:
        n_freq_depth = afft.CL_ST.shape[2]
    else:
        n_freq_depth = min(n_freq_depth, afft.CL_ST.shape[2])

    # Pre-compute the interpolation point as array to avoid repeated array creation
    point = np.array([Re_val, AOA_val])

    results = {}

    for field in fields:
        ST_out = np.zeros(n_freq_depth)
        amp_out = np.zeros(n_freq_depth)
        phase_out = np.zeros(n_freq_depth)

        # Get the appropriate cached interpolators
        if field == 'CL':
            st_interps, amp_interps, pha_interps = afft._cl_st_interps, afft._cl_amp_interps, afft._cl_pha_interps
        elif field == 'CD':
            st_interps, amp_interps, pha_interps = afft._cd_st_interps, afft._cd_amp_interps, afft._cd_pha_interps
        elif field == 'CF':
            st_interps, amp_interps, pha_interps = afft._cf_st_interps, afft._cf_amp_interps, afft._cf_pha_interps
        elif field == 'CM':
            # CM not cached in current implementation, fall back to original method
            return {field: interpolate_fft_spectrum(afft, Re_val, AOA_val, field, n_freq_depth)}
        else:
            raise ValueError(f"Invalid field symbol: {field}")

        # Vectorized interpolation using cached interpolators
        for k in range(n_freq_depth):
            ST_out[k] = st_interps[k](point)
            amp_out[k] = amp_interps[k](point)
            phase_out[k] = pha_interps[k](point)

        results[field] = (ST_out, amp_out, phase_out)

    return results


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