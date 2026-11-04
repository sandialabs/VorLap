"""Stationary and nonstationary spectral signal reconstruction helpers."""

from __future__ import annotations

import math
import warnings

import numpy as np


_EPS = 1.0e-12


def _smooth_time_series_adaptive(
    values: np.ndarray,
    freq_ref: np.ndarray,
    tvec: np.ndarray,
    smoothing_cycles: float,
) -> np.ndarray:
    """
    First-order adaptive smoother using a time constant measured in oscillation cycles.

    A constant number of cycles is converted to seconds as tau = cycles / f_ref.
    """
    if smoothing_cycles <= 0.0:
        return values

    out = np.array(values, dtype=float, copy=True)
    freq_ref = np.asarray(freq_ref, dtype=float)
    tvec = np.asarray(tvec, dtype=float)
    for i in range(1, out.shape[0]):
        dt = tvec[i] - tvec[i - 1]
        f_loc = max(0.5 * (freq_ref[i] + freq_ref[i - 1]), _EPS)
        tau = max(smoothing_cycles / f_loc, _EPS)
        alpha = 1.0 - math.exp(-dt / tau)
        out[i] = out[i - 1] + alpha * (out[i] - out[i - 1])
    return out


def reconstruct_nonstationary_signal(
    freqs: np.ndarray,
    amps: np.ndarray,
    phases: np.ndarray,
    tvec: np.ndarray,
    smoothing_cycles: float = 0.0,
) -> np.ndarray:
    """
    Reconstruct a nonstationary harmonic signal with phase continuity.

    This is an oscillator-bank reconstruction where instantaneous phase is
    integrated from frequency (`dphi/dt = 2*pi*f(t)`) to avoid discontinuities
    when inflow conditions vary in time. The provided phase is used as the
    initial phase for each harmonic; subsequent phase evolution comes from
    frequency integration.
    """
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    phases = np.asarray(phases, dtype=float)
    tvec = np.asarray(tvec, dtype=float).reshape(-1)

    if freqs.ndim != 2 or amps.ndim != 2 or phases.ndim != 2:
        raise ValueError("freqs, amps, and phases must be 2D arrays [ntime, nfreq].")
    if freqs.shape != amps.shape or freqs.shape != phases.shape:
        raise ValueError("freqs, amps, and phases must have matching shapes.")
    if freqs.shape[0] != tvec.shape[0]:
        raise ValueError("time vector length must match first dimension of freqs/amps/phases.")
    if tvec.size < 2:
        raise ValueError("tvec must have at least two samples.")

    dt = np.diff(tvec)
    if np.any(dt <= 0.0):
        raise ValueError("tvec must be strictly increasing.")

    fnyq_min = 0.5 / np.max(dt)
    fmax = float(np.nanmax(freqs)) if freqs.size else 0.0
    if fmax > fnyq_min + np.finfo(float).eps * max(1.0, fnyq_min):
        warnings.warn(
            (
                f"Max nonstationary frequency {fmax:.6g} Hz exceeds minimum Nyquist "
                f"{fnyq_min:.6g} Hz implied by tvec; reconstruction may alias."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    ntime, nfreq = freqs.shape
    signal = np.zeros(ntime, dtype=float)
    if nfreq == 0:
        return signal

    # DC term is interpreted as the time-varying mean.
    signal += amps[:, 0]

    for k in range(1, nfreq):
        freq_k = np.maximum(freqs[:, k], 0.0)
        amp_k = amps[:, k]
        if smoothing_cycles > 0.0:
            freq_k = _smooth_time_series_adaptive(freq_k, freq_k, tvec, smoothing_cycles)
            amp_k = _smooth_time_series_adaptive(amp_k, np.maximum(freq_k, _EPS), tvec, smoothing_cycles)

        phase_k = np.zeros(ntime, dtype=float)
        phase_k[0] = phases[0, k]
        for i in range(1, ntime):
            f_mid = 0.5 * (freq_k[i] + freq_k[i - 1])
            phase_k[i] = phase_k[i - 1] + 2.0 * np.pi * f_mid * (tvec[i] - tvec[i - 1])

        signal += amp_k * np.cos(phase_k)

    return signal


def reconstruct_signal(
    freqs: np.ndarray,
    amps: np.ndarray,
    phases: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a time-domain signal from frequency, peak amplitude, and phase.

    DC entries carry the mean value in `amps`. Positive-frequency amplitudes
    are peak amplitudes with a cosine phase convention. Negative frequencies
    are ignored as redundant positive-frequency content.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    tvec = np.asarray(tvec, dtype=np.float64)

    if not (len(freqs) == len(amps) == len(phases)):
        raise ValueError("freqs/amps/phases must be the same length")
    if tvec.ndim != 1:
        raise ValueError("tvec must be one-dimensional")
    if tvec.size < 2:
        raise ValueError("tvec must have at least 2 samples")

    dt_all = np.diff(tvec)
    if np.any(dt_all <= 0.0):
        raise ValueError("tvec must be strictly increasing with positive spacing")
    dt = float(dt_all[0])
    if not np.allclose(dt_all, dt, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("tvec must have uniform spacing for stationary reconstruction")
    fs = 1.0 / dt
    fnyq = 0.5 * fs

    eps = np.finfo(np.float64).eps
    fmax = float(np.max(freqs)) if freqs.size else 0.0
    if fmax > (fnyq + eps * max(1.0, fnyq)):
        warnings.warn(
            (
                f"Max frequency {fmax:.6g} Hz exceeds Nyquist {fnyq:.6g} Hz implied by tvec; "
                "reconstruction may alias."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    signal = np.zeros(tvec.shape[0], dtype=np.float64)

    zero_mask = np.isclose(freqs, 0.0, rtol=0.0, atol=eps)
    if np.any(zero_mask):
        signal += np.sum(amps[zero_mask])

    pos_mask = freqs > 0.0
    if np.any(pos_mask):
        fpos = freqs[pos_mask]
        apos = amps[pos_mask]
        ppos = phases[pos_mask]
        omega = 2.0 * np.pi * fpos
        for w, amplitude, phase in zip(omega, apos, ppos):
            signal += amplitude * np.cos(w * tvec + phase)

    return signal


def one_sided_power_spectrum(
    tvec: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a one-sided periodogram for a uniformly sampled real signal.

    The signal mean is removed before the FFT so the returned positive-frequency
    power describes dynamic content rather than static load bias.
    """
    tvec = np.asarray(tvec, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    if tvec.size != values.size:
        raise ValueError("tvec and values must have the same length")
    if tvec.size < 3:
        raise ValueError("at least three samples are required for spectral analysis")
    if not np.all(np.isfinite(tvec)) or not np.all(np.isfinite(values)):
        raise ValueError("tvec and values must be finite")

    dt_all = np.diff(tvec)
    if np.any(dt_all <= 0.0):
        raise ValueError("tvec must be strictly increasing")
    dt = float(dt_all[0])
    if not np.allclose(dt_all, dt, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("tvec must have uniform spacing for spectral analysis")

    centered = values - float(np.mean(values))
    frequency_hz = np.fft.rfftfreq(tvec.size, d=dt)
    spectrum = np.fft.rfft(centered)
    sample_rate_hz = 1.0 / dt
    power = (np.abs(spectrum) ** 2) / (sample_rate_hz * tvec.size)
    if power.size > 2:
        if tvec.size % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0

    return frequency_hz, power


def _dominant_spectral_peaks_from_power(
    frequency_hz: np.ndarray,
    power: np.ndarray,
    *,
    top_n: int,
    min_frequency_hz: float,
) -> list[dict[str, float]]:
    mask = frequency_hz > float(min_frequency_hz)
    if not np.any(mask):
        return []
    candidate_indices = np.flatnonzero(mask)
    sorted_indices = candidate_indices[np.argsort(power[candidate_indices])[::-1]]
    total_power = float(np.sum(power[mask]))
    peaks: list[dict[str, float]] = []
    for idx in sorted_indices[: max(0, int(top_n))]:
        peak_power = float(power[idx])
        peaks.append(
            {
                "frequency_hz": float(frequency_hz[idx]),
                "power": peak_power,
                "power_fraction": peak_power / total_power if total_power > 0.0 else 0.0,
            }
        )
    return peaks


def dominant_spectral_peaks(
    tvec: np.ndarray,
    values: np.ndarray,
    *,
    top_n: int = 5,
    min_frequency_hz: float = 0.0,
) -> list[dict[str, float]]:
    """Return the strongest positive-frequency bins in descending power order."""
    frequency_hz, power = one_sided_power_spectrum(tvec, values)
    return _dominant_spectral_peaks_from_power(
        frequency_hz,
        power,
        top_n=top_n,
        min_frequency_hz=min_frequency_hz,
    )


def verify_expected_spectral_content(
    tvec: np.ndarray,
    values: np.ndarray,
    expected_frequencies_hz: np.ndarray,
    *,
    tolerance_hz: float | None = None,
    min_power_fraction: float = 0.01,
    min_total_power: float = 1.0e-20,
) -> dict[str, object]:
    """
    Check that expected positive frequencies carry measurable signal power.

    Each expected frequency passes when the summed power in the local frequency
    band is at least `min_power_fraction` of the positive-frequency dynamic
    power. The default tolerance is one FFT bin.
    """
    frequency_hz, power = one_sided_power_spectrum(tvec, values)
    positive = frequency_hz > 0.0
    total_power = float(np.sum(power[positive]))
    if frequency_hz.size > 1:
        bin_width_hz = float(frequency_hz[1] - frequency_hz[0])
    else:
        bin_width_hz = 0.0
    tolerance = float(bin_width_hz if tolerance_hz is None else tolerance_hz)
    expected = sorted({float(freq) for freq in np.asarray(expected_frequencies_hz, dtype=float).reshape(-1) if freq > 0.0})

    comparisons: list[dict[str, object]] = []
    for expected_hz in expected:
        band = np.abs(frequency_hz - expected_hz) <= tolerance + np.finfo(float).eps
        if np.any(band):
            band_indices = np.flatnonzero(band)
            local_index = band_indices[int(np.argmax(power[band_indices]))]
            band_power = float(np.sum(power[band_indices]))
        else:
            local_index = int(np.argmin(np.abs(frequency_hz - expected_hz)))
            band_power = 0.0
        power_fraction = band_power / total_power if total_power > min_total_power else 0.0
        comparisons.append(
            {
                "expected_hz": expected_hz,
                "observed_hz": float(frequency_hz[local_index]),
                "frequency_error_hz": abs(float(frequency_hz[local_index]) - expected_hz),
                "band_power": band_power,
                "power_fraction": power_fraction,
                "passed": bool(total_power > min_total_power and power_fraction >= float(min_power_fraction)),
            }
        )

    return {
        "passed": bool(comparisons) and all(bool(row["passed"]) for row in comparisons),
        "sample_count": int(np.asarray(tvec).size),
        "frequency_resolution_hz": bin_width_hz,
        "tolerance_hz": tolerance,
        "total_dynamic_power": total_power,
        "expected_frequency_count": len(expected),
        "min_power_fraction": float(min_power_fraction),
        "comparisons": comparisons,
    }


def verify_frequency_band_spectral_content(
    tvec: np.ndarray,
    values: np.ndarray,
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
    tolerance_hz: float | None = None,
    min_band_power_fraction: float = 0.95,
    min_total_power: float = 1.0e-20,
    top_n: int = 5,
) -> dict[str, object]:
    """Check that dynamic signal power lies inside an allowed frequency band."""
    frequency_hz, power = one_sided_power_spectrum(tvec, values)
    positive = frequency_hz > 0.0
    total_power = float(np.sum(power[positive]))
    if frequency_hz.size > 1:
        bin_width_hz = float(frequency_hz[1] - frequency_hz[0])
    else:
        bin_width_hz = 0.0
    tolerance = float(bin_width_hz if tolerance_hz is None else tolerance_hz)
    fmin = max(0.0, float(min_frequency_hz) - tolerance)
    fmax = float(max_frequency_hz) + tolerance
    allowed = positive & (frequency_hz >= fmin) & (frequency_hz <= fmax)
    band_power = float(np.sum(power[allowed]))
    band_power_fraction = band_power / total_power if total_power > min_total_power else 0.0
    peaks = _dominant_spectral_peaks_from_power(
        frequency_hz,
        power,
        top_n=top_n,
        min_frequency_hz=0.0,
    )
    dominant_peak = peaks[0] if peaks else None
    dominant_in_band = bool(
        dominant_peak is not None
        and fmin <= float(dominant_peak["frequency_hz"]) <= fmax
    )

    return {
        "passed": bool(total_power > min_total_power and band_power_fraction >= float(min_band_power_fraction) and dominant_in_band),
        "sample_count": int(np.asarray(tvec).size),
        "frequency_resolution_hz": bin_width_hz,
        "tolerance_hz": tolerance,
        "min_frequency_hz": float(min_frequency_hz),
        "max_frequency_hz": float(max_frequency_hz),
        "total_dynamic_power": total_power,
        "band_power": band_power,
        "band_power_fraction": band_power_fraction,
        "min_band_power_fraction": float(min_band_power_fraction),
        "dominant_peak": dominant_peak,
        "dominant_in_band": dominant_in_band,
        "peaks": peaks,
    }


__all__ = [
    "dominant_spectral_peaks",
    "one_sided_power_spectrum",
    "reconstruct_nonstationary_signal",
    "reconstruct_signal",
    "verify_expected_spectral_content",
    "verify_frequency_band_spectral_content",
]
