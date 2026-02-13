import numpy as np
import pytest

from vorlap.interpolation import (
    interpolate_fft_spectrum,
    interpolate_fft_spectrum_batch,
    interpolate_fft_spectrum_optimized,
    lookup_fft_spectrum_nearest,
)
from vorlap.structs import AirfoilFFT

from conftest import make_linear_airfoil_fft


def test_interpolate_fft_spectrum_bilinear_linear_field():
    afft = make_linear_airfoil_fft()
    st, amp, pha = interpolate_fft_spectrum(afft, Re_val=2.0, AOA_val=5.0, field="CL", n_freq_depth=2)
    np.testing.assert_allclose(st, np.array([12.0, -0.05]), atol=1e-12)
    np.testing.assert_allclose(amp, np.array([16.0, 3.95]), atol=1e-12)
    np.testing.assert_allclose(pha, np.array([0.01, 0.01]), atol=1e-12)


def test_interpolate_fft_spectrum_flat_extrapolation():
    afft = make_linear_airfoil_fft()
    st, amp, _ = interpolate_fft_spectrum(afft, Re_val=-1.0, AOA_val=30.0, field="CL", n_freq_depth=1)
    # Clamped to Re=1 and AOA=10 -> base=21, amp=25
    np.testing.assert_allclose(st, np.array([21.0]), atol=1e-12)
    np.testing.assert_allclose(amp, np.array([25.0]), atol=1e-12)


def test_interpolate_singleton_grid():
    re_grid = np.array([2.0])
    aoa_grid = np.array([5.0])
    data = np.array([[[1.0, 2.0]]], dtype=float)
    zeros = np.zeros_like(data)
    afft = AirfoilFFT(
        name="single",
        Re=re_grid,
        AOA=aoa_grid,
        Thickness=0.1,
        CL_ST=data,
        CD_ST=data + 10.0,
        CM_ST=data + 20.0,
        CF_ST=data + 30.0,
        CL_Amp=data + 40.0,
        CD_Amp=data + 50.0,
        CM_Amp=data + 60.0,
        CF_Amp=data + 70.0,
        CL_Pha=zeros + 0.1,
        CD_Pha=zeros + 0.2,
        CM_Pha=zeros + 0.3,
        CF_Pha=zeros + 0.4,
    )

    out = interpolate_fft_spectrum_optimized(afft, Re_val=999.0, AOA_val=-999.0, fields=["CL"], n_freq_depth=2)
    st, amp, pha = out["CL"]
    np.testing.assert_allclose(st, np.array([1.0, 2.0]), atol=1e-12)
    np.testing.assert_allclose(amp, np.array([41.0, 42.0]), atol=1e-12)
    np.testing.assert_allclose(pha, np.array([0.1, 0.1]), atol=1e-12)


def test_lookup_nearest_tie_breaks_to_lower_index():
    afft = make_linear_airfoil_fft()
    out = lookup_fft_spectrum_nearest(afft, Re_val=2.0, AOA_val=5.0, fields=["CL"], n_freq_depth=1)
    st, amp, _ = out["CL"]
    # Tie case on both axes chooses lower index -> Re=1, AOA=0.
    np.testing.assert_allclose(st, np.array([1.0]), atol=1e-12)
    np.testing.assert_allclose(amp, np.array([5.0]), atol=1e-12)


def test_interpolate_batch_supports_cm():
    afft = make_linear_airfoil_fft()
    re_vals = np.array([1.0, 2.0, 3.0])
    aoa_vals = np.array([0.0, 5.0, 10.0])
    st, amp, pha = interpolate_fft_spectrum_batch(afft, re_vals, aoa_vals, field="CM", n_freq_depth=2)
    assert st.shape == (3, 2)
    assert amp.shape == (3, 2)
    assert pha.shape == (3, 2)


def test_interpolate_batch_shape_mismatch_raises():
    afft = make_linear_airfoil_fft()
    with pytest.raises(ValueError, match="same shape"):
        interpolate_fft_spectrum_batch(afft, np.array([1.0, 2.0]), np.array([1.0]), field="CL")
