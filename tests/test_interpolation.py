import warnings

import numpy as np
import pytest

from vorlap.interpolation import (
    AirfoilRangeWarning,
    interpolate_fft_spectrum,
    interpolate_fft_spectrum_batch,
    interpolate_fft_spectra_batch_prepared,
    interpolate_fft_spectra_batch_packed,
    interpolate_fft_spectrum_optimized,
    lookup_fft_spectrum_nearest,
    prepare_fft_spectra_packed_tensor,
)
from vorlap.structs import AirfoilFFT

from conftest import make_linear_airfoil_fft


def _valid_airfoil_kwargs() -> dict:
    re_grid = np.array([1.0, 2.0], dtype=float)
    aoa_grid = np.array([-5.0, 5.0], dtype=float)
    st = np.zeros((2, 2, 2), dtype=float)
    st[:, :, 1] = 0.25
    amp = np.ones((2, 2, 2), dtype=float)
    pha = np.zeros((2, 2, 2), dtype=float)
    return {
        "name": "valid",
        "Re": re_grid,
        "AOA": aoa_grid,
        "Thickness": 0.12,
        "CL_ST": st.copy(),
        "CD_ST": st.copy(),
        "CM_ST": st.copy(),
        "CF_ST": st.copy(),
        "CL_Amp": amp.copy(),
        "CD_Amp": amp.copy(),
        "CM_Amp": amp.copy(),
        "CF_Amp": amp.copy(),
        "CL_Pha": pha.copy(),
        "CD_Pha": pha.copy(),
        "CM_Pha": pha.copy(),
        "CF_Pha": pha.copy(),
    }


def _phase_wrap_airfoil_fft() -> AirfoilFFT:
    re_grid = np.array([1.0], dtype=float)
    aoa_grid = np.array([0.0, 10.0], dtype=float)
    shape = (re_grid.size, aoa_grid.size, 2)
    st = np.zeros(shape, dtype=float)
    st[:, :, 1] = 0.25
    amp = np.ones(shape, dtype=float)
    pha = np.zeros(shape, dtype=float)
    pha[0, :, 1] = np.deg2rad([179.0, -179.0])
    return AirfoilFFT(
        name="phase-wrap",
        Re=re_grid,
        AOA=aoa_grid,
        Thickness=0.12,
        CL_ST=st.copy(),
        CD_ST=st.copy(),
        CM_ST=st.copy(),
        CF_ST=st.copy(),
        CL_Amp=amp.copy(),
        CD_Amp=amp.copy(),
        CM_Amp=amp.copy(),
        CF_Amp=amp.copy(),
        CL_Pha=pha.copy(),
        CD_Pha=pha.copy(),
        CM_Pha=pha.copy(),
        CF_Pha=pha.copy(),
    )


def _assert_phase_near_pi(phase: float) -> None:
    assert np.cos(phase) == pytest.approx(-1.0, abs=1.0e-12)
    assert np.sin(phase) == pytest.approx(0.0, abs=1.0e-12)


def test_airfoil_fft_accepts_signed_dc_and_valid_harmonics():
    kwargs = _valid_airfoil_kwargs()
    kwargs["CL_Amp"][:, :, 0] = -0.5

    afft = AirfoilFFT(**kwargs)

    assert afft.CL_Amp[0, 0, 0] == pytest.approx(-0.5)


def test_airfoil_fft_requires_dc_first():
    kwargs = _valid_airfoil_kwargs()
    kwargs["CL_ST"][:, :, 0] = 0.1

    with pytest.raises(ValueError, match="CL_ST spectral index 0 must be DC"):
        AirfoilFFT(**kwargs)


def test_airfoil_fft_rejects_nonpositive_active_harmonic_strouhal():
    kwargs = _valid_airfoil_kwargs()
    kwargs["CD_ST"][:, :, 1] = 0.0
    kwargs["CD_Amp"][:, :, 1] = 0.1

    with pytest.raises(ValueError, match="positive Strouhal"):
        AirfoilFFT(**kwargs)


def test_airfoil_fft_rejects_negative_harmonic_amplitude():
    kwargs = _valid_airfoil_kwargs()
    kwargs["CF_Amp"][:, :, 1] = -0.1

    with pytest.raises(ValueError, match="harmonic amplitudes"):
        AirfoilFFT(**kwargs)


def test_airfoil_fft_rejects_unsorted_axes():
    kwargs = _valid_airfoil_kwargs()
    kwargs["Re"] = np.array([2.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="strictly increasing"):
        AirfoilFFT(**kwargs)


def test_airfoil_fft_verifies_declared_reynolds_independent_planes():
    kwargs = _valid_airfoil_kwargs()
    kwargs["reynolds_model"] = "replicated_reynolds_independent"
    kwargs["CL_Amp"][1, 0, 0] += 0.01

    with pytest.raises(ValueError, match="not exact replicas"):
        AirfoilFFT(**kwargs)


def test_interpolate_fft_spectrum_bilinear_linear_field():
    afft = make_linear_airfoil_fft()
    st, amp, pha = interpolate_fft_spectrum(afft, Re_val=2.0, AOA_val=5.0, field="CL", n_freq_depth=2)
    np.testing.assert_allclose(st, np.array([0.0, 0.45]), atol=1e-12)
    np.testing.assert_allclose(amp, np.array([16.0, 4.45]), atol=1e-12)
    np.testing.assert_allclose(pha, np.array([0.01, 0.01]), atol=1e-12)


def test_interpolate_fft_spectrum_flat_extrapolation():
    afft = make_linear_airfoil_fft()
    with pytest.warns(AirfoilRangeWarning, match="outside source grid"):
        st, amp, _ = interpolate_fft_spectrum(afft, Re_val=-1.0, AOA_val=30.0, field="CL", n_freq_depth=1)
    # Clamped to Re=1 and AOA=10 -> base=21, amp=25
    np.testing.assert_allclose(st, np.array([0.0]), atol=1e-12)
    np.testing.assert_allclose(amp, np.array([25.0]), atol=1e-12)


def test_interpolate_fft_spectrum_bounds_policy_raise():
    afft = make_linear_airfoil_fft()

    with pytest.raises(ValueError, match="outside source grid"):
        interpolate_fft_spectrum(
            afft,
            Re_val=2.0,
            AOA_val=-90.0,
            field="CL",
            n_freq_depth=1,
            bounds_policy="raise",
        )


def test_strict_policy_warns_only_for_lower_re_on_verified_independent_source():
    kwargs = _valid_airfoil_kwargs()
    kwargs["reynolds_model"] = "replicated_reynolds_independent"
    afft = AirfoilFFT(**kwargs)

    with pytest.warns(AirfoilRangeWarning, match="warning-only"):
        _st, amp, _pha = interpolate_fft_spectrum(
            afft,
            Re_val=0.5,
            AOA_val=0.0,
            field="CL",
            n_freq_depth=1,
            bounds_policy="raise",
        )
    np.testing.assert_allclose(amp, [1.0])

    with pytest.raises(ValueError, match="outside source grid"):
        interpolate_fft_spectrum(
            afft,
            Re_val=3.0,
            AOA_val=0.0,
            field="CL",
            n_freq_depth=1,
            bounds_policy="raise",
        )

    with pytest.raises(ValueError, match="outside source grid"):
        interpolate_fft_spectrum(
            afft,
            Re_val=1.5,
            AOA_val=90.0,
            field="CL",
            n_freq_depth=1,
            bounds_policy="raise",
        )


def test_strict_policy_rejects_lower_re_for_ordinary_source():
    afft = AirfoilFFT(**_valid_airfoil_kwargs())

    with pytest.raises(ValueError, match="outside source grid"):
        interpolate_fft_spectrum(
            afft,
            Re_val=0.5,
            AOA_val=0.0,
            field="CL",
            n_freq_depth=1,
            bounds_policy="raise",
        )


def test_interpolate_fft_spectrum_bounds_policy_clip_is_silent():
    afft = make_linear_airfoil_fft()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st, amp, _ = interpolate_fft_spectrum(
            afft,
            Re_val=10.0,
            AOA_val=30.0,
            field="CL",
            n_freq_depth=1,
            bounds_policy="clip",
        )

    assert not [warning for warning in caught if issubclass(warning.category, AirfoilRangeWarning)]
    np.testing.assert_allclose(st, np.array([0.0]), atol=1e-12)
    np.testing.assert_allclose(amp, np.array([27.0]), atol=1e-12)


def test_interpolate_fft_spectrum_batch_warns_once_for_source_range_clip():
    afft = make_linear_airfoil_fft()

    with pytest.warns(AirfoilRangeWarning, match="outside source grid"):
        st, amp, _ = interpolate_fft_spectrum_batch(
            afft,
            np.array([1.0, 2.0, 5.0]),
            np.array([0.0, -20.0, 5.0]),
            field="CD",
            n_freq_depth=1,
        )

    np.testing.assert_allclose(st[:, 0], 0.0, atol=1e-12)
    assert amp.shape == (3, 1)


def test_interpolate_fft_spectrum_interpolates_phase_circularly_at_wrap():
    afft = _phase_wrap_airfoil_fft()

    _st, _amp, pha = interpolate_fft_spectrum(afft, Re_val=1.0, AOA_val=5.0, field="CL", n_freq_depth=2)

    # Linear radian interpolation of +179 and -179 deg would incorrectly give 0 deg.
    _assert_phase_near_pi(float(pha[1]))


def test_interpolate_fft_spectrum_batch_interpolates_phase_circularly_at_wrap():
    afft = _phase_wrap_airfoil_fft()

    _st, _amp, pha = interpolate_fft_spectrum_batch(
        afft,
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 5.0, 10.0]),
        field="CL",
        n_freq_depth=2,
    )

    assert pha[0, 1] == pytest.approx(np.deg2rad(179.0), abs=1.0e-12)
    _assert_phase_near_pi(float(pha[1, 1]))
    assert pha[2, 1] == pytest.approx(np.deg2rad(-179.0), abs=1.0e-12)


def test_prepared_multi_field_interpolation_is_exactly_equal_to_scalar_batches():
    afft = make_linear_airfoil_fft()
    re_vals = np.array([1.0, 1.5, 2.25, 3.0])
    aoa_vals = np.array([0.0, 2.5, 7.5, 10.0])
    phase_phasors = {
        field: (np.cos(getattr(afft, f"{field}_Pha")), np.sin(getattr(afft, f"{field}_Pha")))
        for field in ("CL", "CD")
    }

    prepared = interpolate_fft_spectra_batch_prepared(
        afft,
        re_vals,
        aoa_vals,
        ["CL", "CD"],
        n_freq_depth=2,
        phase_phasors=phase_phasors,
    )

    for field in ("CL", "CD"):
        reference = interpolate_fft_spectrum_batch(
            afft,
            re_vals,
            aoa_vals,
            field=field,
            n_freq_depth=2,
        )
        for actual, expected in zip(prepared[field], reference):
            np.testing.assert_array_equal(actual, expected)


def test_packed_multi_field_interpolation_exactly_matches_prepared_path():
    afft = make_linear_airfoil_fft()
    reynolds = np.array([1.1, 1.6])
    aoa = np.array([3.0, 7.0])
    fields = ["CL", "CD"]
    prepared = interpolate_fft_spectra_batch_prepared(
        afft,
        reynolds,
        aoa,
        fields,
        n_freq_depth=2,
        bounds_policy="raise",
        phase_phasors={
            field: (
                np.cos(getattr(afft, f"{field}_Pha")),
                np.sin(getattr(afft, f"{field}_Pha")),
            )
            for field in fields
        },
    )
    packed = interpolate_fft_spectra_batch_packed(
        afft,
        reynolds,
        aoa,
        fields,
        n_freq_depth=2,
        bounds_policy="raise",
        packed_tensor=prepare_fft_spectra_packed_tensor(
            afft,
            fields,
            n_freq_depth=2,
        ),
    )
    for field in fields:
        for actual, expected in zip(packed[field], prepared[field]):
            np.testing.assert_array_equal(actual, expected)


def test_prepared_interpolation_preserves_circular_phase_wrap_exactly():
    afft = _phase_wrap_airfoil_fft()
    prepared = interpolate_fft_spectra_batch_prepared(
        afft,
        np.array([1.0]),
        np.array([5.0]),
        ["CL"],
        n_freq_depth=2,
        phase_phasors={"CL": (np.cos(afft.CL_Pha), np.sin(afft.CL_Pha))},
    )
    reference = interpolate_fft_spectrum_batch(
        afft,
        np.array([1.0]),
        np.array([5.0]),
        field="CL",
        n_freq_depth=2,
    )

    for actual, expected in zip(prepared["CL"], reference):
        np.testing.assert_array_equal(actual, expected)
    packed = interpolate_fft_spectra_batch_packed(
        afft,
        np.array([1.0]),
        np.array([5.0]),
        ["CL"],
        n_freq_depth=2,
        bounds_policy="warn",
        packed_tensor=prepare_fft_spectra_packed_tensor(
            afft,
            ["CL"],
            n_freq_depth=2,
        ),
    )
    for actual, expected in zip(packed["CL"], reference):
        np.testing.assert_array_equal(actual, expected)


def test_interpolate_singleton_grid():
    re_grid = np.array([2.0])
    aoa_grid = np.array([5.0])
    st_data = np.array([[[0.0, 2.0]]], dtype=float)
    amp_data = np.array([[[1.0, 2.0]]], dtype=float)
    zeros = np.zeros_like(st_data)
    afft = AirfoilFFT(
        name="single",
        Re=re_grid,
        AOA=aoa_grid,
        Thickness=0.1,
        CL_ST=st_data,
        CD_ST=st_data + np.array([[[0.0, 10.0]]], dtype=float),
        CM_ST=st_data + np.array([[[0.0, 20.0]]], dtype=float),
        CF_ST=st_data + np.array([[[0.0, 30.0]]], dtype=float),
        CL_Amp=amp_data + 40.0,
        CD_Amp=amp_data + 50.0,
        CM_Amp=amp_data + 60.0,
        CF_Amp=amp_data + 70.0,
        CL_Pha=zeros + 0.1,
        CD_Pha=zeros + 0.2,
        CM_Pha=zeros + 0.3,
        CF_Pha=zeros + 0.4,
    )

    with pytest.warns(AirfoilRangeWarning, match="outside source grid"):
        out = interpolate_fft_spectrum_optimized(afft, Re_val=999.0, AOA_val=-999.0, fields=["CL"], n_freq_depth=2)
    st, amp, pha = out["CL"]
    np.testing.assert_allclose(st, np.array([0.0, 2.0]), atol=1e-12)
    np.testing.assert_allclose(amp, np.array([41.0, 42.0]), atol=1e-12)
    np.testing.assert_allclose(pha, np.array([0.1, 0.1]), atol=1e-12)


def test_lookup_nearest_tie_breaks_to_lower_index():
    afft = make_linear_airfoil_fft()
    out = lookup_fft_spectrum_nearest(afft, Re_val=2.0, AOA_val=5.0, fields=["CL"], n_freq_depth=1)
    st, amp, _ = out["CL"]
    # Tie case on both axes chooses lower index -> Re=1, AOA=0.
    np.testing.assert_allclose(st, np.array([0.0]), atol=1e-12)
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
