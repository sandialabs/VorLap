"""Airfoil FFT and plotting-coordinate loading helpers."""

import os
import numpy as np
import h5py
import warnings
from .structs import AirfoilFFT


def load_airfoil_fft(
    path: str,
    *,
    repair_legacy_dc_strouhal: bool = True,
    repair_legacy_zero_st_harmonics: bool = True,
) -> AirfoilFFT:
    """
    Loads a processed airfoil unsteady FFT dataset from an HDF5 file.

    Args:
        path: Path to the HDF5 file containing the airfoil FFT data.
        repair_legacy_dc_strouhal: If true, legacy files that store mean
            coefficients at spectral index 0 but have nonzero index-0 Strouhal
            metadata are repaired by forcing only index-0 Strouhal values to
            zero, with a RuntimeWarning.
        repair_legacy_zero_st_harmonics: If true, legacy files that contain
            non-DC amplitudes at zero Strouhal are repaired by zeroing those
            amplitudes and phases, with a RuntimeWarning. Zero-frequency
            non-DC entries cannot represent oscillatory shedding, so dropping
            them is safer than inventing frequencies.

    Returns:
        AirfoilFFT object containing the loaded data.

    Expected HDF5 File Format:
        The file must contain the following datasets:
        - `Airfoilname` :: String — Name of the airfoil (e.g., "NACA0012")
        - `Re` :: Vector{Float64} — Reynolds number values (assumed constant across all entries)
        - `Thickness` :: Vector{Float64} — Thickness ratio(s) used
        - `AOA` :: Vector{Float64} — Angle of attack values in degrees
        - `CL_ST`, `CD_ST`, `CM_ST`, `CF_ST` :: 3D Arrays [Re x AOA x freq] — Strouhal numbers for the legacy CL/CD channels, stored pitching moment metadata, and combined force
        - `CL_Amp`, `CD_Amp`, `CM_Amp`, `CF_Amp` :: 3D Arrays [Re x AOA x freq] — FFT amplitudes for the legacy CL/CD channels, stored pitching moment metadata, and combined force
        - `CL_Pha`, `CD_Pha`, `CM_Pha`, `CF_Pha` :: 3D Arrays [Re x AOA x freq] — FFT phases in radians for each quantity

    Assumptions:
        - All arrays must share dimensions [Re, AOA, NFreq].
        - Spectral index 0 is the mandatory DC/mean coefficient entry and must
          have Strouhal number 0 for CL, CD, CM, and CF.
        - Entries 1..NFreq-1 are oscillatory peaks with nonnegative amplitudes;
          any nonzero harmonic amplitude must have a positive Strouhal number.
        - Phase data is in radians. DC phase is ignored but must be finite.
        - Strouhal data represents unsteady aerodynamics due to vortex shedding.
        - File fields keep CL/CD names for compatibility. Current force
          assembly maps CL to the section-normal coefficient channel and CD to
          the chordwise coefficient channel.
        - No ragged or missing data is allowed.
    """
    with h5py.File(path, 'r') as h5:
        name = h5['Airfoilname'][()] if 'Airfoilname' in h5 else os.path.basename(path)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        reynolds_model = h5.attrs.get("reynolds_model", "")
        if isinstance(reynolds_model, bytes):
            reynolds_model = reynolds_model.decode("utf-8")

        Re = np.asarray(h5['Re'][()], dtype=float).reshape(-1)
        Thickness = np.asarray(h5['Thickness'][()])
        AOA = np.asarray(h5['AOA'][()], dtype=float).reshape(-1)

        CL_ST = h5['CL_ST'][()]
        CD_ST = h5['CD_ST'][()]
        CM_ST = h5['CM_ST'][()]
        CF_ST = h5['CF_ST'][()]

        CL_Amp = h5['CL_Amp'][()]
        CD_Amp = h5['CD_Amp'][()]
        CM_Amp = h5['CM_Amp'][()]
        CF_Amp = h5['CF_Amp'][()]

        CL_Pha = h5['CL_Pha'][()]
        CD_Pha = h5['CD_Pha'][()]
        CM_Pha = h5['CM_Pha'][()]
        CF_Pha = h5['CF_Pha'][()]

        expected_prefix = (len(Re), len(AOA))

        def orient_dimensions(arr: np.ndarray, arr_name: str) -> np.ndarray:
            """Orient tensor dimensions to [Re, AOA, freq]."""
            arr = np.asarray(arr)
            if arr.ndim != 3:
                raise ValueError(f"{arr_name} must be 3D, got shape {arr.shape}")
            if arr.shape[:2] == expected_prefix:
                return arr

            for perm in (
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            ):
                candidate = np.transpose(arr, perm)
                if candidate.shape[:2] == expected_prefix:
                    warnings.warn(
                        f"Transposing {arr_name} from shape {arr.shape} to {candidate.shape}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return candidate

            raise ValueError(
                f"Could not orient {arr_name} to [Re, AOA, freq]. "
                f"Got shape {arr.shape}, expected prefix {expected_prefix}."
            )

        arrays = {
            "CL_ST": orient_dimensions(CL_ST, "CL_ST"),
            "CD_ST": orient_dimensions(CD_ST, "CD_ST"),
            "CM_ST": orient_dimensions(CM_ST, "CM_ST"),
            "CF_ST": orient_dimensions(CF_ST, "CF_ST"),
            "CL_Amp": orient_dimensions(CL_Amp, "CL_Amp"),
            "CD_Amp": orient_dimensions(CD_Amp, "CD_Amp"),
            "CM_Amp": orient_dimensions(CM_Amp, "CM_Amp"),
            "CF_Amp": orient_dimensions(CF_Amp, "CF_Amp"),
            "CL_Pha": orient_dimensions(CL_Pha, "CL_Pha"),
            "CD_Pha": orient_dimensions(CD_Pha, "CD_Pha"),
            "CM_Pha": orient_dimensions(CM_Pha, "CM_Pha"),
            "CF_Pha": orient_dimensions(CF_Pha, "CF_Pha"),
        }

        common_depth = min(arr.shape[2] for arr in arrays.values())
        for arr_name, arr in list(arrays.items()):
            if arr.shape[2] != common_depth:
                warnings.warn(
                    f"Trimming {arr_name} frequency depth from {arr.shape[2]} to {common_depth} for consistency.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            arrays[arr_name] = arr[:, :, :common_depth]

        st_keys = ("CL_ST", "CD_ST", "CM_ST", "CF_ST")
        dc_ok = {
            key: np.allclose(arrays[key][:, :, 0], 0.0, rtol=0.0, atol=1.0e-12)
            for key in st_keys
        }
        if not all(dc_ok.values()) and repair_legacy_dc_strouhal:
            harmonic_has_zero_st = any(
                np.any(np.isclose(arrays[key][:, :, 1:], 0.0, rtol=0.0, atol=1.0e-12))
                for key in st_keys
            )
            if harmonic_has_zero_st:
                raise ValueError(
                    "Airfoil FFT appears to contain a zero-Strouhal entry outside spectral index 0; "
                    "refusing legacy DC repair because the DC location is ambiguous."
                )
            warnings.warn(
                (
                    f"Repairing legacy airfoil FFT '{name}': forcing spectral index 0 "
                    "Strouhal values to 0 for the required DC/mean entry."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            for key in st_keys:
                arrays[key] = arrays[key].copy()
                arrays[key][:, :, 0] = 0.0

        if repair_legacy_zero_st_harmonics:
            for field in ("CL", "CD", "CM", "CF"):
                st_key = f"{field}_ST"
                amp_key = f"{field}_Amp"
                pha_key = f"{field}_Pha"
                st = arrays[st_key]
                amp = arrays[amp_key]
                if st.shape[2] <= 1:
                    continue
                invalid_active = (np.abs(amp[:, :, 1:]) > 1.0e-12) & (st[:, :, 1:] <= 1.0e-12)
                if not np.any(invalid_active):
                    continue
                warnings.warn(
                    (
                        f"Repairing legacy airfoil FFT '{name}': zeroing {field} "
                        "harmonic amplitudes with zero/nonpositive Strouhal metadata."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                arrays[amp_key] = amp.copy()
                arrays[pha_key] = arrays[pha_key].copy()
                arrays[amp_key][:, :, 1:][invalid_active] = 0.0
                arrays[pha_key][:, :, 1:][invalid_active] = 0.0

        for field in ("CL", "CD", "CM", "CF"):
            amp_key = f"{field}_Amp"
            pha_key = f"{field}_Pha"
            amp = arrays[amp_key].copy()
            pha = arrays[pha_key].copy()
            harmonic_amp = amp[:, :, 1:]
            neg = harmonic_amp < 0.0
            if np.any(neg):
                warnings.warn(
                    (
                        f"Converting signed harmonic amplitudes to nonnegative "
                        f"amplitude/phase form for {field} in airfoil FFT '{name}'."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                harmonic_amp[neg] *= -1.0
                harmonic_phase = pha[:, :, 1:]
                harmonic_phase[neg] += np.pi
            arrays[amp_key] = amp
            arrays[pha_key] = pha

        if np.isscalar(Thickness) or np.ndim(Thickness) == 0:
            thickness_value = float(np.asarray(Thickness, dtype=float))
        else:
            thickness_arr = np.asarray(Thickness, dtype=float).reshape(-1)
            thickness_value = float(thickness_arr[0]) if thickness_arr.size else 0.0

        return AirfoilFFT(
            name=name,
            Re=Re,
            AOA=AOA,
            Thickness=thickness_value,
            CL_ST=arrays["CL_ST"],
            CD_ST=arrays["CD_ST"],
            CM_ST=arrays["CM_ST"],
            CF_ST=arrays["CF_ST"],
            CL_Amp=arrays["CL_Amp"],
            CD_Amp=arrays["CD_Amp"],
            CM_Amp=arrays["CM_Amp"],
            CF_Amp=arrays["CF_Amp"],
            CL_Pha=arrays["CL_Pha"],
            CD_Pha=arrays["CD_Pha"],
            CM_Pha=arrays["CM_Pha"],
            CF_Pha=arrays["CF_Pha"],
            reynolds_model=str(reynolds_model),
        )


def load_airfoil_coords(afpath: str = "") -> np.ndarray:
    """
    Loads an airfoil shape from a 2-column text file (x, z), normalized to unit chord length.
    If no file is specified, or if loading fails, returns a built-in 200-point Clark Y airfoil shape.

    Args:
        afpath: Optional path to a text file with two columns: x and z coordinates.

    Returns:
        xy: Nx2 matrix of normalized (x, y) coordinates representing the airfoil surface.

    Notes:
        - If loading from file, x-coordinates are normalized to span [0, 1].
        - The default fallback airfoil is a symmetric approximation of the Clark Y shape.
        - This airfoil is primarily used for visualization, not aerodynamic calculations.
    """
    if afpath and os.path.isfile(afpath):
        try:
            xy = np.loadtxt(afpath, delimiter=',')
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError(f"Expected Nx2 coordinates, got {xy.shape}")
            xy[:, 0] -= np.min(xy[:, 0])
            x_max = np.max(xy[:, 0])
            y_span = np.max(xy[:, 1]) - np.min(xy[:, 1])
            if x_max <= 0.0 or y_span <= 0.0:
                raise ValueError("Airfoil coordinates have zero span in x or y.")
            xy[:, 0] /= x_max
            xy[:, 1] /= y_span
            return xy
        except Exception as e:
            warnings.warn(f"Could not load airfoil file: {e}. Falling back to default Clark Y profile for plotting.")
    else:
        warnings.warn("Could not load airfoil file used for plotting. Falling back to default Clark Y profile for plotting.")

    # Fallback Clark Y airfoil coordinates (from airfoiltools.com)
    xy = np.array([
        [1.0000000, 0.0],
        [0.9900000, 0.0029690],
        [0.9800000, 0.0053335],
        [0.9700000, 0.0076868],
        [0.9600000, 0.0100232],
        [0.9400000, 0.0146239],
        [0.9200000, 0.0191156],
        [0.9000000, 0.0235025],
        [0.8800000, 0.0277891],
        [0.8600000, 0.0319740],
        [0.8400000, 0.0360536],
        [0.8200000, 0.0400245],
        [0.8000000, 0.0438836],
        [0.7800000, 0.0476281],
        [0.7600000, 0.0512565],
        [0.7400000, 0.0547675],
        [0.7200000, 0.0581599],
        [0.7000000, 0.0614329],
        [0.6800000, 0.0645843],
        [0.6600000, 0.0676046],
        [0.6400000, 0.0704822],
        [0.6200000, 0.0732055],
        [0.6000000, 0.0757633],
        [0.5800000, 0.0781451],
        [0.5600000, 0.0803480],
        [0.5400000, 0.0823712],
        [0.5200000, 0.0842145],
        [0.5000000, 0.0858772],
        [0.4800000, 0.0873572],
        [0.4600000, 0.0886427],
        [0.4400000, 0.0897175],
        [0.4200000, 0.0905657],
        [0.4000000, 0.0911712],
        [0.3800000, 0.0915212],
        [0.3600000, 0.0916266],
        [0.3400000, 0.0915079],
        [0.3200000, 0.0911857],
        [0.3000000, 0.0906804],
        [0.2800000, 0.0900016],
        [0.2600000, 0.0890840],
        [0.2400000, 0.0878308],
        [0.2200000, 0.0861433],
        [0.2000000, 0.0839202],
        [0.1800000, 0.0810687],
        [0.1600000, 0.0775707],
        [0.1400000, 0.0734360],
        [0.1200000, 0.0686204],
        [0.1000000, 0.0629981],
        [0.0800000, 0.0564308],
        [0.0600000, 0.0487571],
        [0.0500000, 0.0442753],
        [0.0400000, 0.0391283],
        [0.0300000, 0.0330215],
        [0.0200000, 0.0253735],
        [0.0120000, 0.0178581],
        [0.0080000, 0.0137350],
        [0.0040000, 0.0089238],
        [0.0020000, 0.0058025],
        [0.0010000, 0.0037271],
        [0.0005000, 0.0023390],
        [0.0000000, 0.0000000],
        [0.0005000, -0.0046700],
        [0.0010000, -0.0059418],
        [0.0020000, -0.0078113],
        [0.0040000, -0.0105126],
        [0.0080000, -0.0142862],
        [0.0120000, -0.0169733],
        [0.0200000, -0.0202723],
        [0.0300000, -0.0226056],
        [0.0400000, -0.0245211],
        [0.0500000, -0.0260452],
        [0.0600000, -0.0271277],
        [0.0800000, -0.0284595],
        [0.1000000, -0.0293786],
        [0.1200000, -0.0299633],
        [0.1400000, -0.0302404],
        [0.1600000, -0.0302546],
        [0.1800000, -0.0300490],
        [0.2000000, -0.0296656],
        [0.2200000, -0.0291445],
        [0.2400000, -0.0285181],
        [0.2600000, -0.0278164],
        [0.2800000, -0.0270696],
        [0.3000000, -0.0263079],
        [0.3200000, -0.0255565],
        [0.3400000, -0.0248176],
        [0.3600000, -0.0240870],
        [0.3800000, -0.0233606],
        [0.4000000, -0.0226341],
        [0.4200000, -0.0219042],
        [0.4400000, -0.0211708],
        [0.4600000, -0.0204353],
        [0.4800000, -0.0196986],
        [0.5000000, -0.0189619],
        [0.5200000, -0.0182262],
        [0.5400000, -0.0174914],
        [0.5600000, -0.0167572],
        [0.5800000, -0.0160232],
        [0.6000000, -0.0152893],
        [0.6200000, -0.0145551],
        [0.6400000, -0.0138207],
        [0.6600000, -0.0130862],
        [0.6800000, -0.0123515],
        [0.7000000, -0.0116169],
        [0.7200000, -0.0108823],
        [0.7400000, -0.0101478],
        [0.7600000, -0.0094133],
        [0.7800000, -0.0086788],
        [0.8000000, -0.0079443],
        [0.8200000, -0.0072098],
        [0.8400000, -0.0064753],
        [0.8600000, -0.0057408],
        [0.8800000, -0.0050063],
        [0.9000000, -0.0042718],
        [0.9200000, -0.0035373],
        [0.9400000, -0.0028028],
        [0.9600000, -0.0020683],
        [0.9700000, -0.0017011],
        [0.9800000, -0.0013339],
        [0.9900000, -0.0009666],
        [1.0, 0]
    ])

    xy[:, 0] -= np.min(xy[:, 0])
    xy[:, 0] /= np.max(xy[:, 0])
    xy[:, 1] /= (np.max(xy[:, 1]) - np.min(xy[:, 1]))

    return xy


__all__ = [
    "load_airfoil_coords",
    "load_airfoil_fft",
]
