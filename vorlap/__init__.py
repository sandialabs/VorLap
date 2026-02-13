"""VorLap public package interface."""

import os

# Repository root directory.
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .graphics import calc_structure_vectors_andplot
from .interpolation import interpolate_fft_spectrum, interpolate_fft_spectrum_batch, interpolate_fft_spectrum_optimized, resample_airfoil, lookup_fft_spectrum_nearest
from .structs import (
    Component,
    AirfoilFFT,
    VIV_Params
)

from .fileio import (
    load_components_from_csv,
    load_airfoil_fft,
    load_airfoil_coords,
    write_force_time_series
)

from .computations import (
    compute_thrust_torque_spectrum,
    compute_thrust_torque_spectrum_optimized,
    reconstruct_signal,
    rotate_vector,
    rotationMatrix
)

__version__ = "0.1.0"

__all__ = [
    "AirfoilFFT",
    "Component",
    "VIV_Params",
    "calc_structure_vectors_andplot",
    "compute_thrust_torque_spectrum",
    "compute_thrust_torque_spectrum_optimized",
    "interpolate_fft_spectrum",
    "interpolate_fft_spectrum_batch",
    "interpolate_fft_spectrum_optimized",
    "load_airfoil_coords",
    "load_airfoil_fft",
    "load_components_from_csv",
    "lookup_fft_spectrum_nearest",
    "reconstruct_signal",
    "resample_airfoil",
    "rotate_vector",
    "rotationMatrix",
    "write_force_time_series",
]
