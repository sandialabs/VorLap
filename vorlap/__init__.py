"""
VorLap - Vortex Lattice Method for Wind Turbine Analysis

This package provides tools for analyzing wind turbine aerodynamics using the vortex lattice method.
It includes functionality for loading and processing airfoil data, computing forces and moments,
and analyzing vortex-induced vibrations.
"""

import os

# Get the repository root directory (3 levels up from this file)
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .graphics import calc_structure_vectors_andplot
from .interpolation import interpolate_fft_spectrum, interpolate_fft_spectrum_batch, interpolate_fft_spectrum_optimized, resample_airfoil
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
    mock_compute_thrust_torque_spectrum,
    reconstruct_signal,
    rotate_vector,
    rotationMatrix
)

__version__ = "0.1.0"
