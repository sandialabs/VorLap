"""
VorLap - Vortex Lattice Method for Wind Turbine Analysis

This package provides tools for analyzing wind turbine aerodynamics using the vortex lattice method.
It includes functionality for loading and processing airfoil data, computing forces and moments,
and analyzing vortex-induced vibrations.
"""

from .structs import (
    Component,
    AirfoilFFT,
    VIV_Params
)

from .fileio import (
    load_components_from_csv,
    load_airfoil_fft,
    load_airfoil_coords,
    resample_airfoil,
    interpolate_fft_spectrum,
    interpolate_fft_spectrum_optimized,
    interpolate_fft_spectrum_batch,
    write_force_time_series
)

from .vorlap_utils import (
    compute_thrust_torque_spectrum,
    compute_thrust_torque_spectrum_optimized,
    reconstruct_signal,
    rotate_vector,
    rotationMatrix,
    calc_structure_vectors_andplot
)

__version__ = "0.1.0"
