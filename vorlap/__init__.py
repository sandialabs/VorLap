"""VorLap public package interface."""

import os

# Repository root directory.
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .graphics import calc_structure_vectors_andplot
from .qblade_controller import (
    QBladeController,
    SwapArrayAdapter,
    SwapBlockSpec,
    VorLapController,
    VorLapQBladeController,
)
from .qblade_runtime import (
    VorLapQBladeRuntime,
    build_external_library_table_spec,
    build_qblade_external_config,
    create_runtime,
    load_airfoil_fft_directory,
    patch_qblade_structural_definition,
    patch_qblade_turbine_definition,
    write_qblade_external_config,
)
from .interpolation import interpolate_fft_spectrum, interpolate_fft_spectrum_batch, interpolate_fft_spectrum_optimized, resample_airfoil, lookup_fft_spectrum_nearest
from .structs import (
    Component,
    AirfoilFFT,
    InflowTimeSeries,
    VIV_Params
)

from .fileio import (
    convert_qblade_to_vorlap_inputs,
    load_components_from_csv,
    load_airfoil_fft,
    load_airfoil_coords,
    load_qblade_blade_definition,
    load_qblade_simulation_definition,
    load_qblade_turbine_definition,
    load_inflow_time_series,
    write_components_to_csv,
    write_qblade_loading_file,
    write_force_time_series
)

from .computations import (
    compute_time_varying_force_history,
    compute_time_varying_force_history_optimized,
    compute_thrust_torque_spectrum,
    compute_thrust_torque_spectrum_optimized,
    reconstruct_nonstationary_signal,
    reconstruct_signal,
    rotate_vector,
    rotationMatrix
)

__version__ = "0.1.0"

__all__ = [
    "AirfoilFFT",
    "Component",
    "InflowTimeSeries",
    "QBladeController",
    "SwapArrayAdapter",
    "SwapBlockSpec",
    "VIV_Params",
    "calc_structure_vectors_andplot",
    "build_external_library_table_spec",
    "build_qblade_external_config",
    "compute_time_varying_force_history",
    "compute_time_varying_force_history_optimized",
    "compute_thrust_torque_spectrum",
    "compute_thrust_torque_spectrum_optimized",
    "convert_qblade_to_vorlap_inputs",
    "create_runtime",
    "interpolate_fft_spectrum",
    "interpolate_fft_spectrum_batch",
    "interpolate_fft_spectrum_optimized",
    "load_airfoil_coords",
    "load_airfoil_fft",
    "load_airfoil_fft_directory",
    "load_components_from_csv",
    "load_qblade_blade_definition",
    "load_qblade_simulation_definition",
    "load_qblade_turbine_definition",
    "load_inflow_time_series",
    "lookup_fft_spectrum_nearest",
    "reconstruct_nonstationary_signal",
    "reconstruct_signal",
    "resample_airfoil",
    "patch_qblade_structural_definition",
    "patch_qblade_turbine_definition",
    "rotate_vector",
    "rotationMatrix",
    "VorLapController",
    "VorLapQBladeRuntime",
    "VorLapQBladeController",
    "write_qblade_external_config",
    "write_components_to_csv",
    "write_qblade_loading_file",
    "write_force_time_series",
]
