"""VorLap public package interface."""

import os
import sys


def _resolve_repo_dir() -> str:
    """Return the repository root or PyInstaller extraction root."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and frozen_root:
        return os.path.abspath(str(frozen_root))
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(source_root, "data")):
        return source_root
    installed_data_root = os.path.join(sys.prefix, "share", "vorlap")
    if os.path.isdir(os.path.join(installed_data_root, "data")):
        return os.path.abspath(installed_data_root)
    return source_root


# Repository root directory, or the bundled app extraction root when frozen.
repo_dir = _resolve_repo_dir()

from .graphics import calc_structure_vectors_andplot
from .qblade_controller import (
    ControllerSourceState,
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
from .qblade_bridge_config import QBladeBridgeConfig
from .qblade_bridge_service import QBladeBridgeService
from .qblade_response_model import (
    QBladeResponseDiagnostics,
    QBladeResponseDiagnosticsV2,
    QBladeResponseModel,
    QBladeResponseModelConfig,
    QBladeResponseResult,
    QBladeSeededResponseDiagnostics,
    QBladeSeededResponseDiagnosticsV2,
    QBladeStartupSeedConfig,
)
from .qblade_modal_power_model import (
    ExternalModalPowerTarget,
    QBladeModalPowerDiagnostics,
    QBladeModalPowerLimitError,
    QBladeModalPowerModel,
    QBladeModalPowerModelConfig,
    QBladeModalPowerResult,
    TabulatedAerodynamicPowerCurve,
)
from .qblade_airfoil_work_map import (
    AirfoilForceMotionWorkMap,
    AirfoilWorkMapQuery,
)
from .qblade_spectral_lockin import (
    HVAWTSpectralLockInSource,
    HybridSpectralStepResult,
    ModalDampingBalance,
    SpectralLockInCaptureConfig,
    SpectralLockInModeConfig,
    SpectralPartitionResult,
    evaluate_modal_damping_balance,
    partition_lift_spectrum,
)
from .offline_analysis import VorLapOfflineAnalysisConfig, load_airfoils_from_folder, run_offline_analysis
from .interpolation import AirfoilRangeWarning, interpolate_fft_spectrum, interpolate_fft_spectrum_batch, interpolate_fft_spectrum_optimized, resample_airfoil, lookup_fft_spectrum_nearest
from .structs import (
    Component,
    AirfoilFFT,
    InflowTimeSeries,
    VIV_Params,
    VorLapAnalysisResult,
)

from .fileio import (
    compare_qblade_loading_file,
    convert_qblade_to_vorlap_inputs,
    load_components_from_csv,
    load_airfoil_fft,
    load_airfoil_coords,
    load_qblade_blade_definition,
    load_qblade_simulation_definition,
    load_qblade_turbine_definition,
    read_qblade_loading_file,
    load_inflow_time_series,
    write_components_to_csv,
    write_qblade_loading_file,
    write_force_time_series
)

from .geometry_frames import (
    prepare_component_frames,
    rotate_vector,
    rotationMatrix,
)
from .spectral_reconstruction import reconstruct_nonstationary_signal, reconstruct_signal
from .stationary_sweep import compute_thrust_torque_spectrum, compute_thrust_torque_spectrum_optimized
from .time_varying_history import compute_time_varying_force_history, compute_time_varying_force_history_optimized

__version__ = "1.1.0"

__all__ = [
    "AirfoilFFT",
    "AirfoilForceMotionWorkMap",
    "AirfoilRangeWarning",
    "AirfoilWorkMapQuery",
    "Component",
    "ControllerSourceState",
    "ExternalModalPowerTarget",
    "InflowTimeSeries",
    "QBladeController",
    "QBladeBridgeConfig",
    "QBladeBridgeService",
    "QBladeModalPowerDiagnostics",
    "QBladeModalPowerLimitError",
    "QBladeModalPowerModel",
    "QBladeModalPowerModelConfig",
    "QBladeModalPowerResult",
    "QBladeResponseDiagnostics",
    "QBladeResponseDiagnosticsV2",
    "QBladeResponseModel",
    "QBladeResponseModelConfig",
    "QBladeResponseResult",
    "QBladeSeededResponseDiagnosticsV2",
    "HVAWTSpectralLockInSource",
    "HybridSpectralStepResult",
    "ModalDampingBalance",
    "SwapArrayAdapter",
    "SwapBlockSpec",
    "SpectralLockInCaptureConfig",
    "SpectralLockInModeConfig",
    "SpectralPartitionResult",
    "TabulatedAerodynamicPowerCurve",
    "VIV_Params",
    "VorLapAnalysisResult",
    "VorLapOfflineAnalysisConfig",
    "calc_structure_vectors_andplot",
    "build_external_library_table_spec",
    "build_qblade_external_config",
    "compute_time_varying_force_history",
    "compute_time_varying_force_history_optimized",
    "compute_thrust_torque_spectrum",
    "compute_thrust_torque_spectrum_optimized",
    "compare_qblade_loading_file",
    "convert_qblade_to_vorlap_inputs",
    "create_runtime",
    "evaluate_modal_damping_balance",
    "interpolate_fft_spectrum",
    "interpolate_fft_spectrum_batch",
    "interpolate_fft_spectrum_optimized",
    "load_airfoil_coords",
    "load_airfoil_fft",
    "load_airfoils_from_folder",
    "load_airfoil_fft_directory",
    "load_components_from_csv",
    "load_qblade_blade_definition",
    "load_qblade_simulation_definition",
    "load_qblade_turbine_definition",
    "read_qblade_loading_file",
    "load_inflow_time_series",
    "lookup_fft_spectrum_nearest",
    "prepare_component_frames",
    "reconstruct_nonstationary_signal",
    "reconstruct_signal",
    "resample_airfoil",
    "patch_qblade_structural_definition",
    "patch_qblade_turbine_definition",
    "partition_lift_spectrum",
    "rotate_vector",
    "rotationMatrix",
    "run_offline_analysis",
    "VorLapController",
    "VorLapQBladeRuntime",
    "VorLapQBladeController",
    "write_qblade_external_config",
    "write_components_to_csv",
    "write_qblade_loading_file",
    "write_force_time_series",
]
