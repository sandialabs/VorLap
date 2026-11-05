"""Non-GUI offline VorLap analysis service."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import glob
from pathlib import Path
import time
import warnings

import numpy as np

from . import graphics
from .airfoil_io import load_airfoil_fft
from .inflow_io import load_inflow_time_series, write_force_time_series
from .qblade_conversion import convert_qblade_to_vorlap_inputs
from .qblade_loading_file import write_qblade_loading_file
from .stationary_sweep import compute_thrust_torque_spectrum_optimized
from .structs import AirfoilFFT, Component, VIV_Params, VorLapAnalysisResult
from .time_varying_history import compute_time_varying_force_history_optimized


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class VorLapOfflineAnalysisConfig:
    """Inputs required to run one offline VorLap sweep and force-history export."""

    viv_params: VIV_Params
    natural_frequencies: np.ndarray
    inflow_profile_path: str | Path
    components: Sequence[Component] = ()
    use_qblade_geometry: bool = False
    qblade_sim_path: str | Path | None = None
    include_qblade_tower: bool = True
    save_dir: str | Path | None = None
    qblade_loading_path: str | Path | None = None


def load_airfoils_from_folder(airfoil_folder: str | Path) -> dict[str, AirfoilFFT]:
    """Load all airfoil FFT HDF5 files from a folder with the legacy default fallback."""
    affts: dict[str, AirfoilFFT] = {}
    for file in glob.glob(str(Path(airfoil_folder) / "*.h5")):
        afft = load_airfoil_fft(file)
        affts[afft.name] = afft
    if "default" not in affts and affts:
        affts["default"] = next(iter(affts.values()))
    return affts


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _validated_natural_frequencies(values: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(values, dtype=float).reshape(-1)
    if frequencies.size == 0:
        raise ValueError("No natural frequencies loaded.")
    if np.any(~np.isfinite(frequencies)):
        raise ValueError("Natural frequencies must contain only finite values.")
    if np.any(frequencies <= 0.0):
        raise ValueError("Natural frequencies must be positive; omit rigid-body modes.")
    return frequencies


def _resolve_qblade_loading_path(save_dir: Path, qblade_loading_path: str | Path | None) -> Path:
    if qblade_loading_path not in (None, ""):
        loading_file = Path(qblade_loading_path)
        return loading_file if loading_file.is_absolute() else save_dir / loading_file
    return save_dir / "qblade_external_loading.txt"


def run_offline_analysis(
    config: VorLapOfflineAnalysisConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> VorLapAnalysisResult:
    """Run the offline VorLap analysis without depending on Tk widgets."""
    viv_params = config.viv_params
    natfreqs = _validated_natural_frequencies(config.natural_frequencies)
    components = list(config.components)
    qblade_node_ids: tuple[str, ...] | None = None
    qblade_sim_path = ""

    _emit(progress_callback, "Starting VorLap analysis...\n")
    if config.save_dir not in (None, ""):
        _emit(progress_callback, f"  Save directory: {config.save_dir}\n")

    if config.use_qblade_geometry:
        if config.qblade_sim_path in (None, ""):
            raise ValueError(
                "QBlade Geometry is enabled but no QBlade simulation file is set. "
                "Choose a `.sim` file or uncheck QBlade Geometry."
            )
        qblade_sim_path = str(config.qblade_sim_path)
        _emit(progress_callback, f"Converting QBlade model from: {qblade_sim_path}\n")
        components, qblade_viv_params, qblade_ids = convert_qblade_to_vorlap_inputs(
            qblade_sim_path,
            include_tower=bool(config.include_qblade_tower),
        )
        qblade_node_ids = tuple(str(node_id) for node_id in qblade_ids)
        # Keep user sweep settings, but align key fluid/flow frame fields with QBlade.
        viv_params.fluid_density = qblade_viv_params.fluid_density
        viv_params.fluid_dynamicviscosity = qblade_viv_params.fluid_dynamicviscosity
        viv_params.rotation_axis = qblade_viv_params.rotation_axis
        viv_params.rotation_axis_offset = qblade_viv_params.rotation_axis_offset
        viv_params.inflow_vec = qblade_viv_params.inflow_vec
        _emit(
            progress_callback,
            f"Imported {len(components)} components and {len(qblade_node_ids)} QBlade load targets\n",
        )

    if not components:
        raise ValueError("No components loaded.")

    if config.inflow_profile_path in (None, ""):
        raise ValueError(
            "Inflow profile CSV is required for force output. "
            "Provide a file with: time, inflow_speed, inflow_direction_deg."
        )
    inflow_profile = load_inflow_time_series(str(config.inflow_profile_path))
    _emit(
        progress_callback,
        f"Loaded inflow profile with {inflow_profile.time.size} samples: {config.inflow_profile_path}\n",
    )

    affts = load_airfoils_from_folder(viv_params.airfoil_folder)
    if not affts:
        raise ValueError(f"No airfoil data loaded from {viv_params.airfoil_folder}.")
    _emit(progress_callback, f"Loaded {len(affts)} airfoil files from {viv_params.airfoil_folder}\n")

    graphics.calc_structure_vectors_andplot(
        components,
        viv_params,
        show_plot=False,
        return_fig=False,
    )

    _emit(progress_callback, "Running thrust/torque spectrum computation...\n")
    start_time = time.time()
    warning_messages: list[str] = []
    seen_warning_messages: set[str] = set()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        (
            percdiff_matrix,
            percdiff_info,
            total_global_force_vector,
            total_global_moment_vector,
            global_force_vector_nodes,
        ) = compute_thrust_torque_spectrum_optimized(components, affts, viv_params, natfreqs)

        _emit(progress_callback, "Running time-varying force history reconstruction...\n")
        (
            inflow_time,
            tv_total_global_force_vector,
            tv_total_global_moment_vector,
            tv_global_force_vector_nodes,
        ) = compute_time_varying_force_history_optimized(
            components=components,
            affts=affts,
            viv_params=viv_params,
            inflow_profile=inflow_profile,
            azimuth_deg=viv_params.output_azimuth_vinf[0],
            smoothing_cycles=0.25,
        )

    for warning_record in captured_warnings:
        warning_message = str(warning_record.message).strip()
        if not warning_message or warning_message in seen_warning_messages:
            continue
        seen_warning_messages.add(warning_message)
        warning_messages.append(warning_message)
        _emit(progress_callback, f"Warning: {warning_message}\n")
    execution_time = time.time() - start_time
    _emit(progress_callback, f"Computation completed in {execution_time:.4f} seconds\n")

    result = VorLapAnalysisResult(
        percdiff_matrix=percdiff_matrix,
        percdiff_info=percdiff_info,
        total_global_force_vector=total_global_force_vector,
        total_global_moment_vector=total_global_moment_vector,
        global_force_vector_nodes_selected_case=global_force_vector_nodes,
        global_force_vector_nodes=tv_global_force_vector_nodes,
        inflow_time=inflow_time,
        time_varying_total_global_force_vector=tv_total_global_force_vector,
        time_varying_total_global_moment_vector=tv_total_global_moment_vector,
        qblade_node_ids=qblade_node_ids,
        qblade_sim_path=qblade_sim_path,
        viv_params=viv_params,
        components=tuple(components),
        warnings=tuple(warning_messages),
        execution_time_s=execution_time,
        metadata={
            "stationary_include_mean_load": True,
            "time_varying_include_mean_load": True,
        },
    )

    if config.save_dir not in (None, ""):
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        force_file = save_dir / "forces_output.csv"
        write_force_time_series(str(force_file), inflow_time, tv_global_force_vector_nodes)
        result.artifacts["force_time_series"] = str(force_file)
        _emit(progress_callback, f"Force time series saved to: {force_file}\n")

        if qblade_node_ids:
            qblade_loading_file = _resolve_qblade_loading_path(save_dir, config.qblade_loading_path)
            qblade_loading_file.parent.mkdir(parents=True, exist_ok=True)
            write_qblade_loading_file(
                str(qblade_loading_file),
                inflow_time,
                tv_global_force_vector_nodes,
                qblade_node_ids,
                local=False,
            )
            result.artifacts["qblade_loading_file"] = str(qblade_loading_file)
            _emit(progress_callback, f"QBlade loading file saved to: {qblade_loading_file}\n")

    return result
