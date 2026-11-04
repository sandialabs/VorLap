"""Typed batch planning and setup helpers for QBlade/VorLap campaigns."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import copy
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator

from .qblade_case_prep import validate_force_scale
from .qblade_files import (
    read_qblade_keyword,
    render_qblade_value,
    replace_qblade_keyword_file,
    resolve_qblade_path,
)
from .qblade_full_structure_output import (
    OUTPUT_MODE as FULL_STRUCTURE_OUTPUT_MODE,
    configure_full_structure_outputs,
    write_full_structure_filter,
)
from .qblade_process import launch_isolated_process, terminate_process_tree
from .qblade_run_support import (
    atomic_write_json,
    file_sha256,
    find_qblade_library,
    prepend_env_path,
    qblade_library_dirs,
    qblade_runtime_asset_fingerprint,
)
from .qblade_sil import resolve_sil_history_mode


try:
    import fcntl
except ImportError:  # pragma: no cover - Slurm target is Linux.
    fcntl = None


_DEFAULT_REFERENCE_CASE_DIR = "ReferenceHVAWTQBladeModel"
_DEFAULT_REFERENCE_SIM_NAME = "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"

WAKE_PROFILE_REFERENCE = "reference"
WAKE_PROFILE_PARKED_REDUCED_V1 = "parked-reduced-v1"
WAKE_PROFILE_PARKED_REDUCED_V2 = "parked-reduced-v2"
WAKE_PROFILE_CHOICES = (
    WAKE_PROFILE_REFERENCE,
    WAKE_PROFILE_PARKED_REDUCED_V1,
    WAKE_PROFILE_PARKED_REDUCED_V2,
)
_PARKED_REDUCED_V1_RPM = 1.0
_PARKED_REDUCED_V1_TIMESTEP_S = 0.0075
_PARKED_REDUCED_PROFILES = {
    WAKE_PROFILE_PARKED_REDUCED_V1,
    WAKE_PROFILE_PARKED_REDUCED_V2,
}

# The reference profile deliberately leaves the tracked turbine definition
# untouched. The reduced profile is generated-case-only and keeps the LLFVW
# physics active while making the wake-age zoning meaningful for a 1 RPM
# parked/idling rotor. Values are counted in timesteps when WAKELENGTHTYPE=1.
_WAKE_PROFILE_SETTINGS: dict[str, dict[str, float | int]] = {
    WAKE_PROFILE_REFERENCE: {},
    WAKE_PROFILE_PARKED_REDUCED_V1: {
        "WAKELENGTHTYPE": 1,
        "MAXWAKESIZE": 120000,
        "MAXWAKEDIST": 30,
        "WAKEREDUCTION": 0.005,
        "NEARWAKELENGTH": 200,
        "ZONE1LENGTH": 800,
        "ZONE2LENGTH": 2400,
        "ZONE3LENGTH": 6400,
    },
    WAKE_PROFILE_PARKED_REDUCED_V2: {
        # Retain the v1 timestep-based zones, EF integration, panel counts,
        # 120k element cap, and circulation threshold. The additional values
        # are selected from the maintained UNH cross-flow validation setup.
        "DYNSTALLTYPE": 1,
        "TF_OYE": 8,
        "WAKELENGTHTYPE": 1,
        "MAXWAKESIZE": 120000,
        "MAXWAKEDIST": 10,
        "WAKEREDUCTION": 0.005,
        "NEARWAKELENGTH": 200,
        "ZONE1LENGTH": 800,
        "ZONE2LENGTH": 2400,
        "ZONE3LENGTH": 6400,
        "BOUNDCORERADIUS": 1.0,
        "WAKECORERADIUS": 0.05,
        "VORTEXVISCOSITY": 80.0,
    },
}

_WAKE_PROFILE_REASONS = {
    WAKE_PROFILE_PARKED_REDUCED_V1: (
        "Reduce LLFVW cost for the prescribed 1 RPM parked/idling campaign by "
        "counting wake zones in timesteps and bounding far-wake retention while "
        "preserving rollup, shed/trailing vortices, convection, and vortex-core settings."
    ),
    WAKE_PROFILE_PARKED_REDUCED_V2: (
        "Use the accepted 10-diameter LLFVW cutoff and selected maintained UNH "
        "cross-flow validation parameters (Oye TF=8, bound/free-wake core radii, "
        "and vortex viscosity) while retaining the qualified VorLap timestep-based "
        "wake zones, EF integration, panel counts, element cap, and reduction threshold."
    ),
}

_KEYWORD_COMMENTS = {
    "MEANINF": "the mean inflow velocity, overridden if a windfield or hubheight file is use",
    "NUMTIMESTEPS": "the number of timesteps",
    "TIMESTEP": "the timestep size in [s]",
    "PRESCRIBETYPE": "rotor RPM prescribe type (0 = ramp-up; 1 = whole sim; 3 = no RPM prescribed)",
    "RPMPRESCRIBED": "the prescribed rotor RPM [-]",
    "STOREFROM": "the simulation stores data from this point in time, in [s]",
    "STOREREPLAY": "store a replay of the simulation (warning, large memory will be required) [bool]",
    "WNDTYPE": "use a number: 0 = steady; 1 = windfield; 2 = hubheight",
    "WNDNAME": "filename of the turbsim input file, mann input file or hubheight file (with extension), leave blank if unused",
    "HORANGLE": "the horizontal inflow angle [deg]",
    "VERTANGLE": "the vertical inflow angle [deg]",
    "SHEAREXP": "the power-law shear exponent [-]",
    "REFHEIGHT": "the reference height used to construct the boundary-layer profile [m]",
    "DENSITYAIR": "the air density [kg/m^3]",
    "VISCOSITYAIR": "the air kinematic viscosity [m^2/s]",
    "RAMPUP": "the rampup time for the structural model",
    "ADDDAMP": "the initial time with additional damping",
    "STRITERATIONS": "number of iterations for the time integration (used when integrator is HHT or Euler)",
    "DYNSTALLTYPE": "dynamic stall model (0 = none; 1 = OYE; 2 = IAG; 3 = GORMONT-BERG; 4 = ATEFLAP)",
    "TF_OYE": "Tf constant for the OYE dynamic stall model",
    "WAKELENGTHTYPE": "wake age unit (0 = rotor revolutions; 1 = timesteps)",
    "MAXWAKESIZE": "maximum number of wake elements [-]",
    "MAXWAKEDIST": "maximum wake distance from the rotor plane, normalized by diameter [-]",
    "WAKEREDUCTION": "circulation-based wake reduction factor [-]",
    "NEARWAKELENGTH": "near-wake zone length [revolutions or timesteps]",
    "ZONE1LENGTH": "wake zone 1 length [revolutions or timesteps]",
    "ZONE2LENGTH": "wake zone 2 length [revolutions or timesteps]",
    "ZONE3LENGTH": "wake zone 3 length [revolutions or timesteps]",
    "BOUNDCORERADIUS": "fixed bound-vortex core radius, fraction of local chord [-]",
    "WAKECORERADIUS": "initial free-wake core radius, fraction of local chord [-]",
    "VORTEXVISCOSITY": "turbulent vortex viscosity coefficient [-]",
}


def wake_profile_settings(profile: str) -> dict[str, float | int]:
    """Return a copy of the generated-case wake overrides for ``profile``."""
    normalized = str(profile).strip().lower()
    if normalized not in _WAKE_PROFILE_SETTINGS:
        raise ValueError(
            f"Unknown QBlade wake profile {profile!r}; expected one of {', '.join(WAKE_PROFILE_CHOICES)}."
        )
    return dict(_WAKE_PROFILE_SETTINGS[normalized])


@dataclass(frozen=True)
class QBladeBatchCase:
    index: int
    label: str
    speed_label: str
    azimuth_label: str
    wind_speed: float
    azimuth_deg: float
    case_dir: str
    sim: str
    result_dir: str
    result_name: str
    result_filter: str | None
    storage_filter: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "label": self.label,
            "speed_label": self.speed_label,
            "azimuth_label": self.azimuth_label,
            "wind_speed": self.wind_speed,
            "azimuth_deg": self.azimuth_deg,
            "case_dir": self.case_dir,
            "sim": self.sim,
            "result_dir": self.result_dir,
            "result_name": self.result_name,
            "result_filter": self.result_filter,
            "storage_filter": self.storage_filter,
        }


@dataclass(frozen=True)
class QBladeBatchSettings:
    wind_speeds_mps: list[float]
    rpm: float
    rpm_prescribe_type: int
    azimuths_deg: list[float]
    timesteps: int
    timestep: float
    store_from: float
    store_replay: bool
    result_filter_source: str | None
    result_filter_sha256: str | None
    result_filter_channel_count: int
    full_structure_strain_output: bool
    n_freq_depth: int
    source_bounds_policy: str
    source_frequency_truncation: str
    source_frequency_cap_hz: float | None
    force_scale: float
    diagnostic_force_scale: bool
    wind_type: int
    wind_name: str
    horizontal_inflow_angle_deg: float
    vertical_inflow_angle_deg: float
    shear_exponent: float
    reference_height_m: float
    air_density_kg_m3: float
    air_kinematic_viscosity_m2_s: float
    wake_profile: str
    wake_settings: dict[str, float | int]
    stable_support_patch: bool
    rampup: float
    add_damp: float
    str_iterations: int | None
    allow_partial: bool
    coupling: str

    def as_dict(self) -> dict[str, object]:
        return {
            "wind_speeds_mps": self.wind_speeds_mps,
            "rpm": self.rpm,
            "rpm_prescribe_type": self.rpm_prescribe_type,
            "azimuths_deg": self.azimuths_deg,
            "timesteps": self.timesteps,
            "timestep": self.timestep,
            "store_from": self.store_from,
            "store_replay": self.store_replay,
            "result_filter_source": self.result_filter_source,
            "result_filter_sha256": self.result_filter_sha256,
            "result_filter_channel_count": self.result_filter_channel_count,
            "full_structure_strain_output": self.full_structure_strain_output,
            "structural_output_mode": (
                FULL_STRUCTURE_OUTPUT_MODE if self.full_structure_strain_output else "selected-stations"
            ),
            "n_freq_depth": self.n_freq_depth,
            "source_bounds_policy": self.source_bounds_policy,
            "source_frequency_truncation": self.source_frequency_truncation,
            "source_frequency_cap_hz": self.source_frequency_cap_hz,
            "force_scale": self.force_scale,
            "diagnostic_force_scale": self.diagnostic_force_scale,
            "wind_type": self.wind_type,
            "wind_name": self.wind_name,
            "horizontal_inflow_angle_deg": self.horizontal_inflow_angle_deg,
            "vertical_inflow_angle_deg": self.vertical_inflow_angle_deg,
            "shear_exponent": self.shear_exponent,
            "reference_height_m": self.reference_height_m,
            "air_density_kg_m3": self.air_density_kg_m3,
            "air_kinematic_viscosity_m2_s": self.air_kinematic_viscosity_m2_s,
            "wake_profile": self.wake_profile,
            "wake_settings": self.wake_settings,
            "stable_support_patch": self.stable_support_patch,
            "rampup": self.rampup,
            "add_damp": self.add_damp,
            "str_iterations": self.str_iterations,
            "allow_partial": self.allow_partial,
            "coupling": self.coupling,
        }


@dataclass(frozen=True)
class QBladeBatchManifest:
    created_unix: float
    qblade_root: str
    vorlap_root: str
    python: str
    reference_runtime_assets: list[dict[str, object]]
    reference_runtime_asset_fingerprint: str
    settings: QBladeBatchSettings
    cases: list[QBladeBatchCase]

    def as_dict(self) -> dict[str, object]:
        return {
            "created_unix": self.created_unix,
            "qblade_root": self.qblade_root,
            "vorlap_root": self.vorlap_root,
            "python": self.python,
            "reference_runtime_assets": self.reference_runtime_assets,
            "reference_runtime_asset_fingerprint": self.reference_runtime_asset_fingerprint,
            "settings": self.settings.as_dict(),
            "cases": [case.as_dict() for case in self.cases],
        }

    @classmethod
    def from_args(cls, args: Any, *, created_unix: float | None = None) -> "QBladeBatchManifest":
        speeds = parse_wind_speeds(args.wind_speeds)
        azimuths = parse_azimuths(args.azimuths)
        raw_result_filter = getattr(args, "result_filter", None)
        result_filter_source = None
        result_filter_sha256 = None
        result_filter_channel_count = 0
        result_filter_name = None
        if raw_result_filter is not None and str(raw_result_filter).strip():
            result_filter_path = Path(raw_result_filter).expanduser().resolve()
            if not result_filter_path.is_file():
                raise FileNotFoundError(f"QBlade result filter does not exist: {result_filter_path}")
            channels = [
                line.strip()
                for line in result_filter_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith(("#", "!"))
            ]
            if not channels:
                raise ValueError(f"QBlade result filter has no variable names: {result_filter_path}")
            if len(channels) != len(set(channels)):
                raise ValueError(f"QBlade result filter contains duplicate variable names: {result_filter_path}")
            identity = file_sha256(result_filter_path)
            if identity is None:
                raise FileNotFoundError(f"Could not hash QBlade result filter: {result_filter_path}")
            result_filter_source = str(result_filter_path)
            result_filter_sha256 = str(identity["sha256"])
            result_filter_channel_count = len(channels)
            result_filter_name = result_filter_path.name
        coupling = str(args.coupling)
        reference_sim_name = reference_sim_name_for_root(Path(args.vorlap_root))
        reference_sim = reference_case_dir_for_root(Path(args.vorlap_root)) / reference_sim_name
        reference_assets, reference_asset_fingerprint = qblade_runtime_asset_fingerprint(reference_sim)
        cases: list[QBladeBatchCase] = []
        for speed in speeds:
            speed_text = speed_label(speed)
            for azimuth in azimuths:
                azi_text = azimuth_label(azimuth)
                label = speed_text if len(azimuths) == 1 else f"{speed_text}_{azi_text}"
                case_dir = Path(args.run_root) / "cases" / label
                result_dir = Path(args.run_root) / "results" / label
                cases.append(
                    QBladeBatchCase(
                        index=len(cases),
                        label=label,
                        speed_label=speed_text,
                        azimuth_label=azi_text,
                        wind_speed=float(speed),
                        azimuth_deg=float(azimuth),
                        case_dir=str(case_dir),
                        sim=str(case_dir / reference_sim_name),
                        result_dir=str(result_dir),
                        result_name=f"{label}_qblade_results",
                        result_filter=None
                        if result_filter_name is None
                        else str(case_dir / result_filter_name),
                        storage_filter=None
                        if result_filter_name is None
                        else str(
                            case_dir
                            / (
                                f"{Path(result_filter_name).stem}.sil-runtime.filter"
                                if coupling == "sil-vorlap"
                                else result_filter_name
                            )
                        ),
                    )
                )
        frequency_truncation = str(getattr(args, "source_frequency_truncation", "none")).strip().lower()
        if frequency_truncation not in {"none", "nyquist"}:
            raise ValueError("source_frequency_truncation must be 'none' or 'nyquist'.")
        frequency_cap_hz = None
        if frequency_truncation == "nyquist":
            timestep = float(args.timestep)
            if not math.isfinite(timestep) or timestep <= 0.0:
                raise ValueError("timestep must be finite and positive for Nyquist truncation.")
            frequency_cap_hz = 0.5 / timestep
        wake_profile = str(
            getattr(args, "wake_profile", WAKE_PROFILE_REFERENCE)
        ).strip().lower()
        wake_settings = wake_profile_settings(wake_profile)
        if wake_profile in _PARKED_REDUCED_PROFILES:
            if not math.isclose(
                float(args.rpm),
                _PARKED_REDUCED_V1_RPM,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{wake_profile} requires prescribed "
                    f"rpm={_PARKED_REDUCED_V1_RPM:g}; got {float(args.rpm):g}."
                )
            if not math.isclose(
                float(args.timestep),
                _PARKED_REDUCED_V1_TIMESTEP_S,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{wake_profile} requires "
                    f"timestep={_PARKED_REDUCED_V1_TIMESTEP_S:g} s because its wake zones "
                    f"are timestep-counted; got {float(args.timestep):g} s."
                )
        settings = QBladeBatchSettings(
            wind_speeds_mps=speeds,
            rpm=args.rpm,
            rpm_prescribe_type=1,
            azimuths_deg=azimuths,
            timesteps=args.timesteps,
            timestep=args.timestep,
            store_from=args.store_from,
            store_replay=args.store_replay,
            result_filter_source=result_filter_source,
            result_filter_sha256=result_filter_sha256,
            result_filter_channel_count=result_filter_channel_count,
            full_structure_strain_output=bool(
                getattr(args, "full_structure_strain_output", False)
            ),
            n_freq_depth=args.n_freq_depth,
            source_bounds_policy=args.source_bounds_policy,
            source_frequency_truncation=frequency_truncation,
            source_frequency_cap_hz=frequency_cap_hz,
            force_scale=args.force_scale,
            diagnostic_force_scale=args.diagnostic_force_scale,
            wind_type=args.wind_type,
            wind_name=args.wind_name,
            horizontal_inflow_angle_deg=float(getattr(args, "horizontal_inflow_angle", 0.0)),
            vertical_inflow_angle_deg=float(getattr(args, "vertical_inflow_angle", 0.0)),
            shear_exponent=float(getattr(args, "shear_exponent", 0.23)),
            reference_height_m=float(getattr(args, "reference_height", 18.0)),
            air_density_kg_m3=float(getattr(args, "air_density", 1.0296)),
            air_kinematic_viscosity_m2_s=float(
                getattr(args, "air_kinematic_viscosity", 1.647e-5)
            ),
            wake_profile=wake_profile,
            wake_settings=wake_settings,
            stable_support_patch=not args.skip_stable_support_patch,
            rampup=args.rampup,
            add_damp=args.add_damp,
            str_iterations=args.str_iterations,
            allow_partial=args.allow_partial,
            coupling=coupling,
        )
        return cls(
            created_unix=time.time() if created_unix is None else float(created_unix),
            qblade_root=str(args.qblade_root),
            vorlap_root=str(args.vorlap_root),
            python=args.python,
            reference_runtime_assets=reference_assets,
            reference_runtime_asset_fingerprint=reference_asset_fingerprint,
            settings=settings,
            cases=cases,
        )


@dataclass(frozen=True)
class QBladeBatchRunOptions:
    """Typed run options for one QBlade batch case."""

    python: str
    qblade_root: Path
    vorlap_root: Path
    threads: int
    coupling: str
    progress_every: int
    use_opencl: bool = False
    opencl_device: int = 0
    opencl_group_size: int = 32
    qblade_steps: int | None = None
    allow_partial: bool = False
    checkpoint_every: int = 0
    resume_initialize: bool = False
    record_sil_vorlap_history: bool = False
    sil_vorlap_history_mode: str | None = None
    store_project: bool = False
    native_run_full_simulation: bool = False
    result_filter: Path | None = None
    completed_checkpoint_export_only: bool = False

    def __post_init__(self) -> None:
        resolved_history_mode = resolve_sil_history_mode(
            record_history=bool(self.record_sil_vorlap_history),
            history_mode=self.sil_vorlap_history_mode,
        )
        if self.sil_vorlap_history_mode is not None:
            object.__setattr__(self, "sil_vorlap_history_mode", resolved_history_mode)
        if self.coupling != "sil-vorlap" and (
            self.record_sil_vorlap_history or self.sil_vorlap_history_mode is not None
        ):
            raise ValueError(
                "SIL VorLap history options require coupling='sil-vorlap'; "
                f"got coupling={self.coupling!r}."
            )

    @classmethod
    def from_args(cls, args: Any) -> "QBladeBatchRunOptions":
        legacy_record_history = bool(args.record_sil_vorlap_history)
        explicit_history_mode = getattr(args, "sil_vorlap_history_mode", None)
        resolved_history_mode = resolve_sil_history_mode(
            record_history=legacy_record_history,
            history_mode=explicit_history_mode,
        )
        return cls(
            python=str(args.python),
            qblade_root=Path(args.qblade_root),
            vorlap_root=Path(args.vorlap_root),
            threads=int(args.threads),
            coupling=str(args.coupling),
            progress_every=int(args.progress_every),
            use_opencl=bool(args.use_opencl),
            opencl_device=int(args.opencl_device),
            opencl_group_size=int(args.opencl_group_size),
            qblade_steps=None if args.qblade_steps is None else int(args.qblade_steps),
            allow_partial=bool(args.allow_partial),
            checkpoint_every=int(args.checkpoint_every or 0),
            resume_initialize=bool(args.resume_initialize),
            record_sil_vorlap_history=legacy_record_history,
            sil_vorlap_history_mode=(
                resolved_history_mode if explicit_history_mode is not None else None
            ),
            store_project=bool(args.store_project),
            native_run_full_simulation=bool(getattr(args, "native_run_full_simulation", False)),
            result_filter=None
            if not getattr(args, "result_filter", None)
            else Path(args.result_filter).expanduser().resolve(),
            completed_checkpoint_export_only=bool(
                getattr(args, "completed_checkpoint_export_only", False)
            ),
        )

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{self.vorlap_root}{os.pathsep}{self.qblade_root / 'SIL_Interface'}"
            f"{os.pathsep}{env.get('PYTHONPATH', '')}"
        )
        prepend_env_path(env, "LD_LIBRARY_PATH", qblade_library_dirs(self.qblade_root))
        env["OMP_NUM_THREADS"] = str(self.threads)
        return env

    def command_for_case(
        self,
        case: Mapping[str, object],
        *,
        runner_path: Path,
        case_metadata_path: Path,
    ) -> list[str]:
        command = [
            self.python,
            str(runner_path),
            "--qblade-root",
            str(self.qblade_root),
            "--sim",
            str(case["sim"]),
            "--output-dir",
            str(case["result_dir"]),
            "--threads",
            str(self.threads),
            "--result-name",
            str(case["result_name"]),
            "--coupling-mode",
            self.coupling,
            "--case-metadata",
            str(case_metadata_path),
            "--progress-every",
            str(self.progress_every),
        ]
        if self.use_opencl:
            command.extend(
                [
                    "--use-opencl",
                    "--device",
                    str(self.opencl_device),
                    "--group-size",
                    str(self.opencl_group_size),
                ]
            )
        if self.qblade_steps is not None:
            command.extend(["--steps", str(self.qblade_steps)])
        case_filter = case.get("result_filter")
        result_filter = (
            Path(str(case_filter)).resolve()
            if case_filter and Path(str(case_filter)).is_file()
            else self.result_filter
        )
        if result_filter is not None:
            command.extend(["--result-filter", str(result_filter)])
        if self.allow_partial:
            command.append("--allow-partial")
        if self.checkpoint_every:
            command.extend(["--checkpoint-every", str(self.checkpoint_every), "--resume-from-checkpoint"])
            if self.resume_initialize:
                command.append("--resume-initialize")
        if self.coupling == "sil-vorlap":
            command.append("--sil-vorlap")
            if self.record_sil_vorlap_history:
                command.append("--record-sil-vorlap-history")
            elif self.sil_vorlap_history_mode is not None:
                command.extend(
                    ["--sil-vorlap-history-mode", self.sil_vorlap_history_mode]
                )
        if self.store_project:
            command.append("--store-project")
        if self.native_run_full_simulation:
            command.append("--native-run-full-simulation")
        if self.completed_checkpoint_export_only:
            command.append("--completed-checkpoint-export-only")
        return command


def parse_float_spec(spec: str, *, name: str) -> list[float]:
    text = str(spec).strip()
    if not text:
        raise ValueError(f"{name} specification is empty.")
    if text.startswith("linspace:"):
        parts = text.split(":")
        if len(parts) != 4:
            raise ValueError(f"Use linspace:start:stop:count for {name} sweeps.")
        start, stop, count = float(parts[1]), float(parts[2]), int(parts[3])
        if count < 2:
            return [start]
        step = (stop - start) / (count - 1)
        return [start + i * step for i in range(count)]
    if text.startswith("range:"):
        parts = text.split(":")
        if len(parts) != 4:
            raise ValueError(f"Use range:start:stop:step for {name} sweeps.")
        start, stop, step = float(parts[1]), float(parts[2]), float(parts[3])
        if step == 0.0:
            raise ValueError(f"{name} range step cannot be zero.")
        if (stop - start) * step < 0.0:
            raise ValueError(f"{name} range step has the wrong sign for the requested stop value.")
        values: list[float] = []
        value = start
        tolerance = abs(step) * 1.0e-9 + 1.0e-12
        if step > 0.0:
            while value <= stop + tolerance:
                values.append(value)
                value += step
        else:
            while value >= stop - tolerance:
                values.append(value)
                value += step
        return values
    values = [part for part in re.split(r"[\s,]+", text) if part]
    return [float(value) for value in values]


def parse_wind_speeds(spec: str) -> list[float]:
    return parse_float_spec(spec, name="wind-speed")


def parse_azimuths(spec: str) -> list[float]:
    azimuths = parse_float_spec(spec, name="azimuth")
    for azimuth in azimuths:
        if not math.isfinite(azimuth) or azimuth < 0.0 or azimuth >= 360.0:
            raise ValueError(f"Azimuths must be finite values in [0, 360) deg; got {azimuth!r}.")
    return azimuths


def speed_label(speed: float) -> str:
    text = f"{speed:05.2f}".replace(".", "p")
    return f"U{text}ms"


def azimuth_label(azimuth_deg: float) -> str:
    text = f"{azimuth_deg:06.2f}".replace(".", "p")
    return f"A{text}deg"


def reference_case_dir_for_root(vorlap_root: Path) -> Path:
    """Return the repository reference QBlade case directory.

    The 10 m H-VAWT example is the default. Other model directories must be
    selected explicitly with VORLAP_QBLADE_REFERENCE_CASE.
    """
    root = Path(vorlap_root)
    selected = os.environ.get("VORLAP_QBLADE_REFERENCE_CASE", _DEFAULT_REFERENCE_CASE_DIR)
    reference = root / "examples" / selected
    if reference.is_dir():
        return reference
    raise FileNotFoundError(
        f"VorLap reference QBlade model not found: {reference}. "
        f"Set VORLAP_QBLADE_REFERENCE_CASE to a directory below {root / 'examples'} if needed."
    )


def reference_sim_name_for_root(vorlap_root: Path) -> str:
    """Return the simulation filename for the selected reference case."""
    override = os.environ.get("VORLAP_QBLADE_REFERENCE_SIM")
    if override:
        return Path(override).name
    reference = reference_case_dir_for_root(vorlap_root)
    if (reference / _DEFAULT_REFERENCE_SIM_NAME).is_file():
        return _DEFAULT_REFERENCE_SIM_NAME
    sims = sorted(path.name for path in reference.glob("*.sim"))
    if len(sims) == 1:
        return sims[0]
    if not sims:
        raise FileNotFoundError(f"No QBlade .sim file found in reference case: {reference}")
    raise FileNotFoundError(
        f"Multiple QBlade .sim files found in {reference}; set VORLAP_QBLADE_REFERENCE_SIM explicitly."
    )


def turbine_definition_path_for_sim(sim_path: Path) -> Path:
    """Resolve the QBlade TURBFILE path referenced by a simulation file."""
    turbine_file = read_qblade_keyword(sim_path, "TURBFILE")
    if turbine_file is None or not turbine_file.strip():
        raise ValueError(f"Could not find TURBFILE in {sim_path}")
    return resolve_qblade_path(sim_path, turbine_file)


def turbine_dir_for_sim(sim_path: Path) -> Path:
    """Return the model directory containing the TURBFILE referenced by a .sim."""
    return turbine_definition_path_for_sim(sim_path).parent


def vorlap_airfoil_dir_for_sim(sim_path: Path) -> Path:
    """Return the VorLapAirfoils directory for a generated QBlade case."""
    turbine_dir = turbine_dir_for_sim(sim_path)
    direct = turbine_dir / "VorLapAirfoils"
    if direct.is_dir():
        return direct
    matches = sorted(turbine_dir.rglob("VorLapAirfoils"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"VorLapAirfoils directory not found below {turbine_dir}")


def structural_file_for_sim(sim_path: Path) -> Path | None:
    """Resolve the turbine structural model file when the .trb defines one."""
    turbine_file = turbine_definition_path_for_sim(sim_path)
    structural_file = read_qblade_keyword(turbine_file, "STRUCTURALFILE")
    if structural_file is None or not structural_file.strip():
        return None
    return resolve_qblade_path(turbine_file, structural_file)


def qblade_aerodynamic_reference_issues(sim_path: Path) -> list[str]:
    """Return missing or unreadable QBlade blade, polar, and airfoil references."""
    sim = Path(sim_path)
    issues: list[str] = []
    try:
        turbine_file = turbine_definition_path_for_sim(sim)
    except (FileNotFoundError, ValueError) as exc:
        return [str(exc)]
    if not turbine_file.is_file():
        return [f"missing QBlade turbine definition: {turbine_file}"]
    if not os.access(turbine_file, os.R_OK):
        return [f"unreadable QBlade turbine definition: {turbine_file}"]

    blade_value = read_qblade_keyword(turbine_file, "BLADEFILE")
    if blade_value is None or not blade_value.strip():
        return issues
    blade_file = resolve_qblade_path(turbine_file, blade_value)
    if not blade_file.is_file():
        return [f"missing QBlade blade definition: {blade_file}"]
    if not os.access(blade_file, os.R_OK):
        return [f"unreadable QBlade blade definition: {blade_file}"]

    blade_text = blade_file.read_text(encoding="utf-8", errors="ignore")
    polar_values = sorted(set(re.findall(r"(?i)(?<!\S)(\S+\.plr)(?!\S)", blade_text)))
    for polar_value in polar_values:
        polar_file = resolve_qblade_path(blade_file, polar_value)
        if not polar_file.is_file():
            issues.append(f"missing QBlade polar file: {polar_file}")
            continue
        if not os.access(polar_file, os.R_OK):
            issues.append(f"unreadable QBlade polar file: {polar_file}")
            continue
        foil_value = read_qblade_keyword(polar_file, "FOILNAME")
        if foil_value is None or not foil_value.strip():
            issues.append(f"QBlade polar has no FOILNAME reference: {polar_file}")
            continue
        foil_file = resolve_qblade_path(polar_file, foil_value)
        if not foil_file.is_file():
            issues.append(f"missing QBlade airfoil geometry file: {foil_file}")
        elif not os.access(foil_file, os.R_OK):
            issues.append(f"unreadable QBlade airfoil geometry file: {foil_file}")
    return issues


def build_case_manifest(args: Any) -> dict[str, object]:
    return QBladeBatchManifest.from_args(args).as_dict()


def comparable_case_manifest(manifest: dict[str, object]) -> dict[str, object]:
    comparable = copy.deepcopy(manifest)
    comparable.pop("created_unix", None)
    for case in comparable.get("cases", []):
        if isinstance(case, dict):
            case.pop("runtime_assets", None)
            case.pop("runtime_asset_fingerprint", None)
            case.pop("storage_filter_sha256", None)
            case.pop("storage_filter_channel_count", None)
            case.pop("storage_filter_added_channels", None)
            case.pop("full_structure_strain_output", None)
    return comparable


def manifest_matches_setup_inputs(existing_manifest: dict[str, object], planned_manifest: dict[str, object]) -> bool:
    return comparable_case_manifest(existing_manifest) == comparable_case_manifest(planned_manifest)


def qblade_setting_matches(actual: str | None, expected: str) -> bool:
    if actual == expected:
        return True
    if actual is None:
        return False
    try:
        return math.isclose(float(actual), float(expected), rel_tol=1.0e-10, abs_tol=1.0e-12)
    except ValueError:
        return False


def qblade_filter_channels(path: Path) -> list[str]:
    """Read ordered, non-comment QBlade filter channels."""
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]


def write_sil_storage_filter(
    *,
    result_filter: Path,
    storage_filter: Path,
    parameter_file: Path,
) -> dict[str, object]:
    """Retain SIL input channels in QBlade while exportResults stays compact."""
    config = json.loads(Path(parameter_file).read_text(encoding="utf-8"))
    rows = config.get("external_input_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"VorLap external config has no external_input_rows: {parameter_file}")

    result_channels = qblade_filter_channels(result_filter)
    required_channels: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"Invalid external input row in {parameter_file}: {row!r}")
        variable = str(row.get("variable", "")).strip()
        if not variable or variable == "Timestep [-]":
            continue
        required_channels.append(variable)

    channels = list(result_channels)
    seen = set(channels)
    added: list[str] = []
    for variable in required_channels:
        if variable in seen:
            continue
        seen.add(variable)
        channels.append(variable)
        added.append(variable)

    storage_filter.parent.mkdir(parents=True, exist_ok=True)
    storage_filter.write_text("\n".join(channels) + "\n", encoding="utf-8")
    identity = file_sha256(storage_filter)
    if identity is None:
        raise FileNotFoundError(f"Could not hash generated QBlade storage filter: {storage_filter}")
    return {
        "path": str(storage_filter),
        "sha256": str(identity["sha256"]),
        "channel_count": len(channels),
        "added_channels": added,
    }


def setup_artifact_issues(manifest: dict[str, object]) -> list[str]:
    """Return generated-case settings that are missing or stale for a manifest."""
    settings = manifest.get("settings")
    cases = manifest.get("cases")
    if not isinstance(settings, Mapping) or not isinstance(cases, list):
        return ["case manifest is missing its settings or cases list"]

    expected_common = {
        "PRESCRIBETYPE": settings.get("rpm_prescribe_type", 1),
        "RPMPRESCRIBED": settings.get("rpm"),
        "NUMTIMESTEPS": settings.get("timesteps"),
        "TIMESTEP": settings.get("timestep"),
        "STOREFROM": settings.get("store_from"),
        "STOREREPLAY": str(settings.get("store_replay")).lower(),
        "WNDTYPE": settings.get("wind_type"),
        "HORANGLE": settings.get("horizontal_inflow_angle_deg"),
        "VERTANGLE": settings.get("vertical_inflow_angle_deg"),
        "SHEAREXP": settings.get("shear_exponent"),
        "REFHEIGHT": settings.get("reference_height_m"),
        "DENSITYAIR": settings.get("air_density_kg_m3"),
        "VISCOSITYAIR": settings.get("air_kinematic_viscosity_m2_s"),
    }
    result_filter_source = str(settings.get("result_filter_source", "") or "")
    wind_name_template = str(settings.get("wind_name", "") or "")
    issues: list[str] = []
    for raw_case in cases:
        if not isinstance(raw_case, Mapping):
            issues.append("case manifest contains a non-mapping case entry")
            continue
        sim_path = Path(str(raw_case.get("sim", "")))
        if not sim_path.is_file():
            issues.append(f"missing generated simulation file: {sim_path}")
            continue

        expected = {
            **expected_common,
            "MEANINF": raw_case.get("wind_speed"),
            "INITIAL_AZIMUTH": raw_case.get("azimuth_deg"),
        }
        if result_filter_source:
            result_filter_path = Path(str(raw_case.get("result_filter", "")))
            storage_filter_path = Path(
                str(raw_case.get("storage_filter", "") or result_filter_path)
            )
            expected["FILTERFILE"] = storage_filter_path.name
            if not result_filter_path.is_file():
                issues.append(f"missing generated QBlade result filter: {result_filter_path}")
            else:
                identity = file_sha256(result_filter_path)
                expected_sha256 = str(settings.get("result_filter_sha256", "") or "")
                observed_sha256 = "" if identity is None else str(identity.get("sha256", "") or "")
                if not expected_sha256 or observed_sha256 != expected_sha256:
                    issues.append(
                        f"{result_filter_path}: result-filter SHA-256 is {observed_sha256!r}, "
                        f"expected {expected_sha256!r}"
                    )
            if not storage_filter_path.is_file():
                issues.append(f"missing generated QBlade storage filter: {storage_filter_path}")
            else:
                identity = file_sha256(storage_filter_path)
                expected_sha256 = str(raw_case.get("storage_filter_sha256", "") or "")
                observed_sha256 = "" if identity is None else str(identity.get("sha256", "") or "")
                if not expected_sha256 or observed_sha256 != expected_sha256:
                    issues.append(
                        f"{storage_filter_path}: storage-filter SHA-256 is {observed_sha256!r}, "
                        f"expected {expected_sha256!r}"
                    )
        if wind_name_template:
            expected["WNDNAME"] = wind_name_template.format(
                speed=float(raw_case["wind_speed"]),
                label=raw_case["label"],
                speed_label=raw_case["speed_label"],
                azimuth=float(raw_case["azimuth_deg"]),
                azimuth_label=raw_case["azimuth_label"],
            )
        for keyword, value in expected.items():
            expected_value = render_qblade_value(value)
            actual_value = read_qblade_keyword(sim_path, keyword)
            if not qblade_setting_matches(actual_value, expected_value):
                issues.append(
                    f"{sim_path}: {keyword} is {actual_value!r}, expected {expected_value!r}"
                )
        wake_settings = settings.get("wake_settings", {})
        if isinstance(wake_settings, Mapping) and wake_settings:
            try:
                turbine_path = turbine_definition_path_for_sim(sim_path)
            except (FileNotFoundError, ValueError) as exc:
                issues.append(str(exc))
            else:
                for keyword, value in wake_settings.items():
                    expected_value = render_qblade_value(value)
                    actual_value = read_qblade_keyword(turbine_path, str(keyword))
                    if not qblade_setting_matches(actual_value, expected_value):
                        issues.append(
                            f"{turbine_path}: {keyword} is {actual_value!r}, expected {expected_value!r}"
                        )
        if read_qblade_keyword(sim_path, "TURBFILE") is not None:
            issues.extend(qblade_aerodynamic_reference_issues(sim_path))
        recorded_assets = raw_case.get("runtime_assets")
        recorded_fingerprint = raw_case.get("runtime_asset_fingerprint")
        actual_assets, actual_fingerprint = qblade_runtime_asset_fingerprint(sim_path)
        if not isinstance(recorded_assets, list) or not recorded_fingerprint:
            issues.append(f"{sim_path}: generated runtime-asset fingerprint is missing")
        elif str(recorded_fingerprint) != actual_fingerprint:
            issues.append(
                f"{sim_path}: generated runtime-asset fingerprint is {actual_fingerprint}, "
                f"expected recorded {recorded_fingerprint}"
            )
        missing_assets = [str(row.get("path")) for row in actual_assets if not row.get("sha256")]
        for missing_asset in missing_assets:
            issues.append(f"{sim_path}: runtime asset is missing or unreadable: {missing_asset}")
    return issues


def find_bridge_library(search_root: Path) -> Path | None:
    """Find an installed VorLap/QBlade bridge library below a build or install root."""
    names = (
        "libvorlap_qblade_bridge.so",
        "libvorlap_qblade_bridge.dylib",
        "libvorlap_qblade_bridge.dll",
    )
    for name in names:
        matches = sorted(search_root.rglob(name))
        if matches:
            return matches[0]
    return None


def validate_bridge_library_loadable(library_path: Path) -> None:
    """Fail if the installed native bridge cannot be loaded by this process."""
    path = Path(library_path)
    if not path.is_file():
        raise FileNotFoundError(f"VorLap bridge library does not exist: {path}")
    ctypes.CDLL(str(path))


def validate_qblade_root(qblade_root: Path) -> None:
    """Validate the QBlade root pieces needed by the batch setup and runner."""
    root = Path(qblade_root)
    missing: list[str] = []
    for label, path in (
        ("QBlade SIL_Interface directory", root / "SIL_Interface"),
        ("QBlade ControllerFiles directory", root / "ControllerFiles"),
    ):
        if not path.is_dir():
            missing.append(f"missing {label}: {path}")
    try:
        find_qblade_library(root)
    except FileNotFoundError:
        patterns = "*QBlade*.so*, *QBlade*.dll, *QBlade*.dylib"
        missing.append(f"missing QBlade shared library in {root} matching one of: {patterns}")
    if missing:
        searched = [
            str(root / "SIL_Interface"),
            str(root / "ControllerFiles"),
            f"{root}/*QBlade*.so*",
            f"{root}/*QBlade*.dll",
            f"{root}/*QBlade*.dylib",
        ]
        raise FileNotFoundError(
            "Incomplete QBlade root. Searched: "
            + "; ".join(searched)
            + ". Missing: "
            + "; ".join(missing)
        )


def run_checked(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    process = launch_isolated_process(command, cwd=cwd, env=env)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        terminate_process_tree(process)
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


@contextmanager
def setup_lock(run_root: Path) -> Iterator[None]:
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / "setup.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def replace_qblade_keyword(path: Path, keyword: str, value: float | int | str) -> None:
    comment = _KEYWORD_COMMENTS.get(keyword, "")
    replace_qblade_keyword_file(path, keyword, value, comment=comment)


def replace_structural_keyword(path: Path, keyword: str, value: float | int | str) -> dict[str, str]:
    """Replace value-leading QBlade structural file keywords."""
    return replace_qblade_keyword_file(
        path,
        keyword,
        value,
        comment=None,
        value_width=42,
        keyword_width=0,
    )


def relative_case_path(case_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(case_dir))
    except ValueError:
        return str(path)


def copy_reference_case(vorlap_root: Path, output_dir: Path, force: bool) -> Path:
    """Copy the repository's reference QBlade model into a generated case directory."""
    source = reference_case_dir_for_root(vorlap_root)
    sim_name = reference_sim_name_for_root(vorlap_root)
    sim_source = source / sim_name
    if not sim_source.is_file():
        raise FileNotFoundError(f"VorLap reference QBlade simulation file not found: {sim_source}")
    aerodynamic_issues = qblade_aerodynamic_reference_issues(sim_source)
    if aerodynamic_issues:
        details = "\n".join(f"  - {issue}" for issue in aerodynamic_issues)
        raise FileNotFoundError(
            f"Incomplete QBlade aerodynamic reference model: {sim_source}\n{details}"
        )
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    ignored = shutil.ignore_patterns(".DS_Store", "__pycache__", "*.pyc")
    for child in source.iterdir():
        if child.name == ".DS_Store":
            continue
        destination = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, ignore=ignored)
        else:
            shutil.copy2(child, destination)
    return output_dir / sim_name


def apply_stable_support_patch(
    case_dir: Path,
    sim_path: Path,
    timestep: float,
    rampup: float,
    add_damp: float,
    str_iterations: int | None,
) -> None:
    """Patch the reference case to a headless-stable parked support configuration."""
    turbine_dir = turbine_dir_for_sim(sim_path)
    main_structure = structural_file_for_sim(sim_path)
    structural_patches = []
    structural_note = ""
    remove_subfile = parse_bool_env(os.environ.get("VORLAP_STABLE_PATCH_REMOVE_SUBFILE"), default=False)
    if main_structure is not None and remove_subfile:
        structural_patches.append(replace_structural_keyword(main_structure, "SUBFILE", ""))
    elif main_structure is None:
        structural_note = "No STRUCTURALFILE was defined by the selected QBlade turbine."
    else:
        structural_note = (
            "The structural SUBFILE is preserved. Set VORLAP_STABLE_PATCH_REMOVE_SUBFILE=1 "
            "only when substructure removal is an intended model change."
        )

    sim_patches = []
    for keyword, value in (
        ("RELAXSTEPS", 0),
        ("RELAXTIMESTEP", timestep),
        ("RAMPUP", rampup),
        ("ADDDAMP", add_damp),
    ):
        replace_qblade_keyword(sim_path, keyword, value)
        sim_patches.append({"file": str(sim_path), "keyword": keyword, "new_value": render_qblade_value(value)})
    if str_iterations is not None:
        replace_qblade_keyword(sim_path, "STRITERATIONS", str_iterations)
        sim_patches.append(
            {"file": str(sim_path), "keyword": "STRITERATIONS", "new_value": render_qblade_value(str_iterations)}
        )

    for row in structural_patches + sim_patches:
        row["file"] = relative_case_path(case_dir, Path(row["file"]))

    manifest_path = turbine_dir / "Control" / "vorlap_stable_support_patch.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        manifest_path,
        {
            "applied": True,
            "scope": "generated_case_only",
            "reason": (
                "Setup records generated-case-only parked solver keyword changes here. "
                "Substructure removal changes the structural model and is applied only "
                "when explicitly requested with VORLAP_STABLE_PATCH_REMOVE_SUBFILE=1."
            ),
            "structural_patches": structural_patches,
            "structural_note": structural_note,
            "legacy_subfile_removal": remove_subfile,
            "sim_patches": sim_patches,
            "settings": {
                "preserved_rotor_support_keywords": [
                    "TWRHEIGHT",
                    "TRQTBCLEAR",
                    "HUBPOS",
                    "TRQTBCONN",
                    "RTRCLEAR",
                ],
                "relax_steps": 0,
                "relax_timestep_s": timestep,
                "rampup_s": rampup,
                "add_damp_s": add_damp,
                "str_iterations": str_iterations,
            },
        },
    )


def parse_bool_env(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def pip_trusted_hosts() -> list[str]:
    hosts = os.environ.get("VORLAP_PIP_TRUSTED_HOSTS", "pypi.org files.pythonhosted.org pypi.python.org")
    return [host for host in re.split(r"[\s,]+", hosts.strip()) if host]


def pip_trusted_host_args() -> list[str]:
    args: list[str] = []
    for host in pip_trusted_hosts():
        args.extend(["--trusted-host", host])
    return args


def install_vorlap(args: Any) -> None:
    if args.skip_vorlap_install:
        return
    command = [args.python, "-m", "pip", "install", "--no-deps"]
    if parse_bool_env(os.environ.get("VORLAP_PIP_NO_BUILD_ISOLATION"), default=True):
        command.append("--no-build-isolation")
    command.extend(pip_trusted_host_args())
    command.extend(["-e", str(args.vorlap_root)])
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    if pip_trusted_hosts():
        env.setdefault("PIP_TRUSTED_HOST", " ".join(pip_trusted_hosts()))
    run_checked(command, env=env)


def build_bridge(args: Any) -> None:
    if args.skip_bridge_build:
        print("Skipping VorLap bridge build.", flush=True)
        return
    lock_root = args.qblade_root / "ControllerFiles" / ".vorlap_bridge_build"
    with setup_lock(lock_root):
        _build_bridge_locked(args)


def _build_bridge_locked(args: Any) -> None:
    """Validate or build the bridge while holding the QBlade-root setup lock."""
    install_dir = args.qblade_root / "ControllerFiles"
    build_dir = args.run_root / "build" / "qblade_external_linux"
    existing = install_dir / "libvorlap_qblade_bridge.so"
    force_build = bool(args.force_bridge_build)
    force_token = os.environ.get("VORLAP_FORCE_SETUP_TOKEN", "").strip()
    force_marker = install_dir / ".vorlap_bridge_build" / "force-token"
    if force_build and force_token and force_marker.is_file():
        try:
            completed_force_token = json.loads(
                force_marker.read_text(encoding="utf-8")
            ).get("token")
        except (json.JSONDecodeError, OSError, AttributeError):
            completed_force_token = None
        if completed_force_token == force_token:
            force_build = False
            print(
                f"Forced bridge setup already completed for job token {force_token}; validating installed bridge.",
                flush=True,
            )
    if existing.is_file() and not force_build:
        print(f"Using existing VorLap bridge: {existing}", flush=True)
        try:
            validate_bridge_library_loadable(existing)
            return
        except OSError as exc:
            print(
                f"Existing VorLap bridge is not loadable with {args.python}: {exc}. Rebuilding.",
                file=sys.stderr,
                flush=True,
            )
    if not os.access(install_dir, os.W_OK):
        raise PermissionError(f"QBlade ControllerFiles directory is not writable: {install_dir}")

    script = args.vorlap_root / "scripts" / "build_qblade_external_linux.sh"
    env = os.environ.copy()
    env["PYTHON_EXE"] = args.python
    run_checked([str(script), str(build_dir), str(install_dir)], cwd=args.vorlap_root, env=env)
    installed_library = find_bridge_library(install_dir)
    if installed_library is None:
        raise FileNotFoundError(f"VorLap bridge was not installed into {install_dir}")
    validate_bridge_library_loadable(installed_library)
    if args.force_bridge_build and force_token:
        atomic_write_json(force_marker, {"token": force_token})


def prepare_cases(args: Any, manifest: dict[str, object]) -> None:
    for case in manifest["cases"]:
        case_dir = Path(str(case["case_dir"]))
        sim_path = Path(str(case["sim"]))
        if case_dir.exists():
            if args.force_setup:
                shutil.rmtree(case_dir)
            else:
                continue
        copy_reference_case(args.vorlap_root, case_dir, force=False)
        raw_result_filter = getattr(args, "result_filter", None)
        if raw_result_filter is not None and str(raw_result_filter).strip():
            source_filter = Path(raw_result_filter).expanduser().resolve()
            case_filter = Path(str(case["result_filter"]))
            case_filter.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_filter, case_filter)
        replace_qblade_keyword(sim_path, "MEANINF", float(case["wind_speed"]))
        replace_qblade_keyword(sim_path, "INITIAL_AZIMUTH", float(case["azimuth_deg"]))
        replace_qblade_keyword(sim_path, "PRESCRIBETYPE", 1)
        replace_qblade_keyword(sim_path, "RPMPRESCRIBED", args.rpm)
        replace_qblade_keyword(sim_path, "NUMTIMESTEPS", args.timesteps)
        replace_qblade_keyword(sim_path, "TIMESTEP", args.timestep)
        replace_qblade_keyword(sim_path, "STOREFROM", args.store_from)
        replace_qblade_keyword(sim_path, "STOREREPLAY", str(args.store_replay).lower())
        replace_qblade_keyword(sim_path, "WNDTYPE", args.wind_type)
        replace_qblade_keyword(sim_path, "HORANGLE", getattr(args, "horizontal_inflow_angle", 0.0))
        replace_qblade_keyword(sim_path, "VERTANGLE", getattr(args, "vertical_inflow_angle", 0.0))
        replace_qblade_keyword(sim_path, "SHEAREXP", getattr(args, "shear_exponent", 0.23))
        replace_qblade_keyword(sim_path, "REFHEIGHT", getattr(args, "reference_height", 18.0))
        replace_qblade_keyword(sim_path, "DENSITYAIR", getattr(args, "air_density", 1.0296))
        replace_qblade_keyword(
            sim_path,
            "VISCOSITYAIR",
            getattr(args, "air_kinematic_viscosity", 1.647e-5),
        )
        if args.wind_name:
            wind_name = args.wind_name.format(
                speed=float(case["wind_speed"]),
                label=case["label"],
                speed_label=case["speed_label"],
                azimuth=float(case["azimuth_deg"]),
                azimuth_label=case["azimuth_label"],
            )
            replace_qblade_keyword(sim_path, "WNDNAME", wind_name)
        wake_settings = manifest.get("settings", {}).get("wake_settings", {})
        if isinstance(wake_settings, Mapping) and wake_settings:
            turbine_path = turbine_definition_path_for_sim(sim_path)
            wake_patches = [
                replace_qblade_keyword_file(
                    turbine_path,
                    str(keyword),
                    value,
                    comment=_KEYWORD_COMMENTS.get(str(keyword), ""),
                )
                for keyword, value in wake_settings.items()
            ]
            wake_manifest = turbine_path.parent / "Control" / "vorlap_wake_profile.json"
            wake_manifest.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                wake_manifest,
                {
                    "profile": manifest.get("settings", {}).get("wake_profile"),
                    "scope": "generated_case_only",
                    "reason": _WAKE_PROFILE_REASONS.get(
                        str(manifest.get("settings", {}).get("wake_profile")),
                        "Apply the archived generated-case QBlade aerodynamic/wake overrides.",
                    ),
                    "patches": wake_patches,
                },
            )
        if not args.skip_stable_support_patch:
            apply_stable_support_patch(
                case_dir,
                sim_path,
                args.timestep,
                args.rampup,
                args.add_damp,
                args.str_iterations,
            )

        prep_script = args.vorlap_root / "scripts" / "prepare_qblade_external_case.py"
        airfoils = vorlap_airfoil_dir_for_sim(sim_path)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{args.vorlap_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        prep_command = [
            args.python,
            str(prep_script),
            "--sim",
            str(sim_path),
            "--airfoils",
            str(airfoils),
            "--node-source",
            "structural",
            "--n-freq-depth",
            str(args.n_freq_depth),
            "--source-bounds-policy",
            str(args.source_bounds_policy),
            "--force-scale",
            str(args.force_scale),
            "--debug",
            "--absolute-source-parameter-dir",
        ]
        frequency_cap_hz = manifest.get("settings", {}).get("source_frequency_cap_hz")
        if frequency_cap_hz is not None:
            prep_command.extend(["--source-frequency-cap-hz", f"{float(frequency_cap_hz):.17g}"])
        if args.diagnostic_force_scale:
            prep_command.append("--diagnostic-force-scale")
        run_checked(prep_command, env=env)
        if getattr(args, "full_structure_strain_output", False):
            case_filter = Path(str(case["result_filter"]))
            output_metadata_path = (
                turbine_dir_for_sim(sim_path)
                / "Control"
                / "vorlap_full_structure_strain_output.json"
            )
            output_metadata = configure_full_structure_outputs(
                sim_path,
                case_filter,
                metadata_path=output_metadata_path,
            )
            case["full_structure_strain_output"] = {
                "mode": FULL_STRUCTURE_OUTPUT_MODE,
                "metadata_path": str(output_metadata_path),
                "metadata_sha256": file_sha256(output_metadata_path)["sha256"],
                "monitor_count": output_metadata["monitor_count"],
                "result_filter_channel_count": output_metadata[
                    "result_filter_channel_count"
                ],
            }
        if raw_result_filter is not None and str(raw_result_filter).strip():
            case_filter = Path(str(case["result_filter"]))
            storage_filter = Path(str(case.get("storage_filter") or case_filter))
            if str(args.coupling) == "sil-vorlap":
                turbine_path = turbine_definition_path_for_sim(sim_path)
                parameter_value = read_qblade_keyword(turbine_path, "LIBPARAMETERFILE_1")
                if parameter_value is None or not parameter_value.strip():
                    raise ValueError(f"Could not find LIBPARAMETERFILE_1 in {turbine_path}")
                parameter_file = resolve_qblade_path(turbine_path, parameter_value)
                storage_metadata = write_sil_storage_filter(
                    result_filter=case_filter,
                    storage_filter=storage_filter,
                    parameter_file=parameter_file,
                )
            else:
                identity = file_sha256(case_filter)
                if identity is None:
                    raise FileNotFoundError(f"Could not hash QBlade result filter: {case_filter}")
                storage_metadata = {
                    "path": str(case_filter),
                    "sha256": str(identity["sha256"]),
                    "channel_count": len(qblade_filter_channels(case_filter)),
                    "added_channels": [],
                }
            case["storage_filter"] = str(storage_filter)
            case["storage_filter_sha256"] = storage_metadata["sha256"]
            case["storage_filter_channel_count"] = storage_metadata["channel_count"]
            case["storage_filter_added_channels"] = storage_metadata["added_channels"]
            replace_qblade_keyword(sim_path, "FILTERFILE", storage_filter.name)
        runtime_assets, runtime_asset_fingerprint = qblade_runtime_asset_fingerprint(sim_path)
        case["runtime_assets"] = runtime_assets
        case["runtime_asset_fingerprint"] = runtime_asset_fingerprint


def setup(args: Any) -> dict[str, object]:
    args.run_root.mkdir(parents=True, exist_ok=True)
    validate_qblade_root(args.qblade_root)
    sentinel = args.run_root / "setup_complete.json"
    manifest_path = args.run_root / "cases.json"

    with setup_lock(args.run_root):
        if getattr(args, "full_structure_strain_output", False):
            generated_filter = args.run_root / "full_structure_strain_results.filter"
            requested_filter = getattr(args, "result_filter", None)
            if requested_filter is not None and Path(requested_filter).resolve() != generated_filter.resolve():
                raise ValueError(
                    "--full-structure-strain-output generates its own model-matched result "
                    "filter and cannot be combined with a different --result-filter."
                )
            reference_sim = (
                reference_case_dir_for_root(Path(args.vorlap_root))
                / reference_sim_name_for_root(Path(args.vorlap_root))
            )
            write_full_structure_filter(
                reference_sim,
                generated_filter,
                metadata_path=args.run_root / "full_structure_strain_results.json",
            )
            args.result_filter = generated_filter
        planned_manifest = build_case_manifest(args)
        regenerate_existing_cases = False
        if sentinel.is_file() and manifest_path.is_file() and not args.force_setup:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_matches_setup_inputs(existing_manifest, planned_manifest):
                artifact_issues = setup_artifact_issues(existing_manifest)
                if not artifact_issues:
                    if not getattr(args, "skip_vorlap_install", True):
                        install_vorlap(args)
                    if not getattr(args, "skip_bridge_build", True):
                        build_bridge(args)
                    return existing_manifest
                print(
                    "Existing setup artifacts do not match the cached manifest; regenerating cases:",
                    file=sys.stderr,
                )
                for issue in artifact_issues:
                    print(f"  - {issue}", file=sys.stderr)
                regenerate_existing_cases = True
            else:
                print(
                    "Existing setup manifest does not match current setup inputs; regenerating cases.",
                    file=sys.stderr,
                )
                regenerate_existing_cases = True
        elif not args.force_setup:
            existing_case_dirs = [
                Path(str(case["case_dir"]))
                for case in planned_manifest["cases"]
                if Path(str(case["case_dir"])).exists()
            ]
            if existing_case_dirs:
                print(
                    "Found generated case directories without a complete reusable setup; regenerating cases.",
                    file=sys.stderr,
                )
                regenerate_existing_cases = True

        install_vorlap(args)
        build_bridge(args)
        setup_args = args
        if regenerate_existing_cases:
            setup_args = copy.copy(args)
            setup_args.force_setup = True
        prepare_cases(setup_args, planned_manifest)
        atomic_write_json(manifest_path, planned_manifest)
        atomic_write_json(sentinel, {"setup_complete_unix": time.time(), "cases": len(planned_manifest["cases"])})
        return planned_manifest


def completed_filtered_export_ready(
    case: Mapping[str, object],
    result_filter: Path,
    *,
    require_completed_checkpoint_export_only: bool = False,
) -> bool:
    """Return true for a complete, successful, hash-verified filtered export.

    Recovery callers can additionally require evidence that the existing export
    was itself produced from a completed checkpoint without advancing QBlade.
    A normal completed run must not suppress that one-time recovery because its
    summary does not carry the narrower recovery identity policy.
    """
    result_dir = Path(str(case["result_dir"]))
    summary_path = result_dir / f"{Path(str(case['sim'])).stem}_run_summary.json"
    result_path = result_dir / f"{case['result_name']}.txt"
    if not summary_path.is_file() or not result_path.is_file() or result_path.stat().st_size <= 0:
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    requested_steps = int(summary.get("requested_steps", 0) or 0)
    completed_steps = int(summary.get("completed_steps", 0) or 0)
    if (
        requested_steps <= 0
        or completed_steps != requested_steps
        or bool(summary.get("failed", False))
        or not bool(summary.get("exported_results", False))
    ):
        return False
    restart = summary.get("restart")
    if require_completed_checkpoint_export_only and (
        not isinstance(restart, Mapping)
        or not bool(restart.get("completed_checkpoint_export_only", False))
    ):
        return False
    filter_identity = file_sha256(result_filter)
    result_identity = file_sha256(result_path)
    recorded_filter_identity = summary.get("result_filter_identity")
    recorded_result_identity = summary.get("result_file_identity")
    if not all(
        isinstance(identity, dict)
        for identity in (
            filter_identity,
            result_identity,
            recorded_filter_identity,
            recorded_result_identity,
        )
    ):
        return False
    return (
        recorded_filter_identity.get("sha256") == filter_identity.get("sha256")
        and recorded_filter_identity.get("size_bytes") == filter_identity.get("size_bytes")
        and recorded_result_identity.get("sha256") == result_identity.get("sha256")
        and recorded_result_identity.get("size_bytes") == result_identity.get("size_bytes")
    )


def load_manifest(args: Any) -> dict[str, object]:
    manifest_path = args.run_root / "cases.json"
    if not manifest_path.is_file():
        return setup(args)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def qblade_env(args: Any) -> dict[str, str]:
    return QBladeBatchRunOptions.from_args(args).env()


__all__ = [
    "QBladeBatchCase",
    "QBladeBatchManifest",
    "QBladeBatchRunOptions",
    "QBladeBatchSettings",
    "apply_stable_support_patch",
    "azimuth_label",
    "build_bridge",
    "build_case_manifest",
    "completed_filtered_export_ready",
    "comparable_case_manifest",
    "copy_reference_case",
    "find_bridge_library",
    "find_qblade_library",
    "install_vorlap",
    "load_manifest",
    "manifest_matches_setup_inputs",
    "parse_azimuths",
    "parse_float_spec",
    "parse_wind_speeds",
    "prepend_env_path",
    "prepare_cases",
    "qblade_aerodynamic_reference_issues",
    "qblade_env",
    "qblade_library_dirs",
    "reference_case_dir_for_root",
    "reference_sim_name_for_root",
    "relative_case_path",
    "replace_qblade_keyword",
    "replace_structural_keyword",
    "run_checked",
    "setup",
    "setup_artifact_issues",
    "setup_lock",
    "speed_label",
    "structural_file_for_sim",
    "turbine_definition_path_for_sim",
    "turbine_dir_for_sim",
    "validate_force_scale",
    "validate_bridge_library_loadable",
    "validate_qblade_root",
    "vorlap_airfoil_dir_for_sim",
    "WAKE_PROFILE_CHOICES",
    "WAKE_PROFILE_PARKED_REDUCED_V1",
    "WAKE_PROFILE_PARKED_REDUCED_V2",
    "WAKE_PROFILE_REFERENCE",
    "wake_profile_settings",
]
