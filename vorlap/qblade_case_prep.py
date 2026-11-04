"""Importable service for preparing QBlade cases for VorLap coupling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import h5py
import json
import math
import numpy as np
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .qblade_conversion import convert_qblade_to_vorlap_inputs
from .qblade_parser import (
    load_qblade_simulation_definition,
    load_qblade_turbine_definition,
)
from .qblade_nodes import (
    extract_structural_node_ids,
    infer_geometry_inclusions,
    parse_qblade_node_id,
)
from .qblade_external_table import build_hvawt_response_table_spec
from .qblade_hvawt_response import load_tracked_hvawt_modal_basis
from .qblade_runtime import (
    build_external_library_table_spec,
    build_qblade_external_config,
    patch_qblade_structural_definition,
    patch_qblade_turbine_definition,
    write_qblade_external_config,
    write_qblade_geometry_audit,
)

QBladeNodeSource = Literal["structural", "converted"]


@dataclass(frozen=True)
class QBladeCasePrepRequest:
    """Inputs needed to prepare one QBlade case for VorLap external-library coupling."""

    sim_path: str | Path
    airfoil_dir: str | Path
    library_stem: str = "libvorlap_qblade_bridge"
    function_name: str = "update"
    parameter_file: str | Path | None = None
    sample_step: float | None = None
    default_airfoil_id: str = "default"
    include_struts: bool = True
    include_tower: bool = False
    node_source: QBladeNodeSource = "structural"
    include_tower_nodes: bool = False
    tower_airfoil_id: str = "cylinder"
    n_freq_depth: int | None = None
    force_scale: float = 1.0
    diagnostic_force_scale: bool = False
    include_mean_load: bool = False
    source_bounds_policy: str = "raise"
    source_frequency_cap_hz: float | None = None
    response_model: Mapping[str, Any] | None = None
    hvawt_response_model: Mapping[str, Any] | None = None
    hvawt_response_artifact_dir: str | Path | None = None
    use_qblade_local_inflow: bool = False
    fluid_density: float | None = None
    fluid_dynamicviscosity: float | None = None
    rotation_axis: Sequence[float] | None = None
    rotation_axis_offset: Sequence[float] | None = None
    inflow_vec: Sequence[float] | None = None
    debug: bool = False
    log_file: str | Path | None = None
    python_path_entries: Sequence[str | Path] = ()
    include_default_python_path: bool = True
    default_python_path: str | Path | None = None
    absolute_source_parameter_dir: bool = False
    geometry_audit_file: str | Path | None = None
    write_geometry_audit: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class QBladeCasePrepResult:
    """Resolved outputs and side effects from QBlade case preparation."""

    sim_path: Path
    turbine_path: Path
    structural_path: Path
    parameter_file_path: Path
    geometry_audit_path: Path | None
    sim_path_for_config: str
    airfoil_dir_for_config: str
    node_source: QBladeNodeSource
    node_ids: tuple[str, ...]
    structural_node_ids: tuple[str, ...]
    converted_node_ids: tuple[str, ...]
    convert_include_struts: bool
    convert_include_tower: bool
    swap_size: int
    component_count: int
    runtime_config: dict[str, Any]
    planned_write_paths: tuple[Path, ...]
    written_files: tuple[Path, ...]
    dry_run: bool


@dataclass(frozen=True)
class QBladeModelReferenceValidation:
    """Resolved QBlade model references and any validation errors."""

    sim_path: Path
    turbine_path: Path | None
    blade_path: Path | None
    structural_path: Path | None
    structural_node_ids: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


@dataclass(frozen=True)
class QBladeAirfoilAvailability:
    """Airfoil FFT ids available to a prepared QBlade/VorLap case."""

    airfoil_dir: Path
    available_ids: tuple[str, ...]
    default_source_id: str | None
    fallback_rows: tuple[dict[str, object], ...]


def resolve_parameter_file_path(
    trb_path: Path, parameter_file: str | Path | None
) -> Path:
    """Resolve the generated VorLap runtime-config path for a QBlade turbine file."""
    if parameter_file:
        param_path = Path(parameter_file)
        if not param_path.is_absolute():
            param_path = (trb_path.parent / param_path).resolve()
        return param_path
    return (trb_path.parent / "Control" / "vorlap_qblade_external.json").resolve()


def path_for_config(path: Path, base_dir: Path) -> str:
    """Store paths relative to the config file when they live under that tree."""
    resolved = path.resolve()
    base = base_dir.resolve()
    if resolved == base or base in resolved.parents:
        return Path(os.path.relpath(str(resolved), start=str(base))).as_posix()
    return str(resolved)


def validate_force_scale(
    force_scale: float, diagnostic_force_scale: bool = False
) -> None:
    """Require explicit labeling for exaggerated force-scale diagnostics."""
    if not math.isfinite(force_scale) or force_scale < 0.0:
        raise ValueError("--force-scale must be finite and non-negative.")
    if force_scale > 1.0 and not diagnostic_force_scale:
        raise ValueError(
            "--force-scale values greater than 1.0 require --diagnostic-force-scale "
            "so exaggerated forcing cannot be mistaken for a nominal physical run."
        )


def python_path_entries_for_config(
    raw_entries: Sequence[str | Path],
    *,
    include_default_path: bool,
    config_dir: Path,
    default_python_path: str | Path | None = None,
) -> list[str]:
    """Resolve embedded-Python import paths for the generated runtime config."""
    resolved: list[Path] = []
    if include_default_path:
        default_path = (
            Path(default_python_path)
            if default_python_path is not None
            else Path(__file__).resolve().parents[1]
        )
        resolved.append(default_path.resolve())
    for raw in raw_entries:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        resolved.append(candidate)

    entries: list[str] = []
    seen: set[str] = set()
    for path in resolved:
        text = path_for_config(path, config_dir)
        if text not in seen:
            seen.add(text)
            entries.append(text)
    return entries


def _resolve_output_path(base_dir: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _relative_or_absolute(path: Path, base_dir: Path) -> str:
    try:
        return Path(os.path.relpath(str(path), start=str(base_dir))).as_posix()
    except ValueError:
        return str(path)


def _child_or_absolute(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path)


def _airfoil_name_from_h5(path: Path) -> str:
    try:
        with h5py.File(path, "r") as h5:
            raw_name = h5["Airfoilname"][()] if "Airfoilname" in h5 else path.name
    except Exception as exc:
        raise ValueError(f"Could not read airfoil FFT name from {path}: {exc}") from exc
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode("utf-8")
    name = str(raw_name).strip()
    return name or path.name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_numeric_dataset(h5: h5py.File, name: str) -> list[float]:
    if name not in h5:
        return []
    values = h5[name][()]
    if values is None:
        return []
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def build_airfoil_source_manifest(airfoil_dir: str | Path) -> list[dict[str, object]]:
    """Return portable hashes and physical/source bounds for every FFT file."""
    resolved_dir = Path(airfoil_dir).expanduser().resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(resolved_dir.glob("*.h5")):
        try:
            with h5py.File(path, "r") as h5:
                airfoil_id = _airfoil_name_from_h5(path)
                reynolds = _optional_numeric_dataset(h5, "Re")
                aoa = _optional_numeric_dataset(h5, "AOA")
                thickness = _optional_numeric_dataset(h5, "Thickness")
                processing_raw = h5.attrs.get("processing_config_json")
                processing = None
                if processing_raw not in (None, ""):
                    if isinstance(processing_raw, bytes):
                        processing_raw = processing_raw.decode("utf-8")
                    processing = json.loads(str(processing_raw))
                rows.append(
                    {
                        "airfoil_id": airfoil_id,
                        "file": path.name,
                        "sha256": _sha256_file(path),
                        "bytes": path.stat().st_size,
                        "reynolds": reynolds,
                        "reynolds_min": min(reynolds) if reynolds else None,
                        "reynolds_max": max(reynolds) if reynolds else None,
                        "aoa_min_deg": min(aoa) if aoa else None,
                        "aoa_max_deg": max(aoa) if aoa else None,
                        "thickness_ratio": (
                            thickness[0] if len(thickness) == 1 else thickness
                        ),
                        "generator": str(h5.attrs.get("generator", "")),
                        "source_data_sha256": str(
                            h5.attrs.get("source_data_sha256", "")
                        ),
                        "reynolds_model": str(h5.attrs.get("reynolds_model", "")),
                        "reynolds_replication_justification": str(
                            h5.attrs.get("reynolds_replication_justification", "")
                        ),
                        "processing_config": processing,
                    }
                )
        except Exception as exc:
            raise ValueError(
                f"Could not build airfoil source manifest for {path}: {exc}"
            ) from exc
    return rows


def _available_airfoil_ids(
    airfoil_dir: str | Path,
) -> tuple[Path, tuple[str, ...], str]:
    resolved_dir = Path(airfoil_dir).expanduser().resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Airfoil directory does not exist: {resolved_dir}")
    h5_files = sorted(resolved_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(
            f"No airfoil FFT .h5 files were found in: {resolved_dir}"
        )
    ids: list[str] = []
    seen: set[str] = set()
    for path in h5_files:
        airfoil_id = _airfoil_name_from_h5(path)
        if airfoil_id not in seen:
            seen.add(airfoil_id)
            ids.append(airfoil_id)
    if "default" in seen:
        default_source_id = "default"
    else:
        default_source_id = ids[0]
        ids.append("default")
    return resolved_dir, tuple(ids), default_source_id


def validate_component_airfoils(
    components: Sequence[Any], airfoil_dir: str | Path
) -> QBladeAirfoilAvailability:
    """Validate component airfoil ids against available FFT datasets and default fallback."""
    resolved_dir, available_ids, default_source_id = _available_airfoil_ids(airfoil_dir)
    available = set(available_ids)
    fallback_rows: list[dict[str, object]] = []

    for component in components:
        component_id = str(getattr(component, "id", "component"))
        shape = getattr(component, "shape_xyz", [])
        try:
            node_count = len(shape)
        except TypeError:
            node_count = 0
        airfoil_ids = list(getattr(component, "airfoil_ids", []) or [])
        if len(airfoil_ids) < node_count:
            fill_id = airfoil_ids[-1] if airfoil_ids else "default"
            airfoil_ids = [*airfoil_ids, *([fill_id] * (node_count - len(airfoil_ids)))]
        else:
            airfoil_ids = airfoil_ids[:node_count]

        for node_index, raw_airfoil_id in enumerate(airfoil_ids):
            airfoil_id = str(raw_airfoil_id)
            if airfoil_id in available:
                if airfoil_id == "default" and default_source_id != "default":
                    fallback_rows.append(
                        {
                            "component": component_id,
                            "node_index": node_index,
                            "requested_airfoil_id": airfoil_id,
                            "fallback_airfoil_id": "default",
                            "fallback_source_airfoil_id": default_source_id,
                        }
                    )
                continue
            if "default" not in available:
                raise KeyError(
                    f"Airfoil '{airfoil_id}' requested by component '{component_id}' node {node_index} "
                    f"was not found in {resolved_dir}, and no default fallback is available."
                )
            fallback_rows.append(
                {
                    "component": component_id,
                    "node_index": node_index,
                    "requested_airfoil_id": airfoil_id,
                    "fallback_airfoil_id": "default",
                    "fallback_source_airfoil_id": default_source_id,
                }
            )

    return QBladeAirfoilAvailability(
        airfoil_dir=resolved_dir,
        available_ids=available_ids,
        default_source_id=default_source_id,
        fallback_rows=tuple(fallback_rows),
    )


def validate_generated_external_table(node_ids: Sequence[str], table_spec: Any) -> None:
    """Validate generated QBlade swap-table dimensions before patching case files."""
    node_count = len(tuple(node_ids))
    if node_count < 1:
        raise ValueError(
            "Generated QBlade external table has no nodes; cannot patch external-library tables."
        )
    swap_layout = getattr(table_spec, "swap_layout", {})
    has_rpm = "lss_rpm" in swap_layout
    has_positions = "position_global_m" in swap_layout
    if has_rpm != has_positions:
        raise ValueError(
            "Generated H-VAWT table must contain both RPM and position blocks."
        )
    hvawt_response_table = has_rpm and has_positions
    expected_swap_size = (
        (4 + 9 * node_count) if hvawt_response_table else (3 + 6 * node_count)
    )
    expected_input_rows = (
        (4 + 6 * node_count) if hvawt_response_table else (3 + 3 * node_count)
    )
    expected_output_rows = 3 * node_count
    if tuple(str(node_id) for node_id in node_ids) != tuple(
        str(node_id) for node_id in table_spec.node_ids
    ):
        raise ValueError(
            "Generated QBlade external table node order does not match the selected node list."
        )
    if int(table_spec.swap_size) != expected_swap_size:
        raise ValueError(
            f"Generated QBlade external table swap size {table_spec.swap_size} does not match "
            f"expected size {expected_swap_size} for {node_count} node(s)."
        )
    if len(table_spec.input_rows) != expected_input_rows:
        raise ValueError(
            f"Generated QBlade external input row count {len(table_spec.input_rows)} does not match "
            f"expected count {expected_input_rows} for {node_count} node(s)."
        )
    if len(table_spec.output_rows) != expected_output_rows:
        raise ValueError(
            f"Generated QBlade external output row count {len(table_spec.output_rows)} does not match "
            f"expected count {expected_output_rows} for {node_count} node(s)."
        )


def validate_qblade_model_references(
    sim_path: str | Path,
    *,
    include_tower_nodes: bool = False,
) -> QBladeModelReferenceValidation:
    """Validate the QBlade files referenced by one simulation definition."""
    resolved_sim = Path(sim_path).expanduser().resolve()
    errors: list[str] = []
    turbine_path: Path | None = None
    blade_path: Path | None = None
    structural_path: Path | None = None
    structural_node_ids: tuple[str, ...] = ()

    if not resolved_sim.is_file():
        errors.append(f"QBlade simulation file does not exist: {resolved_sim}")
        return QBladeModelReferenceValidation(
            sim_path=resolved_sim,
            turbine_path=None,
            blade_path=None,
            structural_path=None,
            structural_node_ids=(),
            errors=tuple(errors),
        )

    try:
        sim_info = load_qblade_simulation_definition(str(resolved_sim))
    except Exception as exc:
        errors.append(f"Could not read QBlade simulation file {resolved_sim}: {exc}")
        return QBladeModelReferenceValidation(
            sim_path=resolved_sim,
            turbine_path=None,
            blade_path=None,
            structural_path=None,
            structural_node_ids=(),
            errors=tuple(errors),
        )

    turbfile = str(sim_info.get("turbfile") or "").strip()
    if not turbfile:
        errors.append(f"QBlade .sim file does not specify TURBFILE: {resolved_sim}")
    else:
        turbine_path = Path(str(sim_info.get("turbfile_path"))).expanduser().resolve()
        if not turbine_path.is_file():
            errors.append(f"QBlade .sim TURBFILE does not exist: {turbine_path}")

    if turbine_path is not None and turbine_path.is_file():
        try:
            trb_info = load_qblade_turbine_definition(str(turbine_path))
        except Exception as exc:
            errors.append(f"Could not read QBlade turbine file {turbine_path}: {exc}")
        else:
            bladefile = str(trb_info.get("bladefile") or "").strip()
            if not bladefile:
                errors.append(
                    f"QBlade .trb file does not specify BLADEFILE: {turbine_path}"
                )
            else:
                blade_path = (
                    Path(str(trb_info.get("bladefile_path"))).expanduser().resolve()
                )
                if not blade_path.is_file():
                    errors.append(f"QBlade .trb BLADEFILE does not exist: {blade_path}")

            structuralfile = str(trb_info.get("structuralfile") or "").strip()
            if not structuralfile:
                errors.append(
                    f"QBlade .trb file does not specify STRUCTURALFILE: {turbine_path}"
                )
            else:
                structural_path = (
                    Path(str(trb_info.get("structuralfile_path")))
                    .expanduser()
                    .resolve()
                )
                if not structural_path.is_file():
                    errors.append(
                        f"QBlade .trb STRUCTURALFILE does not exist: {structural_path}"
                    )

    if structural_path is not None and structural_path.is_file():
        structural_text = structural_path.read_text(encoding="utf-8", errors="ignore")
        raw_structural_node_ids = tuple(
            extract_structural_node_ids(
                structural_text,
                include_tower_nodes=include_tower_nodes,
                unique=False,
            )
        )
        structural_node_ids = tuple(
            extract_structural_node_ids(
                structural_text,
                include_tower_nodes=include_tower_nodes,
            )
        )
        seen_nodes: set[str] = set()
        duplicate_nodes: list[str] = []
        for node_id in raw_structural_node_ids:
            if node_id in seen_nodes and node_id not in duplicate_nodes:
                duplicate_nodes.append(node_id)
            seen_nodes.add(node_id)
        if duplicate_nodes:
            errors.append(
                f"Duplicate structural node id(s) in {structural_path}: {', '.join(duplicate_nodes)}"
            )
        for node_id in structural_node_ids:
            try:
                parse_qblade_node_id(node_id, validate_position=True)
            except ValueError as exc:
                errors.append(f"Invalid structural node id in {structural_path}: {exc}")

    return QBladeModelReferenceValidation(
        sim_path=resolved_sim,
        turbine_path=turbine_path,
        blade_path=blade_path,
        structural_path=structural_path,
        structural_node_ids=structural_node_ids,
        errors=tuple(errors),
    )


def prepare_qblade_external_case(
    request: QBladeCasePrepRequest,
) -> QBladeCasePrepResult:
    """Prepare one QBlade case for VorLap external-library coupling."""
    if not isinstance(request.use_qblade_local_inflow, bool):
        raise ValueError("use_qblade_local_inflow must be a JSON boolean.")
    validate_force_scale(request.force_scale, request.diagnostic_force_scale)

    sim_path = Path(request.sim_path).resolve()
    airfoil_arg = Path(request.airfoil_dir)
    airfoil_dir = (
        airfoil_arg.resolve()
        if airfoil_arg.is_absolute()
        else (Path.cwd() / airfoil_arg).resolve()
    )
    if not airfoil_dir.is_dir():
        raise FileNotFoundError(f"Airfoil directory does not exist: {airfoil_dir}")

    # Validate the QBlade file chain before conversion so patching never starts
    # from a broken .sim/.trb/.bld/.str reference set.
    model_validation = validate_qblade_model_references(
        sim_path,
        include_tower_nodes=request.include_tower_nodes,
    )
    model_validation.raise_for_errors()

    sim_info = load_qblade_simulation_definition(str(sim_path))
    turbine_path = Path(str(sim_info["turbfile_path"])).resolve()
    trb_info = load_qblade_turbine_definition(str(turbine_path))
    structural_path = Path(str(trb_info["structuralfile_path"])).resolve()
    structural_text = structural_path.read_text(encoding="utf-8", errors="ignore")

    structural_node_ids = tuple(
        extract_structural_node_ids(
            structural_text,
            include_tower_nodes=request.include_tower_nodes,
        )
    )
    preliminary_node_ids = (
        structural_node_ids if request.node_source == "structural" else ()
    )
    convert_include_struts, convert_include_tower = infer_geometry_inclusions(
        preliminary_node_ids,
        request.include_struts,
        request.include_tower,
    )

    components, _viv_params, converted_node_ids_list = convert_qblade_to_vorlap_inputs(
        str(sim_path),
        default_airfoil_id=request.default_airfoil_id,
        include_struts=convert_include_struts,
        include_tower=convert_include_tower,
        tower_airfoil_id=request.tower_airfoil_id,
    )
    airfoil_validation = validate_component_airfoils(components, airfoil_dir)
    converted_node_ids = tuple(str(node_id) for node_id in converted_node_ids_list)
    if request.node_source == "structural":
        node_ids = structural_node_ids if structural_node_ids else converted_node_ids
    else:
        node_ids = converted_node_ids

    if not node_ids:
        raise ValueError(
            "Generated QBlade external table has no nodes; cannot patch external-library tables."
        )
    try:
        table_spec = (
            build_hvawt_response_table_spec(node_ids)
            if request.hvawt_response_model is not None or request.use_qblade_local_inflow
            else build_external_library_table_spec(node_ids)
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid QBlade external table node list before patching: {exc}"
        ) from exc
    # The node list defines both QBlade table rows and VorLap swap shape; check
    # the generated table before any case files are written.
    validate_generated_external_table(node_ids, table_spec)
    parameter_file_path = resolve_parameter_file_path(
        turbine_path, request.parameter_file
    )
    log_file_path = _resolve_output_path(parameter_file_path.parent, request.log_file)
    if log_file_path is None and request.debug:
        log_file_path = (
            parameter_file_path.parent / "vorlap_qblade_debug.log"
        ).resolve()

    if request.geometry_audit_file:
        geometry_audit_path = _resolve_output_path(
            parameter_file_path.parent, request.geometry_audit_file
        )
    elif request.write_geometry_audit:
        geometry_audit_path = (
            parameter_file_path.parent / "vorlap_qblade_geometry_audit.csv"
        ).resolve()
    else:
        geometry_audit_path = None

    hvawt_artifact_copies: list[tuple[Path, Path]] = []
    if request.hvawt_response_model is not None:
        if request.hvawt_response_artifact_dir is None:
            raise ValueError(
                "H-VAWT response case preparation requires an explicit modal artifact directory."
            )
        artifacts = request.hvawt_response_model.get("modal_artifacts")
        expected_artifact_roles = {
            "identification",
            "shapes",
            "duals",
            "review_marker",
        }
        if (
            not isinstance(artifacts, Mapping)
            or set(artifacts) != expected_artifact_roles
        ):
            raise ValueError(
                "hvawt_response_model.modal_artifacts must define identification, shapes, "
                "duals, and review_marker."
            )
        artifact_dir = Path(request.hvawt_response_artifact_dir).expanduser().resolve()
        resolved_sources: dict[str, Path] = {}
        for role in sorted(expected_artifact_roles):
            raw_name = artifacts[role]
            if (
                not isinstance(raw_name, str)
                or not raw_name
                or Path(raw_name).name != raw_name
            ):
                raise ValueError(
                    f"H-VAWT modal artifact {role!r} must be one local filename."
                )
            source = (artifact_dir / raw_name).resolve()
            if source.parent != artifact_dir or not source.is_file():
                raise FileNotFoundError(
                    f"Missing H-VAWT modal artifact {role!r}: {source}"
                )
            destination = (parameter_file_path.parent / raw_name).resolve()
            resolved_sources[role] = source
            hvawt_artifact_copies.append((source, destination))
        calibration_blocks = [
            (name, request.hvawt_response_model.get(name))
            for name in ("engineering_lockin", "griffin_lockin")
            if request.hvawt_response_model.get(name) is not None
        ]
        if len(calibration_blocks) > 1:
            raise ValueError(
                "H-VAWT response model cannot stage multiple lock-in calibrations."
            )
        if calibration_blocks:
            calibration_role, calibration_block = calibration_blocks[0]
            if not isinstance(calibration_block, Mapping):
                raise ValueError(
                    f"hvawt_response_model.{calibration_role} must be a mapping."
                )
            calibration_name = calibration_block.get("calibration_path")
            expected_calibration_sha256 = calibration_block.get("calibration_sha256")
            if (
                not isinstance(calibration_name, str)
                or not calibration_name
                or Path(calibration_name).name != calibration_name
            ):
                raise ValueError(
                    "H-VAWT lock-in calibration_path must be one local filename."
                )
            calibration_source = (artifact_dir / calibration_name).resolve()
            if (
                calibration_source.parent != artifact_dir
                or not calibration_source.is_file()
            ):
                raise FileNotFoundError(
                    f"Missing H-VAWT lock-in calibration: {calibration_source}"
                )
            if (
                _sha256_file(calibration_source)
                != str(expected_calibration_sha256).lower()
            ):
                raise ValueError(
                    "H-VAWT lock-in calibration SHA-256 differs from the model config."
                )
            hvawt_artifact_copies.append(
                (
                    calibration_source,
                    (parameter_file_path.parent / calibration_name).resolve(),
                )
            )
            unified_name = calibration_block.get("unified_gain_model_path")
            if unified_name is not None:
                expected_unified_sha256 = calibration_block.get(
                    "unified_gain_model_sha256"
                )
                if (
                    not isinstance(unified_name, str)
                    or not unified_name
                    or Path(unified_name).name != unified_name
                ):
                    raise ValueError(
                        "H-VAWT unified_gain_model_path must be one local filename."
                    )
                unified_source = (artifact_dir / unified_name).resolve()
                if unified_source.parent != artifact_dir or not unified_source.is_file():
                    raise FileNotFoundError(
                        f"Missing H-VAWT unified gain model: {unified_source}"
                    )
                if _sha256_file(unified_source) != str(expected_unified_sha256).lower():
                    raise ValueError(
                        "H-VAWT unified gain model SHA-256 differs from the model config."
                    )
                hvawt_artifact_copies.append(
                    (
                        unified_source,
                        (parameter_file_path.parent / unified_name).resolve(),
                    )
                )
        load_tracked_hvawt_modal_basis(
            identification_path=resolved_sources["identification"],
            shapes_path=resolved_sources["shapes"],
            duals_path=resolved_sources["duals"],
            review_marker_path=resolved_sources["review_marker"],
        )

    python_path_entries = python_path_entries_for_config(
        request.python_path_entries,
        include_default_path=request.include_default_python_path,
        config_dir=parameter_file_path.parent,
        default_python_path=request.default_python_path,
    )
    sim_path_for_config = _relative_or_absolute(sim_path, parameter_file_path.parent)
    airfoil_dir_for_config = _relative_or_absolute(
        airfoil_dir, parameter_file_path.parent
    )
    parameter_file_for_turbine = _child_or_absolute(
        parameter_file_path, turbine_path.parent
    )
    source_parameter_dir_for_config = (
        str(parameter_file_path.parent.resolve())
        if request.absolute_source_parameter_dir
        else "."
    )

    config = build_qblade_external_config(
        sim_path=sim_path_for_config,
        airfoil_dir=airfoil_dir_for_config,
        node_ids=node_ids,
        library_stem=request.library_stem,
        function_name=request.function_name,
        parameter_file=parameter_file_for_turbine,
        sample_step=request.sample_step,
        default_airfoil_id=request.default_airfoil_id,
        include_struts=request.include_struts,
        include_tower=request.include_tower,
        tower_airfoil_id=request.tower_airfoil_id,
        n_freq_depth=request.n_freq_depth,
        force_scale=request.force_scale,
        include_mean_load=request.include_mean_load,
        source_bounds_policy=request.source_bounds_policy,
        source_frequency_cap_hz=request.source_frequency_cap_hz,
        response_model=request.response_model,
        hvawt_response_model=request.hvawt_response_model,
        use_qblade_local_inflow=request.use_qblade_local_inflow,
        fluid_density=request.fluid_density,
        fluid_dynamicviscosity=request.fluid_dynamicviscosity,
        rotation_axis=request.rotation_axis,
        rotation_axis_offset=request.rotation_axis_offset,
        inflow_vec=request.inflow_vec,
        source_parameter_dir=source_parameter_dir_for_config,
        python_path_entries=python_path_entries,
        debug=request.debug,
        log_file=(
            None
            if log_file_path is None
            else path_for_config(log_file_path, parameter_file_path.parent)
        ),
    )
    config["available_airfoil_ids"] = list(airfoil_validation.available_ids)
    config["default_airfoil_source_id"] = airfoil_validation.default_source_id
    config["airfoil_fallbacks"] = [
        dict(row) for row in airfoil_validation.fallback_rows
    ]
    config["airfoil_source_manifest"] = build_airfoil_source_manifest(airfoil_dir)
    if geometry_audit_path is not None:
        config["geometry_audit_file"] = path_for_config(
            geometry_audit_path, parameter_file_path.parent
        )

    turbine_text = turbine_path.read_text(encoding="utf-8", errors="ignore")
    patched_turbine_text = patch_qblade_turbine_definition(
        turbine_text,
        library_stem=request.library_stem,
        function_name=request.function_name,
        swap_size=table_spec.swap_size,
        parameter_file=parameter_file_for_turbine,
    )
    patched_structural_text = patch_qblade_structural_definition(
        structural_text,
        table_spec=table_spec,
    )

    planned_write_paths = [parameter_file_path, turbine_path, structural_path]
    if geometry_audit_path is not None:
        planned_write_paths.append(geometry_audit_path)
    planned_write_paths.extend(
        destination for _source, destination in hvawt_artifact_copies
    )

    written_files: list[Path] = []
    if not request.dry_run:
        write_qblade_external_config(str(parameter_file_path), config)
        written_files.append(parameter_file_path)
        if geometry_audit_path is not None:
            write_qblade_geometry_audit(str(geometry_audit_path), components, node_ids)
            written_files.append(geometry_audit_path)
        for source, destination in hvawt_artifact_copies:
            if source != destination:
                destination.write_bytes(source.read_bytes())
                written_files.append(destination)
        turbine_path.write_text(patched_turbine_text, encoding="utf-8")
        written_files.append(turbine_path)
        structural_path.write_text(patched_structural_text, encoding="utf-8")
        written_files.append(structural_path)

    return QBladeCasePrepResult(
        sim_path=sim_path,
        turbine_path=turbine_path,
        structural_path=structural_path,
        parameter_file_path=parameter_file_path,
        geometry_audit_path=geometry_audit_path,
        sim_path_for_config=sim_path_for_config,
        airfoil_dir_for_config=airfoil_dir_for_config,
        node_source=request.node_source,
        node_ids=tuple(node_ids),
        structural_node_ids=structural_node_ids,
        converted_node_ids=converted_node_ids,
        convert_include_struts=convert_include_struts,
        convert_include_tower=convert_include_tower,
        swap_size=table_spec.swap_size,
        component_count=len(components),
        runtime_config=config,
        planned_write_paths=tuple(planned_write_paths),
        written_files=tuple(written_files),
        dry_run=request.dry_run,
    )


__all__ = [
    "QBladeAirfoilAvailability",
    "QBladeCasePrepRequest",
    "QBladeCasePrepResult",
    "QBladeModelReferenceValidation",
    "QBladeNodeSource",
    "path_for_config",
    "prepare_qblade_external_case",
    "python_path_entries_for_config",
    "resolve_parameter_file_path",
    "validate_component_airfoils",
    "validate_qblade_model_references",
    "validate_generated_external_table",
    "validate_force_scale",
]
