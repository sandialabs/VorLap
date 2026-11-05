"""Prepare a QBlade case for live VorLap external-library coupling."""

import argparse
from dataclasses import replace
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vorlap
from vorlap.qblade_case_prep import (
    QBladeCasePrepRequest,
    path_for_config as _case_path_for_config,
    prepare_qblade_external_case,
    python_path_entries_for_config as _case_python_path_entries_for_config,
    resolve_parameter_file_path as _case_resolve_parameter_file_path,
    validate_force_scale as _case_validate_force_scale,
)
from vorlap.qblade_nodes import (
    extract_structural_node_ids as _extract_structural_node_ids,
    infer_geometry_inclusions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="Path to the QBlade .sim file.")
    parser.add_argument(
        "--airfoils",
        default=str(Path(vorlap.repo_dir) / "data" / "airfoils"),
        help="Directory containing VorLap airfoil FFT .h5 files.",
    )
    parser.add_argument(
        "--library-stem",
        default="libvorlap_qblade_bridge",
        help="Shared-library stem used in the QBlade .trb file.",
    )
    parser.add_argument(
        "--function-name",
        default="update",
        help="Exported function name that QBlade should call.",
    )
    parser.add_argument(
        "--parameter-file",
        default=None,
        help=(
            "Relative or absolute path for the generated JSON runtime config. "
            "Defaults to Control/vorlap_qblade_external.json next to the turbine file."
        ),
    )
    parser.add_argument(
        "--sample-step",
        type=float,
        default=None,
        help="Optional QBlade SAMPLESTEP value to embed in the config.",
    )
    parser.add_argument(
        "--default-airfoil-id",
        default="default",
        help="Fallback airfoil id used when QBlade airfoil names are not present in the VorLap database.",
    )
    parser.add_argument(
        "--include-struts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include struts when converting the QBlade case.",
    )
    parser.add_argument(
        "--include-tower",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the tower in the converted VorLap geometry.",
    )
    parser.add_argument(
        "--node-source",
        choices=("structural", "converted"),
        default="structural",
        help="Choose node ids from the structural file output list or from converted VorLap geometry.",
    )
    parser.add_argument(
        "--include-tower-nodes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --node-source=structural, include TWR_* nodes in EXTERNAL_1 tables.",
    )
    parser.add_argument(
        "--tower-airfoil-id",
        default="cylinder",
        help="Airfoil id used for tower nodes.",
    )
    parser.add_argument(
        "--n-freq-depth",
        type=int,
        default=None,
        help="Override the frequency depth used by the VorLap controller.",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to VorLap external forces before returning to QBlade.",
    )
    parser.add_argument(
        "--diagnostic-force-scale",
        action="store_true",
        help=(
            "Allow force scales greater than 1.0 for explicit diagnostic or "
            "amplitude-sensitivity runs. Physical nominal runs should use 1.0; "
            "baseline identity runs should use 0.0."
        ),
    )
    parser.add_argument(
        "--include-mean-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include VorLap DC/mean-load coefficients in the external force callback.",
    )
    parser.add_argument(
        "--use-qblade-local-inflow",
        action="store_true",
        help=(
            "Require explicitly supplied nodal XYZ wind vectors in the Python SIL loop. "
            "Adds global position inputs to the external table; automatic sampling is not configured."
        ),
    )
    parser.add_argument(
        "--source-bounds-policy",
        choices=("warn", "raise", "clip"),
        default="raise",
        help="How runtime interpolation handles Re/AOA queries outside the airfoil-source grid.",
    )
    parser.add_argument(
        "--source-frequency-cap-hz",
        type=float,
        default=None,
        help=(
            "Optional fixed upper frequency [Hz] for VorLap oscillatory CL/CD tones. "
            "Tones above the cap are discarded, not clamped."
        ),
    )
    parser.add_argument(
        "--hvawt-response-config",
        type=Path,
        default=None,
        help=(
            "JSON file containing the audited two-mode H-VAWT response-model block. "
            "This selects the 355-channel RPM/position SIL table and requires --force-scale 0."
        ),
    )
    parser.add_argument(
        "--fluid-density",
        type=float,
        default=None,
        help="Optional VorLap fluid-density override in kg/m^3. Defaults to the QBlade-converted value.",
    )
    parser.add_argument(
        "--fluid-dynamicviscosity",
        type=float,
        default=None,
        help="Optional VorLap dynamic-viscosity override in Pa s. Defaults to the QBlade-converted value.",
    )
    parser.add_argument(
        "--rotation-axis",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional VorLap rotation-axis override.",
    )
    parser.add_argument(
        "--rotation-axis-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional VorLap rotation-axis offset override in meters.",
    )
    parser.add_argument(
        "--inflow-vec",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional VorLap inflow-direction vector override.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose runtime diagnostic messages for QBlade update_message().",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Optional log file path for runtime diagnostics. "
            "Defaults to vorlap_qblade_debug.log next to the generated JSON when --debug is enabled."
        ),
    )
    parser.add_argument(
        "--python-path-entry",
        action="append",
        default=[],
        help=(
            "Additional path prepended to embedded Python sys.path before importing VorLap. "
            "May be repeated. Relative paths are resolved from the current working directory."
        ),
    )
    parser.add_argument(
        "--no-default-python-path",
        action="store_true",
        help=(
            "Do not store this checkout root in python_path_entries. Use this only when VorLap "
            "is installed in the Python environment embedded by QBlade."
        ),
    )
    parser.add_argument(
        "--absolute-source-parameter-dir",
        action="store_true",
        help=(
            "Store the generated Control directory as an absolute source_parameter_dir fallback. "
            "Use this for generated run artifacts that QBlade may copy into TEMP before calling update_init()."
        ),
    )
    parser.add_argument(
        "--geometry-audit-file",
        default=None,
        help=(
            "Optional path for the generated per-node geometry/swap audit CSV. "
            "Defaults to vorlap_qblade_geometry_audit.csv next to the generated JSON."
        ),
    )
    parser.add_argument(
        "--skip-geometry-audit",
        action="store_true",
        help="Do not write the pre-run geometry/swap audit CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate the case preparation steps without writing files.",
    )
    parser.add_argument(
        "--backup-originals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before writing, save .vorlap.bak backups for existing files that will be patched "
            "or overwritten. Enabled by default."
        ),
    )
    return parser.parse_args()


def _resolve_parameter_file_path(trb_path: Path, parameter_file: Optional[str]) -> Path:
    return _case_resolve_parameter_file_path(trb_path, parameter_file)


def _path_for_config(path: Path, base_dir: Path) -> str:
    """Store paths relative to the config file when they live under that tree."""
    return _case_path_for_config(path, base_dir)


def _validate_force_scale(force_scale: float, diagnostic_force_scale: bool = False) -> None:
    """Require explicit labeling for exaggerated force-scale diagnostics."""
    _case_validate_force_scale(force_scale, diagnostic_force_scale)


def _python_path_entries_for_config(
    raw_entries: List[str],
    *,
    include_repo_root: bool,
    config_dir: Path,
) -> List[str]:
    """Resolve embedded-Python import paths for the generated runtime config."""
    return _case_python_path_entries_for_config(
        raw_entries,
        include_default_path=include_repo_root,
        config_dir=config_dir,
        default_python_path=REPO_ROOT.resolve(),
    )


def extract_structural_node_ids(structural_text: str, include_tower_nodes: bool = False) -> List[str]:
    """Extract aerodynamic node ids from a QBlade structural file."""
    return _extract_structural_node_ids(
        structural_text,
        include_tower_nodes=include_tower_nodes,
    )


def _geometry_flags_for_requested_nodes(
    node_ids: List[str],
    include_struts: bool,
    include_tower: bool,
) -> tuple[bool, bool]:
    """Include geometry classes required by the node ids selected for export."""
    return infer_geometry_inclusions(node_ids, include_struts, include_tower)


def _request_from_args(args: argparse.Namespace) -> QBladeCasePrepRequest:
    hvawt_response_model = None
    hvawt_response_artifact_dir = None
    if args.hvawt_response_config is not None:
        config_path = args.hvawt_response_config.expanduser().resolve()
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--hvawt-response-config must contain one JSON object.")
        hvawt_response_model = loaded
        hvawt_response_artifact_dir = config_path.parent
    return QBladeCasePrepRequest(
        sim_path=args.sim,
        airfoil_dir=args.airfoils,
        library_stem=args.library_stem,
        function_name=args.function_name,
        parameter_file=args.parameter_file,
        sample_step=args.sample_step,
        default_airfoil_id=args.default_airfoil_id,
        include_struts=args.include_struts,
        include_tower=args.include_tower,
        node_source=args.node_source,
        include_tower_nodes=args.include_tower_nodes,
        tower_airfoil_id=args.tower_airfoil_id,
        n_freq_depth=args.n_freq_depth,
        force_scale=args.force_scale,
        diagnostic_force_scale=args.diagnostic_force_scale,
        include_mean_load=args.include_mean_load,
        source_bounds_policy=args.source_bounds_policy,
        source_frequency_cap_hz=args.source_frequency_cap_hz,
        hvawt_response_model=hvawt_response_model,
        hvawt_response_artifact_dir=hvawt_response_artifact_dir,
        use_qblade_local_inflow=args.use_qblade_local_inflow,
        fluid_density=args.fluid_density,
        fluid_dynamicviscosity=args.fluid_dynamicviscosity,
        rotation_axis=None if args.rotation_axis is None else tuple(args.rotation_axis),
        rotation_axis_offset=None if args.rotation_axis_offset is None else tuple(args.rotation_axis_offset),
        inflow_vec=None if args.inflow_vec is None else tuple(args.inflow_vec),
        debug=args.debug,
        log_file=args.log_file,
        python_path_entries=tuple(args.python_path_entry),
        include_default_python_path=not args.no_default_python_path,
        default_python_path=REPO_ROOT.resolve(),
        absolute_source_parameter_dir=args.absolute_source_parameter_dir,
        geometry_audit_file=args.geometry_audit_file,
        write_geometry_audit=not args.skip_geometry_audit,
        dry_run=args.dry_run,
    )


def _next_backup_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.vorlap.bak")
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.vorlap.bak.{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _backup_existing_paths(paths: tuple[Path, ...]) -> list[tuple[Path, Path]]:
    backups: list[tuple[Path, Path]] = []
    for path in paths:
        if not path.is_file():
            continue
        backup_path = _next_backup_path(path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        backups.append((path, backup_path))
    return backups


def main() -> None:
    args = parse_args()
    request = _request_from_args(args)
    backups: list[tuple[Path, Path]] = []
    if not request.dry_run and args.backup_originals:
        plan = prepare_qblade_external_case(replace(request, dry_run=True))
        backups = _backup_existing_paths(plan.planned_write_paths)

    result = prepare_qblade_external_case(request)

    if result.dry_run:
        print(f"Dry-run QBlade case preparation: {result.sim_path}")
        print("Dry run: no files were written.")
        print("Planned writes:")
        for path in result.planned_write_paths:
            print(f"  {path}")
    else:
        print(f"Prepared QBlade case: {result.sim_path}")
        print(f"Patched turbine definition: {result.turbine_path}")
        print(f"Patched structural definition: {result.structural_path}")
        print(f"Wrote runtime config: {result.parameter_file_path}")
        if result.geometry_audit_path is not None:
            print(f"Wrote geometry audit: {result.geometry_audit_path}")
        if backups:
            print("Backups written:")
            for source, backup in backups:
                print(f"  {source} -> {backup}")
    print(f"Config sim_path: {result.sim_path_for_config}")
    print(f"Config airfoil_dir: {result.airfoil_dir_for_config}")
    print(f"External swap size: {result.swap_size}")
    print(f"Node count: {len(result.node_ids)}")
    print(f"Node source: {result.node_source}")
    print(f"VorLap components built: {result.component_count}")


if __name__ == "__main__":
    main()
