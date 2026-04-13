"""Prepare a QBlade case for live VorLap external-library coupling."""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vorlap
from vorlap.qblade_runtime import (
    build_external_library_table_spec,
    build_qblade_external_config,
    patch_qblade_structural_definition,
    patch_qblade_turbine_definition,
    write_qblade_external_config,
)

_BLD_NODE_RE = re.compile(r"^BLD_\d+_[-+]?\d*\.?\d+$")
_STR_NODE_RE = re.compile(r"^STR_\d+_\d+_[-+]?\d*\.?\d+$")
_TWR_NODE_RE = re.compile(r"^TWR_(?:\d+_)?[-+]?\d*\.?\d+$")


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
    return parser.parse_args()


def _resolve_parameter_file_path(trb_path: Path, parameter_file: Optional[str]) -> Path:
    if parameter_file:
        param_path = Path(parameter_file)
        if not param_path.is_absolute():
            param_path = (trb_path.parent / param_path).resolve()
        return param_path
    return (trb_path.parent / "Control" / "vorlap_qblade_external.json").resolve()


def extract_structural_node_ids(structural_text: str, include_tower_nodes: bool = False) -> List[str]:
    """Extract aerodynamic node ids from a QBlade structural file."""
    node_ids: List[str] = []
    seen = set()
    for raw in structural_text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        is_blade = bool(_BLD_NODE_RE.match(line))
        is_strut = bool(_STR_NODE_RE.match(line))
        is_tower = include_tower_nodes and bool(_TWR_NODE_RE.match(line))
        if (is_blade or is_strut or is_tower) and line not in seen:
            seen.add(line)
            node_ids.append(line)
    return node_ids


def main() -> None:
    args = parse_args()
    sim_path = Path(args.sim).resolve()
    airfoil_dir_arg = Path(args.airfoils)
    airfoil_dir = (
        airfoil_dir_arg.resolve()
        if airfoil_dir_arg.is_absolute()
        else (Path.cwd() / airfoil_dir_arg).resolve()
    )
    if not airfoil_dir.is_dir():
        raise FileNotFoundError(f"Airfoil directory does not exist: {airfoil_dir}")

    sim_info = vorlap.load_qblade_simulation_definition(str(sim_path))
    turbfile_path = Path(sim_info["turbfile_path"]).resolve()
    trb_info = vorlap.load_qblade_turbine_definition(str(turbfile_path))
    structural_path = Path(trb_info["structuralfile_path"]).resolve()

    components, _viv_params, converted_node_ids = vorlap.convert_qblade_to_vorlap_inputs(
        str(sim_path),
        default_airfoil_id=args.default_airfoil_id,
        include_struts=args.include_struts,
        include_tower=args.include_tower,
        tower_airfoil_id=args.tower_airfoil_id,
    )

    str_text = structural_path.read_text(encoding="utf-8", errors="ignore")
    structural_node_ids = extract_structural_node_ids(
        str_text, include_tower_nodes=args.include_tower_nodes
    )
    if args.node_source == "structural":
        node_ids = structural_node_ids if structural_node_ids else list(converted_node_ids)
    else:
        node_ids = list(converted_node_ids)
    table_spec = build_external_library_table_spec(node_ids)

    parameter_file_path = _resolve_parameter_file_path(turbfile_path, args.parameter_file)
    try:
        sim_path_for_config = Path(
            os.path.relpath(str(sim_path), start=str(parameter_file_path.parent))
        ).as_posix()
    except ValueError:
        # Relative conversion can fail on Windows when drives differ.
        sim_path_for_config = str(sim_path)

    try:
        airfoil_dir_for_config = Path(
            os.path.relpath(str(airfoil_dir), start=str(parameter_file_path.parent))
        ).as_posix()
    except ValueError:
        # Relative conversion can fail on Windows when drives differ.
        airfoil_dir_for_config = str(airfoil_dir)

    config = build_qblade_external_config(
        sim_path=sim_path_for_config,
        airfoil_dir=airfoil_dir_for_config,
        node_ids=node_ids,
        library_stem=args.library_stem,
        function_name=args.function_name,
        parameter_file=str(parameter_file_path.relative_to(turbfile_path.parent).as_posix())
        if parameter_file_path.is_relative_to(turbfile_path.parent)
        else str(parameter_file_path),
        sample_step=args.sample_step,
        default_airfoil_id=args.default_airfoil_id,
        include_struts=args.include_struts,
        include_tower=args.include_tower,
        tower_airfoil_id=args.tower_airfoil_id,
        n_freq_depth=args.n_freq_depth,
        force_scale=args.force_scale,
        source_parameter_dir=str(parameter_file_path.parent),
    )
    write_qblade_external_config(str(parameter_file_path), config)

    trb_text = turbfile_path.read_text(encoding="utf-8", errors="ignore")
    patched_trb = patch_qblade_turbine_definition(
        trb_text,
        library_stem=args.library_stem,
        function_name=args.function_name,
        swap_size=table_spec.swap_size,
        parameter_file=str(parameter_file_path.relative_to(turbfile_path.parent).as_posix())
        if parameter_file_path.is_relative_to(turbfile_path.parent)
        else str(parameter_file_path),
    )
    turbfile_path.write_text(patched_trb, encoding="utf-8")

    patched_str = patch_qblade_structural_definition(str_text, table_spec=table_spec)
    structural_path.write_text(patched_str, encoding="utf-8")

    print(f"Prepared QBlade case: {sim_path}")
    print(f"Patched turbine definition: {turbfile_path}")
    print(f"Patched structural definition: {structural_path}")
    print(f"Wrote runtime config: {parameter_file_path}")
    print(f"Config sim_path: {sim_path_for_config}")
    print(f"Config airfoil_dir: {airfoil_dir_for_config}")
    print(f"External swap size: {table_spec.swap_size}")
    print(f"Node count: {len(node_ids)}")
    print(f"Node source: {args.node_source}")
    print(f"VorLap components built: {len(components)}")


if __name__ == "__main__":
    main()
