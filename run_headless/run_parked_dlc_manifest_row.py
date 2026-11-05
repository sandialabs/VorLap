#!/usr/bin/env python3
"""Run one validated parked-DLC manifest row for wind generation or QBlade."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
VORLAP_ROOT = SCRIPT_DIR.parent
if str(VORLAP_ROOT) not in sys.path:
    sys.path.insert(0, str(VORLAP_ROOT))

from vorlap.dlc_manifest import (  # noqa: E402
    DLCManifestError,
    build_wind_record,
    load_manifest,
    selected_row,
    sha256_file,
    validate_wind_record,
    wind_record_path,
    write_json,
)
from vorlap.execution_provenance import build_execution_provenance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--phase", choices=("wind", "qblade"), required=True)
    parser.add_argument(
        "--parent-submit",
        type=Path,
        default=SCRIPT_DIR / "submit_parked_demo_campaign.sj",
    )
    parser.add_argument(
        "--parent-dry-run",
        action="store_true",
        help="Run the selected row through the parent campaign's DRY_RUN_ONLY path.",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow a draft manifest only with --parent-dry-run.",
    )
    return parser.parse_args()


def row_environment(
    manifest: dict,
    row: dict,
    *,
    phase: str,
    parent_submit: Path,
    parent_dry_run: bool,
) -> dict[str, str]:
    wind = manifest["wind_settings"]
    execution = manifest["execution_settings"]
    result_filter = Path(str(execution["result_filter_path"])).expanduser().resolve()
    if not result_filter.is_file():
        raise DLCManifestError(f"QBlade result filter does not exist: {result_filter}")
    observed_filter_sha256 = sha256_file(result_filter)
    if observed_filter_sha256 != execution["result_filter_sha256"]:
        raise DLCManifestError(
            "QBlade result filter hash does not match the immutable DLC manifest: "
            f"{result_filter}"
        )
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("SLURM_ARRAY_"):
            env.pop(name, None)
    env.update(
        {
            "VORLAP_SUBMIT_SCRIPT": str(parent_submit),
            "RUN_ROOT": str(row["run_root"]),
            "WIND_SPEEDS": f"{float(row['wind_speed_mps']):.12g}",
            "PRODUCTION_WIND_SPEEDS": f"{float(row['wind_speed_mps']):.12g}",
            "QBLADE_AZIMUTHS": f"{float(row['parked_azimuth_deg']):.12g}",
            "PRODUCTION_AZIMUTHS": f"{float(row['parked_azimuth_deg']):.12g}",
            "QBLADE_RPM": f"{float(row['rpm']):.12g}",
            "PRODUCTION_RPM": f"{float(row['rpm']):.12g}",
            "QBLADE_DURATION": f"{float(manifest['duration_s']):.12g}",
            "QBLADE_STORE_FROM": f"{float(manifest['store_from_s']):.12g}",
            "QBLADE_TIMESTEP": f"{float(manifest['timestep_s']):.12g}",
            "POSTPROCESS_MIN_FATIGUE_DURATION": "600",
            "POSTPROCESS_EQUIVALENT_CYCLES": "600",
            "CAMPAIGN_SCENARIOS": " ".join(manifest["scenarios"]),
            "CASE_WORKERS": str(int(execution["case_workers"])),
            "VORLAP_AUTO_NODE_THREADS": "1" if execution["auto_node_threads"] else "0",
            "QBLADE_USE_OPENCL": "true" if execution["use_opencl"] else "false",
            "QBLADE_STORE_REPLAY": "true" if execution["store_replay"] else "false",
            "QBLADE_STORE_PROJECT": "true" if execution["store_project"] else "false",
            "QBLADE_CHECKPOINT_EVERY": str(int(execution["checkpoint_every"])),
            "QBLADE_WAKE_PROFILE": str(execution["wake_profile"]),
            "QBLADE_RESULT_FILTER": str(result_filter),
            "VORLAP_SOURCE_FREQUENCY_TRUNCATION": str(execution["source_frequency_truncation"]),
            "VORLAP_SOURCE_BOUNDS_POLICY": str(execution["source_bounds_policy"]),
            "QBLADE_WIND_NAME": str(row["wind_file"]),
            "REQUIRE_WIND_NAME": "1",
            "REQUIRE_PRODUCTION_MATRIX": "1",
            "REQUIRE_DLC_MANIFEST_ROW": "1",
            "REQUIRE_WIND_RECORD": "0" if parent_dry_run or phase == "wind" else "1",
            "REQUIRE_SOURCE_COVERAGE": "1" if execution["require_source_coverage"] else "0",
            "REQUIRE_POSTPROCESS_ARTIFACTS": "1" if execution["require_postprocess_artifacts"] else "0",
            "RUN_FINAL_AUDIT": "1" if execution["run_final_audit"] else "0",
            "RUN_SIGN_PROBES": "0",
            "REQUIRE_FORCE_SIGN_PROBES": "0",
            "DLC_MANIFEST_ROW_ARCHIVE": str(Path(str(row["run_root"])) / "dlc_manifest_row.json"),
            "TURBSIM_WIND_ROOT": str(Path(str(row["wind_file"])).parent),
            "TURBSIM_TIMESTEP": f"{float(wind['time_step_s']):.12g}",
            "TURBSIM_ANALYSIS_TIME": f"{float(wind['analysis_time_s']):.12g}",
            "TURBSIM_NUM_GRID_Y": str(int(wind["num_grid_y"])),
            "TURBSIM_NUM_GRID_Z": str(int(wind["num_grid_z"])),
            "TURBSIM_GRID_WIDTH": f"{float(wind['grid_width_m']):.12g}",
            "TURBSIM_GRID_HEIGHT": f"{float(wind['grid_height_m']):.12g}",
            "TURBSIM_HUB_HEIGHT": f"{float(wind['hub_height_m']):.12g}",
            "TURBSIM_REF_HEIGHT": f"{float(wind['ref_height_m']):.12g}",
            "QBLADE_HORIZONTAL_INFLOW_ANGLE": f"{float(wind['horizontal_inflow_angle_deg']):.12g}",
            "QBLADE_VERTICAL_INFLOW_ANGLE": f"{float(wind['vertical_inflow_angle_deg']):.12g}",
            "QBLADE_SHEAR_EXPONENT": f"{float(wind['shear_exponent']):.12g}",
            "QBLADE_REFERENCE_HEIGHT": f"{float(wind['ref_height_m']):.12g}",
            "QBLADE_AIR_DENSITY": f"{float(wind['air_density_kg_m3']):.12g}",
            "QBLADE_AIR_KINEMATIC_VISCOSITY": f"{float(wind['air_kinematic_viscosity_m2_s']):.12g}",
            "TURBSIM_SEED_BASE": str(int(row["seed"])),
            "TURBSIM_TURB_MODEL": str(row["turbsim_turb_model"]),
            "TURBSIM_IEC_STANDARD": str(row["turbsim_iec_standard"]),
            "TURBSIM_IEC_TURBC": str(row["turbsim_iec_turbc"]),
            "TURBSIM_IEC_WIND_TYPE": str(row["turbsim_iec_wind_type"]),
            "GENERATE_TURBSIM": "1" if phase == "wind" else "0",
            "WIND_ONLY": "1" if phase == "wind" else "0",
            "AUTO_SETUP_FIRST_RUN": "0",
            "AUTO_SETUP_TURBSIM": "0",
            "FORCE_PYTHON_SETUP": "0",
            "FORCE_OPENFAST_TURBSIM_BUILD": "0",
            "FORCE_BRIDGE_BUILD": "0",
            "FORCE_CASE_SETUP": "0",
            "FORCE_GENERATE_TURBSIM": "0",
            "SKIP_VORLAP_INSTALL": "1",
            "SKIP_BRIDGE_BUILD": "1",
            "RUN_PAPER_PREFLIGHT": "0" if phase == "wind" else env.get("RUN_PAPER_PREFLIGHT", "1"),
            "DRY_RUN_ONLY": "1" if parent_dry_run else "0",
        }
    )
    env.pop("QBLADE_STEPS", None)
    return env


def main() -> int:
    args = parse_args()
    if args.allow_draft and not args.parent_dry_run:
        print("ERROR: --allow-draft is permitted only with --parent-dry-run.", file=sys.stderr)
        return 2
    try:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = load_manifest(manifest_path, require_approved=not args.allow_draft)
        row = selected_row(manifest, args.index)
        parent_submit = args.parent_submit.expanduser().resolve()
        if not parent_submit.is_file():
            raise FileNotFoundError(f"Parent campaign submitter not found: {parent_submit}")

        run_root = Path(str(row["run_root"]))
        run_root.mkdir(parents=True, exist_ok=True)
        row_archive = {
            "schema_version": 1,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "row_index": args.index,
            "row": row,
        }
        write_json(run_root / "dlc_manifest_row.json", row_archive)

        if args.phase == "qblade" and not args.parent_dry_run:
            validate_wind_record(row, manifest)

        env = row_environment(
            manifest,
            row,
            phase=args.phase,
            parent_submit=parent_submit,
            parent_dry_run=args.parent_dry_run,
        )
        if args.allow_draft:
            env["REQUIRE_DLC_MANIFEST_ROW"] = "0"
        execution_provenance = build_execution_provenance(
            manifest=manifest,
            row=row,
            row_index=args.index,
            phase=args.phase,
            environment=env,
            vorlap_root=VORLAP_ROOT,
        )
        write_json(
            run_root / f"execution_environment_{args.phase}.json",
            execution_provenance,
        )
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "row_index": args.index,
                    "row_id": row["row_id"],
                    "run_root": row["run_root"],
                    "wind_file": row["wind_file"],
                    "parent_dry_run": args.parent_dry_run,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        completed = subprocess.run(["bash", str(parent_submit)], env=env, check=False)
        if completed.returncode != 0:
            return completed.returncode

        if args.phase == "wind" and not args.parent_dry_run:
            record = build_wind_record(row, manifest)
            write_json(wind_record_path(row), record)
            write_json(run_root / "wind_record.json", record)
        return 0
    except (DLCManifestError, FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
