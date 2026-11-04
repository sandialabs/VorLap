#!/usr/bin/env python3
"""Fail closed unless a sampled-DLC U03 row can reuse its completed baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
VORLAP_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(VORLAP_ROOT) not in sys.path:
    sys.path.insert(0, str(VORLAP_ROOT))

from audit_parked_campaign import approved_reynolds_envelope_extension_matches  # noqa: E402
from vorlap.dlc_manifest import DLCManifestError, load_manifest, selected_row  # noqa: E402
from vorlap.qblade_case_prep import build_airfoil_source_manifest  # noqa: E402
from vorlap_qblade_batch import completed_filtered_export_ready  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--vorlap-root", type=Path, default=VORLAP_ROOT)
    return parser.parse_args()


def verify_recovery(manifest_path: Path, index: int, vorlap_root: Path) -> dict[str, object]:
    manifest = load_manifest(manifest_path.expanduser().resolve(), require_approved=True)
    row = selected_row(manifest, index)
    if index not in {0, 1, 2} or float(row["wind_speed_mps"]) != 3.0:
        raise DLCManifestError(
            f"U03 recovery accepts only approved manifest rows 0, 1, and 2; got row {index} "
            f"({row['row_id']})."
        )

    source_dir = (
        vorlap_root.expanduser().resolve()
        / "examples"
        / "ReferenceHVAWTQBladeModel"
        / "10m_HVAWT_reference_turbine"
        / "VorLapAirfoils"
    )
    source_rows = [
        source
        for source in build_airfoil_source_manifest(source_dir)
        if source.get("airfoil_id") == "NACA0018"
    ]
    if len(source_rows) != 1:
        raise DLCManifestError("Recovery requires exactly one active NACA0018 source file.")
    source = source_rows[0]
    parent_sha256 = str(manifest["basis"]["airfoil_source_sha256"])
    if not approved_reynolds_envelope_extension_matches(source, parent_sha256):
        raise DLCManifestError(
            "Active NACA0018 source is not the approved Re=10,000 envelope extension of the "
            "production manifest parent. Pull the recovery commit before resubmitting."
        )

    run_root = Path(str(row["run_root"]))
    baseline_manifest_path = run_root / "baseline" / "cases.json"
    if not baseline_manifest_path.is_file():
        raise DLCManifestError(f"Completed baseline case manifest is missing: {baseline_manifest_path}")
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    cases = baseline_manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise DLCManifestError("U03 recovery requires exactly one existing baseline case.")
    result_filter = Path(str(manifest["execution_settings"]["result_filter_path"]))
    if not completed_filtered_export_ready(cases[0], result_filter):
        raise DLCManifestError(
            "The U03 baseline is not a complete hash-verified filtered export; refusing a recovery "
            "that could rerun it."
        )

    return {
        "passed": True,
        "row_index": index,
        "row_id": row["row_id"],
        "run_root": str(run_root),
        "baseline_reuse": "complete_hash_verified_export",
        "baseline_will_advance_timesteps": False,
        "naca0018_parent_sha256": parent_sha256,
        "naca0018_runtime_sha256": source["sha256"],
        "naca0018_reynolds": source["reynolds"],
    }


def main() -> int:
    args = parse_args()
    try:
        report = verify_recovery(args.manifest, args.index, args.vorlap_root)
    except (DLCManifestError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
