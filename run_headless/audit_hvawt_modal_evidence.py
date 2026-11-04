#!/usr/bin/env python3
"""Fail-closed clean-checkout audit for tracked H-VAWT modal evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from identify_hvawt_response_modes import compare_independent_action_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "run_headless" / "config"
SOURCE_SUBSTRUCTURE_RELATIVE = (
    Path("examples")
    / "ReferenceHVAWTQBladeModel"
    / "10m_HVAWT_reference_turbine"
    / "Structure"
    / "10m_HVAWT_reference_substructure.str"
)
EXPECTED_OWENS_COMMIT = "883e721aa003efc0d952241d4af1bf609d51f238"
EXPECTED_PROJECT_SHA256 = "a5424041322ca3e655f7546e8b75a58c0f4d5c12411e0f53cd211840bcab3f3c"
EXPECTED_MANIFEST_SHA256 = "65a608d4afa9ddcec5f5180baf379e4a73c787f355577ae24b4521c8fda26405"
EXPECTED_SOURCE_SUBSTRUCTURE_SHA256 = "0093c6970a257e7bbd206ee15d81f84c70b5e138decb15e752c838c388e747dc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _compare_nested(actual: object, expected: object, label: str) -> None:
    if isinstance(expected, float):
        _require(
            isinstance(actual, (float, int))
            and math.isclose(float(actual), expected, rel_tol=2.0e-12, abs_tol=2.0e-14),
            f"{label} differs: {actual!r} versus {expected!r}",
        )
    elif isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{label} is not an object")
        _require(set(actual) == set(expected), f"{label} keys differ")
        for key, value in expected.items():
            _compare_nested(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list), f"{label} is not a list")
        _require(len(actual) == len(expected), f"{label} length differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare_nested(actual_item, expected_item, f"{label}[{index}]")
    else:
        _require(actual == expected, f"{label} differs: {actual!r} versus {expected!r}")


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition(":")
        _require(bool(separator), f"Malformed metadata line: {raw!r}")
        result[key.strip()] = value.strip()
    return result


def audit(
    *,
    repo_root: Path = REPO_ROOT,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    config_dir = config_dir.resolve()
    marker_path = config_dir / "hvawt_modal_evidence_review_marker.json"
    identification_path = config_dir / "hvawt_response_mode_identification.json"
    marker = _json(marker_path)
    identification = _json(identification_path)

    tracked_files = {
        "run_headless/config/hvawt_response_mode_identification.json": identification_path,
        "run_headless/config/hvawt_response_mode_shapes.csv": config_dir / "hvawt_response_mode_shapes.csv",
        "run_headless/config/hvawt_response_mode_force_duals.csv": config_dir / "hvawt_response_mode_force_duals.csv",
        "run_headless/config/hvawt_response_mode_shapes_repeat.csv": config_dir / "hvawt_response_mode_shapes_repeat.csv",
        "run_headless/config/hvawt_response_mode_force_duals_repeat.csv": config_dir / "hvawt_response_mode_force_duals_repeat.csv",
        "run_headless/config/hvawt_owens_request.json": config_dir / "hvawt_owens_request.json",
        "run_headless/config/hvawt_owens_run_metadata.txt": config_dir / "hvawt_owens_run_metadata.txt",
        "run_headless/config/hvawt_campbell_mode_tracking.csv": config_dir / "hvawt_campbell_mode_tracking.csv",
        "run_headless/config/hvawt_campbell_mode_properties.csv": config_dir / "hvawt_campbell_mode_properties.csv",
    }
    for label, path in tracked_files.items():
        _require(path.is_file(), f"Tracked evidence is missing: {label}")
        if path.suffix == ".csv":
            raw = path.read_bytes()
            _require(b"\r" not in raw, f"Tracked CSV is not LF-only: {label}")
            _require(raw.endswith(b"\n"), f"Tracked CSV lacks final LF: {label}")
    marker_hashes = marker.get("tracked_evidence", {})
    _require(set(marker_hashes) == set(tracked_files), "Marker tracked-evidence set differs")
    for label, path in tracked_files.items():
        _require(_sha256(path) == marker_hashes[label], f"Tracked hash mismatch: {label}")

    implementation_files = {
        "run_headless/qblade_to_owens.py": repo_root / "run_headless" / "qblade_to_owens.py",
        "run_headless/run_owens_campbell.jl": repo_root / "run_headless" / "run_owens_campbell.jl",
        "run_headless/identify_hvawt_response_modes.py": repo_root / "run_headless" / "identify_hvawt_response_modes.py",
        "run_headless/audit_hvawt_modal_evidence.py": repo_root / "run_headless" / "audit_hvawt_modal_evidence.py",
    }
    marker_implementation = marker.get("implementation", {})
    _require(
        set(marker_implementation) == set(implementation_files),
        "Marker implementation set differs",
    )
    for label, path in implementation_files.items():
        _require(_sha256(path) == marker_implementation[label], f"Implementation hash mismatch: {label}")

    request = _json(tracked_files["run_headless/config/hvawt_owens_request.json"])
    raw_modal = identification["provenance"]["modal_artifacts"]
    expected_request = {
        "schema": "vorlap-owens-campbell-request-v1",
        "conversion_manifest_sha256": raw_modal["conversion_manifest.json"]["sha256"],
        "runner_sha256": _sha256(implementation_files["run_headless/run_owens_campbell.jl"]),
        "owens_project": "OWENS.jl",
        "owens_project_toml_sha256": EXPECTED_PROJECT_SHA256,
        "owens_manifest_toml_sha256": EXPECTED_MANIFEST_SHA256,
        "julia_executable": "julia",
        "execution_provenance": {
            "python_version": "3.11.5",
            "python_implementation": "CPython",
            "julia_version_output": "julia version 1.12.6",
            "owens_git_commit": EXPECTED_OWENS_COMMIT,
            "owens_git_dirty": False,
            "owens_git_status_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "rpms": "0,0.25,0.5,0.75,1",
        "modes": 12,
        "save_rpms": "0,1",
        "save_modes": "1,2,3,4,5,6,7,8,9,10,11,12",
        "vtk_modes": "6,12",
        "mode_scaling": 3.0,
        "vtk_steps": 4,
        "gravity": True,
        "structural_damping": False,
    }
    _compare_nested(request, expected_request, "owens_request")

    metadata = _metadata(
        tracked_files["run_headless/config/hvawt_owens_run_metadata.txt"]
    )
    expected_metadata = {
        "Julia version": "1.12.6",
        "RPM values": "0.0,0.25,0.5,0.75,1.0",
        "Modes": "12",
        "Gravity": "true",
        "Structural damping": "false (matches OWENS AutoCampbellDiagram GX default)",
        "elements.csv beta storage": "qblade_2_0_9_7_documented candidate, unselected and unused by the eigensolve",
    }
    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"OWENS metadata {key!r} differs")

    tracking = _rows(
        tracked_files["run_headless/config/hvawt_campbell_mode_tracking.csv"]
    )
    _require(len(tracking) == 60, "Tracked Campbell mode tracking must have 60 rows")
    tracking_keys = {(float(row["rpm"]), int(row["tracked_mode"])) for row in tracking}
    _require(
        tracking_keys
        == {(rpm, mode) for rpm in (0.0, 0.25, 0.5, 0.75, 1.0) for mode in range(1, 13)},
        "Tracked Campbell mode coverage differs",
    )
    clusters: dict[int, list[int]] = {}
    for row in tracking:
        if float(row["rpm"]) == 0.0:
            clusters.setdefault(int(row["reference_cluster"]), []).append(
                int(row["tracked_mode"])
            )
    degenerate = [sorted(values) for _, values in sorted(clusters.items()) if len(values) > 1]
    _require(
        degenerate == [[1, 2], [3, 4], [7, 8], [9, 10, 11]],
        "Zero-RPM degenerate clusters differ",
    )

    properties = _rows(
        tracked_files["run_headless/config/hvawt_campbell_mode_properties.csv"]
    )
    _require(len(properties) == 24, "Tracked modal properties must have 24 rows")
    property_keys = {(float(row["rpm"]), int(row["tracked_mode"])) for row in properties}
    _require(
        property_keys == {(rpm, mode) for rpm in (0.0, 1.0) for mode in range(1, 13)},
        "Tracked modal-property coverage differs",
    )
    _require(
        "elements_csv_beta_projection_raw_unselected" in properties[0],
        "Tracked properties do not label beta projection as unselected",
    )

    repeatability = compare_independent_action_evidence(
        tracked_files["run_headless/config/hvawt_response_mode_shapes.csv"],
        tracked_files["run_headless/config/hvawt_response_mode_force_duals.csv"],
        tracked_files["run_headless/config/hvawt_response_mode_shapes_repeat.csv"],
        tracked_files["run_headless/config/hvawt_response_mode_force_duals_repeat.csv"],
    )
    _require(repeatability["passed"], "Tracked independent repeatability gates failed")
    _compare_nested(
        repeatability,
        identification["independent_regeneration_evidence"],
        "independent_regeneration_evidence",
    )
    _compare_nested(
        repeatability["summary"], marker["independent_regeneration"]["summary"],
        "marker.independent_regeneration.summary",
    )

    clean_evidence = identification["provenance"]["tracked_clean_checkout_evidence"]
    expected_clean_names = {
        Path(label).name for label in tracked_files if Path(label).name not in {
            "hvawt_response_mode_identification.json",
            "hvawt_response_mode_shapes.csv",
            "hvawt_response_mode_force_duals.csv",
        }
    }
    _require(set(clean_evidence) == expected_clean_names, "Identification clean-evidence set differs")
    for name, identity in clean_evidence.items():
        path = config_dir / name
        _require(identity["sha256"] == _sha256(path), f"Identification hash differs: {name}")
        _require(not Path(str(identity["label"])).is_absolute(), f"Absolute evidence label: {name}")

    _require(
        identification["damping_evidence"]["selected_damping_candidate_id"] is None,
        "A damping candidate was selected",
    )
    for mode in identification["modes"]:
        _require(mode["selected_damping_candidate_id"] is None, "Mode damping was selected")
    source_artifacts = {
        item["label"]: item["sha256"]
        for item in identification["provenance"]["conversion_source_artifacts"]
    }
    source_label = SOURCE_SUBSTRUCTURE_RELATIVE.as_posix()
    _require(
        source_artifacts[source_label] == EXPECTED_SOURCE_SUBSTRUCTURE_SHA256,
        "Serialized source-substructure hash differs",
    )
    _require(
        _sha256(repo_root / source_label) == EXPECTED_SOURCE_SUBSTRUCTURE_SHA256,
        "Tracked source substructure was modified",
    )

    absolute_pattern = re.compile(r"(?<![A-Za-z0-9])/(?:Users|private|home|tmp)/|[A-Za-z]:\\")
    portability_files = [marker_path, *tracked_files.values()]
    for path in portability_files:
        text = path.read_text(encoding="utf-8")
        _require(
            absolute_pattern.search(text) is None,
            f"Absolute host path leaked into tracked evidence: {path.name}",
        )

    return {
        "schema": "vorlap-hvawt-modal-evidence-audit-v1",
        "status": "pass",
        "tracked_file_count": len(tracked_files),
        "owens_commit": EXPECTED_OWENS_COMMIT,
        "julia_version": "1.12.6",
        "degenerate_zero_rpm_clusters": degenerate,
        "repeatability_summary": repeatability["summary"],
        "selected_damping_candidate_id": None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(audit(repo_root=args.repo_root, config_dir=args.config_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
