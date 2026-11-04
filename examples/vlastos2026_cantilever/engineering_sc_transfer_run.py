#!/usr/bin/env python3
"""Execute and assess the frozen eight-case Belloli Scruton transfer screen.

This adapter intentionally keeps the Belloli transfer campaign separate from
the original schema-v2 holdout.  The Sc=2.94 and Sc=7.02 targets informed the
transfer-form selection, so a passing result is an engineering transfer screen,
not independent validation.  Execution remains conditional on a passing
24-case Sc=5.18 calibration-readiness artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Package imports in tests and ``python -m`` use.
    from examples.vlastos2026_cantilever import engineering_holdout
    from examples.vlastos2026_cantilever import engineering_holdout_run
    from examples.vlastos2026_cantilever import engineering_sc_transfer
    from examples.vlastos2026_cantilever import response_benchmark
except ImportError:  # Direct execution from this example directory.
    import engineering_holdout  # type: ignore[no-redef]
    import engineering_holdout_run  # type: ignore[no-redef]
    import engineering_sc_transfer  # type: ignore[no-redef]
    import response_benchmark  # type: ignore[no-redef]


PLAN_SCHEMA = "vlastos2026-belloli-sc-transfer-execution-plan-v1"
ASSESSMENT_SCHEMA = "vlastos2026-belloli-sc-transfer-live-screen-v1"
EXECUTION_INTENT = "transfer_tuning_engineering_screen_not_independent_validation"
SCOPE = (
    "Eight-case live-QBlade Belloli Scruton transfer tuning and engineering "
    "screen; targets informed transfer-form selection, so this is not "
    "independent validation, full-curve validation, fatigue validation, or "
    "H-VAWT validation."
)


def _json_file(path: Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _file_identity(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": source.as_posix(),
        "sha256": response_benchmark.sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _passing_readiness(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return engineering_holdout_run._load_passing_readiness(Path(path))


def _manifest_file_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["path"]): {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "bytes": int(row["bytes"]),
            "size_bytes": int(row["bytes"]),
        }
        for row in manifest["files"]
    }


def _effective_fingerprints(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(entry["case_id"]): str(
            entry["effective_parameter_record"][
                "effective_parameter_fingerprint_sha256"
            ]
        )
        for entry in manifest["cases"]
    }


def build_execution_plan(
    manifest_path: Path,
    calibration_readiness_path: Path,
    results_root: Path,
    *,
    qblade_root: Path,
    checkpoint_every: int = 0,
    threads: int = 1,
) -> dict[str, Any]:
    """Build the canonical local/HPC plan without launching QBlade."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    campaign = engineering_sc_transfer.validate_transfer_campaign(manifest_file)
    readiness, readiness_identity = _passing_readiness(calibration_readiness_path)
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads < 8:
        raise ValueError("threads must be an integer from 1 through 7.")
    if (
        isinstance(checkpoint_every, bool)
        or not isinstance(checkpoint_every, int)
        or checkpoint_every < 0
    ):
        raise ValueError("checkpoint_every must be a non-negative integer.")

    campaign_root = manifest_file.parent
    results = Path(results_root).expanduser().resolve()
    qblade = Path(qblade_root).expanduser().resolve()
    response_benchmark._validate_v2_results_root(
        results,
        campaign_root=campaign_root,
        qblade_root=qblade,
    )
    runtime_identities = response_benchmark._ambient_v2_plan_runtime_identities(
        campaign_root,
        campaign,
        qblade,
    )
    contract = campaign["runtime_contract"]
    runner = (engineering_sc_transfer.REPO_ROOT / contract["runner"]["path"]).resolve()
    result_filter = (campaign_root / contract["result_filter"]["path"]).resolve()
    commands: list[dict[str, Any]] = []
    for entry in campaign["cases"]:
        case_output = results / str(entry["case_id"])
        argv = [
            sys.executable,
            str(runner),
            "--sim",
            str(campaign_root / str(entry["sim_path"])),
            "--qblade-root",
            str(qblade),
            "--output-dir",
            str(case_output),
            "--result-name",
            "result",
            "--result-filter",
            str(result_filter),
            "--threads",
            str(threads),
            "--sil-vorlap",
            "--coupling-mode",
            "sil-vorlap",
            "--sil-vorlap-history-mode",
            "compact",
        ]
        if checkpoint_every:
            argv.extend(
                (
                    "--checkpoint-every",
                    str(checkpoint_every),
                    "--checkpoint-dir",
                    str(case_output / "checkpoints"),
                )
            )
        commands.append(
            {
                "case_id": entry["case_id"],
                "role": "transfer_tuning_engineering_screen",
                "argv": argv,
            }
        )

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "execution_intent": EXECUTION_INTENT,
        "scientific_scope": SCOPE,
        "independent_validation": False,
        "campaign_manifest": _file_identity(manifest_file),
        "campaign_manifest_fingerprint_sha256": campaign[
            "manifest_fingerprint_sha256"
        ],
        "transfer_definition_fingerprint_sha256": campaign["sc_transfer"][
            "definition_fingerprint_sha256"
        ],
        "analytical_gate_fingerprint_sha256": campaign["analytical_gate"][
            "assessment_fingerprint_sha256"
        ],
        "effective_parameter_fingerprints_by_case": _effective_fingerprints(
            campaign
        ),
        "calibration_readiness": readiness_identity,
        "calibration_readiness_fingerprint_sha256": readiness[
            "assessment_fingerprint_sha256"
        ],
        "results_root": results.as_posix(),
        "qblade_root": qblade.as_posix(),
        "python_executable": str(Path(sys.executable).resolve()),
        "source_repo_root": engineering_sc_transfer.REPO_ROOT.resolve().as_posix(),
        "threads": threads,
        "checkpoint_every": checkpoint_every,
        "runtime_identities": runtime_identities,
        "case_ids": [str(entry["case_id"]) for entry in campaign["cases"]],
        "commands": commands,
    }
    plan["plan_fingerprint_sha256"] = engineering_sc_transfer.canonical_sha256(plan)
    return plan


def _canonical_plan_from_supplied(
    supplied: Mapping[str, Any],
    manifest_path: Path,
    calibration_readiness_path: Path,
) -> dict[str, Any]:
    return build_execution_plan(
        manifest_path,
        calibration_readiness_path,
        Path(str(supplied.get("results_root", ""))),
        qblade_root=Path(str(supplied.get("qblade_root", ""))),
        checkpoint_every=supplied.get("checkpoint_every"),
        threads=supplied.get("threads"),
    )


def _assert_transfer_post_dispatch(
    campaign_root: Path,
    manifest_path: Path,
    baseline_tree: tuple[tuple[Any, ...], ...],
    row: Mapping[str, Any],
) -> None:
    if response_benchmark._campaign_tree_snapshot(campaign_root) != baseline_tree:
        raise RuntimeError("Transfer subprocess mutated the immutable campaign tree.")
    engineering_sc_transfer.validate_transfer_campaign(manifest_path)
    argv = list(row["argv"])
    case_output = Path(str(argv[argv.index("--output-dir") + 1]))
    summaries = sorted(case_output.glob("*_run_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            "Transfer subprocess must publish exactly one run summary; "
            f"observed {len(summaries)} for {row['case_id']}."
        )
    summary = _json_file(summaries[0])
    evidence = summary.get("native_callback_evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeError("Transfer run summary lacks native-callback evidence.")
    activity = (
        bool(evidence.get("log_file_exists"))
        or int(evidence.get("update_init_count", 0) or 0) != 0
        or int(evidence.get("update_count", 0) or 0) != 0
        or int(evidence.get("update_count_observed_max", 0) or 0) != 0
    )
    if activity:
        raise RuntimeError("Unexpected native callback activity in transfer SIL execution.")


def execute_case(
    plan: Mapping[str, Any],
    manifest_path: Path,
    calibration_readiness_path: Path,
    *,
    case_index: int,
    acknowledge_transfer_tuning_screen: bool,
) -> str:
    """Execute exactly one zero-based array index from a canonical plan."""

    if acknowledge_transfer_tuning_screen is not True:
        raise RuntimeError(
            "Execution requires explicit acknowledgement that this is a "
            "transfer-tuning engineering screen, not independent validation."
        )
    if isinstance(case_index, bool) or not isinstance(case_index, int):
        raise TypeError("case_index must be an integer.")
    if not isinstance(plan, dict):
        raise TypeError("Transfer execution plan must be a JSON object.")
    manifest_file = Path(manifest_path).expanduser().resolve()
    expected = _canonical_plan_from_supplied(
        plan,
        manifest_file,
        Path(calibration_readiness_path).expanduser().resolve(),
    )
    if plan != expected:
        raise RuntimeError("Transfer execution plan differs from the canonical plan.")
    if not 0 <= case_index < len(plan["commands"]):
        raise IndexError(
            f"case_index must be from 0 through {len(plan['commands']) - 1}."
        )

    campaign_root = manifest_file.parent
    campaign = engineering_sc_transfer.validate_transfer_campaign(manifest_file)
    row = plan["commands"][case_index]
    response_benchmark._assert_v2_case_outputs_fresh(plan, [row])
    baseline_tree = response_benchmark._campaign_tree_snapshot(campaign_root)
    by_id = {str(entry["case_id"]): entry for entry in campaign["cases"]}
    entry = by_id[str(row["case_id"])]
    response_benchmark._preflight_response_runtime(
        campaign_root / str(entry["parameter_file"])
    )
    subprocess.run(row["argv"], check=True)
    _assert_transfer_post_dispatch(
        campaign_root,
        manifest_file,
        baseline_tree,
        row,
    )
    return str(row["case_id"])


def _nonempty_result_directories(results_root: Path) -> set[str]:
    root = Path(results_root).expanduser().resolve()
    if not root.exists():
        return set()
    if not root.is_dir():
        raise NotADirectoryError(root)
    return {
        child.name
        for child in root.iterdir()
        if child.is_dir() and any(child.iterdir())
    }


def _incomplete_assessment(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    readiness: Mapping[str, Any],
    results_root: Path,
    missing: Sequence[str],
    unexpected: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": ASSESSMENT_SCHEMA,
        "status": "not_evaluated",
        "evaluated": False,
        "pass": False,
        "scientific_scope": SCOPE,
        "independent_validation": False,
        "campaign_manifest_sha256": response_benchmark.sha256_file(manifest_path),
        "campaign_manifest_fingerprint_sha256": manifest[
            "manifest_fingerprint_sha256"
        ],
        "transfer_definition_fingerprint_sha256": manifest["sc_transfer"][
            "definition_fingerprint_sha256"
        ],
        "effective_parameter_fingerprints_by_case": _effective_fingerprints(
            manifest
        ),
        "calibration_readiness_fingerprint_sha256": readiness[
            "assessment_fingerprint_sha256"
        ],
        "results_root": Path(results_root).expanduser().resolve().as_posix(),
        "expected_case_count": engineering_sc_transfer.EXPECTED_CASE_COUNT,
        "completed_case_count": engineering_sc_transfer.EXPECTED_CASE_COUNT
        - len(missing),
        "missing_case_ids": list(missing),
        "unexpected_nonempty_result_directories": list(unexpected),
        "checks": {
            "all_results_present": not missing,
            "exact_result_directory_set": not missing and not unexpected,
            "case_semantics_pass": False,
            "runtime_provenance_execution_limits_pass": False,
            "stationarity_pass": False,
            "all_timestep_pairs_pass": False,
            "both_peak_amplitudes_pass": False,
            "both_selected_peaks_lock_in_frequency_pass": False,
            "transfer_definition_and_effective_parameters_bound": True,
        },
        "by_scruton_number": [],
        "claim_boundary": {
            "allowed": "engineering transfer-screen adequacy if every declared gate passes",
            "forbidden": [
                "blind independent validation",
                "full lock-in curve validation",
                "fatigue validation",
                "H-VAWT aerodynamic validation",
                "certification qualification",
            ],
        },
    }
    result["assessment_fingerprint_sha256"] = (
        engineering_sc_transfer.canonical_sha256(result)
    )
    return result


def _assess_complete_results(
    manifest: Mapping[str, Any],
    readiness: Mapping[str, Any],
    response_rows: Sequence[Mapping[str, Any]],
    runtime_audits: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    results_root: Path,
) -> dict[str, Any]:
    expected_ids = [str(entry["case_id"]) for entry in manifest["cases"]]
    response_by_id = {str(row["case_id"]): row for row in response_rows}
    runtime_by_id = {str(row["case_id"]): row for row in runtime_audits}
    if (
        len(response_by_id) != len(response_rows)
        or len(runtime_by_id) != len(runtime_audits)
        or set(response_by_id) != set(expected_ids)
        or set(runtime_by_id) != set(expected_ids)
    ):
        raise ValueError("Complete transfer evidence must contain exactly eight unique cases.")
    by_entry = {str(entry["case_id"]): entry for entry in manifest["cases"]}

    case_semantics_pass = all(
        float(response_by_id[case_id].get("scruton_number", math.nan))
        == float(by_entry[case_id]["scruton_number"])
        and float(response_by_id[case_id].get("reduced_velocity", math.nan))
        == float(by_entry[case_id]["reduced_velocity"])
        and response_by_id[case_id].get("timestep_level")
        == by_entry[case_id]["timestep_level"]
        and response_by_id[case_id].get("role")
        == "transfer_tuning_engineering_screen"
        and response_by_id[case_id].get("runtime_audit_evaluated") is True
        and response_by_id[case_id].get("runtime_audit_pass") is True
        for case_id in expected_ids
    )

    runtime_pass = all(
        engineering_holdout._runtime_audit_pass(
            runtime_by_id[case_id],
            expected_response_config_fingerprint=by_entry[case_id][
                "runtime_response_config_fingerprint_sha256"
            ],
        )
        for case_id in expected_ids
    )
    stationarity_pass = all(
        bool(response_by_id[case_id].get("stationarity_evaluated"))
        and bool(response_by_id[case_id].get("stationarity_pass"))
        for case_id in expected_ids
    )

    per_scruton: list[dict[str, Any]] = []
    targets = {
        sc: float(
            engineering_holdout._HELDOUT_CASES[sc][
                "target_peak_amplitude_over_diameter"
            ]
        )
        for sc in engineering_sc_transfer.TRANSFER_TUNING_SCRUTON_NUMBERS
    }
    for sc in engineering_sc_transfer.TRANSFER_TUNING_SCRUTON_NUMBERS:
        target = targets[sc]
        timestep_pairs: list[dict[str, Any]] = []
        fine_candidates: list[Mapping[str, Any]] = []
        for u_star in engineering_sc_transfer.CASE_GRID[sc]:
            levels = {
                str(row["timestep_level"]): row
                for row in response_rows
                if float(row["scruton_number"]) == sc
                and float(row["reduced_velocity"]) == u_star
            }
            if set(levels) != set(engineering_sc_transfer.TIMESTEP_LEVELS):
                raise ValueError(f"Transfer pair Sc={sc:g}, U*={u_star:g} is incomplete.")
            coarse = levels["dt"]
            fine = levels["dt_half"]
            fine_candidates.append(fine)
            coarse_amplitude = float(coarse["tip_peak_amplitude_over_diameter"])
            fine_amplitude = float(fine["tip_peak_amplitude_over_diameter"])
            if not all(
                math.isfinite(value) and value >= 0.0
                for value in (coarse_amplitude, fine_amplitude)
            ):
                raise ValueError("Transfer response amplitude must be finite and nonnegative.")
            amplitude_difference = abs(coarse_amplitude - fine_amplitude)
            amplitude_tolerance = max(
                engineering_holdout.TIMESTEP_ABSOLUTE_AMPLITUDE_TOLERANCE,
                engineering_holdout.TIMESTEP_RELATIVE_AMPLITUDE_TOLERANCE
                * max(coarse_amplitude, fine_amplitude),
            )
            active = max(coarse_amplitude, fine_amplitude) >= (
                engineering_holdout.ACTIVE_AMPLITUDE_FRACTION_OF_TARGET * target
            )
            if active:
                coarse_frequency = float(coarse["terminal_dominant_frequency_over_fn"])
                fine_frequency = float(fine["terminal_dominant_frequency_over_fn"])
                if not all(
                    math.isfinite(value) and value > 0.0
                    for value in (coarse_frequency, fine_frequency)
                ):
                    raise ValueError("Active transfer response frequency must be positive.")
                frequency_difference = abs(coarse_frequency - fine_frequency)
                frequency_pass = frequency_difference <= (
                    engineering_holdout.TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN
                    + engineering_holdout.NUMERICAL_COMPARISON_SLACK
                )
            else:
                coarse_frequency = None
                fine_frequency = None
                frequency_difference = None
                frequency_pass = True
            timestep_pairs.append(
                {
                    "reduced_velocity": u_star,
                    "active": active,
                    "coarse_amplitude_over_diameter": coarse_amplitude,
                    "half_dt_amplitude_over_diameter": fine_amplitude,
                    "amplitude_difference_over_diameter": amplitude_difference,
                    "amplitude_tolerance_over_diameter": amplitude_tolerance,
                    "amplitude_pass": amplitude_difference
                    <= amplitude_tolerance
                    + engineering_holdout.NUMERICAL_COMPARISON_SLACK,
                    "coarse_frequency_over_fn": coarse_frequency,
                    "half_dt_frequency_over_fn": fine_frequency,
                    "frequency_difference_over_fn": frequency_difference,
                    "frequency_tolerance_over_fn": (
                        engineering_holdout.TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN
                    ),
                    "frequency_pass": frequency_pass,
                }
            )

        selected = max(
            fine_candidates,
            key=lambda row: float(row["tip_peak_amplitude_over_diameter"]),
        )
        selected_amplitude = float(selected["tip_peak_amplitude_over_diameter"])
        selected_frequency = float(selected["terminal_dominant_frequency_over_fn"])
        peak_error = abs(selected_amplitude - target)
        peak_tolerance = (
            engineering_holdout.PEAK_RELATIVE_TOLERANCE * target
            + engineering_holdout.DIGITIZATION_UNCERTAINTY_OVER_DIAMETER
        )
        timestep_pass = all(
            row["amplitude_pass"] and row["frequency_pass"]
            for row in timestep_pairs
        )
        frequency_pass = (
            engineering_holdout.LOCK_IN_FREQUENCY_MINIMUM_OVER_FN
            - engineering_holdout.NUMERICAL_COMPARISON_SLACK
            <= selected_frequency
            <= engineering_holdout.LOCK_IN_FREQUENCY_MAXIMUM_OVER_FN
            + engineering_holdout.NUMERICAL_COMPARISON_SLACK
        )
        amplitude_pass = peak_error <= (
            peak_tolerance + engineering_holdout.NUMERICAL_COMPARISON_SLACK
        )
        per_scruton.append(
            {
                "scruton_number": sc,
                "evidence_role": "transfer_tuning_engineering_screen",
                "target_series": "experiment_lupi2021",
                "timestep_pairs": timestep_pairs,
                "timestep_pass": timestep_pass,
                "selected_peak_case_id": selected["case_id"],
                "selected_peak_reduced_velocity": float(selected["reduced_velocity"]),
                "selected_peak_amplitude_over_diameter": selected_amplitude,
                "target_peak_amplitude_over_diameter": target,
                "digitization_uncertainty_over_diameter": (
                    engineering_holdout.DIGITIZATION_UNCERTAINTY_OVER_DIAMETER
                ),
                "peak_amplitude_absolute_error": peak_error,
                "peak_amplitude_tolerance": peak_tolerance,
                "peak_amplitude_pass": amplitude_pass,
                "selected_peak_frequency_over_fn": selected_frequency,
                "lock_in_frequency_pass": frequency_pass,
                "pass": timestep_pass and amplitude_pass and frequency_pass,
            }
        )

    checks = {
        "all_results_present": True,
        "exact_result_directory_set": True,
        "case_semantics_pass": case_semantics_pass,
        "runtime_provenance_execution_limits_pass": runtime_pass,
        "stationarity_pass": stationarity_pass,
        "all_timestep_pairs_pass": all(row["timestep_pass"] for row in per_scruton),
        "both_peak_amplitudes_pass": all(
            row["peak_amplitude_pass"] for row in per_scruton
        ),
        "both_selected_peaks_lock_in_frequency_pass": all(
            row["lock_in_frequency_pass"] for row in per_scruton
        ),
        "transfer_definition_and_effective_parameters_bound": all(
            entry["transfer_definition_fingerprint_sha256"]
            == manifest["sc_transfer"]["definition_fingerprint_sha256"]
            and entry["effective_parameter_record"][
                "effective_parameter_fingerprint_sha256"
            ]
            == _effective_fingerprints(manifest)[str(entry["case_id"])]
            for entry in manifest["cases"]
        ),
    }
    passed = all(checks.values()) and all(row["pass"] for row in per_scruton)
    result: dict[str, Any] = {
        "schema": ASSESSMENT_SCHEMA,
        "status": "pass" if passed else "fail",
        "evaluated": True,
        "pass": passed,
        "scientific_scope": SCOPE,
        "independent_validation": False,
        "campaign_manifest_sha256": response_benchmark.sha256_file(manifest_path),
        "campaign_manifest_fingerprint_sha256": manifest[
            "manifest_fingerprint_sha256"
        ],
        "transfer_definition_fingerprint_sha256": manifest["sc_transfer"][
            "definition_fingerprint_sha256"
        ],
        "effective_parameter_fingerprints_by_case": _effective_fingerprints(
            manifest
        ),
        "calibration_readiness_fingerprint_sha256": readiness[
            "assessment_fingerprint_sha256"
        ],
        "results_root": Path(results_root).expanduser().resolve().as_posix(),
        "expected_case_count": engineering_sc_transfer.EXPECTED_CASE_COUNT,
        "completed_case_count": engineering_sc_transfer.EXPECTED_CASE_COUNT,
        "missing_case_ids": [],
        "unexpected_nonempty_result_directories": [],
        "checks": checks,
        "by_scruton_number": per_scruton,
        "evidence_content_sha256": {
            "response_rows": engineering_sc_transfer.canonical_sha256(
                list(response_rows)
            ),
            "runtime_audits": engineering_sc_transfer.canonical_sha256(
                list(runtime_audits)
            ),
        },
        "claim_boundary": {
            "allowed": "engineering transfer-screen adequacy if every declared gate passes",
            "forbidden": [
                "blind independent validation",
                "full lock-in curve validation",
                "fatigue validation",
                "H-VAWT aerodynamic validation",
                "certification qualification",
            ],
        },
    }
    result["assessment_fingerprint_sha256"] = (
        engineering_sc_transfer.canonical_sha256(result)
    )
    return result


def analyze_results(
    manifest_path: Path,
    calibration_readiness_path: Path,
    results_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Audit all eight cases, or publish an explicit incomplete assessment."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = engineering_sc_transfer.validate_transfer_campaign(manifest_file)
    readiness, _ = _passing_readiness(calibration_readiness_path)
    results = Path(results_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Analysis output must be new or empty: {output}.")
    output.mkdir(parents=True, exist_ok=True)

    expected_ids = [str(entry["case_id"]) for entry in manifest["cases"]]
    present = _nonempty_result_directories(results)
    missing = sorted(
        case_id
        for case_id in expected_ids
        if not (results / case_id / "result.txt").is_file()
    )
    unexpected = sorted(present.difference(expected_ids))
    if missing or unexpected:
        assessment = _incomplete_assessment(
            manifest_path=manifest_file,
            manifest=manifest,
            readiness=readiness,
            results_root=results,
            missing=missing,
            unexpected=unexpected,
        )
        _write_json(output / "engineering_sc_transfer_assessment.json", assessment)
        return assessment

    file_index = _manifest_file_index(manifest)
    response_rows: list[dict[str, Any]] = []
    runtime_audits: list[dict[str, Any]] = []
    result_errors: list[dict[str, str]] = []
    for entry in manifest["cases"]:
        case_id = str(entry["case_id"])
        case_directory = results / case_id
        result_path = case_directory / "result.txt"
        runtime = response_benchmark.runtime_case_audit(
            case_directory,
            entry=entry,
            campaign_root=manifest_file.parent,
            manifest_files=file_index,
            result_path=result_path,
            expected_history_mode="compact",
            runtime_contract=manifest["runtime_contract"],
        )
        runtime_audits.append({"case_id": case_id, **runtime})
        try:
            response = response_benchmark.response_amplitude_from_result(
                result_path,
                natural_frequency_hz=float(entry["structure"]["natural_frequency_hz"]),
                response_window_cycles=40.0,
                expected_timestep_s=float(entry["timestep_s"]),
                expected_samples=int(entry["num_steps"]),
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            result_errors.append({"case_id": case_id, "error": str(exc)})
            continue
        response_rows.append(
            {
                "case_id": case_id,
                "scruton_number": entry["scruton_number"],
                "role": entry["role"],
                "reduced_velocity": entry["reduced_velocity"],
                "timestep_level": entry["timestep_level"],
                "specified_timestep_s": entry["timestep_s"],
                "runtime_audit_evaluated": bool(runtime.get("evaluated")),
                "runtime_audit_pass": bool(runtime.get("pass")),
                **response,
            }
        )

    response_benchmark._write_csv(output / "response_by_case.csv", response_rows)
    (output / "runtime_case_audits.json").write_text(
        json.dumps(runtime_audits, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if result_errors:
        assessment = _incomplete_assessment(
            manifest_path=manifest_file,
            manifest=manifest,
            readiness=readiness,
            results_root=results,
            missing=[row["case_id"] for row in result_errors],
            unexpected=[],
        )
        assessment["result_errors"] = result_errors
        assessment.pop("assessment_fingerprint_sha256", None)
        assessment["assessment_fingerprint_sha256"] = (
            engineering_sc_transfer.canonical_sha256(assessment)
        )
    else:
        assessment = _assess_complete_results(
            manifest,
            readiness,
            response_rows,
            runtime_audits,
            manifest_path=manifest_file,
            results_root=results,
        )
    _write_json(output / "engineering_sc_transfer_assessment.json", assessment)
    return assessment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Write a canonical local/HPC run plan.")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--calibration-readiness", type=Path, required=True)
    plan.add_argument("--results", type=Path, required=True)
    plan.add_argument("--qblade-root", type=Path, required=True)
    plan.add_argument("--checkpoint-every", type=int, default=0)
    plan.add_argument("--threads", type=int, default=1)
    plan.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run one zero-based plan index.")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--calibration-readiness", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--case-index", type=int, required=True)
    run.add_argument("--acknowledge-transfer-tuning-screen", action="store_true")

    analyze = subparsers.add_parser("analyze", help="Audit and assess all eight results.")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--calibration-readiness", type=Path, required=True)
    analyze.add_argument("--results", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        plan = build_execution_plan(
            args.manifest,
            args.calibration_readiness,
            args.results,
            qblade_root=args.qblade_root,
            checkpoint_every=args.checkpoint_every,
            threads=args.threads,
        )
        _write_json(args.output, plan)
        print(json.dumps({"plan": str(args.output.resolve()), "case_count": 8}))
    elif args.command == "run":
        case_id = execute_case(
            _json_file(args.plan),
            args.manifest,
            args.calibration_readiness,
            case_index=args.case_index,
            acknowledge_transfer_tuning_screen=args.acknowledge_transfer_tuning_screen,
        )
        print(json.dumps({"completed_case_id": case_id}))
    elif args.command == "analyze":
        assessment = analyze_results(
            args.manifest,
            args.calibration_readiness,
            args.results,
            args.output,
        )
        print(json.dumps(assessment, indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
