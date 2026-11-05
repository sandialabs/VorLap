import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "vlastos2026_cantilever"
    / "response_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("vlastos2026_response_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


@pytest.fixture(scope="module")
def ambient_v2_full_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("ambient_v2_full") / "campaign"
    return benchmark.generate_campaign(root, response_schema_version=2)


def _copy_campaign(manifest_path, destination):
    shutil.copytree(manifest_path.parent, destination)
    return destination / "campaign_manifest.json"


def _write_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime_identity_for_summary(path):
    return benchmark._actual_file_identity(path)


def _force_vector_digest(values) -> str:
    numeric = [float(value) for value in values]
    payload = b"vorlap-force-vector-sha256-v1\0"
    payload += struct.pack("!Q", len(numeric))
    payload += struct.pack(f"!{len(numeric)}d", *numeric)
    return hashlib.sha256(payload).hexdigest()


def test_structural_cases_report_published_and_qblade_implied_scruton_numbers():
    cases = benchmark.load_structural_cases()

    assert sorted(cases) == [2.94, 5.18, 7.02]
    assert cases[5.18].natural_frequency_hz == 15.38
    assert cases[5.18].equivalent_mass_kg_per_m == 0.4311
    assert cases[5.18].damping_ratio == 0.0028
    assert benchmark.derived_scruton_number(cases[2.94]) == pytest.approx(2.8448211418563956)
    assert benchmark.derived_scruton_number(cases[5.18]) == pytest.approx(4.953017025691645)
    assert benchmark.derived_scruton_number(cases[7.02]) == pytest.approx(6.677266689645888)


def test_calibration_api_can_operate_with_only_the_sc5p18_parameter_row(tmp_path):
    source = benchmark._reference_path("vlastos2026_model_parameters.csv")
    destination = tmp_path / source.name
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(next(row for row in rows if float(row["scruton_number"]) == 5.18))

    coefficients = benchmark.calibrate_response_coefficients(
        figure9_path=benchmark._reference_path("sc5p18_figure9_digitized.csv"),
        model_parameters_path=destination,
    )

    assert coefficients["calibration_scruton_number"] == 5.18
    assert coefficients["held_out_targets_accessed_during_calibration"] is False
    assert coefficients["held_out_scruton_numbers"] == [2.94, 7.02]
    assert "figure10" not in json.dumps(coefficients).lower()
    assert "experiment_lupi2021" not in json.dumps(coefficients).lower()
    assert coefficients["analytical_calibration_fit"]["rmse_tip_amplitude_over_diameter"] < 0.025


def test_postfreeze_gate_records_simple_model_failure_without_retuning():
    coefficients = benchmark.calibrate_response_coefficients()
    before = json.dumps(coefficients["frozen_parameters"], sort_keys=True)

    gate = benchmark.evaluate_postfreeze_analytical_gate(coefficients)

    assert gate["status"] == "fail"
    assert gate["pass"] is False
    assert gate["coefficients_changed_after_held_out_access"] is False
    assert json.dumps(coefficients["frozen_parameters"], sort_keys=True) == before
    comparisons = {row["scruton_number"]: row for row in gate["comparisons"]}
    assert comparisons[2.94]["postfreeze_prediction_peak_over_diameter"] == 0.0
    assert comparisons[2.94]["hgast_reference_peak_over_diameter"] == 0.54
    assert comparisons[2.94]["pass"] is False
    assert comparisons[7.02]["pass"] is False


def test_ambient_v2_coefficients_are_portable_across_two_source_roots(
    tmp_path,
):
    frozen_objects = []
    legacy_fingerprints = []
    for root_name in ("source_root_a", "source_root_b"):
        source_root = tmp_path / root_name
        source_root.mkdir()
        figure9 = source_root / "sc5p18_figure9_digitized.csv"
        parameters = source_root / "vlastos2026_model_parameters.csv"
        shutil.copy2(benchmark._reference_path(figure9.name), figure9)
        shutil.copy2(benchmark._reference_path(parameters.name), parameters)
        coefficients = benchmark.calibrate_response_coefficients(
            figure9_path=figure9,
            model_parameters_path=parameters,
        )
        legacy_before = json.dumps(
            coefficients,
            sort_keys=True,
            separators=(",", ":"),
        )
        legacy_fingerprints.append(
            coefficients["coefficient_fingerprint_sha256"]
        )
        frozen_objects.append(
            benchmark.portable_ambient_v2_coefficients(coefficients)
        )
        assert json.dumps(
            coefficients,
            sort_keys=True,
            separators=(",", ":"),
        ) == legacy_before

    assert legacy_fingerprints[0] != legacy_fingerprints[1]
    assert frozen_objects[0] == frozen_objects[1]
    assert frozen_objects[0]["coefficient_fingerprint_sha256"] == (
        benchmark.AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
    )
    assert frozen_objects[0]["figure9_path"] == (
        "reference_data/sc5p18_figure9_digitized.csv"
    )
    assert frozen_objects[0]["figure9_bytes"] == (
        benchmark._reference_path("sc5p18_figure9_digitized.csv").stat().st_size
    )
    assert frozen_objects[0]["model_parameters_path"] == (
        "reference_data/vlastos2026_model_parameters.csv"
    )
    assert frozen_objects[0]["model_parameters_bytes"] == (
        benchmark._reference_path("vlastos2026_model_parameters.csv").stat().st_size
    )
    assert frozen_objects[0]["frozen_parameters"] == frozen_objects[1][
        "frozen_parameters"
    ]

    tampered = copy.deepcopy(frozen_objects[0])
    tampered["figure9_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        benchmark.portable_ambient_v2_coefficients(tampered)


def test_campaign_spec_has_all_three_cases_and_at_least_50_steps_per_cycle():
    entries = benchmark.build_campaign_spec(
        structural_cases=benchmark.load_structural_cases(),
        reduced_velocities=(6.25,),
        timestep_s=5.0e-4,
        include_dt_halving=True,
        duration_s=8.0,
    )

    assert len(entries) == 6
    assert {entry["scruton_number"] for entry in entries} == {2.94, 5.18, 7.02}
    assert {entry["timestep_level"] for entry in entries} == {"dt", "dt_half"}
    assert min(entry["steps_per_natural_cycle"] for entry in entries) >= 50.0
    assert all(entry["scruton_number"] != entry["qblade_implied_scruton_number"] for entry in entries)


def test_campaign_spec_defaults_to_frozen_case_specific_stationarity_caps():
    entries = benchmark.build_campaign_spec(
        structural_cases=benchmark.load_structural_cases(),
        reduced_velocities=benchmark.DEFAULT_REDUCED_VELOCITIES,
        timestep_s=benchmark.DEFAULT_TIMESTEP_S,
        include_dt_halving=True,
        duration_s=None,
    )

    assert len(entries) == 72
    by_key = {
        (
            float(entry["scruton_number"]),
            float(entry["reduced_velocity"]),
            str(entry["timestep_level"]),
        ): entry
        for entry in entries
    }
    peak_coarse = by_key[(5.18, 6.25, "dt")]
    peak_fine = by_key[(5.18, 6.25, "dt_half")]
    slow_onset = by_key[(5.18, 5.80, "dt")]
    assert peak_coarse["stationary_maximum_cycles"] == 320
    assert peak_coarse["num_steps"] == 41_613
    assert peak_fine["num_steps"] == 83_226
    assert peak_coarse["duration_s"] == peak_fine["duration_s"] == 20.8065
    assert slow_onset["stationary_maximum_cycles"] == 2520
    assert slow_onset["duration_s"] == 163.8495
    assert {entry["duration_policy"] for entry in entries} == {
        "frozen_stationary_cycle_cap"
    }

    with pytest.raises(ValueError, match="Stationary-duration generation requires"):
        benchmark.build_campaign_spec(
            structural_cases=benchmark.load_structural_cases(),
            reduced_velocities=(6.25,),
            timestep_s=0.0004,
            include_dt_halving=True,
            duration_s=None,
        )


def test_response_campaign_uses_finite_seed_without_persistent_source(tmp_path):
    manifest_path = benchmark.generate_campaign(
        tmp_path / "campaign",
        reduced_velocities=(6.25,),
        timestep_s=0.0005,
        include_dt_halving=False,
        duration_s=0.05,
        selected_scruton_numbers=(5.18,),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["cases"][0]
    config_path = manifest_path.parent / entry["parameter_file"]
    runtime = json.loads(config_path.read_text(encoding="utf-8"))

    assert runtime["force_scale"] == 0.0
    assert runtime["response_model"]["startup_seed"] == {
        "schema_version": 1,
        "enabled": True,
        "duration_cycles": 5.0,
        "target_amplitude_ratio": 0.005,
        "structural_damping_ratio": 0.0028,
        "taper": "hann_sine_squared",
        "carrier": "natural_frequency",
        "nodal_weighting": "projection_mass_times_mode",
        "nodal_force_limit_source": "response_model_max_abs_nodal_force_n",
    }
    assert manifest["qblade_mapping"]["persistent_fixed_wake_source"] == (
        "disabled with force_scale=0"
    )
    assert manifest["qblade_mapping"]["startup_seed"][
        "exactly_zero_after_seed"
    ] is True
    assert manifest["numerics"]["duration_policy"] == (
        "explicit_fixed_diagnostic_duration"
    )
    assert manifest["numerics"]["qualification_input_eligible"] is False


@pytest.mark.requires_qblade
def test_ambient_v2_generation_uses_disjoint_case_and_checkpoint_namespace(
    tmp_path,
):
    manifest_path = benchmark.generate_campaign(
        tmp_path / "ambient_v2_campaign",
        reduced_velocities=(6.25,),
        timestep_s=0.0005,
        include_dt_halving=True,
        duration_s=0.05,
        selected_scruton_numbers=(2.94, 5.18, 7.02),
        response_schema_version=2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _root, loaded = benchmark._load_campaign(manifest_path)

    assert loaded["response_model_semantics"] == (
        benchmark.stationary_protocol.ambient_v2_response_semantics()
    )
    assert len(manifest["cases"]) == 6
    assert sum(row["role"] == "calibration" for row in manifest["cases"]) == 2
    assert sum(row["role"] == "held_out" for row in manifest["cases"]) == 4
    assert all(
        row["case_id"].startswith(
            f"{benchmark.AMBIENT_V2_CASE_ID_NAMESPACE}_"
        )
        for row in manifest["cases"]
    )
    assert {
        row["response_model_config"]["schema_version"]
        for row in manifest["cases"]
    } == {2}
    assert {
        row["response_model_config"]["aerodynamic_speed_basis"]
        for row in manifest["cases"]
    } == {benchmark.AMBIENT_V2_AERODYNAMIC_SPEED_BASIS}
    assert all(
        "response_section_width_m" not in row["response_model_config"]
        for row in manifest["cases"]
    )
    assert all(
        row["mean_inflow_mps"]
        == pytest.approx(
            row["reduced_velocity"]
            * row["structure"]["natural_frequency_hz"]
            * row["response_model_config"]["reference_length_m"]
        )
        for row in manifest["cases"]
    )

    legacy_entries = benchmark.build_campaign_spec(
        structural_cases=benchmark.load_structural_cases(),
        reduced_velocities=(6.25,),
        timestep_s=0.0005,
        include_dt_halving=True,
        duration_s=0.05,
    )
    assert {row["case_id"] for row in legacy_entries}.isdisjoint(
        {row["case_id"] for row in manifest["cases"]}
    )

    with pytest.raises(ValueError, match="full 72-case"):
        benchmark.build_run_plan(
            manifest_path,
            tmp_path / "shared_results_root",
            stage="all",
            qblade_root=(
                benchmark.REPO_ROOT
                / "run_headless"
                / "QBladeCE_2.0.9.7"
            ),
            checkpoint_every=100,
            threads=4,
            diagnostic_only=True,
        )


@pytest.mark.requires_qblade
def test_ambient_v2_full_tree_is_portable_across_copied_reference_roots(
    tmp_path,
    monkeypatch,
):
    source_roots = []
    for label in ("source_a", "source_b"):
        source_root = tmp_path / label
        source_root.mkdir()
        for filename in benchmark.stationary_protocol.AMBIENT_V2_REFERENCE_IDENTITIES:
            shutil.copyfile(benchmark._reference_path(filename), source_root / filename)
        source_roots.append(source_root)

    trees = []
    for index, source_root in enumerate(source_roots):
        monkeypatch.setattr(
            benchmark,
            "_reference_path",
            lambda name, root=source_root: root / name,
        )
        manifest_path = benchmark.generate_campaign(
            tmp_path / f"portable_campaign_{index}",
            response_schema_version=2,
        )
        trees.append(
            {
                path.relative_to(manifest_path.parent).as_posix(): path.read_bytes()
                for path in sorted(manifest_path.parent.rglob("*"))
                if path.is_file()
            }
        )

    assert trees[0] == trees[1]
    manifest_text = trees[0]["campaign_manifest.json"].decode("utf-8")
    assert source_roots[0].as_posix() not in manifest_text
    assert source_roots[1].as_posix() not in manifest_text
    assert (
        json.loads(manifest_text)["calibration_policy"][
            "coefficient_fingerprint_sha256"
        ]
        == benchmark.AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
    )


@pytest.mark.requires_qblade
def test_ambient_v2_plan_is_canonical_and_rejects_api_mutation(
    ambient_v2_full_campaign,
    tmp_path,
    monkeypatch,
):
    qblade_root = (
        benchmark.REPO_ROOT / "run_headless" / "QBladeCE_2.0.9.7"
    )
    plan = benchmark.build_run_plan(
        ambient_v2_full_campaign,
        tmp_path / "results",
        stage="calibration",
        qblade_root=qblade_root,
        checkpoint_every=100,
        threads=4,
    )
    assert plan["schema"] == benchmark.AMBIENT_V2_RUN_PLAN_SCHEMA
    assert len(plan["commands"]) == 24
    assert plan["stationary_protocol_sha256"]
    assert plan["runtime_identities"]["qblade_sil_wrapper"] == (
        benchmark.stationary_protocol.AMBIENT_V2_QBLADE_SIL_WRAPPER_IDENTITY
    )
    assert len(plan["runtime_identities"]["qblade_runtime_dependencies"]) >= 2

    mutations = []
    changed = copy.deepcopy(plan)
    changed["commands"][0]["argv"][1] = "/usr/bin/touch"
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["coefficient_fingerprint_sha256"] = "0" * 64
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["stage"] = "all"
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["results_root"] = str((tmp_path / "redirected").resolve())
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["threads"] = 3
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["checkpoint_every"] = 0
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["stationary_protocol_sha256"] = "0" * 64
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["runtime_identities"]["result_filter"]["sha256"] = "0" * 64
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["commands"].append(copy.deepcopy(changed["commands"][0]))
    mutations.append(changed)
    changed = copy.deepcopy(plan)
    changed["commands"][0]["case_id"] = "unknown_case"
    mutations.append(changed)

    monkeypatch.setattr(benchmark, "_preflight_response_runtime", lambda _path: None)
    for changed in mutations:
        with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
            benchmark.execute_run_plan(changed, ambient_v2_full_campaign)

    first_case = Path(plan["results_root"]) / plan["commands"][0]["case_id"]
    first_case.mkdir(parents=True)
    (first_case / "residue.txt").write_text("not a fresh run\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty"):
        benchmark.execute_run_plan(plan, ambient_v2_full_campaign)


@pytest.mark.requires_qblade
def test_ambient_v2_results_containment_and_checkpoint_types_fail_closed(
    ambient_v2_full_campaign,
    tmp_path,
):
    qblade_root = (
        benchmark.REPO_ROOT / "run_headless" / "QBladeCE_2.0.9.7"
    )
    campaign_root = ambient_v2_full_campaign.parent
    forbidden_results = (
        Path("/"),
        benchmark.REPO_ROOT.parent,
        campaign_root.parent,
        benchmark.REPO_ROOT,
        Path.home(),
    )
    for results_root in forbidden_results:
        with pytest.raises(ValueError, match="results_root"):
            benchmark.build_run_plan(
                ambient_v2_full_campaign,
                results_root,
                stage="calibration",
                qblade_root=qblade_root,
            )
    for invalid in (True, -1, 1.5):
        with pytest.raises(ValueError, match="checkpoint_every"):
            benchmark.build_run_plan(
                ambient_v2_full_campaign,
                tmp_path / f"results_{str(invalid).replace('.', '_')}",
                stage="calibration",
                qblade_root=qblade_root,
                checkpoint_every=invalid,
            )


@pytest.mark.requires_qblade
def test_ambient_v2_dispatch_rechecks_output_and_rejects_native_activity(
    ambient_v2_full_campaign,
    tmp_path,
    monkeypatch,
):
    qblade_root = (
        benchmark.REPO_ROOT / "run_headless" / "QBladeCE_2.0.9.7"
    )
    plan = benchmark.build_run_plan(
        ambient_v2_full_campaign,
        tmp_path / "dispatch_results",
        stage="calibration",
        qblade_root=qblade_root,
        threads=4,
    )
    monkeypatch.setattr(benchmark, "_preflight_response_runtime", lambda _path: None)
    calls = []

    def fake_run(argv, check):
        assert check is True
        calls.append(list(argv))
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "case_run_summary.json").write_text(
            json.dumps(
                {
                    "native_callback_evidence": {
                        "log_file_exists": False,
                        "update_init_count": 0,
                        "update_count": 0,
                        "update_count_observed_max": 0,
                    }
                }
            ),
            encoding="utf-8",
        )
        next_output = (
            Path(plan["results_root"]) / plan["commands"][1]["case_id"]
        )
        next_output.mkdir(parents=True, exist_ok=True)
        (next_output / "residue.txt").write_text("TOCTOU residue\n", encoding="utf-8")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    with pytest.raises(FileExistsError, match="nonempty"):
        benchmark.execute_run_plan(plan, ambient_v2_full_campaign)
    assert len(calls) == 1

    campaign_root, manifest = benchmark._load_campaign(ambient_v2_full_campaign)
    entry = manifest["cases"][0]
    log_path = (
        campaign_root / entry["parameter_file"]
    ).parent / "vorlap_qblade_bridge_cpp.log"
    log_path.write_text("update entered; count=1\n", encoding="utf-8")
    try:
        with pytest.raises(
            benchmark.stationary_protocol.ProtocolError,
            match="campaign tree|native bridge",
        ):
            benchmark._load_campaign(ambient_v2_full_campaign)
    finally:
        log_path.unlink()


@pytest.mark.requires_qblade
def test_ambient_v2_runtime_contract_and_response_fingerprint_are_exact(
    ambient_v2_full_campaign,
    tmp_path,
):
    campaign_root, manifest = benchmark._load_campaign(ambient_v2_full_campaign)
    contract = manifest["runtime_contract"]
    runner = benchmark.REPO_ROOT / contract["runner"]["path"]
    result_filter = campaign_root / contract["result_filter"]["path"]

    def reported(identity, prefix):
        return {
            "path": f"{prefix}/{identity['path']}",
            "sha256": identity["sha256"],
            "size_bytes": identity["bytes"],
        }

    reproducibility = {
        "command_line": [sys.executable, str(runner)],
        "runner": _runtime_identity_for_summary(runner),
        "input_files": {
            "result_filter": _runtime_identity_for_summary(result_filter),
        },
        "qblade": {
            "library": reported(contract["qblade_library"], "/runtime"),
        },
        "bridge_library": reported(contract["bridge_library"], "/runtime"),
        "vorlap_python_runtime_assets": contract[
            "vorlap_python_runtime_assets"
        ],
        "vorlap_python_runtime_fingerprint": contract[
            "vorlap_python_runtime_fingerprint"
        ],
        "qblade_runtime_dependencies": contract[
            "qblade_runtime_dependencies"
        ],
        "qblade_runtime_dependency_fingerprint": contract[
            "qblade_runtime_dependency_fingerprint"
        ],
        "python": {
            "implementation": contract["numerical_runtime"][
                "python_implementation"
            ],
            "version": contract["numerical_runtime"]["python_version"],
        },
        "packages": contract["numerical_runtime"]["packages"],
    }
    missing = []
    mismatches = []
    benchmark._audit_ambient_v2_runtime_contract(
        reproducibility,
        campaign_root=campaign_root,
        runtime_contract=contract,
        missing=missing,
        mismatches=mismatches,
    )
    assert missing == []
    assert mismatches == []

    drifted = copy.deepcopy(reproducibility)
    drifted["qblade_runtime_dependencies"][0]["sha256"] = "0" * 64
    mismatches = []
    benchmark._audit_ambient_v2_runtime_contract(
        drifted,
        campaign_root=campaign_root,
        runtime_contract=contract,
        missing=[],
        mismatches=mismatches,
    )
    assert any("qblade_runtime_dependencies" in value for value in mismatches)

    response_config = manifest["cases"][0]["response_model_config"]
    expected = benchmark._response_config_fingerprint(response_config)
    changed = ("0" if expected[0] != "0" else "1") + expected[1:]
    with pytest.raises(ValueError, match="fingerprint"):
        benchmark._validated_response_config_fingerprint(
            response_config, changed
        )

    result_directory = tmp_path / "checkpoint_case"
    result_directory.mkdir()
    checkpoint_path = result_directory / "step.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "sil_vorlap_state": {
                    "runtime": {
                        "controller": {
                            "response_model_state": {
                                "config_fingerprint": expected
                            },
                            "response_diagnostics_summary": {
                                "response_config_fingerprint_sha256": changed
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint_mismatches = []
    benchmark._audit_ambient_v2_checkpoint_fingerprint(
        {"restart": {"latest_checkpoint": str(checkpoint_path)}},
        result_directory=result_directory,
        expected_fingerprint=expected,
        runtime_contract=None,
        missing=[],
        mismatches=checkpoint_mismatches,
    )
    assert checkpoint_mismatches == [
        "checkpoint response-diagnostics fingerprint differs"
    ]


def test_ambient_v2_equation_gate_is_exact_and_fail_closed():
    coefficients = benchmark.portable_ambient_v2_coefficients(
        benchmark.calibrate_response_coefficients()
    )
    gate = benchmark.ambient_v2_equation_reproduction_spec(
        benchmark.load_structural_cases()[benchmark.CALIBRATION_SC],
        coefficients,
    )
    result_rows = []
    for table_row in gate["analytical_table"]:
        for timestep_level in ("dt", "dt_half"):
            result_rows.append(
                {
                    "scruton_number": benchmark.CALIBRATION_SC,
                    "reduced_velocity": table_row["reduced_velocity"],
                    "timestep_level": timestep_level,
                    "response_estimator_final_mean_amplitude_ratio": table_row[
                        "analytical_equilibrium_amplitude_over_diameter"
                    ],
                }
            )

    rows, summary = benchmark.ambient_v2_equation_reproduction_gate(
        result_rows,
        gate,
    )

    assert len(rows) == 12
    assert summary["evaluated"] is True
    assert summary["pass"] is True
    assert summary["fine_estimator_rmse_over_diameter"] == pytest.approx(0.0)
    assert summary["fine_estimator_maximum_absolute_error_over_diameter"] == (
        pytest.approx(0.0)
    )
    assert summary["observed_active_reduced_velocities"] == [
        5.8,
        6.2,
        6.25,
        6.5,
    ]

    changed_timestep = copy.deepcopy(result_rows)
    changed_timestep[2 * 7][
        "response_estimator_final_mean_amplitude_ratio"
    ] += 0.006
    _rows, failed = benchmark.ambient_v2_equation_reproduction_gate(
        changed_timestep,
        gate,
    )
    assert failed["evaluated"] is True
    assert failed["pass"] is False
    assert (
        failed[
            "coarse_fine_estimator_maximum_absolute_difference_over_diameter"
        ]
        > gate[
            "coarse_fine_estimator_maximum_absolute_difference_tolerance_over_diameter"
        ]
    )

    relaxed_gate = copy.deepcopy(gate)
    relaxed_gate["fine_estimator_rmse_tolerance_over_diameter"] = 1.0
    with pytest.raises(
        benchmark.stationary_protocol.ProtocolError,
        match="frozen",
    ):
        benchmark.ambient_v2_equation_reproduction_gate(
            result_rows,
            relaxed_gate,
        )


def test_calibration_curve_comparison_covers_every_hgast_point():
    u_star, amplitude = benchmark._read_figure9_series(
        benchmark._reference_path("sc5p18_figure9_digitized.csv"),
        benchmark.CALIBRATION_SERIES,
    )
    curve = [
        {
            "scruton_number": 5.18,
            "reduced_velocity": float(u_value),
            "tip_peak_amplitude_over_diameter": float(amplitude_value),
        }
        for u_value, amplitude_value in zip(u_star, amplitude)
    ]

    rows, summary = benchmark.calibration_curve_comparison(
        curve,
        benchmark._reference_path("sc5p18_figure9_digitized.csv"),
    )

    assert summary["evaluated"] is True
    assert summary["pass"] is True
    assert summary["rmse_tip_amplitude_over_diameter"] == pytest.approx(0.0)
    assert len(rows) == len(u_star)


def _write_synthetic_result(
    path: Path,
    *,
    frequency_hz: float,
    timestep_s: float,
    amplitude_ratio: float,
    final_amplitude_ratio: float | None = None,
    duration_cycles: float = 45.0,
    sample_count: int | None = None,
    time_format: str = "%.18e",
    accumulate_export_time: bool = False,
) -> int:
    if sample_count is None:
        duration_s = duration_cycles / frequency_hz
        time = np.arange(0.0, duration_s + 0.5 * timestep_s, timestep_s)
    else:
        time = np.arange(sample_count, dtype=float) * timestep_s
    amplitude = np.linspace(
        amplitude_ratio,
        amplitude_ratio if final_amplitude_ratio is None else final_amplitude_ratio,
        time.size,
    )
    displacement = (
        amplitude
        * benchmark.DIAMETER_M
        * np.sin(2.0 * math.pi * frequency_hz * time)
    )
    export_time = time
    if accumulate_export_time:
        export_time = np.empty_like(time)
        export_time[0] = 0.0
        np.cumsum(
            np.full(time.size - 1, timestep_s, dtype=float),
            out=export_time[1:],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack((export_time, displacement)),
        delimiter="\t",
        header=f"Time [s]\t{benchmark.RESULT_CHANNEL}",
        comments="",
        fmt=(time_format, "%.18e"),
    )
    return int(time.size)


def _write_response_history(
    path: Path,
    *,
    natural_frequency_hz: float = 5.0,
    amplitude_ratio_start: float = 0.20,
    amplitude_ratio_end: float = 0.20,
    raw_gain: float = 5.0,
    nodal_force_n: float = 10.0,
    gain_limited_node_samples: int = 0,
) -> int:
    timestep_s = 0.01
    duration_s = 45.0 / natural_frequency_hz
    time = np.arange(0.0, duration_s + 0.5 * timestep_s, timestep_s)
    amplitude_ratio = np.linspace(
        amplitude_ratio_start,
        amplitude_ratio_end,
        time.size,
    )
    rows = [
        {
            "force_history_schema": "vorlap-sil-force-history-v2",
            "step_index": index,
            "time_s": float(time_value),
            "response_model_type": "modal_negative_damping",
            "response_enabled": True,
            "response_amplitude_ratio": float(amplitude_value),
            "response_raw_gain_min": -raw_gain,
            "response_raw_gain_max": raw_gain,
            "response_gain_limited_nodes": (
                gain_limited_node_samples if index == len(time) - 1 else 0
            ),
            "response_force_limited_nodes": 0,
            "response_max_abs_nodal_force_n": nodal_force_n,
        }
        for index, (time_value, amplitude_value) in enumerate(
            zip(time, amplitude_ratio)
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_compact_seeded_response_history(
    path: Path,
    *,
    natural_frequency_hz: float = 5.0,
    amplitude_ratio: float = 0.005,
    component_count: int = 63,
    response_schema_version: int = 1,
    ambient_response_speed_mps: float = 1.5625,
    configured_reduced_velocity: float = 6.25,
) -> int:
    timestep_s = 0.01
    duration_s = 45.0 / natural_frequency_hz
    seed_duration_cycles = 5.0
    seed_duration_s = seed_duration_cycles / natural_frequency_hz
    target_ratio = 0.005
    target_amplitude_m = target_ratio * benchmark.DIAMETER_M
    acceleration_scale = 0.3
    time = np.arange(0.0, duration_s + 0.5 * timestep_s, timestep_s)
    zero_digest = _force_vector_digest([0.0] * component_count)
    generated_digests = [
        hashlib.sha256(f"generated-{index}".encode("ascii")).hexdigest()
        for index in range(time.size)
    ]
    velocity_digests = [
        hashlib.sha256(f"velocity-{index}".encode("ascii")).hexdigest()
        for index in range(time.size)
    ]
    rows = []
    for index, time_value in enumerate(time):
        seed_active = time_value < seed_duration_s
        elapsed_cycles = min(time_value * natural_frequency_hz, seed_duration_cycles)
        taper = (
            math.sin(math.pi * time_value / seed_duration_s) ** 2
            if seed_active
            else 0.0
        )
        acceleration = (
            acceleration_scale
            * taper
            * math.sin(2.0 * math.pi * natural_frequency_hz * time_value)
            if seed_active
            else 0.0
        )
        row = {
                "force_history_schema": "vorlap-sil-force-history-compact-v1",
                "force_vector_digest_schema": "vorlap-force-vector-sha256-v1",
                "force_vector_component_count": component_count,
                "step_index": index,
                "time_s": float(time_value),
                "generated_step_index": index,
                "generated_time_s": float(time_value),
                "applied_source_step_index": "" if index == 0 else index - 1,
                "applied_source_time_s": "" if index == 0 else float(time[index - 1]),
                "applied_startup_zero": index == 0,
                "applied_force_lag_steps": 1,
                "generated_force_vector_sha256": generated_digests[index],
                "applied_force_vector_sha256": (
                    zero_digest if index == 0 else generated_digests[index - 1]
                ),
                "structural_velocity_vector_sha256": velocity_digests[index],
                "applied_force_dot_structural_velocity_w": 0.0,
                "applied_power_sign_convention": (
                    "positive_when_applied_force_does_work_on_current_structural_velocity"
                ),
                "response_model_type": "modal_negative_damping",
                "response_enabled": True,
                "response_amplitude_ratio": amplitude_ratio,
                "response_raw_gain_min": -5.0,
                "response_raw_gain_max": 5.0,
                "response_gain_limited_nodes": 0,
                "response_force_limited_nodes": 0,
                "response_max_abs_nodal_force_n": 0.01,
                "response_startup_seed_enabled": True,
                "response_startup_seed_active": seed_active,
                "response_startup_seed_elapsed_cycles": elapsed_cycles,
                "response_startup_seed_duration_cycles": seed_duration_cycles,
                "response_startup_seed_duration_s": seed_duration_s,
                "response_startup_seed_taper": taper,
                "response_startup_seed_target_amplitude_ratio": target_ratio,
                "response_startup_seed_target_amplitude_m": target_amplitude_m,
                "response_startup_seed_ideal_terminal_amplitude_m": target_amplitude_m,
                "response_startup_seed_modal_acceleration_scale_mps2": acceleration_scale,
                "response_startup_seed_modal_acceleration_mps2": acceleration,
                "response_startup_seed_raw_generalized_force_n": acceleration,
                "response_startup_seed_raw_max_abs_nodal_force_n": abs(acceleration) * 0.01,
                "response_startup_seed_force_limited_nodes": 0,
                "response_startup_seed_modal_power_w": acceleration * 0.01,
            }
        if response_schema_version == 2:
            row.update(
                {
                    "response_response_model_schema_version": 2,
                    "response_aerodynamic_speed_basis": (
                        benchmark.AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
                    ),
                    "response_response_declared_ambient_crossflow_speed_min_mps": (
                        ambient_response_speed_mps
                    ),
                    "response_response_declared_ambient_crossflow_speed_max_mps": (
                        ambient_response_speed_mps
                    ),
                    "response_response_relative_crossflow_speed_min_mps": (
                        max(0.0, ambient_response_speed_mps - 0.1)
                    ),
                    "response_response_relative_crossflow_speed_max_mps": (
                        ambient_response_speed_mps + 0.1
                    ),
                    "response_response_section_width_basis": (
                        "declared_ambient_crossflow_angle"
                    ),
                    "response_response_section_width_min_m": (
                        benchmark.DIAMETER_M
                    ),
                    "response_response_section_width_max_m": (
                        benchmark.DIAMETER_M
                    ),
                    "response_configured_response_reduced_velocity_min": (
                        configured_reduced_velocity
                    ),
                    "response_configured_response_reduced_velocity_max": (
                        configured_reduced_velocity
                    ),
                    "response_realized_response_reduced_velocity_min": (
                        configured_reduced_velocity
                    ),
                    "response_realized_response_reduced_velocity_max": (
                        configured_reduced_velocity
                    ),
                    "response_speed_basis_invariant_violations": 0,
                    "response_activation_invariant_violations": 0,
                }
            )
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _portable_file_identity(path: Path, reported_path: str) -> dict[str, object]:
    return {
        "path": reported_path,
        "sha256": benchmark.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_bound_runtime_case(
    campaign_root: Path,
    results_root: Path,
    *,
    case_id: str,
    mean_inflow_mps: float,
    amplitude_ratio: float = 0.20,
    compact_seeded: bool = False,
) -> tuple[dict[str, object], dict[str, dict[str, object]], Path, Path]:
    natural_frequency_hz = 5.0
    timestep_s = 0.01
    sim_relative = f"cases/{case_id}/{case_id}.sim"
    config_relative = f"cases/{case_id}/model/Control/vorlap_qblade_external.json"
    turbine_relative = f"cases/{case_id}/model/cantilever.trb"
    sim_path = campaign_root / sim_relative
    config_path = campaign_root / config_relative
    turbine_path = campaign_root / turbine_relative
    for path in (sim_path, config_path, turbine_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    sim_path.write_text(
        f"{mean_inflow_mps} MEANINF\n{timestep_s} TIMESTEP\nmodel/cantilever.trb TURBFILE\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps({"case_id": case_id}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    turbine_path.write_text(f"OBJECTNAME {case_id}\n", encoding="utf-8")

    result_directory = results_root / case_id
    result_path = result_directory / "result.txt"
    history_path = result_directory / "result_sil_vorlap_history.csv"
    _write_synthetic_result(
        result_path,
        frequency_hz=natural_frequency_hz,
        timestep_s=timestep_s,
        amplitude_ratio=amplitude_ratio,
    )
    if compact_seeded:
        samples = _write_compact_seeded_response_history(
            history_path,
            natural_frequency_hz=natural_frequency_hz,
            amplitude_ratio=amplitude_ratio,
        )
        history_mode = "compact"
        history_schema = "vorlap-sil-force-history-compact-v1"
    else:
        samples = _write_response_history(
            history_path,
            natural_frequency_hz=natural_frequency_hz,
            amplitude_ratio_start=amplitude_ratio,
            amplitude_ratio_end=amplitude_ratio,
        )
        history_mode = "full"
        history_schema = "vorlap-sil-force-history-v2"
    with history_path.open(newline="", encoding="utf-8") as handle:
        history_columns = len(next(csv.reader(handle)))

    manifest_files: dict[str, dict[str, object]] = {}
    for relative, path in (
        (sim_relative, sim_path),
        (config_relative, config_path),
        (turbine_relative, turbine_path),
    ):
        manifest_files[relative] = _portable_file_identity(
            path, f"/manifest-root/{relative}"
        )
        manifest_files[relative]["path"] = relative

    source_row = {field: 0 for field in benchmark._SOURCE_ZERO_FIELDS}
    source_row["samples"] = samples
    sim_identity = _portable_file_identity(sim_path, f"/container/{sim_relative}")
    config_identity = _portable_file_identity(
        config_path, f"/container/{config_relative}"
    )
    turbine_identity = _portable_file_identity(
        turbine_path, f"/container/{turbine_relative}"
    )
    result_identity = _portable_file_identity(
        result_path, f"/container/results/{case_id}/result.txt"
    )
    history_identity = _portable_file_identity(
        history_path,
        f"/container/results/{case_id}/result_sil_vorlap_history.csv",
    )
    run_summary = {
        "sim": f"/container/{sim_relative}",
        "result_file": f"/container/results/{case_id}/result.txt",
        "result_file_identity": result_identity,
        "sil_vorlap_parameter_file": f"/container/{config_relative}",
        "sil_vorlap_history_mode": history_mode,
        "sil_vorlap_history": {
            "path": f"/container/results/{case_id}/result_sil_vorlap_history.csv",
            "rows": samples,
            "columns": history_columns,
            "schema": history_schema,
            "mode": history_mode,
            "file_identity": history_identity,
        },
        "failed": False,
        "stopped": False,
        "completed_steps": samples,
        "requested_steps": samples,
        "sil_vorlap_updates": samples,
        "coupling_mode": "sil-vorlap",
        "qblade_export_time_diagnostics": {"samples": samples},
        "sil_vorlap_response_diagnostics_summary": {
            "model_type": "modal_negative_damping",
            "samples": samples,
            "gain_limited_steps": 0,
            "force_limited_steps": 0,
            "gain_limited_node_samples": 0,
            "force_limited_node_samples": 0,
        },
        "sil_vorlap_source_coverage": [source_row],
        "reproducibility": {
            "input_files": {
                "sim": sim_identity,
                "vorlap_config": config_identity,
                "turbine": turbine_identity,
            },
            "qblade_case_settings": {
                "MEANINF": str(mean_inflow_mps),
                "TIMESTEP": str(timestep_s),
                "NUMTIMESTEPS": str(samples),
                "TURBFILE": "model/cantilever.trb",
            },
        },
    }
    if compact_seeded:
        run_summary["sil_vorlap_history"]["vector_digest_schema"] = (
            "vorlap-force-vector-sha256-v1"
        )
        run_summary["sil_vorlap_history"].update(
            {
                "streaming": True,
                "retained_rows": 0,
                "node_count": 21,
                "discarded_uncheckpointed_tail_bytes": 0,
            }
        )
        run_summary["sil_vorlap_response_diagnostics_summary"].update(
            {
                "startup_seed_samples": samples,
                "startup_seed_active_steps": 100,
                "startup_seed_force_limited_steps": 0,
                "startup_seed_force_limited_node_samples": 0,
                "max_startup_seed_force_limited_nodes": 0,
                "max_startup_seed_raw_nodal_force_n": 0.003,
                "startup_seed_target_amplitude_ratio": 0.005,
                "startup_seed_target_amplitude_m": 0.00025,
            }
        )
    (result_directory / "result_run_summary.json").write_text(
        json.dumps(run_summary), encoding="utf-8"
    )
    entry: dict[str, object] = {
        "case_id": case_id,
        "sim_path": sim_relative,
        "parameter_file": config_relative,
        "mean_inflow_mps": mean_inflow_mps,
        "timestep_s": timestep_s,
        "num_steps": samples,
        "structure": {"natural_frequency_hz": natural_frequency_hz},
        "response_model_config": {
            "max_abs_gain": 10.0,
            "max_abs_nodal_force_n": 25.0,
            **(
                {
                    "node_ids": [f"BLD_1_{index / 20:.6f}" for index in range(21)],
                    "startup_seed": {
                        "enabled": True,
                        "duration_cycles": 5.0,
                        "target_amplitude_ratio": 0.005,
                    },
                }
                if compact_seeded
                else {}
            ),
        },
    }
    return entry, manifest_files, result_directory, result_path


def test_enhanced_qualification_thresholds_are_predeclared():
    assert {
        "stationarity_window_cycles": benchmark.STATIONARITY_WINDOW_CYCLES,
        "stationarity_absolute_amplitude_ratio": benchmark.STATIONARITY_ABSOLUTE_AMPLITUDE_RATIO,
        "stationarity_relative_amplitude": benchmark.STATIONARITY_RELATIVE_AMPLITUDE,
        "stationarity_frequency_ratio": benchmark.STATIONARITY_FREQUENCY_RATIO,
        "active_amplitude_ratio": benchmark.ACTIVE_AMPLITUDE_RATIO,
        "lock_in_frequency_tolerance_ratio": benchmark.LOCK_IN_FREQUENCY_TOLERANCE_RATIO,
        "lock_in_minimum_mode_band_power_fraction": benchmark.LOCK_IN_MINIMUM_MODE_BAND_POWER_FRACTION,
        "lock_in_minimum_consecutive_points": benchmark.LOCK_IN_MINIMUM_CONSECUTIVE_POINTS,
        "lock_in_reference_peak_neighborhood_u_star": benchmark.LOCK_IN_REFERENCE_PEAK_NEIGHBORHOOD_USTAR,
        "lock_in_endpoint_tolerance_u_star": benchmark.LOCK_IN_ENDPOINT_TOLERANCE_USTAR,
        "timestep_frequency_tolerance_ratio": benchmark.TIMESTEP_FREQUENCY_TOLERANCE_RATIO,
        "timestep_peak_location_tolerance_u_star": benchmark.TIMESTEP_PEAK_LOCATION_TOLERANCE_USTAR,
        "timestep_band_endpoint_tolerance_u_star": benchmark.TIMESTEP_BAND_ENDPOINT_TOLERANCE_USTAR,
        "limit_headroom_fraction": benchmark.LIMIT_HEADROOM_FRACTION,
        "live_analytical_rmse_tolerance": benchmark.LIVE_ANALYTICAL_RMSE_TOLERANCE,
        "live_analytical_max_abs_tolerance": benchmark.LIVE_ANALYTICAL_MAX_ABS_TOLERANCE,
        "live_analytical_peak_location_tolerance_u_star": benchmark.LIVE_ANALYTICAL_PEAK_LOCATION_TOLERANCE_USTAR,
        "coordinate_absolute_amplitude_ratio_tolerance": benchmark.COORDINATE_ABSOLUTE_AMPLITUDE_RATIO_TOLERANCE,
        "coordinate_relative_amplitude_tolerance": benchmark.COORDINATE_RELATIVE_AMPLITUDE_TOLERANCE,
        "material_improvement_minimum_skill": benchmark.MATERIAL_IMPROVEMENT_MINIMUM_SKILL,
        "fatigue_proxy_slope": benchmark.FATIGUE_PROXY_SLOPE,
        "startup_seed_observation_cycles": benchmark.STARTUP_SEED_OBSERVATION_CYCLES,
        "startup_seed_realized_minimum_amplitude_ratio": benchmark.STARTUP_SEED_REALIZED_MINIMUM_AMPLITUDE_RATIO,
        "startup_seed_realized_maximum_amplitude_ratio": benchmark.STARTUP_SEED_REALIZED_MAXIMUM_AMPLITUDE_RATIO,
        "lupi_experiment_curve_rmse_tolerance": benchmark.LUPI_EXPERIMENT_CURVE_RMSE_TOLERANCE,
        "lupi_experiment_peak_location_tolerance_u_star": benchmark.LUPI_EXPERIMENT_PEAK_LOCATION_TOLERANCE_USTAR,
        "lupi_experiment_peak_absolute_tolerance": benchmark.LUPI_EXPERIMENT_PEAK_ABSOLUTE_TOLERANCE,
        "lupi_experiment_peak_relative_tolerance": benchmark.LUPI_EXPERIMENT_PEAK_RELATIVE_TOLERANCE,
    } == {
        "stationarity_window_cycles": 20.0,
        "stationarity_absolute_amplitude_ratio": 0.005,
        "stationarity_relative_amplitude": 0.05,
        "stationarity_frequency_ratio": 0.01,
        "active_amplitude_ratio": 0.05,
        "lock_in_frequency_tolerance_ratio": 0.05,
        "lock_in_minimum_mode_band_power_fraction": 0.80,
        "lock_in_minimum_consecutive_points": 3,
        "lock_in_reference_peak_neighborhood_u_star": 0.30,
        "lock_in_endpoint_tolerance_u_star": 0.30,
        "timestep_frequency_tolerance_ratio": 0.01,
        "timestep_peak_location_tolerance_u_star": 0.25,
        "timestep_band_endpoint_tolerance_u_star": 0.15,
        "limit_headroom_fraction": 0.90,
        "live_analytical_rmse_tolerance": 0.03,
        "live_analytical_max_abs_tolerance": 0.05,
        "live_analytical_peak_location_tolerance_u_star": 0.25,
        "coordinate_absolute_amplitude_ratio_tolerance": 0.01,
        "coordinate_relative_amplitude_tolerance": 0.05,
        "material_improvement_minimum_skill": 0.50,
        "fatigue_proxy_slope": 10.0,
        "startup_seed_observation_cycles": 1.0,
        "startup_seed_realized_minimum_amplitude_ratio": 0.001,
        "startup_seed_realized_maximum_amplitude_ratio": 0.010,
        "lupi_experiment_curve_rmse_tolerance": 0.05,
        "lupi_experiment_peak_location_tolerance_u_star": 0.30,
        "lupi_experiment_peak_absolute_tolerance": 0.03,
        "lupi_experiment_peak_relative_tolerance": 0.15,
    }


def test_sc5p18_peak_velocities_come_from_digitized_figure9_series():
    figure9 = benchmark._reference_path("sc5p18_figure9_digitized.csv")

    hgast = benchmark.digitized_series_peak(figure9, benchmark.CALIBRATION_SERIES)
    lupi = benchmark.digitized_series_peak(figure9, "experiment_lupi2021")

    assert hgast == {
        "reduced_velocity": 6.245825,
        "tip_amplitude_over_diameter": 0.264791,
    }
    assert lupi == {
        "reduced_velocity": 6.350126,
        "tip_amplitude_over_diameter": 0.235286,
    }
    assert hgast["reduced_velocity"] != benchmark.load_structural_cases()[
        benchmark.CALIBRATION_SC
    ].u_star_max_figure_digitized


def _lupi_matching_fine_curve():
    figure9 = benchmark._reference_path("sc5p18_figure9_digitized.csv")
    experimental_u, experimental_amplitude = benchmark._read_figure9_series(
        figure9, "experiment_lupi2021"
    )
    return [
        {
            "scruton_number": benchmark.CALIBRATION_SC,
            "reduced_velocity": u_star,
            "tip_peak_amplitude_over_diameter": float(
                np.interp(u_star, experimental_u, experimental_amplitude)
            ),
            "selected_timestep_level": "dt_half",
        }
        for u_star in benchmark.DEFAULT_REDUCED_VELOCITIES
    ]


def test_lupi_experimental_curve_screen_is_separate_complete_and_fail_closed():
    figure9 = benchmark._reference_path("sc5p18_figure9_digitized.csv")
    curve = _lupi_matching_fine_curve()

    comparison_rows, screen = benchmark.lupi_experimental_curve_screen(
        curve, figure9
    )

    assert screen["schema"] == "vlastos2026-lupi-experimental-curve-screen-v1"
    assert screen["figure9_sha256"] == benchmark.sha256_file(figure9)
    assert screen["evaluated"] is True
    assert screen["pass"] is True
    assert screen["complete_12_point_fine_response_curve"] is True
    assert screen["experimental_marker_count"] == 27
    assert screen["in_range_marker_count"] == 27
    assert screen["out_of_range_marker_count"] == 0
    assert len(comparison_rows) == 27
    assert screen["curve_rmse_tip_amplitude_over_diameter"] < 0.01
    assert screen["curve_rmse_tolerance"] == 0.05
    assert screen["curve_rmse_pass"] is True
    assert screen["response_peak_u_star"] == 6.5
    assert screen["experimental_peak_u_star"] == 6.350126
    assert screen["peak_location_absolute_error_u_star"] == pytest.approx(
        0.149874
    )
    assert screen["peak_location_tolerance_u_star"] == 0.30
    assert screen["peak_location_pass"] is True
    assert screen["experimental_peak_amplitude_over_diameter"] == 0.235286
    assert screen["peak_amplitude_tolerance"] == pytest.approx(0.0352929)
    assert screen["peak_amplitude_pass"] is True
    assert "not a calibration input" in screen["scope"]

    incomplete_rows, incomplete = benchmark.lupi_experimental_curve_screen(
        curve[:-1], figure9
    )
    assert incomplete_rows == []
    assert incomplete["evaluated"] is False
    assert incomplete["pass"] is False
    assert incomplete["complete_12_point_fine_response_curve"] is False

    failed_curve = copy.deepcopy(curve)
    for row in failed_curve:
        row["tip_peak_amplitude_over_diameter"] = 0.0
    failed_rows, failed = benchmark.lupi_experimental_curve_screen(
        failed_curve, figure9
    )
    assert len(failed_rows) == 27
    assert failed["evaluated"] is True
    assert failed["pass"] is False
    assert failed["curve_rmse_pass"] is False
    assert failed["peak_amplitude_pass"] is False

    invalid_curve = copy.deepcopy(curve)
    invalid_curve[0]["tip_peak_amplitude_over_diameter"] = math.nan
    invalid_rows, invalid = benchmark.lupi_experimental_curve_screen(
        invalid_curve, figure9
    )
    assert invalid_rows == []
    assert invalid["evaluated"] is False
    assert invalid["pass"] is False
    assert invalid["reason"] == "the complete response curve contains invalid values"


def _passing_calibration_stage_inputs():
    cases = []
    result_rows = []
    runtime_audits = []
    for sc in (2.94, 5.18, 7.02):
        role = "calibration" if sc == benchmark.CALIBRATION_SC else "held_out"
        for u_star in benchmark.DEFAULT_REDUCED_VELOCITIES:
            for timestep_level in ("dt", "dt_half"):
                case_id = (
                    f"sc{str(sc).replace('.', 'p')}_"
                    f"u{str(u_star).replace('.', 'p')}_{timestep_level}"
                )
                cases.append(
                    {
                        "case_id": case_id,
                        "scruton_number": sc,
                        "role": role,
                        "reduced_velocity": u_star,
                        "timestep_level": timestep_level,
                    }
                )
                if role != "calibration":
                    continue
                result_rows.append(
                    {
                        "case_id": case_id,
                        "scruton_number": sc,
                        "role": role,
                        "stationarity_evaluated": True,
                        "stationarity_pass": True,
                    }
                )
                runtime_audits.append(
                    {
                        "case_id": case_id,
                        "evaluated": True,
                        "provenance_evaluated": True,
                        "provenance_pass": True,
                        "identity_mismatch": False,
                        "execution_pass": True,
                        "source_coverage_complete": True,
                        "source_limits_pass": True,
                        "response_summary_limits_pass": True,
                        "startup_seed_summary_pass": True,
                        "startup_seed_response": {
                            "evaluated": True,
                            "pass": True,
                        },
                        "history": {
                            "runtime_model_pass": True,
                            "compact_lag_provenance_pass": True,
                            "applied_power_sign_convention_pass": True,
                            "limit_headroom_pass": True,
                            "startup_seed_integrity_evaluated": True,
                            "startup_seed_integrity_pass": True,
                            "response_estimator_stationarity_evaluated": True,
                            "response_estimator_stationarity_pass": True,
                        },
                        "pass": True,
                    }
                )
    dt_rows = [
        {
            "scruton_number": benchmark.CALIBRATION_SC,
            "reduced_velocity": u_star,
            "frequency_evaluated": True,
            "frequency_pass": True,
            "both_stationary": True,
            "both_runtime_audits_pass": True,
            "pass": True,
        }
        for u_star in benchmark.DEFAULT_REDUCED_VELOCITIES
    ]
    return {
        "manifest": {
            "numerics": {"qualification_input_eligible": True},
            "cases": cases,
        },
        "result_rows": result_rows,
        "runtime_audits": runtime_audits,
        "dt_rows": dt_rows,
        "dt_curve_rows": [
            {
                "scruton_number": benchmark.CALIBRATION_SC,
                "evaluated": True,
                "pass": True,
            }
        ],
        "mechanism_rows": [
            {
                "scruton_number": benchmark.CALIBRATION_SC,
                "evaluated": True,
                "pass": True,
            }
        ],
        "live_analytical_summary": {
            "evaluated": True,
            "pass": True,
            "coordinate_consistency_evaluated": True,
            "coordinate_consistency_pass": True,
        },
        "calibration_curve_summary": {"evaluated": True, "pass": True},
        "calibration_band_summary": {"evaluated": True, "pass": True},
        "peak_rows": [
            {
                "scruton_number": benchmark.CALIBRATION_SC,
                "role": "calibration",
                "enhanced_curve_complete": True,
                "external_code_comparison_pass": True,
            }
        ],
    }


def test_calibration_stage_gate_passes_exact_24_cases_without_heldout_results():
    inputs = _passing_calibration_stage_inputs()

    assessment = benchmark.calibration_stage_assessment(**inputs)

    assert assessment["schema"] == "vlastos2026-calibration-stage-assessment-v1"
    assert assessment["status"] == "calibration_stage_pass"
    assert assessment["evaluated"] is True
    assert assessment["pass"] is True
    assert assessment["expected_case_count"] == 24
    assert assessment["manifest_calibration_case_count"] == 24
    assert assessment["completed_calibration_case_count"] == 24
    assert assessment["runtime_calibration_case_count"] == 24
    assert len(inputs["manifest"]["cases"]) == 72
    assert len(inputs["result_rows"]) == 24
    assert not any("lupi" in name for name in assessment["checks"])
    assert set(assessment) == {
        "schema",
        "status",
        "evaluated",
        "pass",
        "scope",
        "expected_scruton_number",
        "expected_case_count",
        "expected_reduced_velocities",
        "expected_timestep_levels",
        "manifest_calibration_case_count",
        "completed_calibration_case_count",
        "runtime_calibration_case_count",
        "checks",
    }
    assert all(assessment["checks"].values())


def test_calibration_stage_gate_fails_closed_for_every_required_check():
    failure_mutations = (
        (
            "runtime_provenance_pass",
            lambda value: value["runtime_audits"][0].update(
                provenance_pass=False
            ),
        ),
        (
            "runtime_execution_and_model_pass",
            lambda value: value["runtime_audits"][0].update(execution_pass=False),
        ),
        (
            "runtime_limits_pass",
            lambda value: value["runtime_audits"][0].update(
                response_summary_limits_pass=False
            ),
        ),
        (
            "startup_seed_pass",
            lambda value: value["runtime_audits"][0].update(
                startup_seed_summary_pass=False
            ),
        ),
        (
            "result_and_estimator_stationarity_pass",
            lambda value: value["result_rows"][0].update(stationarity_pass=False),
        ),
        (
            "runtime_case_audits_pass",
            lambda value: value["runtime_audits"][0].update({"pass": False}),
        ),
        (
            "timestep_amplitude_pass",
            lambda value: value["dt_rows"][0].update({"pass": False}),
        ),
        (
            "timestep_frequency_pass",
            lambda value: value["dt_rows"][0].update(frequency_pass=False),
        ),
        (
            "timestep_pair_runtime_stationarity_pass",
            lambda value: value["dt_rows"][0].update(both_stationary=False),
        ),
        (
            "timestep_curve_properties_pass",
            lambda value: value["dt_curve_rows"][0].update({"pass": False}),
        ),
        (
            "live_vs_analytical_pass",
            lambda value: value["live_analytical_summary"].update({"pass": False}),
        ),
        (
            "coordinate_consistency_pass",
            lambda value: value["live_analytical_summary"].update(
                coordinate_consistency_pass=False
            ),
        ),
        (
            "lock_in_mechanism_pass",
            lambda value: value["mechanism_rows"][0].update({"pass": False}),
        ),
        (
            "external_hgast_peak_pass",
            lambda value: value["peak_rows"][0].update(
                external_code_comparison_pass=False
            ),
        ),
        (
            "external_hgast_curve_pass",
            lambda value: value["calibration_curve_summary"].update({"pass": False}),
        ),
        (
            "external_hgast_band_pass",
            lambda value: value["calibration_band_summary"].update({"pass": False}),
        ),
    )

    for check_name, mutate in failure_mutations:
        inputs = copy.deepcopy(_passing_calibration_stage_inputs())
        mutate(inputs)
        assessment = benchmark.calibration_stage_assessment(**inputs)
        assert assessment["evaluated"] is True, check_name
        assert assessment["status"] == "calibration_stage_fail", check_name
        assert assessment["pass"] is False, check_name
        assert assessment["checks"][check_name] is False, check_name

    incomplete = _passing_calibration_stage_inputs()
    incomplete["result_rows"].pop()
    assessment = benchmark.calibration_stage_assessment(**incomplete)
    assert assessment["status"] == "not_evaluated"
    assert assessment["evaluated"] is False
    assert assessment["pass"] is False
    assert assessment["checks"]["all_calibration_results_complete"] is False


def test_ambient_v2_calibration_stage_requires_strict_equation_gate():
    inputs = _passing_calibration_stage_inputs()
    inputs["manifest"]["equation_reproduction_gate"] = {
        "schema": benchmark.AMBIENT_V2_EQUATION_GATE_SCHEMA
    }
    inputs["equation_reproduction_summary"] = {
        "evaluated": True,
        "pass": False,
    }

    failed = benchmark.calibration_stage_assessment(**inputs)

    assert failed["evaluated"] is True
    assert failed["status"] == "calibration_stage_fail"
    assert failed["pass"] is False
    assert failed["checks"][
        "ambient_v2_equation_reproduction_evaluated"
    ] is True
    assert failed["checks"]["ambient_v2_equation_reproduction_pass"] is False

    unevaluated_inputs = copy.deepcopy(inputs)
    unevaluated_inputs["equation_reproduction_summary"] = {
        "evaluated": False,
        "pass": False,
    }
    unevaluated = benchmark.calibration_stage_assessment(**unevaluated_inputs)
    assert unevaluated["evaluated"] is False
    assert unevaluated["status"] == "not_evaluated"
    assert unevaluated["pass"] is False

    passing_inputs = copy.deepcopy(inputs)
    passing_inputs["equation_reproduction_summary"]["pass"] = True
    passed = benchmark.calibration_stage_assessment(**passing_inputs)
    assert passed["evaluated"] is True
    assert passed["status"] == "calibration_stage_pass"
    assert passed["pass"] is True


def test_response_metrics_distinguish_stationary_lock_in_drift_and_short_pilot(
    tmp_path,
):
    stable_path = tmp_path / "stable.txt"
    stable_samples = _write_synthetic_result(
        stable_path,
        frequency_hz=5.0,
        timestep_s=0.002,
        amplitude_ratio=0.20,
    )
    stable = benchmark.response_amplitude_from_result(
        stable_path,
        natural_frequency_hz=5.0,
        response_window_cycles=40.0,
        expected_timestep_s=0.002,
        expected_samples=stable_samples,
    )

    assert stable["tip_peak_amplitude_over_diameter"] == pytest.approx(0.20)
    assert stable["stationarity_evaluated"] is True
    assert stable["stationarity_pass"] is True
    assert stable["terminal_dominant_frequency_over_fn"] == pytest.approx(
        1.0, abs=2.0e-4
    )
    assert stable["terminal_mode_band_power_fraction"] > 0.99

    drift_path = tmp_path / "drift.txt"
    drift_samples = _write_synthetic_result(
        drift_path,
        frequency_hz=5.0,
        timestep_s=0.002,
        amplitude_ratio=0.10,
        final_amplitude_ratio=0.20,
    )
    drift = benchmark.response_amplitude_from_result(
        drift_path,
        natural_frequency_hz=5.0,
        response_window_cycles=40.0,
        expected_timestep_s=0.002,
        expected_samples=drift_samples,
    )
    assert drift["stationarity_evaluated"] is True
    assert drift["stationarity_amplitude_pass"] is False
    assert drift["stationarity_pass"] is False

    short_path = tmp_path / "short.txt"
    short_samples = _write_synthetic_result(
        short_path,
        frequency_hz=5.0,
        timestep_s=0.002,
        amplitude_ratio=0.20,
        duration_cycles=10.0,
    )
    short = benchmark.response_amplitude_from_result(
        short_path,
        natural_frequency_hz=5.0,
        response_window_cycles=8.0,
        expected_timestep_s=0.002,
        expected_samples=short_samples,
    )
    assert short["tip_peak_amplitude_over_diameter"] == pytest.approx(0.20)
    assert short["stationarity_evaluated"] is False
    assert short["stationarity_pass"] is False


def test_response_metrics_reconstruct_quantized_fine_export_grid_fail_closed(
    tmp_path,
):
    timestep_s = 0.00025
    expected_samples = 62_420
    natural_frequency_hz = 5.0
    result_path = tmp_path / "fine_quantized.txt"
    written_samples = _write_synthetic_result(
        result_path,
        frequency_hz=natural_frequency_hz,
        timestep_s=timestep_s,
        amplitude_ratio=0.20,
        sample_count=expected_samples,
        time_format=benchmark.QBLADE_RESULT_TIME_FORMAT,
        accumulate_export_time=True,
    )
    assert written_samples == expected_samples

    exported = np.loadtxt(result_path, delimiter="\t", skiprows=1)
    exported_difference = np.diff(exported[:, 0])
    assert np.any(np.isclose(exported_difference, 0.0002, atol=1.0e-12))
    assert np.any(np.isclose(exported_difference, 0.0003, atol=1.0e-12))
    assert f"{exported[-1, 0]:.5E}" == "1.56047E+01"

    response = benchmark.response_amplitude_from_result(
        result_path,
        natural_frequency_hz=natural_frequency_hz,
        response_window_cycles=40.0,
        expected_timestep_s=timestep_s,
        expected_samples=expected_samples,
    )
    audit = response["export_time_grid_audit"]
    assert response["configured_analysis_timestep_s"] == timestep_s
    assert response["tip_peak_amplitude_over_diameter"] == pytest.approx(0.20)
    assert response["terminal_dominant_frequency_over_fn"] == pytest.approx(
        1.0, abs=2.0e-4
    )
    assert audit["schema"] == benchmark.QBLADE_RESULT_TIME_GRID_SCHEMA
    assert audit["export_format"] == "%.5E"
    assert audit["expected_samples"] == expected_samples
    assert audit["duplicate_exported_time_count"] == 0
    assert audit["maximum_export_time_absolute_error_s"] == pytest.approx(
        5.0e-5, abs=1.0e-12
    )
    assert 5.0e-5 <= audit["maximum_export_time_rounding_tolerance_s"] < 5.0001e-5
    assert audit["maximum_export_time_renderer_mismatch_s"] == 0.0
    assert audit["maximum_export_time_renderer_match_tolerance_s"] < 1.0e-12
    assert audit["two_token_rounding_boundary_sample_count"] > 0
    assert audit["maximum_repeated_addition_time_bound_s"] > 0.0

    inserted_duplicate = exported.copy()
    inserted_duplicate[1000, 0] = inserted_duplicate[999, 0]
    inserted_duplicate_path = tmp_path / "inserted_duplicate.txt"
    np.savetxt(
        inserted_duplicate_path,
        inserted_duplicate,
        delimiter="\t",
        header=f"Time [s]\t{benchmark.RESULT_CHANNEL}",
        comments="",
        fmt=(benchmark.QBLADE_RESULT_TIME_FORMAT, "%.18e"),
    )
    with pytest.raises(ValueError, match="inconsistent with the configured grid"):
        benchmark.response_amplitude_from_result(
            inserted_duplicate_path,
            natural_frequency_hz=natural_frequency_hz,
            response_window_cycles=40.0,
            expected_timestep_s=timestep_s,
            expected_samples=expected_samples,
        )

    true_jitter = exported.copy()
    true_jitter[50_000, 0] += 1.0e-4
    true_jitter_path = tmp_path / "true_jitter.txt"
    np.savetxt(
        true_jitter_path,
        true_jitter,
        delimiter="\t",
        header=f"Time [s]\t{benchmark.RESULT_CHANNEL}",
        comments="",
        fmt=(benchmark.QBLADE_RESULT_TIME_FORMAT, "%.18e"),
    )
    with pytest.raises(ValueError, match="renderer mismatch"):
        benchmark.response_amplitude_from_result(
            true_jitter_path,
            natural_frequency_hz=natural_frequency_hz,
            response_window_cycles=40.0,
            expected_timestep_s=timestep_s,
            expected_samples=expected_samples,
        )

    decreasing = exported.copy()
    decreasing[1000, 0] = decreasing[999, 0] - 1.0e-4
    decreasing_path = tmp_path / "decreasing.txt"
    np.savetxt(
        decreasing_path,
        decreasing,
        delimiter="\t",
        header=f"Time [s]\t{benchmark.RESULT_CHANNEL}",
        comments="",
        fmt=(benchmark.QBLADE_RESULT_TIME_FORMAT, "%.18e"),
    )
    with pytest.raises(ValueError, match="must remain nondecreasing"):
        benchmark.response_amplitude_from_result(
            decreasing_path,
            natural_frequency_hz=natural_frequency_hz,
            response_window_cycles=40.0,
            expected_timestep_s=timestep_s,
            expected_samples=expected_samples,
        )

    with pytest.raises(ValueError, match="sample count does not match"):
        benchmark.response_amplitude_from_result(
            result_path,
            natural_frequency_hz=natural_frequency_hz,
            response_window_cycles=40.0,
            expected_timestep_s=timestep_s,
            expected_samples=expected_samples + 1,
        )
    with pytest.raises(ValueError, match="inconsistent with the configured grid"):
        benchmark.response_amplitude_from_result(
            result_path,
            natural_frequency_hz=natural_frequency_hz,
            response_window_cycles=40.0,
            expected_timestep_s=0.000251,
            expected_samples=expected_samples,
        )


def test_time_grid_reconstruction_accepts_renderer_duplicates_after_100_seconds():
    timestep_s = 0.00025
    expected_samples = 400_005
    exact_time = np.arange(expected_samples, dtype=float) * timestep_s
    exported_time = np.fromiter(
        (float(benchmark.QBLADE_RESULT_TIME_FORMAT % value) for value in exact_time),
        dtype=float,
        count=expected_samples,
    )
    assert exact_time[-1] > 100.0
    assert np.any(np.diff(exported_time) == 0.0)

    reconstructed_time, audit = benchmark._validated_qblade_result_time_grid(
        exported_time,
        expected_timestep_s=timestep_s,
        expected_samples=expected_samples,
    )

    assert np.array_equal(reconstructed_time, exact_time)
    assert audit["pass"] is True
    assert audit["nondecreasing_exported_time"] is True
    assert audit["duplicate_exported_time_count"] > 0
    assert audit["exported_adjacent_time_difference_min_s"] == 0.0
    assert audit["maximum_export_time_absolute_error_s"] == pytest.approx(
        5.0e-4, abs=1.0e-12
    )
    assert 5.0e-4 <= audit["maximum_export_time_rounding_tolerance_s"] < 5.001e-4
    assert audit["maximum_export_time_renderer_mismatch_s"] == 0.0

    shifted_time = exported_time.copy()
    shifted_time[400_001] += 0.001
    assert np.all(np.diff(shifted_time) >= 0.0)
    with pytest.raises(ValueError, match="inconsistent with the configured grid"):
        benchmark._validated_qblade_result_time_grid(
            shifted_time,
            expected_timestep_s=timestep_s,
            expected_samples=expected_samples,
        )


def test_response_history_and_runtime_audits_require_stationarity_and_limit_headroom(
    tmp_path,
):
    case_dir = tmp_path / "standalone_history"
    history_path = case_dir / "result_sil_vorlap_history.csv"
    _write_response_history(history_path)
    history = benchmark.response_history_audit(
        history_path,
        natural_frequency_hz=5.0,
        max_abs_gain=10.0,
        max_abs_nodal_force_n=25.0,
    )
    assert history["response_estimator_stationarity_pass"] is True
    assert history["limit_headroom_pass"] is True

    entry, manifest_files, runtime_dir, result_path = _write_bound_runtime_case(
        tmp_path / "campaign",
        tmp_path / "results",
        case_id="case_a",
        mean_inflow_mps=1.25,
    )
    runtime = benchmark.runtime_case_audit(
        runtime_dir,
        entry=entry,
        campaign_root=tmp_path / "campaign",
        manifest_files=manifest_files,
        result_path=result_path,
    )
    assert runtime["evaluated"] is True
    assert runtime["provenance_evaluated"] is True
    assert runtime["provenance_pass"] is True
    assert runtime["source_coverage_complete"] is True
    assert runtime["pass"] is True

    summary_path = runtime_dir / "result_run_summary.json"
    legacy_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    legacy_summary["sil_vorlap_history"].pop("file_identity")
    summary_path.write_text(json.dumps(legacy_summary), encoding="utf-8")
    legacy = benchmark.runtime_case_audit(
        runtime_dir,
        entry=entry,
        campaign_root=tmp_path / "campaign",
        manifest_files=manifest_files,
        result_path=result_path,
    )
    assert legacy["evaluated"] is True
    assert legacy["provenance_evaluated"] is False
    assert legacy["provenance_pass"] is False
    assert legacy["identity_mismatch"] is False
    assert legacy["pass"] is False

    limited_path = tmp_path / "limited.csv"
    _write_response_history(
        limited_path,
        raw_gain=9.1,
        gain_limited_node_samples=1,
    )
    limited = benchmark.response_history_audit(
        limited_path,
        natural_frequency_hz=5.0,
        max_abs_gain=10.0,
        max_abs_nodal_force_n=25.0,
    )
    assert limited["limit_headroom_pass"] is False


def test_ambient_v2_history_audit_binds_speed_width_and_reduced_velocity(
    tmp_path,
):
    history_path = tmp_path / "ambient_v2_history.csv"
    _write_compact_seeded_response_history(
        history_path,
        response_schema_version=2,
        ambient_response_speed_mps=1.5625,
        configured_reduced_velocity=6.25,
    )
    response_config = {
        "schema_version": 2,
        "aerodynamic_speed_basis": (
            benchmark.AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
        ),
    }
    seed_config = {
        "duration_cycles": 5.0,
        "target_amplitude_ratio": 0.005,
    }

    audit = benchmark.response_history_audit(
        history_path,
        natural_frequency_hz=5.0,
        max_abs_gain=10.0,
        max_abs_nodal_force_n=25.0,
        history_mode="compact",
        expected_force_vector_component_count=63,
        startup_seed_config=seed_config,
        response_model_config=response_config,
        expected_reduced_velocity=6.25,
        expected_ambient_response_speed_mps=1.5625,
    )

    assert audit["runtime_model_pass"] is True
    assert audit["speed_basis_provenance_pass"] is True
    assert audit["response_model_schema_version"] == 2
    assert audit["aerodynamic_speed_basis"] == (
        benchmark.AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
    )
    assert audit["ambient_response_speed_min_mps"] == pytest.approx(1.5625)
    assert audit["ambient_response_speed_max_mps"] == pytest.approx(1.5625)
    assert audit["response_section_width_min_m"] == pytest.approx(
        benchmark.DIAMETER_M
    )
    assert audit["response_section_width_max_m"] == pytest.approx(
        benchmark.DIAMETER_M
    )

    with pytest.raises(ValueError, match=r"requires expected U\*"):
        benchmark.response_history_audit(
            history_path,
            natural_frequency_hz=5.0,
            max_abs_gain=10.0,
            max_abs_nodal_force_n=25.0,
            history_mode="compact",
            expected_force_vector_component_count=63,
            startup_seed_config=seed_config,
            response_model_config=response_config,
        )


def test_runtime_audit_rejects_swapped_case_and_modified_result(tmp_path):
    campaign_root = tmp_path / "campaign"
    results_root = tmp_path / "results"
    entry_a, files_a, _, _ = _write_bound_runtime_case(
        campaign_root,
        results_root,
        case_id="case_a",
        mean_inflow_mps=1.25,
    )
    _, files_b, directory_b, result_b = _write_bound_runtime_case(
        campaign_root,
        results_root,
        case_id="case_b",
        mean_inflow_mps=2.50,
    )
    manifest_files = {**files_a, **files_b}

    swapped = benchmark.runtime_case_audit(
        directory_b,
        entry=entry_a,
        campaign_root=campaign_root,
        manifest_files=manifest_files,
        result_path=result_b,
    )
    assert swapped["pass"] is False
    assert swapped["provenance_pass"] is False
    assert swapped["identity_mismatch"] is True
    assert any(
        "does not identify this case" in problem
        or "campaign manifest" in problem
        or "MEANINF" in problem
        for problem in swapped["problems"]
    )

    entry_b = {
        **entry_a,
        "case_id": "case_b",
        "sim_path": "cases/case_b/case_b.sim",
        "parameter_file": "cases/case_b/model/Control/vorlap_qblade_external.json",
        "mean_inflow_mps": 2.50,
    }
    result_b.write_text(result_b.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    modified = benchmark.runtime_case_audit(
        directory_b,
        entry=entry_b,
        campaign_root=campaign_root,
        manifest_files=manifest_files,
        result_path=result_b,
    )
    assert modified["pass"] is False
    assert modified["identity_mismatch"] is True
    assert any("result SHA-256" in problem for problem in modified["problems"])


def test_compact_seeded_runtime_audit_binds_lag_seed_and_realized_response(tmp_path):
    entry, manifest_files, runtime_dir, result_path = _write_bound_runtime_case(
        tmp_path / "campaign",
        tmp_path / "results",
        case_id="seeded_case",
        mean_inflow_mps=1.25,
        amplitude_ratio=0.005,
        compact_seeded=True,
    )

    audit = benchmark.runtime_case_audit(
        runtime_dir,
        entry=entry,
        campaign_root=tmp_path / "campaign",
        manifest_files=manifest_files,
        result_path=result_path,
        expected_history_mode="compact",
    )

    assert audit["evaluated"] is True
    assert audit["provenance_pass"] is True
    assert audit["history"]["compact_lag_provenance_pass"] is True
    assert audit["history"]["startup_seed_integrity_pass"] is True
    assert audit["startup_seed_summary_pass"] is True
    assert audit["startup_seed_response"]["realized_amplitude_ratio"] == pytest.approx(
        0.005
    )
    assert audit["startup_seed_response"]["pass"] is True
    assert audit["pass"] is True

    history_path = runtime_dir / "result_sil_vorlap_history.csv"
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[5]["applied_force_vector_sha256"] = "f" * 64
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = runtime_dir / "result_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["sil_vorlap_history"]["file_identity"] = _portable_file_identity(
        history_path,
        "/container/results/seeded_case/result_sil_vorlap_history.csv",
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    tampered = benchmark.runtime_case_audit(
        runtime_dir,
        entry=entry,
        campaign_root=tmp_path / "campaign",
        manifest_files=manifest_files,
        result_path=result_path,
        expected_history_mode="compact",
    )
    assert tampered["pass"] is False
    assert any("prior generated packet" in problem for problem in tampered["problems"])


def test_startup_seed_realized_response_gate_rejects_underseeded_motion(tmp_path):
    result_path = tmp_path / "underseeded.txt"
    samples = _write_synthetic_result(
        result_path,
        frequency_hz=5.0,
        timestep_s=0.002,
        amplitude_ratio=0.0005,
    )

    audit = benchmark.startup_seed_response_audit(
        result_path,
        natural_frequency_hz=5.0,
        seed_duration_cycles=5.0,
        expected_timestep_s=0.002,
        expected_samples=samples,
    )

    assert audit["realized_amplitude_ratio"] == pytest.approx(0.0005)
    assert audit["pass"] is False


def test_startup_seed_audit_accepts_only_exact_grid_renderer_duplicates(
    tmp_path, monkeypatch
):
    timestep_s = 0.00025
    expected_samples = 400_005
    natural_frequency_hz = 5.0
    exact_time = np.arange(expected_samples, dtype=float) * timestep_s
    exported_time = np.fromiter(
        (
            float(benchmark.QBLADE_RESULT_TIME_FORMAT % value)
            for value in exact_time
        ),
        dtype=float,
        count=expected_samples,
    )
    displacement = (
        0.005
        * benchmark.DIAMETER_M
        * np.sin(2.0 * math.pi * natural_frequency_hz * exact_time)
    )
    columns = {
        "Time [s]": exported_time,
        benchmark.RESULT_CHANNEL: displacement,
    }
    monkeypatch.setattr(
        benchmark,
        "read_qblade_result_numeric_columns",
        lambda _path: columns,
    )

    assert exact_time[-1] > 100.0
    assert np.any(np.diff(exported_time) == 0.0)
    audit = benchmark.startup_seed_response_audit(
        tmp_path / "rendered-result.txt",
        natural_frequency_hz=natural_frequency_hz,
        seed_duration_cycles=5.0,
        expected_timestep_s=timestep_s,
        expected_samples=expected_samples,
    )
    assert audit["pass"] is True
    assert audit["realized_amplitude_ratio"] == pytest.approx(0.005)
    assert audit["export_time_grid_audit"]["pass"] is True
    assert audit["export_time_grid_audit"]["duplicate_exported_time_count"] > 0

    inserted_duplicate = exported_time.copy()
    inserted_duplicate[1000] = inserted_duplicate[999]
    columns["Time [s]"] = inserted_duplicate
    with pytest.raises(ValueError, match="inconsistent with the configured grid"):
        benchmark.startup_seed_response_audit(
            tmp_path / "inserted-duplicate.txt",
            natural_frequency_hz=natural_frequency_hz,
            seed_duration_cycles=5.0,
            expected_timestep_s=timestep_s,
            expected_samples=expected_samples,
        )

    true_jitter = exported_time.copy()
    true_jitter[400_001] += 0.001
    assert np.all(np.diff(true_jitter) >= 0.0)
    columns["Time [s]"] = true_jitter
    with pytest.raises(ValueError, match="inconsistent with the configured grid"):
        benchmark.startup_seed_response_audit(
            tmp_path / "jittered.txt",
            natural_frequency_hz=natural_frequency_hz,
            seed_duration_cycles=5.0,
            expected_timestep_s=timestep_s,
            expected_samples=expected_samples,
        )

    columns["Time [s]"] = exported_time
    with pytest.raises(ValueError, match="inconsistent with the configured grid"):
        benchmark.startup_seed_response_audit(
            tmp_path / "wrong-grid.txt",
            natural_frequency_hz=natural_frequency_hz,
            seed_duration_cycles=5.0,
            expected_timestep_s=0.000251,
            expected_samples=expected_samples,
        )


def test_response_history_audit_streams_large_wide_history_without_dict_rows(
    tmp_path, monkeypatch
):
    history_path = tmp_path / "large_history.csv"
    fieldnames = [
        "force_history_schema",
        "step_index",
        "time_s",
        "response_model_type",
        "response_enabled",
        "response_amplitude_ratio",
        "response_raw_gain_min",
        "response_raw_gain_max",
        "response_gain_limited_nodes",
        "response_force_limited_nodes",
        "response_max_abs_nodal_force_n",
        *[f"unused_node_column_{index}" for index in range(300)],
    ]
    samples = 100_000
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        unused = ["0"] * 300
        for index in range(samples):
            writer.writerow(
                [
                    "vorlap-sil-force-history-v2",
                    index,
                    0.1 * index,
                    "modal_negative_damping",
                    "true",
                    0.2,
                    -5.0,
                    5.0,
                    0,
                    0,
                    10.0,
                    *unused,
                ]
            )

    def reject_dict_reader(*_args, **_kwargs):
        raise AssertionError("response history must not materialize DictReader rows")

    monkeypatch.setattr(benchmark.csv, "DictReader", reject_dict_reader)
    audit = benchmark.response_history_audit(
        history_path,
        natural_frequency_hz=1.0,
        max_abs_gain=10.0,
        max_abs_nodal_force_n=25.0,
    )

    assert audit["samples"] == samples
    assert audit["columns"] == len(fieldnames)
    assert audit["streaming_passes"] == 2
    assert audit["rows_retained_in_memory"] == 1
    assert audit["terminal_window_samples_processed"] <= 402
    assert audit["response_estimator_stationarity_pass"] is True


def test_material_improvement_requires_skill_over_zero_and_frozen_baselines():
    u_star = [0.0, 1.0, 2.0, 3.0, 4.0]
    reference = [0.0, 0.4, 1.0, 0.4, 0.0]
    frozen = [0.0, 0.3, 0.2, 0.1, 0.0]

    improved = benchmark.material_improvement_metrics(
        reference_u=u_star,
        reference_amplitude=reference,
        response_u=u_star,
        response_amplitude=reference,
        frozen_u=u_star,
        frozen_amplitude=frozen,
    )
    assert improved["evaluated"] is True
    assert improved["skill_over_no_viv"] == pytest.approx(1.0)
    assert improved["skill_over_frozen"] == pytest.approx(1.0)
    assert improved["pass"] is True
    assert "shared reference" in improved["baseline_pairing"]

    no_improvement = benchmark.material_improvement_metrics(
        reference_u=u_star,
        reference_amplitude=reference,
        response_u=u_star,
        response_amplitude=frozen,
        frozen_u=u_star,
        frozen_amplitude=frozen,
    )
    assert no_improvement["skill_over_frozen"] == pytest.approx(0.0)
    assert no_improvement["pass"] is False

    # Peak and RMSE gates use the same paired reference grid; a raw-only point
    # cannot be mixed into just one of the comparisons.
    raw_dense_u = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]
    raw_dense_response = [0.0, 0.4, 5.0, 1.0, 0.4, 0.0]
    paired = benchmark.material_improvement_metrics(
        reference_u=u_star,
        reference_amplitude=reference,
        response_u=raw_dense_u,
        response_amplitude=raw_dense_response,
        frozen_u=u_star,
        frozen_amplitude=frozen,
    )
    assert paired["response_rmse_tip_amplitude_over_diameter"] == pytest.approx(0.0)
    assert paired["response_peak_amplitude_error"] == pytest.approx(0.0)
    assert paired["response_peak_location_error_u_star"] == pytest.approx(0.0)


def test_generate_plan_and_analyze_synthetic_all_sc_campaign(tmp_path):
    campaign_root = tmp_path / "campaign"
    manifest_path = benchmark.generate_campaign(
        campaign_root,
        reduced_velocities=(6.25,),
        duration_s=3.2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["calibration_policy"]["postfreeze_analytical_gate_status"] == "fail"
    assert manifest["qblade_mapping"]["scruton_mapping"]["not_an_exact_scruton_match"] is True
    fingerprints = {entry["coefficient_fingerprint_sha256"] for entry in manifest["cases"]}
    assert fingerprints == {manifest["calibration_policy"]["coefficient_fingerprint_sha256"]}
    for entry in manifest["cases"]:
        response = entry["response_model_config"]
        assert response["type"] == "modal_negative_damping"
        assert response["schema_version"] == 1
        assert response["force_direction"] == "normal"
        assert response["enabled"] is True
        assert response["bandpass_damping_ratio"] == 0.1
        assert len(response["projection_weights"]) == benchmark.N_STATIONS
        assert response["projection_weights"][0] == pytest.approx(
            0.5 * response["projection_weights"][1]
        )

    results_root = tmp_path / "results"
    target_by_sc = {2.94: 0.54, 5.18: 0.27, 7.02: 0.15}
    for entry in manifest["cases"]:
        _write_synthetic_result(
            results_root / entry["case_id"] / "result.txt",
            frequency_hz=float(entry["structure"]["natural_frequency_hz"]),
            timestep_s=float(entry["timestep_s"]),
            amplitude_ratio=target_by_sc[float(entry["scruton_number"])],
            sample_count=int(entry["num_steps"]),
        )

    plan = benchmark.build_run_plan(
        manifest_path,
        results_root,
        stage="held_out",
        qblade_root=Path("run_headless/QBladeCE_2.0.9.7"),
    )
    assert len(plan["commands"]) == 4
    assert plan["postfreeze_analytical_gate_status"] == "fail"
    assert plan["warning"] is not None
    assert plan["threads_per_qblade_run"] == 4
    assert plan["sil_vorlap_history_mode"] == "compact"
    assert all(
        "--sil-vorlap-history-mode" in row["argv"]
        and "compact" in row["argv"]
        and "--record-sil-vorlap-history" not in row["argv"]
        for row in plan["commands"]
    )
    with pytest.raises(RuntimeError, match="analytical held-out gate failed"):
        benchmark.execute_run_plan(plan, manifest_path)

    summary_path = benchmark.analyze_campaign(
        manifest_path,
        results_root,
        tmp_path / "analysis",
        response_window_cycles=40.0,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "not_evaluated"
    assert summary["calibration_stage_status"] == "not_evaluated"
    assert summary["calibration_stage_evaluated"] is False
    assert summary["calibration_stage_pass"] is False
    assert (
        summary["calibration_stage_assessment"]["schema"]
        == "vlastos2026-calibration-stage-assessment-v1"
    )
    assert summary["lupi_experimental_curve_screen_evaluated"] is False
    assert summary["lupi_experimental_curve_screen_pass"] is False
    assert (
        summary["lupi_experimental_curve_screen"]["schema"]
        == "vlastos2026-lupi-experimental-curve-screen-v1"
    )
    assert summary["calibration_digitized_peak_locations"] == {
        "hgast_reduced_velocity": 6.245825,
        "hgast_tip_amplitude_over_diameter": 0.264791,
        "lupi_reduced_velocity": 6.350126,
        "lupi_tip_amplitude_over_diameter": 0.235286,
        "source": "sampled maxima of hash-bound digitized Figure 9 series",
    }
    peak_by_sc = {
        row["scruton_number"]: row for row in summary["peak_comparisons"]
    }
    assert peak_by_sc[5.18]["hgast_reference_peak_u_star"] == 6.245825
    assert peak_by_sc[5.18]["experiment_reference_peak_u_star"] == 6.350126
    assert peak_by_sc[2.94]["hgast_reference_peak_u_star"] == 6.8
    assert peak_by_sc[7.02]["hgast_reference_peak_u_star"] == 6.2
    assert summary["enhanced_assessment_status"] == "not_evaluated"
    assert summary["enhanced_qualification_status"] == "not_evaluated"
    assert summary["assessment_pass"] is False
    assert summary["qualification_pass"] is False
    assert summary["checks"]["dt_halving_pass"] is True
    assert summary["checks"]["calibration_external_code_peak_comparison_pass"] is True
    assert summary["checks"]["calibration_external_code_curve_comparison_pass"] is False
    assert summary["checks"]["held_out_external_code_peak_comparison_pass"] is True
    assert summary["held_out_targets_accessed_during_calibration"] is False
    assert summary["completed_case_count"] == 6
    assert summary["enhanced_checks"]["all_result_windows_stationary"] is True
    assert summary["enhanced_checks"]["runtime_artifacts_complete"] is False
    assert summary["enhanced_checks"]["live_vs_analytical_evaluated"] is False
    assert summary["enhanced_checks"]["lock_in_mechanism_evaluated"] is False
    assert summary["enhanced_checks"][
        "heldout_material_improvement_over_matched_frozen_evaluated"
    ] is False
    assert summary["enhanced_checks"][
        "heldout_material_improvement_over_matched_frozen_pass"
    ] is False
    assert summary["material_improvement_over_no_and_frozen"]["evaluated"] is False
    assert (summary_path.parent / "response_peak_vs_reduced_velocity.csv").is_file()
    assert (summary_path.parent / "response_peak_vs_reduced_velocity.png").is_file()

    # A second, now-incomplete analysis may not reuse the successful output
    # directory and leave stale CSV/PDF artifacts looking current.
    first_result = results_root / manifest["cases"][0]["case_id"] / "result.txt"
    first_result.unlink()
    with pytest.raises(FileExistsError, match="nonempty analysis output"):
        benchmark.analyze_campaign(
            manifest_path,
            results_root,
            summary_path.parent,
            response_window_cycles=40.0,
        )

    # Every campaign input is size/hash checked before a new analysis begins.
    sim_path = campaign_root / manifest["cases"][0]["sim_path"]
    sim_path.write_text(sim_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Campaign manifest file (size|hash) changed"):
        benchmark.analyze_campaign(
            manifest_path,
            results_root,
            tmp_path / "analysis_after_modified_input",
            response_window_cycles=40.0,
        )
