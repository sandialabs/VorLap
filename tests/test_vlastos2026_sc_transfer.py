from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from examples.vlastos2026_cantilever import engineering_sc_transfer as transfer
from examples.vlastos2026_cantilever import response_benchmark


def _portable_hash(value: object) -> str:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_frozen_transfer_parameters_are_exact_at_sc5_and_bounded() -> None:
    definition = transfer.load_transfer_definition()
    assert definition["closure_fingerprint_sha256"] == (
        transfer.EXPECTED_DEFINITION_FINGERPRINT_SHA256
    )
    assert definition["selected_form_id"] == "shift_1__h1_none__asat_1"
    assert definition["selection_disclosure"]["independent_validation_claim_allowed"] is False

    base = {
        name: float(definition["base_frozen_parameters"][name])
        for name in transfer._RUNTIME_PARAMETER_NAMES
    }
    assert transfer.effective_frozen_parameters(5.18) == base
    assert transfer.effective_parameter_record(5.18)["activation_shift_u_star"] == 0.0
    assert transfer.effective_parameter_record(5.18)["saturation_amplitude_scale"] == 1.0

    low_sc = transfer.effective_frozen_parameters(2.94)
    assert low_sc["aerodynamic_damping_gain_h1"] == base[
        "aerodynamic_damping_gain_h1"
    ]
    assert low_sc["saturation_amplitude_ratio"] == pytest.approx(
        0.48973553846602763, abs=1.0e-15
    )
    assert low_sc["reduced_velocity_low"] == pytest.approx(
        5.68602101956694, abs=1.0e-15
    )
    assert low_sc["reduced_velocity_peak"] == pytest.approx(
        6.569315198350391, abs=1.0e-15
    )
    assert low_sc["reduced_velocity_high"] == pytest.approx(
        7.306292049413173, abs=1.0e-15
    )

    high_sc = transfer.effective_frozen_parameters(7.02)
    assert high_sc["aerodynamic_damping_gain_h1"] == base[
        "aerodynamic_damping_gain_h1"
    ]
    assert high_sc["saturation_amplitude_ratio"] == pytest.approx(
        0.1933182113365147, abs=1.0e-15
    )
    assert high_sc["reduced_velocity_low"] == pytest.approx(
        5.0968067297144435, abs=1.0e-15
    )
    assert high_sc["reduced_velocity_peak"] == pytest.approx(
        5.980100908497894, abs=1.0e-15
    )
    assert high_sc["reduced_velocity_high"] == pytest.approx(
        6.717077759560676, abs=1.0e-15
    )

    for invalid in (2.939999, 7.020001):
        with pytest.raises(ValueError, match="outside the frozen"):
            transfer.effective_frozen_parameters(invalid)
    with pytest.raises(TypeError, match="not boolean"):
        transfer.effective_frozen_parameters(True)
    with pytest.raises(ValueError, match="finite"):
        transfer.effective_frozen_parameters(float("nan"))


def test_dense_engineering_gate_passes_and_retains_original_failure() -> None:
    gate = transfer.analytical_dense_gate()
    assert gate["schema"] == transfer.ANALYTICAL_GATE_SCHEMA
    assert gate["pass"] is True
    assert gate["status"] == "pass"
    assert gate["independent_validation"] is False
    assert gate["sc5_exact_identity"] is True
    assert gate["sc5_maximum_parameter_absolute_difference"] == 0.0
    assert gate["dense_grid"]["points"] == 13001
    assert len(gate["comparisons"]) == 12

    rows = {
        (
            row["published_scruton_number"],
            row["model"],
            row["reference_series"],
        ): row
        for row in gate["comparisons"]
    }
    sc2 = rows[(2.94, "belloli_sc_transfer", "hgast_hartlen_currie")]
    assert sc2["predicted_peak_amplitude_over_diameter"] == pytest.approx(
        0.43232823669448345
    )
    assert sc2["predicted_peak_u_star"] == pytest.approx(6.5835)
    assert sc2["peak_amplitude_absolute_error_over_diameter"] == pytest.approx(
        0.10767176330551659
    )
    assert sc2["peak_amplitude_tolerance_over_diameter"] == pytest.approx(0.172)
    assert sc2["peak_location_absolute_error_u_star"] == pytest.approx(0.2165)
    assert sc2["engineering_joint_pass"] is True

    sc7 = rows[(7.02, "belloli_sc_transfer", "hgast_hartlen_currie")]
    assert sc7["predicted_peak_amplitude_over_diameter"] == pytest.approx(
        0.1269237433950313
    )
    assert sc7["predicted_peak_u_star"] == pytest.approx(5.9955)
    assert sc7["peak_amplitude_absolute_error_over_diameter"] == pytest.approx(
        0.023076256604968703
    )
    assert sc7["peak_amplitude_tolerance_over_diameter"] == pytest.approx(0.055)
    assert sc7["peak_location_absolute_error_u_star"] == pytest.approx(0.2045)
    assert sc7["engineering_joint_pass"] is True

    assert all(
        row["engineering_joint_pass"]
        for row in gate["comparisons"]
        if row["model"] == "belloli_sc_transfer"
    )
    original_sc2 = rows[(2.94, "original_schema_v2", "hgast_hartlen_currie")]
    assert original_sc2["engineering_joint_pass"] is False
    assert original_sc2["peak_amplitude_relative_error"] == pytest.approx(
        0.5183009064139986
    )


def test_transfer_layer_does_not_change_schema_v1_or_v2_config_bytes() -> None:
    coefficients = response_benchmark.portable_ambient_v2_coefficients(
        response_benchmark.calibrate_response_coefficients()
    )
    case = response_benchmark.load_structural_cases()[5.18]
    span = np.linspace(0.0, response_benchmark.HEIGHT_M, response_benchmark.N_STATIONS)
    mode = response_benchmark.fixed_wake_case.first_mode(
        span / response_benchmark.HEIGHT_M
    )
    node_ids = [
        f"BLD_1_{value:.6f}" for value in span / response_benchmark.HEIGHT_M
    ]
    before = {
        schema: response_benchmark.response_model_config(
            case,
            node_ids,
            mode,
            coefficients,
            response_schema_version=schema,
        )
        for schema in (1, 2)
    }
    transfer.analytical_dense_gate()
    after = {
        schema: response_benchmark.response_model_config(
            case,
            node_ids,
            mode,
            coefficients,
            response_schema_version=schema,
        )
        for schema in (1, 2)
    }
    assert before == after
    reference = json.loads(
        (Path(__file__).parent / "fixtures/vlastos2026_schema_v1_v2_reference.json").read_text()
    )
    # Preserve the original exact configuration identities. The analytical
    # unit-normalized mode alone has last-bit libm differences across systems.
    assert _portable_hash(reference["1"]) == (
        "1620fe60b7c6a7dea406256973c21ae11157883f8e2e22823c35adc97ec5496d"
    )
    assert _portable_hash(reference["2"]) == (
        "b4fb161873c6a67355b444f1859744397007c24f0e9d54ceeacba17c612d1734"
    )
    for schema, actual in after.items():
        expected = reference[str(schema)]
        assert {k: v for k, v in actual.items() if k != "mode_shape"} == {
            k: v for k, v in expected.items() if k != "mode_shape"
        }
        np.testing.assert_allclose(
            actual["mode_shape"], expected["mode_shape"],
            rtol=0.0, atol=8.0 * np.finfo(float).eps,
        )


def test_publication_artifacts_are_traceable_and_explicitly_not_validation(
    tmp_path: Path,
) -> None:
    artifacts = transfer.render_analytical_artifacts(tmp_path / "evidence")
    assert set(artifacts) == {
        "gate",
        "comparison_csv",
        "latex_table",
        "figure",
    }
    assert artifacts["figure"].read_bytes().startswith(b"%PDF")
    assert artifacts["figure"].stat().st_size > 20_000
    gate = json.loads(artifacts["gate"].read_text(encoding="utf-8"))
    assert gate["pass"] is True
    assert gate["independent_validation"] is False
    assert "_curve_records" not in gate
    with artifacts["comparison_csv"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    table = artifacts["latex_table"].read_text(encoding="utf-8")
    assert "transfer form and are tuning evidence" in table
    assert "not independent validation" in table
    assert table.count(" & pass \\") == 6


@pytest.mark.requires_qblade
def test_exact_eight_campaign_generates_validates_and_plans_without_execution(
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "campaign"
    manifest_path = transfer.generate_transfer_campaign(campaign_root)
    manifest = transfer.validate_transfer_campaign(manifest_path)
    assert manifest["case_count"] == 8
    assert manifest["execution_state"] == "inputs_only_not_run"
    assert manifest["sc_transfer"]["independent_validation"] is False
    assert manifest["base_response_model"]["runtime_code_modified_for_transfer"] is False
    observed = {
        (
            float(row["scruton_number"]),
            float(row["reduced_velocity"]),
            str(row["timestep_level"]),
        )
        for row in manifest["cases"]
    }
    assert observed == set(transfer._expected_case_keys())

    plan = transfer.build_dry_run_plan(
        manifest_path,
        execution_campaign_root=PurePosixPath(
            "/work/VorLap/run_headless/runs/vlastos2026_belloli_sc_transfer"
        ),
        results_root=PurePosixPath("/results/vlastos2026_belloli_sc_transfer"),
        qblade_root=PurePosixPath(
            "/work/VorLap/run_headless/QBladeCE_2.0.9.7"
        ),
        threads=4,
    )
    assert plan["dry_run"] is True
    assert plan["execution_authorized"] is False
    assert len(plan["commands"]) == 8
    for row in plan["commands"]:
        argv = row["argv"]
        assert argv[0] == "/usr/bin/python3"
        assert "--threads" in argv
        assert argv[argv.index("--threads") + 1] == "4"
        assert "--sil-vorlap" in argv
        assert "--execute" not in argv

    first_parameter = campaign_root / manifest["cases"][0]["parameter_file"]
    runtime_config = json.loads(first_parameter.read_text(encoding="utf-8"))
    runtime_config["response_model"]["saturation_amplitude_ratio"] += 0.001
    first_parameter.write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash changed"):
        transfer.validate_transfer_campaign(manifest_path)


@pytest.mark.parametrize("threads", [0, 8, True])
def test_dry_plan_rejects_invalid_thread_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threads: object
) -> None:
    monkeypatch.setattr(
        transfer,
        "validate_transfer_campaign",
        lambda _: {
            "scope": "test",
            "manifest_fingerprint_sha256": "a" * 64,
            "sc_transfer": {"definition_fingerprint_sha256": "b" * 64},
            "cases": [],
        },
    )
    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="threads"):
        transfer.build_dry_run_plan(
            manifest_path,
            execution_campaign_root=PurePosixPath("/work/campaign"),
            results_root=PurePosixPath("/results"),
            qblade_root=PurePosixPath("/work/qblade"),
            threads=threads,  # type: ignore[arg-type]
        )
