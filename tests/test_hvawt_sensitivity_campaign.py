from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from vorlap import hvawt_sensitivity_campaign as sensitivity


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADED_V3 = (
    Path(os.environ.get("VORLAP_ARCHIVE_RUNS", ROOT / "run_headless/runs"))
    / "hvawt_complete_dlc_three_path_v3"
)


requires_archived_campaign = pytest.mark.skipif(
    not (DOWNLOADED_V3 / "campaign_manifest.json").is_file(),
    reason="Historical campaign archive required; set VORLAP_ARCHIVE_RUNS.",
)

def _runtime_config() -> dict:
    return {
        "sample_step": sensitivity.TIMESTEP_S,
        "source_frequency_cap_hz": sensitivity.NOMINAL_CAP_HZ,
        "source_parameter_dir": "/pscratch/old/accepted_v3/Control",
        "force_scale": 0.0,
        "unrelated": {"must": "remain"},
        "hvawt_response_model": {
            "model": "engineering-v8",
            "engineering_lockin": {
                "residual_source_scale": 1.0,
                "response_force_scale": 1.0,
                "velocity_basis": "local_rigid_body_relative_unscaled",
                "unrelated_gain": 17.0,
            },
        },
    }


def _write_force_history(
    path: Path,
    *,
    response: float = 0.0,
    residual: float = 2.0,
    total: float = 2.0,
    generalized: float = 0.0,
    nodal: float = 0.0,
) -> None:
    fields = [
        "response_engineering_maximum_abs_nodal_force_n",
        "response_spectral_applied_residual_maximum_nodal_force_n",
        "response_total_maximum_nodal_force_n",
        "response_engineering_generalized_power_w",
        "response_engineering_nodal_flexible_power_w",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                fields[0]: response,
                fields[1]: residual,
                fields[2]: total,
                fields[3]: generalized,
                fields[4]: nodal,
            }
        )


def _write_grid(path: Path, *, timestep_s: float, rows: int) -> None:
    fields = [
        "step_index",
        "time_s",
        "qblade_timestep",
        "generated_max_abs_force_component_n",
        "applied_max_abs_force_component_n",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(rows):
            writer.writerow(
                {
                    "step_index": index,
                    "time_s": index * timestep_s,
                    "qblade_timestep": float(index),
                    "generated_max_abs_force_component_n": 1.0,
                    "applied_max_abs_force_component_n": 1.0,
                }
            )


@requires_archived_campaign
def test_downloaded_v3_receipts_match_the_frozen_sensitivity_sources():
    expected = (
        (
            DOWNLOADED_V3 / "campaign_manifest.json",
            sensitivity.BASE_CAMPAIGN_FINGERPRINT,
        ),
        (
            DOWNLOADED_V3 / sensitivity.SELECTION_NAME,
            sensitivity.SELECTION_FINGERPRINT,
        ),
        (
            DOWNLOADED_V3 / sensitivity.RECOVERY_NAME,
            sensitivity.RECOVERY_FINGERPRINT,
        ),
        (
            DOWNLOADED_V3 / sensitivity.RECOVERED_ANALYSIS_NAME,
            sensitivity.RECOVERED_ANALYSIS_FINGERPRINT,
        ),
        (
            DOWNLOADED_V3 / sensitivity.PROMOTION_RECEIPT_RELATIVE,
            sensitivity.BASE_PROMOTION_FINGERPRINT,
        ),
    )
    for path, fingerprint in expected:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sensitivity._validate_fingerprint(payload, label=path.name)
        assert payload["fingerprint"] == fingerprint

    promotion = DOWNLOADED_V3 / sensitivity.PROMOTION_RECEIPT_RELATIVE
    assert sensitivity._sha256(promotion) == sensitivity.BASE_PROMOTION_FILE_SHA256
    promotion_payload = json.loads(promotion.read_text(encoding="utf-8"))
    comparison = promotion_payload["source_comparison"]
    assert comparison["sha256_matches_manifest"] is False
    assert comparison["size_matches_manifest"] is False
    assert (
        comparison["manifest_record"]["sha256"]
        == sensitivity.BASE_CAMPAIGN_PAPER_HELPER_SHA256
    )
    assert (
        comparison["live_postcampaign_helper"]["sha256"]
        == sensitivity.BASE_POSTCAMPAIGN_PAPER_HELPER_SHA256
    )
    modal = ROOT / sensitivity.MODAL_QUALIFICATION_RELATIVE
    assert (
        sensitivity._sha256(modal)
        == sensitivity.MODAL_QUALIFICATION_FILE_SHA256
    )


@requires_archived_campaign
def test_base_preflight_accepts_receipt_bound_postcampaign_paper_helper(tmp_path):
    base = tmp_path / "accepted_v3"
    base.mkdir()
    for relative in (
        Path("campaign_manifest.json"),
        Path(sensitivity.SELECTION_NAME),
        Path(sensitivity.RECOVERY_NAME),
        Path(sensitivity.RECOVERED_ANALYSIS_NAME),
        sensitivity.PROMOTION_RECEIPT_RELATIVE,
    ):
        source = DOWNLOADED_V3 / relative
        destination = base / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = json.loads(
        (base / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    for selected in sensitivity.SELECTED_ROWS:
        for case in sensitivity._scenario_cases(
            manifest, str(selected["row_id"])
        ).values():
            case_dir = base / str(case["case_dir"])
            sim = base / str(case["sim"])
            result_dir = base / str(case["result_dir"])
            case_dir.mkdir(parents=True, exist_ok=True)
            sim.parent.mkdir(parents=True, exist_ok=True)
            sim.touch()
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / f"{sim.stem}_run_summary.json").touch()
            (result_dir / f"{case['result_name']}.txt").touch()

    state = sensitivity.validate_base_preflight(base)

    assert state["manifest"]["fingerprint"] == (
        sensitivity.BASE_CAMPAIGN_FINGERPRINT
    )
    comparison = json.loads(
        (base / sensitivity.PROMOTION_RECEIPT_RELATIVE).read_text(
            encoding="utf-8"
        )
    )["source_comparison"]
    assert comparison["sha256_matches_manifest"] is False


@requires_archived_campaign
def test_selected_base_validation_rehashes_only_the_two_used_winds(monkeypatch):
    manifest = json.loads(
        (DOWNLOADED_V3 / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    observed_rows = []

    def accepted_wind_record(row, _approved):
        row_id = str(row["row_id"])
        observed_rows.append(row_id)
        return json.loads(
            (
                DOWNLOADED_V3
                / "inputs/wind_records"
                / f"{row_id}.json"
            ).read_text(encoding="utf-8")
        )

    monkeypatch.setattr(
        sensitivity, "validate_wind_record", accepted_wind_record
    )
    records = sensitivity.validate_selected_base_inputs(
        {
            "base_campaign_root": DOWNLOADED_V3,
            "manifest": manifest,
        }
    )

    expected = [str(row["row_id"]) for row in sensitivity.SELECTED_ROWS]
    assert observed_rows == expected
    assert set(records) == set(expected)


@requires_archived_campaign
def test_selected_base_validation_rejects_external_wind_drift(monkeypatch):
    manifest = json.loads(
        (DOWNLOADED_V3 / "campaign_manifest.json").read_text(encoding="utf-8")
    )

    def changed_wind_record(row, _approved):
        row_id = str(row["row_id"])
        record = json.loads(
            (
                DOWNLOADED_V3
                / "inputs/wind_records"
                / f"{row_id}.json"
            ).read_text(encoding="utf-8")
        )
        record["sha256"] = "0" * 64
        return record

    monkeypatch.setattr(sensitivity, "validate_wind_record", changed_wind_record)
    with pytest.raises(
        sensitivity.HVAWTSensitivityError,
        match="External wind record changed",
    ):
        sensitivity.validate_selected_base_inputs(
            {
                "base_campaign_root": DOWNLOADED_V3,
                "manifest": manifest,
            }
        )


def test_six_case_matrix_is_exact_and_additive():
    assert len(sensitivity.SELECTED_ROWS) == 2
    assert len(sensitivity.VARIANTS) == 3
    assert [row["name"] for row in sensitivity.VARIANTS] == [
        "complete_response_off",
        "complete_half_timestep",
        "complete_lower_cap",
    ]
    assert {row["source_scenario"] for row in sensitivity.VARIANTS} == {
        "complete_hybrid_v8"
    }

    response_off, half, lower = sensitivity.VARIANTS
    assert response_off["force_contract"] == {
        "coupling_mode": "sil-vorlap",
        "legacy_vorlap_force_scale": 0.0,
        "residual_source_scale": 1.0,
        "response_force_scale": 0.0,
    }
    assert response_off["timestep_s"] == sensitivity.TIMESTEP_S
    assert response_off["total_steps"] == 88_000
    assert response_off["retained_rows"] == 80_000
    assert response_off["source_frequency_cap_hz"] == pytest.approx(200.0 / 3.0)

    assert half["force_contract"] == sensitivity.COMPLETE_CONTRACT
    assert half["timestep_s"] == 0.00375
    assert half["total_steps"] == 176_000
    assert half["retained_rows"] == 160_000
    assert half["source_frequency_cap_hz"] == pytest.approx(200.0 / 3.0)

    assert lower["force_contract"] == sensitivity.COMPLETE_CONTRACT
    assert lower["timestep_s"] == sensitivity.TIMESTEP_S
    assert lower["total_steps"] == 88_000
    assert lower["retained_rows"] == 80_000
    assert lower["source_frequency_cap_hz"] == pytest.approx(100.0 / 3.0)
    assert (
        lower["source_frequency_cap_hz"]
        / sensitivity.HIGHEST_REVIEWED_MODE_HZ
        == pytest.approx(7.289810994295803)
    )


def test_orchestration_identities_are_repository_relative_and_portable():
    identities = sensitivity._orchestration_identities(ROOT)
    assert set(identities) == set(sensitivity.ORCHESTRATION_SOURCE_RELATIVE)
    for role, record in identities.items():
        assert record["path"] == sensitivity.ORCHESTRATION_SOURCE_RELATIVE[
            role
        ].as_posix()
        assert not Path(str(record["path"])).is_absolute()
        assert record["sha256"] == sensitivity._sha256(
            ROOT / str(record["path"])
        )


def test_runtime_receipt_rejects_library_or_orchestration_drift(tmp_path):
    path = tmp_path / sensitivity.RUNTIME_RECEIPT_NAME
    receipt = {
        "schema": "vorlap-hvawt-targeted-sensitivity-runtime-v1",
        "base_runtime": {
            "schema": "vorlap-hvawt-complete-dlc-flight-runtime-v1",
            "qblade_library": {"sha256": sensitivity.QBLADE_LIBRARY_SHA256},
            "bridge_library": {"sha256": sensitivity.BRIDGE_LIBRARY_SHA256},
        },
        "sensitivity_orchestration_sources": (
            sensitivity._orchestration_identities(ROOT)
        ),
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert sensitivity._load_runtime(tmp_path) == receipt

    receipt["base_runtime"]["bridge_library"]["sha256"] = "0" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sensitivity.HVAWTSensitivityError, match="invalid"):
        sensitivity._load_runtime(tmp_path)


def test_execution_authorization_binds_all_six_ids_and_exact_smoke(
    tmp_path, monkeypatch
):
    cases = [{"case_id": f"case-{index}"} for index in range(6)]
    manifest = {"fingerprint": "campaign", "cases": cases}
    smoke = {"fingerprint": "smoke"}
    runtime = {"schema": "test-runtime"}
    receipt = {
        "schema": sensitivity.AUTHORIZATION_SCHEMA,
        "status": "authorized_after_6_case_input_smoke",
        "campaign_fingerprint": "campaign",
        "authorized_case_indices": list(range(6)),
        "authorized_case_ids": [case["case_id"] for case in cases],
        "smoke_receipt": {
            "path": sensitivity.SMOKE_RECEIPT_NAME,
            "size_bytes": 1,
            "sha256": "identity",
        },
        "smoke_fingerprint": "smoke",
        "runtime_receipt": {
            "path": sensitivity.RUNTIME_RECEIPT_NAME,
            "size_bytes": 1,
            "sha256": "runtime-identity",
        },
        "runtime_identity_fingerprint": sensitivity.canonical_fingerprint(
            runtime
        ),
        "fresh_start_only": True,
    }
    receipt["fingerprint"] = sensitivity.canonical_fingerprint(receipt)
    (tmp_path / sensitivity.AUTHORIZATION_NAME).write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    monkeypatch.setattr(
        sensitivity,
        "_verify_relative_identity",
        lambda root, record, **_kwargs: (
            Path(root) / str(record["path"])
        ).resolve(),
    )
    monkeypatch.setattr(sensitivity, "_load_runtime", lambda *_args: runtime)
    monkeypatch.setattr(
        sensitivity, "execute_smoke_receipt", lambda *_args, **_kwargs: smoke
    )
    assert sensitivity._load_authorization(tmp_path, manifest) == receipt

    receipt["authorized_case_ids"][-1] = "wrong-case"
    content = dict(receipt)
    content.pop("fingerprint")
    receipt["fingerprint"] = sensitivity.canonical_fingerprint(content)
    (tmp_path / sensitivity.AUTHORIZATION_NAME).write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(sensitivity.HVAWTSensitivityError, match="invalid"):
        sensitivity._load_authorization(tmp_path, manifest)


def test_smoke_receipt_rechecks_artifact_semantics_and_force_partition(
    tmp_path, monkeypatch
):
    runtime = {"schema": "test-runtime", "identity": "bound"}
    runtime_path = tmp_path / sensitivity.RUNTIME_RECEIPT_NAME
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    monkeypatch.setattr(sensitivity, "_load_runtime", lambda *_args: runtime)
    cases = []
    recorded_cases = []
    for case_index in range(6):
        variant = sensitivity.VARIANTS[case_index % len(sensitivity.VARIANTS)]
        case = {
            "index": case_index,
            "case_id": f"case-{case_index}",
            "row_id": f"row-{case_index // 3}",
            "variant": variant["name"],
            "source_scenario": "complete_hybrid_v8",
            "force_contract": variant["force_contract"],
        }
        cases.append(case)
        result_dir = tmp_path / f"smoke-{case_index}"
        result_dir.mkdir()
        summary_path = result_dir / "summary.json"
        result_path = result_dir / "result.txt"
        metadata_path = result_dir / "metadata.json"
        history_path = result_dir / "history.csv"
        response_on = (
            float(variant["force_contract"]["response_force_scale"]) == 1.0
        )
        _write_force_history(
            history_path,
            response=0.1 if response_on else 0.0,
            residual=2.0,
            total=2.0,
            generalized=0.2 if response_on else 0.0,
            nodal=0.2 if response_on else 0.0,
        )
        summary_path.write_text(
            json.dumps(
                {
                    "failed": False,
                    "stopped": False,
                    "exported_results": True,
                    "requested_steps": 4,
                    "completed_steps": 4,
                    "coupling_mode": "sil-vorlap",
                }
            ),
            encoding="utf-8",
        )
        result_path.write_text("finite smoke result\n", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "campaign_fingerprint": "campaign",
                    "case_id": case["case_id"],
                    "case_index": case_index,
                    "row_id": case["row_id"],
                    "variant": case["variant"],
                    "source_scenario": case["source_scenario"],
                    "force_contract": case["force_contract"],
                    "runtime_identity_fingerprint": (
                        sensitivity.canonical_fingerprint(runtime)
                    ),
                    "smoke_only": True,
                }
            ),
            encoding="utf-8",
        )
        force_metrics = sensitivity._read_history_force_metrics(history_path)
        recorded_cases.append(
            {
                "case_index": case_index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "variant": case["variant"],
                "source_scenario": case["source_scenario"],
                "force_contract": case["force_contract"],
                "completed_steps": 4,
                "maximum_residual_force_n": force_metrics[
                    "maximum_residual_force_n"
                ],
                "maximum_response_force_n": force_metrics[
                    "maximum_response_force_n"
                ],
                "maximum_total_force_n": force_metrics["maximum_total_force_n"],
                "force_metrics": force_metrics,
                "summary": sensitivity._relative_identity(summary_path, tmp_path),
                "result": sensitivity._relative_identity(result_path, tmp_path),
                "metadata": sensitivity._relative_identity(
                    metadata_path, tmp_path
                ),
                "history": sensitivity._relative_identity(history_path, tmp_path),
            }
        )
    manifest = {"fingerprint": "campaign", "cases": cases}
    receipt = {
        "schema": sensitivity.SMOKE_SCHEMA,
        "status": "passed_all_6_four_step_input_smokes",
        "campaign_fingerprint": "campaign",
        "runtime_receipt": sensitivity._relative_identity(
            runtime_path, tmp_path
        ),
        "runtime_identity_fingerprint": sensitivity.canonical_fingerprint(
            runtime
        ),
        "case_count": 6,
        "completed_steps_per_case": 4,
        "cases": recorded_cases,
    }
    receipt["fingerprint"] = sensitivity.canonical_fingerprint(receipt)
    receipt_path = tmp_path / sensitivity.SMOKE_RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert sensitivity.execute_smoke_receipt(tmp_path, manifest) == receipt

    metadata_path = tmp_path / str(recorded_cases[0]["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["variant"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    recorded_cases[0]["metadata"] = sensitivity._relative_identity(
        metadata_path, tmp_path
    )
    receipt.pop("fingerprint")
    receipt["fingerprint"] = sensitivity.canonical_fingerprint(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(
        sensitivity.HVAWTSensitivityError, match="artifact semantics changed"
    ):
        sensitivity.execute_smoke_receipt(tmp_path, manifest)


def test_protocol_matches_the_executable_case_matrix_and_claim_boundary():
    protocol = json.loads(
        (ROOT / sensitivity.PROTOCOL_RELATIVE).read_text(encoding="utf-8")
    )
    assert protocol["case_count"] == 6
    assert [row["name"] for row in protocol["variants"]] == [
        row["name"] for row in sensitivity.VARIANTS
    ]
    assert protocol["acceptance_gates"][
        "response_off_effect_closeness_gate_applied"
    ] is False
    assert protocol["acceptance_gates"][
        "response_off_requires_exactly_zero_response_force_and_power"
    ] is True
    assert protocol["acceptance_gates"][
        "nominal_complete_finite_record_gate_required"
    ] is True
    assert protocol["acceptance_gates"][
        "active_mode_classification_consistency_required"
    ] is True
    assert "only when the same reviewed mode exceeds" in protocol[
        "acceptance_gates"
    ]["active_mode_intervariant_frequency_gate_applicability"]
    assert protocol["acceptance_gates"][
        "complete_path_numerical_load_symmetric_log_ratio_limit"
    ] == pytest.approx(math.log1p(sensitivity.LOAD_RELATIVE_LIMIT))
    assert protocol["acceptance_gates"][
        "active_mode_intervariant_frequency_relative_limit"
    ] == sensitivity.MODE_RELATIVE_LIMIT
    assert protocol["acceptance_gates"][
        "active_mode_absolute_frequency_relative_limit"
    ] == sensitivity.MODE_ABSOLUTE_BAND_LIMIT
    ratio_interpretation = protocol["analysis_contract"]["ratio_interpretation"]
    assert "not independent gates" in ratio_interpretation
    assert "matched three-path numerical convergence" in ratio_interpretation
    assert "does not establish full timestep convergence" in protocol[
        "claim_boundary"
    ]
    assert protocol["variants"][0]["residual_source_scale"] == 1.0
    assert protocol["variants"][0]["response_force_scale"] == 0.0
    assert protocol["variants"][0]["velocity_basis"] == (
        "local_rigid_body_relative_unscaled"
    )


def test_selected_nominal_evidence_rejects_identity_or_frozen_del_drift(
    tmp_path, monkeypatch
):
    cases = []
    accepted_cases = []
    recovered_cases = []
    selected_rows = []
    paths_by_index: dict[int, dict[str, Path]] = {}
    expected_dels: dict[tuple[str, str], float] = {}
    first_result: Path | None = None

    for row_number, selected in enumerate(sensitivity.SELECTED_ROWS, start=1):
        row_id = str(selected["row_id"])
        scenario_indices = {
            "raw_qblade": int(selected["complete_case_index"]) - 2,
            "one_way_spectral_vorlap": (
                int(selected["complete_case_index"]) - 1
            ),
            "complete_hybrid_v8": int(selected["complete_case_index"]),
        }
        scenario_ids = {
            scenario: f"{row_id}__{scenario}" for scenario in scenario_indices
        }
        row_dels = {
            "raw": 100.0 * row_number,
            "one_way": 110.0 * row_number,
            "complete": 120.0 * row_number,
        }
        selected_rows.append(
            {
                "row_id": row_id,
                "case_indices": scenario_indices,
                "case_ids": scenario_ids,
                "absolute_del_Nm": row_dels,
            }
        )
        for scenario, case_index in scenario_indices.items():
            case_id = scenario_ids[scenario]
            case = {
                "index": case_index,
                "row_id": row_id,
                "scenario": scenario,
                "case_id": case_id,
            }
            cases.append(case)
            case_root = tmp_path / f"case_{case_index}"
            case_root.mkdir()
            paths = {
                "result": case_root / "result.txt",
                "summary": case_root / "summary.json",
                "metadata": case_root / "metadata.json",
            }
            if scenario != "raw_qblade":
                paths["history"] = case_root / "history.csv"
            for role, path in paths.items():
                path.write_text(
                    f"{row_id}:{scenario}:{role}\n", encoding="utf-8"
                )
            if first_result is None:
                first_result = paths["result"]
            paths_by_index[case_index] = paths
            artifacts = {
                role: sensitivity._relative_identity(path, tmp_path)
                for role, path in paths.items()
            }
            accepted: dict[str, object] = {
                "case_index": case_index,
                "case_id": case_id,
                "row_id": row_id,
                "scenario": scenario,
            }
            if scenario == "raw_qblade":
                accepted["receipt"] = artifacts
            elif scenario == "one_way_spectral_vorlap":
                recovered_cases.append(
                    {
                        "case_index": case_index,
                        "case_id": case_id,
                        "row_id": row_id,
                        "scenario": scenario,
                        "artifacts": artifacts,
                    }
                )
            else:
                accepted["receipt"] = {
                    **artifacts,
                    "coupled_history_audit": {
                        "history": artifacts["history"]
                    },
                }
            accepted_cases.append(accepted)
            del_key = {
                "raw_qblade": "raw",
                "one_way_spectral_vorlap": "one_way",
                "complete_hybrid_v8": "complete",
            }[scenario]
            expected_dels[(row_id, scenario)] = row_dels[del_key]

    manifest = {"cases": cases}
    selection = {"selected_physical_rows": selected_rows}
    recovery = {"cases": accepted_cases, "recovered_cases": recovered_cases}
    monkeypatch.setattr(
        sensitivity,
        "_case_result_paths",
        lambda _root, case: paths_by_index[int(case["index"])],
    )

    def fake_metric(**kwargs):
        scenario = str(kwargs["label"]).removeprefix("accepted_")
        return {"del_Nm": expected_dels[(str(kwargs["row_id"]), scenario)]}

    monkeypatch.setattr(sensitivity, "_load_metric", fake_metric)
    base = {
        "base_campaign_root": tmp_path,
        "manifest": manifest,
        "selection": selection,
        "recovery": recovery,
    }

    evidence = sensitivity.validate_selected_nominal_evidence(base)
    assert set(evidence) == {
        str(row["row_id"]) for row in sensitivity.SELECTED_ROWS
    }

    assert first_result is not None
    original = first_result.read_bytes()
    first_result.write_bytes(original + b"drift")
    with pytest.raises(sensitivity.HVAWTSensitivityError, match="Identity changed"):
        sensitivity.validate_selected_nominal_evidence(base)
    first_result.write_bytes(original)

    changed_selection = json.loads(json.dumps(selection))
    changed_selection["selected_physical_rows"][0]["absolute_del_Nm"][
        "complete"
    ] += 1.0
    with pytest.raises(
        sensitivity.HVAWTSensitivityError,
        match="does not reproduce the frozen receipt",
    ):
        sensitivity.validate_selected_nominal_evidence(
            {**base, "selection": changed_selection}
        )


@requires_archived_campaign
def test_generator_clones_six_cases_without_writing_the_accepted_root(
    tmp_path, monkeypatch
):
    base = tmp_path / "accepted_v3"
    base.mkdir()
    copied_sources = {
        "campaign_manifest.json": DOWNLOADED_V3 / "campaign_manifest.json",
        sensitivity.SELECTION_NAME: DOWNLOADED_V3 / sensitivity.SELECTION_NAME,
        sensitivity.RECOVERY_NAME: DOWNLOADED_V3 / sensitivity.RECOVERY_NAME,
        sensitivity.RECOVERED_ANALYSIS_NAME: (
            DOWNLOADED_V3 / sensitivity.RECOVERED_ANALYSIS_NAME
        ),
    }
    for name, source in copied_sources.items():
        shutil.copy2(source, base / name)
    promotion = base / sensitivity.PROMOTION_RECEIPT_RELATIVE
    promotion.parent.mkdir(parents=True)
    shutil.copy2(
        DOWNLOADED_V3 / sensitivity.PROMOTION_RECEIPT_RELATIVE,
        promotion,
    )
    filter_path = base / "inputs/result.filter"
    filter_path.parent.mkdir()
    filter_path.write_text("filter\n", encoding="utf-8")
    evidence = base / "result_evidence"
    evidence.mkdir()
    for name in ("result.txt", "summary.json", "metadata.json", "history.csv"):
        (evidence / name).write_text(name, encoding="utf-8")

    cases = []
    for selected in sensitivity.SELECTED_ROWS:
        row_id = str(selected["row_id"])
        source_dir = base / "source_cases" / row_id
        source_dir.mkdir(parents=True)
        sim = source_dir / "case.sim"
        sim.write_text(
            "0.0075 RELAXTIMESTEP\n"
            "0.0075 TIMESTEP\n"
            "88000 NUMTIMESTEPS\n"
            "60.0 STOREFROM\n",
            encoding="utf-8",
        )
        config = source_dir / "vorlap_qblade_params.json"
        config.write_text(json.dumps(_runtime_config()), encoding="utf-8")
        (source_dir / "storage.filter").write_text("storage\n", encoding="utf-8")
        for scenario_index, scenario in enumerate(
            ("raw_qblade", "one_way_spectral_vorlap", "complete_hybrid_v8")
        ):
            complete = scenario == "complete_hybrid_v8"
            cases.append(
                {
                    "index": (
                        selected["complete_case_index"]
                        if complete
                        else selected["complete_case_index"] - (2 - scenario_index)
                    ),
                    "row_id": row_id,
                    "seed": selected["seed"],
                    "scenario": scenario,
                    "case_id": f"{row_id}__{scenario}",
                    "case_dir": source_dir.relative_to(base).as_posix(),
                    "sim": sim.relative_to(base).as_posix(),
                    "runtime_config": (
                        config.name if complete else None
                    ),
                    "storage_filter": "storage.filter",
                    "result_dir": "result_evidence",
                    "result_name": "result",
                    "runtime_asset_fingerprint": "assets",
                }
            )
    manifest = {
        "fingerprint": sensitivity.BASE_CAMPAIGN_FINGERPRINT,
        "inputs": {"result_filter": {"path": "inputs/result.filter"}},
        "cases": cases,
    }
    selection = json.loads(
        (base / sensitivity.SELECTION_NAME).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        sensitivity,
        "validate_base_preflight",
        lambda *_args, **_kwargs: {
            "base_campaign_root": base,
            "manifest": manifest,
            "selection": selection,
        },
    )
    selected_validation_calls = []
    monkeypatch.setattr(
        sensitivity,
        "validate_selected_base_inputs",
        lambda *_args, **_kwargs: selected_validation_calls.append(True),
    )
    monkeypatch.setattr(
        sensitivity,
        "validate_selected_nominal_evidence",
        lambda *_args, **_kwargs: {
            str(row["row_id"]): {} for row in sensitivity.SELECTED_ROWS
        },
    )
    monkeypatch.setattr(
        sensitivity,
        "qblade_runtime_asset_fingerprint",
        lambda *_args, **_kwargs: ([], "assets"),
    )
    monkeypatch.setattr(
        sensitivity,
        "_case_result_paths",
        lambda *_args, **_kwargs: {
            "result": evidence / "result.txt",
            "summary": evidence / "summary.json",
            "metadata": evidence / "metadata.json",
            "history": evidence / "history.csv",
        },
    )
    before = sensitivity._tree_inventory(base)
    target = tmp_path / "sensitivity"

    manifest_path = sensitivity.generate_campaign(
        target, base_campaign_root=base, repository=ROOT
    )

    assert manifest_path == target / sensitivity.MANIFEST_NAME
    assert selected_validation_calls == [True]
    assert sensitivity._tree_inventory(base) == before
    state = sensitivity.validate_campaign(target)
    assert state["manifest"]["case_count"] == 6
    protocol = json.loads(
        (ROOT / sensitivity.PROTOCOL_RELATIVE).read_text(encoding="utf-8")
    )
    assert state["manifest"]["acceptance_gates"] == protocol[
        "acceptance_gates"
    ]
    assert state["manifest"]["claim_boundary"] == protocol["claim_boundary"]
    assert [case["index"] for case in state["manifest"]["cases"]] == list(
        range(6)
    )
    for case in state["manifest"]["cases"]:
        assert Path(case["case_dir"]).parts[0] == "cases"
        runtime = json.loads(
            (
                target / str(case["case_dir"]) / str(case["runtime_config"])
            ).read_text(encoding="utf-8")
        )
        engineering = runtime["hvawt_response_model"]["engineering_lockin"]
        assert runtime["source_parameter_dir"] == "."
        assert engineering["residual_source_scale"] == 1.0
        assert engineering["velocity_basis"] == (
            "local_rigid_body_relative_unscaled"
        )


@pytest.mark.parametrize("variant", sensitivity.VARIANTS)
def test_runtime_patch_changes_only_the_declared_controls(tmp_path, variant):
    path = tmp_path / "vorlap_qblade_params.json"
    original = _runtime_config()
    path.write_text(json.dumps(original), encoding="utf-8")

    sensitivity._patch_runtime_config(path, variant=variant)

    patched = json.loads(path.read_text(encoding="utf-8"))
    assert patched["sample_step"] == variant["timestep_s"]
    assert (
        patched["source_frequency_cap_hz"]
        == variant["source_frequency_cap_hz"]
    )
    assert patched["source_parameter_dir"] == "."
    assert patched["force_scale"] == 0.0
    assert patched["unrelated"] == original["unrelated"]
    engineering = patched["hvawt_response_model"]["engineering_lockin"]
    assert engineering["residual_source_scale"] == 1.0
    assert (
        engineering["response_force_scale"]
        == variant["force_contract"]["response_force_scale"]
    )
    assert engineering["velocity_basis"] == (
        "local_rigid_body_relative_unscaled"
    )
    assert engineering["unrelated_gain"] == 17.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cfg: cfg["hvawt_response_model"]["engineering_lockin"].update(
                residual_source_scale=0.0
            ),
            "Source complete runtime contract changed",
        ),
        (
            lambda cfg: cfg["hvawt_response_model"]["engineering_lockin"].update(
                response_force_scale=0.0
            ),
            "Source complete runtime contract changed",
        ),
        (
            lambda cfg: cfg["hvawt_response_model"]["engineering_lockin"].update(
                velocity_basis="inflow_only"
            ),
            "Source complete runtime contract changed",
        ),
    ],
)
def test_runtime_patch_refuses_force_partition_or_velocity_drift(
    tmp_path, mutation, message
):
    path = tmp_path / "vorlap_qblade_params.json"
    config = _runtime_config()
    mutation(config)
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(sensitivity.HVAWTSensitivityError, match=message):
        sensitivity._patch_runtime_config(
            path, variant=sensitivity.VARIANTS[0]
        )


def test_case_tree_inventory_rejects_empty_symlink_and_hardlink_trees(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(sensitivity.HVAWTSensitivityError, match="empty"):
        sensitivity._tree_inventory(empty)

    plain = tmp_path / "plain"
    plain.mkdir()
    source = plain / "source.dat"
    source.write_bytes(b"immutable")
    inventory = sensitivity._tree_inventory(plain)
    assert inventory["source.dat"]["sha256"] == sensitivity._sha256(source)

    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "source.dat").write_bytes(b"immutable")
    os.link(linked / "source.dat", linked / "alias.dat")
    with pytest.raises(sensitivity.HVAWTSensitivityError, match="hard link"):
        sensitivity._tree_inventory(linked)

    symlinked = tmp_path / "symlinked"
    symlinked.mkdir()
    (symlinked / "source.dat").write_bytes(b"immutable")
    (symlinked / "alias.dat").symlink_to(symlinked / "source.dat")
    with pytest.raises(sensitivity.HVAWTSensitivityError, match="symlink"):
        sensitivity._tree_inventory(symlinked)


def test_compact_history_audit_enforces_exact_physical_grid(tmp_path):
    path = tmp_path / "history.csv"
    _write_grid(path, timestep_s=0.5, rows=4)

    audit = sensitivity._audit_compact_history_grid(
        path, timestep_s=0.5, total_steps=4
    )
    assert audit["rows"] == 4
    assert audit["maximum_time_grid_error_s"] == 0.0

    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("2,1.0,2.0", "2,1.1,2.0"), encoding="utf-8")
    with pytest.raises(
        sensitivity.HVAWTSensitivityError, match="physical time grid"
    ):
        sensitivity._audit_compact_history_grid(
            path, timestep_s=0.5, total_steps=4
        )


def test_response_off_history_preserves_residual_and_zeroes_response_work(
    tmp_path,
):
    path = tmp_path / "history.csv"
    _write_force_history(path)

    metrics = sensitivity._read_history_force_metrics(path)

    assert metrics["rows"] == 1
    assert metrics["maximum_residual_force_n"] == 2.0
    assert metrics["maximum_total_force_n"] == 2.0
    assert metrics["maximum_response_force_n"] == 0.0
    assert metrics["maximum_engineering_generalized_power_w"] == 0.0
    assert metrics["maximum_engineering_nodal_power_w"] == 0.0
    assert metrics["maximum_total_residual_force_error_n"] == 0.0
    sensitivity._validate_complete_force_partition_metrics(
        metrics, response_force_scale=0.0, require_response_signal=True
    )

    _write_force_history(path, response=0.1, generalized=0.2, nodal=0.2)
    drifted = sensitivity._read_history_force_metrics(path)
    assert drifted["maximum_response_force_n"] > 0.0
    assert drifted["maximum_engineering_generalized_power_w"] > 0.0
    with pytest.raises(
        sensitivity.HVAWTSensitivityError, match="applied response work"
    ):
        sensitivity._validate_complete_force_partition_metrics(
            drifted, response_force_scale=0.0, require_response_signal=True
        )

    _write_force_history(path, residual=0.0, total=0.0)
    no_residual = sensitivity._read_history_force_metrics(path)
    with pytest.raises(sensitivity.HVAWTSensitivityError, match="lost residual"):
        sensitivity._validate_complete_force_partition_metrics(
            no_residual, response_force_scale=0.0, require_response_signal=True
        )


class _FakeSeries:
    def __init__(self, values: np.ndarray):
        self._values = values

    def to_numpy(self, _dtype):
        return self._values


class _FakeFrame:
    def __init__(self, channel: str, values: np.ndarray):
        self._channel = channel
        self._values = values

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key: str):
        if key != self._channel:
            raise KeyError(key)
        return _FakeSeries(self._values)


@pytest.mark.parametrize(
    ("timestep_s", "retained_rows", "expected_prefix"),
    [(0.0075, 80_000, 64_000), (0.00375, 160_000, 128_000)],
)
def test_metric_windows_are_equal_physical_durations(
    tmp_path, monkeypatch, timestep_s, retained_rows, expected_prefix
):
    path = tmp_path / "result.txt"
    path.write_text("identity", encoding="utf-8")
    channel = "target"
    values = np.resize(np.array([-2.0, 2.0]), retained_rows)
    lengths: list[int] = []
    monkeypatch.setattr(
        sensitivity,
        "read_qblade_results",
        lambda *_args, **_kwargs: _FakeFrame(channel, values),
    )

    def fake_damage(signal, _exponent):
        lengths.append(len(signal))
        return float(len(signal)) * 2.0, 20.0, np.array([]), np.array([])

    monkeypatch.setattr(sensitivity, "rainflow_damage", fake_damage)

    metric = sensitivity._load_metric(
        path=path,
        identity_root=tmp_path,
        channel=channel,
        timestep_s=timestep_s,
        retained_rows=retained_rows,
        label="case",
        row_id="row",
        source_kind="test",
    )

    assert lengths == [retained_rows, expected_prefix]
    assert metric["finite_record_del_relative_change_480_to_600s"] == pytest.approx(
        0.0
    )
    assert metric["monitored_channel_absolute_extreme_480_Nm"] == 2.0
    assert metric["monitored_channel_absolute_extreme_600_Nm"] == 2.0


def test_finite_record_gate_uses_damage_rate_del_and_record_maximum():
    gates: list[dict[str, object]] = []
    sensitivity._append_finite_record_sensitivity_gates(
        gates,
        row_id="row",
        family="complete_path_nominal_finite_record_sensitivity",
        label="nominal_complete",
        metric={
            "finite_record_del_relative_change_480_to_600s": 0.049,
            "finite_record_monitored_extreme_relative_change_480_to_600s": 0.051,
        },
    )

    assert [gate["comparison"] for gate in gates] == [
        "duration_normalized_damage_rate_del_480_to_600s",
        "monitored_record_maximum_magnitude_480_to_600s",
    ]
    assert [gate["passed"] for gate in gates] == [True, False]


def test_modal_gates_record_activity_and_applicability_explicitly():
    nominal = {
        "modes": {
            "6": {
                "maximum_activation": 0.5,
                "dominant_frequency_hz": sensitivity.MODE_FREQUENCIES_HZ[6],
            },
            "12": {
                "maximum_activation": 1.4,
                "dominant_frequency_hz": sensitivity.MODE_FREQUENCIES_HZ[12],
            },
        }
    }
    variants = {
        "complete_half_timestep": {
            "modes": {
                "6": {
                    "maximum_activation": 0.4,
                    "dominant_frequency_hz": sensitivity.MODE_FREQUENCIES_HZ[6],
                },
                "12": {
                    "maximum_activation": 1.3,
                    "dominant_frequency_hz": (
                        1.02 * sensitivity.MODE_FREQUENCIES_HZ[12]
                    ),
                },
            }
        },
        "complete_lower_cap": {
            "modes": {
                "6": {
                    "maximum_activation": 1.1,
                    "dominant_frequency_hz": sensitivity.MODE_FREQUENCIES_HZ[6],
                },
                "12": {
                    "maximum_activation": 0.9,
                    "dominant_frequency_hz": sensitivity.MODE_FREQUENCIES_HZ[12],
                },
            }
        },
    }
    gates: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    sensitivity._append_modal_diagnostics_and_gates(
        gates,
        diagnostics,
        row_id="row",
        nominal_history=nominal,
        variant_histories=variants,
    )

    assert len(diagnostics) == 4
    assert sum(
        row["intervariant_frequency_gate_applied"] is True
        for row in diagnostics
    ) == 1
    lower_mode6 = next(
        row
        for row in diagnostics
        if row["comparison_variant"] == "complete_lower_cap"
        and row["mode"] == 6
    )
    assert lower_mode6["nominal_active"] is False
    assert lower_mode6["variant_active"] is True
    assert lower_mode6["variant_absolute_band_gate_applied"] is True
    assert lower_mode6["intervariant_not_applicable_reason"] == (
        "nominal_below_activation_threshold"
    )
    assert sum(
        gate["metric"] == "nominal_complete_mode12_natural_frequency"
        for gate in gates
    ) == 1
    classification = [
        gate
        for gate in gates
        if gate["family"]
        == "complete_path_active_mode_classification_consistency"
    ]
    assert len(classification) == 4
    assert sum(gate["passed"] is False for gate in classification) == 2


def test_symmetric_ten_percent_gate_is_inclusive_and_fail_closed():
    gates: list[dict[str, object]] = []
    limit = math.log1p(0.1)
    sensitivity._gate(
        gates,
        row_id="row",
        family="test",
        metric="at_limit",
        observed=limit,
        limit=limit,
        comparison="test",
    )
    sensitivity._gate(
        gates,
        row_id="row",
        family="test",
        metric="above_limit",
        observed=limit + 1.0e-12,
        limit=limit,
        comparison="test",
    )
    assert gates[0]["passed"] is True
    assert gates[1]["passed"] is False
    with pytest.raises(sensitivity.HVAWTSensitivityError, match="nonfinite"):
        sensitivity._gate(
            [],
            row_id="row",
            family="test",
            metric="nan",
            observed=math.nan,
            limit=limit,
            comparison="test",
        )


def test_numerical_load_gates_exclude_response_ablation_and_contextual_ratios():
    row_id = "row"
    metrics = {
        (row_id, "nominal_complete"): {
            "del_Nm": 100.0,
            "monitored_channel_absolute_extreme_Nm": 200.0,
        },
        (row_id, "complete_half_timestep"): {
            "del_Nm": 105.0,
            "monitored_channel_absolute_extreme_Nm": 210.0,
        },
        (row_id, "complete_lower_cap"): {
            "del_Nm": 95.0,
            "monitored_channel_absolute_extreme_Nm": 190.0,
        },
    }
    gates: list[dict[str, object]] = []

    sensitivity._append_complete_path_numerical_load_gates(
        gates, row_id=row_id, metric_by_key=metrics
    )

    assert len(gates) == 4
    assert {gate["family"] for gate in gates} == {
        "complete_path_numerical_load_sensitivity"
    }
    assert all("response_off" not in str(gate["metric"]) for gate in gates)
    assert all("ratio" not in str(gate["metric"]) for gate in gates)


def test_command_uses_sealed_case_specific_checkpoint_policy(
    tmp_path, monkeypatch
):
    sim = tmp_path / "case.sim"
    sim.write_text("sim", encoding="utf-8")
    case = {
        "index": 0,
        "sim": sim.relative_to(tmp_path).as_posix(),
        "result_dir": "results/case",
        "result_name": "result",
        "result_filter": "filter.txt",
        "force_contract": sensitivity.COMPLETE_CONTRACT,
        "checkpoint_every": 48_000,
        "progress_every": 2_000,
        "runtime_asset_fingerprint": "assets",
    }
    monkeypatch.setattr(
        sensitivity,
        "validate_campaign",
        lambda *_args, **_kwargs: {
            "campaign_root": tmp_path,
            "manifest": {"cases": [case]},
        },
    )
    monkeypatch.setattr(
        sensitivity,
        "qblade_runtime_asset_fingerprint",
        lambda *_args, **_kwargs: ([], "assets"),
    )

    command = sensitivity.command_for_case(
        tmp_path,
        case_index=0,
        qblade_root=tmp_path,
        python_executable="python",
        threads=112,
    )

    assert command[command.index("--checkpoint-every") + 1] == "48000"
    assert command[command.index("--progress-every") + 1] == "2000"
    assert command[command.index("--coupling-mode") + 1] == "sil-vorlap"
    assert "--sil-vorlap" in command
    assert command[command.index("--sil-vorlap-history-mode") + 1] == "compact"
    with pytest.raises(
        sensitivity.HVAWTSensitivityError, match="Checkpoint interval"
    ):
        sensitivity.command_for_case(
            tmp_path,
            case_index=0,
            qblade_root=tmp_path,
            threads=112,
            checkpoint_every=24_000,
        )


def test_slurm_launchers_separate_prepare_execution_and_analysis():
    prepare = (
        ROOT / "run_headless/submit_hvawt_sensitivity_prepare.sj"
    ).read_text(encoding="utf-8")
    execute = (
        ROOT / "run_headless/submit_hvawt_sensitivity_flight.sj"
    ).read_text(encoding="utf-8")
    analyze = (
        ROOT / "run_headless/submit_hvawt_sensitivity_analysis.sj"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --cpus-per-task=4" in prepare
    assert "hvawt_sensitivity_campaign.py prepare" in prepare
    assert 'mkdir "${INSTALL_LOCK}"' in prepare
    assert 'mv "${PREPARING_ROOT}" "${SENSITIVITY_ROOT}"' in prepare
    assert "#SBATCH --array=0-5%6" in execute
    assert "#SBATCH --exclusive" in execute
    assert "#SBATCH --time=47:59:00" in execute
    assert "hvawt_sensitivity_campaign.py run" in execute
    assert "--checkpoint-every" not in execute
    assert "--progress-every" not in execute
    assert "hvawt_sensitivity_campaign.py analyze" in analyze
    assert "hvawt_sensitivity_campaign.py promote" in analyze
    assert "qblade-root" not in analyze
    assert "hvawt_sensitivity_campaign.py run" not in analyze
    for source in (prepare, execute, analyze):
        assert "set -euo pipefail" in source
        assert "rsync --delete" not in source
        assert " +  --" not in source
        assert not any(line.strip() == "+" for line in source.splitlines())


def test_slurm_launcher_shell_dry_run_passes_exact_cli_argv(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    argv_log = tmp_path / "argv.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"${ARGV_LOG}\"\n"
        "if [[ \"${2:-}\" == prepare ]]; then\n"
        "  args=(\"$@\")\n"
        "  for ((i=0; i<${#args[@]}; i++)); do\n"
        "    if [[ \"${args[$i]}\" == --campaign-root ]]; then\n"
        "      mkdir -p \"${args[$((i + 1))]}\"\n"
        "    fi\n"
        "  done\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    (fake_bin / "uname").write_text(
        "#!/bin/bash\n"
        "if [[ \"${1:-}\" == -s ]]; then echo Linux; else echo x86_64; fi\n",
        encoding="utf-8",
    )
    (fake_bin / "uname").chmod(0o755)
    (fake_bin / "sha256sum").write_text(
        "#!/bin/bash\n"
        f"echo '{sensitivity.QBLADE_LIBRARY_SHA256}  '$1\n",
        encoding="utf-8",
    )
    (fake_bin / "sha256sum").chmod(0o755)
    bash_env = tmp_path / "bash_env"
    bash_env.write_text("module() { return 0; }\n", encoding="utf-8")

    qblade = tmp_path / "qblade"
    (qblade / "Libraries").mkdir(parents=True)
    (qblade / "ControllerFiles").mkdir()
    (qblade / "Binaries/IPCBridge").mkdir(parents=True)
    (qblade / "SIL_Interface").mkdir()
    (qblade / "libQBladeCE_2.0.9.7.so.1.0.0").write_bytes(b"runtime")
    base = tmp_path / "base"
    base.mkdir()
    (base / "campaign_manifest.json").write_text("{}", encoding="utf-8")
    (base / sensitivity.SELECTION_NAME).write_text("{}", encoding="utf-8")
    target = tmp_path / "sensitivity"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "BASH_ENV": str(bash_env),
        "ARGV_LOG": str(argv_log),
        "VORLAP_ROOT": str(ROOT),
        "PYTHON_EXE": str(fake_python),
        "QBLADE_ROOT": str(qblade),
        "BASE_CAMPAIGN_ROOT": str(base),
        "SENSITIVITY_ROOT": str(target),
        "SLURM_JOB_ID": "7001",
        "SLURM_CPUS_PER_TASK": "4",
    }
    subprocess.run(
        ["bash", str(ROOT / "run_headless/submit_hvawt_sensitivity_prepare.sj")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    for name in (
        sensitivity.MANIFEST_NAME,
        sensitivity.RUNTIME_RECEIPT_NAME,
        sensitivity.SMOKE_RECEIPT_NAME,
        sensitivity.AUTHORIZATION_NAME,
    ):
        (target / name).write_text("{}", encoding="utf-8")

    environment.update(
        {
            "SLURM_ARRAY_TASK_ID": "5",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1",
            "SLURM_CPUS_ON_NODE": "112",
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "run_headless/submit_hvawt_sensitivity_flight.sj")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["bash", str(ROOT / "run_headless/submit_hvawt_sensitivity_analysis.sj")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    invocations = argv_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 5
    assert "hvawt_sensitivity_campaign.py prepare --campaign-root" in invocations[0]
    assert " + " not in invocations[0]
    assert "hvawt_sensitivity_campaign.py validate --campaign-root" in invocations[1]
    assert "hvawt_sensitivity_campaign.py run --campaign-root" in invocations[2]
    assert "--case-index 5 --threads 112" in invocations[2]
    assert "hvawt_sensitivity_campaign.py analyze --campaign-root" in invocations[3]
    assert "hvawt_sensitivity_campaign.py promote --campaign-root" in invocations[4]


def test_promotion_is_atomic_idempotent_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    manifest = {"fingerprint": "campaign"}
    monkeypatch.setattr(
        sensitivity,
        "validate_campaign",
        lambda *_args, **_kwargs: {
            "campaign_root": tmp_path,
            "manifest": manifest,
        },
    )
    analysis = {
        "schema": sensitivity.ANALYSIS_SCHEMA,
        "status": (
            "completed_targeted_screen_with_complete_path_numerical_gates_"
            "passed_and_response_ablation_audited"
        ),
        "campaign_fingerprint": "campaign",
        "complete_case_count": 6,
        "expected_case_count": 6,
        "all_execution_audits_passed": True,
        "response_ablation_execution_audits_passed": True,
        "all_complete_path_numerical_gates_passed": True,
        "all_response_ablation_diagnostics_passed": True,
        "all_preregistered_screening_gates_passed": True,
        "response_ablation_effect_closeness_gate_applied": False,
        "passed_gate_count": 1,
        "gate_count": 1,
        "metrics": [{"row_id": "row", "del_Nm": 1.0}],
        "gates": [{"row_id": "row", "passed": True}],
        "ratio_records": [{"row_id": "row", "diagnostic_only": False}],
        "modal_diagnostics": [
            {
                "row_id": "row",
                "mode": 12,
                "intervariant_frequency_gate_applied": True,
            }
        ],
        "modal_applicability": {
            "diagnostic_record_count": 1,
            "intervariant_gate_applied_count": 1,
        },
        "response_attribution": [{"row_id": "row", "effect": 0.1}],
        "claim_boundary": "bounded test claim",
    }
    analysis["fingerprint"] = sensitivity.canonical_fingerprint(analysis)
    analysis_path = tmp_path / sensitivity.ANALYSIS_NAME
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

    first = sensitivity.promote_analysis(tmp_path)
    second = sensitivity.promote_analysis(tmp_path)
    assert second == first
    target = tmp_path / sensitivity.PAPER_EXPORT_NAME
    assert (
        target / "hvawt_sensitivity_modal_diagnostics_v1.csv"
    ).is_file()
    before = {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }

    analysis["metrics"][0]["del_Nm"] = 2.0
    analysis.pop("fingerprint")
    analysis["fingerprint"] = sensitivity.canonical_fingerprint(analysis)
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(
        sensitivity.HVAWTSensitivityError,
        match="differs from this analysis",
    ):
        sensitivity.promote_analysis(tmp_path)
    after = {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_versioned_analysis_json_accepts_only_exact_retry(tmp_path):
    path = tmp_path / sensitivity.ANALYSIS_NAME
    payload = {"schema": "test", "value": 1}

    first = sensitivity._write_new_or_exact_json(
        path,
        payload,
        label="test analysis",
        mismatch_message="version required",
    )
    original = path.read_bytes()
    second = sensitivity._write_new_or_exact_json(
        path,
        payload,
        label="test analysis",
        mismatch_message="version required",
    )
    assert second == first

    with pytest.raises(sensitivity.HVAWTSensitivityError, match="version required"):
        sensitivity._write_new_or_exact_json(
            path,
            {"schema": "test", "value": 2},
            label="test analysis",
            mismatch_message="version required",
        )
    assert path.read_bytes() == original
