import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

import vorlap.hvawt_modal_decay_campaign as campaign


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def generated_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("hvawt_modal_campaign")
    source = root / "source"
    shutil.copytree(REPO_ROOT / "examples/ReferenceHVAWTQBladeModel", source)
    campaign.replace_qblade_keyword_file(
        source / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim",
        "RPMPRESCRIBED",
        1,
    )
    files = {
        path.relative_to(source).as_posix(): _sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    contract = {
        "schema": campaign.SOURCE_CONTRACT_SCHEMA,
        "sim_relative_path": "baseline_10m_HVAWT_reference_turbine_Wwnd.sim",
        "substructure_relative_path": (
            "10m_HVAWT_reference_turbine/Structure/"
            "10m_HVAWT_reference_substructure.str"
        ),
        "main_structural_relative_path": (
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_main.str"
        ),
        "legacy_external_config_relative_path": (
            "10m_HVAWT_reference_turbine/Control/"
            "vorlap_10m_HVAWT_reference_external.json"
        ),
        "blade_strut_station_damping_relative_paths": [
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_blade.str",
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_lower_strut.str",
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_upper_strut.str",
        ],
        "rayleigh_scalar_relative_paths": [
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_blade.str",
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_lower_strut.str",
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_upper_strut.str",
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_torque_tube.str",
            "10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_tower.str",
        ],
        "expected_sim_settings": {
            "PRESCRIBETYPE": "1",
            "RPMPRESCRIBED": "1",
            "TIMESTEP": "0.007500",
        },
        "files_sha256": files,
    }
    contract_path = root / "source_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    qblade_root = root / "qblade"
    (qblade_root / "ControllerFiles").mkdir(parents=True)
    (qblade_root / "Libraries").mkdir()
    (qblade_root / "SIL_Interface").mkdir()
    (qblade_root / "ControllerFiles/libvorlap_qblade_bridge.so").write_bytes(b"bridge")
    (qblade_root / "Libraries/libChronoEngine.so").write_bytes(b"chrono")
    (qblade_root / "SIL_Interface/QBladeLibrary.py").write_text("# wrapper\n")
    (qblade_root / "libQBlade.so").write_bytes(b"qblade")
    output = root / "campaign"
    manifest = campaign.generate_campaign(
        source_root=source,
        source_contract_path=contract_path,
        output_root=output,
        action_shapes_path=REPO_ROOT / "run_headless/config/hvawt_response_mode_shapes.csv",
        force_duals_path=(
            REPO_ROOT / "run_headless/config/hvawt_response_mode_force_duals.csv"
        ),
        qblade_root=qblade_root,
    )
    return {
        "root": output,
        "source": source,
        "contract": contract_path,
        "qblade_root": qblade_root,
        "manifest": manifest,
    }


def test_locked_matrix_is_exactly_twelve_m12_paired_cases():
    cases = campaign.campaign_cases()
    assert len(cases) == 12
    assert {case.variant for case in cases} == set(campaign.VARIANTS)
    assert all(case.target_mode == 12 for case in cases)
    assert all(case.rotor_speed_rpm == 1.0 for case in cases)
    assert all(case.timestep_s == 0.0005 for case in cases)
    assert all(case.total_wrapper_updates == 13602 for case in cases)
    for variant in campaign.VARIANTS:
        pair = [case for case in cases if case.variant == variant]
        assert {case.role for case in pair} == {"baseline", "excited"}
        baseline = next(case for case in pair if case.role == "baseline")
        excited = next(case for case in pair if case.role == "excited")
        assert baseline.generalized_force_amplitude_n == 0.0
        assert excited.nominal_translation_m == 0.001
        assert excited.generalized_force_amplitude_n == pytest.approx(13.00885879625848)
    frequency = campaign.MODAL_FREQUENCIES_HZ[1.0][12]
    assert (campaign.END_TIME_S - campaign.BURST_CYCLES / frequency) * frequency > 20


def test_locked_hann_force_mapping_matches_analytical_and_numerical_sdof():
    stiffness = campaign.NOMINAL_1RPM_STIFFNESS_N_PER_M[12]
    frequency = campaign.MODAL_FREQUENCIES_HZ[1.0][12]
    mass = stiffness / (2.0 * math.pi * frequency) ** 2
    force = campaign.nominal_hann_force_amplitude_n(12, 0.001)
    analytical = campaign.analytical_hann_sdof_realized_amplitude_m(
        stiffness_n_per_m=stiffness,
        force_amplitude_n=force,
    )
    numerical = campaign.numerical_hann_sdof_realized_amplitude_m(
        stiffness_n_per_m=stiffness,
        mass_kg=mass,
        force_amplitude_n=force,
        steps_per_cycle=1000,
    )
    assert analytical == pytest.approx(0.001, rel=1.0e-14)
    assert numerical == pytest.approx(analytical, rel=1.0e-8)


def test_substructure_tail_transforms_preserve_every_unrelated_byte(tmp_path):
    header = (
        b"SUBELEMENTS\n"
        b"c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 "
        b"DIA_[m]  Drag[-]  DAMP[-]\n"
    )
    row = b"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000 1.0 0.02\n"
    original = b"prefix  with spacing\n" + header + row + b"\nSUBMEMBERS\nsuffix\n"
    for variant in ("O", "S", "C20", "all_beta_zero"):
        path = tmp_path / f"{variant}.str"
        path.write_bytes(original)
        audit = campaign.apply_substructure_variant(path, variant)
        assert audit["only_subelements_tail_changed"] is True
        observed = path.read_bytes()
        if variant == "O":
            assert observed == original
        elif variant == "S":
            assert observed == original.replace(
                b"1.000 1.0 0.02\n", b"1.000 0.02 1.0\n"
            )
        elif variant == "C20":
            expected = original.replace(b"Drag[-]  ", b"").replace(b"1.0 ", b"")
            assert observed == expected
        else:
            expected = original.replace(b"Drag[-]  ", b"").replace(
                b"1.0 0.02\n", b"0.00\n"
            )
            assert observed == expected
            assert audit["tail_variant"] == "C20_ZERO"


def test_station_and_scalar_zeroing_preserve_byte_width_and_scope(tmp_path):
    path = tmp_path / "blade.str"
    original = (
        b"0.0011   RAYLEIGHDMP\r\n"
        b"LEN MASS RAYLEIGHDMP_Vec [-]\r\n"
        b"0 1 0.0011\r\n"
        b"1 1 0.00025\r\n"
        b"RGBCOLOR\r\n"
    )
    path.write_bytes(original)
    scalar = campaign.zero_rayleigh_scalar(path)
    station = campaign.zero_rayleigh_station_vector(path)
    observed = path.read_bytes()
    assert len(observed) == len(original)
    assert scalar["old_value"] == "0.0011"
    assert scalar["new_value"] == "0.0000"
    assert station["changed_station_count"] == 2
    assert observed == original.replace(b"0.0011   RAY", b"0.0000   RAY").replace(
        b"0 1 0.0011", b"0 1 0.0000"
    ).replace(b"1 1 0.00025", b"1 1 0.00000")


def test_result_filter_exports_all_positions_velocities_and_timing(generated_campaign):
    case_root = (
        generated_campaign["root"] / "cases/m12_rpm1_dt0500us_O_excited"
    )
    config = json.loads(
        (
            case_root
            / "10m_HVAWT_reference_turbine/Control/"
            "vorlap_10m_HVAWT_reference_external.json"
        ).read_text()
    )
    channels = campaign._result_channels(config)
    assert len(channels) == 238
    assert channels[:4] == [
        campaign.TIME_CHANNEL,
        campaign.TIMESTEP_CHANNEL,
        campaign.AZIMUTH_CHANNEL,
        campaign.RPM_CHANNEL,
    ]
    assert sum("_g Pos." in name for name in channels) == 117
    assert sum("_g Vel." in name for name in channels) == 117
    assert (case_root / "modal_decay_result.filter").read_text().splitlines() == channels


def test_generated_campaign_is_sealed_exact_and_runtime_bound(generated_campaign):
    manifest = campaign.validate_campaign(generated_campaign["root"])
    assert manifest["schema"] == campaign.CAMPAIGN_SCHEMA
    assert manifest["full_case_count"] == 12
    assert manifest["no_inflow_cases"] is True
    assert len(manifest["cases"]) == 12
    assert manifest["qblade_runtime_contract"]["dependency_fingerprint"]
    all_zero = next(
        row
        for row in manifest["cases"]
        if row["case_id"] == "m12_rpm1_dt0500us_all_beta_zero_excited"
    )
    assert len(all_zero["expected_modified_structural_relative_paths"]) == 6
    assert all_zero["declared_free_cycles"] > 20
    assert all_zero["analytical_undamped_realized_translation_m"] == pytest.approx(0.001)
    assert all_zero["numerical_undamped_realized_translation_m"] == pytest.approx(
        0.001, rel=1.0e-8
    )


def test_exact_source_contract_binds_one_rpm_intended_case_when_available():
    contract_path = REPO_ROOT / "run_headless/config/hvawt_modal_decay_source_contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract["expected_sim_settings"]["PRESCRIBETYPE"] == "1"
    assert contract["expected_sim_settings"]["RPMPRESCRIBED"] == "1"
    assert contract["expected_sim_settings"]["TIMESTEP"] == "0.0075"
    assert len(contract["files_sha256"]) == 20
    candidate = REPO_ROOT / contract["source_root_hint"]
    if candidate.is_dir():
        audit = campaign.validate_source_case(candidate, contract_path)
        assert audit["passed"] is True
        assert audit["file_count"] == 20


def test_modal_projection_rotates_back_to_body_frame_and_uses_velocities(
    generated_campaign,
):
    root = generated_campaign["root"]
    case_root = root / "cases/m12_rpm1_dt0500us_O_excited"
    config = json.loads(
        (
            case_root
            / "10m_HVAWT_reference_turbine/Control/"
            "vorlap_10m_HVAWT_reference_external.json"
        ).read_text()
    )
    shape = campaign._load_mode_shape_vector(
        root / "inputs/hvawt_response_mode_shapes.csv",
        rpm=1.0,
        target_mode=12,
        external_config=config,
    )
    time = np.arange(0.0, 6.8001, 0.01)
    omega = 2.0 * math.pi * campaign.MODAL_FREQUENCIES_HZ[1.0][12]
    sigma = 0.05 * omega
    coordinate = 0.001 * np.exp((-sigma + 1j * omega) * time)
    rate = (-sigma + 1j * omega) * coordinate
    basis = np.column_stack((shape.real, -shape.imag))
    body_position = (np.column_stack((coordinate.real, coordinate.imag)) @ basis.T).reshape(
        (-1, 39, 3)
    )
    body_velocity = (np.column_stack((rate.real, rate.imag)) @ basis.T).reshape(
        (-1, 39, 3)
    )
    azimuth = 6.0 * time
    angle = np.deg2rad(azimuth)
    cosine = np.cos(angle)[:, None]
    sine = np.sin(angle)[:, None]
    rotor_omega = 2.0 * math.pi / 60.0
    rotating_velocity = body_velocity.copy()
    rotating_velocity[:, :, 0] -= rotor_omega * body_position[:, :, 1]
    rotating_velocity[:, :, 1] += rotor_omega * body_position[:, :, 0]
    global_position = np.empty_like(body_position)
    global_velocity = np.empty_like(body_velocity)
    for source, destination in (
        (body_position, global_position),
        (rotating_velocity, global_velocity),
    ):
        destination[:, :, 0] = cosine * source[:, :, 0] - sine * source[:, :, 1]
        destination[:, :, 1] = sine * source[:, :, 0] + cosine * source[:, :, 1]
        destination[:, :, 2] = source[:, :, 2]
    recovered, recovered_rate, condition = campaign.project_modal_coordinates(
        global_position_residual_m=global_position.reshape((-1, 117)),
        global_velocity_residual_m_per_s=global_velocity.reshape((-1, 117)),
        azimuth_deg=azimuth,
        mode_shape=shape,
    )
    assert condition < 100
    assert np.max(np.abs(recovered - coordinate)) < 1.0e-14
    assert np.max(np.abs(recovered_rate - rate)) < 1.0e-12
    fit = campaign.fit_modal_decay(
        time,
        recovered,
        target_frequency_hz=campaign.MODAL_FREQUENCIES_HZ[1.0][12],
    )
    assert fit.classification == "interpretable_underdamped"
    assert fit.frequency_hz == pytest.approx(campaign.MODAL_FREQUENCIES_HZ[1.0][12])
    assert fit.damping_ratio == pytest.approx(0.05 / math.sqrt(1.0 + 0.05**2))


def test_rayleigh_classifier_resolves_parser_and_scalar_station_contributions():
    def fit(zeta):
        return campaign.ModalDecayFit(
            classification="interpretable_underdamped",
            frequency_hz=4.57,
            damping_ratio=zeta,
            normalized_rmse=0.01,
            decay_rates_per_s=(0.1, 0.1, 0.1, 0.1),
            peak_translation_m=0.001,
            burst_end_translation_m=0.001,
            available_free_cycles=26.0,
            sample_count=1000,
        )

    result = campaign.classify_rayleigh_discriminator(
        {
            "O": fit(0.031),
            "S": fit(0.10),
            "C20": fit(0.031),
            "C20_station_zero": fit(0.016),
            "C20_scalar_zero": fit(0.021),
            "all_beta_zero": fit(0.001),
        }
    )
    assert result["complete_and_interpretable"] is True
    assert result["selection_allowed"] is True
    assert result["selected_unambiguous_substructure_encoding"] == "C20"
    assert result["parser_semantics"] == "header_21st_token_is_damping"
    assert result["damping_source"] == "both_file_scalar_and_station_vector"
    assert result["file_scalar_removal_effect_damping_ratio"] == pytest.approx(0.010)
    assert result["station_vector_removal_effect_damping_ratio"] == pytest.approx(0.015)
    assert result["substructure_contribution_separately_identified"] is False
    assert result["scalar_station_interaction_separately_identified"] is False


def test_run_case_uses_only_explicit_modal_mode_and_fixed_output_names(
    generated_campaign, monkeypatch, tmp_path
):
    captured = {}
    sentinel = object()

    def fake_run(request):
        captured["request"] = request
        return sentinel

    monkeypatch.setattr(campaign, "run_qblade_case", fake_run)
    result = campaign.run_campaign_case(
        campaign_root=generated_campaign["root"],
        case_id="m12_rpm1_dt0500us_C20_excited",
        qblade_root=generated_campaign["qblade_root"],
        results_root=tmp_path / "results",
    )
    assert result is sentinel
    request = captured["request"]
    assert request.coupling_mode == "sil-modal-burst"
    assert request.sil_modal_burst_plan.name == "modal_burst_plan.json"
    assert request.result_name == "modal_decay_results"
    assert request.steps == 13602
    assert request.allow_partial is False


def test_sealed_campaign_rejects_any_input_byte_change(generated_campaign):
    target = (
        generated_campaign["root"]
        / "cases/m12_rpm1_dt0500us_O_excited/modal_decay_result.filter"
    )
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"unexpected\n")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            campaign.validate_campaign(generated_campaign["root"])
    finally:
        target.write_bytes(original)
    campaign.validate_campaign(generated_campaign["root"])
