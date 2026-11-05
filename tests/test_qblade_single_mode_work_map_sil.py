import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.ehrmann2014_naca0018 import campaign
import vorlap.qblade_single_mode_work_map_sil as work_map_sil
from vorlap.qblade_modal_power_model import ExternalModalPowerTarget
from vorlap.qblade_sil import SilVorLapCoupler, find_qblade_external_parameter_file
from vorlap.qblade_single_mode_work_map_sil import (
    _lift_unit,
    _mode_shape_in_node_order,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_SPECTRUM_SHA256 = (
    "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369"
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("ehrmann-work-map") / "campaign"
    campaign.generate(root, runtime_vorlap_root=REPO_ROOT)
    return root


def _map_payload(*, signed_at_first=0.012, signed_at_last=-0.006):
    axes = {
        "incidence_deg": [-180.0, 180.0],
        "reynolds_number": [1.0e4, 2.0e6],
        "reduced_frequency_k": [0.01, 10.0],
        "amplitude_ratio": [0.0, 0.02, 0.5],
        "direction_deg_mod_180": [0.0, 90.0],
    }
    shape = tuple(len(axis) for axis in axes.values())
    entrainment = np.ones(shape)
    signed = np.empty(shape)
    uncertainty = np.zeros(shape)
    for amplitude_index, value in enumerate((0.0, signed_at_first, signed_at_last)):
        signed[:, :, :, amplitude_index, :] = value
    entrainment[:, :, :, 0, :] = 0.0
    return {
        "schema": "vorlap-airfoil-force-motion-work-map-v1",
        "status": "calibrated",
        "profile": "NACA0018",
        "motion": "prescribed_crossflow_translation",
        "axes": axes,
        "fields": {
            "entrainment_fraction": entrainment.tolist(),
            "reference_available_force_coefficient": np.full(shape, 0.2).tolist(),
            "signed_power_coefficient": signed.tolist(),
            "power_coefficient_standard_uncertainty": uncertainty.tolist(),
        },
        "conventions": {
            "power_coefficient": "mean_incremental_force_dot_velocity/(0.5*rho*U^3*c*span)",
            "power_sign": "positive_flow_to_structure",
            "reduced_frequency": "omega*c/(2*U)",
            "reynolds_interpolation": "linear_in_log_reynolds",
            "amplitude_origin_interpolation": "quadratic_first_interval",
            "out_of_domain": "error",
            "base_model_subtraction": "high_fidelity_minus_exact_runtime_base",
        },
        "sources": [
            {
                "role": "prescribed_motion_high_fidelity",
                "label": "synthetic synchronized force-motion fixture",
                "sha256": "1" * 64,
            },
            {
                "role": "exact_runtime_base",
                "label": "synthetic paired QBlade fixture",
                "sha256": "2" * 64,
            },
            {
                "role": "stationary_force_spectrum",
                "label": "tracked NACA0018 stationary spectrum",
                "sha256": TRACKED_SPECTRUM_SHA256,
            },
        ],
    }


def _write_map(path):
    raw = (
        json.dumps(_map_payload(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _prepare_case(generated, tmp_path, *, modal_enabled=True, h1=0.0):
    del tmp_path
    manifest = json.loads((generated / "campaign_manifest.json").read_text())
    row = manifest["cases"][0]
    source_config_path = generated / row["runtime_config"]
    baseline_path = source_config_path.with_name("baseline_original_config.json")
    if not baseline_path.exists():
        baseline_path.write_bytes(source_config_path.read_bytes())
    config = json.loads(baseline_path.read_text())
    map_path = source_config_path.parent / "synthetic_work_map.json"
    map_bytes = _write_map(map_path)
    config["response_model"]["aerodynamic_damping_gain_h1"] = h1
    config["response_model"]["startup_seed"]["enabled"] = False
    config.pop("single_mode_spectral_lockin")
    config["single_mode_airfoil_work_map"] = {
        "schema_version": 1,
        "type": "single_mode_airfoil_force_motion_work_map_v1",
        "profile": "NACA0018",
        "kinematics": "fixed_nonrotating",
        "work_map": {
            "path": map_path.name,
            "sha256": hashlib.sha256(map_bytes).hexdigest(),
            "confidence_multiplier": 0.0,
            "require_nonpositive_high_amplitude_work": True,
        },
        "static_spectrum": {
            "channel": "CL",
            "force_axis": "wind_axis_lift",
            "amplitude_measure": "rss_nonzero_frequency_harmonics",
            "source_frequency_measure": "energy_centroid",
            "bounds_policy": "raise",
            "source_sha256": TRACKED_SPECTRUM_SHA256,
        },
        "modal_power": {
            "rms_time_constant_s": 0.1,
            "bandpass_damping_ratio": 0.1,
            "max_abs_target_power_w": 1000.0,
            "max_abs_generalized_force_n": 1000.0,
            "max_abs_nodal_force_n": 1000.0,
            "limit_action": "error",
            "enabled": modal_enabled,
        },
        "structural_damping_ratio": row["structural_damping_ratio"],
        "residual_source_scale": 0.0,
    }
    source_config_path.write_text(json.dumps(config, indent=2) + "\n")
    return generated / row["sim"]


def _set_velocity(coupler, value):
    spec = coupler.runtime.layout["velocity"]
    velocity = np.zeros((len(coupler.node_ids), 3))
    velocity[:, 1] = value
    coupler.swap[spec.offset : spec.offset + spec.size] = velocity.reshape(-1)


@pytest.mark.parametrize(
    "angle_deg,expected",
    [
        (0.0, [0.0, 1.0, 0.0]),
        (90.0, [-1.0, 0.0, 0.0]),
        (180.0, [0.0, -1.0, 0.0]),
    ],
)
def test_wind_axis_lift_direction_is_correct_and_rotation_covariant(
    angle_deg, expected
):
    angle = np.deg2rad(angle_deg)
    chord = np.array([[1.0, 0.0, 0.0]])
    normal = np.array([[0.0, 1.0, 0.0]])
    velocity_chord = np.array([np.cos(angle)])
    velocity_normal = np.array([np.sin(angle)])
    lift = _lift_unit(
        chord, normal, velocity_chord, velocity_normal, np.array([1.0])
    )
    np.testing.assert_allclose(lift[0], expected, atol=1e-15)

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = _lift_unit(
        chord @ rotation.T,
        normal @ rotation.T,
        velocity_chord,
        velocity_normal,
        np.array([1.0]),
    )
    np.testing.assert_allclose(rotated, lift @ rotation.T, atol=1e-15)


def test_work_map_adapter_uses_lift_energy_and_injects_positive_modal_power(
    generated, tmp_path
):
    sim = _prepare_case(generated, tmp_path)
    coupler = SilVorLapCoupler(sim, history_mode="off")
    adapter = coupler.hvawt_response_adapter
    assert type(adapter).__name__ == "QBladeSingleModeWorkMapSilAdapter"
    adapter.step(coupler.swap, response_time_s=0.0)
    _set_velocity(coupler, 0.05)
    force = adapter.step(coupler.swap, response_time_s=0.05)
    assert float(np.sum(force.reshape((-1, 3))[:, 1] * 0.05)) > 0.0
    diagnostics = adapter.last_diagnostics
    assert diagnostics.target_aerodynamic_power_w > 0.0
    assert diagnostics.available_lift_coefficient_max > 0.0
    assert diagnostics.incidence_min_deg == pytest.approx(-90.0)
    assert diagnostics.local_amplitude_ratio_max > 0.0
    assert adapter.seed_model.config.aerodynamic_damping_gain_h1 == 0.0
    summary = adapter.diagnostics_summary()
    assert summary["residual_source_scale"] == 0.0
    assert summary["static_spectrum_source_sha256"] == TRACKED_SPECTRUM_SHA256
    assert summary["work_map"]["target_aerodynamic_power_w"] > 0.0


def test_work_map_modal_force_is_energy_conjugate_for_nonmodal_velocity(
    generated, tmp_path
):
    sim = _prepare_case(generated, tmp_path)
    adapter = SilVorLapCoupler(sim, history_mode="off").hvawt_response_adapter
    model = adapter.power_model
    node_ids = adapter.node_ids
    directions = np.zeros((len(node_ids), 3), dtype=float)
    directions[:, 1] = 1.0
    velocities = np.zeros_like(directions)

    omega = 2.0 * np.pi * model.config.natural_frequency_hz

    def target(amplitude_m):
        return ExternalModalPowerTarget(
            target_cycle_mean_power_w=0.5 * 3.0 * omega**2 * amplitude_m**2,
            origin_equivalent_velocity_gain_kgps=3.0,
            context_fingerprint="a" * 64,
        )

    model.step(
        0.0,
        node_ids=node_ids,
        direction_basis=directions,
        structural_velocity_mps=velocities,
        target_power_evaluator=target,
    )
    # Deliberately not proportional to the configured mode: the old
    # span-weighted force dual could give the opposite sign of actual F dot v.
    velocities[:, 1] = np.linspace(1.0, -0.35, len(node_ids))
    result = model.step(
        0.05,
        node_ids=node_ids,
        direction_basis=directions,
        structural_velocity_mps=velocities,
        target_power_evaluator=target,
    )
    diagnostics = result.diagnostics
    actual_power = float(np.sum(result.additive_force_n * velocities))
    conjugate_power = (
        diagnostics.applied_generalized_force_n * diagnostics.raw_qdot_mps
    )
    assert abs(diagnostics.applied_generalized_force_n) > 0.0
    assert actual_power == pytest.approx(conjugate_power, rel=2e-14, abs=1e-14)
    assert diagnostics.instantaneous_structural_power_w == pytest.approx(
        conjugate_power, rel=2e-14, abs=1e-14
    )


def test_work_map_mode_shape_association_is_node_id_invariant(generated, tmp_path):
    sim = _prepare_case(generated, tmp_path)
    adapter = SilVorLapCoupler(sim, history_mode="off").hvawt_response_adapter
    canonical = tuple(adapter.power_model._canonical_node_ids)
    permuted = tuple(reversed(canonical))
    expected = np.asarray(list(reversed(adapter.power_model._mode_shape)))
    np.testing.assert_array_equal(
        _mode_shape_in_node_order(adapter.power_model, permuted), expected
    )
    by_node = dict(zip(canonical, adapter.power_model._mode_shape))
    np.testing.assert_array_equal(
        adapter._mode_shape,
        np.asarray([by_node[node_id] for node_id in adapter.node_ids]),
    )


def test_work_map_adapter_checkpoint_restart_is_exact(generated, tmp_path):
    sim = _prepare_case(generated, tmp_path)
    first = SilVorLapCoupler(sim, history_mode="off")
    first.hvawt_response_adapter.step(first.swap, response_time_s=0.0)
    _set_velocity(first, 0.04)
    first.hvawt_response_adapter.step(first.swap, response_time_s=0.05)
    state = first.hvawt_response_adapter.checkpoint_state()
    _set_velocity(first, -0.03)
    expected = first.hvawt_response_adapter.step(first.swap, response_time_s=0.10)

    restarted = SilVorLapCoupler(sim, history_mode="off")
    restarted.hvawt_response_adapter.restore_checkpoint_state(state)
    _set_velocity(restarted, -0.03)
    observed = restarted.hvawt_response_adapter.step(
        restarted.swap, response_time_s=0.10
    )
    np.testing.assert_array_equal(observed, expected)
    assert (
        restarted.hvawt_response_adapter.history_fields()
        == first.hvawt_response_adapter.history_fields()
    )


def test_work_map_adapter_disabled_path_is_exact_zero(generated, tmp_path):
    sim = _prepare_case(generated, tmp_path, modal_enabled=False)
    coupler = SilVorLapCoupler(sim, history_mode="off")
    adapter = coupler.hvawt_response_adapter
    adapter.step(coupler.swap, response_time_s=0.0)
    _set_velocity(coupler, 0.05)
    force = adapter.step(coupler.swap, response_time_s=0.05)
    np.testing.assert_array_equal(force, np.zeros_like(force))
    assert adapter.last_diagnostics.target_aerodynamic_power_w > 0.0
    assert adapter.last_power_result.diagnostics.applied_target_cycle_mean_power_w == 0.0


def test_work_map_adapter_rejects_cylinder_gain_and_competing_adapters(
    generated, tmp_path
):
    sim = _prepare_case(generated, tmp_path, h1=1.0)
    with pytest.raises(ValueError, match="Cylinder-derived H1"):
        SilVorLapCoupler(sim, history_mode="off")

    sim = _prepare_case(generated, tmp_path)
    config_file = find_qblade_external_parameter_file(sim)
    config = json.loads(config_file.read_text())
    config["single_mode_spectral_lockin"] = {
        "deliberately": "conflicting"
    }
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        SilVorLapCoupler(sim, history_mode="off")


def test_work_map_adapter_rejects_map_config_and_runtime_spectrum_substitution(
    tmp_path, monkeypatch
):
    configured_root = tmp_path / "configured-source-mismatch"
    campaign.generate(configured_root, runtime_vorlap_root=REPO_ROOT)
    sim = _prepare_case(configured_root, tmp_path)
    config_file = find_qblade_external_parameter_file(sim)
    config = json.loads(config_file.read_text())
    config["single_mode_airfoil_work_map"]["static_spectrum"]["source_sha256"] = (
        "0" * 64
    )
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    with pytest.raises(ValueError, match="differs from the work-map source"):
        SilVorLapCoupler(sim, history_mode="off")

    runtime_root = tmp_path / "runtime-source-mismatch"
    campaign.generate(runtime_root, runtime_vorlap_root=REPO_ROOT)
    sim = _prepare_case(runtime_root, tmp_path)
    manifest = json.loads((runtime_root / "campaign_manifest.json").read_text())
    original_loader = work_map_sil.load_airfoil_fft

    def load_with_changed_content(path):
        airfoil = original_loader(path)
        airfoil.CL_Amp = np.array(airfoil.CL_Amp, copy=True)
        airfoil.CL_Amp.reshape(-1)[-1] += 1.0e-12
        return airfoil

    monkeypatch.setattr(work_map_sil, "load_airfoil_fft", load_with_changed_content)
    with pytest.raises(ValueError, match="airfoil content differs"):
        SilVorLapCoupler(sim, history_mode="off")
    monkeypatch.setattr(work_map_sil, "load_airfoil_fft", original_loader)

    source = (
        runtime_root
        / manifest["cases"][0]["case_dir"]
        / "model"
        / "VorLapAirfoils"
        / "NACA0018.h5"
    )
    source.write_bytes(source.read_bytes() + b"substituted")
    with pytest.raises(ValueError, match="Runtime-loaded airfoil source differs"):
        SilVorLapCoupler(sim, history_mode="off")


def test_ehrmann_v2_campaign_reuses_native_contract_and_removes_cylinder_h1(
    tmp_path,
):
    map_path = tmp_path / "work_map.json"
    map_bytes = _write_map(map_path)
    root = tmp_path / "campaign-v2"
    manifest_path = campaign.generate(
        root,
        runtime_vorlap_root=REPO_ROOT,
        work_map=map_path,
    )
    state = campaign.validate(manifest_path)
    manifest = state["manifest"]
    assert manifest["schema"] == campaign.WORK_MAP_SCHEMA
    assert manifest["comparison_series"] == list(campaign.WORK_MAP_SCENARIOS)
    assert manifest["qblade_runtime_identity"] == {
        "version": campaign.FROZEN_QBLADE_VERSION,
        "library_sha256": campaign.FROZEN_QBLADE_LIBRARY_SHA256,
        "runtime_dependency_fingerprint": (
            campaign.FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
        ),
    }
    assert manifest["work_map_identity"]["sha256"] == hashlib.sha256(
        map_bytes
    ).hexdigest()
    assert (
        manifest["work_map_identity"]["stationary_force_spectrum_sha256"]
        == TRACKED_SPECTRUM_SHA256
    )
    assert len(manifest["cases"]) == 40
    pairs = {}
    for row in manifest["cases"]:
        pairs.setdefault(row["native_physical_id"], {})[row["scenario"]] = row
    assert len(pairs) == 20
    for rows in pairs.values():
        raw = rows["raw_qblade"]
        hybrid = rows["latest_airfoil_work_map_bridge"]
        assert raw["native_case_identity"] == hybrid["native_case_identity"]
        for row, enabled in ((raw, False), (hybrid, True)):
            config = json.loads((root / row["runtime_config"]).read_text())
            assert config["response_model"]["aerodynamic_damping_gain_h1"] == 0.0
            assert config["response_model"]["startup_seed"]["enabled"] is enabled
            adapter = config["single_mode_airfoil_work_map"]
            assert adapter["modal_power"]["enabled"] is enabled
            assert adapter["residual_source_scale"] == 0.0
            assert (
                adapter["static_spectrum"]["source_sha256"]
                == TRACKED_SPECTRUM_SHA256
            )
            assert "single_mode_spectral_lockin" not in config

    first = manifest["cases"][0]
    staged_spectrum = (
        root
        / first["case_dir"]
        / "model"
        / "VorLapAirfoils"
        / "NACA0018.h5"
    )
    staged_spectrum.write_bytes(staged_spectrum.read_bytes() + b"changed")
    with pytest.raises(campaign.CampaignError, match="stationary-spectrum identity"):
        campaign.validate(root)
