import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts.prepare_qblade_external_case import (
    _backup_existing_paths,
    _geometry_flags_for_requested_nodes,
    _next_backup_path,
    _path_for_config,
    _python_path_entries_for_config,
    _validate_force_scale,
    extract_structural_node_ids,
)
from vorlap.qblade_case_prep import (
    QBladeCasePrepRequest,
    prepare_qblade_external_case,
    validate_component_airfoils,
    validate_generated_external_table,
    validate_qblade_model_references,
)
from vorlap.qblade_runtime import (
    build_external_library_table_spec,
    patch_qblade_structural_definition,
    patch_qblade_turbine_definition,
)
from vorlap.qblade_external_table import build_hvawt_response_table_spec
from vorlap.qblade_conversion import convert_qblade_to_vorlap_inputs
from vorlap.qblade_component_selection import (
    component_span_vectors_global,
    resample_component,
)
from vorlap.force_model import force_from_chord_normal
from vorlap.geometry_frames import prepare_component_frames
from vorlap.qblade_geometry_audit import build_qblade_geometry_audit_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODEL_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "legacy_hvawt_models"


def _write_minimal_qblade_case(base: Path) -> tuple[Path, Path, Path, Path]:
    case_dir = base / "case"
    aero_dir = case_dir / "Aero"
    structure_dir = case_dir / "Structure"
    airfoil_dir = base / "airfoils"
    aero_dir.mkdir(parents=True)
    structure_dir.mkdir(parents=True)
    airfoil_dir.mkdir()

    sim = case_dir / "case.sim"
    trb = case_dir / "case.trb"
    bld = aero_dir / "case.bld"
    structural = structure_dir / "case.str"

    sim.write_text(
        "\n".join(
            [
                "case.trb TURBFILE - turbine file",
                "0.0 INITIAL_AZIMUTH - initial azimuth",
                "0.05 TIMESTEP - timestep",
                "20 NUMTIMESTEPS - steps",
                "1.30 DENSITYAIR - density",
                "1.00E-05 VISCOSITYAIR - kinematic viscosity",
                "4.2 MEANINF - mean inflow",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trb.write_text(
        "\n".join(
            [
                "Aero/case.bld BLADEFILE - blade file",
                "1 NUMBLADES - blades",
                "Structure/case.str STRUCTURALFILE - structural file",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bld.write_text(
        "\n".join(
            [
                "1 NUMBLADES - blades",
                "HEIGHT_[m] CHORD_[m] RADIUS_[m] TOFFSET_[m] TWIST_[deg] CIRCANGLE_[deg] P_AXIS_[-] POLAR_FILE",
                "0.0 0.10 0.50 0.00 0.0 0.0 0.25 Polars/NACA_0018.plr",
                "1.0 0.10 0.50 0.00 0.0 0.0 0.25 Polars/NACA_0018.plr",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    structural.write_text(
        "BLD_1_0.000000\n" "BLD_1_1.000000\n",
        encoding="utf-8",
    )
    _write_airfoil_name(airfoil_dir / "NACA0018.h5", "NACA0018")
    return sim, trb, structural, airfoil_dir


def _write_airfoil_name(path: Path, name: str) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("Airfoilname", data=name.encode("utf-8"))


def test_convert_cylinder_polar_uses_diameter_as_thickness_reference(tmp_path: Path):
    case_dir = tmp_path / "cylinder"
    (case_dir / "Aero").mkdir(parents=True)
    (case_dir / "case.sim").write_text(
        "case.trb TURBFILE\n0.01 TIMESTEP\n10 NUMTIMESTEPS\n5 MEANINF\n",
        encoding="utf-8",
    )
    (case_dir / "case.trb").write_text(
        "Aero/case.bld BLADEFILE\n0 TURBTYPE\n1 NUMBLADES\n",
        encoding="utf-8",
    )
    (case_dir / "Aero" / "case.bld").write_text(
        "HAWT ROTORTYPE\n1 NUMBLADES\n"
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE\n"
        "0.0 0.05 0.0 0.0 0.0 0.5 Polars/CYLINDER.plr\n"
        "1.0 0.05 0.0 0.0 0.0 0.5 Polars/CYLINDER.plr\n",
        encoding="utf-8",
    )

    components, _params, _node_ids = convert_qblade_to_vorlap_inputs(
        str(case_dir / "case.sim"), include_struts=False, include_tower=False
    )

    assert components[0].airfoil_ids == ["CYLINDER", "CYLINDER"]
    np.testing.assert_allclose(components[0].thickness, [1.0, 1.0])


def test_convert_hawt_geometry_uses_span_offsets_and_hub_x_rotation(tmp_path: Path):
    case_dir = tmp_path / "hawt"
    (case_dir / "Aero").mkdir(parents=True)
    sim = case_dir / "case.sim"
    trb = case_dir / "case.trb"
    bld = case_dir / "Aero" / "case.bld"
    sim.write_text(
        "case.trb TURBFILE - turbine\n"
        "0.01 TIMESTEP - step\n10 NUMTIMESTEPS - count\n18 MEANINF - inflow\n",
        encoding="utf-8",
    )
    trb.write_text(
        "Aero/case.bld BLADEFILE - blade\n"
        "0 TURBTYPE - HAWT\n"
        "1 NUMBLADES - count\n"
        "12 TOWERHEIGHT - hub height\n"
        "3 OVERHANG - hub x offset\n",
        encoding="utf-8",
    )
    bld.write_text(
        "HAWT ROTORTYPE - rotor type\n"
        "1 NUMBLADES - count\n"
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE\n"
        "2.0 4.0 10.0 0.1 -0.2 0.50 Polars/FFA-W3-360.plr\n"
        "98.0 1.0 0.0 0.3 -1.2 0.45 Polars/FFA-W3-211.plr\n",
        encoding="utf-8",
    )

    components, params, node_ids = convert_qblade_to_vorlap_inputs(
        str(sim), include_struts=True, include_tower=False
    )

    assert [component.id for component in components] == ["BLD_1"]
    np.testing.assert_allclose(
        components[0].shape_xyz, [[-0.2, 0.1, 2.0], [-1.2, 0.3, 98.0]]
    )
    np.testing.assert_allclose(components[0].translation, [3.0, 0.0, 12.0])
    np.testing.assert_allclose(components[0].rotation, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(params.rotation_axis, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(params.rotation_axis_offset, [3.0, 0.0, 12.0])
    assert components[0].airfoil_ids == ["FFA_W3_360", "FFA_W3_211"]
    np.testing.assert_allclose(components[0].thickness, [0.36, 0.211])
    assert node_ids == ["BLD_1_0.000000", "BLD_1_1.000000"]


def test_convert_hawt_preserves_unique_iea_station_source_ids(tmp_path: Path):
    case_dir = tmp_path / "hawt"
    (case_dir / "Aero").mkdir(parents=True)
    (case_dir / "case.sim").write_text(
        "case.trb TURBFILE\n0.006 TIMESTEP\n10 NUMTIMESTEPS\n18 MEANINF\n",
        encoding="utf-8",
    )
    (case_dir / "case.trb").write_text(
        "Aero/case.bld BLADEFILE\n0 TURBTYPE\n1 NUMBLADES\n",
        encoding="utf-8",
    )
    (case_dir / "Aero" / "case.bld").write_text(
        "HAWT ROTORTYPE\n1 NUMBLADES\n"
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE\n"
        "2.0 4.0 0.0 0.0 0.0 0.5 Polars/IEA10_STATION_00.plr\n"
        "98.0 1.0 0.0 0.0 0.0 0.5 Polars/IEA10-STATION-29.plr\n",
        encoding="utf-8",
    )

    components, _params, _node_ids = convert_qblade_to_vorlap_inputs(
        str(case_dir / "case.sim"), include_struts=False, include_tower=False
    )

    assert components[0].airfoil_ids == ["IEA10_STATION_00", "IEA10_STATION_29"]


def test_convert_hawt_maps_qblade_chord_and_normal_axes_without_swapping(
    tmp_path: Path,
):
    case_dir = tmp_path / "hawt_axes"
    (case_dir / "Aero").mkdir(parents=True)
    sim = case_dir / "case.sim"
    sim.write_text(
        "case.trb TURBFILE\n0.01 TIMESTEP\n10 NUMTIMESTEPS\n18 MEANINF\n",
        encoding="utf-8",
    )
    (case_dir / "case.trb").write_text(
        "Aero/case.bld BLADEFILE\n0 TURBTYPE\n1 NUMBLADES\n",
        encoding="utf-8",
    )
    (case_dir / "Aero" / "case.bld").write_text(
        "HAWT ROTORTYPE\n1 NUMBLADES\n"
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE\n"
        "0.0 0.05 0.0 0.0 0.0 0.5 Polars/CYLINDER.plr\n"
        "1.0 0.05 0.0 0.0 0.0 0.5 Polars/CYLINDER.plr\n",
        encoding="utf-8",
    )

    components, _params, node_ids = convert_qblade_to_vorlap_inputs(
        str(sim), include_struts=False, include_tower=False
    )
    prepare_component_frames(components)
    component = components[0]

    # QBlade +Y_b is chordwise.  VorLap uses -X_b as its normal so that the
    # stored (chord, normal, span) ordering remains right-handed.
    np.testing.assert_allclose(
        component.chord_vector, [[0.0, 1.0, 0.0]] * 2, atol=1.0e-12
    )
    np.testing.assert_allclose(
        component.normal_vector, [[-1.0, 0.0, 0.0]] * 2, atol=1.0e-12
    )
    force = force_from_chord_normal(
        np.array([2.0]),
        np.array([3.0]),
        component.chord_vector[0],
        component.normal_vector[0],
    )
    np.testing.assert_allclose(force, [[-3.0, 2.0, 0.0]], atol=1.0e-12)

    audit_rows = build_qblade_geometry_audit_rows(components, node_ids)
    assert all(row["basis_status"] == "ok" for row in audit_rows)
    assert all(row["basis_determinant"] == pytest.approx(1.0) for row in audit_rows)


def test_resampled_swept_hawt_basis_remains_orthonormal(tmp_path: Path):
    case_dir = tmp_path / "hawt_swept_axes"
    (case_dir / "Aero").mkdir(parents=True)
    sim = case_dir / "case.sim"
    sim.write_text(
        "case.trb TURBFILE\n0.01 TIMESTEP\n10 NUMTIMESTEPS\n18 MEANINF\n",
        encoding="utf-8",
    )
    (case_dir / "case.trb").write_text(
        "Aero/case.bld BLADEFILE\n0 TURBTYPE\n1 NUMBLADES\n",
        encoding="utf-8",
    )
    (case_dir / "Aero" / "case.bld").write_text(
        "HAWT ROTORTYPE\n1 NUMBLADES\n"
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE\n"
        "0.0 0.05 0.0 0.000 0.000 0.5 Polars/CYLINDER.plr\n"
        "0.3 0.05 3.0 0.012 0.018 0.5 Polars/CYLINDER.plr\n"
        "0.7 0.05 6.0 0.026 0.011 0.5 Polars/CYLINDER.plr\n"
        "1.0 0.05 9.0 0.041 -0.004 0.5 Polars/CYLINDER.plr\n",
        encoding="utf-8",
    )

    components, _params, _node_ids = convert_qblade_to_vorlap_inputs(
        str(sim), include_struts=False, include_tower=False
    )
    sampled = resample_component(components[0], [0.0, 0.17, 0.53, 0.86, 1.0])
    prepare_component_frames([sampled])
    spans = component_span_vectors_global(sampled.shape_xyz_global)

    np.testing.assert_allclose(
        np.linalg.norm(sampled.chord_vector, axis=1), 1.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(sampled.normal_vector, axis=1), 1.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.sum(sampled.chord_vector * spans, axis=1), 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.sum(sampled.normal_vector * spans, axis=1), 0.0, atol=1.0e-12
    )
    determinants = np.sum(
        spans * np.cross(sampled.chord_vector, sampled.normal_vector), axis=1
    )
    np.testing.assert_allclose(determinants, 1.0, atol=1.0e-12)


def test_extract_structural_node_ids_filters_expected_targets():
    text = """
BLD_1_0.047438
BLD_1_0.047438
STR_1_2_0.172168
TWR_0.901408 // tower sensor
MOO_1_0.500000
"""
    ids = extract_structural_node_ids(text, include_tower_nodes=False)
    assert ids == ["BLD_1_0.047438", "STR_1_2_0.172168"]

    ids_with_tower = extract_structural_node_ids(text, include_tower_nodes=True)
    assert ids_with_tower == ["BLD_1_0.047438", "STR_1_2_0.172168", "TWR_0.901408"]


def test_validate_qblade_model_references_resolves_required_files(tmp_path):
    sim, trb, structural, _airfoil_dir = _write_minimal_qblade_case(tmp_path)

    result = validate_qblade_model_references(sim)

    assert result.ok is True
    assert result.sim_path == sim.resolve()
    assert result.turbine_path == trb.resolve()
    assert result.blade_path == (sim.parent / "Aero" / "case.bld").resolve()
    assert result.structural_path == structural.resolve()
    assert result.structural_node_ids == ("BLD_1_0.000000", "BLD_1_1.000000")


def test_validate_qblade_model_references_reports_missing_turbine_blade_and_structural_files(
    tmp_path,
):
    sim, trb, structural, _airfoil_dir = _write_minimal_qblade_case(tmp_path)
    trb.unlink()
    result = validate_qblade_model_references(sim)

    assert result.ok is False
    assert result.errors == (f"QBlade .sim TURBFILE does not exist: {trb.resolve()}",)

    trb.write_text(
        "Aero/missing.bld BLADEFILE - blade file\n"
        "Structure/missing.str STRUCTURALFILE - structural file\n",
        encoding="utf-8",
    )
    structural.unlink()
    result = validate_qblade_model_references(sim)

    assert result.ok is False
    assert any(
        "QBlade .trb BLADEFILE does not exist" in error for error in result.errors
    )
    assert any(
        "QBlade .trb STRUCTURALFILE does not exist" in error for error in result.errors
    )


def test_validate_qblade_model_references_rejects_invalid_structural_node_positions(
    tmp_path,
):
    sim, _trb, structural, _airfoil_dir = _write_minimal_qblade_case(tmp_path)
    structural.write_text("BLD_1_1.500000\n", encoding="utf-8")

    result = validate_qblade_model_references(sim)

    assert result.ok is False
    assert any(
        "positions must be finite and within [0, 1]" in error for error in result.errors
    )


def test_validate_qblade_model_references_rejects_duplicate_structural_nodes(tmp_path):
    sim, _trb, structural, _airfoil_dir = _write_minimal_qblade_case(tmp_path)
    structural.write_text(
        "BLD_1_0.000000\n" "BLD_1_0.000000\n" "BLD_1_1.000000\n",
        encoding="utf-8",
    )

    result = validate_qblade_model_references(sim)

    assert result.ok is False
    assert result.structural_node_ids == ("BLD_1_0.000000", "BLD_1_1.000000")
    assert any("Duplicate structural node id(s)" in error for error in result.errors)
    assert any("BLD_1_0.000000" in error for error in result.errors)


def test_validate_generated_external_table_rejects_bad_swap_dimensions():
    table_spec = SimpleNamespace(
        node_ids=("BLD_1_0.000000",),
        swap_size=8,
        input_rows=[(0, "Time [s]"), (1, "Timestep [-]"), (2, "Azimuth [deg]")],
        output_rows=[],
    )

    with pytest.raises(ValueError, match="swap size 8 does not match expected size 9"):
        validate_generated_external_table(("BLD_1_0.000000",), table_spec)


def test_validate_generated_external_table_accepts_hvawt_rpm_position_extension():
    node_ids = tuple(f"BLD_1_{index / 38.0:.6f}" for index in range(39))
    table_spec = build_hvawt_response_table_spec(node_ids)

    validate_generated_external_table(node_ids, table_spec)
    assert table_spec.swap_size == 355


def test_validate_component_airfoils_requires_h5_files_and_records_fallback(tmp_path):
    airfoil_dir = tmp_path / "airfoils"
    airfoil_dir.mkdir()
    component = SimpleNamespace(
        id="BLD_1",
        shape_xyz=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        airfoil_ids=["NACA0018", "missing_airfoil"],
    )

    with pytest.raises(FileNotFoundError, match="No airfoil FFT .h5 files"):
        validate_component_airfoils([component], airfoil_dir)

    _write_airfoil_name(airfoil_dir / "default.h5", "default")

    result = validate_component_airfoils([component], airfoil_dir)

    assert result.available_ids == ("default",)
    assert result.default_source_id == "default"
    assert result.fallback_rows == (
        {
            "component": "BLD_1",
            "node_index": 0,
            "requested_airfoil_id": "NACA0018",
            "fallback_airfoil_id": "default",
            "fallback_source_airfoil_id": "default",
        },
        {
            "component": "BLD_1",
            "node_index": 1,
            "requested_airfoil_id": "missing_airfoil",
            "fallback_airfoil_id": "default",
            "fallback_source_airfoil_id": "default",
        },
    )


def test_prepare_case_records_airfoil_fallback_context_in_runtime_config(tmp_path):
    sim, _trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    (airfoil_dir / "NACA0018.h5").unlink()
    _write_airfoil_name(airfoil_dir / "default.h5", "default")

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            source_bounds_policy="raise",
            source_frequency_cap_hz=50.0,
            include_default_python_path=False,
            write_geometry_audit=False,
            dry_run=True,
        )
    )

    assert result.runtime_config["available_airfoil_ids"] == ["default"]
    assert result.runtime_config["source_bounds_policy"] == "raise"
    assert result.runtime_config["source_frequency_cap_hz"] == 50.0
    assert result.runtime_config["default_airfoil_source_id"] == "default"
    assert result.runtime_config["airfoil_fallbacks"] == [
        {
            "component": "BLD_1",
            "node_index": 0,
            "requested_airfoil_id": "NACA0018",
            "fallback_airfoil_id": "default",
            "fallback_source_airfoil_id": "default",
        },
        {
            "component": "BLD_1",
            "node_index": 1,
            "requested_airfoil_id": "NACA0018",
            "fallback_airfoil_id": "default",
            "fallback_source_airfoil_id": "default",
        },
    ]
    assert (
        result.runtime_config["airfoil_source_manifest"][0]["airfoil_id"] == "default"
    )
    assert len(result.runtime_config["airfoil_source_manifest"][0]["sha256"]) == 64


def test_prepare_case_records_airfoil_source_bounds_and_provenance(tmp_path):
    sim, _trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    source = airfoil_dir / "NACA0018.h5"
    justification = "Conservative Reynolds-independent parked-DLC model"
    with h5py.File(source, "a") as h5:
        h5.create_dataset("Re", data=[3.0e4, 5.0e5, 2.0e6])
        h5.create_dataset("AOA", data=[-180.0, 0.0, 176.0])
        h5.create_dataset("Thickness", data=0.18)
        h5.attrs["generator"] = "scripts/process_naluout_to_h5.py"
        h5.attrs["source_data_sha256"] = "a" * 64
        h5.attrs["reynolds_model"] = "replicated_reynolds_independent"
        h5.attrs["reynolds_replication_justification"] = justification
        h5.attrs["processing_config_json"] = json.dumps(
            {"source_reynolds": [5.0e5], "output_reynolds": [3.0e4, 5.0e5, 2.0e6]}
        )

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            include_default_python_path=False,
            write_geometry_audit=False,
            dry_run=True,
        )
    )

    row = result.runtime_config["airfoil_source_manifest"][0]
    assert row["airfoil_id"] == "NACA0018"
    assert row["file"] == "NACA0018.h5"
    assert row["reynolds"] == pytest.approx([3.0e4, 5.0e5, 2.0e6])
    assert row["reynolds_min"] == pytest.approx(3.0e4)
    assert row["reynolds_max"] == pytest.approx(2.0e6)
    assert row["aoa_min_deg"] == pytest.approx(-180.0)
    assert row["aoa_max_deg"] == pytest.approx(176.0)
    assert row["thickness_ratio"] == pytest.approx(0.18)
    assert row["reynolds_model"] == "replicated_reynolds_independent"
    assert row["reynolds_replication_justification"] == justification
    assert row["processing_config"]["source_reynolds"] == pytest.approx([5.0e5])
    assert len(row["sha256"]) == 64


def test_prepare_case_rejects_empty_generated_node_list_before_patching(
    tmp_path, monkeypatch
):
    sim, _trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    patched: list[str] = []

    monkeypatch.setattr(
        "vorlap.qblade_case_prep.convert_qblade_to_vorlap_inputs",
        lambda *_args, **_kwargs: ([], None, []),
    )
    monkeypatch.setattr(
        "vorlap.qblade_case_prep.patch_qblade_turbine_definition",
        lambda *args, **kwargs: patched.append("turbine") or "",
    )

    with pytest.raises(ValueError, match="external table has no nodes"):
        prepare_qblade_external_case(
            QBladeCasePrepRequest(
                sim_path=sim,
                airfoil_dir=airfoil_dir,
                node_source="converted",
                include_default_python_path=False,
                write_geometry_audit=False,
            )
        )

    assert patched == []


def test_geometry_flags_follow_structural_node_requirements():
    include_struts, include_tower = _geometry_flags_for_requested_nodes(
        ["BLD_1_0.047438", "STR_1_2_0.172168", "TWR_0.901408"],
        include_struts=False,
        include_tower=False,
    )

    assert include_struts is True
    assert include_tower is True


def test_prepare_case_rejects_unlabeled_exaggerated_force_scale():
    _validate_force_scale(0.0)
    _validate_force_scale(1.0)
    _validate_force_scale(10.0, diagnostic_force_scale=True)

    with pytest.raises(ValueError, match="diagnostic-force-scale"):
        _validate_force_scale(10.0)

    with pytest.raises(ValueError, match="finite and non-negative"):
        _validate_force_scale(-1.0)


def test_prepare_case_stores_local_paths_relative_to_config_dir(tmp_path):
    config_dir = tmp_path / "case" / "Control"
    config_dir.mkdir(parents=True)
    local_log = config_dir / "vorlap_qblade_debug.log"
    external_log = tmp_path / "outside.log"

    assert _path_for_config(local_log, config_dir) == "vorlap_qblade_debug.log"
    assert _path_for_config(external_log, config_dir) == str(external_log.resolve())


def test_python_path_entries_for_config_deduplicates_and_relativizes(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "case" / "Control"
    config_dir.mkdir(parents=True)
    local_package = config_dir / "python"
    local_package.mkdir()
    external_package = tmp_path / "external_vorlap"
    external_package.mkdir()
    monkeypatch.chdir(tmp_path)

    entries = _python_path_entries_for_config(
        ["case/Control/python", str(local_package), str(external_package)],
        include_repo_root=False,
        config_dir=config_dir,
    )

    assert entries == ["python", str(external_package.resolve())]


def test_prepare_qblade_external_case_service_writes_runtime_config_and_patches_case(
    tmp_path,
):
    sim, trb, structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            include_default_python_path=False,
            write_geometry_audit=False,
        )
    )

    assert result.written_files == (
        result.parameter_file_path,
        result.turbine_path,
        result.structural_path,
    )
    assert (
        result.parameter_file_path
        == (trb.parent / "Control" / "vorlap_qblade_external.json").resolve()
    )
    assert result.node_ids == ("BLD_1_0.000000", "BLD_1_1.000000")
    assert result.swap_size == 15
    assert result.component_count == 1

    config = json.loads(result.parameter_file_path.read_text(encoding="utf-8"))
    assert config["node_ids"] == list(result.node_ids)
    assert config["python_path_entries"] == []
    assert config["parameter_file"] == "Control/vorlap_qblade_external.json"
    assert config["source_parameter_dir"] == "."

    assert "LIBPARAMETERFILE_1" in trb.read_text(encoding="utf-8")
    structural_text = structural.read_text(encoding="utf-8")
    assert "EXTERNAL_1_IN" in structural_text
    assert "EXTERNAL_1_OUT" in structural_text
    assert "ADDFORCE  BLD_1" in structural_text


def test_prepare_local_inflow_case_writes_matching_position_tables(tmp_path):
    sim, _trb, structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    result = prepare_qblade_external_case(QBladeCasePrepRequest(
        sim_path=sim, airfoil_dir=airfoil_dir,
        use_qblade_local_inflow=True, include_default_python_path=False,
        write_geometry_audit=False,
    ))
    config = json.loads(result.parameter_file_path.read_text(encoding="utf-8"))
    assert config["use_qblade_local_inflow"] is True
    assert result.swap_size == config["swap_size"] == 22
    rows = config["external_input_rows"]
    assert [row["index"] for row in rows[-6:]] == list(range(16, 22))
    structural_text = structural.read_text(encoding="utf-8")
    for row in rows[-6:]:
        assert row["variable"] in structural_text


def test_prepare_qblade_external_case_can_write_absolute_source_parameter_dir(tmp_path):
    sim, trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            include_default_python_path=False,
            absolute_source_parameter_dir=True,
            write_geometry_audit=False,
        )
    )

    config = json.loads(result.parameter_file_path.read_text(encoding="utf-8"))

    assert config["source_parameter_dir"] == str((trb.parent / "Control").resolve())


def test_prepare_qblade_external_case_dry_run_leaves_case_unchanged(tmp_path):
    sim, trb, structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    original_trb = trb.read_text(encoding="utf-8")
    original_structural = structural.read_text(encoding="utf-8")

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            include_default_python_path=False,
            write_geometry_audit=False,
            dry_run=True,
        )
    )

    assert result.dry_run is True
    assert result.written_files == ()
    assert result.parameter_file_path in result.planned_write_paths
    assert not result.parameter_file_path.exists()
    assert trb.read_text(encoding="utf-8") == original_trb
    assert structural.read_text(encoding="utf-8") == original_structural


def test_prepare_hvawt_response_case_stages_validated_modal_artifacts(tmp_path):
    sim, _trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    artifact_dir = Path(__file__).resolve().parents[1] / "run_headless" / "config"
    response_config = json.loads(
        (LEGACY_MODEL_FIXTURES / "hvawt_response_model_v1.json").read_text(
            encoding="utf-8"
        )
    )

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            hvawt_response_model=response_config,
            hvawt_response_artifact_dir=artifact_dir,
            include_default_python_path=False,
            write_geometry_audit=False,
        )
    )

    control_dir = result.parameter_file_path.parent
    expected_names = set(response_config["modal_artifacts"].values())
    staged = {path.name for path in result.written_files if path.parent == control_dir}
    assert expected_names <= staged
    for name in expected_names:
        destination = control_dir / name
        assert destination in result.planned_write_paths
        assert destination.read_bytes() == (artifact_dir / name).read_bytes()


def test_prepare_schema_v3_hvawt_case_stages_bound_engineering_calibration(tmp_path):
    sim, _trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    artifact_dir = Path(__file__).resolve().parents[1] / "run_headless" / "config"
    response_config = json.loads(
        (LEGACY_MODEL_FIXTURES / "hvawt_response_model_v3.json").read_text(
            encoding="utf-8"
        )
    )

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            hvawt_response_model=response_config,
            hvawt_response_artifact_dir=artifact_dir,
            include_default_python_path=False,
            write_geometry_audit=False,
        )
    )

    calibration_name = response_config["engineering_lockin"]["calibration_path"]
    staged = result.parameter_file_path.parent / calibration_name
    assert staged in result.written_files
    assert staged.read_bytes() == (artifact_dir / calibration_name).read_bytes()

    bad = json.loads(json.dumps(response_config))
    bad["engineering_lockin"]["calibration_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="calibration SHA-256"):
        prepare_qblade_external_case(
            QBladeCasePrepRequest(
                sim_path=sim,
                airfoil_dir=airfoil_dir,
                hvawt_response_model=bad,
                hvawt_response_artifact_dir=artifact_dir,
                include_default_python_path=False,
                write_geometry_audit=False,
            )
        )


def test_prepare_schema_v4_hvawt_case_stages_bound_griffin_calibration(tmp_path):
    sim, _trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)
    artifact_dir = Path(__file__).resolve().parents[1] / "run_headless" / "config"
    response_config = json.loads(
        (LEGACY_MODEL_FIXTURES / "hvawt_response_model_v4.json").read_text(
            encoding="utf-8"
        )
    )

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            hvawt_response_model=response_config,
            hvawt_response_artifact_dir=artifact_dir,
            include_default_python_path=False,
            write_geometry_audit=False,
        )
    )

    calibration_name = response_config["griffin_lockin"]["calibration_path"]
    staged = result.parameter_file_path.parent / calibration_name
    assert staged in result.written_files
    assert staged.read_bytes() == (artifact_dir / calibration_name).read_bytes()

    bad = json.loads(json.dumps(response_config))
    bad["griffin_lockin"]["calibration_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="calibration SHA-256"):
        prepare_qblade_external_case(
            QBladeCasePrepRequest(
                sim_path=sim,
                airfoil_dir=airfoil_dir,
                hvawt_response_model=bad,
                hvawt_response_artifact_dir=artifact_dir,
                include_default_python_path=False,
                write_geometry_audit=False,
            )
        )


def test_prepare_qblade_external_case_passes_include_mean_load(tmp_path):
    sim, _trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            include_mean_load=True,
            include_default_python_path=False,
            write_geometry_audit=False,
        )
    )

    config = json.loads(result.parameter_file_path.read_text(encoding="utf-8"))

    assert config["include_mean_load"] is True


def test_prepare_qblade_external_case_passes_viv_parameter_overrides(tmp_path):
    sim, _trb, _structural, airfoil_dir = _write_minimal_qblade_case(tmp_path)

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim,
            airfoil_dir=airfoil_dir,
            fluid_density=1.3,
            fluid_dynamicviscosity=1.7e-5,
            rotation_axis=(0.0, 0.0, 1.0),
            rotation_axis_offset=(1.0, 2.0, 3.0),
            inflow_vec=(1.0, 1.0, 0.0),
            include_default_python_path=False,
            write_geometry_audit=False,
        )
    )

    config = json.loads(result.parameter_file_path.read_text(encoding="utf-8"))

    assert config["fluid_density"] == 1.3
    assert config["fluid_dynamicviscosity"] == 1.7e-5
    assert config["rotation_axis"] == [0.0, 0.0, 1.0]
    assert config["rotation_axis_offset"] == [1.0, 2.0, 3.0]
    assert config["inflow_vec"] == [1.0, 1.0, 0.0]


def test_prepare_cli_backup_helpers_copy_existing_files_without_overwriting_backups(
    tmp_path,
):
    target = tmp_path / "case.trb"
    missing = tmp_path / "missing.str"
    first_backup = tmp_path / "case.trb.vorlap.bak"
    target.write_text("original turbine\n", encoding="utf-8")
    first_backup.write_text("previous backup\n", encoding="utf-8")

    assert _next_backup_path(target) == tmp_path / "case.trb.vorlap.bak.1"

    backups = _backup_existing_paths((target, missing))

    assert backups == [(target, tmp_path / "case.trb.vorlap.bak.1")]
    assert target.read_text(encoding="utf-8") == "original turbine\n"
    assert first_backup.read_text(encoding="utf-8") == "previous backup\n"
    assert (tmp_path / "case.trb.vorlap.bak.1").read_text(
        encoding="utf-8"
    ) == "original turbine\n"
    assert not (tmp_path / "missing.str.vorlap.bak").exists()


def test_structural_patch_is_idempotent_and_preserves_positions():
    base = """TWR_1.0\n"""
    spec = build_external_library_table_spec(["BLD_1_0.047438", "STR_2_3_0.174510"])

    first = patch_qblade_structural_definition(base, table_spec=spec)
    second = patch_qblade_structural_definition(first, table_spec=spec)

    assert first == second
    assert "EXTERNAL_1_IN" in second
    assert "EXTERNAL_1_OUT" in second
    assert '"X_g Vel. BLD_1 pos 0.047 [m/s]"' in second
    assert "ADDFORCE  BLD_1" in second
    assert "0.047438" in second
    assert "ADDFORCE  STR_2_3" in second
    assert "0.174510" in second


def test_structural_patch_preserves_content_after_external_tables():
    base = """TWR_1.0

EXTERNAL_1_IN
SWAP DATA
0    "Time [s]"

EXTERNAL_1_OUT
SWAP  ACTION     ID     POS  DIR  LOCAL
3    ADDFORCE  BLD_1    0.500000 X    false

CUSTOM_TRAILER
42 KEEP_ME
"""
    spec = build_external_library_table_spec(["BLD_1_0.047438"])

    patched = patch_qblade_structural_definition(base, table_spec=spec)
    second = patch_qblade_structural_definition(patched, table_spec=spec)

    assert patched == second
    assert "CUSTOM_TRAILER\n42 KEEP_ME" in patched
    assert patched.count("EXTERNAL_1_IN") == 1
    assert patched.count("EXTERNAL_1_OUT") == 1
    assert '"X_g Vel. BLD_1 pos 0.047 [m/s]"' in patched
    assert "0.500000 X" not in patched


def test_turbine_patch_is_idempotent():
    text = """----------------------------------------Turbine Controller-----------------------------------------------------------
1                                                  CONTROLLERTYPE     - the type of turbine controller 0 = none, 1 = BLADED, 2 = DTU, 3 = TUB
libexample_controller                            CONTROLLERFILE     - the controller file name, WITHOUT file ending (.dll or .so ) - leave blank if unused
Control/example_controller.in                      PARAMETERFILE      - the controller parameter file name (leave blank if unused)
"""
    first = patch_qblade_turbine_definition(
        text,
        library_stem="libvorlap_qblade_bridge",
        function_name="update",
        swap_size=237,
        parameter_file="Control/vorlap_qblade_external.json",
    )
    second = patch_qblade_turbine_definition(
        first,
        library_stem="libvorlap_qblade_bridge",
        function_name="update",
        swap_size=237,
        parameter_file="Control/vorlap_qblade_external.json",
    )

    assert first == second
    assert "0                                                 CONTROLLERTYPE" in second
    assert "LIBARRAYSIZE_1" in second
    assert "237" in second


def test_turbine_patch_preserves_sections_after_external_libraries():
    text = """----------------------------------------Turbine Controller-----------------------------------------------------------
1                                                  CONTROLLERTYPE     - the type of turbine controller 0 = none, 1 = BLADED, 2 = DTU, 3 = TUB
libexample_controller                            CONTROLLERFILE     - the controller file name, WITHOUT file ending (.dll or .so ) - leave blank if unused
Control/example_controller.in                      PARAMETERFILE      - the controller parameter file name (leave blank if unused)
----------------------------------------External Libraries-----------------------------------------------------------
old                                                LIBFILE_1          - old library
old_update                                         LIBFUNCTION_1      - old function
----------------------------------------Custom Section-----------------------------------------------------------
99                                                 CUSTOMKEY          - keep this later section
"""

    patched = patch_qblade_turbine_definition(
        text,
        library_stem="libvorlap_qblade_bridge",
        function_name="update",
        swap_size=237,
        parameter_file="Control/vorlap_qblade_external.json",
    )
    second = patch_qblade_turbine_definition(
        patched,
        library_stem="libvorlap_qblade_bridge",
        function_name="update",
        swap_size=237,
        parameter_file="Control/vorlap_qblade_external.json",
    )

    assert patched == second
    assert "old_update" not in patched
    assert "Custom Section" in patched
    assert "CUSTOMKEY" in patched
    assert patched.count("External Libraries") == 1
