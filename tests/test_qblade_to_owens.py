from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_headless" / "qblade_to_owens.py"
IDENTIFIER = ROOT / "run_headless" / "identify_hvawt_response_modes.py"
SIM = ROOT / "examples" / "ReferenceHVAWTQBladeModel" / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
EXTERNAL_CONFIG = (
    ROOT
    / "examples"
    / "ReferenceHVAWTQBladeModel"
    / "10m_HVAWT_reference_turbine"
    / "Control"
    / "vorlap_10m_HVAWT_reference_external.json"
)


def load_converter():
    spec = importlib.util.spec_from_file_location("qblade_to_owens", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["qblade_to_owens"] = module
    spec.loader.exec_module(module)
    return module


def load_identifier():
    spec = importlib.util.spec_from_file_location("identify_hvawt_response_modes", IDENTIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["identify_hvawt_response_modes"] = module
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def matrix(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [[float(row[f"{prefix}_{i}{j}"]) for j in range(1, 7)] for i in range(1, 7)],
        dtype=float,
    )


def test_reference_qblade_conversion_preserves_topology_mass_and_inertia(tmp_path):
    converter = load_converter()
    manifest = converter.convert_qblade_model(SIM, tmp_path)

    assert manifest["schema"] == "vorlap-qblade-to-owens-v1"
    assert manifest["qblade"]["number_blades"] == 3
    assert manifest["qblade"]["number_struts_per_blade"] == 2
    assert manifest["qblade"]["hub_inertia_kg_m2"] == pytest.approx(414.5)
    assert manifest["qblade"]["generator_inertia_reflected_kg_m2"] == pytest.approx(0.23)
    assert manifest["mesh"] == {
        "node_count": 182,
        "element_count": 185,
        "physical_element_count": 171,
        "connector_element_count": 14,
        "point_mass_node_count": 20,
        "component_ids": converter.COMPONENT_IDS,
        "kind_ids": converter.KIND_IDS,
        "fixed_nodes": [1],
    }
    assert manifest["mass_checks"]["point_mass_kg"] == pytest.approx((1000.0 + 6 * 15.04))
    # A model replacement must conserve the mass encoded by its own positive
    # sectional matrices, rather than retain a private turbine mass gold value.
    assert manifest["mass_checks"]["total_mass_kg"] > manifest["mass_checks"]["point_mass_kg"]
    assert len(manifest["converter_sha256"]) == 64
    assert all(len(digest) == 64 for digest in manifest["inputs"].values())

    elements = read_rows(tmp_path / "elements.csv")
    integrated_mass = sum(float(row["length_m"]) * float(row["m_11"])
                          for row in elements if int(row["kind_id"]) == converter.KIND_IDS["physical"])
    assert integrated_mass == pytest.approx(manifest["mass_checks"]["distributed_mass_kg"])
    assert integrated_mass + (1000.0 + 6 * 15.04) == pytest.approx(manifest["mass_checks"]["total_mass_kg"])
    kind_counts = {
        name: sum(int(row["kind_id"]) == kind_id for row in elements)
        for name, kind_id in converter.KIND_IDS.items()
    }
    assert kind_counts == {"physical": 171, "rigid": 6, "released": 6, "bearing": 1, "drivetrain": 1}

    for row in elements:
        frame = np.array(
            [[float(row[f"frame_{i}{j}"]) for j in range(1, 4)] for i in range(1, 4)],
            dtype=float,
        )
        assert frame.T @ frame == pytest.approx(np.eye(3), abs=2.0e-12)
        assert np.linalg.det(frame) == pytest.approx(1.0, abs=2.0e-12)
        stiffness = matrix(row, "k")
        assert stiffness == pytest.approx(stiffness.T, abs=1.0e-8)
        assert np.linalg.eigvalsh(stiffness).min() > 0.0
        if int(row["kind_id"]) == converter.KIND_IDS["physical"]:
            mass = matrix(row, "m")
            assert mass == pytest.approx(mass.T, abs=1.0e-10)
            assert np.linalg.eigvalsh(mass).min() > 0.0

    substructure_physical = [
        row
        for row in elements
        if int(row["component_id"]) == converter.COMPONENT_IDS["substructure"]
        and int(row["kind_id"]) == converter.KIND_IDS["physical"]
    ]
    assert len(substructure_physical) == 2
    # Official 2.0.9.7 and archived 2.0.4 semantics both assign zero-based
    # row[19]=DAMP and optional row[20]=drag.  The conflicting source header is
    # retained as a separate unresolved candidate rather than overriding that
    # documented parser contract.
    assert {
        float(row[f"damping_{index}"])
        for row in substructure_physical
        for index in range(1, 7)
    } == {1.0}

    damping_candidates = read_rows(tmp_path / "element_damping_candidates.csv")
    sub_candidates = [
        row
        for row in damping_candidates
        if int(row["component_id"]) == converter.COMPONENT_IDS["substructure"]
        and int(row["kind_id"]) == converter.KIND_IDS["physical"]
    ]
    assert {
        float(row["qblade_2_0_9_7_documented_beta_s"])
        for row in sub_candidates
    } == {1.0}
    assert {
        float(row["reference_header_intended_beta_s"])
        for row in sub_candidates
    } == {0.02}
    assert manifest["damping_semantics"]["selected_damping_candidate_id"] is None
    assert manifest["damping_semantics"]["reference_header_conflict"]["source_rows"][0][
        "raw_text"
    ].strip().endswith("1.0         0.02")

    point_masses = read_rows(tmp_path / "point_masses.csv")
    assert sum(float(row["mass_11"]) for row in point_masses) == pytest.approx((1000.0 + 6 * 15.04))
    assert max(float(row["mass_66"]) for row in point_masses) == pytest.approx(414.73)

    node_metadata = read_rows(tmp_path / "node_metadata.csv")
    assert len(node_metadata) == 182
    assert len({int(row["node_id"]) for row in node_metadata}) == 182
    assert any(row["label"].startswith("blade_1_") for row in node_metadata)
    assert any(row["label"].startswith("strut_2_blade_3_") for row in node_metadata)


def test_released_hinge_regularization_is_independent_of_penalty_factor(tmp_path):
    converter = load_converter()
    low = tmp_path / "low"
    high = tmp_path / "high"
    converter.convert_qblade_model(SIM, low, penalty_factor=1.0e3)
    converter.convert_qblade_model(SIM, high, penalty_factor=1.0e5)

    released_id = converter.KIND_IDS["released"]
    rigid_id = converter.KIND_IDS["rigid"]
    low_rows = read_rows(low / "elements.csv")
    high_rows = read_rows(high / "elements.csv")
    low_released = [matrix(row, "k") for row in low_rows if int(row["kind_id"]) == released_id]
    high_released = [matrix(row, "k") for row in high_rows if int(row["kind_id"]) == released_id]
    assert len(low_released) == len(high_released) == 6
    for low_matrix, high_matrix in zip(low_released, high_released):
        # QBlade releases rY (zero-based DOF 4); its tiny regularization must not
        # grow when the five constrained coordinates are tightened.
        assert low_matrix[4, 4] == pytest.approx(high_matrix[4, 4])
        assert high_matrix[0, 0] / low_matrix[0, 0] == pytest.approx(100.0)

    low_rigid = next(matrix(row, "k") for row in low_rows if int(row["kind_id"]) == rigid_id)
    high_rigid = next(matrix(row, "k") for row in high_rows if int(row["kind_id"]) == rigid_id)
    assert high_rigid[0, 0] / low_rigid[0, 0] == pytest.approx(100.0)


def test_section_mass_uses_rotated_principal_inertia_and_parallel_axis_transport():
    converter = load_converter()
    rho_a = 5.0
    chord = 2.0
    rg_x = 0.2 * chord
    rg_y = 0.3 * chord
    angle = math.radians(30.0)
    mass_y = 0.25
    mass_z = -0.10

    actual = np.array(
        converter._physical_section_mass_matrix(
            rho_a,
            0.2,
            0.3,
            chord,
            math.degrees(angle),
            mass_y,
            mass_z,
        )
    )

    jyy_center = rho_a * (math.cos(angle) ** 2 * rg_x**2 + math.sin(angle) ** 2 * rg_y**2)
    jzz_center = rho_a * (math.sin(angle) ** 2 * rg_x**2 + math.cos(angle) ** 2 * rg_y**2)
    jyz_center = rho_a * math.sin(angle) * math.cos(angle) * (rg_x**2 - rg_y**2)
    expected = np.zeros((6, 6))
    expected[:3, :3] = rho_a * np.eye(3)
    expected[0, 4] = expected[4, 0] = rho_a * mass_z
    expected[0, 5] = expected[5, 0] = -rho_a * mass_y
    expected[1, 3] = expected[3, 1] = -rho_a * mass_z
    expected[2, 3] = expected[3, 2] = rho_a * mass_y
    expected[3, 3] = rho_a * (rg_x**2 + rg_y**2 + mass_y**2 + mass_z**2)
    expected[4, 4] = jyy_center + rho_a * mass_z**2
    expected[5, 5] = jzz_center + rho_a * mass_y**2
    expected[4, 5] = expected[5, 4] = jyz_center - rho_a * mass_y * mass_z

    assert actual == pytest.approx(expected, abs=1.0e-12)
    assert np.linalg.eigvalsh(actual).min() > 0.0


def test_damping_parser_keeps_documented_and_header_candidates_separate():
    converter = load_converter()
    twenty_columns = [float(index) for index in range(20)]
    candidates = converter._subelement_damping_candidates(twenty_columns)
    assert candidates["qblade_2_0_9_7_documented"] == 19.0
    assert candidates["qblade_2_0_4_archived_scalar"] == 19.0
    assert candidates["reference_header_intended"] is None
    assert candidates["legacy_converter_header_scalar"] is None

    twenty_one_columns = [*twenty_columns, 20.0]
    candidates = converter._subelement_damping_candidates(twenty_one_columns)
    assert candidates["qblade_2_0_9_7_documented"] == 19.0
    assert candidates["reference_header_intended"] == 20.0
    with pytest.raises(ValueError, match="20 base numeric entries"):
        converter._subelement_damping_candidates(twenty_columns[:-1])

    structure_dir = (
        ROOT
        / "examples"
        / "ReferenceHVAWTQBladeModel"
        / "10m_HVAWT_reference_turbine"
        / "Structure"
    )
    blade = converter._parse_section_file(structure_dir / "10m_HVAWT_reference_blade.str")
    lower = converter._parse_section_file(structure_dir / "10m_HVAWT_reference_lower_strut.str")
    tower = converter._parse_section_file(structure_dir / "10m_HVAWT_reference_tower.str")
    blade_candidates, blade_source = converter._section_damping_candidates(blade, 0.5)
    lower_candidates, _ = converter._section_damping_candidates(lower, 0.0)
    tower_candidates, tower_source = converter._section_damping_candidates(tower, 0.5)
    assert blade_source["distributed_policy"] == "linear_station_RAYLEIGHDMP_Vec"
    assert blade_candidates["qblade_2_0_9_7_documented"] == pytest.approx(0.0011)
    assert blade_candidates["qblade_2_0_4_archived_scalar"] == pytest.approx(0.0011)
    assert lower_candidates["qblade_2_0_9_7_documented"] == pytest.approx(0.00025)
    assert lower_candidates["qblade_2_0_4_archived_scalar"] == pytest.approx(0.0011)
    assert tower_source["distributed_policy"] == "scalar_RAYLEIGHDMP_fallback_no_station_column"
    assert len(set(tower_candidates.values())) == 1


def test_force_dual_rejects_rank_deficient_action_basis():
    identifier = load_identifier()
    phi = np.ones((117, 12), dtype=complex)
    _, evidence = identifier._minimum_norm_force_dual(
        phi, target_modes=(6, 12), basis_modes=list(range(1, 13))
    )
    assert evidence["rank"] == 1
    assert evidence["passed"] is False
    assert math.isinf(evidence["condition_number"])


def test_action_mapping_rejects_duplicate_or_shifted_output_indices():
    identifier = load_identifier()
    config = json.loads(EXTERNAL_CONFIG.read_text(encoding="utf-8"))
    force_rows = [
        row for row in config["external_output_rows"] if row.get("action") == "ADDFORCE"
    ]
    force_rows[1]["index"] = force_rows[0]["index"]
    with pytest.raises(ValueError, match="unique output indices"):
        identifier._action_sites(config)

    config = json.loads(EXTERNAL_CONFIG.read_text(encoding="utf-8"))
    force_rows = [
        row for row in config["external_output_rows"] if row.get("action") == "ADDFORCE"
    ]
    force_rows[-1]["index"] += 1
    with pytest.raises(ValueError, match="contiguous QBlade output indices"):
        identifier._action_sites(config)


def test_matching_owens_request_requires_complete_artifact_set(tmp_path):
    converter = load_converter()
    expected = {"schema": "test", "rpms": "0,1"}
    (tmp_path / "owens_request.json").write_text(json.dumps(expected), encoding="utf-8")
    for name in (
        "node_metadata.csv",
        "element_damping_candidates.csv",
        "campbell_frequencies.csv",
        "campbell_eigenvalues.csv",
        "campbell_mode_tracking.csv",
        "campbell_mode_shapes.csv",
        "campbell_mode_properties.csv",
        "campbell_mode_energy_by_component.csv",
        "campbell_mode_energy_by_element.csv",
        "campbell_mode_orthogonality.csv",
        "owens_run_metadata.txt",
    ):
        (tmp_path / name).write_text("test\n", encoding="utf-8")
    (tmp_path / "paraview").mkdir()

    assert not converter._request_is_reusable(tmp_path, expected)
    (tmp_path / "paraview" / "converted_qblade_geometry.vtu").write_text("test\n", encoding="utf-8")
    assert converter._request_is_reusable(tmp_path, expected)
    assert not converter._request_is_reusable(tmp_path, {**expected, "rpms": "0,2"})


def test_hvawt_identifier_maps_all_actions_and_normalizes_synthetic_modes(tmp_path):
    identifier = load_identifier()
    modal = tmp_path / "modal"
    modal.mkdir()

    config = json.loads(EXTERNAL_CONFIG.read_text(encoding="utf-8"))
    sites = identifier._action_sites(config)
    components = sorted({str(site["component"]) for site in sites})
    assert len(sites) == 39
    assert len(components) == 9

    node_rows = []
    metadata_rows = []
    component_nodes = {}
    node_id = 1
    for component_index, component in enumerate(components, start=1):
        component_nodes[component] = []
        for fraction in (0.0, 1.0):
            if component.startswith("BLD_"):
                blade = int(component.split("_")[1])
                label = f"blade_{blade}_{fraction:.8f}"
                converted_component = "blade"
            else:
                _, strut, blade = component.split("_")
                label = f"strut_{int(strut)}_blade_{int(blade)}_{fraction:.8f}"
                converted_component = "lower_strut" if int(strut) == 1 else "upper_strut"
            node_rows.append(
                {
                    "node_id": node_id,
                    "x_m": component_index,
                    "y_m": fraction,
                    "z_m": component_index + fraction,
                }
            )
            metadata_rows.append(
                {"node_id": node_id, "component": converted_component, "label": label}
            )
            component_nodes[component].append((fraction, node_id))
            node_id += 1

    def write_dict_rows(path: Path, rows: list[dict[str, object]]):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_dict_rows(modal / "nodes.csv", node_rows)
    write_dict_rows(modal / "node_metadata.csv", metadata_rows)

    frequency_rows = []
    for rpm in (0.0, 1.0):
        row = {"rpm": rpm}
        for mode in range(1, 13):
            base = 0.5 * mode
            row[f"mode_{mode:02d}_hz"] = base + rpm * 0.01 * mode
        frequency_rows.append(row)
    write_dict_rows(modal / "campbell_frequencies.csv", frequency_rows)

    shape_rows = []
    for rpm in (0.0, 1.0):
        for mode in range(1, 13):
            for row in node_rows:
                values: dict[str, object] = {
                    "rpm": rpm,
                    "tracked_mode": mode,
                    "eigen_index": mode,
                    "node_id": row["node_id"],
                    "x_m": row["x_m"],
                    "y_m": row["y_m"],
                    "z_m": row["z_m"],
                }
                for index, name in enumerate(identifier.STATE_NAMES):
                    coordinate = 12 * (int(row["node_id"]) - 1) + index + 1
                    amplitude = (
                        math.sin(0.071 * mode * coordinate)
                        + 0.37 * math.cos(0.113 * (mode + 1) * coordinate)
                        + 0.01 * mode
                    )
                    values[f"{name}_real"] = amplitude
                    values[f"{name}_imag"] = 0.005 * rpm * amplitude
                shape_rows.append(values)
    write_dict_rows(modal / "campbell_mode_shapes.csv", shape_rows)

    property_rows = []
    component_energy_rows = []
    for rpm in (0.0, 1.0):
        for mode in (6, 12):
            frequency = frequency_rows[int(rpm)][f"mode_{mode:02d}_hz"]
            omega = 2.0 * math.pi * frequency
            velocity_mass = 100.0 + mode
            component_terms = ((4, 0.6, 0.001), (5, 0.4, 0.01))
            damping_numerator = velocity_mass * sum(
                elastic_fraction * beta for _, elastic_fraction, beta in component_terms
            )
            property_rows.append(
                {
                    "rpm": rpm,
                    "tracked_mode": mode,
                    "eigen_index": mode,
                    "frequency_hz": frequency,
                    "eigen_residual_relative": 1.0e-12,
                    "left_right_mass_real": 1.0,
                    "left_right_mass_imag": 0.0,
                    "distributed_velocity_mass_raw": velocity_mass - 10.0,
                    "point_velocity_mass_raw": 10.0,
                    "velocity_mass_raw": velocity_mass,
                    "elastic_stiffness_raw": velocity_mass,
                    "rayleigh_damping_raw_s_times_stiffness": damping_numerator,
                    "raw_modal_mass_velocity_over_omega2": velocity_mass / omega**2,
                    "stiffness_to_inertia_ratio": 1.0,
                    "rayleigh_energy_projected_damping_ratio": damping_numerator
                    * omega
                    / (2.0 * velocity_mass),
                }
            )
            for component_id, elastic_fraction, beta in component_terms:
                elastic = velocity_mass * elastic_fraction
                rayleigh = elastic * beta
                component_energy_rows.append(
                    {
                        "rpm": rpm,
                        "tracked_mode": mode,
                        "eigen_index": mode,
                        "component_id": component_id,
                        "elastic_stiffness_raw": elastic,
                        "elastic_stiffness_fraction": elastic_fraction,
                        "rayleigh_damping_raw_s_times_stiffness": rayleigh,
                        "effective_beta_s": beta,
                        "rayleigh_energy_projected_damping_ratio_contribution": rayleigh
                        * omega
                        / (2.0 * velocity_mass),
                    }
                )
    write_dict_rows(modal / "campbell_mode_properties.csv", property_rows)
    write_dict_rows(modal / "campbell_mode_energy_by_component.csv", component_energy_rows)

    element_energy_rows = []
    for rpm in (0.0, 1.0):
        for mode in (6, 12):
            velocity_mass = 100.0 + mode
            for element_id, (component_id, elastic_fraction, _) in enumerate(
                ((4, 0.6, 0.001), (5, 0.4, 0.01)), start=1
            ):
                element_energy_rows.append(
                    {
                        "rpm": rpm,
                        "tracked_mode": mode,
                        "eigen_index": mode,
                        "element_id": element_id,
                        "component_id": component_id,
                        "kind_id": 1,
                        "distributed_velocity_mass_raw": velocity_mass * elastic_fraction,
                        "distributed_velocity_mass_fraction_of_total": elastic_fraction,
                        "elastic_stiffness_raw": velocity_mass * elastic_fraction,
                        "elastic_stiffness_fraction": elastic_fraction,
                    }
                )
    write_dict_rows(modal / "campbell_mode_energy_by_element.csv", element_energy_rows)

    write_dict_rows(
        modal / "elements.csv",
        [
                {
                    "element_id": 1,
                    "component_id": 4,
                    "kind_id": 1,
                **{f"damping_{index}": 0.001 for index in range(1, 7)},
            },
                {
                    "element_id": 2,
                    "component_id": 5,
                    "kind_id": 1,
                **{f"damping_{index}": 0.01 for index in range(1, 7)},
            },
        ],
    )
    write_dict_rows(
        modal / "element_damping_candidates.csv",
        [
            {
                "element_id": 1,
                "component_id": 4,
                "kind_id": 1,
                "source_label": "blade.str",
                "source_table": "section_properties",
                "source_header_line": 1,
                "source_row_line_lower": 2,
                "source_row_line_upper": 3,
                "source_fraction": 0.5,
                "distributed_policy": "linear_station_RAYLEIGHDMP_Vec",
                "scalar_beta_s": 0.002,
                "station_beta_column_zero_based": 18,
                "qblade_2_0_9_7_documented_beta_s": 0.001,
                "reference_header_intended_beta_s": 0.0015,
                "qblade_2_0_4_archived_scalar_beta_s": 0.002,
                "legacy_converter_header_scalar_beta_s": 0.0025,
            },
            {
                "element_id": 2,
                "component_id": 5,
                "kind_id": 1,
                "source_label": "strut.str",
                "source_table": "section_properties",
                "source_header_line": 1,
                "source_row_line_lower": 2,
                "source_row_line_upper": 3,
                "source_fraction": 0.5,
                "distributed_policy": "linear_station_RAYLEIGHDMP_Vec",
                "scalar_beta_s": 0.02,
                "station_beta_column_zero_based": 18,
                "qblade_2_0_9_7_documented_beta_s": 0.01,
                "reference_header_intended_beta_s": 0.015,
                "qblade_2_0_4_archived_scalar_beta_s": 0.02,
                "legacy_converter_header_scalar_beta_s": 0.025,
            },
        ],
    )
    write_dict_rows(modal / "point_masses.csv", [{"node_id": 1, "mass_11": 1.0}])
    write_dict_rows(
        modal / "campbell_mode_orthogonality.csv",
        [
            {
                "rpm": 1.0,
                "mode_a": 6,
                "mode_b": 12,
                "system_left_mass_right_real": 0.0,
                "system_left_mass_right_imag": 0.0,
                "physical_velocity_mass_mac": 0.0,
            }
        ],
    )
    write_dict_rows(
        modal / "campbell_mode_tracking.csv",
        [
            {
                "rpm": 0.0,
                "tracked_mode": 6,
                "reference_cluster": 6,
                "eigen_index": 6,
                "frequency_hz": 3.0,
                "corruption_index": 0.0,
            },
            {
                "rpm": 1.0,
                "tracked_mode": 6,
                "reference_cluster": 6,
                "eigen_index": 6,
                "frequency_hz": 3.06,
                "corruption_index": 0.0,
            },
        ],
    )
    (modal / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "schema": "test",
                "inputs": {"structure.str": "0" * 64},
                "input_sizes_bytes": {"structure.str": 1},
                "mesh": {"component_ids": {"blade": 4, "lower_strut": 5}},
                "damping_semantics": {
                    "selected_damping_candidate_id": None,
                    "candidates": {
                        candidate: {"evidence": "synthetic"}
                        for candidate in identifier.DAMPING_CANDIDATE_IDS
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (modal / "owens_run_metadata.txt").write_text("test\n", encoding="utf-8")
    (modal / "owens_request.json").write_text("{}\n", encoding="utf-8")

    reviewed = tmp_path / "reviewed.csv"
    write_dict_rows(reviewed, frequency_rows)
    output_json = tmp_path / "identified.json"
    output_csv = tmp_path / "shapes.csv"
    result = identifier.identify(
        modal,
        EXTERNAL_CONFIG,
        output_json,
        output_csv,
        rpm=1.0,
        comparison_rpm=0.0,
        modes=(6, 12),
        reviewed_frequencies=reviewed,
    )

    assert result["external_action_mapping"]["site_count"] == 39
    assert result["external_action_mapping"]["all_sites_bracketed"] is True
    assert len(result["modes"]) == 2
    assert all(mode["generalized_modal_mass_kg"] > 0.0 for mode in result["modes"])
    assert all(mode["selected_damping_candidate_id"] is None for mode in result["modes"])
    assert all(mode["rayleigh_energy_projected_damping_ratio"] is None for mode in result["modes"])
    assert all(len(mode["damping_candidates"]) == 4 for mode in result["modes"])
    assert all(
        all(candidate["available"] for candidate in mode["damping_candidates"])
        for mode in result["modes"]
    )
    assert all(
        mode["action_translation_complex_mode_purity"]["single_real_scalar_gate"]["passed"]
        for mode in result["modes"]
    )
    documented = result["modes"][0]["damping_candidates"][0]
    assert {
        row["component"] for row in documented["damping_component_breakdown"]
    } == {"blade", "lower_strut"}
    assert documented["strain_energy_weighted_effective_beta_s"] == pytest.approx(
        0.0046
    )
    rows = read_rows(output_csv)
    assert len(rows) == 2 * 12 * 39
    assert {
        int(row[f"qblade_force_output_index_{direction}"])
        for row in rows
        for direction in ("x", "y", "z")
    } == set(range(120, 237))
    for rpm in (0.0, 1.0):
        for mode in range(1, 13):
            selected = [
                row
                for row in rows
                if int(row["tracked_mode"]) == mode and float(row["rpm"]) == rpm
            ]
            assert len(selected) == 39
            maximum = max(
                math.sqrt(
                    sum(
                        float(row[f"{name}_per_m_real"]) ** 2
                        + float(row[f"{name}_per_m_imag"]) ** 2
                        for name in ("ux", "uy", "uz")
                    )
                )
                for row in selected
            )
            assert maximum == pytest.approx(1.0, abs=2.0e-15)

    dual_rows = read_rows(output_csv.with_name("shapes_force_duals.csv"))
    assert len(dual_rows) == 2 * 2 * 39
    for path in (output_csv, output_csv.with_name("shapes_force_duals.csv")):
        raw = path.read_bytes()
        assert b"\r" not in raw
        assert raw.endswith(b"\n")
    assert all(
        evidence["passed"]
        for evidence in result["minimum_norm_force_duals"]["svd_evidence_by_rpm"]
    )


def test_campaign_runs_owens_preprocess_once_on_array_leader():
    text = (ROOT / "run_headless" / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")

    assert 'RUN_OWENS_PREPROCESS="${RUN_OWENS_PREPROCESS:-auto}"' in text
    assert "run_owens_preprocess_once" in text
    assert "is_array_leader || return 0" in text
    assert 'command+=(--if-available)' in text
    assert 'command+=(--reuse-existing)' in text
    assert '--output-dir "${output_dir}"' in text
