import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from vorlap.fileio import (
    compare_qblade_loading_file,
    convert_qblade_to_vorlap_inputs,
    load_airfoil_coords,
    load_airfoil_fft,
    load_components_from_csv,
    load_inflow_time_series,
    load_qblade_blade_definition,
    load_qblade_simulation_definition,
    load_qblade_turbine_definition,
    read_qblade_loading_file,
    write_components_to_csv,
    write_force_time_series,
    write_qblade_loading_file,
)
from vorlap import airfoil_io, component_io, inflow_io, qblade_conversion, qblade_loading_file, qblade_parser


def test_qblade_loading_file_helpers_are_reexported_from_fileio():
    assert write_qblade_loading_file is qblade_loading_file.write_qblade_loading_file
    assert read_qblade_loading_file is qblade_loading_file.read_qblade_loading_file
    assert compare_qblade_loading_file is qblade_loading_file.compare_qblade_loading_file


def test_component_csv_helpers_are_reexported_from_fileio():
    assert write_components_to_csv is component_io.write_components_to_csv
    assert load_components_from_csv is component_io.load_components_from_csv


def test_inflow_csv_helpers_are_reexported_from_fileio():
    assert load_inflow_time_series is inflow_io.load_inflow_time_series
    assert write_force_time_series is inflow_io.write_force_time_series


def test_qblade_parser_and_conversion_helpers_are_reexported_from_fileio():
    assert load_qblade_simulation_definition is qblade_parser.load_qblade_simulation_definition
    assert load_qblade_turbine_definition is qblade_parser.load_qblade_turbine_definition
    assert load_qblade_blade_definition is qblade_parser.load_qblade_blade_definition
    assert convert_qblade_to_vorlap_inputs is qblade_conversion.convert_qblade_to_vorlap_inputs


def test_airfoil_helpers_are_reexported_from_fileio():
    assert load_airfoil_fft is airfoil_io.load_airfoil_fft
    assert load_airfoil_coords is airfoil_io.load_airfoil_coords


def _write_component_csv(path: Path, include_offset: bool = True) -> None:
    columns = ["x", "y", "z", "chord", "twist", "thickness"]
    if include_offset:
        columns.append("offset")
    columns.append("airfoil_id")

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["meta_header"])
        writer.writerow(["blade1", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        writer.writerow(columns)
        row = [0.0, 0.0, 0.0, 1.2, 3.0, 0.12]
        if include_offset:
            row.append(0.05)
        row.append("default")
        writer.writerow(row)


def test_load_components_from_csv_parses_component(tmp_path: Path):
    comp_dir = tmp_path / "components"
    comp_dir.mkdir()
    _write_component_csv(comp_dir / "blade.csv")

    components = load_components_from_csv(str(comp_dir))
    assert len(components) == 1
    comp = components[0]
    assert comp.id == "blade1"
    np.testing.assert_allclose(comp.translation, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(comp.rotation, np.array([4.0, 5.0, 6.0]))
    assert comp.airfoil_ids == ["default"]


def test_write_components_to_csv_roundtrip(tmp_path: Path):
    comp_dir = tmp_path / "components_in"
    comp_dir.mkdir()
    _write_component_csv(comp_dir / "blade.csv")
    components = load_components_from_csv(str(comp_dir))

    out_dir = tmp_path / "components_out"
    written = write_components_to_csv(str(out_dir), components)

    assert len(written) == 1
    assert Path(written[0]).exists()

    reloaded = load_components_from_csv(str(out_dir))
    assert len(reloaded) == 1
    np.testing.assert_allclose(reloaded[0].shape_xyz, components[0].shape_xyz)
    np.testing.assert_allclose(reloaded[0].chord, components[0].chord)
    np.testing.assert_allclose(reloaded[0].twist, components[0].twist)
    np.testing.assert_allclose(reloaded[0].thickness, components[0].thickness)
    np.testing.assert_allclose(reloaded[0].offset, components[0].offset)
    assert reloaded[0].airfoil_ids == components[0].airfoil_ids


def test_load_components_from_csv_missing_required_column_raises(tmp_path: Path):
    comp_dir = tmp_path / "components"
    comp_dir.mkdir()
    _write_component_csv(comp_dir / "blade.csv", include_offset=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_components_from_csv(str(comp_dir))


def test_write_force_time_series_roundtrip(tmp_path: Path):
    out_file = tmp_path / "forces.csv"
    output_time = np.array([0.0, 0.5])
    forces = np.array(
        [
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            [[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]],
        ]
    )

    write_force_time_series(str(out_file), output_time, forces)
    lines = out_file.read_text().strip().splitlines()
    assert lines[0] == "time, node1x, node1y, node1z, node2x, node2y, node2z"
    assert lines[1].startswith("0.0, 1.0, 2.0, 3.0")
    assert lines[2].startswith("0.5, 7.0, 8.0, 9.0")


def test_write_force_time_series_validates_shape(tmp_path: Path):
    out_file = tmp_path / "forces.csv"
    with pytest.raises(ValueError, match="shape"):
        write_force_time_series(str(out_file), np.array([0.0]), np.array([[1.0, 2.0, 3.0]]))


def test_load_airfoil_fft_orients_dimensions(tmp_path: Path):
    h5_path = tmp_path / "airfoil.h5"
    re = np.array([1.0, 2.0])
    aoa = np.array([-5.0, 5.0])
    freq = 3

    # Stored as [freq, AOA, Re], loader should transpose to [Re, AOA, freq].
    valid_st = np.zeros((re.size, aoa.size, freq), dtype=float)
    valid_st[:, :, 1] = 0.2
    valid_st[:, :, 2] = 0.4
    valid_amp = np.ones((re.size, aoa.size, freq), dtype=float)
    valid_pha = np.zeros((re.size, aoa.size, freq), dtype=float)
    raw_st = np.transpose(valid_st, (2, 1, 0))
    raw_amp = np.transpose(valid_amp, (2, 1, 0))
    raw_pha = np.transpose(valid_pha, (2, 1, 0))

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("Airfoilname", data=b"synthetic")
        h5.create_dataset("Re", data=re)
        h5.create_dataset("AOA", data=aoa)
        h5.create_dataset("Thickness", data=np.array([0.1]))
        for key in ("CL_ST", "CD_ST", "CM_ST", "CF_ST"):
            h5.create_dataset(key, data=raw_st)
        for key in ("CL_Amp", "CD_Amp", "CM_Amp", "CF_Amp"):
            h5.create_dataset(key, data=raw_amp)
        for key in ("CL_Pha", "CD_Pha", "CM_Pha", "CF_Pha"):
            h5.create_dataset(key, data=raw_pha)

    with pytest.warns(RuntimeWarning):
        afft = load_airfoil_fft(str(h5_path))
    assert afft.CL_ST.shape == (2, 2, 3)
    np.testing.assert_allclose(afft.CL_ST[0, 0, :], valid_st[0, 0, :])


def test_load_airfoil_fft_accepts_scalar_thickness(tmp_path: Path):
    h5_path = tmp_path / "airfoil_scalar_thickness.h5"
    re = np.array([1.0, 2.0])
    aoa = np.array([-5.0, 5.0])
    arr = np.zeros((2, 2, 2), dtype=float)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("Airfoilname", data=b"scalar")
        h5.create_dataset("Re", data=re)
        h5.create_dataset("AOA", data=aoa)
        h5.create_dataset("Thickness", data=np.array(0.18))
        for key in ("CL_ST", "CD_ST", "CM_ST", "CF_ST", "CL_Amp", "CD_Amp", "CM_Amp", "CF_Amp", "CL_Pha", "CD_Pha", "CM_Pha", "CF_Pha"):
            h5.create_dataset(key, data=arr)

    afft = load_airfoil_fft(str(h5_path))
    assert afft.Thickness == pytest.approx(0.18)


def test_load_airfoil_fft_requires_dc_strouhal_at_index_zero(tmp_path: Path):
    h5_path = tmp_path / "airfoil_missing_dc.h5"
    re = np.array([1.0, 2.0])
    aoa = np.array([-5.0, 5.0])
    arr = np.zeros((2, 2, 2), dtype=float)
    bad_st = arr.copy()
    bad_st[:, :, 0] = 0.2
    bad_st[:, :, 1] = 0.0

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("Airfoilname", data=b"missing_dc")
        h5.create_dataset("Re", data=re)
        h5.create_dataset("AOA", data=aoa)
        h5.create_dataset("Thickness", data=np.array(0.18))
        for key in ("CL_ST", "CD_ST", "CM_ST", "CF_ST"):
            h5.create_dataset(key, data=bad_st)
        for key in ("CL_Amp", "CD_Amp", "CM_Amp", "CF_Amp"):
            h5.create_dataset(key, data=arr)
        for key in ("CL_Pha", "CD_Pha", "CM_Pha", "CF_Pha"):
            h5.create_dataset(key, data=arr)

    with pytest.raises(ValueError, match="index 0 must be DC"):
        load_airfoil_fft(str(h5_path), repair_legacy_dc_strouhal=False)


def test_load_airfoil_fft_repairs_legacy_nonzero_dc_strouhal_and_signed_harmonics(tmp_path: Path):
    h5_path = tmp_path / "airfoil_legacy_dc_st.h5"
    re = np.array([1.0, 2.0])
    aoa = np.array([-5.0, 5.0])
    st = np.ones((2, 2, 2), dtype=float) * 0.2
    st[:, :, 1] = 0.5
    amp = np.ones((2, 2, 2), dtype=float)
    amp[:, :, 1] = -0.25
    pha = np.zeros((2, 2, 2), dtype=float)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("Airfoilname", data=b"legacy")
        h5.create_dataset("Re", data=re)
        h5.create_dataset("AOA", data=aoa)
        h5.create_dataset("Thickness", data=np.array(0.18))
        for key in ("CL_ST", "CD_ST", "CM_ST", "CF_ST"):
            h5.create_dataset(key, data=st)
        for key in ("CL_Amp", "CD_Amp", "CM_Amp", "CF_Amp"):
            h5.create_dataset(key, data=amp)
        for key in ("CL_Pha", "CD_Pha", "CM_Pha", "CF_Pha"):
            h5.create_dataset(key, data=pha)

    with pytest.warns(RuntimeWarning):
        afft = load_airfoil_fft(str(h5_path))

    np.testing.assert_allclose(afft.CL_ST[:, :, 0], 0.0)
    np.testing.assert_allclose(afft.CL_Amp[:, :, 1], 0.25)
    np.testing.assert_allclose(afft.CL_Pha[:, :, 1], np.pi)


def test_load_airfoil_fft_zeros_legacy_active_zero_strouhal_harmonics(tmp_path: Path):
    h5_path = tmp_path / "airfoil_legacy_zero_st_harmonics.h5"
    re = np.array([1.0, 2.0])
    aoa = np.array([-5.0, 5.0])
    st = np.zeros((2, 2, 3), dtype=float)
    st[:, :, 2] = 0.5
    amp = np.zeros((2, 2, 3), dtype=float)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = 0.25
    amp[:, :, 2] = 0.5
    pha = np.ones((2, 2, 3), dtype=float)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("Airfoilname", data=b"legacy_zero_st")
        h5.create_dataset("Re", data=re)
        h5.create_dataset("AOA", data=aoa)
        h5.create_dataset("Thickness", data=np.array(0.18))
        for key in ("CL_ST", "CD_ST", "CM_ST", "CF_ST"):
            h5.create_dataset(key, data=st)
        for key in ("CL_Amp", "CD_Amp", "CM_Amp", "CF_Amp"):
            h5.create_dataset(key, data=amp)
        for key in ("CL_Pha", "CD_Pha", "CM_Pha", "CF_Pha"):
            h5.create_dataset(key, data=pha)

    with pytest.warns(RuntimeWarning) as caught:
        afft = load_airfoil_fft(str(h5_path))

    assert any("zeroing CL harmonic amplitudes" in str(warning.message) for warning in caught)
    np.testing.assert_allclose(afft.CL_Amp[:, :, 1], 0.0)
    np.testing.assert_allclose(afft.CL_Pha[:, :, 1], 0.0)
    np.testing.assert_allclose(afft.CL_Amp[:, :, 2], 0.5)
    np.testing.assert_allclose(afft.CL_ST[:, :, 2], 0.5)


def test_load_airfoil_coords_falls_back_with_warning():
    with pytest.warns(UserWarning):
        coords = load_airfoil_coords("does/not/exist.csv")
    assert coords.ndim == 2
    assert coords.shape[1] == 2


def test_load_inflow_time_series_from_direction_deg(tmp_path: Path):
    profile = tmp_path / "inflow_profile.csv"
    profile.write_text(
        "time,inflow_speed,inflow_direction_deg\n"
        "0.0,5.0,0.0\n"
        "0.5,6.0,90.0\n"
        "1.0,7.0,180.0\n"
    )

    inflow = load_inflow_time_series(str(profile))
    np.testing.assert_allclose(inflow.time, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(inflow.inflow_speeds, np.array([5.0, 6.0, 7.0]))
    np.testing.assert_allclose(inflow.inflow_directions[0], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(inflow.inflow_directions[1], np.array([0.0, 1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(inflow.inflow_directions[2], np.array([-1.0, 0.0, 0.0]), atol=1e-12)


def test_load_inflow_time_series_from_vector_columns(tmp_path: Path):
    profile = tmp_path / "inflow_profile.csv"
    profile.write_text(
        "time,inflow_speed,inflow_dir_x,inflow_dir_y,inflow_dir_z\n"
        "0.0,5.0,2.0,0.0,0.0\n"
        "1.0,5.0,0.0,3.0,4.0\n"
    )

    inflow = load_inflow_time_series(str(profile))
    np.testing.assert_allclose(inflow.inflow_directions[0], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(inflow.inflow_directions[1], np.array([0.0, 0.6, 0.8]), atol=1e-12)


def test_load_inflow_time_series_requires_strictly_increasing_time(tmp_path: Path):
    profile = tmp_path / "inflow_profile.csv"
    profile.write_text(
        "time,inflow_speed,inflow_direction_deg\n"
        "0.0,5.0,0.0\n"
        "0.0,6.0,10.0\n"
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_inflow_time_series(str(profile))


def _write_qblade_minimal_files(base: Path):
    sim = base / "case.sim"
    trb = base / "case.trb"
    bld_dir = base / "Aero"
    bld_dir.mkdir(parents=True, exist_ok=True)
    bld = bld_dir / "case.bld"

    sim.write_text(
        "\n".join(
            [
                "TURB_1",
                "case.trb TURBFILE - turbine file",
                "15.0 INITIAL_AZIMUTH - initial azimuth",
                "1.0 GLOBPOS_X - x",
                "2.0 GLOBPOS_Y - y",
                "3.0 GLOBPOS_Z - z",
                "END_TURB_1",
                "0.05 TIMESTEP - timestep",
                "20 NUMTIMESTEPS - steps",
                "1.30 DENSITYAIR - density",
                "1.00E-05 VISCOSITYAIR - kinematic viscosity",
                "4.2 MEANINF - mean inflow",
                "30.0 HORANGLE - horizontal angle",
                "10.0 VERTANGLE - vertical angle",
            ]
        )
        + "\n"
    )

    trb.write_text(
        "\n".join(
            [
                "Aero/case.bld BLADEFILE - blade file",
                "3 NUMBLADES - blades",
                "Structure/unused.str STRUCTURALFILE - structural file",
            ]
        )
        + "\n"
    )

    bld.write_text(
        "\n".join(
            [
                "3 NUMBLADES - blades",
                "HEIGHT_[m] CHORD_[m] RADIUS_[m] TOFFSET_[m] TWIST_[deg] CIRCANGLE_[deg] P_AXIS_[-] POLAR_FILE",
                "0.0 0.10 0.50 0.00 0.0 0.0 0.25 Polars/NACA_0018.plr",
                "1.0 0.10 0.50 0.00 0.0 0.0 0.25 Polars/NACA_0018.plr",
                "",
                "STRUT_1",
                "0.20 CHORDHUB_STR - hub chord",
                "0.10 CHORDBLD_STR - blade chord",
                "0.0 ANGLE_STR - angle",
                "1.0 HGTBLD_STR - blade height",
                "0.5 HGTHUB_STR - hub height",
                "0.0 DSTHUB_STR - hub distance",
                "0.30 PAXISHUB_STR - pitch axis hub",
                "0.30 PAXISBLD_STR - pitch axis blade",
                "Polars/flat_plate.plr POLAR_STR - polar file",
                "END_STRUT_1",
            ]
        )
        + "\n"
    )

    return sim, trb, bld


def test_load_qblade_definition_files(tmp_path: Path):
    sim, trb, bld = _write_qblade_minimal_files(tmp_path)

    sim_data = load_qblade_simulation_definition(str(sim))
    assert Path(str(sim_data["turbfile_path"])) == trb
    assert sim_data["fluid_density"] == pytest.approx(1.30)
    assert sim_data["fluid_dynamic_viscosity"] == pytest.approx(1.3e-5)

    trb_data = load_qblade_turbine_definition(str(trb))
    assert Path(str(trb_data["bladefile_path"])) == bld
    assert trb_data["num_blades"] == 3

    bld_data = load_qblade_blade_definition(str(bld))
    assert bld_data["num_blades"] == 3
    assert len(bld_data["blade_rows"]) == 2
    assert len(bld_data["strut_rows"]) == 1


def test_convert_qblade_to_vorlap_inputs_reports_malformed_qblade_files(tmp_path: Path):
    sim_missing_turb = tmp_path / "missing_turb.sim"
    sim_missing_turb.write_text("0.05 TIMESTEP - timestep\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not specify TURBFILE"):
        convert_qblade_to_vorlap_inputs(str(sim_missing_turb))

    sim_missing_reference = tmp_path / "missing_reference.sim"
    sim_missing_reference.write_text("missing.trb TURBFILE - turbine file\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="QBlade file does not exist"):
        convert_qblade_to_vorlap_inputs(str(sim_missing_reference))

    trb_missing_blade = tmp_path / "missing_blade.trb"
    trb_missing_blade.write_text("3 NUMBLADES - blades\n", encoding="utf-8")
    sim = tmp_path / "case.sim"
    sim.write_text("missing_blade.trb TURBFILE - turbine file\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not specify BLADEFILE"):
        convert_qblade_to_vorlap_inputs(str(sim))

    malformed_bld = tmp_path / "malformed.bld"
    malformed_bld.write_text("3 NUMBLADES - blades\nNo blade geometry table here\n", encoding="utf-8")
    trb = tmp_path / "case.trb"
    trb.write_text("malformed.bld BLADEFILE - blade file\n3 NUMBLADES - blades\n", encoding="utf-8")
    sim.write_text("case.trb TURBFILE - turbine file\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not locate blade data table"):
        convert_qblade_to_vorlap_inputs(str(sim))


def test_convert_qblade_to_vorlap_inputs(tmp_path: Path):
    sim, _trb, _bld = _write_qblade_minimal_files(tmp_path)

    components, viv_params, node_ids = convert_qblade_to_vorlap_inputs(str(sim))
    assert len(components) == 6  # 3 blades + (1 strut * 3 blades)
    assert len(node_ids) == 12   # (3*2 blade nodes) + (3*2 strut nodes)
    assert node_ids[0].startswith("BLD_1_")
    assert any(node_id.startswith("STR_1_2_") for node_id in node_ids)

    assert viv_params.fluid_density == pytest.approx(1.30)
    assert viv_params.fluid_dynamicviscosity == pytest.approx(1.3e-5)
    np.testing.assert_allclose(viv_params.rotation_axis_offset, np.array([1.0, 2.0, 3.0]))


def test_convert_qblade_to_vorlap_inputs_tower_defaults_to_cylinder(tmp_path: Path):
    sim, trb, _bld = _write_qblade_minimal_files(tmp_path)
    with trb.open("a") as f:
        f.write("12.0 TOWERHEIGHT - tower height\n")
        f.write("0.50 TOWERTOPRAD - tower top radius\n")
        f.write("0.70 TOWERBOTRAD - tower bottom radius\n")

    components, _viv_params, _node_ids = convert_qblade_to_vorlap_inputs(str(sim), include_tower=True)
    tower = next(comp for comp in components if comp.id == "TWR_1")
    assert set(tower.airfoil_ids) == {"cylinder"}


def test_write_qblade_loading_file(tmp_path: Path):
    outfile = tmp_path / "qblade_loading.txt"
    time = np.array([0.0, 0.5], dtype=float)
    force = np.array(
        [
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            [[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]],
        ],
        dtype=float,
    )
    torque = np.array(
        [
            [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
            [[0.7, 1.0], [0.8, 1.1], [0.9, 1.2]],
        ],
        dtype=float,
    )
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]

    write_qblade_loading_file(str(outfile), time, force, node_ids, local=True, node_torque_vector_nodes=torque)
    lines = outfile.read_text().splitlines()

    assert any("VorLap-generated QBlade external loading file" in line for line in lines)
    assert "BLD_1_0.000000 LOCAL" in lines
    i = lines.index("BLD_1_0.000000 LOCAL")
    data_line = lines[i + 1].split()
    assert len(data_line) == 7
    assert data_line[0] == "0"
    assert data_line[1] == "1"
    assert data_line[2] == "2"
    assert data_line[3] == "3"
    assert data_line[4] == "0.1"
    assert data_line[5] == "0.2"
    assert data_line[6] == "0.3"

    parsed = read_qblade_loading_file(str(outfile))
    assert list(parsed) == node_ids
    for inode, node_id in enumerate(node_ids):
        block = parsed[node_id]
        assert block["local"] is True
        data = block["data"]
        np.testing.assert_allclose(data[:, 0], time)
        np.testing.assert_allclose(data[:, 1:4], force[:, :, inode])
        np.testing.assert_allclose(data[:, 4:7], torque[:, :, inode])

    node0 = parsed[node_ids[0]]["data"]
    midpoint = np.array([np.interp(0.25, node0[:, 0], node0[:, column]) for column in range(1, 7)])
    np.testing.assert_allclose(midpoint, np.array([4.0, 5.0, 6.0, 0.4, 0.5, 0.6]))


def test_compare_qblade_loading_file_reports_parity_and_mismatches(tmp_path: Path):
    outfile = tmp_path / "qblade_loading.txt"
    time = np.array([0.0, 0.5], dtype=float)
    force = np.array(
        [
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            [[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]],
        ],
        dtype=float,
    )
    torque = 0.1 * force
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    write_qblade_loading_file(str(outfile), time, force, node_ids, local=True, node_torque_vector_nodes=torque)

    report = compare_qblade_loading_file(
        str(outfile),
        time,
        force,
        node_ids,
        local=True,
        node_torque_vector_nodes=torque,
    )

    assert report["passed"] is True
    assert report["node_count"] == 2
    assert report["compared_rows"] == 4
    assert report["node_order_matches"] is True
    assert report["max_force_abs_error_n"] == pytest.approx(0.0)
    assert report["max_torque_abs_error_nm"] == pytest.approx(0.0)

    expected_force = force.copy()
    expected_force[1, 2, 0] += 1.0
    mismatch = compare_qblade_loading_file(
        str(outfile),
        time,
        expected_force,
        node_ids,
        local=False,
        node_torque_vector_nodes=torque,
    )

    assert mismatch["passed"] is False
    assert mismatch["local_mismatches"] == node_ids
    assert mismatch["force_mismatches"] == ["BLD_1_0.000000"]
    assert mismatch["max_force_abs_error_n"] == pytest.approx(1.0)


def test_write_qblade_loading_file_validates_interpolation_inputs(tmp_path: Path):
    force = np.zeros((2, 3, 1), dtype=float)
    node_ids = ["BLD_1_0.000000"]

    with pytest.raises(ValueError, match="strictly increasing"):
        write_qblade_loading_file(str(tmp_path / "bad_time.txt"), np.array([0.0, 0.0]), force, node_ids)

    bad_force = force.copy()
    bad_force[1, 2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write_qblade_loading_file(str(tmp_path / "bad_force.txt"), np.array([0.0, 0.5]), bad_force, node_ids)
