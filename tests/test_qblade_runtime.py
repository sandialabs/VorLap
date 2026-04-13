import numpy as np

from conftest import make_component, make_constant_airfoil_fft, make_viv_params
from vorlap import QBladeController
from vorlap import qblade_runtime
from vorlap.qblade_runtime import (
    VorLapQBladeRuntime,
    build_external_library_table_spec,
    build_qblade_external_config,
    patch_qblade_structural_definition,
    patch_qblade_turbine_definition,
    write_qblade_external_config,
)


def test_build_external_library_table_spec_matches_node_count():
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    spec = build_external_library_table_spec(node_ids)

    assert spec.swap_size == 15
    assert spec.swap_layout["time"].offset == 0
    assert spec.swap_layout["velocity"].shape == (2, 3)
    assert spec.swap_layout["force"].offset == 9
    assert spec.input_rows[0] == (0, "Time [s]")
    assert spec.input_rows[2] == (2, "LSS Azimuthal Pos. [deg]")
    assert spec.output_rows[0] == (9, "ADDFORCE", "BLD_1", 0.0, "X", False)


def test_build_external_library_table_spec_keeps_action_positions_per_node():
    spec = build_external_library_table_spec(["BLD_2_0.230455", "STR_1_3_0.172168"])
    assert spec.output_rows[0] == (9, "ADDFORCE", "BLD_2", 0.230455, "X", False)
    assert spec.output_rows[3] == (12, "ADDFORCE", "STR_1_3", 0.172168, "X", False)


def test_patch_qblade_turbine_definition_inserts_external_library_section():
    text = """----------------------------------------Turbine Controller-----------------------------------------------------------\n1                                                  CONTROLLERTYPE     - the type of turbine controller 0 = none, 1 = BLADED, 2 = DTU, 3 = TUB\nlibaero-lib-xflowenergy                            CONTROLLERFILE     - the controller file name, WITHOUT file ending (.dll or .so ) - leave blank if unused\nControl/discon_xflowenergy.in                      PARAMETERFILE      - the controller parameter file name (leave blank if unused)\n"""
    patched = patch_qblade_turbine_definition(
        text,
        library_stem="libvorlap_qblade_bridge",
        function_name="update",
        swap_size=15,
        parameter_file="Control/vorlap_qblade_external.json",
    )

    assert "CONTROLLERTYPE" in patched
    assert "1                                                  CONTROLLERTYPE" not in patched
    assert "LIBFILE_1" in patched
    assert "libvorlap_qblade_bridge" in patched
    assert "LIBARRAYSIZE_1" in patched
    assert "15" in patched
    assert "LIBPARAMETERFILE_1" in patched


def test_patch_qblade_structural_definition_appends_external_tables():
    text = "TWR_1.0\n"
    spec = build_external_library_table_spec(["BLD_1_0.000000"])
    patched = patch_qblade_structural_definition(text, table_spec=spec)

    assert "EXTERNAL_1_IN" in patched
    assert "EXTERNAL_1_OUT" in patched
    assert '"Time [s]"' in patched
    assert "ADDFORCE" in patched


def test_runtime_updates_forces_in_place_and_reuses_cache(monkeypatch):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    viv_params = make_viv_params()
    viv_params.output_time = np.array([0.0, 0.25], dtype=float)
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["timestep"].offset] = 0.25
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 6] = np.array(
        [2.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32
    )

    import vorlap.interpolation as interpolation

    calls = {"count": 0}
    original = interpolation.interpolate_fft_spectrum_batch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(interpolation, "interpolate_fft_spectrum_batch", wrapped)

    first = runtime.update(swap)
    assert first.shape == (2, 3)
    assert np.isfinite(first).all()
    assert calls["count"] > 0

    first_count = calls["count"]
    second = runtime.update(swap.copy())
    assert calls["count"] == first_count
    assert second is first
    np.testing.assert_allclose(second, first)
    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 6],
        first.reshape(-1),
    )


def test_build_qblade_external_config_and_write(tmp_path):
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
        sample_step=0.1,
    )
    assert config["swap_size"] == 9
    assert config["force_scale"] == 1.0

    out_path = tmp_path / "vorlap_qblade_external.json"
    written = write_qblade_external_config(str(out_path), config)
    assert out_path.exists()
    assert written == str(out_path.resolve())


def test_build_qblade_external_config_preserves_relative_airfoil_dir():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="../VorLapAirfoils",
        node_ids=["BLD_1_0.000000"],
    )
    assert config["airfoil_dir"] == "../VorLapAirfoils"


def test_build_qblade_external_config_preserves_relative_sim_path():
    config = build_qblade_external_config(
        sim_path="../../baseline_wMinSagSnubbers-Wwnd.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
    )
    assert config["sim_path"] == "../../baseline_wMinSagSnubbers-Wwnd.sim"


def test_build_qblade_external_config_accepts_force_scale():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
        force_scale=100.0,
    )
    assert config["force_scale"] == 100.0


def test_build_qblade_external_config_accepts_source_parameter_dir():
    config = build_qblade_external_config(
        sim_path="../../baseline_wMinSagSnubbers-Wwnd.sim",
        airfoil_dir="../VorLapAirfoils",
        node_ids=["BLD_1_0.000000"],
        source_parameter_dir="/tmp/original/Control",
    )
    assert config["source_parameter_dir"] == "/tmp/original/Control"


def test_runtime_resolves_relative_paths_from_original_parameter_dir_when_copied_to_temp(tmp_path, monkeypatch):
    original_control = tmp_path / "qblade_xflow" / "wMinSagSnubbers" / "Control"
    original_control.mkdir(parents=True)
    temp_control = tmp_path / "QBladeCE" / "TEMP" / "run1"
    temp_control.mkdir(parents=True)

    sim_path = tmp_path / "qblade_xflow" / "baseline.sim"
    sim_path.write_text("sim", encoding="utf-8")
    airfoil_dir = tmp_path / "qblade_xflow" / "wMinSagSnubbers" / "VorLapAirfoils"
    airfoil_dir.mkdir(parents=True)
    (airfoil_dir / "default.h5").write_text("placeholder", encoding="utf-8")

    config = build_qblade_external_config(
        sim_path="../../baseline.sim",
        airfoil_dir="../VorLapAirfoils",
        node_ids=["BLD_1_0.000000"],
        source_parameter_dir=str(original_control),
    )
    cfg_path = temp_control / "vorlap_qblade_external.json"
    write_qblade_external_config(str(cfg_path), config)

    calls = {}

    def fake_convert(sim_arg, **kwargs):
        calls["sim_path"] = sim_arg
        component = make_component(n_nodes=1, airfoil_id="default")
        component.id = "BLD_1"
        viv_params = make_viv_params()
        return [component], viv_params, ["BLD_1_0.000000"]

    def fake_load_airfoils(airfoil_arg):
        calls["airfoil_dir"] = airfoil_arg
        return {"default": make_constant_airfoil_fft()}

    monkeypatch.setattr(qblade_runtime, "convert_qblade_to_vorlap_inputs", fake_convert)
    monkeypatch.setattr(qblade_runtime, "load_airfoil_fft_directory", fake_load_airfoils)

    runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))

    assert runtime is not None
    assert calls["sim_path"] == str(sim_path)
    assert calls["airfoil_dir"] == str(airfoil_dir)


def test_build_qblade_external_config_infers_required_geometry_flags():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000", "STR_1_1_1.000000", "TWR_1_0.000000"],
        include_struts=False,
        include_tower=False,
    )

    assert config["include_struts"] is True
    assert config["include_tower"] is True
