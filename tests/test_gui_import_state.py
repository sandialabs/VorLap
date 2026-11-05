from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import json

import numpy as np
import pytest


class DummyValue:
    def __init__(self, value="") -> None:
        self.value = value

    def get(self):
        return self.value


def test_gui_modules_import_without_starting_mainloop():
    from vorlap.gui.app import NOTEBOOK_TAB_LABELS, VorLapApp
    from vorlap.gui.tabs import PlotsOutputsTab, QBladeBridgeTab, SimulationSetupTab

    assert VorLapApp is not None
    assert NOTEBOOK_TAB_LABELS == ("Simulation Setup", "Plots & Outputs", "QBlade-Bridge")
    assert QBladeBridgeTab is not None
    assert PlotsOutputsTab is not None
    assert SimulationSetupTab is not None


def test_plots_tab_no_matplotlib_update_logs_skip(monkeypatch):
    from vorlap.gui.tabs import plots as plots_module

    tab = object.__new__(plots_module.PlotsOutputsTab)
    messages = []
    tab.app = SimpleNamespace(log=messages.append, analysis_results=None)
    tab.plot_type = DummyValue("geometry")
    monkeypatch.setattr(plots_module, "MATPLOTLIB_OK", False)

    plots_module.PlotsOutputsTab.update_plots(tab)

    assert messages == ["Plot update skipped: Matplotlib is not available.\n"]


def test_prepare_result_plot_data_is_headless_and_copies_arrays():
    from vorlap.gui.tabs.plots import prepare_result_plot_data

    force = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    moment = force + 100.0
    results = {
        "viv_params": SimpleNamespace(
            azimuths=np.array([0.0, 45.0, 90.0]),
            inflow_speeds=np.array([5.0, 10.0]),
        ),
        "percdiff_matrix": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "total_global_force_vector": force,
        "total_global_moment_vector": moment,
    }

    percdiff = prepare_result_plot_data(results, "percdiff")
    assert percdiff.title == "Worst Percent Difference"
    assert percdiff.colorbar_label == "Freq % Diff"
    assert percdiff.cmap == "viridis_r"
    assert percdiff.extent == (0.0, 90.0, 5.0, 10.0)
    assert percdiff.vmin == 0.0
    assert percdiff.vmax == 50.0
    assert percdiff.zero_center_norm is False
    np.testing.assert_allclose(percdiff.data, results["percdiff_matrix"])

    fx = prepare_result_plot_data(results, "fx")
    assert fx.title == "Force FX"
    assert fx.colorbar_label == "Force (N)"
    assert fx.cmap == "coolwarm"
    assert fx.zero_center_norm is True
    np.testing.assert_allclose(fx.data, force[:, :, 0])

    mz = prepare_result_plot_data(results, "mz")
    assert mz.title == "Moment MZ"
    assert mz.colorbar_label == "Moment (N-m)"
    np.testing.assert_allclose(mz.data, moment[:, :, 2])

    force[:, :, 0] = 999.0
    np.testing.assert_allclose(fx.data, np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)[:, :, 0])


def test_prepare_result_plot_data_rejects_unsupported_or_empty_inputs():
    from vorlap.gui.tabs.plots import prepare_result_plot_data

    results = {
        "viv_params": SimpleNamespace(azimuths=np.array([]), inflow_speeds=np.array([5.0])),
        "percdiff_matrix": np.zeros((1, 1)),
        "total_global_force_vector": np.zeros((1, 1, 3)),
        "total_global_moment_vector": np.zeros((1, 1, 3)),
    }

    with pytest.raises(ValueError, match="non-empty azimuth"):
        prepare_result_plot_data(results, "percdiff")

    results["viv_params"] = SimpleNamespace(azimuths=np.array([0.0]), inflow_speeds=np.array([5.0]))
    with pytest.raises(ValueError, match="Unsupported plot type"):
        prepare_result_plot_data(results, "bad")

    with pytest.raises(ValueError, match="Geometry plots require"):
        prepare_result_plot_data(results, "geometry")


def test_result_plot_export_writes_csv_and_json_without_tk(tmp_path: Path):
    from vorlap.gui.tabs.plots import prepare_result_plot_export, write_result_plot_export

    force = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    results = {
        "viv_params": SimpleNamespace(
            azimuths=np.array([0.0, 45.0, 90.0]),
            inflow_speeds=np.array([5.0, 10.0]),
        ),
        "percdiff_matrix": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "total_global_force_vector": force,
        "total_global_moment_vector": force + 100.0,
    }

    export = prepare_result_plot_export(results, "fx")
    assert export.metadata["plot_type"] == "fx"
    assert export.metadata["value_unit"] == "N"
    assert export.metadata["row_count"] == 6
    assert export.rows[0] == {"inflow_mps": 5.0, "azimuth_deg": 0.0, "value": 0.0}
    assert export.rows[-1] == {"inflow_mps": 10.0, "azimuth_deg": 90.0, "value": 15.0}

    csv_path = tmp_path / "fx_plot_data.csv"
    json_path = tmp_path / "fx_plot_data.json"
    assert write_result_plot_export(str(csv_path), results, "fx") == str(csv_path)
    assert write_result_plot_export(str(json_path), results, "fx") == str(json_path)

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "plot_type,title,value_label,value_unit,inflow_mps,azimuth_deg,value" in csv_text
    assert "fx,Force FX,Force (N),N,10.0,90.0,15.0" in csv_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["title"] == "Force FX"
    assert payload["rows"][-1]["value"] == 15.0


def test_gui_state_notifies_subscribers_and_copies_frequency_arrays():
    from vorlap.gui.state import VorLapGuiState

    state = VorLapGuiState()
    events = []
    listener = state.subscribe(lambda field_name, value: events.append((field_name, value)))
    frequencies = np.array([0.1, 0.2])

    state.set_components([object()])
    state.set_natural_frequencies(frequencies)
    frequencies[:] = 99.0
    state.set_analysis_results(None)
    state.set_qblade_sim_path("/tmp/case.sim")
    state.unsubscribe(listener)
    state.set_components([])

    assert [field_name for field_name, _value in events] == [
        "components",
        "natural_frequencies",
        "analysis_results",
        "qblade_sim_path",
    ]
    assert len(events[0][1]) == 1
    np.testing.assert_allclose(events[1][1], np.array([0.1, 0.2]))
    np.testing.assert_allclose(state.natural_frequencies, np.array([0.1, 0.2]))
    assert state.qblade_sim_path == "/tmp/case.sim"


def test_vorlap_app_legacy_state_properties_delegate_without_tk_root():
    from vorlap.gui.app import VorLapApp
    from vorlap.gui.state import VorLapGuiState

    app = VorLapApp.__new__(VorLapApp)
    app.state = VorLapGuiState()
    events = []
    app.state.subscribe(lambda field_name, value: events.append((field_name, value)))

    app.components = [object(), object()]
    app.natural_frequencies = np.array([1.0, 2.0])
    app.analysis_results = None
    app.qblade_sim_path = "/tmp/model.sim"

    assert len(app.components) == 2
    np.testing.assert_allclose(app.natural_frequencies, np.array([1.0, 2.0]))
    assert app.analysis_results is None
    assert app.qblade_sim_path == "/tmp/model.sim"
    assert [field_name for field_name, _value in events] == [
        "components",
        "natural_frequencies",
        "analysis_results",
        "qblade_sim_path",
    ]


def test_notebook_tab_refresh_logs_debug_on_selection_error(monkeypatch):
    from vorlap.gui import app as gui_app

    messages = []
    scheduled = []

    def record_debug(message, *args, **kwargs):
        messages.append((message, kwargs.get("exc_info")))

    class BadNotebook:
        def select(self):
            raise gui_app.tk.TclError("selected tab disappeared")

    fake_app = SimpleNamespace(
        after_idle=lambda callback: scheduled.append(callback),
        after=lambda delay_ms, callback: scheduled.append(callback),
    )
    monkeypatch.setattr(gui_app.LOGGER, "debug", record_debug)

    gui_app.VorLapApp._on_tab_changed(fake_app, SimpleNamespace(widget=BadNotebook()))

    assert scheduled == []
    assert messages == [("Could not resolve selected notebook tab for refresh.", True)]


def test_notebook_tab_refresh_logs_debug_on_callback_failure(monkeypatch):
    from vorlap.gui import app as gui_app

    messages = []

    def record_debug(message, *args, **kwargs):
        messages.append((message, kwargs.get("exc_info")))

    class FailingTab:
        def on_tab_selected(self):
            raise RuntimeError("refresh failed")

    class Notebook:
        def select(self):
            return ".tab"

        def nametowidget(self, selected):
            assert selected == ".tab"
            return FailingTab()

    fake_app = SimpleNamespace(
        after_idle=lambda callback: callback(),
        after=lambda delay_ms, callback: callback(),
        update_idletasks=lambda: None,
    )
    monkeypatch.setattr(gui_app.LOGGER, "debug", record_debug)

    gui_app.VorLapApp._on_tab_changed(fake_app, SimpleNamespace(widget=Notebook()))

    assert messages == [
        ("Could not refresh selected notebook tab.", True),
        ("Could not refresh selected notebook tab.", True),
    ]


def test_gui_tabs_exports_only_active_notebook_tabs():
    import vorlap.gui.tabs as tabs
    from vorlap.gui.tabs.setup import SimulationSetupTab

    assert tabs.__all__ == [
        "SimulationSetupTab",
        "PlotsOutputsTab",
        "QBladeBridgeTab",
    ]
    assert not hasattr(tabs, "AnalysisTab")
    assert not hasattr(tabs, "GeometryTab")
    assert not hasattr(SimulationSetupTab, "load_geom")
    assert not hasattr(SimulationSetupTab, "save_geom")


def test_frozen_repo_dir_uses_pyinstaller_extraction_root(monkeypatch, tmp_path):
    import vorlap

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert vorlap._resolve_repo_dir() == str(tmp_path.resolve())
    assert Path(vorlap._resolve_repo_dir()) / "data" / "airfoils" == tmp_path / "data" / "airfoils"


def test_installed_repo_dir_can_fall_back_to_share_vorlap_data(monkeypatch, tmp_path):
    import vorlap

    fake_package = tmp_path / "site-packages" / "vorlap" / "__init__.py"
    installed_data = tmp_path / "prefix" / "share" / "vorlap" / "data"
    fake_package.parent.mkdir(parents=True)
    installed_data.mkdir(parents=True)

    monkeypatch.setattr(vorlap, "__file__", str(fake_package))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "prefix"))
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert vorlap._resolve_repo_dir() == str(installed_data.parent.resolve())


def test_gui_build_scripts_bundle_top_level_data():
    repo = Path(__file__).resolve().parents[1]

    linux = (repo / "scripts" / "build_gui_linux.sh").read_text(encoding="utf-8")
    mac = (repo / "scripts" / "build_gui_mac.sh").read_text(encoding="utf-8")
    windows = (repo / "scripts" / "build_gui_windows.ps1").read_text(encoding="utf-8")

    assert '--add-data "data:data"' in linux
    assert '--add-data "data:data"' in mac
    assert '--add-data "data;data"' in windows


@pytest.fixture(scope="module")
def tk_available():
    """Only unavailable Tk may skip; application regressions must fail."""
    probe = subprocess.run(
        [sys.executable, "-c", "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.destroy()"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout or f"exit status {probe.returncode}").strip()[:500]
        if os.environ.get("VORLAP_REQUIRE_GUI_TESTS") == "1":
            pytest.fail(f"Tk is required but unavailable: {details}")
        pytest.skip(f"Tk is not available for GUI smoke tests: {details}")


def test_qblade_bridge_tab_tk_state_smoke(tk_available):
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    code = """
from types import SimpleNamespace
import tkinter as tk
from vorlap.gui.state import VorLapGuiState
from vorlap.gui.tabs import QBladeBridgeTab

root = tk.Tk()
root.withdraw()
try:
    state = VorLapGuiState()
    app = SimpleNamespace(
        state=state,
        log=lambda _message: None,
    )
    tab = QBladeBridgeTab(root, app)
    root.update_idletasks()
    assert tab.status_var.get() == "idle"
    assert tab.source_bounds_policy_var.get() == "raise"
    assert tab.patch_original_model_var.get() is False
    assert tab.duration_var.get() == "60"
    assert tab.timestep_var.get() == "0.0075"
    assert tab.steps_var.get() == "8000"
    assert tab.rpm_var.get() == "1"
    assert tab.store_from_var.get() == "0"
    assert tab.store_replay_var.get() == "true"
    assert tab.store_project_var.get() is True
    assert tab.checkpoint_every_var.get() == "200"
    assert tab.checkpoint_dir_var.get().endswith("run_headless/runs/gui_qblade_bridge/checkpoints")
    assert tab.resume_checkpoint_var.get() == ""
    assert not hasattr(tab, "native_button")
    assert not hasattr(tab, "sign_probe_button")
finally:
    root.destroy()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_gui_component_change_invalidates_previous_analysis():
    from vorlap.gui.state import VorLapGuiState

    state = VorLapGuiState()
    state.analysis_results = object()
    events = []
    state.subscribe(lambda field_name, value: events.append((field_name, value)))
    state.set_components([object()])

    assert state.analysis_results is None
    assert events[0] == ("analysis_results", None)


def test_full_gui_offline_workflow(tk_available, tmp_path):
    """Exercise real Tk inputs, worker, physical computation, plots and files."""
    repo = Path(__file__).resolve().parents[1]
    code = r'''
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np
import tkinter.messagebox as messagebox
import vorlap
from vorlap.gui.app import VorLapApp
from vorlap.gui.tabs.plots import write_result_plot_export

sys.path.insert(0, str(Path.cwd() / "tests"))
from conftest import make_component

out = Path(sys.argv[1])
app = VorLapApp()
app.withdraw()
errors = []
messagebox.showerror = lambda title, message: errors.append((title, message))
messagebox.showwarning = lambda title, message: errors.append((title, message))
main_thread = threading.get_ident()
state_threads = []
app.state.subscribe(lambda *_args: state_threads.append(threading.get_ident()))
try:
    setup = app.tab_setup
    setup.use_qblade_geometry.set(False)
    setup._on_qblade_toggle()
    component = make_component(n_nodes=3, span=2.0, airfoil_id="NACA0018")
    component.chord[:] = 0.5
    component.thickness[:] = 0.18
    geometry_dir = out / "components"
    vorlap.write_components_to_csv(str(geometry_dir), [component])
    setup.components_path.set(str(geometry_dir))
    setup.load_components()
    assert len(app.components) == 1
    setup._populate_frequency_entries([0.073456789, 0.003456789])
    np.testing.assert_allclose(setup.get_natural_frequencies(), [0.073456789, 0.003456789], atol=1e-12)
    inflow = out / "inflow.csv"
    inflow.write_text("time,inflow_speed,inflow_direction_deg\n0,5,0\n0.01,5,0\n0.02,5,0\n")
    setup.inflow_profile_path.set(str(inflow))
    setup.sim_save_var.set(str(out / "results"))
    setup.param_table.clear()
    values = dict(azimuth_start=0, azimuth_end=20, azimuth_step=10,
                  inflow_speed_start=5, inflow_speed_end=7, inflow_speed_step=1,
                  output_time_start=0, output_time_end=0.03, output_time_step=0.01,
                  output_azimuth=0, output_vinf=5, n_freq_depth=2)
    for key, value in values.items():
        setup.param_table.append_row(["", key, value])
    app.tab_plots.run_and_plot()
    worker = setup._worker
    setup.run_and_save()
    assert setup._worker is worker
    assert str(app.tab_plots.run_button.cget("state")) == "disabled"
    deadline = time.monotonic() + 40
    while setup._worker_active and time.monotonic() < deadline:
        app.update()
        time.sleep(0.01)
    assert not setup._worker_active, "Offline calculation did not complete"
    assert not errors, errors
    assert state_threads and set(state_threads) == {main_thread}
    result = app.analysis_results
    assert result is not None
    assert result.total_global_force_vector.shape == (2, 2, 3)
    assert np.all(np.isfinite(result.total_global_force_vector))
    assert np.all(np.isfinite(result.time_varying_total_global_force_vector))
    assert np.any(np.abs(result.time_varying_total_global_force_vector) > 0)
    assert (out / "results" / "forces_output.csv").is_file()
    write_result_plot_export(str(out / "force_x.csv"), result, "fx")
    write_result_plot_export(str(out / "force_x.json"), result, "fx")
    assert json.loads((out / "force_x.json").read_text())["metadata"]["row_count"] == 4
    # Completed plots remain available while the next setup is being edited.
    setup.param_table.clear()
    setup.param_table.append_row(["", "azimuth_step", "invalid"])
    for plot_type in ("geometry", "percdiff", "fx", "mz"):
        app.tab_plots.plot_type.set(plot_type)
        app.tab_plots.update_plots()
        app.update_idletasks()
    app.tab_plots.fig.savefig(out / "moment_z.png")
    assert not errors, errors
    assert str(app.tab_plots.run_button.cget("state")) == "normal"
    # A failed rerun clears old numerical output and restores the Run control.
    setup.param_table.clear()
    setup.inflow_profile_path.set(str(out / "missing.csv"))
    setup.run_and_save()
    deadline = time.monotonic() + 10
    while setup._worker_active and time.monotonic() < deadline:
        app.update()
        time.sleep(0.01)
    assert not setup._worker_active
    assert app.analysis_results is None
    assert len(errors) == 1 and "missing.csv" in errors[0][1], errors
    assert str(app.tab_plots.run_button.cget("state")) == "normal"
finally:
    app.destroy()
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
