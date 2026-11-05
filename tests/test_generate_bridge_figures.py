import importlib.util
import json
from pathlib import Path
import warnings

import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


REPO = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    REPO
    / "paper_drafts"
    / "NAWEA_QBladeBridge"
    / "figure_generation"
    / "generate_bridge_figures.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_bridge_figures", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_load_channel_accepts_normalized_qblade_label(monkeypatch):
    monkeypatch.delenv("SAMPLE_LOAD_CHANNEL", raising=False)
    generator = load_generator()
    header = [
        "Time [s]",
        "X_l~Mom.~BLD_1~pos~0.770~[Nm]",
        "Y_l Mom. STR_2_1 - BLD_1 z=8.1m [Nm]",
    ]

    index, channel = generator.select_load_channel(header)

    assert index == 1
    assert channel == "X_l~Mom.~BLD_1~pos~0.770~[Nm]"


def test_select_load_channel_requires_named_channel(monkeypatch):
    monkeypatch.delenv("SAMPLE_LOAD_CHANNEL", raising=False)
    generator = load_generator()
    header = [
        "Time [s]",
        "X_l Mom. BLD_1 pos 0.047 [Nm]",
        "X_l Mom. BLD_2 pos 0.770 [Nm]",
        "X_l Mom. BLD_1 pos 0.769 [Nm]",
        "Y_l Mom. BLD_1 pos 0.770 [Nm]",
    ]

    with pytest.raises(ValueError, match="Requested default SAMPLE_LOAD_CHANNEL not found"):
        generator.select_load_channel(header)


def test_fail_on_warnings_records_allowed_replay_repairs():
    generator = load_generator()
    generator.CAPTURED_WARNINGS.clear()

    def emit_allowed():
        warnings.warn("Repairing legacy airfoil FFT 'demo'", RuntimeWarning, stacklevel=2)
        return "ok"

    assert generator.fail_on_warnings("demo replay", emit_allowed) == "ok"
    assert generator.CAPTURED_WARNINGS == [
        {
            "label": "demo replay",
            "category": "RuntimeWarning",
            "message": "Repairing legacy airfoil FFT 'demo'",
            "allowed": True,
        }
    ]


def test_fail_on_warnings_rejects_unexpected_warning():
    generator = load_generator()
    generator.CAPTURED_WARNINGS.clear()

    def emit_unexpected():
        warnings.warn("new runtime diagnostic", RuntimeWarning, stacklevel=2)

    with pytest.raises(RuntimeError, match="new runtime diagnostic"):
        generator.fail_on_warnings("demo replay", emit_unexpected)
    assert generator.CAPTURED_WARNINGS[0]["allowed"] is False


def test_apply_figure_config_overrides_roots_targets_and_channel(tmp_path):
    generator = load_generator()
    config_path = tmp_path / "figure_config.json"
    config_path.write_text(
        json.dumps(
            {
                "verification_run_root": str(tmp_path / "verification"),
                "sample_run_root": str(tmp_path / "sample"),
                "figure_dir": str(tmp_path / "figures"),
                "sample_load_channel": "Y_l Mom. BLD_1 pos 0.500 [Nm]",
                "target_interface_nodes": ["BLD_1_0.500000", "STR_1_1_1.000000"],
                "target_blade_station": 0.5,
                "target_strut_interface_z_m": 7.25,
                "target_interface_labels": {
                    "BLD_1_0.500000": "Blade mid",
                    "STR_1_1_1.000000": "Strut interface",
                },
            }
        ),
        encoding="utf-8",
    )

    generator.apply_figure_config(generator.load_figure_config(config_path))

    assert generator.VERIFICATION_RUN_ROOT == (tmp_path / "verification").resolve()
    assert generator.SAMPLE_RUN_ROOT == (tmp_path / "sample").resolve()
    assert generator.FIG_DIR == (tmp_path / "figures").resolve()
    assert generator.SELECTED_LOAD_CHANNEL == "Y_l Mom. BLD_1 pos 0.500 [Nm]"
    assert generator.TARGET_INTERFACE_NODES == ("BLD_1_0.500000", "STR_1_1_1.000000")
    assert generator.TARGET_BLADE_STATION == pytest.approx(0.5)
    assert generator.TARGET_STRUT_INTERFACE_Z_M == pytest.approx(7.25)
    assert generator.TARGET_INTERFACE_LABELS["BLD_1_0.500000"] == "Blade mid"


def test_apply_figure_config_requires_two_target_nodes():
    generator = load_generator()

    with pytest.raises(ValueError, match="target_interface_nodes"):
        generator.apply_figure_config({"target_interface_nodes": ["BLD_1_0.500000"]})


def test_main_allows_missing_verification_when_sample_outputs_exist(tmp_path, monkeypatch, capsys):
    generator = load_generator()
    sample_root = tmp_path / "sample"
    verification_root = tmp_path / "missing_verification"
    sample_figure = tmp_path / "figures" / "sil_vorlap_sample_diagnostics.pdf"
    del_figure = tmp_path / "figures" / "parked_10s_placeholder_del.pdf"
    target_nodes = ("BLD_1_0.500000", "STR_1_1_1.000000")

    monkeypatch.setattr(generator, "SAMPLE_RUN_ROOT", sample_root)
    monkeypatch.setattr(generator, "VERIFICATION_RUN_ROOT", verification_root)
    monkeypatch.setattr(generator, "FIG_DIR", tmp_path / "figures")
    monkeypatch.setattr(generator, "TARGET_INTERFACE_NODES", target_nodes)

    def fake_missing_summaries(run_root, scenarios):
        if run_root == sample_root:
            return []
        return [run_root / scenarios[0] / "results" / "missing_run_summary.json"]

    def fake_read_summary(_run_root, scenario):
        if scenario == "sil_vorlap":
            return {"sil_vorlap_max_abs_force": 123.0}
        return {}

    replay = {
        "max_abs_diff_n": 0.0,
        "target_max_abs_diff_n": 0.0,
        "target_force_history": {
            node: {"max_abs_diff_n": 0.0, "max_abs_sil_component_n": 1.0, "max_sil_norm_n": 2.0}
            for node in target_nodes
        },
    }

    monkeypatch.setattr(generator, "missing_summaries", fake_missing_summaries)
    monkeypatch.setattr(generator, "read_summary", fake_read_summary)
    monkeypatch.setattr(generator, "isolated_replay_from_history", lambda _summary: replay)
    monkeypatch.setattr(generator, "plot_sample_diagnostics", lambda _summaries, _replay: "sample channel")
    monkeypatch.setattr(
        generator,
        "plot_sample_del_placeholder",
        lambda _root: {"figure": str(del_figure), "del_rows": 1, "signed_rows": 1},
    )

    def fail_verification_plot(*_args, **_kwargs):
        raise AssertionError("verification plot should not run when verification summaries are missing")

    monkeypatch.setattr(generator, "plot_bridge_verification", fail_verification_plot)

    generator.main([])

    payload = json.loads(capsys.readouterr().out)
    assert payload["verification_available"] is False
    assert payload["verification_missing_summaries"] == [
        str(verification_root / "baseline" / "results" / "missing_run_summary.json")
    ]
    assert payload["figures"] == [str(sample_figure), str(del_figure)]
    assert payload["sample_del_placeholder"]["del_rows"] == 1
