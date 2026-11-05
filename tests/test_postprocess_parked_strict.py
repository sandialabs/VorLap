from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


RUN_HEADLESS = Path(__file__).resolve().parents[1] / "run_headless"


def load_postprocess_module():
    path = RUN_HEADLESS / "postprocess_parked_demo.py"
    spec = importlib.util.spec_from_file_location("postprocess_parked_demo_strict", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


postprocess = load_postprocess_module()


def write_result(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    text = [
        "Results Output File created for test",
        "Simulation: strict ingestion fixture",
        "\t".join(columns),
        *("\t".join(str(value) for value in row) for row in rows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def strict_settings(*, scenarios: list[str] | None = None) -> dict:
    return {
        "scenario_names": scenarios or ["baseline", "vorlap"],
        "planned_cases": [
            {
                "label": "U05p00ms",
                "wind_speed_mps": 5.0,
                "parked_azimuth_deg": 270.0,
            }
        ],
        "timestep_s": 0.1,
        "timesteps": 4,
        "qblade_steps_override": 4,
        "store_from_s": 0.1,
    }


@pytest.mark.parametrize("token", ["bad-token", "NaN", "Inf", "-Inf"])
def test_qblade_reader_rejects_malformed_and_nonfinite_tokens(tmp_path: Path, token: str) -> None:
    path = tmp_path / "result.txt"
    write_result(path, ["Time~[s]", "Load~[N]"], [[0.0, token]])
    with pytest.raises(ValueError, match="not numeric|non-finite"):
        postprocess.read_qblade_results(path)


def test_qblade_reader_rejects_duplicate_columns_and_short_rows(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.txt"
    write_result(duplicate, ["Time~[s]", "Time~[s]"], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="duplicate column"):
        postprocess.read_qblade_results(duplicate)

    short = tmp_path / "short.txt"
    write_result(short, ["Time~[s]", "Load~[N]"], [[0.0]])
    with pytest.raises(ValueError, match="has 1 fields; expected 2"):
        postprocess.read_qblade_results(short)


def test_qblade_reader_requires_exact_filter_columns_and_required_channels(tmp_path: Path) -> None:
    path = tmp_path / "result.txt"
    write_result(path, ["Load~[N]", "Time~[s]"], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="do not exactly match"):
        postprocess.read_qblade_results(
            path,
            expected_columns=["Time~[s]", "Load~[N]"],
        )
    with pytest.raises(ValueError, match="missing required unique fatigue"):
        postprocess.read_qblade_results(path, required_columns=["Moment~[Nm]"])


def test_result_filter_parser_ignores_supported_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "result.filter"
    path.write_text(
        "# generated filter\nTime [s]\n! QBlade comment\nLoad [N]\n",
        encoding="utf-8",
    )
    assert postprocess.result_filter_columns(path) == ["Time~[s]", "Load~[N]"]


def test_result_history_rejects_missing_rows_duplicate_time_and_bad_grid(tmp_path: Path) -> None:
    settings = strict_settings()
    valid = postprocess.read_qblade_results(
        _write_simple_time_result(tmp_path / "valid.txt", [0.1, 0.2, 0.3])
    )
    postprocess.validate_result_history(valid, settings, path=tmp_path / "valid.txt")

    missing = postprocess.read_qblade_results(
        _write_simple_time_result(tmp_path / "missing.txt", [0.1, 0.2])
    )
    with pytest.raises(ValueError, match="retained rows; expected exactly 3"):
        postprocess.validate_result_history(missing, settings, path=tmp_path / "missing.txt")

    duplicate = postprocess.read_qblade_results(
        _write_simple_time_result(tmp_path / "duplicate-time.txt", [0.1, 0.2, 0.2])
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        postprocess.validate_result_history(
            duplicate,
            settings,
            path=tmp_path / "duplicate-time.txt",
        )

    off_grid = postprocess.read_qblade_results(
        _write_simple_time_result(tmp_path / "off-grid.txt", [0.1, 0.202, 0.3])
    )
    with pytest.raises(ValueError, match="time grid differs"):
        postprocess.validate_result_history(
            off_grid,
            settings,
            path=tmp_path / "off-grid.txt",
        )


def _write_simple_time_result(path: Path, time: list[float]) -> Path:
    write_result(path, ["Time~[s]", "Load~[N]"], [[value, index] for index, value in enumerate(time)])
    return path


def write_campaign_fixture(tmp_path: Path) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = strict_settings()
    filter_text = "Time [s]\nLoad [N]\n"
    settings["result_filter_sha256"] = hashlib.sha256(filter_text.encode()).hexdigest()
    (tmp_path / "campaign_settings.json").write_text(
        json.dumps(settings),
        encoding="utf-8",
    )
    for scenario in settings["scenario_names"]:
        case_dir = tmp_path / scenario / "cases" / "U05p00ms"
        result_dir = tmp_path / scenario / "results" / "U05p00ms"
        case_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        result_filter = case_dir / "result.filter"
        result_filter.write_text(filter_text, encoding="utf-8")
        result_path = result_dir / "result.txt"
        write_result(
            result_path,
            ["Time~[s]", "Load~[N]"],
            [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]],
        )
        manifest = {
            "cases": [
                {
                    "label": "U05p00ms",
                    "wind_speed": 5.0,
                    "azimuth_deg": 270.0,
                    "result_name": "result",
                    "result_dir": str(result_dir),
                    "result_filter": str(result_filter),
                }
            ]
        }
        (tmp_path / scenario / "cases.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
    return tmp_path, settings


def test_load_campaign_requires_complete_scenario_case_matrix(tmp_path: Path) -> None:
    run_root, settings = write_campaign_fixture(tmp_path)
    cases = postprocess.load_campaign(
        run_root,
        ["baseline", "vorlap"],
        settings=settings,
        required_columns=["Time~[s]", "Load~[N]"],
    )
    assert [(case.scenario, case.label) for case in cases] == [
        ("baseline", "U05p00ms"),
        ("vorlap", "U05p00ms"),
    ]

    (run_root / "vorlap" / "cases.json").write_text('{"cases": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="case matrix is incomplete"):
        postprocess.load_campaign(
            run_root,
            ["baseline", "vorlap"],
            settings=settings,
            required_columns=["Time~[s]"],
        )


def test_load_campaign_rejects_missing_scenario_and_result(tmp_path: Path) -> None:
    run_root, settings = write_campaign_fixture(tmp_path)
    (run_root / "vorlap" / "cases.json").unlink()
    with pytest.raises(FileNotFoundError, match="Missing scenario manifest"):
        postprocess.load_campaign(run_root, ["baseline", "vorlap"], settings=settings)

    run_root, settings = write_campaign_fixture(tmp_path / "missing-result")
    (run_root / "baseline" / "results" / "U05p00ms" / "result.txt").unlink()
    with pytest.raises(FileNotFoundError):
        postprocess.load_campaign(run_root, ["baseline"], settings=settings)


def test_load_campaign_allows_legacy_unfiltered_plan_but_not_missing_planned_filter(
    tmp_path: Path,
) -> None:
    run_root, settings = write_campaign_fixture(tmp_path)
    manifest_path = run_root / "baseline" / "cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["result_filter"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing its planned result filter"):
        postprocess.load_campaign(run_root, ["baseline"], settings=settings)

    legacy_settings = dict(settings)
    legacy_settings["result_filter_sha256"] = None
    legacy_settings["result_filter"] = None
    cases = postprocess.load_campaign(
        run_root,
        ["baseline"],
        settings=legacy_settings,
        required_columns=["Time~[s]", "Load~[N]"],
    )
    assert len(cases) == 1


@pytest.mark.parametrize(
    "operation",
    [
        lambda values: postprocess.turning_points(values),
        lambda values: postprocess.rainflow_ranges(values),
        lambda values: postprocess.rainflow_damage(values, 10.0),
        lambda values: postprocess.equivalent_load_with_cycles(values, 10.0),
        lambda values: postprocess.rms(values),
    ],
)
def test_fatigue_and_rms_routines_reject_nonfinite_histories(operation) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        operation(np.asarray([0.0, math.nan, 1.0]))


def test_rainflow_matches_frozen_full_range_count_fixture_and_reference() -> None:
    values = np.asarray(
        [0, -150, 100, 0, 130, -120, 110, -50, 80, 0, 150, -100, 100, 0],
        dtype=float,
    )
    expected = (
        (80.0, 1.0),
        (100.0, 1.5),
        (150.0, 0.5),
        (160.0, 1.0),
        (200.0, 0.5),
        (250.0, 1.5),
        (300.0, 0.5),
    )
    ranges, counts = postprocess.rainflow_ranges(values)
    reference_ranges, reference_counts = postprocess.reference_rainflow_ranges(values)
    assert postprocess.aggregate_range_count_multiset(ranges, counts) == expected
    assert postprocess.aggregate_range_count_multiset(reference_ranges, reference_counts) == expected
    damage, cycles, _, _ = postprocess.rainflow_damage(values, 4.0)
    assert cycles == sum(count for _, count in expected)
    assert damage == sum(count * cycle_range**4 for cycle_range, count in expected)


def test_rainflow_damage_exposes_direct_pseudo_damage_before_del_normalization() -> None:
    history = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])
    pseudo_damage_a_m, cycles, ranges, counts = postprocess.rainflow_damage(
        history,
        4.0,
    )
    assert pseudo_damage_a_m == 2.0 * 2.0**4
    assert cycles == 2.0
    assert postprocess.aggregate_range_count_multiset(ranges, counts) == ((2.0, 2.0),)
    del_value, _ = postprocess.equivalent_load_with_cycles(
        history,
        4.0,
        equivalent_cycles=600.0,
    )
    assert del_value == pytest.approx((pseudo_damage_a_m / 600.0) ** 0.25)
    assert postprocess.equivalent_load_from_pseudo_damage(
        pseudo_damage_a_m,
        4.0,
        600.0,
    ) == del_value


@pytest.mark.parametrize(
    ("history", "wohler_exponent"),
    [
        ([-1.0e6, 1.0e6, -1.0e6], 100.0),
        ([-1.0e308, 1.0e308, -1.0e308], 10.0),
    ],
)
def test_rainflow_damage_rejects_finite_inputs_that_overflow(
    history: list[float],
    wohler_exponent: float,
) -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        postprocess.rainflow_damage(np.asarray(history), wohler_exponent)


def test_primary_and_independent_rainflow_multisets_match_deterministic_histories() -> None:
    rng = np.random.default_rng(3108)
    for _ in range(50):
        values = np.cumsum(rng.integers(-5, 6, size=80)).astype(float)
        ranges, counts = postprocess.rainflow_ranges(values)
        reference_ranges, reference_counts = postprocess.reference_rainflow_ranges(values)
        assert postprocess.aggregate_range_count_multiset(
            ranges,
            counts,
        ) == postprocess.aggregate_range_count_multiset(reference_ranges, reference_counts)


def test_embedded_rainflow_validation_uses_complex_full_multiset_cases() -> None:
    validation = postprocess.rainflow_validation_table(10.0, 600.0)
    assert validation["passed"].all()
    assert {"nested_reversals", "astm_style_mixed_history"}.issubset(validation["case"])
