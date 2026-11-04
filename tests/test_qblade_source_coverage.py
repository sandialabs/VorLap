import copy

import pytest

from vorlap.qblade_source_coverage import (
    HVAWT_V8_COMPONENT_NODE_COUNTS,
    SOURCE_COVERAGE_COLUMNS,
    SOURCE_COVERAGE_INTEGER_COLUMNS,
    validate_complete_source_coverage,
    validate_complete_velocity_diagnostics,
)


def _coverage(steps=8):
    rows = []
    for component, node_count in HVAWT_V8_COMPONENT_NODE_COUNTS.items():
        row = {name: 0.0 for name in SOURCE_COVERAGE_COLUMNS}
        row.update(
            component=component,
            airfoil_id="NACA0018:test",
            reynolds_model="replicated_reynolds_independent",
        )
        for name in SOURCE_COVERAGE_INTEGER_COLUMNS:
            row[name] = 0
        row["samples"] = steps
        row["node_samples"] = steps * node_count
        row["nyquist_hz"] = 10.0
        row["nyquist_ratio"] = 1.0e12
        rows.append(row)
    return rows


def _velocity(steps=8):
    row = {
        "samples": 39 * steps,
        "min": -1.0,
        "max": 1.0,
        "max_abs": 1.0,
        "all_zero": False,
    }
    return {
        "velocity_channel_count": 117,
        "axis_stats": {axis: dict(row) for axis in ("X", "Y", "Z")},
        "channel_stats": {
            f"{axis}_g Vel. node {index:02d} [m/s]": {
                "samples": steps,
                "min": -1.0,
                "max": 1.0,
                "max_abs": 1.0,
                "all_zero": False,
            }
            for axis in ("X", "Y", "Z")
            for index in range(39)
        },
        "all_zero_axes": [],
        "all_zero_channels": [],
    }


def test_complete_hvawt_coverage_and_velocity_receipts_pass_exact_schema():
    validate_complete_source_coverage(
        _coverage(),
        expected_samples=8,
        component_node_counts=HVAWT_V8_COMPONENT_NODE_COUNTS,
    )
    validate_complete_velocity_diagnostics(
        _velocity(), expected_samples=8, node_count=39
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[0].pop("nonfinite_count"), "columns differ"),
        (lambda rows: rows[0].__setitem__("samples", 7), "completed steps"),
        (lambda rows: rows[0].__setitem__("node_samples", 1), "node_samples"),
        (lambda rows: rows[0].__setitem__("source_clip_count", 1), "forbidden"),
        (lambda rows: rows.append(dict(rows[0])), "component set"),
    ],
)
def test_complete_hvawt_coverage_rejects_malformed_receipts(mutation, message):
    rows = copy.deepcopy(_coverage())
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        validate_complete_source_coverage(
            rows,
            expected_samples=8,
            component_node_counts=HVAWT_V8_COMPONENT_NODE_COUNTS,
        )


def test_complete_hvawt_velocity_rejects_all_zero_warning():
    velocity = _velocity()
    velocity["all_zero_channels"] = ["Vx at node"]
    with pytest.raises(ValueError, match="identically zero"):
        validate_complete_velocity_diagnostics(
            velocity, expected_samples=8, node_count=39
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].__setitem__("aoa_min_deg", 2.0),
        lambda rows: rows[0].__setitem__("freq_max_hz", 20.0),
        lambda rows: rows[0].__setitem__("nyquist_ratio", 999.0),
    ],
)
def test_complete_hvawt_coverage_rejects_inconsistent_derived_fields(mutation):
    rows = _coverage()
    mutation(rows)
    with pytest.raises(ValueError):
        validate_complete_source_coverage(
            rows,
            expected_samples=8,
            component_node_counts=HVAWT_V8_COMPONENT_NODE_COUNTS,
        )


def test_complete_hvawt_velocity_rejects_incomplete_or_inconsistent_channels():
    velocity = _velocity()
    velocity["channel_stats"].pop(next(iter(velocity["channel_stats"])))
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete_velocity_diagnostics(
            velocity, expected_samples=8, node_count=39
        )
    velocity = _velocity()
    next(iter(velocity["channel_stats"].values()))["max_abs"] = 2.0
    with pytest.raises(ValueError, match="ordering"):
        validate_complete_velocity_diagnostics(
            velocity, expected_samples=8, node_count=39
        )
