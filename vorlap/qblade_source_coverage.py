"""Typed QBlade/VorLap source-coverage diagnostic rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


SOURCE_COVERAGE_COLUMNS = (
    "component",
    "airfoil_id",
    "reynolds_model",
    "samples",
    "node_samples",
    "aoa_min_deg",
    "aoa_max_deg",
    "source_aoa_min_deg",
    "source_aoa_max_deg",
    "re_min",
    "re_max",
    "source_re_min",
    "source_re_max",
    "freq_min_hz",
    "freq_max_hz",
    "unfiltered_freq_max_hz",
    "frequency_truncated_count",
    "nyquist_hz",
    "nyquist_ratio",
    "max_nodal_force_n",
    "force_coefficient_max_abs",
    "max_response_nodal_force_n",
    "response_force_coefficient_max_abs",
    "max_total_nodal_force_n",
    "total_force_coefficient_max_abs",
    "source_force_coefficient_max_abs",
    "max_structural_velocity_mps",
    "force_power_mean_w",
    "force_power_max_abs_w",
    "re_lower_bound_warning_count",
    "re_out_of_range_count",
    "aoa_out_of_range_count",
    "aliased_frequency_count",
    "source_clip_count",
    "force_coefficient_out_of_range_count",
    "nonfinite_count",
)

SOURCE_COVERAGE_INTEGER_COLUMNS = (
    "samples",
    "node_samples",
    "frequency_truncated_count",
    "re_lower_bound_warning_count",
    "re_out_of_range_count",
    "aoa_out_of_range_count",
    "aliased_frequency_count",
    "source_clip_count",
    "force_coefficient_out_of_range_count",
    "nonfinite_count",
)

SOURCE_COVERAGE_FORBIDDEN_COUNTS = (
    "re_out_of_range_count",
    "aoa_out_of_range_count",
    "aliased_frequency_count",
    "source_clip_count",
    "force_coefficient_out_of_range_count",
    "nonfinite_count",
)

HVAWT_V8_COMPONENT_NODE_COUNTS = {
    "BLD_1": 5,
    "BLD_2": 5,
    "BLD_3": 5,
    "STR_1_1": 4,
    "STR_1_2": 4,
    "STR_1_3": 4,
    "STR_2_1": 4,
    "STR_2_2": 4,
    "STR_2_3": 4,
}


@dataclass(frozen=True)
class SourceCoverageRow:
    """Typed source-coverage diagnostic row with stable summary/CSV keys."""

    component: str
    airfoil_id: str
    reynolds_model: str
    samples: int | None
    node_samples: int | None
    aoa_min_deg: float | None
    aoa_max_deg: float | None
    source_aoa_min_deg: float | None
    source_aoa_max_deg: float | None
    re_min: float | None
    re_max: float | None
    source_re_min: float | None
    source_re_max: float | None
    freq_min_hz: float | None
    freq_max_hz: float | None
    unfiltered_freq_max_hz: float | None
    frequency_truncated_count: int | None
    nyquist_hz: float | None
    nyquist_ratio: float | None
    max_nodal_force_n: float | None
    force_coefficient_max_abs: float | None
    max_response_nodal_force_n: float | None
    response_force_coefficient_max_abs: float | None
    max_total_nodal_force_n: float | None
    total_force_coefficient_max_abs: float | None
    source_force_coefficient_max_abs: float | None
    max_structural_velocity_mps: float | None
    force_power_mean_w: float | None
    force_power_max_abs_w: float | None
    re_lower_bound_warning_count: int | None
    re_out_of_range_count: int | None
    aoa_out_of_range_count: int | None
    aliased_frequency_count: int | None
    source_clip_count: int | None
    force_coefficient_out_of_range_count: int | None
    nonfinite_count: int | None

    def as_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in SOURCE_COVERAGE_COLUMNS}

    @staticmethod
    def _optional_float(row: Mapping[str, Any], key: str) -> float | None:
        if key not in row or row.get(key) in (None, ""):
            return None
        return float(row[key])

    @staticmethod
    def _optional_int(row: Mapping[str, Any], key: str) -> int | None:
        if key not in row or row.get(key) in (None, ""):
            return None
        return int(row[key])

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SourceCoverageRow":
        return cls(
            component=str(row.get("component", "")),
            airfoil_id=str(row.get("airfoil_id", "")),
            reynolds_model=str(row.get("reynolds_model", "")),
            samples=cls._optional_int(row, "samples"),
            node_samples=cls._optional_int(row, "node_samples"),
            aoa_min_deg=cls._optional_float(row, "aoa_min_deg"),
            aoa_max_deg=cls._optional_float(row, "aoa_max_deg"),
            source_aoa_min_deg=cls._optional_float(row, "source_aoa_min_deg"),
            source_aoa_max_deg=cls._optional_float(row, "source_aoa_max_deg"),
            re_min=cls._optional_float(row, "re_min"),
            re_max=cls._optional_float(row, "re_max"),
            source_re_min=cls._optional_float(row, "source_re_min"),
            source_re_max=cls._optional_float(row, "source_re_max"),
            freq_min_hz=cls._optional_float(row, "freq_min_hz"),
            freq_max_hz=cls._optional_float(row, "freq_max_hz"),
            unfiltered_freq_max_hz=cls._optional_float(row, "unfiltered_freq_max_hz"),
            frequency_truncated_count=cls._optional_int(row, "frequency_truncated_count"),
            nyquist_hz=cls._optional_float(row, "nyquist_hz"),
            nyquist_ratio=cls._optional_float(row, "nyquist_ratio"),
            max_nodal_force_n=cls._optional_float(row, "max_nodal_force_n"),
            force_coefficient_max_abs=cls._optional_float(row, "force_coefficient_max_abs"),
            max_response_nodal_force_n=cls._optional_float(
                row, "max_response_nodal_force_n"
            ),
            response_force_coefficient_max_abs=cls._optional_float(
                row, "response_force_coefficient_max_abs"
            ),
            max_total_nodal_force_n=cls._optional_float(
                row, "max_total_nodal_force_n"
            ),
            total_force_coefficient_max_abs=cls._optional_float(
                row, "total_force_coefficient_max_abs"
            ),
            source_force_coefficient_max_abs=cls._optional_float(row, "source_force_coefficient_max_abs"),
            max_structural_velocity_mps=cls._optional_float(row, "max_structural_velocity_mps"),
            force_power_mean_w=cls._optional_float(row, "force_power_mean_w"),
            force_power_max_abs_w=cls._optional_float(row, "force_power_max_abs_w"),
            re_lower_bound_warning_count=cls._optional_int(row, "re_lower_bound_warning_count"),
            re_out_of_range_count=cls._optional_int(row, "re_out_of_range_count"),
            aoa_out_of_range_count=cls._optional_int(row, "aoa_out_of_range_count"),
            aliased_frequency_count=cls._optional_int(row, "aliased_frequency_count"),
            source_clip_count=cls._optional_int(row, "source_clip_count"),
            force_coefficient_out_of_range_count=cls._optional_int(row, "force_coefficient_out_of_range_count"),
            nonfinite_count=cls._optional_int(row, "nonfinite_count"),
        )

    @classmethod
    def normalize_mapping(cls, row: Mapping[str, Any]) -> dict[str, object]:
        normalized = cls.from_mapping(row).as_dict()
        extras = {str(key): value for key, value in row.items() if str(key) not in SOURCE_COVERAGE_COLUMNS}
        normalized.update(extras)
        return normalized


def validate_complete_source_coverage(
    rows: Any,
    *,
    expected_samples: int,
    component_node_counts: Mapping[str, int],
) -> None:
    """Fail closed on terminal per-component source-coverage receipts."""

    if not isinstance(rows, list) or not rows:
        raise ValueError("Source coverage must be a nonempty list.")
    expected_components = set(component_node_counts)
    observed_components: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(SOURCE_COVERAGE_COLUMNS):
            raise ValueError("Source coverage row columns differ from the schema.")
        component = raw.get("component")
        if (
            not isinstance(component, str)
            or component not in expected_components
            or component in observed_components
        ):
            raise ValueError("Source coverage component set is invalid.")
        observed_components.add(component)
        for name in ("airfoil_id", "reynolds_model"):
            if not isinstance(raw.get(name), str) or not raw[name]:
                raise ValueError(f"Source coverage {name} is invalid.")
        for name in SOURCE_COVERAGE_INTEGER_COLUMNS:
            value = raw.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Source coverage {name} is invalid.")
        if raw["samples"] != expected_samples:
            raise ValueError("Source coverage samples do not match completed steps.")
        if raw["node_samples"] != (
            expected_samples * int(component_node_counts[component])
        ):
            raise ValueError("Source coverage node_samples is invalid.")
        for name in SOURCE_COVERAGE_COLUMNS:
            if name in {"component", "airfoil_id", "reynolds_model"}:
                continue
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Source coverage {name} is not numeric.")
            if not math.isfinite(float(value)):
                raise ValueError(f"Source coverage {name} is nonfinite.")
        if any(raw[name] != 0 for name in SOURCE_COVERAGE_FORBIDDEN_COUNTS):
            raise ValueError("Source coverage contains a forbidden count.")
        ordered_pairs = (
            ("aoa_min_deg", "aoa_max_deg"),
            ("source_aoa_min_deg", "source_aoa_max_deg"),
            ("re_min", "re_max"),
            ("source_re_min", "source_re_max"),
            ("freq_min_hz", "freq_max_hz"),
            ("freq_max_hz", "unfiltered_freq_max_hz"),
        )
        if any(float(raw[low]) > float(raw[high]) for low, high in ordered_pairs):
            raise ValueError("Source coverage range ordering is invalid.")
        nonnegative = (
            "re_min",
            "source_re_min",
            "freq_min_hz",
            "nyquist_hz",
            "nyquist_ratio",
            "max_nodal_force_n",
            "force_coefficient_max_abs",
            "max_response_nodal_force_n",
            "response_force_coefficient_max_abs",
            "max_total_nodal_force_n",
            "total_force_coefficient_max_abs",
            "source_force_coefficient_max_abs",
            "max_structural_velocity_mps",
            "force_power_max_abs_w",
        )
        if any(float(raw[name]) < 0.0 for name in nonnegative):
            raise ValueError("Source coverage nonnegative quantity is invalid.")
        frequency = float(raw["freq_max_hz"])
        nyquist = float(raw["nyquist_hz"])
        if nyquist <= 0.0 or frequency > nyquist + 64.0 * math.ulp(nyquist):
            raise ValueError("Source coverage retained frequency exceeds Nyquist.")
        expected_ratio = nyquist / frequency if frequency > 1.0e-12 else 1.0e12
        if not math.isclose(
            float(raw["nyquist_ratio"]),
            expected_ratio,
            rel_tol=8.0 * math.ulp(1.0),
            abs_tol=8.0 * math.ulp(expected_ratio),
        ):
            raise ValueError("Source coverage Nyquist ratio is inconsistent.")
        coefficient_tolerance = 64.0 * math.ulp(
            max(1.0, float(raw["source_force_coefficient_max_abs"]))
        )
        if (
            float(raw["force_coefficient_max_abs"])
            > float(raw["source_force_coefficient_max_abs"])
            + coefficient_tolerance
        ):
            raise ValueError("Source coverage force coefficient exceeds its source envelope.")
        if abs(float(raw["force_power_mean_w"])) > (
            float(raw["force_power_max_abs_w"])
            + 64.0 * math.ulp(max(1.0, float(raw["force_power_max_abs_w"])))
        ):
            raise ValueError("Source coverage force-power summary is inconsistent.")
    if observed_components != expected_components:
        raise ValueError("Source coverage component set is incomplete.")


def validate_complete_velocity_diagnostics(
    diagnostics: Any,
    *,
    expected_samples: int,
    node_count: int,
) -> None:
    """Require all 3-D structural velocity channels to be live and sampled."""

    if not isinstance(diagnostics, Mapping):
        raise ValueError("Velocity diagnostics are missing.")
    if diagnostics.get("velocity_channel_count") != 3 * node_count:
        raise ValueError("Velocity diagnostic channel count is invalid.")
    if diagnostics.get("all_zero_axes") != []:
        raise ValueError("A structural velocity axis is identically zero.")
    if diagnostics.get("all_zero_channels") != []:
        raise ValueError("A structural velocity channel is identically zero.")
    axis_stats = diagnostics.get("axis_stats")
    if not isinstance(axis_stats, Mapping) or set(axis_stats) != {"X", "Y", "Z"}:
        raise ValueError("Velocity axis diagnostics are invalid.")
    required_axis_samples = expected_samples * node_count
    for axis in ("X", "Y", "Z"):
        row = axis_stats[axis]
        if not isinstance(row, Mapping) or row.get("samples") != required_axis_samples:
            raise ValueError("Velocity axis sample count is invalid.")
        if row.get("all_zero") is not False:
            raise ValueError("A structural velocity axis is identically zero.")
        for name in ("min", "max", "max_abs"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Velocity axis statistic is invalid.")
            if not math.isfinite(float(value)):
                raise ValueError("Velocity axis statistic is nonfinite.")
        minimum = float(row["min"])
        maximum = float(row["max"])
        maximum_abs = float(row["max_abs"])
        if minimum > maximum or maximum_abs <= 0.0:
            raise ValueError("Velocity axis statistic ordering is invalid.")
        required_maximum_abs = max(abs(minimum), abs(maximum))
        if not math.isclose(
            maximum_abs,
            required_maximum_abs,
            rel_tol=64.0 * math.ulp(1.0),
            abs_tol=64.0 * math.ulp(max(1.0, required_maximum_abs)),
        ):
            raise ValueError("Velocity axis maximum magnitude is inconsistent.")
    channel_stats = diagnostics.get("channel_stats")
    expected_channel_count = 3 * node_count
    if not isinstance(channel_stats, Mapping) or len(channel_stats) != expected_channel_count:
        raise ValueError("Velocity channel diagnostics are incomplete.")
    observed_axis_counts = {axis: 0 for axis in ("X", "Y", "Z")}
    for variable, row in channel_stats.items():
        if not isinstance(variable, str) or len(variable) < 3 or variable[0] not in observed_axis_counts:
            raise ValueError("Velocity channel name is invalid.")
        axis = variable[0]
        observed_axis_counts[axis] += 1
        if not isinstance(row, Mapping) or row.get("samples") != expected_samples:
            raise ValueError("Velocity channel sample count is invalid.")
        if row.get("all_zero") is not False:
            raise ValueError("A structural velocity channel is identically zero.")
        values = [row.get(name) for name in ("min", "max", "max_abs")]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("Velocity channel statistic is invalid.")
        minimum, maximum, maximum_abs = (float(value) for value in values)
        expected_max_abs = max(abs(minimum), abs(maximum))
        if (
            minimum > maximum
            or maximum_abs <= 0.0
            or not math.isclose(
                maximum_abs,
                expected_max_abs,
                rel_tol=64.0 * math.ulp(1.0),
                abs_tol=64.0 * math.ulp(max(1.0, expected_max_abs)),
            )
        ):
            raise ValueError("Velocity channel statistic ordering is invalid.")
    if set(observed_axis_counts.values()) != {node_count}:
        raise ValueError("Velocity channel axes are incomplete.")


__all__ = [
    "SOURCE_COVERAGE_COLUMNS",
    "HVAWT_V8_COMPONENT_NODE_COUNTS",
    "SOURCE_COVERAGE_FORBIDDEN_COUNTS",
    "SOURCE_COVERAGE_INTEGER_COLUMNS",
    "SourceCoverageRow",
    "validate_complete_source_coverage",
    "validate_complete_velocity_diagnostics",
]
