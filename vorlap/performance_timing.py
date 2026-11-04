"""Opt-in aggregate timing for long-running VorLap--QBlade cases.

The profiler is intentionally inert unless ``VORLAP_HVAWT_PROFILE=1``.  It
retains aggregate totals plus a bounded deterministic reservoir for
representative percentiles; it never emits per-timestep output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import statistics
import time
from typing import Any


PROFILE_ENVIRONMENT_VARIABLE = "VORLAP_HVAWT_PROFILE"
PROFILE_SCHEMA = "vorlap-hvawt-aggregate-performance-v1"
DEFAULT_SAMPLE_LIMIT = 2048


def profiling_requested(environment: dict[str, str] | None = None) -> bool:
    env = os.environ if environment is None else environment
    return str(env.get(PROFILE_ENVIRONMENT_VARIABLE, "")).strip() == "1"


@dataclass
class _StageAggregate:
    count: int = 0
    total_ns: int = 0
    minimum_ns: int | None = None
    maximum_ns: int = 0
    samples_ns: list[int] = field(default_factory=list)


class AggregatePerformanceTiming:
    """Bounded, deterministic aggregate wall-clock timing accumulator."""

    def __init__(
        self,
        *,
        sample_limit: int = DEFAULT_SAMPLE_LIMIT,
        started_ns: int | None = None,
    ) -> None:
        if isinstance(sample_limit, bool) or int(sample_limit) < 1:
            raise ValueError("Aggregate timing sample_limit must be positive.")
        self.sample_limit = int(sample_limit)
        now_ns = time.perf_counter_ns()
        self.started_ns = now_ns if started_ns is None else int(started_ns)
        if self.started_ns > now_ns:
            raise ValueError("Aggregate timing start cannot be in the future.")
        self._stages: dict[str, _StageAggregate] = {}

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    def record_since(self, stage: str, started_ns: int) -> None:
        self.record_ns(stage, time.perf_counter_ns() - int(started_ns))

    def record_ns(self, stage: str, elapsed_ns: int) -> None:
        name = str(stage).strip()
        value = int(elapsed_ns)
        if not name:
            raise ValueError("Aggregate timing stage must be nonempty.")
        if value < 0:
            raise ValueError("Aggregate timing duration must be non-negative.")
        row = self._stages.setdefault(name, _StageAggregate())
        row.count += 1
        row.total_ns += value
        row.minimum_ns = value if row.minimum_ns is None else min(row.minimum_ns, value)
        row.maximum_ns = max(row.maximum_ns, value)
        if len(row.samples_ns) < self.sample_limit:
            row.samples_ns.append(value)
        else:
            # Hash the observation count before reducing it.  A plain
            # multiplicative map reduced by that same count always selects
            # slot zero; this deterministic mixer retains observations from
            # the full run without global random state.
            hashed_count = row.count & 0xFFFFFFFF
            hashed_count ^= hashed_count >> 16
            hashed_count = (hashed_count * 0x7FEB352D) & 0xFFFFFFFF
            hashed_count ^= hashed_count >> 15
            hashed_count = (hashed_count * 0x846CA68B) & 0xFFFFFFFF
            hashed_count ^= hashed_count >> 16
            slot = hashed_count % row.count
            if slot < self.sample_limit:
                row.samples_ns[slot] = value

    @staticmethod
    def _percentile(sorted_values: list[int], probability: float) -> float | None:
        if not sorted_values:
            return None
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        position = probability * (len(sorted_values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return float(
            sorted_values[lower]
            + fraction * (sorted_values[upper] - sorted_values[lower])
        )

    def as_dict(self) -> dict[str, Any]:
        wall_ns = max(0, time.perf_counter_ns() - self.started_ns)
        rows: dict[str, dict[str, int | float | None]] = {}
        for name, row in sorted(self._stages.items()):
            samples = sorted(row.samples_ns)
            rows[name] = {
                "call_count": row.count,
                "total_seconds": row.total_ns * 1.0e-9,
                "mean_seconds": (
                    None if row.count == 0 else row.total_ns * 1.0e-9 / row.count
                ),
                "minimum_seconds": (
                    None if row.minimum_ns is None else row.minimum_ns * 1.0e-9
                ),
                "sampled_median_seconds": (
                    None if not samples else statistics.median(samples) * 1.0e-9
                ),
                "sampled_p95_seconds": (
                    None
                    if not samples
                    else float(self._percentile(samples, 0.95)) * 1.0e-9
                ),
                "maximum_seconds": row.maximum_ns * 1.0e-9,
                "sample_count": len(samples),
                "fraction_of_profile_wall": (
                    0.0 if wall_ns == 0 else row.total_ns / wall_ns
                ),
            }
        return {
            "schema": PROFILE_SCHEMA,
            "enabled": True,
            "clock": "time.perf_counter_ns",
            "sample_limit_per_stage": self.sample_limit,
            "profile_wall_seconds": wall_ns * 1.0e-9,
            "stages": rows,
        }


__all__ = [
    "AggregatePerformanceTiming",
    "DEFAULT_SAMPLE_LIMIT",
    "PROFILE_ENVIRONMENT_VARIABLE",
    "PROFILE_SCHEMA",
    "profiling_requested",
]
