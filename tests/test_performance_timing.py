import json

import pytest

from vorlap.performance_timing import (
    AggregatePerformanceTiming,
    PROFILE_SCHEMA,
    profiling_requested,
)


def test_profiling_is_strictly_opt_in():
    assert not profiling_requested({})
    assert not profiling_requested({"VORLAP_HVAWT_PROFILE": "true"})
    assert profiling_requested({"VORLAP_HVAWT_PROFILE": "1"})


def test_aggregate_timing_is_bounded_and_json_serializable(monkeypatch):
    clock = iter([1_000, 2_000, 3_000])
    monkeypatch.setattr("vorlap.performance_timing.time.perf_counter_ns", lambda: next(clock))
    timing = AggregatePerformanceTiming(sample_limit=2)
    timing.record_ns("adapter", 10)
    timing.record_ns("adapter", 30)
    timing.record_ns("adapter", 20)
    receipt = timing.as_dict()

    assert receipt["schema"] == PROFILE_SCHEMA
    row = receipt["stages"]["adapter"]
    assert row["call_count"] == 3
    assert row["total_seconds"] == pytest.approx(60.0e-9)
    assert row["mean_seconds"] == pytest.approx(20.0e-9)
    assert row["minimum_seconds"] == pytest.approx(10.0e-9)
    assert row["maximum_seconds"] == pytest.approx(30.0e-9)
    assert row["sample_count"] == 2
    json.dumps(receipt, allow_nan=False)


def test_aggregate_timing_reservoir_does_not_only_replace_first_slot():
    timing = AggregatePerformanceTiming(sample_limit=2)
    for value in range(1, 40):
        timing.record_ns("adapter", value)

    assert timing._stages["adapter"].samples_ns[1] != 2


@pytest.mark.parametrize("sample_limit", [0, -1, True])
def test_aggregate_timing_rejects_invalid_sample_limit(sample_limit):
    with pytest.raises(ValueError, match="positive"):
        AggregatePerformanceTiming(sample_limit=sample_limit)


def test_aggregate_timing_rejects_future_start(monkeypatch):
    monkeypatch.setattr("vorlap.performance_timing.time.perf_counter_ns", lambda: 20)
    with pytest.raises(ValueError, match="future"):
        AggregatePerformanceTiming(started_ns=21)
