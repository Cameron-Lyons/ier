"""Ensure performance reports separate timing, allocation, and result validation."""

from __future__ import annotations

import tracemalloc
from unittest.mock import patch

import pytest
from benchmarks._measurement import measure, measure_many


def test_timing_excludes_tracing_and_retains_last_timed_result() -> None:
    tracing_states = []

    def operation() -> int:
        tracing_states.append(tracemalloc.is_tracing())
        return len(tracing_states)

    with (
        patch("benchmarks._measurement.time.perf_counter", side_effect=[1.0, 1.25, 2.0, 2.75]),
        patch("benchmarks._measurement.gc.collect") as collect,
    ):
        result = measure(operation, repeats=2)

    assert tracing_states == [False, False, True]
    assert result.timings == (0.25, 0.75)
    assert result.median_seconds == 0.5
    assert result.result == 2
    assert collect.call_count == 3
    assert not tracemalloc.is_tracing()


def test_comparisons_alternate_order_and_trace_each_operation_separately() -> None:
    calls = []

    def record(name: str) -> str:
        calls.append((name, tracemalloc.is_tracing()))
        return name

    with patch("benchmarks._measurement.gc.collect"):
        measured = measure_many({"left": lambda: record("a"), "right": lambda: record("b")}, 3)

    assert calls == [
        ("a", False),
        ("b", False),
        ("b", False),
        ("a", False),
        ("a", False),
        ("b", False),
        ("a", True),
        ("b", True),
    ]
    assert list(measured) == ["left", "right"]
    assert measured["left"].result == "a"
    assert measured["right"].result == "b"
    assert all(len(result.timings) == 3 for result in measured.values())


def test_peak_allocation_measures_one_call_in_mib() -> None:
    size = 512 * 1024
    with patch("benchmarks._measurement.gc.collect"):
        result = measure(lambda: bytearray(size), 2)
    assert len(result.result) == size
    assert 0.5 <= result.peak_mib < 0.6


def test_previous_result_destruction_stays_outside_the_timer() -> None:
    timed = False
    destruction_states = []

    class Result:
        def __del__(self) -> None:
            destruction_states.append(timed)

    def clock() -> float:
        nonlocal timed
        timed = not timed
        return 0.0

    with (
        patch("benchmarks._measurement.time.perf_counter", side_effect=clock),
        patch("benchmarks._measurement.gc.collect"),
    ):
        result = measure(Result, 3)
    assert destruction_states == [False, False, False]
    assert isinstance(result.result, Result)


def test_none_results_are_supported() -> None:
    with patch("benchmarks._measurement.gc.collect"):
        result = measure(lambda: None, 1)
    assert result.result is None


@pytest.mark.parametrize("fail_on", [1, 3])
def test_operation_failure_propagates_and_does_not_leave_tracing_active(fail_on: int) -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise LookupError("measurement failed")

    with (
        patch("benchmarks._measurement.gc.collect"),
        pytest.raises(LookupError, match="measurement failed"),
    ):
        measure(operation, 2)
    assert not tracemalloc.is_tracing()


def test_existing_tracing_is_rejected_and_preserved() -> None:
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="disable allocation tracing"):
            measure(lambda: None, 1)
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("repeats", [0, -1])
def test_invalid_repeats_do_not_run_the_operation(repeats: int) -> None:
    with patch("builtins.print") as operation, pytest.raises(ValueError, match="positive"):
        measure(operation, repeats)
    operation.assert_not_called()


def test_empty_comparison_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one operation"):
        measure_many({}, 1)
