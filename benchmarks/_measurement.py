"""Shared benchmark timing and peak-allocation measurement.

Inputs and warmups belong to the caller. Timed calls run without allocation
tracing; one additional call measures peak traced allocation, not process RSS.
"""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

T = TypeVar("T")


@dataclass(frozen=True)
class Measurement(Generic[T]):
    """Untraced timings, peak allocation in MiB, and the last timed result."""

    timings: tuple[float, ...]
    peak_mib: float
    result: T

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.timings)


def measure(operation: Callable[[], T], repeats: int) -> Measurement[T]:
    """Measure one operation without mixing timing and allocation tracing."""
    return measure_many({"operation": operation}, repeats)["operation"]


def measure_many(
    operations: Mapping[str, Callable[[], T]], repeats: int
) -> dict[str, Measurement[T]]:
    """Measure operations in alternating order, then trace each separately.

    Intermediate results are released outside the timer; the final repetition's
    results remain available for caller-owned correctness checks. Allocation
    samples stay alive until tracing stops. Existing allocation tracing is
    rejected without changing its state.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not operations:
        raise ValueError("at least one operation is required")
    if tracemalloc.is_tracing():
        raise RuntimeError("disable allocation tracing before measuring runtime")

    labels = tuple(operations)
    timings: dict[str, list[float]] = {name: [] for name in labels}
    results: dict[str, T] = {}
    for repeat in range(repeats):
        for name in labels if repeat % 2 == 0 else reversed(labels):
            gc.collect()
            started = time.perf_counter()
            result = operations[name]()
            timings[name].append(time.perf_counter() - started)
            if repeat == repeats - 1:
                results[name] = result
            del result

    measurements = {}
    for name in labels:
        gc.collect()
        tracemalloc.start()
        try:
            allocation_result = operations[name]()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        del allocation_result
        measurements[name] = Measurement(tuple(timings[name]), peak / 1024**2, results[name])
    return measurements
