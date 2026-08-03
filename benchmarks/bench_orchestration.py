"""Benchmark respondent-level screen and composite reductions.

Usage:
    uv run python benchmarks/bench_orchestration.py
    uv run python benchmarks/bench_orchestration.py --respondents 1000000 --indices 30
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

from ier.composite import _combine_scores
from ier.screen import _count_flags

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(operation: Callable[[], np.ndarray], repeats: int) -> tuple[float, float]:
    timings: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=500_000)
    parser.add_argument("--indices", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--missing-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    args = parser.parse_args()

    if args.respondents < 1 or args.indices < 1 or args.repeats < 1:
        parser.error("respondents, indices, and repeats must be positive")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    scores: dict[str, np.ndarray] = {}
    flags: dict[str, np.ndarray] = {}
    for index in range(args.indices):
        name = f"index_{index}"
        values = rng.normal(size=args.respondents)
        values[rng.random(args.respondents) < args.missing_rate] = np.nan
        scores[name] = values
        flags[name] = rng.random(args.respondents) < 0.05

    composite_seconds, composite_peak = _measure(
        lambda: _combine_scores(
            scores,
            {},
            args.method,
            args.standardize,
        ),
        args.repeats,
    )
    screen_seconds, screen_peak = _measure(
        lambda: _count_flags(flags, args.respondents),
        args.repeats,
    )

    print(
        f"respondents={args.respondents} indices={args.indices} "
        f"method={args.method} standardize={args.standardize}"
    )
    print(f"composite: median={composite_seconds:.4f}s peak={composite_peak:.1f} MiB")
    print(f"screen flags: median={screen_seconds:.4f}s peak={screen_peak:.1f} MiB")


if __name__ == "__main__":
    main()
