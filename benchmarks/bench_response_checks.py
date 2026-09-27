"""Benchmark missing-response and attention-check scoring.

Usage:
    uv run python benchmarks/bench_response_checks.py
    uv run python benchmarks/bench_response_checks.py --checks 40 --order F

Timing excludes allocation tracing; peak allocation is measured separately.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
import tracemalloc
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ier import infrequency, missing_rate

if TYPE_CHECKING:
    from collections.abc import Callable

    from ier.types import InfrequencyMissingPolicy


def _measure(score: Callable[[], np.ndarray], repeats: int) -> tuple[float, float]:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        score()
        timings.append(time.perf_counter() - started)

    tracemalloc.start()
    try:
        score()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return statistics.median(timings), peak / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--checks", type=int, default=8)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.respondents < 1 or args.items < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("respondents, items, and repeats must be positive; warmup must be nonnegative")
    if not 1 <= args.checks <= args.items:
        parser.error("checks must be between 1 and items")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(1, 6, size=(args.respondents, args.items)).astype(float, order=args.order)
    data[rng.random(data.shape) < args.missing_rate] = np.nan
    applicable = np.asarray(rng.random(data.shape) < 0.8, order=args.order)
    indices = rng.choice(args.items, size=args.checks, replace=False).tolist()
    expected = rng.integers(1, 6, size=args.checks).tolist()
    scorers: dict[str, Callable[[], np.ndarray]] = {
        "missing_all": lambda: missing_rate(data),
        "missing_subset": lambda: missing_rate(data, item_indices=indices),
        "missing_applicable": lambda: missing_rate(data, applicable_mask=applicable),
        "missing_subset_applicable": lambda: missing_rate(
            data, item_indices=indices, applicable_mask=applicable
        ),
    }
    policies: tuple[InfrequencyMissingPolicy, ...] = ("pass", "fail", "omit", "propagate")
    for policy in policies:
        scorers[f"infrequency_{policy}"] = partial(
            infrequency, data, indices, expected, proportion=True, missing=policy
        )

    print(f"Python {platform.python_version()} / NumPy {np.__version__}")
    print(
        f"shape={data.shape} checks={args.checks} order={args.order} "
        f"missing_rate={args.missing_rate} repeats={args.repeats} warmup={args.warmup}"
    )
    for name, score in scorers.items():
        for _ in range(args.warmup):
            score()
        elapsed, peak_mib = _measure(score, args.repeats)
        print(f"{name}: median={elapsed * 1000:.3f}ms peak={peak_mib:.2f} MiB")


if __name__ == "__main__":
    main()
