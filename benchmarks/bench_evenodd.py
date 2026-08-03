"""Benchmark multi-factor even–odd consistency scoring.

Usage:
    uv run python benchmarks/bench_evenodd.py
    uv run python benchmarks/bench_evenodd.py --respondents 200000 --factors 30
    uv run python benchmarks/bench_evenodd.py --factors 1 --factor-items 120 --missing-rate 0.1
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np

from ier import evenodd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--factors", type=int, default=20)
    parser.add_argument("--factor-items", type=int, default=4)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 1 or args.factors < 1 or args.repeats < 1:
        parser.error("respondents, factors, and repeats must be positive")
    if args.factor_items < 4 or args.warmup < 0:
        parser.error("factor-items must be at least 4 and warmup cannot be negative")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    n_items = args.factors * args.factor_items
    data = rng.integers(1, 6, size=(args.respondents, n_items)).astype(float)
    if args.missing_rate > 0.0:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    factors = [args.factor_items] * args.factors

    for _ in range(args.warmup):
        evenodd(data, factors)

    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(args.repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        scored = evenodd(data, factors)
        if not isinstance(scored, np.ndarray):
            raise RuntimeError("benchmark expected score-only even-odd output")
        result = scored
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    if not np.isfinite(result).all():
        raise RuntimeError("benchmark produced non-finite even-odd scores")

    print(
        f"shape={data.shape} factors={args.factors} factor_items={args.factor_items} "
        f"missing_rate={args.missing_rate} repeats={args.repeats} warmup={args.warmup}"
    )
    print(
        f"evenodd: median={statistics.median(timings):.4f}s "
        f"peak={statistics.median(peaks) / 1024 / 1024:.1f} MiB"
    )


if __name__ == "__main__":
    main()
