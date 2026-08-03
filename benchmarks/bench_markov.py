"""Benchmark bounded categorical transition-entropy scoring.

Usage:
    uv run python benchmarks/bench_markov.py
    uv run python benchmarks/bench_markov.py --respondents 200000 --items 100 --states 7
    uv run python benchmarks/bench_markov.py --missing-rate 0.1
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np

from ier import markov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 1 or args.states < 1 or args.repeats < 1:
        parser.error("respondents, states, and repeats must be positive")
    if args.items < 3 or args.warmup < 0:
        parser.error("items must be at least 3 and warmup cannot be negative")
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least 0 and less than 1")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(1, args.states + 1, size=(args.respondents, args.items)).astype(float)
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan

    for _ in range(args.warmup):
        markov(data)

    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(args.repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = markov(data)
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    if not np.isfinite(result).any():
        raise RuntimeError("benchmark produced no finite transition entropy")

    print(
        f"shape={data.shape} states={args.states} missing_rate={args.missing_rate} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    print(
        f"markov: median={statistics.median(timings):.4f}s "
        f"peak={statistics.median(peaks) / 1024 / 1024:.1f} MiB"
    )


if __name__ == "__main__":
    main()
