"""Benchmark bounded person–total correlation scoring.

Usage:
    uv run python benchmarks/bench_person_total.py
    uv run python benchmarks/bench_person_total.py --respondents 200000 --missing-rate 0.05
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np

from ier import person_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 2 or args.items < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents and items must be at least 2, repeats positive, and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least 0 and less than 1")

    rng = np.random.default_rng(args.seed)
    latent = rng.normal(size=(args.respondents, 1))
    item_profile = rng.normal(scale=0.4, size=(1, args.items))
    data = (
        latent
        + item_profile
        + rng.normal(
            scale=0.8,
            size=(args.respondents, args.items),
        )
    )
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan

    for _ in range(args.warmup):
        person_total(data)

    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(args.repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = person_total(data)
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    if not np.isfinite(result).any():
        raise RuntimeError("benchmark produced no finite person–total correlations")

    print(
        f"shape={data.shape} missing_rate={args.missing_rate} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    print(
        f"person_total: median={statistics.median(timings):.4f}s "
        f"peak={statistics.median(peaks) / 1024 / 1024:.1f} MiB"
    )


if __name__ == "__main__":
    main()
