"""Benchmark complete-data lz scoring and discrimination estimation.

Usage:
    uv run python benchmarks/bench_lz.py
    uv run python benchmarks/bench_lz.py --respondents 20000 --items 100 --repeats 5
    uv run python benchmarks/bench_lz.py --respondents 5000 --items 1000 \
        --discrimination-only
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np

from ier import lz
from ier.lz import _estimate_discrimination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discrimination-only", action="store_true")
    args = parser.parse_args()

    if args.respondents < 2 or args.items < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents and items must be at least 2; repeats must be positive; "
            "warmup cannot be negative"
        )

    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, 2, size=(args.respondents, args.items)).astype(float)
    data[0] = 0.0
    data[1] = 1.0
    scorer = _estimate_discrimination if args.discrimination_only else lz
    label = "discrimination" if args.discrimination_only else "lz"

    for _ in range(args.warmup):
        scorer(data)

    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(args.repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = scorer(data)
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    if not np.isfinite(result).all():
        raise RuntimeError("benchmark produced non-finite lz values")
    print(f"shape={data.shape} repeats={args.repeats}")
    print(
        f"{label}: median={statistics.median(timings):.4f}s "
        f"peak={statistics.median(peaks) / 1024 / 1024:.1f} MiB"
    )


if __name__ == "__main__":
    main()
