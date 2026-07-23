#!/usr/bin/env python3
"""Benchmark default screen() on synthetic survey matrices.

Usage:
    uv run python benchmarks/bench_screen.py
    uv run python benchmarks/bench_screen.py --respondents 2000 --items 50 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from ier import screen


def _make_data(n_respondents: int, n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 6, size=(n_respondents, n_items)).astype(float)
    # Inject a few pathological rows
    data[0, :] = 3.0
    if n_items >= 6:
        data[1, :] = np.tile([1.0, 5.0], n_items // 2 + 1)[:n_items]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=500)
    parser.add_argument("--items", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = _make_data(args.respondents, args.items, args.seed)

    for _ in range(args.warmup):
        screen(data, scale_min=1, scale_max=5)

    timings: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = screen(data, scale_min=1, scale_max=5)
        timings.append(time.perf_counter() - start)

    print(f"shape={data.shape} indices={result['n_indices']}")
    print(
        "screen seconds: "
        f"median={statistics.median(timings):.4f} "
        f"mean={statistics.mean(timings):.4f} "
        f"min={min(timings):.4f} max={max(timings):.4f}"
    )


if __name__ == "__main__":
    main()
