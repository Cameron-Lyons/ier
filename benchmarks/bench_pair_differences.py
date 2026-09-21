"""Benchmark predefined semantic, MAD, and acquiescence item-pair scoring.

Usage:
    uv run python benchmarks/bench_pair_differences.py
    uv run python benchmarks/bench_pair_differences.py --respondents 200000 --items 100
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

from ier import acquiescence, mad, semantic_ant, semantic_syn

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(operation: Callable[[], np.ndarray], repeats: int) -> tuple[float, float]:
    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    if not np.isfinite(result).any():
        raise RuntimeError("benchmark produced no finite scores")
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if args.respondents < 1 or args.items < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents and repeats must be positive, items at least 2, and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least 0 and less than 1")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(1, 6, size=(args.respondents, args.items)).astype(float)
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan

    n_pairs = args.items // 2
    positive_items = list(range(n_pairs))
    negative_items = list(range(n_pairs, n_pairs * 2))
    pairs = list(zip(positive_items, negative_items, strict=True))
    operations: dict[str, Callable[[], np.ndarray]] = {
        "acquiescence": lambda: acquiescence(
            data,
            scale_min=1,
            scale_max=5,
            positive_items=positive_items,
            negative_items=negative_items,
        ),
        "semantic_syn": lambda: semantic_syn(data, pairs),
        "semantic_ant": lambda: semantic_ant(data, pairs),
        "mad": lambda: mad(data, item_pairs=pairs, scale_min=1, scale_max=5),
    }

    for _ in range(args.warmup):
        for operation in operations.values():
            operation()

    print(
        f"shape={data.shape} pairs={n_pairs} missing_rate={args.missing_rate} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    for name, operation in operations.items():
        seconds, peak = _measure(operation, args.repeats)
        print(f"{name}: median={seconds:.4f}s peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()
