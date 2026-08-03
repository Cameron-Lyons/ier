"""Benchmark bounded missing-response rate scoring.

Usage:
    uv run python benchmarks/bench_missing.py
    uv run python benchmarks/bench_missing.py --respondents 500000 --items 100
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

from ier import missing_rate

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(score: Callable[[], np.ndarray], repeats: int) -> tuple[float, float]:
    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = score()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    if not np.isfinite(result).any():
        raise RuntimeError("benchmark produced no finite scores")
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=200_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--applicable-rate", type=float, default=0.75)
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
    if not 0.0 < args.applicable_rate <= 1.0:
        parser.error("applicable-rate must be greater than 0 and at most 1")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(1, 6, size=(args.respondents, args.items)).astype(float)
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    applicable = rng.random(data.shape) < args.applicable_rate
    selected = list(range(0, args.items, 2))

    scorers: dict[str, Callable[[], np.ndarray]] = {
        "all_items": lambda: missing_rate(data),
        "required_subset": lambda: missing_rate(data, item_indices=selected),
        "subset_with_mask": lambda: missing_rate(
            data,
            item_indices=selected,
            applicable_mask=applicable,
        ),
    }

    for _ in range(args.warmup):
        for score in scorers.values():
            score()

    print(
        f"shape={data.shape} selected_items={len(selected)} "
        f"missing_rate={args.missing_rate} applicable_rate={args.applicable_rate} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    for name, score in scorers.items():
        elapsed, peak_mib = _measure(score, args.repeats)
        print(f"{name}: median={elapsed:.4f}s peak={peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
