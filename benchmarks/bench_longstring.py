"""Benchmark bounded numeric longest-run and repeating-pattern scoring.

Usage:
    uv run python benchmarks/bench_longstring.py
    uv run python benchmarks/bench_longstring.py --respondents 200000 --items 100
    uv run python benchmarks/bench_longstring.py --missing-rate 0.1
    uv run python benchmarks/bench_longstring.py --items 120 --max-pattern-length 12
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

from ier import longstring_pattern, longstring_scores

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
    if not np.isfinite(result).all():
        raise RuntimeError("benchmark produced non-finite longstring scores")
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--categories", type=int, default=5)
    parser.add_argument("--max-pattern-length", type=int, default=5)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if (
        args.respondents < 1
        or args.items < 2
        or args.categories < 2
        or args.max_pattern_length < 2
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error(
            "respondents and repeats must be positive; items, categories, and "
            "max-pattern-length must be at least 2; warmup must be nonnegative"
        )
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least 0 and less than 1")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(
        1,
        args.categories + 1,
        size=(args.respondents, args.items),
    ).astype(float)
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan

    scorers: dict[str, Callable[[], np.ndarray]] = {
        "longstring_scores": lambda: longstring_scores(data),
        "longstring_pattern": lambda: longstring_pattern(
            data,
            max_pattern_length=args.max_pattern_length,
        ),
    }
    for _ in range(args.warmup):
        for score in scorers.values():
            score()

    print(
        f"shape={data.shape} categories={args.categories} "
        f"max_pattern_length={args.max_pattern_length} missing_rate={args.missing_rate} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    for name, score in scorers.items():
        elapsed, peak = _measure(score, args.repeats)
        print(f"{name}: median={elapsed:.4f}s peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()
