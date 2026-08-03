"""Benchmark bounded row-wise response reductions.

Usage:
    uv run python benchmarks/bench_row_reductions.py
    uv run python benchmarks/bench_row_reductions.py --respondents 200000 --items 100
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

from ier import acquiescence, irv, midpoint_responding, response_pattern, u3_poly

if TYPE_CHECKING:
    from collections.abc import Callable

Score = np.ndarray | dict[str, np.ndarray]


def _measure(score: Callable[[], Score], repeats: int) -> tuple[float, float, Score]:
    timings: list[float] = []
    peaks: list[int] = []
    result: Score | None = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = score()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024, result


def _has_finite_score(result: Score) -> bool:
    arrays = result.values() if isinstance(result, dict) else (result,)
    return all(np.isfinite(values).any() for values in arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--irv-splits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if (
        args.respondents < 1
        or args.items < 1
        or args.repeats < 1
        or args.warmup < 0
        or args.irv_splits < 1
    ):
        parser.error("respondents, items, and repeats must be positive; warmup must be nonnegative")
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least 0 and less than 1")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(1, 6, size=(args.respondents, args.items)).astype(float)
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan

    scorers: dict[str, Callable[[], Score]] = {
        "irv": lambda: irv(data),
        "irv_sections": lambda: irv(data, split=True, num_split=args.irv_splits),
        "acquiescence": lambda: acquiescence(data, scale_min=1, scale_max=5),
        "u3_poly": lambda: u3_poly(data, scale_min=1, scale_max=5),
        "midpoint_responding": lambda: midpoint_responding(data, scale_min=1, scale_max=5),
        "response_pattern": lambda: response_pattern(data, scale_min=1, scale_max=5),
    }

    for _ in range(args.warmup):
        for score in scorers.values():
            score()

    print(
        f"shape={data.shape} missing_rate={args.missing_rate} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    for name, score in scorers.items():
        elapsed, peak_mib, result = _measure(score, args.repeats)
        if not _has_finite_score(result):
            raise RuntimeError(f"{name} produced no finite scores")
        print(f"{name}: median={elapsed:.4f}s peak={peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
