"""Benchmark shared fixed and percentile flagging.

Usage:
    uv run python benchmarks/bench_flagging.py
    uv run python benchmarks/bench_flagging.py --respondents 2000000 --missing-rate 0.2
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

from ier._flagging import threshold_flags

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
    if result.dtype != np.bool_:
        raise RuntimeError("flagging benchmark produced a non-Boolean result")
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=1_000_000)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--direction", choices=["high", "low"], default="high")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if args.respondents < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("respondents and repeats must be positive; warmup must be nonnegative")
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least 0 and less than 1")
    if not np.isfinite(args.threshold):
        parser.error("threshold must be finite")
    if not np.isfinite(args.percentile) or not 0.0 <= args.percentile <= 100.0:
        parser.error("percentile must be finite and between 0 and 100")

    rng = np.random.default_rng(args.seed)
    scores = rng.normal(size=args.respondents)
    if args.missing_rate:
        scores[rng.random(scores.size) < args.missing_rate] = np.nan

    operations: dict[str, Callable[[], np.ndarray]] = {
        "fixed": lambda: threshold_flags(
            scores,
            threshold=args.threshold,
            percentile=args.percentile,
            direction=args.direction,
        ),
        "percentile": lambda: threshold_flags(
            scores,
            threshold=None,
            percentile=args.percentile,
            direction=args.direction,
        ),
    }

    for _ in range(args.warmup):
        for operation in operations.values():
            operation()

    print(
        f"respondents={args.respondents} missing_rate={args.missing_rate} "
        f"direction={args.direction} repeats={args.repeats} warmup={args.warmup}"
    )
    for name, operation in operations.items():
        elapsed, peak_mib = _measure(operation, args.repeats)
        print(f"{name}: median={elapsed:.6f}s peak={peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
