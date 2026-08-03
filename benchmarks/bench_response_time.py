"""Benchmark Gaussian-mixture response-time scoring.

Usage:
    uv run python benchmarks/bench_response_time.py
    uv run python benchmarks/bench_response_time.py --respondents 200000 --items 40
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

from ier import response_time_mixture
from ier.response_time import _em_gaussian_mixture

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(
    operation: Callable[[], np.ndarray],
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        operation()

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
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("benchmark produced invalid mixture probabilities")
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < args.components:
        parser.error("respondents must be at least the component count")
    if args.items < 1 or args.components < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "items and repeats must be positive; components must be at least 2; "
            "warmup cannot be negative"
        )

    rng = np.random.default_rng(args.seed)
    fast = rng.lognormal(mean=-0.7, sigma=0.2, size=args.respondents // 5)
    regular = rng.lognormal(mean=1.2, sigma=0.35, size=args.respondents - len(fast))
    medians = np.concatenate((fast, regular))
    rng.shuffle(medians)
    log_medians = np.log(medians)
    item_noise = rng.lognormal(mean=0.0, sigma=0.15, size=(args.respondents, args.items))
    timings = medians[:, None] * item_noise

    core_seconds, core_peak = _measure(
        lambda: _em_gaussian_mixture(
            log_medians,
            args.components,
            np.random.default_rng(args.seed),
        ),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    workflow_seconds, workflow_peak = _measure(
        lambda: response_time_mixture(
            timings,
            n_components=args.components,
            random_seed=args.seed,
        ),
        repeats=args.repeats,
        warmup=args.warmup,
    )

    print(
        f"shape={timings.shape} components={args.components} repeats={args.repeats} "
        f"warmup={args.warmup}"
    )
    print(f"EM core: median={core_seconds:.4f}s peak={core_peak:.1f} MiB")
    print(f"public workflow: median={workflow_seconds:.4f}s peak={workflow_peak:.1f} MiB")


if __name__ == "__main__":
    main()
