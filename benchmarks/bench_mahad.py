"""Benchmark bounded Mahalanobis distance scoring.

Usage:
    uv run python benchmarks/bench_mahad.py
    uv run python benchmarks/bench_mahad.py --respondents 200000 --items 100
    uv run python benchmarks/bench_mahad.py --respondents 20000 --items 80 --qqplot
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np

from ier import mahad, mahad_qqplot


def _measure(data: np.ndarray, repeats: int, *, qqplot: bool) -> tuple[float, float]:
    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        if qqplot:
            theoretical, observed = mahad_qqplot(data)
            if theoretical.shape != observed.shape:
                raise RuntimeError("benchmark produced misaligned Q-Q values")
            result = theoretical
        else:
            scored = mahad(data)
            if not isinstance(scored, np.ndarray):
                raise RuntimeError("benchmark expected distance-only output")
            result = scored
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    if not np.isfinite(result).all():
        raise RuntimeError("benchmark produced non-finite distances")
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--qqplot", action="store_true")
    args = parser.parse_args()

    if args.respondents < args.items or args.items < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents must be at least items, items and repeats positive, and warmup nonnegative"
        )

    rng = np.random.default_rng(args.seed)
    data = rng.normal(size=(args.respondents, args.items))

    for _ in range(args.warmup):
        if args.qqplot:
            mahad_qqplot(data)
        else:
            mahad(data)

    seconds, peak = _measure(data, args.repeats, qqplot=args.qqplot)
    print(f"shape={data.shape} repeats={args.repeats} warmup={args.warmup} seed={args.seed}")
    operation = "mahad_qqplot" if args.qqplot else "mahad"
    print(f"{operation}: median={seconds:.4f}s peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()
