"""Benchmark psychometric synonym scoring and high-item pair discovery.

Missing responses are confined to one item so correlation discovery still
selects a dense set of pairs among the remaining items. Independent data with a
high cutoff isolates the bounded no-pair discovery path.

Usage:
    uv run python benchmarks/bench_psychsyn.py
    uv run python benchmarks/bench_psychsyn.py --respondents 16000 --items 50
    uv run python benchmarks/bench_psychsyn.py --respondents 200 --items 3000 \
        --critval 0.99 --missing-rate 0 --independent
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np

from ier import psychsyn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=8_000)
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--critval", type=float, default=0.6)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 2 or args.items < 3 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents must be at least 2, items at least 3, repeats positive, "
            "and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if not 0.0 <= args.critval <= 1.0:
        parser.error("critval must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    if args.independent:
        data = rng.normal(size=(args.respondents, args.items))
    else:
        latent = rng.normal(size=(args.respondents, 1))
        data = latent + rng.normal(scale=0.1, size=(args.respondents, args.items))
    missing_count = round(args.respondents * args.missing_rate)
    if missing_count:
        missing_rows = rng.choice(args.respondents, size=missing_count, replace=False)
        data[missing_rows, 0] = np.nan

    for _ in range(args.warmup):
        psychsyn(data, critval=args.critval)

    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    diagnostic: np.ndarray | None = None
    for _ in range(args.repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result, diagnostic = psychsyn(data, critval=args.critval, diag=True)
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None and diagnostic is not None
    if not (np.isfinite(result).all() or np.isnan(result).all()):
        raise RuntimeError("benchmark produced non-finite psychometric synonym scores")

    print(
        f"shape={data.shape} selected_pairs={int(diagnostic.max(initial=0))} "
        f"missing_rate={args.missing_rate} independent={args.independent} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    print(
        f"psychsyn: median={statistics.median(timings):.4f}s "
        f"peak={statistics.median(peaks) / 1024 / 1024:.1f} MiB"
    )


if __name__ == "__main__":
    main()
