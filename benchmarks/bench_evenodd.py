"""Benchmark multi-factor even–odd consistency scoring.

Usage:
    uv run python benchmarks/bench_evenodd.py
    uv run python benchmarks/bench_evenodd.py --respondents 200000 --factors 30
    uv run python benchmarks/bench_evenodd.py --method halves --factor-items 8 --factors 6
"""

from __future__ import annotations

import argparse

import numpy as np
from _measurement import measure

from ier import evenodd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--factors", type=int, default=20)
    parser.add_argument("--factor-items", type=int, default=4)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--method", choices=("item_pairs", "halves"), default="item_pairs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 1 or args.factors < 1 or args.repeats < 1:
        parser.error("respondents, factors, and repeats must be positive")
    if args.factor_items < 4 or args.warmup < 0:
        parser.error("factor-items must be at least 4 and warmup cannot be negative")
    if args.method == "halves" and args.factors < 2:
        parser.error("the halves method requires at least two factors")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    n_items = args.factors * args.factor_items
    data = rng.integers(1, 6, size=(args.respondents, n_items)).astype(float)
    if args.missing_rate > 0.0:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    if args.order == "F":
        data = np.asfortranarray(data)
    factors = [args.factor_items] * args.factors

    for _ in range(args.warmup):
        evenodd(data, factors, method=args.method)

    measurement = measure(lambda: evenodd(data, factors, method=args.method), args.repeats)
    result = measurement.result
    if args.method == "halves":
        # Scores need two factors with both halves observed; constant half means stay NaN.
        observed = np.zeros(args.respondents, dtype=np.intp)
        for start in range(0, n_items, args.factor_items):
            stop = start + args.factor_items
            observed += np.any(~np.isnan(data[:, start:stop:2]), axis=1) & np.any(
                ~np.isnan(data[:, start + 1 : stop : 2]), axis=1
            )
        finite = np.isfinite(result)
        if np.isinf(result).any() or np.any(finite & (observed < 2)) or np.any(abs(result) > 1):
            raise RuntimeError("benchmark produced invalid half-scale even-odd scores")
    else:
        available = np.zeros(args.respondents, dtype=bool)
        for start in range(0, n_items, args.factor_items):
            stop = start + 2 * (args.factor_items // 2)
            paired = ~np.isnan(data[:, start:stop:2]) & ~np.isnan(data[:, start + 1 : stop : 2])
            available |= np.sum(paired, axis=1) >= 2
        if np.isinf(result).any() or not np.array_equal(np.isfinite(result), available):
            raise RuntimeError("benchmark produced scores with incorrect respondent availability")

    print(
        f"shape={data.shape} order={args.order} method={args.method} factors={args.factors} "
        f"factor_items={args.factor_items} "
        f"missing_rate={args.missing_rate} repeats={args.repeats} warmup={args.warmup}"
    )
    print(f"evenodd: median={measurement.median_seconds:.4f}s peak={measurement.peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
