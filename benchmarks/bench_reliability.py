"""Benchmark bounded split-half individual reliability scoring.

Usage:
    uv run python benchmarks/bench_reliability.py
    uv run python benchmarks/bench_reliability.py --respondents 200000 --splits 50
    uv run python benchmarks/bench_reliability.py --order F --missing-rate 0
    uv run python benchmarks/bench_reliability.py --items 48 --factor-items 8
"""

from __future__ import annotations

import argparse

import numpy as np
from _measurement import measure

from ier import individual_reliability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--categories", type=int, default=5)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument(
        "--structure", choices=("categorical", "constant", "near-constant"), default="categorical"
    )
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--factor-items",
        type=int,
        default=0,
        help="Score scale-aware halves of this many items per factor (default: legacy splits)",
    )
    args = parser.parse_args()

    if (
        args.respondents < 1
        or args.items < 4
        or args.categories < 2
        or args.splits < 1
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error(
            "respondents, splits, and repeats must be positive, items at least 4, "
            "categories at least 2, and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least 0 and less than 1")
    factors = None
    if args.factor_items:
        if (
            args.factor_items < 1
            or args.items % args.factor_items
            or args.items < 2 * args.factor_items
        ):
            parser.error("factor-items must divide items into at least two factors")
        factors = [args.factor_items] * (args.items // args.factor_items)

    rng = np.random.default_rng(args.seed)
    data = rng.integers(
        1,
        args.categories + 1,
        size=(args.respondents, args.items),
    ).astype(float)
    if args.structure == "constant":
        data.fill(1.1)
    elif args.structure == "near-constant":
        data *= np.spacing(1.1)
        data += 1.1
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    if args.order == "F":
        data = np.asfortranarray(data)

    for _ in range(args.warmup):
        individual_reliability(
            data,
            n_splits=args.splits,
            random_seed=args.split_seed,
            factors=factors,
        )

    measurement = measure(
        lambda: individual_reliability(
            data, n_splits=args.splits, random_seed=args.split_seed, factors=factors
        ),
        args.repeats,
    )
    result = measurement.result
    if result.shape != (args.respondents,) or np.isinf(result).any() or np.any(result > 1):
        raise RuntimeError("benchmark produced invalid reliability scores")
    if factors is not None and np.any(result < -1):
        raise RuntimeError("benchmark produced scale-aware reliability below -1")
    if args.structure == "constant" and not np.isnan(result).all():
        raise RuntimeError("benchmark assigned reliability to constant responses")
    print(
        f"shape={data.shape} order={args.order} categories={args.categories} splits={args.splits} "
        f"structure={args.structure} missing_rate={args.missing_rate} repeats={args.repeats} "
        f"warmup={args.warmup} factor_items={args.factor_items} "
        f"available_scores={np.count_nonzero(np.isfinite(result))}"
    )
    print(
        f"individual_reliability: median={measurement.median_seconds:.4f}s "
        f"peak={measurement.peak_mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
