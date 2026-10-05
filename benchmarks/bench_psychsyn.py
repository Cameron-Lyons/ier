"""Benchmark psychometric synonym scoring and item-correlation discovery.

Timing excludes allocation tracing; peak allocation is measured separately.

Usage:
    uv run python benchmarks/bench_psychsyn.py
    uv run python benchmarks/bench_psychsyn.py --respondents 16000 --items 50
    uv run python benchmarks/bench_psychsyn.py --structure independent --missing-rate 0
    uv run python benchmarks/bench_psychsyn.py --missing-mode scattered --order F
    uv run python benchmarks/bench_psychsyn.py --operation correlations \
        --item-correlations pairwise --missing-mode scattered --missing-rate 0.02 \
        --respondents 100000 --items 50
"""

from __future__ import annotations

import argparse
import platform

import numpy as np
from _measurement import measure

from ier._column_statistics import column_correlations, pairwise_column_correlations
from ier.psychsyn import psychsyn, psychsyn_critval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=8_000)
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--critval", type=float, default=0.6)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--missing-mode", choices=("item", "scattered"), default="item")
    parser.add_argument("--structure", choices=("correlated", "independent"), default="correlated")
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--dtype", choices=("float64", "int64", "uint64"), default="float64")
    parser.add_argument("--integer-offset", type=int, default=0)
    parser.add_argument(
        "--operation",
        choices=("psychsyn", "psychsyn_critval", "correlations"),
        default="psychsyn",
    )
    parser.add_argument("--item-correlations", choices=("complete", "pairwise"), default="complete")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    minimum_items = 1 if args.operation == "correlations" else 3
    if args.respondents < 2 or args.items < minimum_items or args.repeats < 1 or args.warmup < 0:
        parser.error(
            f"respondents must be at least 2, items at least {minimum_items}, repeats positive, "
            "and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if not 0.0 <= args.critval <= 1.0:
        parser.error("critval must be between 0 and 1")
    integer_input = np.dtype(args.dtype).kind in "iu"
    if integer_input and args.missing_rate:
        parser.error("integer inputs require --missing-rate 0")
    if not integer_input and args.integer_offset:
        parser.error("integer-offset requires an integer dtype")

    rng = np.random.default_rng(args.seed)
    data = rng.normal(size=(args.respondents, args.items))
    if args.structure == "correlated":
        data *= 0.1
        data += rng.normal(size=(args.respondents, 1))
    if integer_input:
        categories = np.rint(data * 16).astype(np.int64)
        minimum, maximum = int(np.min(categories)), int(np.max(categories))
        lower, upper = minimum + args.integer_offset, maximum + args.integer_offset
        limits = np.iinfo(args.dtype)
        if lower < limits.min or upper > limits.max:
            parser.error("integer-offset places generated responses outside the selected dtype")
        data = (categories - minimum).astype(args.dtype)
        data += lower
    if args.missing_mode == "item":
        missing_count = round(args.respondents * args.missing_rate)
        if missing_count:
            missing_rows = rng.choice(args.respondents, size=missing_count, replace=False)
            data[missing_rows, 0] = np.nan
    else:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    data = np.array(data, order=args.order)

    def operation() -> np.ndarray | tuple[np.ndarray, np.ndarray] | list[tuple[int, int, float]]:
        pairwise = args.item_correlations == "pairwise"
        if args.operation == "correlations":
            return pairwise_column_correlations(data) if pairwise else column_correlations(data)
        if args.operation == "psychsyn_critval":
            return psychsyn_critval(
                data, min_correlation=args.critval, item_correlations=args.item_correlations
            )
        return psychsyn(
            data, critval=args.critval, diag=True, item_correlations=args.item_correlations
        )

    for _ in range(args.warmup):
        operation()

    measurement = measure(operation, args.repeats)
    if isinstance(measurement.result, np.ndarray):
        correlations = measurement.result
        if args.item_correlations == "complete":
            available = np.isfinite(data).all(axis=0) & np.any(data != data[0], axis=0)
            usable = np.isfinite(correlations)
            expected = available[:, None] & available
        else:
            # Pair availability depends on shared rows; each item needs three varying responses.
            observed = np.count_nonzero(~np.isnan(data), axis=0)
            spread = np.fmax.reduce(data, axis=0, initial=-np.inf) > np.fmin.reduce(
                data, axis=0, initial=np.inf
            )
            usable = np.isfinite(np.diagonal(correlations))
            expected = (observed >= 3) & spread
        if not np.array_equal(usable, expected):
            raise RuntimeError("benchmark produced correlations with incorrect item availability")
        if np.isinf(correlations).any() or np.any(np.abs(correlations) > 1):
            raise RuntimeError("benchmark produced invalid item correlations")
        selected_pairs = int(np.count_nonzero(np.triu(correlations >= args.critval, k=1)))
    elif isinstance(measurement.result, tuple):
        scores, diagnostic = measurement.result
        selected_pairs = int(diagnostic.max(initial=0))
        if np.isinf(scores).any() or not np.array_equal(np.isfinite(scores), diagnostic >= 2):
            raise RuntimeError("benchmark produced scores with incorrect respondent availability")
    else:
        selected_pairs = len(measurement.result)

    print(f"Python {platform.python_version()} / NumPy {np.__version__}")
    print(
        f"shape={data.shape} selected_pairs={selected_pairs} structure={args.structure} "
        f"missing_rate={args.missing_rate} missing_mode={args.missing_mode} order={args.order} "
        f"dtype={args.dtype} integer_offset={args.integer_offset} "
        f"item_correlations={args.item_correlations} "
        f"repeats={args.repeats} warmup={args.warmup} seed={args.seed}"
    )
    print(
        f"{args.operation}: median={measurement.median_seconds:.4f}s "
        f"peak={measurement.peak_mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
