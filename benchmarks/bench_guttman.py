"""Benchmark bounded Guttman error scoring.

Usage:
    uv run python benchmarks/bench_guttman.py
    uv run python benchmarks/bench_guttman.py --respondents 200000 --items 100
"""

from __future__ import annotations

import argparse

import numpy as np
from _measurement import measure

from ier import guttman
from ier._column_statistics import column_mean_order


def _measure(data: np.ndarray, repeats: int) -> tuple[float, float]:
    measurement = measure(lambda: guttman(data), repeats)
    result = measurement.result
    counts = np.count_nonzero(~np.isnan(data), axis=1)
    np.testing.assert_array_equal(np.isfinite(result), counts >= 2)
    assert np.all((result[counts >= 2] >= 0) & (result[counts >= 2] <= 1))
    # Verify a few respondents against the direct pair definition using the
    # full sample's item order. Keep quadratic checking outside measurement.
    sample = data[:8, column_mean_order(data, ignore_nan=True)]
    errors = np.zeros(len(sample))
    for column in range(1, data.shape[1]):
        errors += np.count_nonzero(sample[:, :column] < sample[:, column, None], axis=1)
    comparisons = counts[:8] * (counts[:8] - 1.0) / 2
    expected = np.divide(
        errors, comparisons, out=np.full(len(sample), np.nan), where=comparisons > 0
    )
    np.testing.assert_array_equal(result[:8], expected)
    return measurement.median_seconds, measurement.peak_mib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--categories", type=int, default=5)
    parser.add_argument("--structure", choices=("categorical", "continuous"), default="categorical")
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if (
        args.respondents < 1
        or args.items < 2
        or args.categories < 2
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error(
            "respondents and repeats must be positive, items and categories at least 2, "
            "and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    data = (
        rng.normal(size=(args.respondents, args.items))
        if args.structure == "continuous"
        else rng.integers(1, args.categories + 1, size=(args.respondents, args.items)).astype(float)
    )
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    data = np.array(data, order=args.order)

    for _ in range(args.warmup):
        guttman(data)

    seconds, peak = _measure(data, args.repeats)
    print(
        f"shape={data.shape} structure={args.structure} categories={args.categories} "
        f"order={args.order} "
        f"missing_rate={args.missing_rate} repeats={args.repeats} "
        f"warmup={args.warmup}"
    )
    print(f"guttman: median={seconds:.4f}s peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()
