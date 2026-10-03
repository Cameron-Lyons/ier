"""Benchmark bounded person–total correlation scoring.

Usage:
    uv run python benchmarks/bench_person_total.py
    uv run python benchmarks/bench_person_total.py --respondents 200000 --missing-rate 0.05
    uv run python benchmarks/bench_person_total.py --structure near-constant --order F
    uv run python benchmarks/bench_person_total.py --structure categorical --scale 5e-324
"""

from __future__ import annotations

import argparse
import warnings
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
from _measurement import measure

from ier import person_total


def _check_exact_scores(data: np.ndarray, scores: np.ndarray) -> None:
    """Check sampled Pearson scores with exact item means and Decimal moments."""
    profile: list[Fraction | None] = []
    for column in data.T:
        observed = column[~np.isnan(column)]
        categories, counts = np.unique(observed, return_counts=True)
        total = sum(
            (
                Fraction(float(value)) * int(count)
                for value, count in zip(categories, counts, strict=True)
            ),
            start=Fraction(),
        )
        profile.append(total / len(observed) if len(observed) else None)

    sampled = np.linspace(0, len(data) - 1, min(len(data), 32), dtype=np.intp)
    expected = np.full(len(sampled), np.nan)
    with localcontext() as context:
        context.prec = 800
        for position, row in enumerate(data[sampled]):
            pairs = [
                (Fraction(float(value)), mean)
                for value, mean in zip(row, profile, strict=True)
                if not np.isnan(value) and mean is not None
            ]
            if len(pairs) < 2:
                continue
            values, means = zip(*pairs, strict=True)
            center_left = sum(values, start=Fraction()) / len(values)
            center_right = sum(means, start=Fraction()) / len(means)
            left = [value - center_left for value in values]
            right = [mean - center_right for mean in means]
            covariance = sum((a * b for a, b in zip(left, right, strict=True)), start=Fraction())
            squares = sum((value * value for value in left), start=Fraction()) * sum(
                (value * value for value in right), start=Fraction()
            )
            if squares == 0:
                continue
            numerator = Decimal(covariance.numerator) / Decimal(covariance.denominator)
            variance = Decimal(squares.numerator) / Decimal(squares.denominator)
            expected[position] = float(numerator / variance.sqrt())
    np.testing.assert_allclose(scores[sampled], expected, rtol=1e-12, atol=1e-13)


def _check_scores(
    data: np.ndarray, scores: np.ndarray, *, na_rm: bool, exact: bool = False
) -> None:
    """Check sampled correlations against the direct paired-observation definition."""
    if not na_rm and np.isnan(data).any():
        np.testing.assert_array_equal(scores, np.full(len(data), np.nan))
        return
    if exact:
        _check_exact_scores(data, scores)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmean(data, axis=0) if na_rm else np.mean(data, axis=0)
    sampled = np.linspace(0, len(data) - 1, min(len(data), 32), dtype=np.intp)
    expected = np.full(len(sampled), np.nan)
    for position, row in enumerate(data[sampled]):
        paired = ~(np.isnan(row) | np.isnan(profile))
        left, right = row[paired], profile[paired]
        if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
            continue
        left = left - left.mean()
        right = right - right.mean()
        expected[position] = (left @ right) / np.linalg.norm(left) / np.linalg.norm(right)
    np.testing.assert_allclose(scores[sampled], expected, rtol=1e-12, atol=1e-13)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--order", choices=["C", "F"], default="C")
    parser.add_argument(
        "--structure",
        choices=["continuous", "categorical", "constant", "near-constant"],
        default="continuous",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true", help="Propagate missing responses")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 2 or args.items < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents and items must be at least 2, repeats positive, and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if not np.isfinite(args.scale) or args.scale <= 0 or not np.isfinite(args.offset):
        parser.error("scale must be positive and finite; offset must be finite")

    rng = np.random.default_rng(args.seed)
    if args.structure == "continuous":
        latent = rng.normal(size=(args.respondents, 1))
        item_profile = rng.normal(scale=0.4, size=(1, args.items))
        data = (
            latent
            + item_profile
            + rng.normal(
                scale=0.8,
                size=(args.respondents, args.items),
            )
        )
    elif args.structure == "constant":
        data = np.full((args.respondents, args.items), 1.1)
    else:
        data = rng.integers(0, 13, size=(args.respondents, args.items)).astype(float)
        if args.structure == "near-constant":
            data *= np.spacing(1.1)
            data += 1.1
    with np.errstate(over="ignore", under="ignore"):
        data *= args.scale
        data += args.offset
    if not np.isfinite(data).all():
        parser.error("scale and offset place responses outside the finite float range")
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    data = np.array(data, order=args.order)
    na_rm = not args.strict

    for _ in range(args.warmup):
        person_total(data, na_rm=na_rm)

    measurement = measure(lambda: person_total(data, na_rm=na_rm), args.repeats)
    result = measurement.result
    _check_scores(
        data,
        result,
        na_rm=na_rm,
        exact=args.structure != "continuous" or args.scale != 1 or args.offset != 0,
    )

    print(
        f"shape={data.shape} missing_rate={args.missing_rate} "
        f"order={args.order} na_rm={na_rm} repeats={args.repeats} warmup={args.warmup} "
        f"structure={args.structure} scale={args.scale} offset={args.offset} "
        f"n_valid={np.count_nonzero(np.isfinite(result))}"
    )
    print(
        f"person_total: median={measurement.median_seconds:.4f}s "
        f"peak={measurement.peak_mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
