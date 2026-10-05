"""Benchmark item-step person-fit statistics beside Guttman error counting.

Usage:
    OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_person_fit.py
    OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_person_fit.py --missing-rate 0.02
    OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_person_fit.py \
        --respondents 55000 --items 20 --categories 2 --operations ht

Operations run in alternating order. Wall-clock timings exclude allocation
tracing, and peak traced allocation is measured in a separate run. A few
respondents are checked against the step-pair and covariance definitions
outside the timed region. ``ht`` scores the responses dichotomized at the
scale midpoint.
"""

from __future__ import annotations

import argparse
import platform
from typing import TYPE_CHECKING

import numpy as np
from _measurement import measure_many

from ier import gpoly, guttman, ht, u3poly

if TYPE_CHECKING:
    from collections.abc import Callable

_OPERATIONS = ("gpoly", "gpoly_raw", "u3poly", "ht", "guttman")


def _check_step_errors(data: np.ndarray, raw: np.ndarray, n_steps: int) -> None:
    """Compare a few raw counts with explicit pairs of popularity-ordered steps."""
    observed = ~np.isnan(data)
    with np.errstate(invalid="ignore"):
        popularity = np.stack(
            [
                np.sum(data >= level, axis=0) / np.sum(observed, axis=0)
                for level in range(1, n_steps + 1)
            ],
            axis=1,
        )
    order = np.argsort(-popularity, axis=None, kind="stable")
    items, levels = np.divmod(order, n_steps)
    for row in range(min(8, len(data))):
        if not observed[row].any():
            continue
        steps = data[row, items]
        passed = steps >= levels + 1
        failed = ~np.isnan(steps) & ~passed
        errors = sum(int(np.sum(failed[:position])) for position in np.flatnonzero(passed))
        if raw[row] != errors:
            raise RuntimeError(
                f"gpoly counted {raw[row]} errors for respondent {row}, not {errors}"
            )


def _check_ht(binary: np.ndarray, scores: np.ndarray) -> None:
    """Compare a few complete respondents with their summed pairwise covariances."""
    complete = np.flatnonzero(~np.isnan(binary).any(axis=1))
    sample = binary[complete].astype(np.int64)
    n_items = sample.shape[1]
    totals = sample.sum(axis=1)
    for position in range(min(8, len(sample))):
        products = totals[position] * totals
        covariance = n_items * (sample @ sample[position]) - products
        maximum = n_items * np.minimum(totals[position], totals) - products
        covariance[position] = maximum[position] = 0
        expected = covariance.sum() / maximum.sum() if maximum.sum() > 0 else np.nan
        if not np.array_equal(scores[complete[position]], expected, equal_nan=True):
            raise RuntimeError(f"ht differs from its definition for respondent {position}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=22_000)
    parser.add_argument("--items", type=int, default=30)
    parser.add_argument("--categories", type=int, default=5)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=_OPERATIONS,
        help="Measure only selected operations (default: all)",
    )
    args = parser.parse_args()

    if args.respondents < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("respondents and repeats must be positive and warmup nonnegative")
    if args.items < 2 or not 2 <= args.categories <= 1024:
        parser.error("items must be at least 2 and categories from 2 to 1024")
    if not 0 <= args.missing_rate < 1:
        parser.error("missing-rate must be in [0, 1)")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, args.categories, size=(args.respondents, args.items)).astype(float)
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    binary = np.where(np.isnan(data), np.nan, data >= args.categories / 2)
    ncat = args.categories
    operations: dict[str, Callable[[], np.ndarray]] = {
        "gpoly": lambda: gpoly(data, ncat=ncat, scale_min=0),
        "gpoly_raw": lambda: gpoly(data, ncat=ncat, scale_min=0, normalize=False),
        "u3poly": lambda: u3poly(data, ncat=ncat, scale_min=0),
        "ht": lambda: ht(binary),
        "guttman": lambda: guttman(data),
    }
    if args.operations is not None:
        operations = {name: operations[name] for name in _OPERATIONS if name in args.operations}

    for _ in range(args.warmup):
        for operation in operations.values():
            operation()
    measurements = measure_many(operations, args.repeats)

    if "gpoly_raw" in measurements:
        _check_step_errors(data, measurements["gpoly_raw"].result, args.categories - 1)
    if "ht" in measurements:
        _check_ht(binary, measurements["ht"].result)
    for name in ("gpoly", "u3poly"):
        if name in measurements:
            scores = measurements[name].result
            if np.any((scores < 0) | (scores > 1)):
                raise RuntimeError(f"{name} produced scores outside [0, 1]")

    print(f"Python {platform.python_version()} / NumPy {np.__version__}")
    print(
        f"shape={data.shape} categories={args.categories} missing_rate={args.missing_rate} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    for name, measurement in measurements.items():
        best = min(measurement.timings)
        print(
            f"{name}: median={measurement.median_seconds:.4f}s min={best:.4f}s "
            f"peak={measurement.peak_mib:.1f} MiB"
        )


if __name__ == "__main__":
    main()
