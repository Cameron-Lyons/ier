"""Benchmark bounded person–total correlation scoring.

Usage:
    uv run python benchmarks/bench_person_total.py
    uv run python benchmarks/bench_person_total.py --respondents 200000 --missing-rate 0.05
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
from _measurement import measure

from ier import person_total


def _check_scores(data: np.ndarray, scores: np.ndarray, *, na_rm: bool) -> None:
    """Check sampled correlations against the direct paired-observation definition."""
    if not na_rm and np.isnan(data).any():
        np.testing.assert_array_equal(scores, np.full(len(data), np.nan))
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

    rng = np.random.default_rng(args.seed)
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
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    data = np.array(data, order=args.order)
    na_rm = not args.strict

    for _ in range(args.warmup):
        person_total(data, na_rm=na_rm)

    measurement = measure(lambda: person_total(data, na_rm=na_rm), args.repeats)
    result = measurement.result
    _check_scores(data, result, na_rm=na_rm)

    print(
        f"shape={data.shape} missing_rate={args.missing_rate} "
        f"order={args.order} na_rm={na_rm} repeats={args.repeats} warmup={args.warmup} "
        f"n_valid={np.count_nonzero(np.isfinite(result))}"
    )
    print(
        f"person_total: median={measurement.median_seconds:.4f}s "
        f"peak={measurement.peak_mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
