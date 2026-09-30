"""Benchmark respondent-level screen and composite reductions.

Usage:
    uv run python benchmarks/bench_orchestration.py
    uv run python benchmarks/bench_orchestration.py --respondents 1000000 --indices 30
"""

from __future__ import annotations

import argparse
import math
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING

import numpy as np
from _measurement import measure

from ier._statistics import logistic_transform
from ier.composite import _combine_scores, _standardize_index_scores
from ier.screen import _reduce_screen_results

if TYPE_CHECKING:
    from ier.types import ScreenIndexSummary


def _check_results(
    scores: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    composite_result: np.ndarray,
    screen_result: tuple[np.ndarray, np.ndarray, dict[str, ScreenIndexSummary]],
    args: argparse.Namespace,
    weights: dict[str, float] | None,
) -> None:
    """Check calibration, summaries, and sampled reductions outside measurement."""
    sampled = np.linspace(0, args.respondents - 1, min(args.respondents, 32), dtype=np.intp)
    sampled_scores: dict[str, np.ndarray] = {}
    expected_counts = np.zeros(len(sampled), dtype=np.intp)
    expected_flags = np.zeros(len(sampled), dtype=np.intp)
    for name, values in scores.items():
        observed = values[~np.isnan(values)]
        constant = bool(len(observed) and np.all(observed == observed[0]))
        prepared = _standardize_index_scores(values) if args.standardize else values
        if args.standardize and len(observed) > 1:
            usable = prepared[~np.isnan(prepared)]
            if constant:
                np.testing.assert_array_equal(usable, np.zeros(len(usable)))
            else:
                np.testing.assert_allclose(np.mean(usable), 0, atol=1e-13)
                np.testing.assert_allclose(np.std(usable), 1, rtol=1e-13)
        sampled_scores[name] = prepared[sampled]
        expected_counts += ~np.isnan(values[sampled])
        expected_flags += flags[name][sampled]
        stats = screen_result[2][name]
        assert stats["n_valid"] == len(observed)
        assert stats["n_flagged"] == np.count_nonzero(flags[name])
        if len(observed):
            assert np.isfinite(stats["mean"]) and np.isfinite(stats["std"])
            assert stats["min"] <= stats["mean"] <= stats["max"]
            if constant:
                assert stats["std"] == 0
            else:
                assert stats["std"] >= 0
                lower_bound = (stats["max"] / 2 - stats["min"] / 2) / math.sqrt(len(observed))
                if lower_bound > 0:
                    assert stats["std"] > 0
        else:
            assert np.isnan([stats["mean"], stats["std"], stats["min"], stats["max"]]).all()
    np.testing.assert_array_equal(screen_result[0][sampled], expected_flags)
    np.testing.assert_array_equal(screen_result[1][sampled], expected_counts)

    expected = np.full(len(sampled), np.nan)
    with localcontext() as context:
        context.prec = 800
        for row in range(len(sampled)):
            weighted_values: list[Decimal] = []
            masses: list[Decimal] = []
            for name, prepared in sampled_scores.items():
                if np.isnan(prepared[row]):
                    continue
                weight = Decimal.from_float(weights[name] if weights is not None else 1.0)
                masses.append(weight)
                weighted_values.append(Decimal.from_float(float(prepared[row])) * weight)
            if args.min_valid_indices is not None and len(weighted_values) < args.min_valid_indices:
                continue
            if args.method == "sum":
                expected[row] = float(sum(weighted_values, Decimal(0)))
            elif weighted_values:
                expected[row] = float(
                    sum(weighted_values) / sum(masses)
                    if args.method == "mean"
                    else max(weighted_values)
                )
    np.testing.assert_allclose(composite_result[sampled], expected, rtol=1e-13, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=500_000)
    parser.add_argument("--indices", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--missing-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--structure", choices=["continuous", "constant", "near-constant"], default="continuous"
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--method", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Apply deterministic nonuniform composite weights",
    )
    parser.add_argument(
        "--min-valid-indices",
        type=int,
        default=None,
        help="Mask respondents with fewer than this many available scores",
    )
    args = parser.parse_args()

    if args.respondents < 1 or args.indices < 1 or args.repeats < 1:
        parser.error("respondents, indices, and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if not np.isfinite(args.scale) or args.scale <= 0 or not np.isfinite(args.offset):
        parser.error("scale must be positive and finite; offset must be finite")
    if not np.isfinite(args.weight_scale) or args.weight_scale <= 0:
        parser.error("weight-scale must be positive and finite")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if args.min_valid_indices is not None and not 1 <= args.min_valid_indices <= args.indices:
        parser.error("min-valid-indices must be between 1 and the index count")

    rng = np.random.default_rng(args.seed)
    scores: dict[str, np.ndarray] = {}
    flags: dict[str, np.ndarray] = {}
    for index in range(args.indices):
        name = f"index_{index}"
        values = rng.normal(size=args.respondents)
        if args.structure == "constant":
            values.fill(1.1)
        elif args.structure == "near-constant":
            values = 1.1 + rng.integers(0, 5, args.respondents) * np.spacing(1.1)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                values *= args.scale
                values += args.offset
            except FloatingPointError:
                parser.error("scale and offset produce non-finite scores")
        values[rng.random(args.respondents) < args.missing_rate] = np.nan
        scores[name] = values
        flags[name] = (rng.random(args.respondents) < 0.05) & ~np.isnan(values)
    weights = (
        {
            name: (0.5 + index / max(args.indices - 1, 1)) * args.weight_scale
            for index, name in enumerate(scores)
        }
        if args.weighted or args.weight_scale != 1.0
        else None
    )
    if weights is not None and any(
        not np.isfinite(value) or value <= 0 for value in weights.values()
    ):
        parser.error("weight-scale must produce positive finite weights")

    for _ in range(args.warmup):
        _combine_scores(
            scores,
            {},
            args.method,
            args.standardize,
            weights,
            min_valid_indices=args.min_valid_indices,
        )
        _reduce_screen_results(scores, flags, args.respondents)

    composite_measurement = measure(
        lambda: _combine_scores(
            scores,
            {},
            args.method,
            args.standardize,
            weights,
            min_valid_indices=args.min_valid_indices,
        ),
        args.repeats,
    )
    screen_measurement = measure(
        lambda: _reduce_screen_results(scores, flags, args.respondents),
        args.repeats,
    )
    probability_scores = np.linspace(-1000.0, 1000.0, args.respondents)
    probability_measurement = measure(
        lambda: logistic_transform(probability_scores),
        args.repeats,
    )
    _check_results(
        scores, flags, composite_measurement.result, screen_measurement.result, args, weights
    )

    print(
        f"respondents={args.respondents} indices={args.indices} "
        f"method={args.method} standardize={args.standardize} weighted={args.weighted} "
        f"min_valid_indices={args.min_valid_indices} structure={args.structure} "
        f"scale={args.scale} offset={args.offset} warmup={args.warmup}"
        f" weight_scale={args.weight_scale}"
    )
    print(
        f"composite: median={composite_measurement.median_seconds:.4f}s "
        f"peak={composite_measurement.peak_mib:.1f} MiB"
    )
    print(
        f"logistic transform: median={probability_measurement.median_seconds:.4f}s "
        f"peak={probability_measurement.peak_mib:.1f} MiB"
    )
    print(
        f"screen reductions: median={screen_measurement.median_seconds:.4f}s "
        f"peak={screen_measurement.peak_mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
