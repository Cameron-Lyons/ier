"""Benchmark bounded row-wise response reductions.

Usage:
    uv run python benchmarks/bench_row_reductions.py
    uv run python benchmarks/bench_row_reductions.py --respondents 200000 --items 100
    uv run python benchmarks/bench_row_reductions.py --irv-splits 10 --order F
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
from _measurement import measure

from ier import acquiescence, irv, midpoint_responding, response_pattern, u3_poly

if TYPE_CHECKING:
    from collections.abc import Callable

Score = np.ndarray | dict[str, np.ndarray]


def _matches_availability(result: Score, available: np.ndarray) -> bool:
    arrays = result.values() if isinstance(result, dict) else (result,)
    return all(
        not np.isinf(values).any() and np.array_equal(np.isfinite(values), available)
        for values in arrays
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument(
        "--dtype", choices=("float32", "float64", "int64", "uint64"), default="float64"
    )
    parser.add_argument("--integer-offset", type=int, default=0)
    parser.add_argument(
        "--structure", choices=("categorical", "constant", "near-constant"), default="categorical"
    )
    parser.add_argument(
        "--strict", action="store_true", help="propagate missing values in IRV and acquiescence"
    )
    parser.add_argument("--irv-splits", type=int, help="also benchmark IRV with this section count")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if args.respondents < 1 or args.items < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("respondents, items, and repeats must be positive; warmup must be nonnegative")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if args.irv_splits is not None and args.irv_splits < 1:
        parser.error("irv-splits must be positive")
    integer_input = np.dtype(args.dtype).kind in "iu"
    scale_min, scale_max = 1, 5
    if integer_input:
        if args.missing_rate or args.structure != "categorical":
            parser.error("integer inputs require --missing-rate 0 and --structure categorical")
        scale_min += args.integer_offset
        scale_max += args.integer_offset
        limits = np.iinfo(args.dtype)
        if scale_min < limits.min or scale_max > limits.max:
            parser.error("integer-offset places responses outside the selected dtype")
    elif args.integer_offset:
        parser.error("integer-offset requires an integer dtype")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, 5, size=(args.respondents, args.items)).astype(args.dtype)
    data += scale_min
    if args.structure == "constant":
        data.fill(1.1)
    elif args.structure == "near-constant":
        baseline = data.dtype.type(1.1)
        data *= np.spacing(baseline)
        data += baseline
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    if args.order == "F":
        data = np.asfortranarray(data)

    scorers: dict[str, Callable[[], Score]] = {
        "irv": lambda: irv(data, na_rm=not args.strict),
        "acquiescence": lambda: acquiescence(
            data, scale_min=scale_min, scale_max=scale_max, na_rm=not args.strict
        ),
        "u3_poly": lambda: u3_poly(data, scale_min=scale_min, scale_max=scale_max),
        "midpoint_responding": lambda: midpoint_responding(
            data, scale_min=scale_min, scale_max=scale_max
        ),
        "response_pattern": lambda: response_pattern(
            data, scale_min=scale_min, scale_max=scale_max
        ),
    }
    if args.irv_splits is not None:
        scorers["irv_split"] = lambda: irv(
            data, split=True, num_split=args.irv_splits, na_rm=not args.strict
        )

    available = np.any(~np.isnan(data), axis=1)
    complete = np.all(~np.isnan(data), axis=1)
    split_available = np.ones(args.respondents, dtype=bool)
    if args.irv_splits is not None:
        for section in np.array_split(data, min(args.items, args.irv_splits), axis=1):
            split_available &= np.any(~np.isnan(section), axis=1)

    for _ in range(args.warmup):
        for score in scorers.values():
            score()

    print(
        f"shape={data.shape} dtype={data.dtype} order={args.order} "
        f"structure={args.structure} integer_offset={args.integer_offset} "
        f"missing_rate={args.missing_rate} strict={args.strict} "
        f"irv_splits={args.irv_splits} repeats={args.repeats} warmup={args.warmup}"
    )
    for name, score in scorers.items():
        measured = measure(score, args.repeats)
        expected_available = split_available if name == "irv_split" else available
        if args.strict and name in {"irv", "irv_split", "acquiescence"}:
            expected_available = complete
        if not _matches_availability(measured.result, expected_available):
            raise RuntimeError(f"{name} produced scores with incorrect respondent availability")
        if integer_input:
            if name == "u3_poly":
                np.testing.assert_array_equal(
                    measured.result, np.mean((data == scale_min) | (data == scale_max), axis=1)
                )
            midpoint_scores = (
                measured.result["midpoint"]
                if isinstance(measured.result, dict)
                else measured.result
                if name == "midpoint_responding"
                else None
            )
            if midpoint_scores is not None:
                np.testing.assert_array_equal(
                    midpoint_scores, np.mean(data == scale_min + 2, axis=1)
                )
        if args.structure == "constant":
            variability = (
                measured.result["variability"]
                if isinstance(measured.result, dict)
                else measured.result
                if name in {"irv", "irv_split"}
                else None
            )
            if variability is not None and np.any(variability[expected_available] != 0):
                raise RuntimeError(f"{name} assigned nonzero variability to constant responses")
        print(f"{name}: median={measured.median_seconds:.4f}s peak={measured.peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
