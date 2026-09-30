"""Benchmark shared fixed and percentile flagging.

Usage:
    uv run python benchmarks/bench_flagging.py
    uv run python benchmarks/bench_flagging.py --respondents 2000000 --missing-rate 0.2
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING

import numpy as np
from _measurement import measure

from ier._flagging import threshold_flags

if TYPE_CHECKING:
    from collections.abc import Callable


def _reference_cutoff(scores: np.ndarray, percentile: float) -> float:
    """Evaluate the sorted linear definition with exact endpoint arithmetic."""
    observed = np.sort(scores[~np.isnan(scores)])
    if not len(observed):
        return 0.0
    position = (len(observed) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(observed) - 1)
    with localcontext() as context:
        context.prec = 800
        fraction = Decimal.from_float(position) - lower
        left = Decimal.from_float(float(observed[lower]))
        right = Decimal.from_float(float(observed[upper]))
        return float(left * (1 - fraction) + right * fraction)


def _measure(
    operation: Callable[[], np.ndarray], repeats: int, expected: np.ndarray
) -> tuple[float, float]:
    measurement = measure(operation, repeats)
    result = measurement.result
    if result.dtype != np.bool_:
        raise RuntimeError("flagging benchmark produced a non-Boolean result")
    np.testing.assert_array_equal(result, expected)
    return measurement.median_seconds, measurement.peak_mib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=1_000_000)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument(
        "--structure",
        choices=["continuous", "constant", "near-constant", "opposite"],
        default="continuous",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--direction", choices=["high", "low"], default="high")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if args.respondents < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("respondents and repeats must be positive; warmup must be nonnegative")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("scale must be positive and finite")
    if not np.isfinite(args.offset):
        parser.error("offset must be finite")
    if not np.isfinite(args.threshold):
        parser.error("threshold must be finite")
    if not np.isfinite(args.percentile) or not 0.0 <= args.percentile <= 100.0:
        parser.error("percentile must be finite and between 0 and 100")

    rng = np.random.default_rng(args.seed)
    if args.structure == "constant":
        scores = np.full(args.respondents, 1.1)
    elif args.structure == "near-constant":
        scores = 1.1 + rng.integers(-2, 3, size=args.respondents) * np.spacing(1.1)
    elif args.structure == "opposite":
        scores = np.resize([-1.0, 1.0], args.respondents)
    else:
        scores = rng.normal(size=args.respondents)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            scores *= args.scale
            scores += args.offset
    except FloatingPointError:
        parser.error("scale and offset must produce finite scores")
    if args.missing_rate:
        scores[rng.random(scores.size) < args.missing_rate] = np.nan

    operations: dict[str, Callable[[], np.ndarray]] = {
        "fixed": lambda: threshold_flags(
            scores,
            threshold=args.threshold,
            percentile=args.percentile,
            direction=args.direction,
        ),
        "percentile": lambda: threshold_flags(
            scores,
            threshold=None,
            percentile=args.percentile,
            direction=args.direction,
        ),
    }

    for _ in range(args.warmup):
        for operation in operations.values():
            operation()

    print(
        f"respondents={args.respondents} missing_rate={args.missing_rate} "
        f"structure={args.structure} scale={args.scale} offset={args.offset} "
        f"direction={args.direction} repeats={args.repeats} warmup={args.warmup}"
    )
    cutoffs = {"fixed": args.threshold, "percentile": _reference_cutoff(scores, args.percentile)}
    for name, operation in operations.items():
        cutoff = cutoffs[name]
        expected = (
            (scores >= cutoff if name == "fixed" else scores > cutoff)
            if args.direction == "high"
            else (scores <= cutoff if name == "fixed" else scores < cutoff)
        )
        elapsed, peak_mib = _measure(operation, args.repeats, expected)
        print(f"{name}: median={elapsed:.6f}s peak={peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
