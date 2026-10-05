"""Benchmark complete and missing-data lz person-fit scoring.

Usage:
    uv run python benchmarks/bench_lz.py
    uv run python benchmarks/bench_lz.py --respondents 20000 --items 100 --repeats 5
    uv run python benchmarks/bench_lz.py --respondents 50000 --items 60
    uv run python benchmarks/bench_lz.py --missing-rate 0.1 --order F
    uv run python benchmarks/bench_lz.py --categories 5 --missing-rate 0.1
    uv run python benchmarks/bench_lz.py --operation discrimination --missing-rate 0.1

Timing excludes allocation tracing; peak allocation is measured separately.
"""

from __future__ import annotations

import argparse
import platform

import numpy as np
from _measurement import measure

from ier import lz
from ier.lz import _dichotomize, _estimate_discrimination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--categories", type=int, default=2)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--model", choices=("1pl", "2pl"), default="2pl")
    parser.add_argument("--operation", choices=("lz", "discrimination"), default="lz")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 2 or args.items < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents and items must be at least 2; repeats must be positive; "
            "warmup cannot be negative"
        )
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be in [0, 1)")
    if args.categories < 2:
        parser.error("categories must be at least 2")
    if args.operation == "discrimination" and args.model != "2pl":
        parser.error("discrimination estimation requires model 2pl")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, args.categories, size=(args.respondents, args.items)).astype(
        float, order=args.order
    )
    data[rng.random(data.shape) < args.missing_rate] = np.nan
    # Keep every item observed for parameter estimation, even at high missingness.
    data[0] = 0.0
    data[1] = args.categories - 1.0

    # Isolate calibration from preprocessing when measuring discrimination alone.
    prepared = _dichotomize(data) if args.operation == "discrimination" else data

    def operation() -> np.ndarray:
        if args.operation == "discrimination":
            return _estimate_discrimination(prepared)
        return lz(data, model=args.model)

    for _ in range(args.warmup):
        operation()

    measurement = measure(operation, args.repeats)
    result = measurement.result
    if args.operation == "discrimination":
        if not np.isfinite(result).all() or np.any((result < 0.2) | (result > 3.0)):
            raise RuntimeError("benchmark produced invalid discrimination estimates")
    else:
        observed = np.any(~np.isnan(data), axis=1)
        if not np.isfinite(result[observed]).all() or not np.isnan(result[~observed]).all():
            raise RuntimeError("benchmark produced invalid lz values")
    print(f"Python {platform.python_version()} / NumPy {np.__version__}")
    print(
        f"shape={data.shape} categories={args.categories} "
        f"missing_rate={args.missing_rate} order={args.order} "
        f"model={args.model} repeats={args.repeats} warmup={args.warmup} seed={args.seed}"
    )
    print(
        f"{args.operation}: median={measurement.median_seconds:.4f}s "
        f"peak={measurement.peak_mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
