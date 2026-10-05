"""Benchmark sequence indices and workflows with controlled missing responses.

Usage:
    OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_sequence_scoring.py
    uv run python benchmarks/bench_sequence_scoring.py --missing-rate 0 --respondents 200
    uv run python benchmarks/bench_sequence_scoring.py --respondents 100 --items 1000 \
        --operations longstring longstring_pattern
    OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_sequence_scoring.py \
        --respondents 23000 --items 40 --missing-rate 0 --operations autocorrelation

Wall-clock timings exclude allocation tracing. Peak traced allocation is measured
in a separate run, excluding input generation and imports.
"""

from __future__ import annotations

import argparse
import platform
from typing import TYPE_CHECKING

import numpy as np
from _measurement import measure

from ier import (
    IndexOptions,
    autocorrelation,
    composite,
    longstring_pattern,
    longstring_scores,
    markov,
    screen,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=[
            "longstring",
            "longstring_pattern",
            "markov",
            "autocorrelation",
            "screen",
            "composite",
        ],
        help="Measure only selected operations (default: all)",
    )
    args = parser.parse_args()
    if args.respondents < 2 or args.items < 3 or args.repeats < 1 or args.warmup < 0:
        parser.error("respondents >= 2, items >= 3, repeats >= 1, and warmup >= 0 are required")
    if not 0 <= args.missing_rate < 1:
        parser.error("missing-rate must be in [0, 1)")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(1, 6, size=(args.respondents, args.items)).astype(float)
    data[rng.random(data.shape) < args.missing_rate] = np.nan
    options = IndexOptions(scale_min=1, scale_max=5)
    operations: dict[str, Callable[[], object]] = {
        "longstring": lambda: longstring_scores(data),
        "longstring_pattern": lambda: longstring_pattern(data),
        "markov": lambda: markov(data),
        "autocorrelation": lambda: autocorrelation(data),
        "screen": lambda: screen(data, options=options),
        "composite": lambda: composite(data),
    }
    if args.items < 4:
        # One usable autocorrelation lag needs at least four items.
        del operations["autocorrelation"]
    print(f"Python {platform.python_version()} / NumPy {np.__version__}")
    print(f"shape={data.shape} missing_rate={args.missing_rate} repeats={args.repeats}")
    for name, operation in operations.items():
        if args.operations is not None and name not in args.operations:
            continue
        for _ in range(args.warmup):
            operation()
        measurement = measure(operation, args.repeats)
        print(
            f"{name}: median={measurement.median_seconds:.6f}s peak={measurement.peak_mib:.2f} MiB"
        )


if __name__ == "__main__":
    main()
