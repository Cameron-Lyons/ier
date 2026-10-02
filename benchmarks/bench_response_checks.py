"""Benchmark missing-response and attention-check scoring.

Usage:
    uv run python benchmarks/bench_response_checks.py
    uv run python benchmarks/bench_response_checks.py --checks 40 --order F

Timing excludes allocation tracing; peak allocation is measured separately.
"""

from __future__ import annotations

import argparse
import math
import platform
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from _measurement import measure

from ier import infrequency, missing_rate

if TYPE_CHECKING:
    from collections.abc import Callable

    from ier.types import InfrequencyMissingPolicy


def _reference_scores(
    name: str,
    data: np.ndarray,
    applicable: np.ndarray,
    indices: list[int],
    expected: list[int],
    positions: np.ndarray,
) -> np.ndarray:
    """Evaluate sampled rows directly using Python's exact numeric comparisons."""
    result = []
    for position in positions:
        row = data[position]
        if name.startswith("missing_"):
            selected = indices if "subset" in name else range(data.shape[1])
            values = [
                row[item].item()
                for item in selected
                if "applicable" not in name or applicable[position, item]
            ]
            result.append(
                sum(math.isnan(value) for value in values) / len(values) if values else math.nan
            )
            continue
        policy = name.removeprefix("infrequency_")
        checks = [
            (row[item].item(), answer) for item, answer in zip(indices, expected, strict=True)
        ]
        observed = [(value, answer) for value, answer in checks if not math.isnan(value)]
        if (policy == "omit" and not observed) or (
            policy == "propagate" and len(observed) < len(checks)
        ):
            result.append(math.nan)
            continue
        failures = sum(value != answer for value, answer in observed)
        if policy == "fail":
            failures += len(checks) - len(observed)
        denominator = len(observed) if policy == "omit" else len(checks)
        result.append(failures / denominator)
    return np.asarray(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--checks", type=int, default=8)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument(
        "--dtype", choices=("float32", "float64", "int64", "uint64"), default="float64"
    )
    parser.add_argument("--integer-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.respondents < 1 or args.items < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("respondents, items, and repeats must be positive; warmup must be nonnegative")
    if not 1 <= args.checks <= args.items:
        parser.error("checks must be between 1 and items")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    integer_input = np.dtype(args.dtype).kind in "iu"
    scale_min = 1 + args.integer_offset
    if integer_input:
        if args.missing_rate:
            parser.error("integer inputs require --missing-rate 0")
        limits = np.iinfo(args.dtype)
        if scale_min < limits.min or scale_min + 4 > limits.max:
            parser.error("integer-offset places responses outside the selected dtype")
    elif args.integer_offset:
        parser.error("integer-offset requires an integer dtype")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, 5, size=(args.respondents, args.items)).astype(
        args.dtype, order=args.order
    )
    data += scale_min
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    applicable = np.asarray(rng.random(data.shape) < 0.8, order=args.order)
    applicable[0] = False
    indices = rng.choice(args.items, size=args.checks, replace=False).tolist()
    expected = [scale_min + category for category in rng.integers(0, 5, size=args.checks).tolist()]
    positions = np.unique(np.linspace(0, len(data) - 1, min(100, len(data)), dtype=int))
    scorers: dict[str, Callable[[], np.ndarray]] = {
        "missing_all": lambda: missing_rate(data),
        "missing_subset": lambda: missing_rate(data, item_indices=indices),
        "missing_applicable": lambda: missing_rate(data, applicable_mask=applicable),
        "missing_subset_applicable": lambda: missing_rate(
            data, item_indices=indices, applicable_mask=applicable
        ),
    }
    policies: tuple[InfrequencyMissingPolicy, ...] = ("pass", "fail", "omit", "propagate")
    for policy in policies:
        scorers[f"infrequency_{policy}"] = partial(
            infrequency, data, indices, expected, proportion=True, missing=policy
        )

    print(f"Python {platform.python_version()} / NumPy {np.__version__}")
    print(
        f"shape={data.shape} checks={args.checks} dtype={data.dtype} order={args.order} "
        f"integer_offset={args.integer_offset} "
        f"missing_rate={args.missing_rate} repeats={args.repeats} warmup={args.warmup}"
    )
    for name, score in scorers.items():
        for _ in range(args.warmup):
            score()
        measured = measure(score, args.repeats)
        np.testing.assert_array_equal(
            measured.result[positions],
            _reference_scores(name, data, applicable, indices, expected, positions),
        )
        print(
            f"{name}: median={measured.median_seconds * 1000:.3f}ms "
            f"peak={measured.peak_mib:.2f} MiB"
        )


if __name__ == "__main__":
    main()
