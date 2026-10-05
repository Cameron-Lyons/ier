"""Benchmark response-time summaries, mixture scoring, and retained-score flagging.

Usage:
    uv run python benchmarks/bench_response_time.py
    uv run python benchmarks/bench_response_time.py --respondents 200000 --items 40
    uv run python benchmarks/bench_response_time.py --operation median --order F
    uv run python benchmarks/bench_response_time.py --operation effort --items 40
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
from _measurement import measure, measure_many

from ier import (
    response_time,
    response_time_consistency,
    response_time_effort,
    response_time_effort_flag,
    response_time_flag,
    response_time_mixture,
    response_time_score_flags,
)
from ier.response_time import _em_gaussian_mixture

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(
    operation: Callable[[], np.ndarray],
    *,
    repeats: int,
    warmup: int,
    probability: bool = False,
    available: np.ndarray | None = None,
) -> tuple[float, float]:
    for _ in range(warmup):
        operation()

    measurement = measure(operation, repeats)
    result = measurement.result
    expected = np.ones(result.shape, dtype=bool) if available is None else available
    if np.isinf(result).any() or not np.array_equal(np.isfinite(result), expected):
        raise RuntimeError("benchmark produced scores with incorrect respondent availability")
    if probability and np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("benchmark produced invalid mixture probabilities")
    return measurement.median_seconds, measurement.peak_mib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--operation", choices=("all", "median", "effort"), default="all")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--no-log-transform", dest="log_transform", action="store_false")
    parser.add_argument(
        "--structure", choices=("mixture", "constant", "near-constant"), default="mixture"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if (
        args.respondents < 1
        or args.items < 1
        or args.components < 2
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error(
            "respondents, items, and repeats must be positive; components must be at least 2; "
            "warmup cannot be negative"
        )
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("scale must be finite and positive")

    rng = np.random.default_rng(args.seed)
    fast = rng.lognormal(mean=-0.7, sigma=0.2, size=args.respondents // 5)
    regular = rng.lognormal(mean=1.2, sigma=0.35, size=args.respondents - len(fast))
    medians = np.concatenate((fast, regular))
    if args.structure == "constant":
        medians.fill(1.1)
    elif args.structure == "near-constant":
        medians = 1.1 + np.concatenate(
            (rng.integers(0, 5, len(fast)), rng.integers(16, 21, len(regular)))
        ) * np.spacing(1.1)
    rng.shuffle(medians)
    with np.errstate(over="ignore", under="ignore"):
        medians *= args.scale
        if args.structure == "mixture":
            item_noise = rng.lognormal(mean=0.0, sigma=0.15, size=(args.respondents, args.items))
            timings = medians[:, None] * item_noise
        else:
            timings = np.repeat(medians[:, None], args.items, axis=1)
    if not np.isfinite(timings).all() or np.any(timings <= 0):
        parser.error("scale produces nonpositive or nonfinite timing values")
    fit_data = np.log(medians) if args.log_transform else medians
    if args.missing_rate:
        timings[rng.random(timings.shape) < args.missing_rate] = np.nan
    if args.order == "F":
        timings = np.asfortranarray(timings)

    available = np.any(~np.isnan(timings), axis=1)
    if args.operation == "all" and np.count_nonzero(available) < args.components:
        parser.error("available timing medians must be at least the component count")
    print(
        f"shape={timings.shape} order={args.order} operation={args.operation} "
        f"components={args.components} missing_rate={args.missing_rate} "
        f"structure={args.structure} scale={args.scale} log_transform={args.log_transform} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    if args.operation == "median":
        seconds, peak = _measure(
            lambda: response_time(timings, metric="median"),
            repeats=args.repeats,
            warmup=args.warmup,
            available=available,
        )
        print(f"median: median={seconds:.4f}s peak={peak:.1f} MiB")
        return
    if args.operation == "effort":
        # Positive timings give every answered item a usable normative threshold.
        effort_operations: dict[str, Callable[[], np.ndarray]] = {
            "effort": lambda: response_time_effort(timings),
            "capped effort": lambda: response_time_effort(timings, max_threshold=0.5),
            "effort flags": lambda: response_time_effort_flag(timings)[0],
        }
        for name, operation in effort_operations.items():
            seconds, peak = _measure(
                operation, repeats=args.repeats, warmup=args.warmup, available=available
            )
            print(f"{name}: median={seconds:.4f}s peak={peak:.1f} MiB")
        return

    retained_scores = response_time(timings, metric="median")
    sensitivity_percentiles = (1.0, 2.5, 5.0, 10.0, 20.0)

    def full_sensitivity() -> np.ndarray:
        return np.concatenate(
            [
                response_time_flag(timings, cutoff_percentile=percentile)
                for percentile in sensitivity_percentiles
            ]
        )

    def reused_sensitivity() -> np.ndarray:
        return np.concatenate(
            [
                response_time_score_flags(
                    retained_scores,
                    cutoff_percentile=percentile,
                )
                for percentile in sensitivity_percentiles
            ]
        )

    np.testing.assert_array_equal(reused_sensitivity(), full_sensitivity())

    summary_operations: dict[str, Callable[[], np.ndarray]] = {
        "mean": lambda: response_time(timings, metric="mean"),
        "median": lambda: response_time(timings, metric="median"),
        "standard deviation": lambda: response_time(timings, metric="sd"),
    }
    if args.items >= 2:
        summary_operations["consistency"] = lambda: response_time_consistency(timings)
    summary_measurements = {
        name: _measure(operation, repeats=args.repeats, warmup=args.warmup, available=available)
        for name, operation in summary_operations.items()
    }

    core_seconds, core_peak = _measure(
        lambda: _em_gaussian_mixture(
            fit_data,
            args.components,
            np.random.default_rng(args.seed),
        ),
        repeats=args.repeats,
        warmup=args.warmup,
        probability=True,
    )
    workflow_seconds, workflow_peak = _measure(
        lambda: response_time_mixture(
            timings,
            n_components=args.components,
            log_transform=args.log_transform,
            random_seed=args.seed,
        ),
        repeats=args.repeats,
        warmup=args.warmup,
        probability=True,
        available=available,
    )
    for _ in range(args.warmup):
        full_sensitivity()
        reused_sensitivity()
    sensitivity = measure_many(
        {"full": full_sensitivity, "reused": reused_sensitivity}, args.repeats
    )
    full = sensitivity["full"]
    reused = sensitivity["reused"]
    np.testing.assert_array_equal(reused.result, full.result)
    full_sensitivity_seconds, full_sensitivity_peak = full.median_seconds, full.peak_mib
    reused_sensitivity_seconds, reused_sensitivity_peak = reused.median_seconds, reused.peak_mib

    for name, (seconds, peak) in summary_measurements.items():
        print(f"{name}: median={seconds:.4f}s peak={peak:.1f} MiB")
    print(f"EM core: median={core_seconds:.4f}s peak={core_peak:.1f} MiB")
    print(f"public workflow: median={workflow_seconds:.4f}s peak={workflow_peak:.1f} MiB")
    print(
        f"five full cutoff scenarios: median={full_sensitivity_seconds:.4f}s "
        f"peak={full_sensitivity_peak:.1f} MiB"
    )
    print(
        f"five reused cutoff scenarios: median={reused_sensitivity_seconds:.4f}s "
        f"peak={reused_sensitivity_peak:.1f} MiB "
        f"speedup={full_sensitivity_seconds / reused_sensitivity_seconds:.1f}x"
    )


if __name__ == "__main__":
    main()
