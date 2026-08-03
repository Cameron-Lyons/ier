"""Benchmark Gaussian-mixture response-time scoring.

Usage:
    uv run python benchmarks/bench_response_time.py
    uv run python benchmarks/bench_response_time.py --respondents 200000 --items 40
"""

from __future__ import annotations

import argparse
import gc
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ier import (
    load_response_time_archive,
    response_time,
    response_time_consistency,
    response_time_flag,
    response_time_mixture,
    response_time_score_flags,
    save_response_time_archive,
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
) -> tuple[float, float]:
    for _ in range(warmup):
        operation()

    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None
    if not np.isfinite(result).all():
        raise RuntimeError("benchmark produced non-finite scores")
    if probability and np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("benchmark produced invalid mixture probabilities")
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < args.components:
        parser.error("respondents must be at least the component count")
    if args.items < 1 or args.components < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "items and repeats must be positive; components must be at least 2; "
            "warmup cannot be negative"
        )
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least 0 and less than 1")

    rng = np.random.default_rng(args.seed)
    fast = rng.lognormal(mean=-0.7, sigma=0.2, size=args.respondents // 5)
    regular = rng.lognormal(mean=1.2, sigma=0.35, size=args.respondents - len(fast))
    medians = np.concatenate((fast, regular))
    rng.shuffle(medians)
    log_medians = np.log(medians)
    item_noise = rng.lognormal(mean=0.0, sigma=0.15, size=(args.respondents, args.items))
    timings = medians[:, None] * item_noise
    if args.missing_rate:
        timings[rng.random(timings.shape) < args.missing_rate] = np.nan

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
        "consistency": lambda: response_time_consistency(timings),
    }
    summary_measurements = {
        name: _measure(operation, repeats=args.repeats, warmup=args.warmup)
        for name, operation in summary_operations.items()
    }

    core_seconds, core_peak = _measure(
        lambda: _em_gaussian_mixture(
            log_medians,
            args.components,
            np.random.default_rng(args.seed),
        ),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    workflow_seconds, workflow_peak = _measure(
        lambda: response_time_mixture(
            timings,
            n_components=args.components,
            random_seed=args.seed,
        ),
        repeats=args.repeats,
        warmup=args.warmup,
        probability=True,
    )
    full_sensitivity_seconds, full_sensitivity_peak = _measure(
        full_sensitivity,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    reused_sensitivity_seconds, reused_sensitivity_peak = _measure(
        reused_sensitivity,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "timing.npz"
        initial_threshold = float(np.nanpercentile(retained_scores, 5.0))
        save_response_time_archive(
            archive_path,
            retained_scores,
            retained_scores < initial_threshold,
            threshold=initial_threshold,
            threshold_source="percentile",
            percentile=5.0,
        )

        def archived_sensitivity() -> np.ndarray:
            saved = load_response_time_archive(archive_path)
            return np.concatenate(
                [
                    response_time_score_flags(
                        saved["scores"],
                        cutoff_percentile=percentile,
                        direction=saved["flag_direction"],
                    )
                    for percentile in sensitivity_percentiles
                ]
            )

        np.testing.assert_array_equal(archived_sensitivity(), reused_sensitivity())
        archived_sensitivity_seconds, archived_sensitivity_peak = _measure(
            archived_sensitivity,
            repeats=args.repeats,
            warmup=args.warmup,
        )

    print(
        f"shape={timings.shape} components={args.components} "
        f"missing_rate={args.missing_rate} repeats={args.repeats} warmup={args.warmup}"
    )
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
    print(
        f"five archived cutoff scenarios: median={archived_sensitivity_seconds:.4f}s "
        f"peak={archived_sensitivity_peak:.1f} MiB "
        f"speedup={full_sensitivity_seconds / archived_sensitivity_seconds:.1f}x"
    )


if __name__ == "__main__":
    main()
