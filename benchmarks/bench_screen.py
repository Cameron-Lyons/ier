"""Benchmark default screen() on synthetic survey matrices.

Usage:
    uv run python benchmarks/bench_screen.py
    uv run python benchmarks/bench_screen.py --respondents 2000 --items 50 --repeats 5
    uv run python benchmarks/bench_screen.py --respondents 20000 --items 80 --workers 4
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
    IndexOptions,
    load_score_archive,
    save_score_archive,
    screen,
    screen_scores,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _peak_mib(operation: Callable[[], object]) -> float:
    """Measure peak traced allocation for one operation."""
    gc.collect()
    tracemalloc.start()
    operation()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1024 / 1024


def _make_data(n_respondents: int, n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 6, size=(n_respondents, n_items)).astype(float)
    data[0, :] = 3.0
    if n_items >= 6:
        data[1, :] = np.tile([1.0, 5.0], n_items // 2 + 1)[:n_items]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=500)
    parser.add_argument("--items", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--sensitivity-scenarios",
        type=int,
        default=5,
        help="Number of tail percentiles to compare with and without score reuse",
    )
    args = parser.parse_args()

    if args.respondents < 1 or args.items < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("respondents, items, and repeats must be positive; warmup cannot be negative")
    if args.workers < 1 or args.sensitivity_scenarios < 1:
        parser.error("workers and sensitivity-scenarios must be positive integers")

    data = _make_data(args.respondents, args.items, args.seed)
    options = IndexOptions(scale_min=1, scale_max=5)

    for _ in range(args.warmup):
        screen(data, options=options, workers=args.workers)

    timings: list[float] = []
    result = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = screen(data, options=options, workers=args.workers)
        timings.append(time.perf_counter() - start)

    assert result is not None
    sensitivity_percentiles = np.linspace(80.0, 99.0, args.sensitivity_scenarios)
    screen_scores(result["scores"], percentile=float(sensitivity_percentiles[0]))
    full_sensitivity_timings: list[float] = []
    reused_sensitivity_timings: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        direct_results = [
            screen(data, options=options, percentile=float(value), workers=args.workers)
            for value in sensitivity_percentiles
        ]
        full_sensitivity_timings.append(time.perf_counter() - start)

        start = time.perf_counter()
        reused_results = [
            screen_scores(result["scores"], percentile=float(value))
            for value in sensitivity_percentiles
        ]
        reused_sensitivity_timings.append(time.perf_counter() - start)

    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "scores.npz"
        save_score_archive(archive_path, result["scores"])
        load_score_archive(archive_path)
        archived_sensitivity_timings: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            archived_results = [
                screen_scores(
                    load_score_archive(archive_path)["scores"],
                    percentile=float(value),
                )
                for value in sensitivity_percentiles
            ]
            archived_sensitivity_timings.append(time.perf_counter() - start)

        archived_sensitivity_peak = _peak_mib(
            lambda: [
                screen_scores(
                    load_score_archive(archive_path)["scores"],
                    percentile=float(value),
                )
                for value in sensitivity_percentiles
            ]
        )

    for direct, reused, archived in zip(
        direct_results,
        reused_results,
        archived_results,
        strict=True,
    ):
        if direct["thresholds"] != reused["thresholds"]:
            raise RuntimeError("reused scores produced different thresholds")
        if direct["thresholds"] != archived["thresholds"]:
            raise RuntimeError("archived scores produced different thresholds")
        np.testing.assert_array_equal(direct["consensus_flags"], reused["consensus_flags"])
        np.testing.assert_array_equal(
            direct["consensus_flags"],
            archived["consensus_flags"],
        )

    full_sensitivity_peak = _peak_mib(
        lambda: [
            screen(data, options=options, percentile=float(value), workers=args.workers)
            for value in sensitivity_percentiles
        ]
    )
    reused_sensitivity_peak = _peak_mib(
        lambda: [
            screen_scores(result["scores"], percentile=float(value))
            for value in sensitivity_percentiles
        ]
    )

    print(f"shape={data.shape} indices={result['n_indices']} workers={args.workers}")
    print(
        "screen seconds: "
        f"median={statistics.median(timings):.4f} "
        f"mean={statistics.mean(timings):.4f} "
        f"min={min(timings):.4f} max={max(timings):.4f}"
    )
    full_median = statistics.median(full_sensitivity_timings)
    reused_median = statistics.median(reused_sensitivity_timings)
    archived_median = statistics.median(archived_sensitivity_timings)
    print(
        f"sensitivity scenarios={args.sensitivity_scenarios}: "
        f"full={full_median:.4f}s reused={reused_median:.4f}s "
        f"speedup={full_median / reused_median:.1f}x "
        f"peak={full_sensitivity_peak:.1f}/{reused_sensitivity_peak:.1f} MiB"
    )
    print(
        f"archived sensitivity: median={archived_median:.4f}s "
        f"speedup={full_median / archived_median:.1f}x "
        f"peak={archived_sensitivity_peak:.1f} MiB"
    )


if __name__ == "__main__":
    main()
