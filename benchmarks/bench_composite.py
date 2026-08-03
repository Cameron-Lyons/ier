"""Benchmark repeated composite sensitivity analysis with reusable scores.

Usage:
    uv run python benchmarks/bench_composite.py
    uv run python benchmarks/bench_composite.py --respondents 10000 --items 80
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
    composite,
    composite_scores,
    composite_summary,
    load_score_archive,
    save_score_archive,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_data(n_respondents: int, n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 6, size=(n_respondents, n_items)).astype(float)
    data[0, :] = 3.0
    if n_items >= 6:
        data[1, :] = np.tile([1.0, 5.0], n_items // 2 + 1)[:n_items]
    return data


def _peak_mib(operation: Callable[[], object]) -> float:
    gc.collect()
    tracemalloc.start()
    operation()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1024 / 1024


def _direct_composite(
    data: np.ndarray,
    options: IndexOptions,
    weights: dict[str, float],
) -> np.ndarray:
    result = composite(data, options=options, weights=weights)
    if isinstance(result, tuple):
        raise TypeError("unexpected diagnostics tuple from default composite scoring")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sensitivity-scenarios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 2 or args.items < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents and items must be at least 2; repeats must be positive; "
            "warmup cannot be negative"
        )
    if args.sensitivity_scenarios < 1:
        parser.error("sensitivity-scenarios must be a positive integer")

    data = _make_data(args.respondents, args.items, args.seed)
    options = IndexOptions(scale_min=1, scale_max=5)
    initial = composite_summary(data, options=options)
    names = initial["indices_used"]
    weight_scenarios = [
        {
            name: 0.5 + ((index + scenario) % len(names)) / max(len(names) - 1, 1)
            for index, name in enumerate(names)
        }
        for scenario in range(args.sensitivity_scenarios)
    ]

    for _ in range(args.warmup):
        _direct_composite(data, options, weight_scenarios[0])
        composite_scores(initial["indices"], weights=weight_scenarios[0])

    full_timings: list[float] = []
    reused_timings: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        direct_results = [_direct_composite(data, options, weights) for weights in weight_scenarios]
        full_timings.append(time.perf_counter() - started)

        started = time.perf_counter()
        reused_results = [
            composite_scores(initial["indices"], weights=weights) for weights in weight_scenarios
        ]
        reused_timings.append(time.perf_counter() - started)

    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "components.npz"
        save_score_archive(
            archive_path,
            initial["indices"],
            result_type="composite",
        )
        load_score_archive(archive_path)
        archived_timings: list[float] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            archived_results = [
                composite_scores(
                    load_score_archive(archive_path)["scores"],
                    weights=weights,
                )
                for weights in weight_scenarios
            ]
            archived_timings.append(time.perf_counter() - started)

        archived_peak = _peak_mib(
            lambda: [
                composite_scores(
                    load_score_archive(archive_path)["scores"],
                    weights=weights,
                )
                for weights in weight_scenarios
            ]
        )

    for direct, reused, archived in zip(
        direct_results,
        reused_results,
        archived_results,
        strict=True,
    ):
        np.testing.assert_allclose(reused, direct, rtol=1e-14, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(archived, direct, rtol=1e-14, atol=1e-14, equal_nan=True)

    full_peak = _peak_mib(
        lambda: [_direct_composite(data, options, weights) for weights in weight_scenarios]
    )
    reused_peak = _peak_mib(
        lambda: [
            composite_scores(initial["indices"], weights=weights) for weights in weight_scenarios
        ]
    )

    full_median = statistics.median(full_timings)
    reused_median = statistics.median(reused_timings)
    archived_median = statistics.median(archived_timings)
    print(f"shape={data.shape} indices={len(names)} scenarios={args.sensitivity_scenarios}")
    print(
        f"composite sensitivity: full={full_median:.4f}s reused={reused_median:.4f}s "
        f"speedup={full_median / reused_median:.1f}x peak={full_peak:.1f}/{reused_peak:.1f} MiB"
    )
    print(
        f"archived sensitivity: median={archived_median:.4f}s "
        f"speedup={full_median / archived_median:.1f}x peak={archived_peak:.1f} MiB"
    )


if __name__ == "__main__":
    main()
