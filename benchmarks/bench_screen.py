"""Benchmark default screen() on synthetic survey matrices.

Usage:
    uv run python benchmarks/bench_screen.py
    uv run python benchmarks/bench_screen.py --respondents 2000 --items 50 --repeats 5
    uv run python benchmarks/bench_screen.py --respondents 20000 --items 80 --workers 4
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np
from _measurement import measure, measure_many

from ier import IndexOptions, screen, screen_scores


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

    measurement = measure(lambda: screen(data, options=options, workers=args.workers), args.repeats)
    result = measurement.result
    timings = measurement.timings
    sensitivity_percentiles = np.linspace(80.0, 99.0, args.sensitivity_scenarios)
    screen_scores(result["scores"], percentile=float(sensitivity_percentiles[0]))
    measured = measure_many(
        {
            "full": lambda: [
                screen(data, options=options, percentile=float(value), workers=args.workers)
                for value in sensitivity_percentiles
            ],
            "reused": lambda: [
                screen_scores(result["scores"], percentile=float(value))
                for value in sensitivity_percentiles
            ],
        },
        args.repeats,
    )
    full, reused = measured["full"], measured["reused"]
    for direct, retained in zip(full.result, reused.result, strict=True):
        if direct["thresholds"] != retained["thresholds"]:
            raise RuntimeError("reused scores produced different thresholds")
        np.testing.assert_array_equal(direct["consensus_flags"], retained["consensus_flags"])

    print(f"shape={data.shape} indices={result['n_indices']} workers={args.workers}")
    print(
        "screen seconds: "
        f"median={statistics.median(timings):.4f} "
        f"mean={statistics.mean(timings):.4f} "
        f"min={min(timings):.4f} max={max(timings):.4f}"
    )
    full_median = full.median_seconds
    reused_median = reused.median_seconds
    print(
        f"sensitivity scenarios={args.sensitivity_scenarios}: "
        f"full={full_median:.4f}s reused={reused_median:.4f}s "
        f"speedup={full_median / reused_median:.1f}x "
        f"peak={full.peak_mib:.1f}/{reused.peak_mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
