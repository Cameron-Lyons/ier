"""Benchmark bounded flag consensus against stacked signal matrices.

Usage:
    uv run python benchmarks/bench_consensus.py
    uv run python benchmarks/bench_consensus.py --respondents 1000000 --signals 20
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

from ier import flag_consensus

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _stacked_consensus(
    flags: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    min_flags: int,
    min_valid_signals: int,
) -> dict[str, object]:
    """Return the expanded respondent-by-signal reference calculation."""
    flag_matrix = np.column_stack(list(flags.values()))
    available_matrix = np.column_stack([~np.isnan(scores[name]) for name in flags])
    flag_counts = np.sum(flag_matrix, axis=1, dtype=np.int_)
    valid_signal_counts = np.sum(available_matrix, axis=1, dtype=np.int_)
    eligible = valid_signal_counts >= min_valid_signals
    return {
        "flag_counts": flag_counts,
        "valid_signal_counts": valid_signal_counts,
        "consensus_eligible": eligible,
        "consensus_flags": (flag_counts >= min_flags) & eligible,
    }


def _measure_pair(
    bounded: Callable[[], Mapping[str, object]],
    stacked: Callable[[], Mapping[str, object]],
    repeats: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, Mapping[str, object]]]:
    """Measure both implementations in alternating order."""
    timings: dict[str, list[float]] = {"bounded": [], "stacked": []}
    peaks: dict[str, list[int]] = {"bounded": [], "stacked": []}
    results: dict[str, Mapping[str, object]] = {}
    operations = {"bounded": bounded, "stacked": stacked}
    for repeat in range(repeats):
        order = ("bounded", "stacked") if repeat % 2 == 0 else ("stacked", "bounded")
        for label in order:
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            results[label] = operations[label]()
            timings[label].append(time.perf_counter() - started)
            peaks[label].append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
    return (
        {name: statistics.median(values) for name, values in timings.items()},
        {name: statistics.median(values) / 1024 / 1024 for name, values in peaks.items()},
        results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=500_000)
    parser.add_argument("--signals", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--flag-rate", type=float, default=0.1)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--min-flags", type=int, default=2)
    parser.add_argument("--min-valid-signals", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 1 or args.signals < 1 or args.repeats < 1:
        parser.error("respondents, signals, and repeats must be positive")
    if not 0.0 <= args.flag_rate <= 1.0 or not 0.0 <= args.missing_rate <= 1.0:
        parser.error("flag-rate and missing-rate must be between 0 and 1")
    if args.min_flags < 1:
        parser.error("min-flags must be positive")
    min_valid_signals = (
        max(1, args.signals - 1) if args.min_valid_signals is None else args.min_valid_signals
    )
    if not 1 <= min_valid_signals <= args.signals:
        parser.error("min-valid-signals must be between 1 and signals")

    rng = np.random.default_rng(args.seed)
    flags: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for index in range(args.signals):
        name = "response_time" if index == args.signals - 1 else f"index_{index}"
        score = rng.normal(size=args.respondents)
        missing = rng.random(args.respondents) < args.missing_rate
        score[missing] = np.nan
        signal_flags = rng.random(args.respondents) < args.flag_rate
        signal_flags[missing] = False
        scores[name] = score
        flags[name] = signal_flags

    flag_consensus(
        flags,
        scores=scores,
        min_flags=args.min_flags,
        min_valid_signals=min_valid_signals,
    )
    _stacked_consensus(flags, scores, args.min_flags, min_valid_signals)
    timings, peaks, results = _measure_pair(
        lambda: flag_consensus(
            flags,
            scores=scores,
            min_flags=args.min_flags,
            min_valid_signals=min_valid_signals,
        ),
        lambda: _stacked_consensus(
            flags,
            scores,
            args.min_flags,
            min_valid_signals,
        ),
        args.repeats,
    )

    for name in (
        "flag_counts",
        "valid_signal_counts",
        "consensus_eligible",
        "consensus_flags",
    ):
        np.testing.assert_array_equal(results["bounded"][name], results["stacked"][name])

    print(
        f"respondents={args.respondents} signals={args.signals} repeats={args.repeats} "
        f"min_flags={args.min_flags} min_valid_signals={min_valid_signals}"
    )
    print(f"bounded consensus: median={timings['bounded']:.4f}s peak={peaks['bounded']:.1f} MiB")
    print(f"stacked reference: median={timings['stacked']:.4f}s peak={peaks['stacked']:.1f} MiB")
    print(
        f"bounded/stacked time={timings['bounded'] / timings['stacked']:.2f}x "
        f"peak={peaks['bounded'] / peaks['stacked']:.2f}x"
    )


if __name__ == "__main__":
    main()
