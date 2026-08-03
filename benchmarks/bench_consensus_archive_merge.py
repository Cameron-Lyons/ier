"""Benchmark validated merging of independently persisted consensus signals.

Usage:
    uv run python benchmarks/bench_consensus_archive_merge.py
    uv run python benchmarks/bench_consensus_archive_merge.py --respondents 1000000
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from ier import (
    flag_consensus,
    merge_flag_consensus_archives,
    save_flag_consensus_archive,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(operation: Callable[[], object], repeats: int) -> tuple[float, float]:
    """Return median wall time and traced peak MiB for one operation."""
    timings: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=500_000)
    parser.add_argument("--signals", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 1 or args.signals < 2 or args.repeats < 1:
        parser.error("respondents and repeats must be positive; signals must be at least 2")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    flags: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for index in range(args.signals):
        name = f"signal_{index}"
        score = rng.normal(size=args.respondents)
        missing = rng.random(args.respondents) < args.missing_rate
        score[missing] = np.nan
        signal_flags = rng.random(args.respondents) < 0.1
        signal_flags[missing] = False
        scores[name] = score
        flags[name] = signal_flags

    split = args.signals // 2
    names = list(flags)
    first_names = names[:split]
    second_names = names[split:]
    min_valid_signals = max(1, args.signals - 1)

    with TemporaryDirectory(prefix="ier-consensus-merge-benchmark-") as directory:
        root = Path(directory)
        first = root / "first.npz"
        second = root / "second.npz"
        save_flag_consensus_archive(
            first,
            {name: flags[name] for name in first_names},
            scores={name: scores[name] for name in first_names},
            min_flags=1,
        )
        save_flag_consensus_archive(
            second,
            {name: flags[name] for name in second_names},
            scores={name: scores[name] for name in second_names},
            min_flags=1,
        )

        merge_time, merge_peak = _measure(
            lambda: merge_flag_consensus_archives(
                [first, second],
                min_flags=2,
                min_valid_signals=min_valid_signals,
            ),
            args.repeats,
        )
        merged = merge_flag_consensus_archives(
            [first, second],
            min_flags=2,
            min_valid_signals=min_valid_signals,
        )
        expected = flag_consensus(
            flags,
            scores=scores,
            min_flags=2,
            min_valid_signals=min_valid_signals,
        )
        assert merged["signal_names"] == names
        assert list(merged["scores"]) == names
        for name in names:
            np.testing.assert_array_equal(merged["flags"][name], flags[name])
            np.testing.assert_array_equal(merged["scores"][name], scores[name])
        np.testing.assert_array_equal(merged["flag_counts"], expected["flag_counts"])
        np.testing.assert_array_equal(
            merged["valid_signal_counts"],
            expected["valid_signal_counts"],
        )
        np.testing.assert_array_equal(
            merged["consensus_eligible"],
            expected["consensus_eligible"],
        )
        np.testing.assert_array_equal(
            merged["consensus_flags"],
            expected["consensus_flags"],
        )
        input_size = (first.stat().st_size + second.stat().st_size) / 1024 / 1024

    print(
        f"respondents={args.respondents} signals={args.signals} repeats={args.repeats} "
        f"missing_rate={args.missing_rate:g}"
    )
    print(f"input archives: {input_size:.1f} MiB")
    print(f"validated merge: median={merge_time:.4f}s peak={merge_peak:.1f} MiB")


if __name__ == "__main__":
    main()
