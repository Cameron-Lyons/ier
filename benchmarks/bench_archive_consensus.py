"""Benchmark validated score-plus-timing archive consensus.

Usage:
    uv run python benchmarks/bench_archive_consensus.py
    uv run python benchmarks/bench_archive_consensus.py --respondents 1000000 --indices 15
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
    flag_consensus_archives,
    index_catalog,
    save_response_time_archive,
    save_score_archive,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ier import FlagConsensusArchive


def _measure(
    operation: Callable[[], FlagConsensusArchive],
    repeats: int,
) -> tuple[float, float, FlagConsensusArchive]:
    """Return median wall time, traced peak MiB, and the final result."""
    timings: list[float] = []
    peaks: list[int] = []
    result: FlagConsensusArchive | None = None
    for _ in range(repeats):
        result = None
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    assert result is not None
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=500_000)
    parser.add_argument("--indices", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    available_names = list(index_catalog())
    if args.respondents < 1 or args.repeats < 1:
        parser.error("respondents and repeats must be positive")
    if not 1 <= args.indices <= len(available_names):
        parser.error(f"indices must be between 1 and {len(available_names)}")
    if not 0.0 <= args.missing_rate < 1.0:
        parser.error("missing-rate must be at least zero and less than one")

    rng = np.random.default_rng(args.seed)
    names = available_names[: args.indices]
    scores: dict[str, np.ndarray] = {}
    for name in names:
        values = rng.normal(size=args.respondents)
        values[rng.random(args.respondents) < args.missing_rate] = np.nan
        scores[name] = values

    timing_scores = rng.lognormal(size=args.respondents)
    timing_scores[rng.random(args.respondents) < args.missing_rate] = np.nan
    timing_threshold = float(np.nanpercentile(timing_scores, 5.0))
    timing_flags = np.isfinite(timing_scores) & (timing_scores < timing_threshold)

    with TemporaryDirectory(prefix="ier-archive-consensus-benchmark-") as directory:
        root = Path(directory)
        score_path = root / "scores.npz"
        timing_path = root / "timing.npz"
        save_score_archive(score_path, scores)
        save_response_time_archive(
            timing_path,
            timing_scores,
            timing_flags,
            threshold=timing_threshold,
            threshold_source="percentile",
            percentile=5.0,
        )

        flag_consensus_archives(
            score_path,
            timing_path,
            min_valid_signals=args.indices,
        )
        seconds, peak, result = _measure(
            lambda: flag_consensus_archives(
                score_path,
                timing_path,
                min_valid_signals=args.indices,
            ),
            args.repeats,
        )
        input_size = (score_path.stat().st_size + timing_path.stat().st_size) / 1024 / 1024

    assert result["n_respondents"] == args.respondents
    assert result["n_signals"] == args.indices + 1
    print(
        f"respondents={args.respondents} registered_indices={args.indices} "
        f"signals={args.indices + 1} repeats={args.repeats} "
        f"missing_rate={args.missing_rate:g}"
    )
    print(f"input archives: {input_size:.1f} MiB")
    print(f"validated archive consensus: median={seconds:.4f}s peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()
