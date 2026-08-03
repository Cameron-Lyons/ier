"""Benchmark validated flag-consensus archive save and load performance.

Usage:
    uv run python benchmarks/bench_consensus_archive.py
    uv run python benchmarks/bench_consensus_archive.py --respondents 1000000 --signals 20
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

from ier import flag_consensus, load_flag_consensus_archive, save_flag_consensus_archive

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

    if args.respondents < 1 or args.signals < 1 or args.repeats < 1:
        parser.error("respondents, signals, and repeats must be positive")
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    flags: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for index in range(args.signals):
        name = "response_time" if index == args.signals - 1 else f"index_{index}"
        score = rng.normal(size=args.respondents)
        missing = rng.random(args.respondents) < args.missing_rate
        score[missing] = np.nan
        signal_flags = rng.random(args.respondents) < 0.1
        signal_flags[missing] = False
        scores[name] = score
        flags[name] = signal_flags

    with TemporaryDirectory(prefix="ier-consensus-benchmark-") as directory:
        destination = Path(directory) / "consensus.npz"

        def save() -> None:
            save_flag_consensus_archive(
                destination,
                flags,
                scores=scores,
                min_flags=2,
                min_valid_signals=max(1, args.signals - 1),
            )

        save_time, save_peak = _measure(save, args.repeats)
        load_time, load_peak = _measure(
            lambda: load_flag_consensus_archive(destination),
            args.repeats,
        )

        def reflag() -> object:
            archived = load_flag_consensus_archive(destination)
            return flag_consensus(
                archived["flags"],
                scores=archived["scores"],
                min_flags=min(3, args.signals),
                min_valid_signals=max(1, args.signals - 2),
            )

        reflag_time, reflag_peak = _measure(reflag, args.repeats)
        loaded = load_flag_consensus_archive(destination)
        assert loaded["n_respondents"] == args.respondents
        assert loaded["n_signals"] == args.signals
        archive_size = destination.stat().st_size / 1024 / 1024

    print(
        f"respondents={args.respondents} signals={args.signals} repeats={args.repeats} "
        f"missing_rate={args.missing_rate:g}"
    )
    print(f"archive size: {archive_size:.1f} MiB")
    print(f"validated save: median={save_time:.4f}s peak={save_peak:.1f} MiB")
    print(f"validated load: median={load_time:.4f}s peak={load_peak:.1f} MiB")
    print(f"validated reload + reflag: median={reflag_time:.4f}s peak={reflag_peak:.1f} MiB")


if __name__ == "__main__":
    main()
