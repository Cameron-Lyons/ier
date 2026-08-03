"""Benchmark forward-only CLI CSV serialization on synthetic screening results.

Usage:
    uv run python benchmarks/bench_cli_csv.py
    uv run python benchmarks/bench_cli_csv.py --respondents 250000 --indices 8
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

from ier.cli import _output_stream, _write_screen_csv

if TYPE_CHECKING:
    from ier.types import ScreenResult


def _make_result(n_respondents: int, n_indices: int) -> ScreenResult:
    names = [f"score_{index}" for index in range(n_indices)]
    values = np.linspace(0.0, 1.0, n_respondents)
    return {
        "scores": {name: values for name in names},
        "flags": {name: values > 0.95 for name in names},
        "thresholds": {name: 0.95 for name in names},
        "flag_counts": np.zeros(n_respondents, dtype=np.int_),
        "consensus_flags": np.zeros(n_respondents, dtype=np.bool_),
        "min_flags": 2,
        "n_indices": n_indices,
        "indices_used": names,
        "errors": {},
        "n_respondents": n_respondents,
        "summary": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--indices", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.respondents < 1 or args.indices < 1 or args.repeats < 1:
        parser.error("respondents, indices, and repeats must be positive")

    result = _make_result(args.respondents, args.indices)
    timings: list[float] = []
    peaks: list[int] = []

    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "screen.csv"
        for _ in range(args.repeats):
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            with _output_stream(destination) as handle:
                _write_screen_csv(handle, result)
            timings.append(time.perf_counter() - started)
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

        print(
            f"respondents={args.respondents} indices={args.indices} "
            f"output_mib={destination.stat().st_size / 1024 / 1024:.1f}"
        )

    print(f"seconds: median={statistics.median(timings):.3f} min={min(timings):.3f}")
    print(f"peak MiB: median={statistics.median(peaks) / 1024 / 1024:.3f}")


if __name__ == "__main__":
    main()
