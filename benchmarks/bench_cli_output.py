"""Benchmark CLI CSV and NPZ serialization on synthetic screening results.

Usage:
    uv run python benchmarks/bench_cli_output.py
    uv run python benchmarks/bench_cli_output.py --format npz --respondents 250000
"""

from __future__ import annotations

import argparse
import gc
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ier._cli_npz import _write_screen_npz
from ier.cli import _output_stream, _write_screen_csv

if TYPE_CHECKING:
    from ier.types import ScreenResult

OutputFormat = Literal["csv", "npz"]


def _make_result(n_respondents: int, n_indices: int) -> ScreenResult:
    names = [f"score_{index}" for index in range(n_indices)]
    values = np.linspace(0.0, 1.0, n_respondents)
    flags = values > 0.95
    return {
        "scores": {name: values for name in names},
        "flags": {name: flags for name in names},
        "thresholds": {name: 0.95 for name in names},
        "flag_counts": np.zeros(n_respondents, dtype=np.int_),
        "consensus_flags": np.zeros(n_respondents, dtype=np.bool_),
        "min_flags": 2,
        "n_indices": n_indices,
        "indices_used": names,
        "errors": {},
        "n_respondents": n_respondents,
        "summary": {
            name: {
                "mean": 0.5,
                "std": 0.3,
                "min": 0.0,
                "max": 1.0,
                "n_flagged": int(np.sum(flags)),
            }
            for name in names
        },
    }


def _write_result(
    output_format: OutputFormat,
    destination: Path,
    result: ScreenResult,
) -> None:
    if output_format == "csv":
        with _output_stream(destination) as handle:
            _write_screen_csv(handle, result)
        return
    _write_screen_npz(destination, result)


def _benchmark(
    output_format: OutputFormat,
    destination: Path,
    result: ScreenResult,
    repeats: int,
) -> tuple[float, float, float]:
    timings: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        _write_result(output_format, destination, result)
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    return (
        statistics.median(timings),
        statistics.median(peaks) / 1024 / 1024,
        destination.stat().st_size / 1024 / 1024,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--indices", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--format", choices=["csv", "npz", "both"], default="both")
    args = parser.parse_args()

    if args.respondents < 1 or args.indices < 1 or args.repeats < 1:
        parser.error("respondents, indices, and repeats must be positive")

    result = _make_result(args.respondents, args.indices)
    formats: list[OutputFormat] = ["csv", "npz"] if args.format == "both" else [args.format]

    print(f"respondents={args.respondents} indices={args.indices}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for output_format in formats:
            destination = root / f"screen.{output_format}"
            seconds, peak_mib, output_mib = _benchmark(
                output_format,
                destination,
                result,
                args.repeats,
            )
            print(
                f"{output_format}: median={seconds:.4f}s "
                f"peak={peak_mib:.3f} MiB output={output_mib:.1f} MiB"
            )


if __name__ == "__main__":
    main()
