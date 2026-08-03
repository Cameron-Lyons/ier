"""Benchmark CLI JSON, CSV, and NPZ serialization on synthetic scoring results.

Usage:
    uv run python benchmarks/bench_cli_output.py
    uv run python benchmarks/bench_cli_output.py --format json --respondents 250000
    uv run python benchmarks/bench_cli_output.py --workflow composite --format all
"""

from __future__ import annotations

import argparse
import gc
import statistics
import tempfile
import time
import tracemalloc
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ier._cli_npz import _write_composite_npz, _write_screen_npz
from ier._cli_output import (
    _output_stream,
    _write_composite_csv,
    _write_composite_json,
    _write_screen_csv,
    _write_screen_json,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ier.types import ScreenResult

OutputFormat = Literal["csv", "json", "npz"]


def _make_screen_result(n_respondents: int, n_indices: int) -> ScreenResult:
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


def _write_screen_result(
    output_format: OutputFormat,
    destination: Path,
    result: ScreenResult,
) -> None:
    if output_format == "csv":
        with _output_stream(destination) as handle:
            _write_screen_csv(handle, result)
        return
    if output_format == "json":
        with _output_stream(destination) as handle:
            _write_screen_json(handle, result)
        return
    _write_screen_npz(destination, result)


def _make_composite_result(
    n_respondents: int,
    n_indices: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    values = np.linspace(0.0, 1.0, n_respondents)
    component_scores = {f"score_{index}": values + index / n_indices for index in range(n_indices)}
    valid_index_counts = np.full(n_respondents, n_indices, dtype=np.int_)
    return values, component_scores, valid_index_counts


def _write_composite_result(
    output_format: OutputFormat,
    destination: Path,
    scores: np.ndarray,
    component_scores: dict[str, np.ndarray],
    valid_index_counts: np.ndarray,
) -> None:
    if output_format == "csv":
        with _output_stream(destination) as handle:
            _write_composite_csv(
                handle,
                scores,
                component_scores=component_scores,
                valid_index_counts=valid_index_counts,
            )
        return
    if output_format == "json":
        with _output_stream(destination) as handle:
            _write_composite_json(
                handle,
                scores,
                "mean",
                component_scores=component_scores,
                valid_index_counts=valid_index_counts,
            )
        return
    _write_composite_npz(
        destination,
        scores,
        "mean",
        component_scores=component_scores,
        valid_index_counts=valid_index_counts,
    )


def _benchmark(
    operation: Callable[[], None],
    destination: Path,
    repeats: int,
) -> tuple[float, float, float]:
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
    parser.add_argument(
        "--workflow",
        choices=["screen", "composite"],
        default="screen",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "npz", "all", "both"],
        default="all",
    )
    args = parser.parse_args()

    if args.respondents < 1 or args.indices < 1 or args.repeats < 1:
        parser.error("respondents, indices, and repeats must be positive")

    screen_result = (
        _make_screen_result(args.respondents, args.indices) if args.workflow == "screen" else None
    )
    composite_result = (
        _make_composite_result(args.respondents, args.indices)
        if args.workflow == "composite"
        else None
    )
    if args.format == "all":
        formats: list[OutputFormat] = ["csv", "json", "npz"]
    elif args.format == "both":
        formats = ["csv", "npz"]
    else:
        formats = [args.format]

    print(f"workflow={args.workflow} respondents={args.respondents} indices={args.indices}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for output_format in formats:
            destination = root / f"{args.workflow}.{output_format}"
            if args.workflow == "screen":
                assert screen_result is not None
                operation = partial(
                    _write_screen_result,
                    output_format,
                    destination,
                    screen_result,
                )
            else:
                assert composite_result is not None
                operation = partial(
                    _write_composite_result,
                    output_format,
                    destination,
                    *composite_result,
                )
            seconds, peak_mib, output_mib = _benchmark(
                operation,
                destination,
                args.repeats,
            )
            print(
                f"{output_format}: median={seconds:.4f}s "
                f"peak={peak_mib:.3f} MiB output={output_mib:.1f} MiB"
            )


if __name__ == "__main__":
    main()
