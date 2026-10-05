"""Benchmark CLI text, JSON, CSV, and NPZ serialization on synthetic scoring results.

Usage:
    uv run python benchmarks/bench_cli_output.py
    uv run python benchmarks/bench_cli_output.py --format json --respondents 250000
    uv run python benchmarks/bench_cli_output.py --format json --compression xz
    uv run python benchmarks/bench_cli_output.py --workflow composite --format all
    uv run python benchmarks/bench_cli_output.py --workflow composite --flagged
    uv run python benchmarks/bench_cli_output.py --workflow composite --probability
    uv run python benchmarks/bench_cli_output.py --format csv --respondent-ids
    uv run python benchmarks/bench_cli_output.py --format text --respondents 1000000 --top 10
"""

from __future__ import annotations

import argparse
import tempfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from _measurement import measure

from ier import index_catalog, screen_scores
from ier._cli_composite import CompositeReport
from ier._cli_npz import _write_composite_npz, _write_screen_npz
from ier._cli_output import (
    _emit_composite_text,
    _emit_screen_text,
    _output_stream,
    _write_composite_csv,
    _write_composite_json,
    _write_output,
    _write_screen_csv,
    _write_screen_json,
)
from ier._cli_streams import _open_text_path
from ier._statistics import logistic_transform

if TYPE_CHECKING:
    from collections.abc import Callable

    from ier.types import ScreenResult

OutputFormat = Literal["text", "csv", "json", "npz"]
Compression = Literal["none", "gzip", "bzip2", "xz"]

_COMPRESSION_SUFFIXES: dict[Compression, str] = {
    "none": "",
    "gzip": ".gz",
    "bzip2": ".bz2",
    "xz": ".xz",
}
_ID_PREFIX = "site,"


def _make_screen_result(n_respondents: int, n_indices: int) -> ScreenResult:
    # NPZ archives validate registered names and recompute every decision, so
    # build a consistent result that flags the top 5% in each suspicious tail.
    catalog = index_catalog()
    names = [name for name, metadata in catalog.items() if metadata["flag_mode"] == "percentile"]
    if n_indices > len(names):
        raise ValueError(f"screen results support at most {len(names)} indices")
    values = np.linspace(0.0, 1.0, n_respondents)
    high = {name: catalog[name]["flag_direction"] == "high" for name in names[:n_indices]}
    return screen_scores(
        {name: values if is_high else 1.0 - values for name, is_high in high.items()},
        thresholds={name: 0.95 if is_high else 0.05 for name, is_high in high.items()},
        min_flags=2,
    )


def _write_screen_result(
    output_format: OutputFormat,
    destination: Path,
    result: ScreenResult,
    top: int = 10,
    respondent_ids: list[str] | None = None,
) -> None:
    if output_format == "text":
        _write_output(_emit_screen_text(result, top, respondent_ids), destination)
        return
    if output_format == "csv":
        with _output_stream(destination) as handle:
            _write_screen_csv(handle, result, respondent_ids)
        return
    if output_format == "json":
        with _output_stream(destination) as handle:
            _write_screen_json(handle, result, respondent_ids)
        return
    _write_screen_npz(destination, result, respondent_ids)


def _make_composite_result(
    n_respondents: int,
    n_indices: int,
    *,
    flagged: bool,
    probability: bool,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    values = np.linspace(0.0, 1.0, n_respondents)
    component_scores = {f"score_{index}": values + index / n_indices for index in range(n_indices)}
    valid_index_counts = np.full(n_respondents, n_indices, dtype=np.int_)
    flags = values >= 0.95 if flagged else None
    probabilities = logistic_transform(values) if probability else None
    return values, component_scores, valid_index_counts, flags, probabilities


def _write_composite_result(
    output_format: OutputFormat,
    destination: Path,
    scores: np.ndarray,
    component_scores: dict[str, np.ndarray],
    valid_index_counts: np.ndarray,
    flags: np.ndarray | None,
    probabilities: np.ndarray | None,
    top: int = 10,
    respondent_ids: list[str] | None = None,
) -> None:
    # Building the report validates it once, as the CLI does before any format.
    report = CompositeReport(
        scores,
        "mean",
        respondent_ids,
        component_scores=component_scores,
        valid_index_counts=valid_index_counts,
        flags=flags,
        flag_threshold=0.95 if flags is not None else None,
        probabilities=probabilities,
    )
    if output_format == "text":
        _write_output(_emit_composite_text(report, top), destination)
        return
    if output_format == "csv":
        with _output_stream(destination) as handle:
            _write_composite_csv(handle, report)
        return
    if output_format == "json":
        with _output_stream(destination) as handle:
            _write_composite_json(handle, report)
        return
    _write_composite_npz(destination, report)


def _benchmark(
    operation: Callable[[], None],
    destination: Path,
    repeats: int,
) -> tuple[float, float, float]:
    measurement = measure(operation, repeats)
    return (
        measurement.median_seconds,
        measurement.peak_mib,
        destination.stat().st_size / 1024 / 1024,
    )


def _respondent_ids(n_respondents: int) -> list[str]:
    """Return identifiers that require CSV quoting, as exported survey IDs often do."""
    return [f"{_ID_PREFIX}{index}" for index in range(n_respondents)]


def _check_text_rows(destination: Path, scores: np.ndarray, top: int) -> None:
    """Verify text preview identities against a complete stable ordering."""
    with _open_text_path(destination, "r") as handle:
        rows = [
            int(line.strip().split("\t")[0].removeprefix(_ID_PREFIX))
            for line in handle
            if "\t" in line
        ]
    positions = np.arange(len(scores))
    expected = np.lexsort((-positions, scores))[::-1][: max(top, 0)]
    np.testing.assert_array_equal(rows, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--indices", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top", type=int, default=10, help="Respondents shown in text previews")
    parser.add_argument(
        "--workflow",
        choices=["screen", "composite"],
        default="screen",
    )
    parser.add_argument(
        "--flagged",
        action="store_true",
        help="Include fixed-threshold composite flags",
    )
    parser.add_argument(
        "--probability",
        action="store_true",
        help="Include uncalibrated logistic composite values",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "json", "npz", "all", "both"],
        default="all",
    )
    parser.add_argument(
        "--compression",
        choices=list(_COMPRESSION_SUFFIXES),
        default="none",
        help="Compress text, CSV, or JSON output with a standard-library codec",
    )
    parser.add_argument(
        "--respondent-ids",
        action="store_true",
        help="Label respondents with string identifiers that CSV must quote",
    )
    args = parser.parse_args()

    if args.respondents < 1 or args.indices < 1 or args.repeats < 1:
        parser.error("respondents, indices, and repeats must be positive")
    if args.flagged and args.workflow != "composite":
        parser.error("--flagged requires --workflow composite")
    if args.probability and args.workflow != "composite":
        parser.error("--probability requires --workflow composite")
    if args.compression != "none" and args.format not in {"text", "csv", "json"}:
        parser.error("--compression requires --format text, csv, or json")

    compression: Compression = args.compression
    respondent_ids = _respondent_ids(args.respondents) if args.respondent_ids else None

    screen_result = (
        _make_screen_result(args.respondents, args.indices) if args.workflow == "screen" else None
    )
    composite_result = (
        _make_composite_result(
            args.respondents,
            args.indices,
            flagged=args.flagged,
            probability=args.probability,
        )
        if args.workflow == "composite"
        else None
    )
    if args.format == "all":
        formats: list[OutputFormat] = ["text", "csv", "json", "npz"]
    elif args.format == "both":
        formats = ["csv", "npz"]
    else:
        formats = [args.format]

    print(
        f"workflow={args.workflow} respondents={args.respondents} "
        f"indices={args.indices} flagged={args.flagged} probability={args.probability} "
        f"compression={compression} top={args.top} respondent_ids={args.respondent_ids}"
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for output_format in formats:
            destination = root / (
                f"{args.workflow}.{output_format}{_COMPRESSION_SUFFIXES[compression]}"
            )
            if args.workflow == "screen":
                assert screen_result is not None
                operation = partial(
                    _write_screen_result,
                    output_format,
                    destination,
                    screen_result,
                    args.top,
                    respondent_ids,
                )
            else:
                assert composite_result is not None
                operation = partial(
                    _write_composite_result,
                    output_format,
                    destination,
                    *composite_result,
                    args.top,
                    respondent_ids,
                )
            seconds, peak_mib, output_mib = _benchmark(
                operation,
                destination,
                args.repeats,
            )
            if output_format == "text":
                if screen_result is not None:
                    ranking_scores = screen_result["flag_counts"]
                else:
                    assert composite_result is not None
                    ranking_scores = composite_result[0]
                _check_text_rows(destination, ranking_scores, args.top)
            print(
                f"{output_format}: median={seconds:.4f}s "
                f"peak={peak_mib:.3f} MiB output={output_mib:.1f} MiB"
            )


if __name__ == "__main__":
    main()
