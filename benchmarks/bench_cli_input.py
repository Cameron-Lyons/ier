"""Benchmark streaming delimited input with and without skipped preamble rows.

Usage:
    uv run python benchmarks/bench_cli_input.py
    uv run python benchmarks/bench_cli_input.py --respondents 250000 --items 40
"""

from __future__ import annotations

import argparse
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from _measurement import measure

from ier._cli_input import _load_input


def _write_fixture(
    path: Path,
    n_respondents: int,
    n_items: int,
    preamble_rows: int,
) -> None:
    header = ",".join(f"item_{index}" for index in range(n_items))
    rows = [
        ",".join(str((index + offset) % 5 + 1) for index in range(n_items)) for offset in range(5)
    ]
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        for index in range(preamble_rows):
            handle.write(f"survey metadata line {index + 1}\n")
        handle.write(f"{header}\n")
        for respondent in range(n_respondents):
            handle.write(f"{rows[respondent % 5]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--preamble-rows", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.respondents < 1 or args.items < 1 or args.repeats < 1:
        parser.error("respondents, items, and repeats must be positive")
    if args.preamble_rows < 1:
        parser.error("preamble rows must be positive")

    print(f"respondents={args.respondents} items={args.items} preamble_rows={args.preamble_rows}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cases = [("default", 0), ("skipped", args.preamble_rows)]
        for name, skip_rows in cases:
            path = root / f"{name}.csv"
            _write_fixture(path, args.respondents, args.items, skip_rows)
            operation = partial(_load_input, path, ",", skip_rows=skip_rows)
            measured = measure(operation, args.repeats)
            matrix, identifiers = measured.result
            if identifiers is not None or matrix.shape != (args.respondents, args.items):
                raise RuntimeError("benchmark fixture loaded incorrectly")
            for offset in range(min(5, args.respondents)):
                observed = matrix[offset::5]
                expected = (np.arange(args.items) + offset) % 5 + 1
                np.testing.assert_array_equal(observed, np.broadcast_to(expected, observed.shape))
            print(f"{name}: median={measured.median_seconds:.4f}s peak={measured.peak_mib:.3f} MiB")


if __name__ == "__main__":
    main()
