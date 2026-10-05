"""Benchmark streaming matrix input and optional respondent-applicability masks.

Usage:
    uv run python benchmarks/bench_cli_input.py
    uv run python benchmarks/bench_cli_input.py --respondents 250000 --items 40
    uv run python benchmarks/bench_cli_input.py --masks
    uv run python benchmarks/bench_cli_input.py --id-column
    uv run python benchmarks/bench_cli_input.py --blank-cells
"""

from __future__ import annotations

import argparse
import mmap
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from _measurement import measure

from ier._cli_input import _load_applicable_mask, _load_input


def _close_file_mapping(values: np.ndarray) -> None:
    """Release an owned mapping even when validation tracebacks retain array views."""
    owner: object = values
    while isinstance(owner, np.ndarray):
        owner = owner.base
    if isinstance(owner, mmap.mmap):
        owner.close()


def _write_fixture(
    path: Path,
    n_respondents: int,
    n_items: int,
    preamble_rows: int,
    *,
    id_column: bool = False,
    blank_cells: bool = False,
) -> None:
    header = ",".join(f"item_{index}" for index in range(n_items))
    rows = [
        ",".join(str((index + offset) % 5 + 1) for index in range(n_items)) for offset in range(5)
    ]
    with path.open(mode="w", encoding="utf-8", newline="") as handle:
        for index in range(preamble_rows):
            handle.write(f"survey metadata line {index + 1}\n")
        handle.write(f"id,{header}\n" if id_column else f"{header}\n")
        for respondent in range(n_respondents):
            prefix = f"r{respondent}," if id_column else ""
            row = rows[respondent % 5]
            if blank_cells:
                # Rotate one blank item through the first, middle, and last positions.
                cells = row.split(",")
                cells[respondent % n_items] = ""
                row = ",".join(cells)
            handle.write(f"{prefix}{row}\n")


def _write_mask_fixture(path: Path, n_respondents: int, n_items: int) -> None:
    """Write alternating binary rows without retaining a full in-memory mask."""
    even_items = np.arange(n_items) % 2 == 0
    if path.suffix == ".npy":
        # NumPy's file-backed fixture writer lacks annotations in supported stubs.
        mask = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            path, mode="w+", dtype=bool, shape=(n_respondents, n_items)
        )
        try:
            mask[::2] = even_items
            mask[1::2] = ~even_items
            mask.flush()
        finally:
            _close_file_mapping(mask)
        return
    rows = [
        ",".join("1" if cell else "0" for cell in even_items),
        ",".join("0" if cell else "1" for cell in even_items),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        for respondent in range(n_respondents):
            handle.write(f"{rows[respondent % 2]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--preamble-rows", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--masks", action="store_true", help="Also measure 0/1 CSV and Boolean .npy masks"
    )
    parser.add_argument(
        "--id-column",
        action="store_true",
        help="Prefix a named respondent ID column and select the item columns around it",
    )
    parser.add_argument(
        "--blank-cells",
        action="store_true",
        help="Leave one item blank in every row, rotating through the item positions",
    )
    args = parser.parse_args()

    if args.respondents < 1 or args.items < 1 or args.repeats < 1:
        parser.error("respondents, items, and repeats must be positive")
    if args.preamble_rows < 1:
        parser.error("preamble rows must be positive")
    if args.blank_cells and args.items < 2:
        parser.error("blank cells need at least two items")

    print(
        f"respondents={args.respondents} items={args.items} "
        f"preamble_rows={args.preamble_rows} id_column={args.id_column} "
        f"blank_cells={args.blank_cells}"
    )
    id_column = "id" if args.id_column else None
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cases = [("default", 0), ("skipped", args.preamble_rows)]
        for name, skip_rows in cases:
            path = root / f"{name}.csv"
            _write_fixture(
                path,
                args.respondents,
                args.items,
                skip_rows,
                id_column=args.id_column,
                blank_cells=args.blank_cells,
            )
            operation = partial(_load_input, path, ",", id_column, skip_rows=skip_rows)
            measured = measure(operation, args.repeats)
            matrix, identifiers = measured.result
            expected_identifiers = (
                [f"r{respondent}" for respondent in range(args.respondents)]
                if args.id_column
                else None
            )
            if identifiers != expected_identifiers or matrix.shape != (
                args.respondents,
                args.items,
            ):
                raise RuntimeError("benchmark fixture loaded incorrectly")
            for offset in range(min(5, args.respondents)):
                observed = matrix[offset::5]
                expected = np.broadcast_to(
                    ((np.arange(args.items) + offset) % 5 + 1).astype(np.float64), observed.shape
                )
                if args.blank_cells:
                    expected = expected.copy()
                    respondents = np.arange(offset, args.respondents, 5)
                    expected[np.arange(len(respondents)), respondents % args.items] = np.nan
                np.testing.assert_array_equal(observed, expected)
            print(f"{name}: median={measured.median_seconds:.4f}s peak={measured.peak_mib:.3f} MiB")
        if args.masks:
            shape = (args.respondents, args.items)
            for suffix in (".csv", ".npy"):
                path = root / f"applicable{suffix}"
                _write_mask_fixture(path, *shape)
                mask_measurement = measure(
                    partial(_load_applicable_mask, path, shape), args.repeats
                )
                mask = mask_measurement.result
                try:
                    if mask.dtype != np.dtype(bool) or mask.shape != shape:
                        raise RuntimeError("benchmark mask fixture loaded incorrectly")
                    if suffix == ".npy" and (
                        not isinstance(mask, np.memmap) or mask.flags.writeable
                    ):
                        raise RuntimeError("NumPy masks must remain read-only memory maps")
                    if suffix == ".csv" and isinstance(mask, np.memmap):
                        raise RuntimeError("text masks must load as Boolean arrays")
                    for offset in range(min(2, args.respondents)):
                        observed = mask[offset::2]
                        expected = (np.arange(args.items) + offset) % 2 == 0
                        np.testing.assert_array_equal(
                            observed, np.broadcast_to(expected, observed.shape)
                        )
                    print(
                        f"mask{suffix}: median={mask_measurement.median_seconds:.4f}s "
                        f"peak={mask_measurement.peak_mib:.3f} MiB"
                    )
                finally:
                    _close_file_mapping(mask)


if __name__ == "__main__":
    main()
