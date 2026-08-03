"""Streaming matrix input helpers for the command-line interface."""

from __future__ import annotations

import csv
import gzip
import sys
from array import array
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


def _row_starts_with_non_numeric_value(row: list[str]) -> bool:
    """Return whether the first non-empty cell cannot be parsed as a number."""
    for cell in row:
        if not cell.strip():
            continue
        try:
            float(cell)
        except ValueError:
            return True
        return False
    return False


def _parse_numeric_cell(cell: str) -> float:
    """Parse a matrix cell, treating blank delimited fields as missing values."""
    return float(cell) if cell.strip() else np.nan


def _iter_rows_from_stream(handle: TextIO, delimiter: str | None) -> Iterator[list[str]]:
    """Yield non-empty rows from a forward-only text stream."""
    sample_lines: list[str] = []
    sample_size = 0
    while sample_size < 4096:
        line = handle.readline()
        if not line:
            break
        sample_lines.append(line)
        sample_size += len(line)

    lines = chain(sample_lines, handle)
    if delimiter is None:
        sample = "".join(sample_lines)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            for line in lines:
                row = line.split()
                if row:
                    yield row
            return

    reader = csv.reader(lines, delimiter=delimiter)
    for row in reader:
        if row and any(cell.strip() for cell in row):
            yield row


def _input_label(path: Path) -> str:
    """Return a readable source label for errors."""
    return "standard input" if path == Path("-") else str(path)


def _load_npy_input(
    path: Path,
    delimiter: str | None,
    id_column: str | None,
    item_columns: list[str] | None,
) -> tuple[np.ndarray, None]:
    """Memory-map one headerless real numeric NumPy matrix."""
    if delimiter is not None:
        raise ValueError("--delimiter is not supported with .npy input")
    if id_column is not None or item_columns is not None:
        raise ValueError("--id-column and --item-columns are not supported with .npy input")

    try:
        loaded = np.load(path, allow_pickle=False, mmap_mode="r")
    except (EOFError, ValueError) as err:
        raise ValueError(f"failed to load NumPy matrix from {path}: {err}") from err

    if not isinstance(loaded, np.ndarray):
        loaded.close()
        raise ValueError(f"expected one NumPy array in {path}, not an archive")
    matrix = loaded

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"expected a non-empty 2D NumPy matrix in {path}")
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
        matrix.dtype, np.complexfloating
    ):
        raise ValueError(f"expected a real numeric NumPy matrix in {path}, got {matrix.dtype}")
    return matrix, None


def _iter_rows(path: Path, delimiter: str | None) -> Iterator[list[str]]:
    """Yield plain, gzip-compressed, or standard-input delimited rows."""
    if delimiter is not None and (len(delimiter) != 1 or delimiter in "\r\n"):
        raise ValueError("delimiter must be exactly one non-newline character")

    found = False
    if path == Path("-"):
        for row in _iter_rows_from_stream(sys.stdin, delimiter):
            found = True
            yield row
    elif path.suffix.casefold() == ".gz":
        with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
            for row in _iter_rows_from_stream(handle, delimiter):
                found = True
                yield row
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in _iter_rows_from_stream(handle, delimiter):
                found = True
                yield row

    if not found:
        raise ValueError(f"no data rows found in {_input_label(path)}")


def _load_input(
    path: Path,
    delimiter: str | None,
    id_column: str | None = None,
    item_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Stream selected numeric items and optionally preserve a named identifier."""
    if path.name.casefold().endswith(".npy.gz"):
        raise ValueError("compressed .npy input is not supported; use uncompressed .npy")
    if path.suffix.casefold() == ".npy":
        return _load_npy_input(path, delimiter, id_column, item_columns)

    source = _input_label(path)
    row_iterator = iter(_iter_rows(path, delimiter))
    first_row = next(row_iterator)

    selected_names: list[str] | None = None
    if item_columns is not None:
        selected_names = [name.strip() for name in item_columns]
        if not selected_names or any(not name for name in selected_names):
            raise ValueError("item columns must include at least one nonblank name")
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("item columns cannot contain duplicate names")

    id_index: int | None = None
    item_indices: list[int] | None = None
    header: list[str] | None = None
    if id_column is not None or selected_names is not None:
        header = [cell.strip() for cell in first_row]
        data_rows: Iterator[list[str]] = row_iterator
        expected_width: int | None = len(header)
    elif _row_starts_with_non_numeric_value(first_row):
        data_rows = row_iterator
        expected_width = None
    else:
        data_rows = chain((first_row,), row_iterator)
        expected_width = len(first_row)

    if id_column is not None:
        assert header is not None
        matches = [index for index, name in enumerate(header) if name == id_column]
        if not matches:
            raise ValueError(f"ID column '{id_column}' was not found in the header")
        if len(matches) > 1:
            raise ValueError(f"ID column '{id_column}' appears more than once in the header")
        id_index = matches[0]

    if selected_names is not None:
        assert header is not None
        item_indices = []
        for name in selected_names:
            matches = [index for index, header_name in enumerate(header) if header_name == name]
            if not matches:
                raise ValueError(f"item column '{name}' was not found in the header")
            if len(matches) > 1:
                raise ValueError(f"item column '{name}' appears more than once in the header")
            if matches[0] == id_index:
                raise ValueError(
                    f"ID column '{id_column}' cannot also be selected as an item column"
                )
            item_indices.append(matches[0])
    elif id_index is not None:
        assert header is not None
        item_indices = [index for index in range(len(header)) if index != id_index]
        if not item_indices:
            raise ValueError("input must contain at least one item column besides the ID column")

    identifiers: list[str] | None = [] if id_index is not None else None
    seen_identifiers: set[str] = set()
    numeric_values = array("d")
    n_rows = 0
    n_items = len(item_indices) if item_indices is not None else None

    for row in data_rows:
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            widths = sorted({expected_width, len(row)})
            raise ValueError(
                f"jagged delimited input in {source}: rows have unequal lengths {widths}; "
                "expected a rectangular respondent×item matrix"
            )

        if id_index is not None:
            assert identifiers is not None
            identifier = row[id_index].strip()
            if not identifier:
                raise ValueError(f"ID column '{id_column}' contains blank values")
            if identifier in seen_identifiers:
                raise ValueError(f"ID column '{id_column}' contains duplicate values")
            identifiers.append(identifier)
            seen_identifiers.add(identifier)

        selected_cells = (
            (row[index] for index in item_indices) if item_indices is not None else iter(row)
        )
        try:
            numeric_values.extend(_parse_numeric_cell(cell) for cell in selected_cells)
        except ValueError as err:
            raise ValueError(f"failed to parse numeric matrix from {source}: {err}") from err

        if n_items is None:
            n_items = len(row)
        n_rows += 1

    if n_rows == 0 or n_items is None:
        raise ValueError(f"no numeric data rows found in {source}")

    matrix = np.frombuffer(numeric_values, dtype=np.float64).reshape(n_rows, n_items)
    return matrix, identifiers


def _load_matrix(path: Path, delimiter: str | None) -> np.ndarray:
    """Load a respondent × item matrix from delimited text or NumPy binary."""
    matrix, _ = _load_input(path, delimiter)
    return matrix
