"""Streaming matrix input helpers for the command-line interface."""

from __future__ import annotations

import csv
import sys
from array import array
from decimal import Decimal, InvalidOperation
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np

from ier._cli_streams import _is_compressed_npy_path, _open_text_path

if TYPE_CHECKING:
    from collections.abc import Iterator

HeaderMode = Literal["auto", "present", "absent"]


def _row_starts_with_non_numeric_value(
    row: list[str], missing_values: frozenset[str] = frozenset()
) -> bool:
    """Return whether the first non-empty cell cannot be parsed as a number."""
    for cell in row:
        stripped = cell.strip()
        if not stripped or stripped in missing_values:
            continue
        try:
            float(stripped)
        except ValueError:
            return True
        return False
    return False


def _parse_numeric_cell(cell: str) -> float:
    """Parse a matrix cell, treating blank delimited fields as missing values."""
    if not cell:
        return np.nan
    try:
        return float(cell)
    except ValueError:
        if not cell.strip():
            return np.nan
        raise


def _parse_numeric_cell_with_missing_values(cell: str, missing_values: frozenset[str]) -> float:
    """Parse a matrix cell with explicit missing-value tokens."""
    stripped = cell.strip()
    return np.nan if not stripped or stripped in missing_values else float(stripped)


def _normalize_missing_values(missing_values: list[str] | None) -> frozenset[str]:
    """Validate and normalize explicit delimited missing-value tokens."""
    if missing_values is None:
        return frozenset()
    normalized = [value.strip() for value in missing_values]
    if any(not value for value in normalized):
        raise ValueError("missing-value tokens must be nonblank")
    if len(set(normalized)) != len(normalized):
        raise ValueError("missing-value tokens cannot contain duplicates")
    return frozenset(normalized)


def _uses_mixed_numeric_whitespace(line: str) -> bool:
    """Recognize extra space-separated numeric cells inside tab-separated fields."""
    tokens = line.split()
    if len(tokens) <= len(line.rstrip("\r\n").split("\t")):
        return False
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def _fallback_delimiter(sample_lines: list[str], *, header_expected: bool = False) -> str | None:
    """Identify a delimited first record even when later rows are jagged.

    The sniffer requires consistent widths, so malformed CSV otherwise falls
    through to whitespace parsing. Inspect fields with CSV quoting rules rather
    than counting punctuation inside quoted names or identifiers.
    """
    best_delimiter = None
    best_score = (0, 0, 1)
    for candidate in (",", "\t", ";"):
        try:
            reader = csv.reader(sample_lines, delimiter=candidate, strict=True)
            first = next(
                (row for row in reader if len(row) > 1 or any(cell.strip() for cell in row)), []
            )
        except csv.Error:
            continue
        if candidate == "\t" and not header_expected:
            # Tabs can also be part of an ordinary whitespace numeric matrix.
            first_line = next((line for line in sample_lines if line.strip()), "")
            if _uses_mixed_numeric_whitespace(first_line):
                continue
        try:
            following = next(
                (row for row in reader if len(row) > 1 or any(cell.strip() for cell in row)), []
            )
        except csv.Error:
            following = []
        score = (int(len(first) == len(following)), int(len(following) > 1), len(first))
        if len(first) > 1 and score > best_score:
            best_delimiter = candidate
            best_score = score
    if best_delimiter is None and any(line.lstrip().startswith('"') for line in sample_lines):
        # A quoted single-column CSV has no field separators to sniff. Preserve
        # its quoting rather than passing literal quotes to whitespace parsing.
        return ","
    return best_delimiter


def _iter_rows_from_stream(
    handle: TextIO,
    delimiter: str | None,
    skip_rows: int = 0,
    *,
    header_expected: bool = False,
) -> Iterator[list[str]]:
    """Yield non-empty rows from a forward-only text stream."""
    for _ in range(skip_rows):
        if not handle.readline():
            break

    sample_lines: list[str] = []
    sample_size = 0
    while sample_size < 4096:
        line = handle.readline()
        if not line:
            break
        if not sample_lines:
            line = line.removeprefix("\ufeff")
        sample_lines.append(line)
        sample_size += len(line)

    lines = chain(sample_lines, handle)
    if delimiter is None:
        sample = "".join(sample_lines)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            delimiter = _fallback_delimiter(sample_lines, header_expected=header_expected)
        if delimiter == "\t" and not header_expected:
            first_line = next((line for line in sample_lines if line.strip()), "")
            if _uses_mixed_numeric_whitespace(first_line):
                delimiter = None
        if delimiter is None:
            for line in lines:
                row = line.split()
                if row:
                    yield row
            return

    record_has_content = False

    def record_lines() -> Iterator[str]:
        nonlocal record_has_content
        for line in lines:
            record_has_content |= bool(line.strip())
            yield line

    reader = csv.reader(record_lines(), delimiter=delimiter, strict=True)
    try:
        for row in reader:
            keep_record = record_has_content
            record_has_content = False
            if row and (keep_record or len(row) > 1):
                yield row
    except csv.Error as err:
        raise ValueError(
            f"malformed delimited input at physical line {skip_rows + reader.line_num}: {err}"
        ) from err


def _input_label(path: Path) -> str:
    """Return a readable source label for errors."""
    return "standard input" if path == Path("-") else str(path)


def _load_npy_input(
    path: Path,
    delimiter: str | None,
    id_column: str | None,
    item_columns: list[str] | None,
    header_mode: HeaderMode,
) -> tuple[np.ndarray, None]:
    """Memory-map one headerless real numeric NumPy matrix."""
    if delimiter is not None:
        raise ValueError("--delimiter is not supported with .npy input")
    if id_column is not None or item_columns is not None:
        raise ValueError("--id-column and --item-columns are not supported with .npy input")
    if header_mode != "auto":
        raise ValueError("--header is not supported with .npy input")

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


def _iter_rows(
    path: Path,
    delimiter: str | None,
    skip_rows: int = 0,
    *,
    header_expected: bool = False,
) -> Iterator[list[str]]:
    """Yield plain, compressed, or standard-input delimited rows."""
    if delimiter is not None and (len(delimiter) != 1 or delimiter in "\r\n"):
        raise ValueError("delimiter must be exactly one non-newline character")

    found = False
    if path == Path("-"):
        for row in _iter_rows_from_stream(
            sys.stdin, delimiter, skip_rows, header_expected=header_expected
        ):
            found = True
            yield row
    else:
        with _open_text_path(path, "r") as handle:
            for row in _iter_rows_from_stream(
                handle, delimiter, skip_rows, header_expected=header_expected
            ):
                found = True
                yield row

    if not found:
        raise ValueError(f"no data rows found in {_input_label(path)}")


def _load_input(
    path: Path,
    delimiter: str | None,
    id_column: str | None = None,
    item_columns: list[str] | None = None,
    header_mode: HeaderMode = "auto",
    missing_values: list[str] | None = None,
    skip_rows: int = 0,
) -> tuple[np.ndarray, list[str] | None]:
    """Stream selected numeric items and optionally preserve a named identifier."""
    missing_value_tokens = _normalize_missing_values(missing_values)
    if isinstance(skip_rows, bool) or not isinstance(skip_rows, int) or skip_rows < 0:
        raise ValueError("skip rows must be a non-negative integer")
    if header_mode not in {"auto", "present", "absent"}:
        raise ValueError("header mode must be 'auto', 'present', or 'absent'")
    if header_mode == "absent" and (id_column is not None or item_columns is not None):
        raise ValueError("--header absent cannot be used with --id-column or --item-columns")
    if _is_compressed_npy_path(path):
        raise ValueError("compressed .npy input is not supported; use uncompressed .npy")
    if path.suffix.casefold() == ".npy":
        if skip_rows:
            raise ValueError("--skip-rows is not supported with .npy input")
        if missing_value_tokens:
            raise ValueError("--missing-value is not supported with .npy input")
        return _load_npy_input(path, delimiter, id_column, item_columns, header_mode)

    source = _input_label(path)
    header_expected = id_column is not None or item_columns is not None or header_mode == "present"
    row_iterator = iter(_iter_rows(path, delimiter, skip_rows, header_expected=header_expected))
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
    if header_expected:
        header = [cell.strip() for cell in first_row]
        data_rows: Iterator[list[str]] = row_iterator
        expected_width = len(header)
        width_reference = "the header"
    elif header_mode == "auto" and _row_starts_with_non_numeric_value(
        first_row, missing_value_tokens
    ):
        data_rows = row_iterator
        expected_width = len(first_row)
        width_reference = "the header"
    else:
        data_rows = chain((first_row,), row_iterator)
        expected_width = len(first_row)
        width_reference = "the first data row"

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
        if len(row) != expected_width:
            raise ValueError(
                f"jagged delimited input in {source}: data row {n_rows + 1} has "
                f"{len(row)} columns; expected {expected_width} to match {width_reference}"
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
            if missing_value_tokens:
                numeric_values.extend(
                    _parse_numeric_cell_with_missing_values(cell, missing_value_tokens)
                    for cell in selected_cells
                )
            else:
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


def _parse_applicable_cell(cell: str) -> bool:
    """Require exactly zero or one without rounding nonbinary decimal tokens."""
    stripped = cell.strip()
    if stripped == "0":
        return False
    if stripped == "1":
        return True
    try:
        value = Decimal(stripped)
    except InvalidOperation as err:
        raise ValueError("mask values must be numeric 0 or 1") from err
    if not value.is_finite() or value not in (0, 1):
        raise ValueError("mask values must be numeric 0 or 1")
    return bool(value)


def _load_applicable_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a Boolean matrix matching the selected response axes, without coercion."""
    try:
        if _is_compressed_npy_path(path):
            raise ValueError("compressed .npy masks are not supported; use uncompressed .npy")
        if path.suffix.casefold() == ".npz":
            raise ValueError("NumPy mask archives are not supported; use one Boolean .npy array")
        if path.suffix.casefold() == ".npy":
            loaded = np.load(path, allow_pickle=False, mmap_mode="r")
            if not isinstance(loaded, np.ndarray):
                loaded.close()
                raise ValueError("expected one Boolean NumPy array, not an archive")
            if loaded.dtype.kind != "b":
                raise ValueError(f"expected a Boolean NumPy mask, got {loaded.dtype}")
            if loaded.shape != shape:
                raise ValueError(f"mask must have shape {shape}, got {loaded.shape}")
            return loaded

        n_rows, n_items = shape
        values = bytearray()
        found_rows = 0
        for found_rows, row in enumerate(_iter_rows(path, None), start=1):
            if found_rows > n_rows:
                raise ValueError(f"mask must have shape {shape}; contains more than {n_rows} rows")
            if len(row) != n_items:
                raise ValueError(
                    f"mask must have shape {shape}; row {found_rows} has {len(row)} columns"
                )
            for column, cell in enumerate(row, start=1):
                try:
                    values.append(_parse_applicable_cell(cell))
                except ValueError as err:
                    raise ValueError(f"row {found_rows}, column {column}: {err}") from err
        if found_rows != n_rows:
            raise ValueError(f"mask must have shape {shape}, got {(found_rows, n_items)}")
        return np.frombuffer(values, dtype=np.bool_).reshape(shape)
    except (OSError, EOFError, ValueError) as err:
        raise ValueError(f"invalid --missing-applicable-mask {path}: {err}") from err
