"""Streaming matrix input helpers for the command-line interface."""

from __future__ import annotations

import csv
import sys
from array import array
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fnmatch import fnmatchcase
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np

from ier._cli_streams import _is_compressed_npy_path, _open_text_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

HeaderMode = Literal["auto", "present", "absent"]
DelimiterMethod = Literal["explicit", "sniffer", "fallback", "whitespace"]
HeaderDecision = Literal["present", "auto-detected", "absent"]


@dataclass
class _InputReport:
    """Parsing decisions recorded by _load_input for ``ier inspect``.

    Recording happens once per input, outside the per-row loop, so supplying a
    report never changes how rows are read or parsed.
    """

    input_format: Literal["delimited", "npy"] = "delimited"
    delimiter: str | None = None
    delimiter_method: DelimiterMethod | None = None
    header: HeaderDecision = "absent"
    column_names: list[str] | None = None
    n_columns: int = 0
    id_index: int | None = None
    item_indices: list[int] | None = None


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


def _parses_as_float(token: str) -> bool:
    """Return whether float() accepts a token unchanged."""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _cell_selector(item_indices: list[int] | None) -> Callable[[list[str]], Sequence[str]] | None:
    """Return a C-level selector for item cells, or None to keep complete rows."""
    if item_indices is None:
        return None
    if len(item_indices) == 1:
        # A one-item getter returns a bare cell; a slice keeps a sequence.
        return itemgetter(slice(item_indices[0], item_indices[0] + 1))
    return itemgetter(*item_indices)


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
    report: _InputReport | None = None,
) -> Iterator[list[str]]:
    """Yield non-empty rows from a forward-only text stream.

    A supplied report records the delimiter and how it was chosen before the
    first row is yielded.
    """
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
    method: DelimiterMethod = "explicit"
    if delimiter is None:
        sample = "".join(sample_lines)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
            method = "sniffer"
        except csv.Error:
            delimiter = _fallback_delimiter(sample_lines, header_expected=header_expected)
            method = "fallback"
        if delimiter == "\t" and not header_expected:
            first_line = next((line for line in sample_lines if line.strip()), "")
            if _uses_mixed_numeric_whitespace(first_line):
                delimiter = None
        if delimiter is None:
            if report is not None:
                report.delimiter, report.delimiter_method = None, "whitespace"
            for line in lines:
                row = line.split()
                if row:
                    yield row
            return
    if report is not None:
        report.delimiter, report.delimiter_method = delimiter, method

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
    item_patterns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> tuple[np.ndarray, None]:
    """Memory-map one headerless real numeric NumPy matrix."""
    if delimiter is not None:
        raise ValueError("--delimiter is not supported with .npy input")
    if id_column is not None or item_columns is not None:
        raise ValueError("--id-column and --item-columns are not supported with .npy input")
    if item_patterns is not None or exclude_columns is not None:
        raise ValueError("--item-pattern and --exclude-column are not supported with .npy input")
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
    report: _InputReport | None = None,
) -> Iterator[list[str]]:
    """Yield plain, compressed, or standard-input delimited rows."""
    if delimiter is not None and (len(delimiter) != 1 or delimiter in "\r\n"):
        raise ValueError("delimiter must be exactly one non-newline character")

    found = False
    if path == Path("-"):
        for row in _iter_rows_from_stream(
            sys.stdin, delimiter, skip_rows, header_expected=header_expected, report=report
        ):
            found = True
            yield row
    else:
        with _open_text_path(path, "r") as handle:
            for row in _iter_rows_from_stream(
                handle, delimiter, skip_rows, header_expected=header_expected, report=report
            ):
                found = True
                yield row

    if not found:
        raise ValueError(f"no data rows found in {_input_label(path)}")


def _normalize_names(names: list[str] | None, noun: str) -> list[str] | None:
    """Strip requested header names, rejecting blank and repeated entries."""
    if names is None:
        return None
    normalized = [name.strip() for name in names]
    if not normalized or any(not name for name in normalized):
        raise ValueError(f"{noun} must include at least one nonblank name")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{noun} cannot contain duplicate names")
    return normalized


def _header_position(header: list[str], name: str, noun: str) -> int:
    """Return the position of one header name that must occur exactly once."""
    matches = [index for index, header_name in enumerate(header) if header_name == name]
    if not matches:
        raise ValueError(f"{noun} '{name}' was not found in the header")
    if len(matches) > 1:
        raise ValueError(f"{noun} '{name}' appears more than once in the header")
    return matches[0]


def _resolve_item_indices(
    header: list[str] | None,
    id_column: str | None,
    item_columns: list[str] | None,
    item_patterns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> tuple[int | None, list[int] | None]:
    """Resolve named identifier and item selections to header positions.

    Exact item names keep their requested order, while glob patterns select
    matching columns once each in header order. Exclusions apply after either
    selection; on their own they select every column except the identifier.
    """
    selected_names = _normalize_names(item_columns, "item columns")
    excluded_names = _normalize_names(exclude_columns, "excluded columns")
    patterns = None
    if item_patterns is not None:
        patterns = [pattern.strip() for pattern in item_patterns]
        if not patterns or any(not pattern for pattern in patterns):
            raise ValueError("item patterns must include at least one nonblank pattern")
    if header is None:
        return None, None

    id_index = None if id_column is None else _header_position(header, id_column, "ID column")
    item_indices: list[int] | None = None
    if selected_names is not None:
        item_indices = []
        for name in selected_names:
            position = _header_position(header, name, "item column")
            if position == id_index:
                raise ValueError(
                    f"ID column '{id_column}' cannot also be selected as an item column"
                )
            item_indices.append(position)
    elif patterns is not None:
        matched: set[int] = set()
        for pattern in patterns:
            matches = {index for index, name in enumerate(header) if fnmatchcase(name, pattern)}
            if not matches:
                raise ValueError(f"item pattern '{pattern}' matched no columns")
            matched |= matches
        item_indices = sorted(matched)
    elif id_index is not None or excluded_names is not None:
        item_indices = [index for index in range(len(header)) if index != id_index]
        if not item_indices:
            raise ValueError("input must contain at least one item column besides the ID column")

    if excluded_names is not None:
        assert item_indices is not None
        excluded = {_header_position(header, name, "excluded column") for name in excluded_names}
        item_indices = [index for index in item_indices if index not in excluded]
    if item_indices is not None and id_index in item_indices:
        raise ValueError(
            f"ID column '{id_column}' cannot also be selected as an item column; "
            "exclude it with --exclude-column"
        )
    if item_indices == []:
        raise ValueError("column selection leaves no item columns to score")
    return id_index, item_indices


def _load_input(
    path: Path,
    delimiter: str | None,
    id_column: str | None = None,
    item_columns: list[str] | None = None,
    header_mode: HeaderMode = "auto",
    missing_values: list[str] | None = None,
    skip_rows: int = 0,
    item_patterns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    *,
    report: _InputReport | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Stream selected numeric items and optionally preserve a named identifier.

    A supplied report receives the detected delimiter, header decision, column
    names, and resolved identifier and item positions.
    """
    missing_value_tokens = _normalize_missing_values(missing_values)
    if isinstance(skip_rows, bool) or not isinstance(skip_rows, int) or skip_rows < 0:
        raise ValueError("skip rows must be a non-negative integer")
    if header_mode not in {"auto", "present", "absent"}:
        raise ValueError("header mode must be 'auto', 'present', or 'absent'")
    if header_mode == "absent" and (id_column is not None or item_columns is not None):
        raise ValueError("--header absent cannot be used with --id-column or --item-columns")
    if header_mode == "absent" and (item_patterns is not None or exclude_columns is not None):
        raise ValueError("--header absent cannot be used with --item-pattern or --exclude-column")
    if _is_compressed_npy_path(path):
        raise ValueError("compressed .npy input is not supported; use uncompressed .npy")
    if path.suffix.casefold() == ".npy":
        if skip_rows:
            raise ValueError("--skip-rows is not supported with .npy input")
        if missing_value_tokens:
            raise ValueError("--missing-value is not supported with .npy input")
        loaded = _load_npy_input(
            path, delimiter, id_column, item_columns, header_mode, item_patterns, exclude_columns
        )
        if report is not None:
            report.input_format = "npy"
            report.n_columns = loaded[0].shape[1]
        return loaded

    source = _input_label(path)
    header_expected = (
        id_column is not None
        or item_columns is not None
        or item_patterns is not None
        or exclude_columns is not None
        or header_mode == "present"
    )
    row_iterator = iter(
        _iter_rows(path, delimiter, skip_rows, header_expected=header_expected, report=report)
    )
    first_row = next(row_iterator)

    header: list[str] | None = None
    column_names: list[str] | None = None
    header_decision: HeaderDecision = "absent"
    if header_expected:
        header = [cell.strip() for cell in first_row]
        column_names = header
        data_rows: Iterator[list[str]] = row_iterator
        expected_width = len(header)
        width_reference = "the header"
        header_decision = "present"
    elif header_mode == "auto" and _row_starts_with_non_numeric_value(
        first_row, missing_value_tokens
    ):
        column_names = [cell.strip() for cell in first_row]
        data_rows = row_iterator
        expected_width = len(first_row)
        width_reference = "the header"
        header_decision = "auto-detected"
    else:
        data_rows = chain((first_row,), row_iterator)
        expected_width = len(first_row)
        width_reference = "the first data row"

    id_index, item_indices = _resolve_item_indices(
        header, id_column, item_columns, item_patterns, exclude_columns
    )
    if report is not None:
        report.header = header_decision
        report.column_names = column_names
        report.n_columns = expected_width
        report.id_index = id_index
        report.item_indices = item_indices

    identifiers: list[str] | None = [] if id_index is not None else None
    seen_identifiers: set[str] = set()
    numeric_values = array("d")
    n_rows = 0
    n_items = len(item_indices) if item_indices is not None else None
    select = _cell_selector(item_indices)
    # float() already parses every complete numeric row exactly as the per-cell
    # path would. Numeric missing-value tokens such as -99 must instead match
    # their exact text, so they keep every row on the per-cell path.
    fast_rows = not any(_parses_as_float(token) for token in missing_value_tokens)

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

        cells = select(row) if select is not None else row
        start = len(numeric_values)
        if fast_rows:
            try:
                numeric_values.extend(map(float, cells))
            except ValueError:
                # Extending retains the converted prefix before the failing cell,
                # and float() already parsed it as the per-cell path would. Only
                # the cells from the failing one onward still need cell rules.
                cells = cells[len(numeric_values) - start :]
            else:
                cells = ()
        if cells:
            try:
                if missing_value_tokens:
                    numeric_values.extend(
                        _parse_numeric_cell_with_missing_values(cell, missing_value_tokens)
                        for cell in cells
                    )
                else:
                    numeric_values.extend(map(_parse_numeric_cell, cells))
            except ValueError as err:
                # The retained prefix length identifies the failing selected cell.
                offset = len(numeric_values) - start
                column = item_indices[offset] if item_indices is not None else offset
                location = (
                    f"column {column + 1}"
                    if column_names is None
                    else f"column {column_names[column]!r} (position {column + 1})"
                )
                raise ValueError(
                    f"failed to parse numeric matrix from {source}: "
                    f"data row {n_rows + 1}, {location}: {err}"
                ) from err

        if n_items is None:
            n_items = len(row)
        n_rows += 1

    if n_rows == 0 or n_items is None:
        raise ValueError(f"no numeric data rows found in {source}")

    matrix = np.frombuffer(numeric_values, dtype=np.float64).reshape(n_rows, n_items)
    return matrix, identifiers


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
