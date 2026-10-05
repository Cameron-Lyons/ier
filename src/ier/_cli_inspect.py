"""Parsing and value summaries for the ``ier inspect`` command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ier._cli_output import _JsonArray, _write_json_value

if TYPE_CHECKING:
    from typing import TextIO

    from ier._cli_input import _InputReport

# Rows summarized at once, bounding workspace for wide or memory-mapped matrices.
_BLOCK_ROWS = 16_384
# Names listed in text output before the remainder is counted.
_PREVIEW_LENGTH = 10
# Observed extremes used by fewer respondents than this share are reported.
_RARE_EXTREME_SHARE = 0.01


@dataclass(frozen=True)
class _Inspection:
    """How one response matrix was parsed and which values it contains."""

    source: str
    report: _InputReport
    id_column: str | None
    n_respondents: int
    item_names: list[str] | None
    item_positions: list[int]
    missing_by_item: np.ndarray
    infinite_cells: int
    observed_min: int | float | None
    observed_max: int | float | None
    distinct_values: int
    non_integer_values: bool
    respondents_at_min: int
    respondents_at_max: int

    @property
    def n_items(self) -> int:
        return len(self.item_positions)

    @property
    def missing_cells(self) -> int:
        return int(self.missing_by_item.sum())

    @property
    def suggested_options(self) -> list[str]:
        """Return the scale options that reproduce the bounds inferred from the data.

        Each option carries its value as ``--name=value``, so a negative value in
        exponent form such as ``-1e-05`` is not mistaken for another option.
        """
        if self.observed_min is None or self.observed_max is None:
            return []
        return [
            f"--scale-min={_format_number(self.observed_min)}",
            f"--scale-max={_format_number(self.observed_max)}",
        ]

    @property
    def warnings(self) -> list[str]:
        """Return observed-scale problems that inferred bounds would silently absorb."""
        messages: list[str] = []
        if self.infinite_cells:
            messages.append(
                f"{self.infinite_cells} infinite cells are excluded from the observed values; "
                "check the input for invalid entries"
            )
        if self.observed_min is None or self.observed_max is None:
            return [*messages, "no finite responses were observed; scale bounds cannot be inferred"]
        if self.distinct_values == 1:
            messages.append(
                f"only one distinct value ({_format_number(self.observed_min)}) was observed; "
                "pass --scale-min and --scale-max explicitly"
            )
            return messages
        for name, option, value, used in (
            ("minimum", "--scale-min", self.observed_min, self.respondents_at_min),
            ("maximum", "--scale-max", self.observed_max, self.respondents_at_max),
        ):
            if used < _RARE_EXTREME_SHARE * self.n_respondents:
                messages.append(
                    f"observed {name} {_format_number(value)} is used by only {used} of "
                    f"{self.n_respondents} respondents ({used / self.n_respondents:.1%}); "
                    f"confirm the response scale and pass {option} explicitly if it differs"
                )
        return messages


def _format_number(value: float) -> str:
    """Format a response value as an exact command-line argument."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        return str(int(value))
    return str(value)


def _inspect_matrix(
    matrix: np.ndarray, report: _InputReport, *, source: str, id_column: str | None
) -> _Inspection:
    """Summarize missing cells, observed values, and respondents at each extreme."""
    n_rows, n_items = matrix.shape
    missing = np.zeros(n_items, dtype=np.int64)
    infinite_cells = 0
    non_integer = False
    blocks: list[np.ndarray] = []
    for start in range(0, n_rows, _BLOCK_ROWS):
        block = np.asarray(matrix[start : start + _BLOCK_ROWS])
        if block.dtype.kind == "f":
            absent = np.isnan(block)
            finite = np.isfinite(block)
            missing += absent.sum(axis=0)
            infinite_cells += int(block.size - np.count_nonzero(absent | finite))
            values = block[finite]
            non_integer = non_integer or bool(np.any(values != np.floor(values)))
        else:
            values = block.ravel()
        blocks.append(np.unique(values))
    # Sorted distinct values also give the observed extremes.
    distinct = np.unique(np.concatenate(blocks))

    observed_min: int | float | None = None
    observed_max: int | float | None = None
    at_min = at_max = 0
    if distinct.size:
        low, high = distinct[0], distinct[-1]
        observed_min, observed_max = low.item(), high.item()
        for start in range(0, n_rows, _BLOCK_ROWS):
            block = np.asarray(matrix[start : start + _BLOCK_ROWS])
            at_min += int(np.count_nonzero(np.any(block == low, axis=1)))
            at_max += int(np.count_nonzero(np.any(block == high, axis=1)))

    positions = list(range(n_items)) if report.item_indices is None else report.item_indices
    names = report.column_names
    return _Inspection(
        source=source,
        report=report,
        id_column=id_column,
        n_respondents=n_rows,
        item_names=None if names is None else [names[index] for index in positions],
        item_positions=[index + 1 for index in positions],
        missing_by_item=missing,
        infinite_cells=infinite_cells,
        observed_min=observed_min,
        observed_max=observed_max,
        distinct_values=int(distinct.size),
        non_integer_values=non_integer,
        respondents_at_min=at_min,
        respondents_at_max=at_max,
    )


def _preview(labels: list[str]) -> str:
    """Join the first labels, counting any that are left out."""
    shown = ", ".join(labels[:_PREVIEW_LENGTH])
    hidden = len(labels) - _PREVIEW_LENGTH
    return f"{shown}, ... (+{hidden} more)" if hidden > 0 else shown


def _emit_inspection_text(inspection: _Inspection) -> str:
    """Render an inspection as a short human-readable report."""
    report = inspection.report
    lines = [f"source: {inspection.source}"]
    if report.input_format == "npy":
        lines.append("format: NumPy .npy matrix")
    elif report.delimiter is None:
        lines.append("delimiter: whitespace")
    else:
        lines.append(f"delimiter: {report.delimiter!r} ({report.delimiter_method})")
    lines.append(f"header: {report.header}")
    if inspection.id_column is not None:
        lines.append(f"id column: {inspection.id_column}")
    unselected = report.n_columns - inspection.n_items - (inspection.id_column is not None)
    column_parts = [f"{inspection.n_items} items"]
    if inspection.id_column is not None:
        column_parts.append("1 ID")
    if unselected:
        column_parts.append(f"{unselected} unselected")
    lines.append(f"columns: {report.n_columns} ({', '.join(column_parts)})")
    lines.append(f"respondents: {inspection.n_respondents}")
    if inspection.item_names is None:
        lines.append(f"items: {inspection.n_items} (unnamed columns 1 to {inspection.n_items})")
    else:
        lines.append(f"items: {inspection.n_items} ({_preview(inspection.item_names)})")

    n_cells = inspection.n_respondents * inspection.n_items
    missing_cells = inspection.missing_cells
    # Loaded matrices always have at least one respondent and one item.
    lines.append(f"missing cells: {missing_cells} of {n_cells} ({missing_cells / n_cells:.1%})")
    if missing_cells:
        labels = inspection.item_names or [
            f"column {position}" for position in inspection.item_positions
        ]
        # Most-missing items first; ties keep their selected order.
        order = np.argsort(-inspection.missing_by_item, kind="stable")
        counts = [
            f"{labels[index]}={int(inspection.missing_by_item[index])}"
            for index in order
            if inspection.missing_by_item[index]
        ]
        lines.append(f"missing by item: {_preview(counts)}")

    if inspection.observed_min is None or inspection.observed_max is None:
        lines.append("observed values: none")
    else:
        n = inspection.n_respondents
        lines.extend(
            [
                f"observed values: {_format_number(inspection.observed_min)} to "
                f"{_format_number(inspection.observed_max)} "
                f"({inspection.distinct_values} distinct)",
                f"non-integer values: {'yes' if inspection.non_integer_values else 'no'}",
                f"respondents using the minimum: {inspection.respondents_at_min} "
                f"({inspection.respondents_at_min / n:.1%})",
                f"respondents using the maximum: {inspection.respondents_at_max} "
                f"({inspection.respondents_at_max / n:.1%})",
                f"suggested options: {' '.join(inspection.suggested_options)}",
            ]
        )
    warnings = inspection.warnings
    if warnings:
        lines.append("warnings:")
        lines.extend(f"  {message}" for message in warnings)
    return "\n".join(lines)


def _write_inspection_json(handle: TextIO, inspection: _Inspection) -> None:
    """Write an inspection as strict JSON through the shared streaming writer."""
    report = inspection.report
    payload: dict[str, object] = {
        "source": inspection.source,
        "input_format": report.input_format,
        "delimiter": report.delimiter,
        "delimiter_detection": report.delimiter_method,
        "header": report.header,
        "id_column": inspection.id_column,
        "n_columns": report.n_columns,
        "n_respondents": inspection.n_respondents,
        "n_items": inspection.n_items,
        "item_names": (
            None if inspection.item_names is None else _JsonArray(inspection.item_names, "string")
        ),
        "item_positions": _JsonArray(inspection.item_positions, "integer"),
        "missing_cells": inspection.missing_cells,
        "missing_by_item": _JsonArray(inspection.missing_by_item, "integer"),
        "infinite_cells": inspection.infinite_cells,
        "observed_min": inspection.observed_min,
        "observed_max": inspection.observed_max,
        "distinct_values": inspection.distinct_values,
        "non_integer_values": inspection.non_integer_values,
        "respondents_at_min": inspection.respondents_at_min,
        "respondents_at_max": inspection.respondents_at_max,
        "suggested_options": inspection.suggested_options,
        "warnings": inspection.warnings,
    }
    _write_json_value(handle, payload)
