"""Chunked CLI writers and whole-row CSV parsing keep per-cell results byte for byte."""

from __future__ import annotations

import csv
import zipfile
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from ier import save_response_time_archive
from ier._cli_composite import CompositeReport, ResponseTimeReport
from ier._cli_input import (
    _load_input,
    _parse_numeric_cell,
    _parse_numeric_cell_with_missing_values,
)
from ier._cli_output import (
    _CSV_CHUNK_SIZE,
    _emit_composite_text,
    _emit_response_time_text,
    _emit_screen_text,
    _JsonArray,
    _respondent_label_values,
    _write_composite_csv,
    _write_composite_json,
    _write_response_time_csv,
    _write_response_time_json,
    _write_screen_csv,
    _write_screen_json,
)
from ier.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from typing import Literal, TextIO

    from ier.types import ScreenResult

_SIZES = [1, 1023, 1024, 1025, 3000]
_JSON_SIZES = [1, 1025, 4097]
_NON_FINITE = {1022: np.nan, 1023: np.inf, 1024: -np.inf, 1025: np.nan}
_QUOTED_LABELS = ("plain", "comma,separated", 'say "hi"', "line\nbreak", "  padded  ")


# Reference implementations: the per-cell serializers the chunked writers replace.


def _reference_csv_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _reference_screen_csv(
    handle: TextIO, result: ScreenResult, respondent_ids: list[str] | None = None
) -> None:
    n = result["n_respondents"]
    scores = result["scores"]
    flags = result["flags"]
    fieldnames = [
        "respondent",
        "flag_count",
        "valid_index_count",
        "consensus_eligible",
        "consensus_flag",
    ]
    for name in result["indices_used"]:
        fieldnames.extend([f"{name}_score", f"{name}_flag"])

    counts = np.asarray(result["flag_counts"])
    valid_counts = np.asarray(result["valid_index_counts"])
    eligible = np.asarray(result["consensus_eligible"])
    consensus = np.asarray(result["consensus_flags"])
    labels = _respondent_label_values(n, respondent_ids)
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(n):
        row: dict[str, object] = {
            "respondent": labels[i],
            "flag_count": int(counts[i]),
            "valid_index_count": int(valid_counts[i]),
            "consensus_eligible": int(bool(eligible[i])),
            "consensus_flag": int(bool(consensus[i])),
        }
        for name in result["indices_used"]:
            row[f"{name}_score"] = _reference_csv_number(scores[name][i])
            row[f"{name}_flag"] = int(bool(flags[name][i]))
        writer.writerow(row)


def _reference_composite_csv(
    handle: TextIO,
    scores: np.ndarray,
    respondent_ids: list[str] | None = None,
    component_scores: Mapping[str, np.ndarray] | None = None,
    valid_index_counts: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> None:
    detail_names = list(component_scores) if component_scores is not None else []
    writer = csv.writer(handle)
    header = ["respondent", "composite_score"]
    if probabilities is not None:
        header.append("composite_probability")
    if flags is not None:
        header.append("composite_flag")
    if component_scores is not None:
        header.extend(["valid_index_count", *(f"{name}_score" for name in detail_names)])
    writer.writerow(header)
    labels = _respondent_label_values(len(scores), respondent_ids)
    for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        row: list[object] = [label, _reference_csv_number(score)]
        if probabilities is not None:
            row.append(_reference_csv_number(probabilities[index]))
        if flags is not None:
            row.append(int(bool(flags[index])))
        if component_scores is not None:
            assert valid_index_counts is not None
            row.append(int(valid_index_counts[index]))
            row.extend(
                _reference_csv_number(component_scores[name][index]) for name in detail_names
            )
        writer.writerow(row)


def _reference_response_time_csv(
    handle: TextIO,
    scores: np.ndarray,
    flags: np.ndarray,
    respondent_ids: list[str] | None = None,
) -> None:
    writer = csv.writer(handle)
    writer.writerow(["respondent", "response_time_score", "response_time_flag"])
    labels = _respondent_label_values(len(scores), respondent_ids)
    for label, score, flag in zip(labels, scores, flags, strict=True):
        writer.writerow([label, _reference_csv_number(score), int(bool(flag))])


def _reference_json_chunk_values(array: _JsonArray, start: int, stop: int) -> list[object]:
    values = array.values[start:stop]
    if array.kind == "number":
        numeric = np.asarray(values, dtype=float)
        chunk: list[object] = numeric.tolist()
        for index in np.flatnonzero(~np.isfinite(numeric)):
            chunk[int(index)] = None
        return chunk
    if array.kind == "integer":
        return [int(value) for value in np.asarray(values, dtype=np.int_)]
    if array.kind == "boolean":
        return [bool(value) for value in np.asarray(values, dtype=np.bool_)]
    return [str(value) for value in values]


def _render(writer: Callable[..., None], *args: Any, **kwargs: Any) -> str:
    output = StringIO()
    writer(output, *args, **kwargs)
    return output.getvalue()


def _composite_report(
    scores: np.ndarray, respondent_ids: list[str] | None = None, **details: Any
) -> CompositeReport:
    """Build a mean composite report, pairing any flags with a fixed threshold."""
    threshold = 1.0 if details.get("flags") is not None else None
    return CompositeReport(scores, "mean", respondent_ids, flag_threshold=threshold, **details)


def _timing_report(
    scores: np.ndarray,
    flags: np.ndarray,
    respondent_ids: list[str] | None = None,
    direction: Literal["high", "low"] = "low",
) -> ResponseTimeReport:
    """Build a median timing report with a fixed cutoff."""
    return ResponseTimeReport(scores, flags, "median", direction, 1.0, respondent_ids)


def _labels(n: int) -> list[str]:
    """Identifiers that need CSV quoting for commas, quotes, and embedded newlines."""
    return [f"{_QUOTED_LABELS[index % len(_QUOTED_LABELS)]}-{index}" for index in range(n)]


def _scores(rng: np.random.Generator, n: int, dtype: type[np.floating[Any]] = np.float64) -> Any:
    """Mixed-magnitude scores with non-finite values on both sides of a chunk boundary."""
    values = rng.normal(size=n) * 10.0 ** rng.integers(-5, 6, size=n)
    values[::7] = np.round(values[::7])
    values[3::11] = -0.0
    for row, value in _NON_FINITE.items():
        if row < n:
            values[row] = value
    values[-1] = np.inf
    return values.astype(dtype)


def _screen_result(n: int) -> ScreenResult:
    rng = np.random.default_rng(n)
    scores = {
        "irv": _scores(rng, n),
        "longstring": rng.integers(0, 20, size=n),
        "mahad": _scores(rng, n, np.float32),
    }
    flags = {
        "irv": rng.random(n) < 0.3,
        "longstring": rng.integers(0, 3, size=n).astype(np.int8),
        "mahad": rng.random(n) < 0.5,
    }
    counts = sum(np.asarray(values, dtype=bool).astype(np.int_) for values in flags.values())
    valid_counts = rng.integers(1, 4, size=n)
    eligible = valid_counts >= 2
    names = list(scores)
    return {
        "scores": scores,
        "flags": flags,
        "thresholds": {name: 1.0 for name in names},
        "threshold_sources": {name: "fixed" for name in names},
        "percentiles": {name: None for name in names},
        "flag_counts": np.asarray(counts),
        "valid_index_counts": valid_counts,
        "consensus_eligible": eligible,
        "consensus_flags": eligible & (np.asarray(counts) >= 2),
        "min_flags": 2,
        "min_valid_indices": 2,
        "n_indices": len(names),
        "indices_used": names,
        "errors": {},
        "n_respondents": n,
        "summary": {
            name: {
                "mean": 0.0,
                "std": 1.0,
                "min": -1.0,
                "max": 1.0,
                "n_valid": n,
                "n_unavailable": 0,
                "n_flagged": 0,
                "flag_rate": 0.0,
            }
            for name in names
        },
    }


def _composite_inputs(n: int) -> dict[str, Any]:
    rng = np.random.default_rng(n + 1)
    scores = _scores(rng, n)
    probabilities = rng.random(n)
    probabilities[~np.isfinite(scores)] = np.nan
    return {
        "scores": scores,
        "component_scores": {
            "irv": _scores(rng, n),
            "longstring": rng.integers(0, 20, size=n).astype(np.float64),
        },
        "valid_index_counts": rng.integers(0, 3, size=n),
        "flags": rng.random(n) < 0.05,
        "probabilities": probabilities,
    }


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("with_ids", [False, True])
def test_screen_csv_matches_per_cell_writer(n: int, with_ids: bool) -> None:
    result = _screen_result(n)
    ids = _labels(n) if with_ids else None

    chunked = _render(_write_screen_csv, result, ids)

    assert chunked == _render(_reference_screen_csv, result, ids)
    assert chunked.count("\r\n") >= n + 1


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("with_ids", [False, True])
@pytest.mark.parametrize("details", [False, True])
def test_composite_csv_matches_per_cell_writer(n: int, with_ids: bool, details: bool) -> None:
    inputs = _composite_inputs(n)
    scores = inputs.pop("scores")
    ids = _labels(n) if with_ids else None
    options = inputs if details else {}

    chunked = _render(_write_composite_csv, _composite_report(scores, ids, **options))

    assert chunked == _render(_reference_composite_csv, scores, ids, **options)


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("with_ids", [False, True])
@pytest.mark.parametrize("direction", ["high", "low"])
def test_response_time_csv_matches_per_cell_writer(
    n: int, with_ids: bool, direction: Literal["high", "low"]
) -> None:
    scores = _scores(np.random.default_rng(n + 2), n)
    flags = scores >= 0.0 if direction == "high" else scores <= 0.0
    ids = _labels(n) if with_ids else None

    chunked = _render(_write_response_time_csv, _timing_report(scores, flags, ids, direction))

    assert chunked == _render(_reference_response_time_csv, scores, flags, ids)


def test_response_time_csv_rejects_misaligned_flags_before_writing() -> None:
    output = StringIO()
    with pytest.raises(ValueError, match="flag length must match score length"):
        _write_response_time_csv(
            output, _timing_report(np.arange(3.0), np.array([True, False, True, False]))
        )
    assert output.getvalue() == ""


def test_screen_csv_rejects_short_respondent_arrays() -> None:
    result = _screen_result(5)
    result["scores"]["irv"] = result["scores"]["irv"][:4]
    with pytest.raises(ValueError, match="zip"):
        _write_screen_csv(StringIO(), result)


@pytest.mark.parametrize("n", _JSON_SIZES)
@pytest.mark.parametrize("with_ids", [False, True])
def test_json_writers_match_per_element_chunks(n: int, with_ids: bool) -> None:
    ids = _labels(n) if with_ids else None
    result = _screen_result(n)
    composite = _composite_inputs(n)
    scores = composite.pop("scores")
    timing = _scores(np.random.default_rng(n + 3), n)
    timing_flags = timing <= 0.0

    def render_all() -> list[str]:
        return [
            _render(_write_screen_json, result, ids),
            _render(_write_composite_json, _composite_report(scores, ids, **composite)),
            _render(_write_response_time_json, _timing_report(timing, timing_flags, ids)),
        ]

    chunked = render_all()
    with patch("ier._cli_output._json_chunk_values", _reference_json_chunk_values):
        reference = render_all()

    assert chunked == reference


def _render_flags(writer: str, output: str, flags: np.ndarray) -> str:
    """Render one writer with every Boolean respondent column set to the given flags."""
    if writer == "screen":
        result = _screen_result(len(flags))
        result["consensus_eligible"] = flags
        result["consensus_flags"] = flags
        result["flags"] = {name: flags for name in result["flags"]}
        if output == "csv":
            return _render(_write_screen_csv, result)
        if output == "json":
            return _render(_write_screen_json, result)
        return _emit_screen_text(result, len(flags))
    if writer == "composite":
        scores = np.linspace(1.5, -0.5, len(flags))
        if output == "csv":
            return _render(_write_composite_csv, _composite_report(scores, flags=flags))
        if output == "json":
            return _render(_write_composite_json, _composite_report(scores, flags=flags))
        return _emit_composite_text(_composite_report(scores, flags=flags), len(flags))
    scores = np.linspace(0.5, 2.0, len(flags))
    if output == "csv":
        return _render(_write_response_time_csv, _timing_report(scores, flags))
    if output == "json":
        return _render(_write_response_time_json, _timing_report(scores, flags))
    return _emit_response_time_text(_timing_report(scores, flags), len(flags))


@pytest.mark.parametrize("writer", ["screen", "composite", "response_time"])
@pytest.mark.parametrize("output", ["csv", "json", "text"])
def test_writers_read_any_nonzero_bool_byte_as_true(writer: str, output: str) -> None:
    # NumPy bool storage only promises zero or nonzero; views and foreign NPZ
    # members can store True as any byte from 1 to 255.
    stored = np.array([2, 0], np.uint8).view(bool)
    assert stored.tolist() == [True, False]

    rendered = _render_flags(writer, output, stored)

    assert rendered == _render_flags(writer, output, np.array([True, False]))
    if output == "csv":
        records = list(csv.DictReader(StringIO(rendered)))
        flag_columns = [name for name in records[0] if name.endswith(("_flag", "_eligible"))]
        assert flag_columns
        for name in flag_columns:
            assert [record[name] for record in records] == ["1", "0"]


@pytest.mark.parametrize("step", [1, -3])
def test_csv_flags_accept_every_nonzero_byte_across_chunks(step: int) -> None:
    flags = (np.arange(2 * _CSV_CHUNK_SIZE + 3) % 256).astype(np.uint8).view(bool)[::step]
    scores = np.zeros(len(flags))

    chunked = _render(_write_response_time_csv, _timing_report(scores, flags))

    assert chunked == _render(_reference_response_time_csv, scores, flags)


def test_response_time_csv_replays_archive_flags_stored_as_nonzero_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scores = np.array([0.5, 2.0, 3.0, 4.0])
    archive = tmp_path / "timing.npz"
    save_response_time_archive(archive, scores, scores <= 1.0, threshold=1.0)
    foreign = tmp_path / "foreign.npz"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(foreign, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "flags.npy":
                member = BytesIO()
                np.save(member, np.array([2, 0, 0, 0], np.uint8).view(bool))
                data = member.getvalue()
            target.writestr(info, data)

    assert main(["response-time-scores", str(foreign), "--format", "csv"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "respondent,response_time_score,response_time_flag",
        "0,0.5,1",
        "1,2.0,0",
        "2,3.0,0",
        "3,4.0,0",
    ]


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _per_cell_matrix(
    rows: list[list[str]], columns: list[int] | None, missing_values: tuple[str, ...] = ()
) -> np.ndarray:
    tokens = frozenset(missing_values)

    def parse(cell: str) -> float:
        if tokens:
            return _parse_numeric_cell_with_missing_values(cell, tokens)
        return _parse_numeric_cell(cell)

    return np.array(
        [
            [parse(row[column]) for column in (range(len(row)) if columns is None else columns)]
            for row in rows
        ],
        dtype=np.float64,
    )


_PADDED = [" 1", "2 ", "\xa03", "1e3", "-0", "inf", "-inf", "nan", "1_0", " 4"]


@pytest.mark.parametrize(
    ("header", "rows", "options", "columns", "missing_values"),
    [
        pytest.param(None, [_PADDED, _PADDED[::-1]], {}, None, (), id="padded-numbers"),
        pytest.param(
            None,
            [["1", "", "3"], ["", " ", "\t"], ["4", "5", "6"]],
            {},
            None,
            (),
            id="blank-cells",
        ),
        pytest.param(
            ["q1", "q2", "q3"],
            [["NA", "2", " N/A "], ["4", "", "6"], ["\x1c7", "8", "9"]],
            {},
            None,
            ("NA", "N/A"),
            id="text-missing-tokens",
        ),
        pytest.param(
            None,
            [["-99", "2", " -99 "], ["-99.0", "nan", "6"], ["7", "", "9"]],
            {"header_mode": "absent"},
            None,
            ("-99", "nan"),
            id="numeric-missing-tokens",
        ),
        pytest.param(
            ["id", "q1", "meta", "q2"],
            [["a", "1", "2024-01-01", " 3"], ["b", "", "note", "4"], ["c", "5", "x", "6"]],
            {"id_column": "id", "item_columns": ["q2", "q1"]},
            [3, 1],
            (),
            id="id-and-selected-columns",
        ),
        pytest.param(
            ["id", "q1", "q2"],
            [["a", "1", "2"], ["b", "", "4"]],
            {"id_column": "id"},
            [1, 2],
            (),
            id="id-column-only",
        ),
        pytest.param(
            ["q1", "meta"],
            [["1", "x"], [" ", "y"], ["3", "z"]],
            {"item_columns": ["q1"]},
            [0],
            (),
            id="single-selected-column",
        ),
    ],
)
def test_whole_row_parsing_matches_per_cell_parsing(
    tmp_path: Path,
    header: list[str] | None,
    rows: list[list[str]],
    options: dict[str, Any],
    columns: list[int] | None,
    missing_values: tuple[str, ...],
) -> None:
    source = tmp_path / "responses.csv"
    _write_rows(source, ([header] if header is not None else []) + rows)

    matrix, identifiers = _load_input(
        source, ",", missing_values=list(missing_values) or None, **options
    )

    expected = _per_cell_matrix(rows, columns, missing_values)
    assert matrix.shape == expected.shape
    assert np.array_equal(matrix, expected, equal_nan=True)
    np.testing.assert_array_equal(np.signbit(matrix), np.signbit(expected))
    if "id_column" in options:
        assert identifiers == [row[0] for row in rows]


@pytest.mark.parametrize(
    ("contents", "missing_values", "expected_calls"),
    [
        # Only the cells from a row's blank cell onward need per-cell rules.
        ("1,2,3\n4,,6\n 7,8 ,9\n", None, 2),
        # Rows resume per-cell at a blank or a text token; numeric rows stay whole.
        ("1,2,3\n4,,6\n7,NA,9\n 10,11 ,12\n", ["NA"], 4),
        # A numeric token must match exactly, so every row uses per-cell rules.
        ("1,2,3\n4,,6\n7,-99,9\n 10,11 ,12\n", ["-99"], 12),
    ],
)
def test_only_rows_needing_cell_rules_use_the_per_cell_parser(
    tmp_path: Path, contents: str, missing_values: list[str] | None, expected_calls: int
) -> None:
    source = tmp_path / "responses.csv"
    source.write_text(contents, encoding="utf-8")
    real_parser = (
        _parse_numeric_cell if missing_values is None else _parse_numeric_cell_with_missing_values
    )

    with patch(f"ier._cli_input.{real_parser.__name__}", side_effect=real_parser) as per_cell:
        _load_input(source, ",", header_mode="absent", missing_values=missing_values)

    assert per_cell.call_count == expected_calls


@pytest.mark.parametrize(("token", "missing_values"), [("", None), ("NA", ["NA"])])
def test_rows_with_missing_cells_convert_each_numeric_cell_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    missing_values: list[str] | None,
) -> None:
    rows = [
        [token, "2", "3", "4"],
        ["1", token, "3", "4"],
        ["1", "2", "3", token],
        ["1", "2", "3", "4"],
    ]
    source = tmp_path / "responses.csv"
    _write_rows(source, rows)
    converted: list[str] = []
    rejected: list[str] = []

    def recording_float(cell: str) -> float:
        try:
            value = float(cell)
        except ValueError:
            rejected.append(cell)
            raise
        converted.append(cell)
        return value

    monkeypatch.setattr("ier._cli_input.float", recording_float, raising=False)
    matrix, _ = _load_input(source, ",", header_mode="absent", missing_values=missing_values)

    # The whole-row attempt keeps its converted prefix, so no cell is parsed twice.
    # Setup tries each missing-value token once; each row then rejects one cell.
    assert converted == [cell for row in rows for cell in row if cell != token]
    assert rejected == [*(missing_values or []), token, token, token]
    expected = _per_cell_matrix(rows, None, tuple(missing_values or ()))
    assert np.array_equal(matrix, expected, equal_nan=True)


# Header (None for headerless input), loader options, and selected item positions.
_POSITION_LAYOUTS: dict[str, tuple[list[str] | None, dict[str, Any], list[int]]] = {
    "headerless": (None, {"header_mode": "absent"}, [0, 1, 2, 3, 4]),
    "detected-header": (["q1", "q2", "q3", "q4", "q5"], {}, [0, 1, 2, 3, 4]),
    "id-column": (["q1", "id", "q2", "q3"], {"id_column": "id"}, [0, 2, 3]),
    "id-and-selected-columns": (
        ["q1", "q2", "id", "q3", "meta", "q4"],
        {"id_column": "id", "item_columns": ["q4", "q1", "q3"]},
        [5, 0, 3],
    ),
    "single-selected-column": (["meta", "q1"], {"item_columns": ["q1"]}, [1]),
}
_SELECTED_POSITIONS = ["first", "middle", "last"]


def _position_rows(header: list[str] | None) -> list[list[str]]:
    """Three numeric rows with identifier and unselected metadata cells filled in."""
    width = 5 if header is None else len(header)
    rows = []
    for row in range(3):
        cells = [str((row + column) % 5 + 1) for column in range(width)]
        for column, name in enumerate(header or []):
            if name == "id":
                cells[column] = f"r{row}"
            elif name == "meta":
                cells[column] = "note"
        rows.append(cells)
    return rows


def _selected_index(columns: list[int], position: str) -> int:
    return {"first": 0, "middle": len(columns) // 2, "last": len(columns) - 1}[position]


@pytest.mark.parametrize("layout", list(_POSITION_LAYOUTS))
@pytest.mark.parametrize("positions", [["first"], ["middle"], ["last"], ["first", "last"]])
@pytest.mark.parametrize(
    ("cell", "missing_values"),
    [("", ()), (" ", ()), (" NA ", ("NA",)), ("-99", ("-99",))],
    ids=["blank", "whitespace", "text-token", "numeric-token"],
)
def test_missing_cells_load_at_every_selected_position(
    tmp_path: Path,
    layout: str,
    positions: list[str],
    cell: str,
    missing_values: tuple[str, ...],
) -> None:
    header, options, columns = _POSITION_LAYOUTS[layout]
    rows = _position_rows(header)
    indices = [_selected_index(columns, position) for position in positions]
    for index in indices:
        rows[1][columns[index]] = cell
    source = tmp_path / "responses.csv"
    _write_rows(source, ([header] if header is not None else []) + rows)

    matrix, identifiers = _load_input(
        source, ",", missing_values=list(missing_values) or None, **options
    )

    expected = _per_cell_matrix(rows, columns, missing_values)
    assert np.isnan(expected[1, indices]).all()
    assert np.count_nonzero(np.isnan(expected)) == len(set(indices))
    assert np.array_equal(matrix, expected, equal_nan=True)
    if "id_column" in options:
        assert identifiers == ["r0", "r1", "r2"]


@pytest.mark.parametrize("layout", list(_POSITION_LAYOUTS))
@pytest.mark.parametrize("position", _SELECTED_POSITIONS)
@pytest.mark.parametrize(
    ("preceding", "missing_values"),
    [(None, ()), ("", ()), ("NA", ("NA",)), ("-99", ("-99",))],
    ids=["numeric-prefix", "after-blank", "after-text-token", "after-numeric-token"],
)
def test_parse_errors_locate_every_selected_position(
    tmp_path: Path,
    layout: str,
    position: str,
    preceding: str | None,
    missing_values: tuple[str, ...],
) -> None:
    header, options, columns = _POSITION_LAYOUTS[layout]
    rows = _position_rows(header)
    index = _selected_index(columns, position)
    target = columns[index]
    rows[1][target] = "x"
    if preceding is not None and index:
        # A missing cell earlier in the row moves the rest onto per-cell rules.
        rows[1][columns[index // 2]] = preceding
    source = tmp_path / "responses.csv"
    _write_rows(source, ([header] if header is not None else []) + rows)

    with pytest.raises(ValueError) as raised:
        _load_input(source, ",", missing_values=list(missing_values) or None, **options)

    location = (
        f"column {target + 1}"
        if header is None
        else f"column {header[target]!r} (position {target + 1})"
    )
    assert str(raised.value) == (
        f"failed to parse numeric matrix from {source}: data row 2, {location}: "
        "could not convert string to float: 'x'"
    )


@pytest.mark.parametrize(
    ("contents", "options", "location", "token"),
    [
        pytest.param(
            "q1,q2,q3\n1,2,3\n4,x,6\n",
            {},
            "data row 2, column 'q2' (position 2)",
            "x",
            id="detected-header",
        ),
        pytest.param(
            "1,2,3\n4,x,6\n",
            {"header_mode": "present"},
            "data row 1, column '2' (position 2)",
            "x",
            id="explicit-numeric-header",
        ),
        pytest.param("1,2,3\n4,5,x\n", {}, "data row 2, column 3", "x", id="no-header"),
        pytest.param(
            "1,2,3\n4,5,x\n",
            {"header_mode": "absent"},
            "data row 2, column 3",
            "x",
            id="headerless-mode",
        ),
        pytest.param(
            "id,q1,meta,q2\na,1,2024-01-01,2\n",
            {"id_column": "id"},
            "data row 1, column 'meta' (position 3)",
            "2024-01-01",
            id="id-column",
        ),
        pytest.param(
            "id,q1,meta,q2\na,1,x,y\n",
            {"id_column": "id", "item_columns": ["q2", "q1"]},
            "data row 1, column 'q2' (position 4)",
            "y",
            id="selected-columns",
        ),
        pytest.param(
            "q1,q2,q3\n,1,x\n",
            {},
            "data row 1, column 'q3' (position 3)",
            "x",
            id="after-blank-cell",
        ),
        pytest.param(
            "1,2,3\n-99,5,x\n",
            {"missing_values": ["-99"]},
            "data row 2, column 3",
            "x",
            id="numeric-missing-token",
        ),
    ],
)
def test_parse_errors_name_the_data_row_and_column(
    tmp_path: Path, contents: str, options: dict[str, Any], location: str, token: str
) -> None:
    source = tmp_path / "responses.csv"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        _load_input(source, ",", **options)

    assert str(raised.value) == (
        f"failed to parse numeric matrix from {source}: {location}: "
        f"could not convert string to float: {token!r}"
    )


def test_command_reports_metadata_column_parse_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "meta.csv"
    source.write_text("q1,q2,date\n1,2,3\n4,5,2024-01-01\n", encoding="utf-8")

    assert main(["screen", str(source), "--indices", "irv"]) == 1

    assert capsys.readouterr().err == (
        f"error: failed to parse numeric matrix from {source}: data row 2, column 'date' "
        "(position 3): could not convert string to float: '2024-01-01'\n"
    )
