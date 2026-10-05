"""`ier inspect` reports parsing decisions, missing cells, and the observed scale."""

from __future__ import annotations

import csv
import gzip
import json
from io import StringIO
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from ier._cli_input import _InputReport, _load_input
from ier.cli import _build_parser, main

if TYPE_CHECKING:
    from pathlib import Path


def _inspect(argv: list[str], stdin: str | None = None) -> tuple[int, str, str]:
    """Run ``ier inspect`` and return its exit status, standard output, and errors."""
    stdout = StringIO()
    stderr = StringIO()
    with (
        patch("sys.stdin", StringIO(stdin or "")),
        patch("sys.stdout", stdout),
        patch("sys.stderr", stderr),
    ):
        code = main(["inspect", *argv])
    return code, stdout.getvalue(), stderr.getvalue()


def _inspect_json(argv: list[str], stdin: str | None = None) -> dict[str, Any]:
    code, output, errors = _inspect([*argv, "--format", "json"], stdin)
    assert code == 0, errors
    payload: dict[str, Any] = json.loads(output)
    return payload


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_sniffed_header_and_values_are_reported(tmp_path: Path) -> None:
    source = _write(tmp_path / "survey.csv", "q1,q2,q3\n1,2,3\n5,4,\n2,2,2\n")

    payload = _inspect_json([str(source)])

    assert payload == {
        "source": str(source),
        "input_format": "delimited",
        "delimiter": ",",
        "delimiter_detection": "sniffer",
        "header": "auto-detected",
        "id_column": None,
        "n_columns": 3,
        "n_respondents": 3,
        "n_items": 3,
        "item_names": ["q1", "q2", "q3"],
        "item_positions": [1, 2, 3],
        "missing_cells": 1,
        "missing_by_item": [0, 0, 1],
        "infinite_cells": 0,
        "observed_min": 1.0,
        "observed_max": 5.0,
        "distinct_values": 5,
        "non_integer_values": False,
        "respondents_at_min": 1,
        "respondents_at_max": 1,
        "suggested_options": ["--scale-min=1", "--scale-max=5"],
        "warnings": [],
    }


def test_text_report_summarizes_selection_and_scale(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "export.csv",
        "ResponseId,StartDate,Q1,Q2,Q3\nr1,today,1,NA,3\nr2,today,5,4,NA\nr3,today,2,NA,1\n",
    )

    code, output, errors = _inspect(
        [
            str(source),
            "--id-column",
            "ResponseId",
            "--item-pattern",
            "Q*",
            "--missing-value",
            "NA",
        ]
    )

    assert code == 0, errors
    assert output.splitlines() == [
        f"source: {source}",
        "delimiter: ',' (sniffer)",
        "header: present",
        "id column: ResponseId",
        "columns: 5 (3 items, 1 ID, 1 unselected)",
        "respondents: 3",
        "items: 3 (Q1, Q2, Q3)",
        "missing cells: 3 of 9 (33.3%)",
        "missing by item: Q2=2, Q3=1",
        "observed values: 1 to 5 (5 distinct)",
        "non-integer values: no",
        "respondents using the minimum: 2 (66.7%)",
        "respondents using the maximum: 1 (33.3%)",
        "suggested options: --scale-min=1 --scale-max=5",
    ]


def test_fallback_detection_is_reported(tmp_path: Path) -> None:
    source = _write(tmp_path / "survey.csv", '"q1;with,punctuation",q2\n1,2\n3,4\n')

    with patch("ier._cli_input.csv.Sniffer.sniff", side_effect=csv.Error("ambiguous")):
        payload = _inspect_json([str(source)])

    assert payload["delimiter"] == ","
    assert payload["delimiter_detection"] == "fallback"
    assert payload["item_names"] == ["q1;with,punctuation", "q2"]


def test_quoted_single_column_uses_fallback_without_patching(tmp_path: Path) -> None:
    source = _write(tmp_path / "single.csv", '"1"\n"2"\n"3"\n')

    payload = _inspect_json([str(source)])

    assert payload["delimiter_detection"] == "fallback"
    assert payload["header"] == "absent"
    assert payload["item_names"] is None
    assert payload["n_items"] == 1


@pytest.mark.parametrize("content", ["1 2 3\n4 5 6\n", "1\t2 3\n4\t5 6\n"])
def test_whitespace_matrices_report_no_delimiter(tmp_path: Path, content: str) -> None:
    source = _write(tmp_path / "matrix.txt", content)

    payload = _inspect_json([str(source)])
    code, output, _ = _inspect([str(source)])

    assert payload["delimiter"] is None
    assert payload["delimiter_detection"] == "whitespace"
    assert payload["header"] == "absent"
    assert payload["item_positions"] == [1, 2, 3]
    assert code == 0
    assert "delimiter: whitespace\n" in output
    assert "items: 3 (unnamed columns 1 to 3)\n" in output


def test_explicit_delimiter_and_header_modes(tmp_path: Path) -> None:
    source = _write(tmp_path / "numeric.csv", "1;2\n3;4\n5;6\n")

    absent = _inspect_json([str(source), "--delimiter", ";"])
    present = _inspect_json([str(source), "--delimiter", ";", "--header", "present"])

    assert absent["delimiter_detection"] == "explicit"
    assert absent["header"] == "absent"
    assert absent["n_respondents"] == 3
    assert present["header"] == "present"
    assert present["item_names"] == ["1", "2"]
    assert present["n_respondents"] == 2


def test_skip_rows_and_exclusions_apply_before_reporting(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "survey.tsv",
        "Survey title\nid\tq1\tmeta\tq2\nA\t1\tx\t2\nB\t3\ty\t4\n",
    )

    payload = _inspect_json(
        [str(source), "--skip-rows", "1", "--id-column", "id", "--exclude-column", "meta"]
    )

    assert payload["delimiter"] == "\t"
    assert payload["id_column"] == "id"
    assert payload["n_columns"] == 4
    assert payload["item_names"] == ["q1", "q2"]
    assert payload["item_positions"] == [2, 4]


def test_long_item_lists_are_truncated_only_in_text(tmp_path: Path) -> None:
    names = [f"item_{index}" for index in range(14)]
    rows = [",".join(names), ",".join("1" for _ in names), ",".join("2" for _ in names)]
    source = _write(tmp_path / "wide.csv", "\n".join(rows) + "\n")

    _, output, _ = _inspect([str(source)])
    payload = _inspect_json([str(source)])

    preview = ", ".join(names[:10])
    assert f"items: 14 ({preview}, ... (+4 more))" in output
    assert payload["item_names"] == names


def test_missing_items_are_ranked_and_truncated(tmp_path: Path) -> None:
    names = [f"q{index}" for index in range(12)]
    first = ["" if index else "1" for index in range(12)]
    second = ["" if index in {5, 11} else "2" for index in range(12)]
    rows = [",".join(names), ",".join(first), ",".join(second)]
    source = _write(tmp_path / "sparse.csv", "\n".join(rows) + "\n")

    _, output, _ = _inspect([str(source)])

    ranked = ["q5=2", "q11=2", *(f"q{index}=1" for index in (1, 2, 3, 4, 6, 7, 8, 9))]
    assert f"missing by item: {', '.join(ranked)}, ... (+1 more)\n" in output
    assert "missing cells: 13 of 24 (54.2%)\n" in output


def test_non_integer_values_are_flagged(tmp_path: Path) -> None:
    source = _write(tmp_path / "means.csv", "1.5,2\n3,4.25\n")

    payload = _inspect_json([str(source)])

    assert payload["non_integer_values"] is True
    assert payload["suggested_options"] == ["--scale-min=1.5", "--scale-max=4.25"]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ("-0.00001,1,2\n0,2,1\n1,0,2\n2,1,0\n", ["--scale-min=-1e-05", "--scale-max=2"]),
        ("-1e20,1\n2,3\n", ["--scale-min=-1e+20", "--scale-max=3"]),
    ],
)
def test_suggested_options_can_be_passed_back_to_scoring_commands(
    tmp_path: Path, rows: str, expected: list[str]
) -> None:
    # Python 3.11 and 3.12 argparse read a separate '-1e-05' token as an option.
    source = _write(tmp_path / "small.csv", rows)

    suggested = _inspect_json([str(source)])["suggested_options"]
    _, output, _ = _inspect([str(source)])
    printed = next(line for line in output.splitlines() if line.startswith("suggested options:"))
    args = _build_parser().parse_args(["screen", str(source), *suggested])

    assert suggested == expected
    assert printed.split()[2:] == expected
    assert args.scale_min == float(expected[0].partition("=")[2])
    assert args.scale_max == float(expected[1].partition("=")[2])
    with patch("sys.stdout", StringIO()), patch("sys.stderr", StringIO()):
        assert main(["screen", str(source), "--indices", "irv", *suggested]) == 0


def test_rarely_used_extremes_are_warned(tmp_path: Path) -> None:
    rows = ["3,3,3"] * 199 + ["1,7,3"]
    rows[:150] = ["1,2,3"] * 150
    source = _write(tmp_path / "typo.csv", "\n".join(rows) + "\n")

    payload = _inspect_json([str(source)])
    _, output, _ = _inspect([str(source)])

    message = (
        "observed maximum 7 is used by only 1 of 200 respondents (0.5%); confirm the "
        "response scale and pass --scale-max explicitly if it differs"
    )
    assert payload["respondents_at_min"] == 151
    assert payload["respondents_at_max"] == 1
    assert payload["warnings"] == [message]
    assert output.endswith(f"warnings:\n  {message}\n")


def test_rare_minimum_is_warned_but_small_samples_are_not(tmp_path: Path) -> None:
    large = _write(tmp_path / "large.csv", "\n".join(["5,4"] * 120 + ["0,4"]) + "\n")
    small = _write(tmp_path / "small.csv", "\n".join(["5,4"] * 50 + ["0,4"]) + "\n")

    warnings = _inspect_json([str(large)])["warnings"]

    assert len(warnings) == 1
    assert warnings[0].startswith("observed minimum 0 is used by only 1 of 121 respondents")
    assert "--scale-min explicitly" in warnings[0]
    assert _inspect_json([str(small)])["warnings"] == []


def test_single_value_missing_and_infinite_matrices_warn(tmp_path: Path) -> None:
    constant = _write(tmp_path / "constant.csv", "3,3\n3,\n")
    empty = _write(tmp_path / "empty.csv", "q1,q2\n,\n,\n")
    infinite = _write(tmp_path / "infinite.csv", "1,inf\n-inf,2\n")

    constant_payload = _inspect_json([str(constant)])
    empty_payload = _inspect_json([str(empty)])
    infinite_payload = _inspect_json([str(infinite)])
    _, empty_text, _ = _inspect([str(empty)])

    assert constant_payload["warnings"] == [
        "only one distinct value (3) was observed; pass --scale-min and --scale-max explicitly"
    ]
    assert empty_payload["observed_min"] is None
    assert empty_payload["distinct_values"] == 0
    assert empty_payload["suggested_options"] == []
    assert empty_payload["warnings"] == [
        "no finite responses were observed; scale bounds cannot be inferred"
    ]
    assert "observed values: none\n" in empty_text
    assert "missing cells: 4 of 4 (100.0%)\n" in empty_text
    assert infinite_payload["infinite_cells"] == 2
    assert infinite_payload["observed_min"] == 1.0
    assert infinite_payload["observed_max"] == 2.0
    assert infinite_payload["warnings"][0] == (
        "2 infinite cells are excluded from the observed values; check the input for "
        "invalid entries"
    )


@pytest.mark.parametrize("dtype", [np.int64, np.uint8, np.float32])
def test_npy_matrices_report_exact_values(tmp_path: Path, dtype: type[np.generic]) -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    if dtype is np.int64:
        values[0, 0] = 2**60 + 1
    source = tmp_path / "matrix.npy"
    np.save(source, values)

    payload = _inspect_json([str(source)])
    _, output, _ = _inspect([str(source)])

    assert payload["input_format"] == "npy"
    assert payload["delimiter"] is None
    assert payload["delimiter_detection"] is None
    assert payload["header"] == "absent"
    assert payload["n_columns"] == 3
    assert payload["observed_min"] == values.min().item()
    assert payload["observed_max"] == values.max().item()
    assert payload["missing_by_item"] == [0, 0, 0]
    assert "format: NumPy .npy matrix\n" in output


def test_npy_float_matrices_count_missing_cells(tmp_path: Path) -> None:
    source = tmp_path / "matrix.npy"
    np.save(source, np.array([[1.0, np.nan], [np.nan, np.nan], [2.0, 3.0]]))

    payload = _inspect_json([str(source)])

    assert payload["missing_by_item"] == [1, 2]
    assert payload["missing_cells"] == 3


def test_blocks_cover_every_respondent(tmp_path: Path) -> None:
    rows = ["2,3"] * 7 + ["1,3", "2,9"]
    source = _write(tmp_path / "blocks.csv", "\n".join(rows) + "\n")

    with patch("ier._cli_inspect._BLOCK_ROWS", 2):
        payload = _inspect_json([str(source)])

    assert payload["observed_min"] == 1.0
    assert payload["observed_max"] == 9.0
    assert payload["distinct_values"] == 4
    assert payload["respondents_at_min"] == 1
    assert payload["respondents_at_max"] == 1


def test_standard_input_and_atomic_compressed_output(tmp_path: Path) -> None:
    destination = tmp_path / "inspection.json.gz"
    destination.write_bytes(b"previous")

    payload = _inspect_json(["-"], stdin="a,b\n1,2\n")
    code, output, errors = _inspect(
        ["-", "--format", "json", "--output", str(destination)], stdin="a,b\n1,2\n"
    )

    assert payload["source"] == "standard input"
    assert code == 0, errors
    assert output == ""
    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == payload


def test_text_output_file_is_replaced(tmp_path: Path) -> None:
    source = _write(tmp_path / "survey.csv", "1,2\n3,4\n")
    destination = _write(tmp_path / "inspection.txt", "previous")

    code, output, _ = _inspect([str(source), "--output", str(destination)])

    assert code == 0
    assert output == ""
    assert destination.read_text(encoding="utf-8").startswith(f"source: {source}\n")


def test_input_errors_are_reported_without_traceback(tmp_path: Path) -> None:
    source = _write(tmp_path / "jagged.csv", "q1,q2\n1,2\n3\n")

    code, output, errors = _inspect([str(source)])

    assert code == 1
    assert output == ""
    assert errors.startswith("error: jagged delimited input")


def test_report_does_not_change_loaded_values(tmp_path: Path) -> None:
    source = _write(tmp_path / "survey.csv", "id,q1,q2\na,1,NA\nb,3,4\n")
    report = _InputReport()

    reported = _load_input(source, None, "id", missing_values=["NA"], report=report)
    plain = _load_input(source, None, "id", missing_values=["NA"])

    np.testing.assert_array_equal(reported[0], plain[0])
    assert reported[1] == plain[1] == ["a", "b"]
    assert report.column_names == ["id", "q1", "q2"]
    assert report.id_index == 0
    assert report.item_indices == [1, 2]
