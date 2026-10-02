"""Survey exports retain their records and fail before overwriting results."""

from __future__ import annotations

import bz2
import csv
import gzip
import json
import lzma
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from ier._cli_input import _load_input
from ier.cli import main

if TYPE_CHECKING:
    from ier._cli_input import HeaderMode


@pytest.mark.parametrize("delimiter", [",", "\t", ";"])
@pytest.mark.parametrize("header", [True, False])
def test_jagged_autodetected_records_report_actual_width(
    tmp_path: Path, delimiter: str, header: bool
) -> None:
    lines = [["q1", "q2", "q3"]] if header else []
    lines += [["1", "2", "3"], ["4", "5"]]
    source = tmp_path / "survey.csv"
    source.write_text("\n".join(delimiter.join(row) for row in lines), encoding="utf-8")
    with pytest.raises(ValueError, match="data row 2 has 2 columns; expected 3"):
        _load_input(source, None)


@pytest.mark.parametrize("transport", ["plain", "gzip", "stdin"])
@pytest.mark.parametrize("command", ["screen", "composite", "response-time"])
@pytest.mark.parametrize("bad_record", ['"participant"trailing,1,2\n', '"participant,1,2\n'])
def test_malformed_quotes_fail_without_changing_existing_output(
    tmp_path: Path, transport: str, command: str, bad_record: str
) -> None:
    content = "Survey export\nid,q1,q2\n" + bad_record
    source = tmp_path / "survey.csv"
    if transport == "gzip":
        source = source.with_suffix(".csv.gz")
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(content)
    else:
        source.write_text(content, encoding="utf-8")
    output = tmp_path / "result.json"
    output.write_text("previous verified result", encoding="utf-8")
    stderr = StringIO()
    options = ["--indices", "irv"] if command != "response-time" else []
    stdin = StringIO(content)
    with patch("sys.stdin", stdin), patch("sys.stderr", stderr):
        code = main(
            [
                command,
                "-" if transport == "stdin" else str(source),
                "--skip-rows",
                "1",
                "--id-column",
                "id",
                *options,
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
    assert code == 1
    assert "malformed delimited input at physical line 3" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
    assert output.read_text(encoding="utf-8") == "previous verified result"
    assert not stdin.closed


def test_large_field_errors_are_reported_through_cli(tmp_path: Path) -> None:
    source = tmp_path / "survey.csv"
    source.write_text("id,q1,notes\na,1," + "x" * (csv.field_size_limit() + 1), encoding="utf-8")
    stderr = StringIO()
    with patch("sys.stderr", stderr):
        code = main(["screen", str(source), "--item-columns", "q1", "--indices", "irv"])
    assert code == 1
    assert "field larger than field limit" in stderr.getvalue()
    assert "physical line 2" in stderr.getvalue()


@pytest.mark.parametrize("compression", ["gzip", "bzip2", "xz"])
@pytest.mark.parametrize("damage", ["truncated", "invalid"])
@pytest.mark.parametrize("command", ["screen", "composite", "response-time"])
def test_damaged_compressed_exports_fail_without_changing_existing_output(
    tmp_path: Path, compression: str, damage: str, command: str
) -> None:
    content = b"id,q1,q2\nfirst,1,2\nsecond,3,4\n"
    compressors = {"gzip": gzip.compress, "bzip2": bz2.compress, "xz": lzma.compress}
    suffixes = {"gzip": ".gz", "bzip2": ".bz2", "xz": ".xz"}
    payload = (
        compressors[compression](content)[:-8]
        if damage == "truncated"
        else b"not a compressed export"
    )
    source = tmp_path / f"survey.csv{suffixes[compression]}"
    source.write_bytes(payload)
    output = tmp_path / "result.json"
    output.write_text("previous verified result", encoding="utf-8")
    stderr = StringIO()
    options = ["--indices", "irv"] if command != "response-time" else []
    with patch("sys.stderr", stderr):
        code = main(
            [
                command,
                str(source),
                "--id-column",
                "id",
                *options,
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
    assert code == 1
    assert f"failed to read compressed text from {source}:" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
    assert output.read_text(encoding="utf-8") == "previous verified result"


def test_corrupt_gzip_deflate_payload_is_reported_through_cli(tmp_path: Path) -> None:
    payload = bytearray(gzip.compress(b"q1,q2\n1,2\n3,4\n"))
    # A reserved deflate block type fails inside zlib after a valid gzip header.
    payload[10] = 0b110
    source = tmp_path / "survey.csv.gz"
    source.write_bytes(payload)
    stderr = StringIO()
    with patch("sys.stderr", stderr):
        code = main(["screen", str(source), "--indices", "irv"])
    assert code == 1
    assert f"failed to read compressed text from {source}:" in stderr.getvalue()
    assert "invalid block type" in stderr.getvalue()


@pytest.mark.parametrize("delimiter", [",", "\t", ";"])
def test_valid_quoted_metadata_and_multiline_identifiers_preserve_alignment(
    tmp_path: Path, delimiter: str
) -> None:
    source = tmp_path / "survey.csv"
    identifiers = ['respondent,one;\t"quoted"', "respondent\ntwo"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["id", "q1", "notes", "q2"])
        writer.writerow([identifiers[0], 1, "metadata,;\t", 1])
        writer.writerow([identifiers[1], 1, "metadata\nsecond line", 5])
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "screen",
                str(source),
                "--id-column",
                "id",
                "--item-columns",
                "q1,q2",
                "--indices",
                "irv",
                "--threshold",
                "irv=0.25",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["respondent_ids"] == identifiers
    assert result["scores"]["irv"] == [0.0, 2.0]
    assert result["flags"]["irv"] == [True, False]


def test_sniffer_fallback_respects_quoted_delimiters(tmp_path: Path) -> None:
    source = tmp_path / "survey.csv"
    source.write_text('"q1;with,punctuation",q2\n1,2\n3,4\n', encoding="utf-8")
    with patch("ier._cli_input.csv.Sniffer.sniff", side_effect=csv.Error("ambiguous")):
        matrix, identifiers = _load_input(source, None)
    np.testing.assert_array_equal(matrix, [[1, 2], [3, 4]])
    assert identifiers is None


def test_numeric_fields_keep_whitespace_blanks_and_nonfinite_values(tmp_path: Path) -> None:
    source = tmp_path / "survey.csv"
    source.write_text(" 1 ,  ,\n+2.5,NaN,inf\n", encoding="utf-8")
    matrix, _ = _load_input(source, ",", header_mode="absent")
    np.testing.assert_array_equal(matrix, [[1, np.nan, np.nan], [2.5, np.nan, np.inf]])


@pytest.mark.parametrize("header_mode", ["auto", "absent"])
@pytest.mark.parametrize("transport", ["plain", "gzip", "bzip2", "xz", "stdin"])
def test_mixed_whitespace_numeric_matrices_keep_all_items(
    tmp_path: Path, header_mode: HeaderMode, transport: str
) -> None:
    content = "  \n+1\t2 3e0\n4\t5 6\n"
    source = tmp_path / "responses.txt"
    if transport == "stdin":
        with patch("sys.stdin", StringIO(content)):
            matrix, identifiers = _load_input(Path("-"), None, header_mode=header_mode)
    else:
        compressors = {"gzip": gzip.compress, "bzip2": bz2.compress, "xz": lzma.compress}
        suffixes = {"gzip": ".gz", "bzip2": ".bz2", "xz": ".xz"}
        if transport == "plain":
            source.write_text(content, encoding="utf-8")
        else:
            source = source.with_suffix(f".txt{suffixes[transport]}")
            source.write_bytes(compressors[transport](content.encode("utf-8")))
        matrix, identifiers = _load_input(source, None, header_mode=header_mode)
    np.testing.assert_array_equal(matrix, [[1, 2, 3], [4, 5, 6]])
    assert identifiers is None


def test_mixed_whitespace_standard_input_scores_every_item() -> None:
    stdout = StringIO()
    with patch("sys.stdin", StringIO("1\t2 3\n4\t5 6\n")), patch("sys.stdout", stdout):
        code = main(["screen", "-", "--indices", "irv", "--format", "json"])
    assert code == 0
    result = json.loads(stdout.getvalue())
    assert result["n_respondents"] == 2
    assert result["scores"]["irv"] == pytest.approx([np.sqrt(2 / 3), np.sqrt(2 / 3)])


def test_explicit_tab_delimiter_keeps_spaces_inside_fields(tmp_path: Path) -> None:
    source = tmp_path / "responses.txt"
    source.write_text("1\t2 3\n4\t5 6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could not convert string to float: '2 3'"):
        _load_input(source, "\t", header_mode="absent")


def test_named_tsv_metadata_keeps_numeric_space_separated_text(tmp_path: Path) -> None:
    source = tmp_path / "responses.tsv"
    source.write_text("q1\tnotes with spaces\tq2\n1\t2 3\t4\n5\t6 7\t8\n", encoding="utf-8")
    matrix, _ = _load_input(source, None, item_columns=["q1", "q2"])
    np.testing.assert_array_equal(matrix, [[1, 4], [5, 8]])


@pytest.mark.parametrize("selection", ["header", "header-and-items", "items", "id"])
@pytest.mark.parametrize("force_fallback", [False, True])
def test_numeric_tsv_header_names_keep_embedded_spaces(
    tmp_path: Path, selection: str, force_fallback: bool
) -> None:
    source = tmp_path / "responses.tsv"
    source.write_text("1\t2 3\t4\n5\t9\t8\n6\t10\t7\n", encoding="utf-8")
    header_mode: HeaderMode = "present" if selection in {"header", "header-and-items"} else "auto"
    item_columns = ["1", "4"] if selection in {"items", "header-and-items"} else None
    id_column = "2 3" if selection == "id" else None
    sniffer = (
        patch("ier._cli_input.csv.Sniffer.sniff", side_effect=csv.Error("ambiguous"))
        if force_fallback
        else nullcontext()
    )
    with sniffer:
        matrix, identifiers = _load_input(
            source,
            None,
            header_mode=header_mode,
            item_columns=item_columns,
            id_column=id_column,
        )
    expected = [[5, 9, 8], [6, 10, 7]] if selection == "header" else [[5, 8], [6, 7]]
    np.testing.assert_array_equal(matrix, expected)
    assert identifiers == (["9", "10"] if selection == "id" else None)


def test_numeric_tsv_identifier_header_scores_through_standard_input() -> None:
    stdout = StringIO()
    content = "1\t2 3\t4\n5\t9\t8\n6\t10\t7\n"
    with patch("sys.stdin", StringIO(content)), patch("sys.stdout", stdout):
        code = main(["screen", "-", "--id-column", "2 3", "--indices", "irv", "--format", "json"])
    assert code == 0
    result = json.loads(stdout.getvalue())
    assert result["respondent_ids"] == ["9", "10"]
    assert result["scores"]["irv"] == [1.5, 0.5]


def test_autodetected_tsv_keeps_blank_fields_as_missing_values(tmp_path: Path) -> None:
    source = tmp_path / "responses.tsv"
    source.write_text("1\t \t3\n4\t5\t6\n", encoding="utf-8")
    matrix, _ = _load_input(source, None)
    np.testing.assert_array_equal(matrix, [[1, np.nan, 3], [4, 5, 6]])


@pytest.mark.parametrize("delimiter", [",", "\t", ";"])
def test_fully_missing_respondents_keep_their_rows(tmp_path: Path, delimiter: str) -> None:
    source = tmp_path / "survey.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["q1", "q2"])
        writer.writerow([1, 2])
        writer.writerow(["", ""])
        writer.writerow([" ", " "])
        writer.writerow([3, 4])
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "screen",
                str(source),
                "--indices",
                "missing_rate",
                "--threshold",
                "missing_rate=0.5",
                "--min-flags",
                "1",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["n_respondents"] == 4
    assert result["scores"]["missing_rate"] == [0.0, 1.0, 1.0, 0.0]
    assert result["consensus_flags"] == [False, True, True, False]


@pytest.mark.parametrize("delimiter", [None, ","])
def test_single_column_quoted_empty_records_are_respondents(
    tmp_path: Path, delimiter: str | None
) -> None:
    source = tmp_path / "survey.csv"
    source.write_text('q1\n1\n\n  \n""\n"   "\n2\n', encoding="utf-8")
    matrix, _ = _load_input(source, delimiter)
    np.testing.assert_array_equal(matrix, [[1], [np.nan], [np.nan], [2]])


def test_quoted_single_column_numeric_input_keeps_first_respondent(tmp_path: Path) -> None:
    source = tmp_path / "survey.csv"
    source.write_text('"1"\n"2"\n"3"\n', encoding="utf-8")
    matrix, _ = _load_input(source, None)
    np.testing.assert_array_equal(matrix, [[1], [2], [3]])


def test_delimiter_fallback_prefers_data_over_header_punctuation(tmp_path: Path) -> None:
    source = tmp_path / "survey.csv"
    source.write_text("q1;q2;q3,q4\n1,2\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data row 2 has 1 columns; expected 2"):
        _load_input(source, None)


def test_delimiter_fallback_skips_only_physically_blank_records(tmp_path: Path) -> None:
    source = tmp_path / "survey.csv"
    source.write_text("  \n\nq1,q2\n,\n\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data row 2 has 1 columns; expected 2"):
        _load_input(source, None)


@pytest.mark.parametrize(("dtype", "answer"), [("int64", 2**60), ("uint64", 2**64 - 2)])
def test_cli_attention_checks_preserve_exact_integer_answers(
    tmp_path: Path, dtype: str, answer: int
) -> None:
    source = tmp_path / "survey.npy"
    np.save(source, np.asarray([[answer - 1], [answer], [answer + 1]], dtype=dtype))
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "screen",
                str(source),
                "--indices",
                "infrequency",
                "--infrequency-item-indices",
                "0",
                "--infrequency-expected-responses",
                str(answer),
                "--threshold",
                "infrequency=1",
                "--min-flags",
                "1",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["scores"]["infrequency"] == [1.0, 0.0, 1.0]
    assert result["consensus_flags"] == [True, False, True]
