"""Exercise exact survey names and respondent-specific planned omissions."""

from __future__ import annotations

import bz2
import csv
import gzip
import json
import lzma
import re
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from ier import irv, load_score_archive, longstring_scores
from ier._cli_input import _load_applicable_mask, _load_input
from ier.cli import main

if TYPE_CHECKING:
    from pathlib import Path

_COMPRESSORS = {"gzip": gzip.compress, "bzip2": bz2.compress, "xz": lzma.compress}
_SUFFIXES = {"gzip": ".gz", "bzip2": ".bz2", "xz": ".xz"}


def _survey_fixture(root: Path) -> tuple[Path, np.ndarray]:
    source = root / "responses.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "q1", "Question, skipped", "q2", "notes"])
        writer.writerows(
            [
                ["a", 1, "", "", "unselected text"],
                ["b", "", "", "", "unselected text"],
                ["c", 3, 3, 3, "unselected text"],
                ["d", "", "", "", "unselected text"],
            ]
        )
    # Columns deliberately match the requested q2, q1, Question order.
    mask = np.array([[False, True, False], [True, False, True], [True, True, False], [False] * 3])
    return source, mask


def _write_mask(root: Path, transport: str, mask: np.ndarray) -> Path:
    source = root / "applicable.npy"
    if transport == "npy":
        np.save(source, mask)
        return source
    source = root / "applicable.csv"
    content = "\n".join(",".join("1" if cell else "0" for cell in row) for row in mask) + "\n"
    if transport == "csv":
        source.write_text(content, encoding="utf-8")
    else:
        source = source.with_suffix(f".csv{_SUFFIXES[transport]}")
        source.write_bytes(_COMPRESSORS[transport](content.encode("utf-8")))
    return source


def _selection() -> list[str]:
    return ["--id-column", "id", "--item-columns", "q2,q1", "--item-column", "Question, skipped"]


@pytest.mark.parametrize("command", ["screen", "composite"])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("transport", ["npy", "csv", "gzip", "bzip2", "xz"])
def test_planned_omissions_follow_selected_axes_and_worker_settings(
    tmp_path: Path, command: str, workers: int, transport: str
) -> None:
    source, mask = _survey_fixture(tmp_path)
    mask_path = _write_mask(tmp_path, transport, mask)
    stdout = StringIO()
    decisions = (
        ["--threshold", "missing_rate=0.6", "--threshold", "irv=-1", "--min-flags", "1"]
        if command == "screen"
        else ["--no-standardize", "--include-components", "--threshold", "0.75"]
    )
    with patch("sys.stdout", stdout):
        code = main(
            [
                command,
                str(source),
                *_selection(),
                "--missing-applicable-mask",
                str(mask_path),
                "--indices",
                "missing_rate",
                "irv",
                "--workers",
                str(workers),
                *decisions,
                "--format",
                "json",
            ]
        )
    assert code == 0
    result = json.loads(stdout.getvalue())
    assert result["respondent_ids"] == ["a", "b", "c", "d"]
    scores = result["scores"] if command == "screen" else result["component_scores"]
    assert scores["missing_rate"] == [0.0, 1.0, 0.0, None]
    if command == "screen":
        assert result["flags"]["missing_rate"] == [False, True, False, False]
        assert result["consensus_flags"] == [False, True, False, False]
    else:
        assert result["scores"] == [0.0, 1.0, 0.0, None]
        assert result["flags"] == [False, True, False, False]


def test_boolean_npy_masks_remain_readonly_memory_maps(tmp_path: Path) -> None:
    path = tmp_path / "applicable.npy"
    expected = np.array([[True, False], [False, True]])
    np.save(path, expected)
    mask = _load_applicable_mask(path, (2, 2))
    assert isinstance(mask, np.memmap)
    assert not mask.flags.writeable
    np.testing.assert_array_equal(mask, expected)
    with pytest.raises(ValueError, match="read-only"):
        mask[0, 0] = False
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)


def test_mask_and_missing_item_subset_exclude_different_cells(tmp_path: Path) -> None:
    source = tmp_path / "responses.csv"
    source.write_text("q1,q2,q3\n,,1\n1,,\n", encoding="utf-8")
    mask = tmp_path / "applicable.csv"
    mask.write_text("1,0,1\n1,1,1\n", encoding="utf-8")
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            [
                "screen",
                str(source),
                "--indices",
                "missing_rate",
                "--missing-applicable-mask",
                str(mask),
                "--missing-item-indices",
                "1,2",
                "--format",
                "json",
            ]
        )
    assert code == 0
    assert json.loads(stdout.getvalue())["scores"]["missing_rate"] == [0.0, 1.0]


@pytest.mark.parametrize("command", ["screen", "composite", "response-time"])
def test_exact_and_shorthand_columns_preserve_interleaved_order(
    tmp_path: Path, command: str
) -> None:
    source = tmp_path / "questions.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "Question, first", "q2", "q3", 'Question "four", last'])
        writer.writerows([["a", 1, 1, 5, 5], ["b", 2, 2, 4, 4]])
    stdout = StringIO()
    options = ["--metric", "median"] if command == "response-time" else ["--indices", "longstring"]
    if command == "composite":
        options += ["--no-standardize", "--include-components"]
    with patch("sys.stdout", stdout):
        code = main(
            [
                command,
                str(source),
                "--id-column",
                "id",
                "--item-column",
                'Question "four", last',
                "--item-columns",
                "q2,q3",
                "--item-column",
                "Question, first",
                *options,
                "--format",
                "json",
            ]
        )
    assert code == 0
    result = json.loads(stdout.getvalue())
    assert result["respondent_ids"] == ["a", "b"]
    if command == "screen":
        # Physical order has length-two runs; the selected alternating order has none.
        assert result["scores"]["longstring"] == [1.0, 1.0]
    elif command == "composite":
        assert result["component_scores"]["longstring"] == [1.0, 1.0]
        assert result["scores"] == [1.0, 1.0]
    else:
        assert result["scores"] == [3.0, 3.0]


@pytest.mark.parametrize(
    "selection",
    [["--item-column", " "], ["--item-column", "q1", "--item-columns", "q1"]],
)
def test_invalid_exact_column_selections_preserve_previous_output(
    tmp_path: Path, selection: list[str]
) -> None:
    source = tmp_path / "responses.csv"
    source.write_text("q1,q2\n1,2\n3,4\n", encoding="utf-8")
    output = tmp_path / "result.json"
    output.write_text("previous verified result", encoding="utf-8")
    stderr = StringIO()
    with patch("sys.stderr", stderr):
        code = main(["screen", str(source), *selection, "--output", str(output)])
    assert code == 1
    assert "column" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
    assert output.read_text(encoding="utf-8") == "previous verified result"


@pytest.mark.parametrize("command", ["screen", "composite"])
def test_masked_scores_can_be_saved_and_reused(tmp_path: Path, command: str) -> None:
    source, mask = _survey_fixture(tmp_path)
    mask_path = _write_mask(tmp_path, "npy", mask)
    archive = tmp_path / "scores.npz"
    options = [] if command == "screen" else ["--include-components", "--no-standardize"]
    assert (
        main(
            [
                command,
                str(source),
                *_selection(),
                "--indices",
                "missing_rate",
                "irv",
                "--missing-applicable-mask",
                str(mask_path),
                *options,
                "--format",
                "npz",
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    saved = load_score_archive(archive)
    np.testing.assert_array_equal(saved["scores"]["missing_rate"], [0, 1, 0, np.nan])
    stdout = StringIO()
    replay_options = [] if command == "screen" else ["--no-standardize", "--include-components"]
    with patch("sys.stdout", stdout):
        code = main([f"{command}-scores", str(archive), *replay_options, "--format", "json"])
    assert code == 0
    result = json.loads(stdout.getvalue())
    assert result["respondent_ids"] == ["a", "b", "c", "d"]
    scores = result["scores"] if command == "screen" else result["component_scores"]
    assert scores["missing_rate"] == [0, 1, 0, None]


@pytest.mark.parametrize("stdin_source", ["responses", "mask"])
def test_one_matrix_can_come_from_forward_only_standard_input(
    tmp_path: Path, stdin_source: str
) -> None:
    source = tmp_path / "responses.csv"
    source.write_text("q1,q2\n1,\n2,3\n", encoding="utf-8")
    mask_path = tmp_path / "mask.csv"
    mask_path.write_text("1,0\n1,1\n", encoding="utf-8")
    content = source.read_text() if stdin_source == "responses" else mask_path.read_text()

    class ForwardOnlyInput(StringIO):
        def seek(self, *args: object, **kwargs: object) -> int:
            raise AssertionError("input must not be rewound")

    stdin = ForwardOnlyInput(content)
    stdout = StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        code = main(
            [
                "screen",
                "-" if stdin_source == "responses" else str(source),
                "--missing-applicable-mask",
                "-" if stdin_source == "mask" else str(mask_path),
                "--indices",
                "missing_rate",
                "--format",
                "json",
            ]
        )
    assert code == 0
    assert not stdin.closed
    assert json.loads(stdout.getvalue())["scores"]["missing_rate"] == [0, 0]


def test_both_inputs_cannot_consume_standard_input() -> None:
    stdin = StringIO("1,2\n")
    stderr = StringIO()
    with patch("sys.stdin", stdin), patch("sys.stderr", stderr):
        code = main(["screen", "-", "--missing-applicable-mask", "-"])
    assert code == 1
    assert "cannot both use stdin" in stderr.getvalue()
    assert stdin.tell() == 0


@pytest.mark.parametrize(
    "token", ["0.5", "1.00000000000000000000001", "1e-4000", "NaN", "sNaN", "inf", "True", ""]
)
def test_text_mask_rejects_nonbinary_values_without_rounding(tmp_path: Path, token: str) -> None:
    source = tmp_path / "mask.csv"
    source.write_text(f"1,{token}\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 1, column 2: mask values must be numeric 0 or 1"):
        _load_applicable_mask(source, (2, 2))


@pytest.mark.parametrize("delimiter", [",", ";", "\t", " "])
def test_text_mask_accepts_exact_numeric_binary_spellings(tmp_path: Path, delimiter: str) -> None:
    source = tmp_path / "mask.txt"
    source.write_text(f"+1{delimiter}-0\n0.000{delimiter}1e0\n", encoding="utf-8")
    mask = _load_applicable_mask(source, (2, 2))
    assert mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])


@pytest.mark.parametrize("command", ["screen", "composite"])
@pytest.mark.parametrize(
    "damage",
    [
        "numeric-npy",
        "object-npy",
        "archive-npy",
        "archive-npz",
        "compressed-npy",
        "wrong-row-npy",
        "wrong-column-npy",
        "not-2d",
        "corrupt-npy",
        "missing-file",
        "blank",
        "fraction",
        "header",
        "jagged",
        "fewer-rows",
        "extra-rows",
        "bad-quote",
        "truncated-gzip",
    ],
)
def test_invalid_masks_are_hard_errors_before_scoring_or_output(
    tmp_path: Path, command: str, damage: str
) -> None:
    source = tmp_path / "responses.csv"
    source.write_text("q1,q2\n1,2\n3,4\n", encoding="utf-8")
    mask_path = tmp_path / "mask.npy"
    if damage == "numeric-npy":
        np.save(mask_path, np.ones((2, 2), dtype=int))
    elif damage == "object-npy":
        np.save(mask_path, np.array([[True, False], [False, True]], dtype=object))
    elif damage in {"archive-npy", "archive-npz"}:
        if damage == "archive-npz":
            mask_path = mask_path.with_suffix(".npz")
        with mask_path.open("wb") as handle:
            np.savez(handle, mask=np.ones((2, 2), dtype=bool))
    elif damage == "compressed-npy":
        np.save(mask_path, np.ones((2, 2), dtype=bool))
        compressed = mask_path.with_suffix(".npy.gz")
        compressed.write_bytes(gzip.compress(mask_path.read_bytes()))
        mask_path = compressed
    elif damage in {"wrong-row-npy", "wrong-column-npy", "not-2d"}:
        shape = {"wrong-row-npy": (3, 2), "wrong-column-npy": (2, 3), "not-2d": (4,)}[damage]
        np.save(mask_path, np.ones(shape, dtype=bool))
    elif damage == "corrupt-npy":
        mask_path.write_bytes(b"not a NumPy matrix")
    elif damage == "missing-file":
        pass
    else:
        content = {
            "blank": "1,\n0,1\n",
            "fraction": "1,0.5\n0,1\n",
            "header": "q1,q2\n1,0\n0,1\n",
            "jagged": "1,0\n1\n",
            "fewer-rows": "1,0\n",
            "extra-rows": "1,0\n0,1\n1,1\n",
            "bad-quote": '1,"0\n0,1\n',
            "truncated-gzip": "1,0\n0,1\n",
        }[damage]
        mask_path = mask_path.with_suffix(".csv")
        if damage == "truncated-gzip":
            mask_path = mask_path.with_suffix(".csv.gz")
            mask_path.write_bytes(gzip.compress(content.encode())[:-8])
        else:
            mask_path.write_text(content, encoding="utf-8")
    output = tmp_path / "result.json"
    output.write_text("previous verified result", encoding="utf-8")
    stderr = StringIO()
    with (
        patch("sys.stderr", stderr),
        patch("ier.cli.screen", side_effect=AssertionError("invalid mask reached scoring")),
        patch("ier.cli.composite", side_effect=AssertionError("invalid mask reached scoring")),
    ):
        code = main(
            [
                command,
                str(source),
                "--missing-applicable-mask",
                str(mask_path),
                "--indices",
                "irv",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
    assert code == 1
    assert "invalid --missing-applicable-mask" in stderr.getvalue()
    assert "warning:" not in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
    assert output.read_text(encoding="utf-8") == "previous verified result"


@pytest.mark.parametrize("command", ["screen-scores", "composite-scores", "response-time-scores"])
@pytest.mark.parametrize("option", ["--missing-applicable-mask", "--item-column"])
def test_fresh_matrix_options_are_unavailable_for_saved_scores(command: str, option: str) -> None:
    with patch("sys.stderr", StringIO()), pytest.raises(SystemExit) as error:
        main([command, "saved.npz", option, "unused"])
    assert error.value.code == 2


def _export_fixture(root: Path) -> tuple[Path, np.ndarray]:
    """Write a Qualtrics-like export with metadata around ten numeric items."""
    rng = np.random.default_rng(31)
    items = rng.integers(1, 6, size=(12, 10)).astype(float)
    items[:3] = 3.0
    items[4, 6] = np.nan
    source = root / "export.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ResponseId", "StartDate", "Duration", *[f"Q{i}" for i in range(1, 11)]])
        for row, values in enumerate(items):
            cells = ["" if np.isnan(value) else str(int(value)) for value in values]
            writer.writerow([f"R_{row:03d}", f"2024-01-{row + 1:02d} 09:30:00", 300 + row, *cells])
    return source, items


@pytest.mark.parametrize(
    "selection",
    [
        ["--item-pattern", "Q*"],
        ["--exclude-column", "StartDate", "--exclude-column", "Duration"],
        ["--item-pattern", "Q?", "--item-pattern", "Q1*"],
    ],
)
@pytest.mark.parametrize("command", ["screen", "composite"])
def test_survey_exports_score_items_selected_by_pattern_or_exclusion(
    tmp_path: Path, selection: list[str], command: str
) -> None:
    source, items = _export_fixture(tmp_path)
    values = selection[1::2]
    patterns = values if selection[0] == "--item-pattern" else None
    matrix, identifiers = _load_input(
        source,
        None,
        "ResponseId",
        item_patterns=patterns,
        exclude_columns=None if patterns else values,
    )
    assert identifiers == [f"R_{row:03d}" for row in range(12)]
    # 'Q?' matches Q1..Q9 and 'Q1*' adds Q10; overlaps keep header order once.
    np.testing.assert_array_equal(matrix, items)
    if patterns == ["Q?", "Q1*"]:
        partial, _ = _load_input(source, None, "ResponseId", item_patterns=["Q?"])
        np.testing.assert_array_equal(partial, items[:, :9])

    stdout = StringIO()
    options = ["--include-components", "--no-standardize"] if command == "composite" else []
    with patch("sys.stdout", stdout):
        code = main(
            [
                command,
                str(source),
                "--id-column",
                "ResponseId",
                *selection,
                "--indices",
                "irv",
                "longstring",
                *options,
                "--format",
                "json",
            ]
        )
    assert code == 0
    result = json.loads(stdout.getvalue())
    assert result["respondent_ids"] == identifiers
    scores = result["scores"] if command == "screen" else result["component_scores"]
    np.testing.assert_allclose(scores["longstring"], longstring_scores(items))
    np.testing.assert_allclose(scores["irv"], irv(items))


def test_patterns_keep_header_order_and_combine_with_exclusions(tmp_path: Path) -> None:
    source = tmp_path / "items.csv"
    source.write_text("id,b2,a1,meta,b1,a2\nx,1,2,note,3,4\ny,5,6,note,7,8\n", encoding="utf-8")
    matrix, identifiers = _load_input(
        source, None, "id", item_patterns=["a*", "b*", "?1"], exclude_columns=["b1"]
    )
    assert identifiers == ["x", "y"]
    # Header order b2, a1, a2 regardless of pattern order or repeated matches.
    np.testing.assert_array_equal(matrix, [[1, 2, 4], [5, 6, 8]])

    matrix, identifiers = _load_input(
        source, None, item_columns=["a2", "b1", "a1"], exclude_columns=["b1"]
    )
    assert identifiers is None
    np.testing.assert_array_equal(matrix, [[4, 2], [8, 6]])

    # Without another selection, exclusions keep every column but the identifier.
    matrix, identifiers = _load_input(source, None, "id", exclude_columns=["meta"])
    np.testing.assert_array_equal(matrix, [[1, 2, 3, 4], [5, 6, 7, 8]])
    matrix, _ = _load_input(source, None, exclude_columns=["meta", "id", " a2 "])
    np.testing.assert_array_equal(matrix, [[1, 2, 3], [5, 6, 7]])


def test_patterns_apply_to_timing_matrices_and_applicability_masks(tmp_path: Path) -> None:
    source = tmp_path / "timing.csv"
    source.write_text("pid,t1,note,t2\na,2,x,4\nb,6,y,\n", encoding="utf-8")
    mask = tmp_path / "mask.csv"
    mask.write_text("1,1\n1,0\n", encoding="utf-8")
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            ["response-time", str(source), "--id-column", "pid", "--item-pattern", "t*"]
            + ["--format", "json"]
        )
    assert code == 0
    assert json.loads(stdout.getvalue())["scores"] == [3.0, 6.0]
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            ["screen", str(source), "--id-column", "pid", "--exclude-column", "note"]
            + ["--indices", "missing_rate", "--missing-applicable-mask", str(mask)]
            + ["--format", "json"]
        )
    assert code == 0
    assert json.loads(stdout.getvalue())["scores"]["missing_rate"] == [0.0, 0.0]


@pytest.mark.parametrize(
    ("contents", "options", "message"),
    [
        ("id,q1,q2\na,1,2\n", {"item_patterns": ["Q*"]}, "item pattern 'Q*' matched no columns"),
        (
            "id,q1,q2\na,1,2\n",
            {"item_patterns": ["q*", "z*"]},
            "item pattern 'z*' matched no columns",
        ),
        ("id,q1,q2\na,1,2\n", {"item_patterns": [" "]}, "nonblank pattern"),
        ("id,q1,q2\na,1,2\n", {"item_patterns": []}, "nonblank pattern"),
        (
            "id,q1,q2\na,1,2\n",
            {"exclude_columns": ["q3"]},
            "excluded column 'q3' was not found in the header",
        ),
        (
            "id,q1,q1\na,1,2\n",
            {"exclude_columns": ["q1"]},
            "excluded column 'q1' appears more than once in the header",
        ),
        (
            "id,q1,q2\na,1,2\n",
            {"exclude_columns": ["q1", "q1"]},
            "excluded columns cannot contain duplicate names",
        ),
        ("id,q1,q2\na,1,2\n", {"exclude_columns": [""]}, "nonblank name"),
        (
            "id,q1,q2\na,1,2\n",
            {"id_column": "id", "item_patterns": ["*"]},
            "ID column 'id' cannot also be selected as an item column; exclude it",
        ),
        (
            "id,q1,q2\na,1,2\n",
            {"id_column": "id", "item_patterns": ["q*"], "exclude_columns": ["q1", "q2"]},
            "column selection leaves no item columns to score",
        ),
        (
            "id,q1,q2\na,1,2\n",
            {"id_column": "id", "exclude_columns": ["q1", "q2"]},
            "column selection leaves no item columns to score",
        ),
        (
            "id,q1\na,1\n",
            {"item_columns": ["q1"], "exclude_columns": ["q1"]},
            "column selection leaves no item columns to score",
        ),
        (
            "id\na\n",
            {"id_column": "id", "exclude_columns": ["id"]},
            "at least one item column besides the ID column",
        ),
        (
            "1,2\n3,4\n",
            {"item_patterns": ["q*"], "header_mode": "absent"},
            "--header absent cannot be used with --item-pattern or --exclude-column",
        ),
        (
            "1,2\n3,4\n",
            {"exclude_columns": ["1"], "header_mode": "absent"},
            "--header absent cannot be used with --item-pattern or --exclude-column",
        ),
    ],
)
def test_invalid_pattern_and_exclusion_selections(
    tmp_path: Path, contents: str, options: dict[str, object], message: str
) -> None:
    source = tmp_path / "items.csv"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        _load_input(source, None, **options)  # type: ignore[arg-type]


@pytest.mark.parametrize("option", ["--item-pattern", "--exclude-column"])
def test_pattern_and_exclusion_selections_reject_binary_matrices(
    tmp_path: Path, option: str
) -> None:
    source = tmp_path / "responses.npy"
    np.save(source, np.ones((3, 2)))
    stderr = StringIO()
    with patch("sys.stderr", stderr):
        code = main(["screen", str(source), option, "q1", "--indices", "irv"])
    assert code == 1
    assert "--item-pattern and --exclude-column are not supported with .npy input" in (
        stderr.getvalue()
    )
    assert "Traceback" not in stderr.getvalue()


@pytest.mark.parametrize(
    "selection",
    [
        ["--item-pattern", "q*", "--item-columns", "q1"],
        ["--item-columns", "q1", "--item-pattern", "q*"],
        ["--item-column", "q1", "--item-pattern", "q*"],
        ["--item-pattern", "q*", "--item-column", "q1"],
    ],
)
@pytest.mark.parametrize("command", ["screen", "composite", "response-time"])
def test_patterns_cannot_be_mixed_with_exact_column_names(
    selection: list[str], command: str
) -> None:
    stderr = StringIO()
    with patch("sys.stderr", stderr), pytest.raises(SystemExit) as error:
        main([command, "responses.csv", *selection])
    assert error.value.code == 2
    assert "not allowed with argument" in stderr.getvalue()
