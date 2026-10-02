"""Exercise saved-score decisions through the CLI with independent expected values."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from ier import load_score_archive, save_score_archive
from ier.cli import main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def saved_scores(tmp_path: Path) -> Path:
    path = tmp_path / "scores.npz"
    save_score_archive(
        path,
        {"irv": [0.0, 2.0, np.nan, 1.0], "longstring": [4.0, 1.0, 3.0, np.nan]},
        respondent_ids=["A", "B", "C", "D"],
        errors={"mad": "paired items were not configured"},
    )
    return path


@pytest.mark.parametrize("output_format", ["json", "csv", "npz", "text"])
def test_reused_screen_cutoffs_coverage_and_provenance(
    saved_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], output_format: str
) -> None:
    destination = tmp_path / f"result.{output_format}"
    # Zero IRV flags low; a long run flags high. C has one signal but lacks coverage.
    argv = [
        "screen-scores",
        str(saved_scores),
        "--threshold",
        "irv=0",
        "--threshold",
        "longstring=3",
        "--min-flags",
        "1",
        "--min-valid-indices",
        "2",
        "--format",
        output_format,
        "--output",
        str(destination),
    ]
    with patch("ier.cli.screen", side_effect=AssertionError("index scoring attempted")):
        assert main(argv) == 0
    assert "mad" in capsys.readouterr().err
    if output_format == "json":
        result = json.loads(destination.read_text())
        assert result["respondent_ids"] == ["A", "B", "C", "D"]
        assert result["flag_counts"] == [2, 0, 1, 0]
        assert result["valid_index_counts"] == [2, 2, 1, 1]
        assert result["consensus_flags"] == [True, False, False, False]
        assert result["threshold_sources"] == {"irv": "fixed", "longstring": "fixed"}
        assert result["errors"] == {"mad": "paired items were not configured"}
    elif output_format == "csv":
        with destination.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["respondent"] for row in rows] == ["A", "B", "C", "D"]
        assert [row["flag_count"] for row in rows] == ["2", "0", "1", "0"]
        assert [row["consensus_flag"] for row in rows] == ["1", "0", "0", "0"]
        assert rows[2]["irv_score"] == ""
    elif output_format == "npz":
        archive = load_score_archive(destination)
        assert archive["errors"] == {"mad": "paired items were not configured"}
        assert archive["respondent_ids"] == ["A", "B", "C", "D"]
        with np.load(destination, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["flag_counts"], [2, 0, 1, 0])
            np.testing.assert_array_equal(result["consensus_flags"], [True, False, False, False])
    else:
        text = destination.read_text()
        assert "mad" in text
        assert "A\t2\t2\t1" in text
        assert "C\t1\t1\t0" in text


@pytest.mark.parametrize("output_format", ["json", "csv", "npz", "text"])
def test_reused_weighted_composite_with_components_flags_and_probability(
    saved_scores: Path, tmp_path: Path, output_format: str
) -> None:
    destination = tmp_path / f"composite.{output_format}"
    argv = [
        "composite-scores",
        str(saved_scores),
        "--no-standardize",
        "--weight",
        "irv=2",
        "--min-valid-indices",
        "2",
        "--threshold",
        "1",
        "--include-components",
        "--include-probability",
        "--format",
        output_format,
        "--output",
        str(destination),
    ]
    with (
        patch("ier.cli.composite", side_effect=AssertionError("index scoring attempted")),
        patch("ier.cli.composite_summary", side_effect=AssertionError("index scoring attempted")),
    ):
        assert main(argv) == 0
    expected = [4.0 / 3.0, -1.0, np.nan, np.nan]
    if output_format == "json":
        result = json.loads(destination.read_text())
        np.testing.assert_allclose(result["scores"][:2], expected[:2])
        assert result["scores"][2:] == [None, None]
        assert result["valid_index_counts"] == [2, 2, 1, 1]
        assert result["flags"] == [True, False, False, False]
        assert result["probabilities"][:2] == pytest.approx([0.7913914727, 0.2689414214])
        assert result["errors"] == {"mad": "paired items were not configured"}
        assert result["respondent_ids"] == ["A", "B", "C", "D"]
    elif output_format == "csv":
        with destination.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["respondent"] for row in rows] == ["A", "B", "C", "D"]
        assert float(rows[0]["composite_score"]) == pytest.approx(4.0 / 3.0)
        assert rows[1]["composite_score"] == "-1.0"
        assert rows[2]["composite_score"] == ""
        assert [row["composite_flag"] for row in rows] == ["1", "0", "0", "0"]
        assert [row["valid_index_count"] for row in rows] == ["2", "2", "1", "1"]
        assert float(rows[0]["composite_probability"]) == pytest.approx(0.7913914727)
    elif output_format == "npz":
        archive = load_score_archive(destination)
        assert archive["respondent_ids"] == ["A", "B", "C", "D"]
        assert archive["errors"] == {"mad": "paired items were not configured"}
        with np.load(destination, allow_pickle=False) as result:
            np.testing.assert_allclose(result["scores"], expected, equal_nan=True)
            np.testing.assert_array_equal(result["flags"], [True, False, False, False])
            np.testing.assert_array_equal(result["valid_index_counts"], [2, 2, 1, 1])
    else:
        text = destination.read_text()
        assert "A\t1.333333" in text
        assert "B\t-1.000000" in text
        assert "mad" in text


@pytest.mark.parametrize("command", ["screen-scores", "composite-scores"])
@pytest.mark.parametrize("output_format", ["json", "csv", "npz", "text"])
def test_saved_failures_retain_original_coverage_requirement(
    saved_scores: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    output_format: str,
) -> None:
    destination = tmp_path / f"coverage.{output_format}"
    options = (
        ["--threshold", "mad=2"]
        if command == "screen-scores"
        else ["--weight", "mad=1e308", "--include-components"]
    )
    assert (
        main(
            [
                command,
                str(saved_scores),
                "--min-valid-indices",
                "3",
                *options,
                "--format",
                output_format,
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    assert "mad" in capsys.readouterr().err
    if output_format == "json":
        result = json.loads(destination.read_text())
        assert result["min_valid_indices"] == 3
        assert result["valid_index_counts"] == [2, 2, 1, 1]
        assert result["errors"] == {"mad": "paired items were not configured"}
        if command == "screen-scores":
            assert result["consensus_eligible"] == [False] * 4
            assert result["consensus_flags"] == [False] * 4
        else:
            assert result["scores"] == [None] * 4
            assert result["weights"] == {"mad": 1e308}
    elif output_format == "npz":
        archive = load_score_archive(destination)
        assert archive["errors"] == {"mad": "paired items were not configured"}
        with np.load(destination, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["valid_index_counts"], [2, 2, 1, 1])
            if command == "screen-scores":
                assert not result["consensus_eligible"].any()
                assert not result["consensus_flags"].any()
            else:
                assert np.isnan(result["scores"]).all()
    elif output_format == "csv":
        with destination.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["respondent"] for row in rows] == ["A", "B", "C", "D"]
        column = "consensus_flag" if command == "screen-scores" else "composite_score"
        expected = "0" if command == "screen-scores" else ""
        assert [row[column] for row in rows] == [expected] * 4
    else:
        text = destination.read_text()
        assert "mad" in text
        if command == "screen-scores":
            assert "consensus eligible: 0 (min_valid_indices=3)" in text
        else:
            assert "minimum valid indices: 3" in text


def test_saved_composite_aggregate_only_honors_failed_coverage(
    saved_scores: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "aggregate.json"
    assert (
        main(
            [
                "composite-scores",
                str(saved_scores),
                "--min-valid-indices",
                "3",
                "--weight",
                "mad=2",
                "--format",
                "json",
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    assert json.loads(destination.read_text())["scores"] == [None] * 4


@pytest.mark.parametrize("command", ["screen-scores", "composite-scores"])
def test_selected_scores_reorder_and_drop_unselected_archive_failures(
    saved_scores: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    destination = tmp_path / "selected.npz"
    options = ["--include-components"] if command == "composite-scores" else []
    assert (
        main(
            [
                command,
                str(saved_scores),
                "--indices",
                "longstring",
                "irv",
                "--strict",
                *options,
                "--format",
                "npz",
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    assert capsys.readouterr().err == ""
    result = load_score_archive(destination)
    assert list(result["scores"]) == ["longstring", "irv"]
    assert result["errors"] == {}


@pytest.mark.parametrize("command", ["screen-scores", "composite-scores"])
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--strict"], "archive index 'mad' failed"),
        (["--indices", "mad"], "index 'mad' has no saved scores: paired items"),
        (["--indices", "markov"], "index 'markov' has no saved scores"),
        (["--indices", "irv", "irv"], "duplicate index"),
    ],
)
def test_reuse_failures_preserve_previous_results(
    saved_scores: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    options: list[str],
    message: str,
) -> None:
    destination = tmp_path / "previous.json"
    destination.write_text("previous result")
    assert (
        main(
            [command, str(saved_scores), *options, "--format", "json", "--output", str(destination)]
        )
        == 1
    )
    assert destination.read_text() == "previous result"
    error = capsys.readouterr().err
    assert message in error
    assert "Traceback" not in error


def test_scores_reuse_actual_cli_archive_after_original_input_removed(tmp_path: Path) -> None:
    source = tmp_path / "survey.csv"
    source.write_text("id,q1,q2,q3\nX,1,1,1\nY,1,2,3\nZ,3,2,1\n")
    archive = tmp_path / "screen.npz"
    assert (
        main(
            [
                "screen",
                str(source),
                "--id-column",
                "id",
                "--indices",
                "irv",
                "longstring",
                "--format",
                "npz",
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    source.unlink()
    assert (
        main(
            [
                "screen-scores",
                str(archive),
                "--index-percentile",
                "irv=50",
                "--min-flags",
                "1",
                "--format",
                "npz",
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    with np.load(archive, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["flag__irv"], [True, False, False])
        assert result["respondent_ids"].tolist() == ["X", "Y", "Z"]
        assert result["threshold_sources"].tolist() == ["percentile", "percentile"]


def test_aggregate_archive_requires_saved_components(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "data.csv"
    source.write_text("1,1,1\n1,2,3\n")
    archive = tmp_path / "aggregate.npz"
    assert (
        main(
            [
                "composite",
                str(source),
                "--indices",
                "longstring",
                "--format",
                "npz",
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    assert main(["composite-scores", str(archive)]) == 1
    assert "--include-components" in capsys.readouterr().err


def test_composite_reuse_omits_screen_only_failures_but_strict_retains_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "screen.npz"
    destination = tmp_path / "composite.npz"
    save_score_archive(
        archive,
        {"irv": [0.1, 0.2], "longstring": [1.0, 2.0]},
        errors={"acquiescence": "invalid polarity pairs"},
    )
    assert (
        main(
            [
                "composite-scores",
                str(archive),
                "--no-standardize",
                "--include-components",
                "--format",
                "npz",
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    assert capsys.readouterr().err == ""
    assert load_score_archive(destination)["errors"] == {}
    with np.load(destination, allow_pickle=False) as result:
        np.testing.assert_allclose(result["scores"], [0.45, 0.9])
    assert main(["composite-scores", str(archive), "--strict"]) == 1
    assert "archive index 'acquiescence' failed" in capsys.readouterr().err


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
def test_score_only_composite_reuse_uses_selected_reduction_and_percentile(
    saved_scores: Path, tmp_path: Path, method: str
) -> None:
    destination = tmp_path / "scores.json"
    assert (
        main(
            [
                "composite-scores",
                str(saved_scores),
                "--method",
                method,
                "--no-standardize",
                "--percentile",
                "50",
                "--format",
                "json",
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    expected = {
        "mean": [2.0, -0.5, 3.0, -1.0],
        "sum": [4.0, -1.0, 3.0, -1.0],
        "max": [4.0, 1.0, 3.0, -1.0],
    }[method]
    result = json.loads(destination.read_text())
    np.testing.assert_allclose(result["scores"], expected)
    assert result["flags"] == [True, False, True, False]
    assert result["method"] == method
    assert result["threshold_source"] == "percentile"
    assert "components" not in result
