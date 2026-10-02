"""Check timing archive reuse against explicit decisions without refitting metrics."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from ier import load_response_time_archive, save_response_time_archive, save_score_archive
from ier.cli import main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("output_format", ["json", "csv", "npz", "text"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("metric", ["median", "mixture"])
def test_reflag_saved_timings_preserves_scores_ids_and_tail(
    tmp_path: Path, output_format: str, fixed: bool, metric: str
) -> None:
    archive = tmp_path / "timing.npz"
    destination = tmp_path / f"revised.{output_format}"
    if metric == "mixture":
        scores = np.array([0.2, 0.6, 0.6, 0.9, np.nan])
        direction = "high"
        initial_cutoff = 0.2
        cutoff = 0.6
        expected_flags = [False, fixed, fixed, True, False]
        ranking = ["D", "B", "C", "A"]
    else:
        scores = np.array([0.5, 1.0, 1.0, 3.0, np.nan])
        direction = "low"
        initial_cutoff = 3.0
        cutoff = 1.0
        expected_flags = [True, fixed, fixed, False, False]
        ranking = ["A", "B", "C", "D"]
    save_response_time_archive(
        archive,
        scores,
        [True, True, True, True, False],
        threshold=initial_cutoff,
        metric=metric,
        flag_direction=direction,
        respondent_ids=["A", "B", "C", "D", "E"],
    )
    decision = ["--threshold", str(cutoff)] if fixed else ["--percentile", "50"]
    with (
        patch("ier.cli._load_input", side_effect=AssertionError("matrix input attempted")),
        patch(
            "ier.cli._score_response_times", side_effect=AssertionError("metric refit attempted")
        ),
    ):
        assert (
            main(
                [
                    "response-time-scores",
                    str(archive),
                    *decision,
                    "--format",
                    output_format,
                    "--output",
                    str(destination),
                ]
            )
            == 0
        )
    if output_format == "json":
        result = json.loads(destination.read_text())
        assert result["scores"] == [*scores[:4], None]
        assert result["flags"] == expected_flags
        assert result["respondent_ids"] == ["A", "B", "C", "D", "E"]
        assert result["metric"] == metric
        assert result["flag_direction"] == direction
        assert result["threshold"] == cutoff
    elif output_format == "csv":
        with destination.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["respondent"] for row in rows] == ["A", "B", "C", "D", "E"]
        assert [float(row["response_time_score"]) for row in rows[:4]] == list(scores[:4])
        assert rows[4]["response_time_score"] == ""
        assert [row["response_time_flag"] for row in rows] == [
            str(int(flag)) for flag in expected_flags
        ]
    elif output_format == "npz":
        result = load_response_time_archive(destination)
        np.testing.assert_array_equal(result["scores"], scores)
        np.testing.assert_array_equal(result["flags"], expected_flags)
        assert result["respondent_ids"] == ["A", "B", "C", "D", "E"]
        assert result["metric"] == metric
        assert result["flag_direction"] == direction
        assert result["threshold"] == cutoff
    else:
        text = destination.read_text()
        assert f"metric: {metric}" in text
        assert f"flag direction: {direction}" in text
        assert f"threshold: {cutoff:g}" in text
        assert f"flagged: {sum(expected_flags)}" in text
        assert [line.strip().split("\t")[0] for line in text.splitlines()[6:]] == ranking


@pytest.mark.parametrize("metric", ["mean", "median", "sd", "min", "consistency", "mixture"])
@pytest.mark.parametrize("inclusive", [False, True])
def test_export_without_cutoff_retains_exact_saved_decisions(
    tmp_path: Path, metric: str, inclusive: bool
) -> None:
    archive = tmp_path / "timing.npz"
    destination = tmp_path / "export.json"
    scores = [0.2, 0.5, 0.5, 0.8, np.nan]
    direction = "high" if metric == "mixture" else "low"
    flags = [False, inclusive, inclusive, True, False]
    if direction == "low":
        flags = [True, inclusive, inclusive, False, False]
    save_response_time_archive(
        archive, scores, flags, threshold=0.5, metric=metric, flag_direction=direction
    )
    with patch("ier.cli.threshold_flags", side_effect=AssertionError("reflagging attempted")):
        assert (
            main(
                [
                    "response-time-scores",
                    str(archive),
                    "--format",
                    "json",
                    "--output",
                    str(destination),
                ]
            )
            == 0
        )
    result = json.loads(destination.read_text())
    assert result["scores"] == [0.2, 0.5, 0.5, 0.8, None]
    assert result["flags"] == flags
    assert result["threshold"] == 0.5
    assert "respondent_ids" not in result


def test_timing_cli_archive_reused_in_place_after_original_input_removed(tmp_path: Path) -> None:
    source = tmp_path / "timings.csv"
    source.write_text("id,t1,t2,t3\nfast,0.4,0.5,0.6\nsteady,1,1,1\nslow,2,3,4\n")
    archive = tmp_path / "timing.npz"
    assert (
        main(
            [
                "response-time",
                str(source),
                "--id-column",
                "id",
                "--threshold",
                "1",
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
                "response-time-scores",
                str(archive),
                "--percentile",
                "50",
                "--format",
                "npz",
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    result = load_response_time_archive(archive)
    np.testing.assert_array_equal(result["scores"], [0.5, 1.0, 3.0])
    np.testing.assert_array_equal(result["flags"], [True, False, False])
    assert result["respondent_ids"] == ["fast", "steady", "slow"]
    assert result["threshold"] == 1.0
    # A second conversion must retain the strict percentile tie decision.
    exported = subprocess.run(
        [
            sys.executable,
            "-m",
            "ier.cli",
            "response-time-scores",
            str(archive),
            "--format",
            "json",
            "--output",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert exported.stderr == ""
    assert json.loads(exported.stdout)["flags"] == [True, False, False]


def test_reused_timing_csv_streams_to_compressed_output(tmp_path: Path) -> None:
    archive = tmp_path / "timing.npz"
    destination = tmp_path / "timing.csv.gz"
    save_response_time_archive(
        archive,
        [0.5, 2.0],
        [True, False],
        threshold=1.0,
        respondent_ids=["quoted,identifier", "plain"],
    )
    assert (
        main(
            [
                "response-time-scores",
                str(archive),
                "--format",
                "csv",
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    with gzip.open(destination, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "respondent": "quoted,identifier",
            "response_time_score": "0.5",
            "response_time_flag": "1",
        },
        {"respondent": "plain", "response_time_score": "2.0", "response_time_flag": "0"},
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--threshold", "nan"], "threshold must be a finite number"),
        (["--threshold", "inf"], "threshold must be a finite number"),
        (["--percentile", "-1"], "percentile must be a finite number between 0 and 100"),
        (["--percentile", "101"], "percentile must be a finite number between 0 and 100"),
        (["--percentile", "nan"], "percentile must be a finite number between 0 and 100"),
    ],
)
def test_invalid_timing_cutoff_preserves_previous_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    message: str,
) -> None:
    archive = tmp_path / "timing.npz"
    destination = tmp_path / "previous.json"
    save_response_time_archive(archive, [0.5, 2.0], [True, False], threshold=1.0)
    destination.write_text("previous result")
    assert (
        main(
            [
                "response-time-scores",
                str(archive),
                *options,
                "--format",
                "json",
                "--output",
                str(destination),
            ]
        )
        == 1
    )
    assert destination.read_text() == "previous result"
    error = capsys.readouterr().err
    assert message in error
    assert "Traceback" not in error


@pytest.mark.parametrize("invalid_source", ["wrong_type", "damaged", "missing"])
def test_timing_archive_validation_preserves_previous_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], invalid_source: str
) -> None:
    archive = tmp_path / "invalid.npz"
    if invalid_source == "wrong_type":
        save_score_archive(archive, {"irv": [0.5, 2.0]})
    elif invalid_source == "damaged":
        archive.write_bytes(b"not a numpy archive")
    destination = tmp_path / "previous.json"
    destination.write_text("previous result")
    assert (
        main(
            [
                "response-time-scores",
                str(archive),
                "--format",
                "json",
                "--output",
                str(destination),
            ]
        )
        == 1
    )
    assert destination.read_text() == "previous result"
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert "Traceback" not in error


@pytest.mark.parametrize("command", ["response-time", "response-time-scores"])
def test_timing_decision_options_are_mutually_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    destination = tmp_path / "previous.json"
    destination.write_text("previous result")
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                command,
                str(tmp_path / "unread-input.npz"),
                "--threshold",
                "1",
                "--percentile",
                "5",
                "--format",
                "json",
                "--output",
                str(destination),
            ]
        )
    assert "not allowed with argument --threshold" in capsys.readouterr().err
    assert destination.read_text() == "previous result"


@pytest.mark.parametrize(
    "options", [["--metric", "mean"], ["--components", "3"], ["--header", "present"]]
)
def test_saved_timing_command_rejects_matrix_and_refitting_options(
    capsys: pytest.CaptureFixture[str], options: list[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["response-time-scores", "unread-input.npz", *options])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_saved_timing_npz_requires_destination_before_input_is_loaded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["response-time-scores", "unread-input.npz", "--format", "npz"]) == 1
    assert "--format npz requires --output" in capsys.readouterr().err
