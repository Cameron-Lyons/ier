"""Failed-only screening results retain rows and provenance during replay."""

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from ier import load_score_archive, screen, screen_scores
from ier.cli import main


@pytest.mark.parametrize("minimum", [None, 1, 2])
def test_failed_only_scores_replay_matches_fresh_screening(
    minimum: int | None,
) -> None:
    settings = {
        "min_flags": 1,
        "min_valid_indices": minimum,
        "thresholds": {"mad": 2.0},
        "percentiles": {"evenodd": 90.0},
    }
    fresh = screen([[1, 2], [2, 3], [3, 4]], indices=["mad", "evenodd"], **settings)

    with patch(
        "ier.screen.score_registered_indices",
        side_effect=AssertionError("saved failures must not trigger index scoring"),
    ):
        replay = screen_scores(
            fresh["scores"],
            errors=fresh["errors"],
            n_respondents=fresh["n_respondents"],
            **settings,
        )

    assert replay["n_respondents"] == 3
    assert replay["n_indices"] == 0
    assert list(replay["errors"]) == ["mad", "evenodd"]
    assert replay["errors"] is not fresh["errors"]
    for key in fresh:
        if isinstance(fresh[key], np.ndarray):
            np.testing.assert_array_equal(replay[key], fresh[key])
        else:
            assert replay[key] == fresh[key]
    np.testing.assert_array_equal(replay["flag_counts"], [0, 0, 0])
    np.testing.assert_array_equal(replay["valid_index_counts"], [0, 0, 0])
    np.testing.assert_array_equal(replay["consensus_flags"], [False, False, False])
    np.testing.assert_array_equal(replay["consensus_eligible"], [minimum is None] * 3)


@pytest.mark.parametrize(
    "count", [0, -1, True, np.bool_(True), 2.5, "3", int(np.iinfo(np.intp).max) + 1]
)
def test_explicit_respondent_count_rejects_invalid_values(count: Any) -> None:
    with pytest.raises(ValueError, match="n_respondents must be a positive integer"):
        screen_scores({}, errors={"mad": "missing configuration"}, n_respondents=count)


def test_explicit_respondent_count_checks_available_scores_and_preserves_arrays() -> None:
    scores = np.array([0.0, 1.0, np.nan])
    replay = screen_scores({"irv": scores}, n_respondents=np.int64(3))
    assert replay["scores"]["irv"] is scores
    assert replay["n_respondents"] == 3
    with pytest.raises(ValueError, match="must match n_respondents"):
        screen_scores({"irv": scores}, n_respondents=2)


def test_failed_only_scores_require_count_and_failure_provenance() -> None:
    with pytest.raises(ValueError, match="at least one registered index"):
        screen_scores({}, errors={"mad": "missing configuration"})
    with pytest.raises(ValueError, match="at least one scored or failed index"):
        screen_scores({}, n_respondents=3)
    with pytest.raises(ValueError, match="selected indices \\(1\\)"):
        screen_scores(
            {}, errors={"mad": "missing configuration"}, n_respondents=3, min_valid_indices=2
        )
    with pytest.raises(ValueError, match="both a threshold and percentile"):
        screen_scores(
            {},
            errors={"mad": "missing configuration"},
            n_respondents=3,
            thresholds={"mad": 2.0},
            percentiles={"mad": 90.0},
        )


def _failed_archive(tmp_path: Path) -> Path:
    source = tmp_path / "responses.csv"
    source.write_text("id,q1,q2\nA,1,2\nB,2,3\nC,3,4\n")
    archive = tmp_path / "failed.npz"
    assert (
        main(
            [
                "screen",
                str(source),
                "--id-column",
                "id",
                "--indices",
                "mad",
                "evenodd",
                "--format",
                "npz",
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    source.unlink()
    return archive


@pytest.mark.parametrize("output_format", ["json", "csv", "npz", "text"])
def test_cli_replays_failed_only_archive_after_original_input_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], output_format: str
) -> None:
    archive = _failed_archive(tmp_path)
    saved = load_score_archive(archive)
    capsys.readouterr()
    destination = tmp_path / f"replayed.{output_format}"
    assert (
        main(
            [
                "screen-scores",
                str(archive),
                "--min-valid-indices",
                "2",
                "--threshold",
                "mad=2",
                "--index-percentile",
                "evenodd=90",
                "--format",
                output_format,
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    diagnostic = capsys.readouterr().err
    assert "mad" in diagnostic and "evenodd" in diagnostic
    if output_format == "json":
        result = json.loads(destination.read_text())
        assert result["n_respondents"] == 3
        assert result["respondent_ids"] == ["A", "B", "C"]
        assert result["errors"] == saved["errors"]
        assert result["scores"] == result["flags"] == result["summary"] == {}
        assert result["valid_index_counts"] == result["flag_counts"] == [0, 0, 0]
        assert result["consensus_flags"] == result["consensus_eligible"] == [False] * 3
    elif output_format == "csv":
        with destination.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["respondent"] for row in rows] == ["A", "B", "C"]
        for row in rows:
            assert row["flag_count"] == row["valid_index_count"] == row["consensus_flag"] == "0"
    elif output_format == "npz":
        replay = load_score_archive(destination)
        assert replay == saved
        with np.load(destination, allow_pickle=False) as members:
            np.testing.assert_array_equal(members["flag_counts"], [0, 0, 0])
            assert not members["consensus_eligible"].any()
            assert not members["consensus_flags"].any()
    else:
        text = destination.read_text()
        assert "mad" in text and "evenodd" in text
        assert "consensus eligible: 0 (min_valid_indices=2)" in text
        for identifier in ["A", "B", "C"]:
            assert f"{identifier}\t0\t0\t0" in text


def test_failed_only_cli_replay_preserves_strict_failure_and_supports_in_place_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _failed_archive(tmp_path)
    original = archive.read_bytes()
    assert (
        main(
            ["screen-scores", str(archive), "--strict", "--format", "npz", "--output", str(archive)]
        )
        == 1
    )
    assert "archive index 'mad' failed" in capsys.readouterr().err
    assert archive.read_bytes() == original
    assert (
        main(
            [
                "screen-scores",
                str(archive),
                "--min-valid-indices",
                "1",
                "--format",
                "npz",
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    saved = load_score_archive(archive)
    assert saved["scores"] == {}
    assert saved["n_respondents"] == 3
    assert saved["respondent_ids"] == ["A", "B", "C"]
    with np.load(archive, allow_pickle=False) as members:
        assert not members["consensus_eligible"].any()
