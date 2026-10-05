"""Score, flag, archive, and replay response time effort through the timing CLI."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import shlex
from io import StringIO
from pathlib import Path
from typing import Any, get_args
from unittest.mock import patch

import numpy as np
import pytest

from ier import (
    load_response_time_archive,
    response_time_effort,
    response_time_effort_flag,
    response_time_score_flags,
    save_response_time_archive,
)
from ier.cli import _EFFORT_CUTOFF, _build_parser, main
from ier.types import ResponseTimeMetric

# NT10 thresholds mark six of the rushed respondent's answers and one of the
# boundary respondent's answers rapid. Boundary effort is exactly 9/10, the
# default cutoff, and the blank respondent answered nothing.
_TIMINGS = (
    "participant,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10\n"
    "steady,12,30,18,20,22,25,19,21,24,20\n"
    "rushed,0.5,1.5,20,0.4,0.3,22,0.6,21,0.2,23\n"
    "partial,14,28,,15,16,17,18,19,20,21\n"
    "boundary,13.5,20.5,22,0.5,20,21,22,23,24,25\n"
    "blank,,,,,,,,,,\n"
)
_IDS = ["steady", "rushed", "partial", "boundary", "blank"]
_ID = ["--id-column", "participant"]
_MATRIX = np.array(
    [
        [float(value) if value else np.nan for value in line.split(",")[1:]]
        for line in _TIMINGS.splitlines()[1:]
    ]
)
_SCORES = [1.0, 0.4, 1.0, 0.9, None]
_DEFAULT_FLAGS = [False, True, False, False, False]
_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def timings(tmp_path: Path) -> Path:
    path = tmp_path / "timings.csv"
    path.write_text(_TIMINGS, encoding="utf-8")
    return path


def _json_values(values: np.ndarray) -> list[object]:
    return [None if np.isnan(value) else float(value) for value in values]


def _json_output(argv: list[str]) -> dict[str, Any]:
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        assert main([*argv, "--format", "json"]) == 0
    payload: dict[str, Any] = json.loads(stdout.getvalue())
    return payload


def _usage_error(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit, match="2"):
        main(argv)
    return capsys.readouterr().err


def _failure(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    assert main(argv) == 1
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert "Traceback" not in error
    return error


def test_fixture_scores_match_the_python_api() -> None:
    scores, flags = response_time_effort_flag(_MATRIX)

    assert _json_values(scores) == _SCORES
    assert flags.tolist() == _DEFAULT_FLAGS


@pytest.mark.parametrize("output_format", ["json", "csv", "npz", "text"])
def test_effort_scores_and_default_flags_in_every_format(
    tmp_path: Path, timings: Path, output_format: str
) -> None:
    destination = tmp_path / f"effort.{output_format}"
    assert (
        main(
            [
                "response-time",
                str(timings),
                "--id-column",
                "participant",
                "--metric",
                "effort",
                "--format",
                output_format,
                "--output",
                str(destination),
            ]
        )
        == 0
    )

    if output_format == "json":
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload == {
            "n_respondents": 5,
            "metric": "effort",
            "flag_direction": "low",
            "threshold": 0.9,
            "scores": _SCORES,
            "flags": _DEFAULT_FLAGS,
            "respondent_ids": _IDS,
        }
    elif output_format == "csv":
        with destination.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["respondent"] for row in rows] == _IDS
        assert [row["response_time_score"] for row in rows] == ["1.0", "0.4", "1.0", "0.9", ""]
        assert [row["response_time_flag"] for row in rows] == ["0", "1", "0", "0", "0"]
    elif output_format == "npz":
        saved = load_response_time_archive(destination)
        assert saved["metric"] == "effort"
        assert saved["flag_direction"] == "low"
        assert saved["threshold"] == _EFFORT_CUTOFF
        assert saved["respondent_ids"] == _IDS
        assert _json_values(saved["scores"]) == _SCORES
        assert saved["flags"].tolist() == _DEFAULT_FLAGS
    else:
        text = destination.read_text(encoding="utf-8").splitlines()
        assert text[:5] == [
            "respondents: 5",
            "metric: effort",
            "flag direction: low",
            "threshold: 0.9",
            "flagged: 1",
        ]
        # Lowest effort ranks first, ties keep input order, and NaN is omitted.
        assert [line.split("\t")[0].strip() for line in text[6:]] == [
            "rushed",
            "boundary",
            "steady",
            "partial",
        ]


@pytest.mark.parametrize(
    ("options", "keywords"),
    [
        ([], {}),
        (["--effort-fraction", "0.5"], {"normative_fraction": 0.5}),
        (["--effort-fraction", "1"], {"normative_fraction": 1.0}),
        (["--effort-max-threshold", "1"], {"max_threshold": 1.0}),
        (
            ["--effort-fraction", "0.5", "--effort-max-threshold", "1"],
            {"normative_fraction": 0.5, "max_threshold": 1.0},
        ),
        (["--effort-threshold", "15"], {"thresholds": 15.0}),
    ],
)
def test_effort_options_reach_the_python_api(
    timings: Path, options: list[str], keywords: dict[str, Any]
) -> None:
    payload = _json_output(["response-time", str(timings), *_ID, "--metric", "effort", *options])

    scores, flags = response_time_effort_flag(_MATRIX, **keywords)
    assert payload["scores"] == _json_values(scores)
    assert payload["flags"] == flags.tolist()
    assert payload["threshold"] == _EFFORT_CUTOFF


@pytest.mark.parametrize(
    ("decision", "threshold", "flags"),
    [
        # A fixed RTE cutoff is strict, so the 9/10 respondent is never flagged at 0.9.
        (["--threshold", "0.9"], 0.9, [False, True, False, False, False]),
        (["--threshold", "1"], 1.0, [False, True, False, True, False]),
        (["--threshold", "0"], 0.0, [False, False, False, False, False]),
        # The median of the available scores is 0.95.
        (["--percentile", "50"], 0.95, [False, True, False, True, False]),
        # Percentile cutoffs exclude ties, including the lowest available score.
        (["--percentile", "0"], 0.4, [False, False, False, False, False]),
    ],
)
def test_effort_cutoffs_flag_strictly_below(
    timings: Path, decision: list[str], threshold: float, flags: list[bool]
) -> None:
    payload = _json_output(["response-time", str(timings), *_ID, "--metric", "effort", *decision])

    assert payload["threshold"] == pytest.approx(threshold)
    assert payload["flags"] == flags


def test_effort_options_do_not_change_other_metrics(timings: Path) -> None:
    command = ["response-time", str(timings), *_ID, "--metric", "median", "--threshold", "18"]

    plain = _json_output(command)
    configured = _json_output([*command, "--effort-threshold", "1"])

    assert configured == plain
    assert plain["flags"] == [False, True, True, False, False]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--effort-fraction", "0"], "argument --effort-fraction: must be a number greater than 0"),
        (["--effort-fraction", "-0.1"], "argument --effort-fraction: must be a number"),
        (["--effort-fraction", "1.5"], "greater than 0 and at most 1"),
        (["--effort-fraction", "nan"], "greater than 0 and at most 1"),
        (["--effort-fraction", "tenth"], "greater than 0 and at most 1"),
        (["--effort-max-threshold", "-10"], "argument --effort-max-threshold: must be a positive"),
        (["--effort-max-threshold", "0"], "must be a positive finite number"),
        (["--effort-max-threshold", "inf"], "must be a positive finite number"),
        (["--effort-threshold", "-1"], "argument --effort-threshold: must be a positive"),
        (["--effort-threshold", "nan"], "must be a positive finite number"),
    ],
)
def test_effort_option_values_are_validated_before_input_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], options: list[str], message: str
) -> None:
    error = _usage_error(
        ["response-time", str(tmp_path / "unread.csv"), "--metric", "effort", *options], capsys
    )

    assert message in error


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            ["--effort-fraction", "0.2", "--effort-threshold", "1"],
            "argument --effort-threshold: not allowed with argument --effort-fraction",
        ),
        (
            ["--effort-threshold", "1", "--effort-fraction", "0.2"],
            "argument --effort-fraction: not allowed with argument --effort-threshold",
        ),
        (
            ["--effort-threshold", "1", "--effort-max-threshold", "5"],
            "argument --effort-max-threshold: not allowed with argument --effort-threshold",
        ),
        (
            ["--effort-max-threshold", "5", "--effort-threshold", "1"],
            "argument --effort-threshold: not allowed with argument --effort-max-threshold",
        ),
    ],
)
def test_shared_effort_threshold_excludes_normative_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], options: list[str], message: str
) -> None:
    destination = tmp_path / "previous.json"
    destination.write_text("previous result", encoding="utf-8")

    error = _usage_error(
        [
            "response-time",
            str(tmp_path / "unread.csv"),
            "--metric",
            "effort",
            *options,
            "--output",
            str(destination),
        ],
        capsys,
    )

    assert message in error
    assert destination.read_text(encoding="utf-8") == "previous result"


@pytest.mark.parametrize("cutoff", ["1.5", "-0.1", "nan", "inf"])
def test_effort_cutoff_outside_the_rte_range_is_rejected_before_input_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cutoff: str
) -> None:
    destination = tmp_path / "previous.json"
    destination.write_text("previous result", encoding="utf-8")

    with patch("ier.cli._load_input", side_effect=AssertionError("input read")):
        error = _failure(
            [
                "response-time",
                str(tmp_path / "unread.csv"),
                "--metric",
                "effort",
                "--threshold",
                cutoff,
                "--format",
                "json",
                "--output",
                str(destination),
            ],
            capsys,
        )

    assert "--threshold for the effort metric must be an RTE proportion between 0 and 1" in error
    assert destination.read_text(encoding="utf-8") == "previous result"


def test_effort_archive_round_trip_replays_new_cutoffs(tmp_path: Path, timings: Path) -> None:
    archive = tmp_path / "effort.npz"
    assert (
        main(
            [
                "response-time",
                str(timings),
                "--id-column",
                "participant",
                "--metric",
                "effort",
                "--effort-max-threshold",
                "10",
                "--format",
                "npz",
                "--compress",
                "--output",
                str(archive),
            ]
        )
        == 0
    )
    saved = load_response_time_archive(archive)
    assert saved["metric"] == "effort"
    assert saved["flags"].tolist() == _DEFAULT_FLAGS

    with patch("ier.cli._score_response_times", side_effect=AssertionError("rescored")):
        preserved = _json_output(["response-time-scores", str(archive)])
        fixed = _json_output(["response-time-scores", str(archive), "--threshold", "1"])
        tied = _json_output(["response-time-scores", str(archive), "--threshold", "0.9"])
        relative = _json_output(["response-time-scores", str(archive), "--percentile", "50"])

    assert preserved["threshold"] == 0.9
    assert preserved["flags"] == _DEFAULT_FLAGS
    assert preserved["respondent_ids"] == _IDS
    assert fixed["metric"] == "effort"
    assert fixed["threshold"] == 1.0
    assert fixed["flags"] == [False, True, False, True, False]
    assert tied["flags"] == _DEFAULT_FLAGS
    assert relative["threshold"] == pytest.approx(0.95)
    assert relative["flags"] == [False, True, False, True, False]
    for payload in (preserved, fixed, tied, relative):
        assert payload["scores"] == _SCORES

    # Reflagged effort archives keep the strict decision and load again.
    revised = tmp_path / "revised.npz"
    assert (
        main(
            [
                "response-time-scores",
                str(archive),
                "--threshold",
                "1",
                "--format",
                "npz",
                "--output",
                str(revised),
            ]
        )
        == 0
    )
    reloaded = load_response_time_archive(revised)
    assert reloaded["threshold"] == 1.0
    assert reloaded["flags"].tolist() == [False, True, False, True, False]


def test_saved_effort_cutoff_outside_the_rte_range_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "effort.npz"
    save_response_time_archive(archive, [0.5, 1.0], [True, False], threshold=0.9, metric="effort")

    error = _failure(["response-time-scores", str(archive), "--threshold", "2"], capsys)

    assert "must be an RTE proportion between 0 and 1" in error


@pytest.mark.parametrize("inclusive", [False, True])
def test_saved_effort_decisions_at_the_cutoff_are_preserved(
    tmp_path: Path, inclusive: bool
) -> None:
    archive = tmp_path / "effort.npz"
    scores = np.array([0.5, 0.9, 1.0, np.nan])
    flags = scores <= 0.9 if inclusive else scores < 0.9
    save_response_time_archive(archive, scores, flags, threshold=0.9, metric="effort")

    loaded = load_response_time_archive(archive)
    exported = _json_output(["response-time-scores", str(archive)])

    assert loaded["metric"] == "effort"
    assert loaded["flag_direction"] == "low"
    assert loaded["flags"].tolist() == flags.tolist()
    assert exported["flags"] == flags.tolist()
    assert exported["threshold"] == 0.9


def test_effort_archive_requires_the_low_tail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metric 'effort' requires 'low' flag_direction"):
        save_response_time_archive(
            tmp_path / "effort.npz",
            [0.5, 1.0],
            [False, True],
            threshold=0.9,
            metric="effort",
            flag_direction="high",
        )


def _write_response_time_members(
    path: Path, metric: str, scores: list[float], threshold: float
) -> None:
    """Write a response-time archive member by member, bypassing the saver."""
    values = np.asarray(scores, dtype=np.float64)
    direction = "high" if metric == "mixture" else "low"
    flags = values >= threshold if direction == "high" else values < threshold
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        result_type=np.asarray("response_time", dtype=np.str_),
        n_respondents=np.asarray(len(values), dtype=np.int64),
        metric=np.asarray(metric, dtype=np.str_),
        flag_direction=np.asarray(direction, dtype=np.str_),
        threshold=np.asarray(threshold, dtype=np.float64),
        scores=values,
        flags=flags,
    )


@pytest.mark.parametrize(
    ("scores", "threshold", "message"),
    [
        # An item time saved as the cutoff, as on the CLI's --threshold check.
        ([0.5, 1.0], 10.0, "effort threshold must be an RTE proportion between 0 and 1"),
        ([0.5, 1.0], 1.5, "effort threshold must be an RTE proportion between 0 and 1"),
        ([0.5, 1.0], -0.1, "effort threshold must be an RTE proportion between 0 and 1"),
        ([0.5, 7.5, np.nan], 0.9, "effort scores must be RTE proportions between 0 and 1"),
        ([-2.0, 0.5], 0.9, "effort scores must be RTE proportions between 0 and 1"),
    ],
)
def test_effort_archive_values_outside_the_rte_range_are_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    scores: list[float],
    threshold: float,
    message: str,
) -> None:
    archive = tmp_path / "effort.npz"
    flags = np.asarray(scores) < threshold

    with pytest.raises(ValueError, match=message):
        save_response_time_archive(archive, scores, flags, threshold=threshold, metric="effort")
    assert not archive.exists()

    # An archive written outside the saver fails to load or replay as well.
    _write_response_time_members(archive, "effort", scores, threshold)
    with pytest.raises(ValueError, match=message):
        load_response_time_archive(archive)
    assert message in _failure(["response-time-scores", str(archive)], capsys)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_effort_archive_accepts_the_rte_bounds(tmp_path: Path, threshold: float) -> None:
    archive = tmp_path / "effort.npz"
    scores = np.array([0.0, 1.0, np.nan])
    save_response_time_archive(
        archive, scores, scores < threshold, threshold=threshold, metric="effort"
    )

    loaded = load_response_time_archive(archive)

    assert loaded["threshold"] == threshold
    assert _json_values(loaded["scores"]) == [0.0, 1.0, None]
    assert loaded["flags"].tolist() == [threshold == 1.0, False, False]


@pytest.mark.parametrize(
    "metric", [name for name in get_args(ResponseTimeMetric) if name != "effort"]
)
def test_rte_range_applies_only_to_effort_archives(tmp_path: Path, metric: str) -> None:
    archive = tmp_path / "timing.npz"
    _write_response_time_members(archive, metric, [-2.0, 7.5, 12.0], 10.0)

    loaded = load_response_time_archive(archive)

    assert loaded["metric"] == metric
    assert loaded["threshold"] == 10.0
    assert loaded["scores"].tolist() == [-2.0, 7.5, 12.0]


@pytest.mark.parametrize(
    "metric", [name for name in get_args(ResponseTimeMetric) if name != "effort"]
)
def test_archives_from_previous_releases_load_unchanged(tmp_path: Path, metric: str) -> None:
    # The exact member layout written before effort joined the metric values.
    direction = "high" if metric == "mixture" else "low"
    scores = np.array([0.2, 0.5, 0.8, np.nan])
    flags = scores >= 0.5 if direction == "high" else scores <= 0.5
    archive = tmp_path / "previous.npz"
    np.savez(
        archive,
        schema_version=np.asarray(1, dtype=np.int64),
        result_type=np.asarray("response_time", dtype=np.str_),
        n_respondents=np.asarray(4, dtype=np.int64),
        metric=np.asarray(metric, dtype=np.str_),
        flag_direction=np.asarray(direction, dtype=np.str_),
        threshold=np.asarray(0.5, dtype=np.float64),
        scores=scores,
        flags=flags,
        respondent_ids=np.asarray(["a", "b", "c", "d"], dtype=np.str_),
    )

    loaded = load_response_time_archive(archive)
    exported = _json_output(["response-time-scores", str(archive), "--threshold", "0.5"])

    assert loaded["schema_version"] == 1
    assert loaded["metric"] == metric
    assert loaded["flags"].tolist() == flags.tolist()
    # Other metrics keep inclusive fixed cutoffs when reflagged.
    assert exported["flags"] == flags.tolist()


def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ier.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_effort_options_come_from_config(tmp_path: Path, timings: Path) -> None:
    config = _config(
        tmp_path,
        '[response-time]\nmetric = "effort"\neffort_fraction = 0.5\n'
        "effort_max_threshold = 1\nthreshold = 0.75\n",
    )

    payload = _json_output(["response-time", str(timings), *_ID, "--config", str(config)])

    scores, flags = response_time_effort_flag(
        _MATRIX, threshold=0.75, normative_fraction=0.5, max_threshold=1.0
    )
    assert payload["metric"] == "effort"
    assert payload["threshold"] == 0.75
    assert payload["scores"] == _json_values(scores)
    assert payload["flags"] == flags.tolist()


@pytest.mark.parametrize(
    ("configured", "argv", "keywords"),
    [
        (
            "effort_fraction = 0.5\neffort_max_threshold = 1\n",
            ["--effort-threshold", "15"],
            {"thresholds": 15.0},
        ),
        ("effort_threshold = 15\n", ["--effort-max-threshold", "1"], {"max_threshold": 1.0}),
        ("effort_threshold = 15\n", ["--effort-fraction", "0.5"], {"normative_fraction": 0.5}),
    ],
)
def test_command_line_effort_rule_displaces_configured_rule(
    tmp_path: Path,
    timings: Path,
    configured: str,
    argv: list[str],
    keywords: dict[str, Any],
) -> None:
    config = _config(tmp_path, f'[response-time]\nmetric = "effort"\n{configured}')

    payload = _json_output(["response-time", str(timings), *_ID, "--config", str(config), *argv])

    assert payload["scores"] == _json_values(response_time_effort(_MATRIX, **keywords))


@pytest.mark.parametrize(
    ("configured", "message"),
    [
        (
            "effort_threshold = 15\neffort_max_threshold = 1\n",
            "argument --effort-max-threshold: not allowed with argument --effort-threshold",
        ),
        ("effort_fraction = 2\n", "argument --effort-fraction: must be a number greater than 0"),
        # One shared threshold or cap is a single value, so an array is never
        # expanded into repeated options where the last element would win.
        (
            "effort_threshold = [1, 15]\n",
            "argument --effort-threshold: must be a positive finite number",
        ),
        (
            "effort-max-threshold = [100, 1]\n",
            "argument --effort-max-threshold: must be a positive finite number",
        ),
        (
            "effort_fraction = [0.5, 0.1]\n",
            "argument --effort-fraction: must be a number greater than 0",
        ),
    ],
)
def test_invalid_configured_effort_options_name_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], configured: str, message: str
) -> None:
    config = _config(tmp_path, f"[response-time]\n{configured}")

    error = _failure(["response-time", "unread.csv", "--config", str(config)], capsys)

    assert f"invalid option in [response-time] of {config}" in error
    assert message in error


def test_configured_effort_replay_uses_the_strict_rule(tmp_path: Path) -> None:
    archive = tmp_path / "effort.npz"
    save_response_time_archive(
        archive, [0.5, 0.9, 1.0], [True, False, False], threshold=0.9, metric="effort"
    )
    config = _config(tmp_path, "[response-time-scores]\nthreshold = 1\n")

    payload = _json_output(["response-time-scores", str(archive), "--config", str(config)])

    assert payload["threshold"] == 1.0
    assert payload["flags"] == [True, True, False]


def test_effort_defaults_match_the_python_api(capsys: pytest.CaptureFixture[str]) -> None:
    effort = inspect.signature(response_time_effort).parameters
    flag = inspect.signature(response_time_effort_flag).parameters
    subcommands = next(
        action
        for action in _build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    parser = subcommands.choices["response-time"]

    assert parser.get_default("effort_fraction") == effort["normative_fraction"].default
    assert parser.get_default("effort_max_threshold") == effort["max_threshold"].default
    assert parser.get_default("effort_threshold") == effort["thresholds"].default
    assert flag["threshold"].default == _EFFORT_CUTOFF

    with pytest.raises(SystemExit, match="0"):
        main(["response-time", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "{mean,median,sd,min,consistency,mixture,effort}" in help_text
    assert "--effort-fraction FRACTION | --effort-threshold TIME" in help_text
    assert "greater than 0 and at most 1 (default: 0.1)" in help_text
    assert "such as 10 seconds (default: None)" in help_text
    assert "(default for effort: 0.9)" in help_text


def _documented_effort_commands() -> list[str]:
    commands = []
    for path in [_ROOT / "README.md", *sorted((_ROOT / "docs").rglob("*.md"))]:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text.startswith("ier response-time") and (
                "--metric effort" in text or "effort.npz" in text
            ):
                commands.append(text)
    return commands


def test_documented_effort_commands_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    commands = _documented_effort_commands()
    assert len(commands) >= 4
    monkeypatch.chdir(tmp_path)
    # Documented commands read a timing matrix without an identifier column.
    numeric = "\n".join(line.partition(",")[2] for line in _TIMINGS.splitlines())
    Path("timings.csv").write_text(numeric + "\n", encoding="utf-8")
    # Create archives before the commands that replay them.
    for command in sorted(commands, key=lambda text: text.startswith("ier response-time-scores")):
        assert not command.endswith("\\"), command
        assert main(shlex.split(command)[1:]) == 0, command
    assert "error:" not in capsys.readouterr().err


def test_documented_effort_reflag_rule_matches_the_cli(tmp_path: Path, timings: Path) -> None:
    archive = tmp_path / "effort.npz"
    command = ["response-time", str(timings), *_ID, "--metric", "effort"]
    assert main([*command, "--format", "npz", "--output", str(archive)]) == 0
    saved = load_response_time_archive(archive)

    # The documented effort rules reproduce both timing commands' strict cutoffs.
    for cutoff in ("0.9", "1"):
        replayed = _json_output(["response-time-scores", str(archive), "--threshold", cutoff])
        rescored = _json_output([*command, "--threshold", cutoff])
        assert (saved["scores"] < float(cutoff)).tolist() == replayed["flags"]
        assert response_time_effort_flag(_MATRIX, float(cutoff))[1].tolist() == rescored["flags"]
    # response_time_score_flags() includes the respondent tied at 0.9, and its
    # default is a percentile rather than the 0.90 effort cutoff.
    tied = response_time_score_flags(saved["scores"], threshold=0.9, direction="low")
    assert tied.tolist() != saved["flags"].tolist()

    # Prose that sends retained timing scores to response_time_score_flags()
    # must also give the strict effort rule.
    qualified = 0
    for path in (_ROOT / "README.md", _ROOT / "docs" / "getting-started.md"):
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            paragraph = " ".join(block.split())
            if "response_time_score_flags()" in paragraph and "effort" in paragraph.lower():
                assert "includes ties at a fixed cutoff" in paragraph, path
                assert "`scores < cutoff`" in paragraph, path
                qualified += 1
    assert qualified >= 2
