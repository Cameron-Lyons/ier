"""Keep CLI options aligned with IndexOptions and exercise the newer survey options."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields, replace
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from ier import IndexOptions, guttman, infrequency, irv, longstring_scores
from ier.cli import (
    _CLI_EXCLUDED_FIELDS,
    _OPTION_CONVERTERS,
    _build_parser,
    _options_from_args,
    _parse_range_list,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path

_FIELD_NAMES = [field.name for field in fields(IndexOptions)]


def _subparsers() -> dict[str, argparse.ArgumentParser]:
    parser = _build_parser()
    action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return dict(action.choices)


def _group_titles(parser: argparse.ArgumentParser) -> list[str | None]:
    return [group.title for group in parser._action_groups]


@pytest.mark.parametrize("command", ["screen", "composite"])
def test_every_index_option_has_a_cli_destination_with_the_shared_default(command: str) -> None:
    parser = _subparsers()[command]
    destinations = {action.dest for action in parser._actions}
    missing = [
        name
        for name in _FIELD_NAMES
        if name not in _CLI_EXCLUDED_FIELDS and name not in destinations
    ]
    assert not missing, (
        f"IndexOptions fields without a same-named CLI destination: {missing}. Add an option "
        "whose dest matches each field (with a converter in _OPTION_CONVERTERS if its text "
        "needs parsing), or list the field in _CLI_EXCLUDED_FIELDS."
    )
    defaults = IndexOptions()
    for name in _FIELD_NAMES:
        if name not in _CLI_EXCLUDED_FIELDS:
            assert parser.get_default(name) == getattr(defaults, name), name


def test_converters_and_exclusions_name_index_options_fields() -> None:
    assert set(_OPTION_CONVERTERS) <= set(_FIELD_NAMES)
    assert set(_CLI_EXCLUDED_FIELDS) <= set(_FIELD_NAMES)
    assert not set(_OPTION_CONVERTERS) & _CLI_EXCLUDED_FIELDS


@pytest.mark.parametrize("command", sorted(_subparsers()))
def test_every_option_documents_itself(command: str) -> None:
    parser = _subparsers()[command]
    undocumented = [
        action.option_strings or action.dest
        for action in parser._actions
        if not action.help or action.help == argparse.SUPPRESS
    ]
    assert not undocumented


def test_scoring_commands_group_their_options() -> None:
    parsers = _subparsers()
    assert _group_titles(parsers["screen"])[2:] == [
        "input",
        "index options",
        "screening decision",
        "output",
    ]
    assert _group_titles(parsers["composite"])[2:] == [
        "input",
        "index options",
        "composite decision",
        "output",
    ]
    assert _group_titles(parsers["response-time"])[2:] == [
        "input",
        "timing options",
        "timing decision",
        "output",
    ]
    assert _group_titles(parsers["screen-scores"])[2:] == ["screening decision", "output"]
    assert _group_titles(parsers["composite-scores"])[2:] == ["composite decision", "output"]
    assert _group_titles(parsers["response-time-scores"])[2:] == ["timing decision", "output"]


def test_screen_help_lists_grouped_options_with_shared_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["screen", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    for title in ["input:", "index options:", "screening decision:", "output:"]:
        assert title in output.splitlines()
    help_text = " ".join(output.split())
    assert "--longstring-max-pattern-length N Longest repeating sub-pattern" in help_text
    assert "at least 2 (default: 5)" in help_text
    assert "antonym pairs (default: -0.6)" in help_text
    assert "--item-pattern GLOB" in help_text
    assert "--exclude-column NAME" in help_text


def test_default_arguments_build_default_index_options() -> None:
    args = _build_parser().parse_args(["screen", "responses.csv"])
    assert _options_from_args(args) == IndexOptions()


def test_full_flag_set_builds_matching_index_options() -> None:
    args = _build_parser().parse_args(
        [
            "composite",
            "responses.csv",
            "--scale-min",
            "1",
            "--scale-max",
            "7",
            "--no-na-rm",
            "--psychsyn-critval",
            "0.5",
            "--psychant-critval",
            "-0.4",
            "--evenodd-factors",
            "3,3",
            "--acquiescence-positive-items",
            "0,2",
            "--acquiescence-negative-items",
            "1,3",
            "--mad-positive-items",
            "0,1",
            "--mad-negative-items",
            "2,3",
            "--mad-scale-min",
            "0",
            "--mad-scale-max",
            "8",
            "--longstring-max-pattern-length",
            "3",
            "--midpoint-tolerance",
            "0.5",
            "--no-guttman-normalize",
            "--onset-window-size",
            "4",
            "--onset-min-items",
            "8",
            "--reliability-n-splits",
            "7",
            "--reliability-random-seed",
            "3",
            "--semantic-item-pairs",
            "0,1;2,3",
            "--infrequency-item-indices",
            "4,5",
            "--infrequency-acceptable-ranges",
            "1:2, :3",
            "--infrequency-proportion",
            "--infrequency-missing",
            "omit",
            "--missing-item-indices",
            "0,5",
            "--irv-num-split",
            "3",
            "--irv-split-points",
            "0,2,6",
        ]
    )
    mask = np.ones((2, 6), dtype=bool)
    expected = IndexOptions(
        na_rm=False,
        psychsyn_critval=0.5,
        psychant_critval=-0.4,
        evenodd_factors=[3, 3],
        mad_positive_items=[0, 1],
        mad_negative_items=[2, 3],
        mad_scale_max=8.0,
        scale_min=1.0,
        scale_max=7.0,
        acquiescence_positive_items=[0, 2],
        acquiescence_negative_items=[1, 3],
        longstring_max_pattern_length=3,
        midpoint_tolerance=0.5,
        guttman_normalize=False,
        onset_window_size=4,
        onset_min_items=8,
        reliability_n_splits=7,
        reliability_random_seed=3,
        semantic_item_pairs=[(0, 1), (2, 3)],
        infrequency_item_indices=[4, 5],
        infrequency_proportion=True,
        infrequency_missing="omit",
        mad_scale_min=0.0,
        missing_item_indices=[0, 5],
        infrequency_acceptable_ranges=[(1, 2), (-math.inf, 3)],
        irv_num_split=3,
        irv_split_points=[0, 2, 6],
    )
    options = _options_from_args(args, missing_applicable_mask=mask)
    assert options.missing_applicable_mask is mask
    # Dataclass equality on Python 3.13+ compares array fields elementwise.
    assert replace(options, missing_applicable_mask=None) == expected

    args = _build_parser().parse_args(
        ["screen", "responses.csv", "--infrequency-expected-responses", "5, 1.5"]
    )
    assert _options_from_args(args).infrequency_expected_responses == [5, 1.5]


def test_empty_acquiescence_lists_keep_their_option_specific_error() -> None:
    args = _build_parser().parse_args(
        ["screen", "responses.csv", "--acquiescence-negative-items", ","]
    )
    with pytest.raises(ValueError, match="--acquiescence-negative-items must include"):
        _options_from_args(args)


def test_acceptable_range_text_supports_open_sides_and_exact_integers() -> None:
    assert _parse_range_list(None) is None
    assert _parse_range_list(" , ") is None
    assert _parse_range_list("4:5,1:2,5:") == [(4, 5), (1, 2), (5, math.inf)]
    parsed = _parse_range_list(":-1.5, 9007199254740993:inf, :")
    assert parsed == [(-math.inf, -1.5), (2**53 + 1, math.inf), (-math.inf, math.inf)]
    assert parsed is not None
    assert isinstance(parsed[1][0], int)
    for invalid in ["4", "1:2:3", "a:5", "1:b"]:
        with pytest.raises(ValueError, match=f"invalid acceptable range '{invalid}'"):
            _parse_range_list(invalid)


def _write_responses(root: Path) -> tuple[Path, np.ndarray]:
    data = np.array(
        [
            [1, 1, 1, 1, 2, 5],
            [2, 3, 2, 3, 4, 4],
            [5, 4, 3, 2, 1, 1],
            [3, 3, 4, 4, 5, 7],
        ],
        dtype=float,
    )
    source = root / "responses.csv"
    rows = [",".join(f"q{column}" for column in range(6))]
    rows.extend(",".join(str(int(value)) for value in row) for row in data)
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return source, data


def _json_scores(arguments: list[str]) -> dict[str, list[float | None]]:
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        code = main([*arguments, "--format", "json"])
    assert code == 0
    scores: dict[str, list[float | None]] = json.loads(stdout.getvalue())["scores"]
    return scores


def test_cli_scores_attention_checks_with_acceptable_ranges(tmp_path: Path) -> None:
    source, data = _write_responses(tmp_path)
    scores = _json_scores(
        ["screen", str(source), "--indices", "infrequency"]
        + ["--infrequency-item-indices", "4,5", "--infrequency-acceptable-ranges", "4:5,:4"]
    )
    expected = infrequency(data, [4, 5], acceptable_ranges=[(4, 5), (-math.inf, 4)])
    assert scores["infrequency"] == expected.tolist() == [2.0, 0.0, 1.0, 1.0]


def test_cli_rejects_both_attention_answer_forms(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            ["screen", "responses.csv", "--infrequency-expected-responses", "5"]
            + ["--infrequency-acceptable-ranges", "4:5"]
        )
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_cli_reports_malformed_acceptable_ranges(tmp_path: Path) -> None:
    source, _ = _write_responses(tmp_path)
    stderr = StringIO()
    with patch("sys.stderr", stderr):
        code = main(
            ["screen", str(source), "--indices", "infrequency"]
            + ["--infrequency-item-indices", "4", "--infrequency-acceptable-ranges", "4-5"]
        )
    assert code == 1
    assert "invalid acceptable range '4-5'" in stderr.getvalue()


@pytest.mark.parametrize(
    ("split", "expected_kwargs"),
    [
        (["--irv-num-split", "2"], {"num_split": 2}),
        (["--irv-split-points", "0,4,6"], {"split_points": [0, 4, 6]}),
    ],
)
def test_cli_reaches_split_irv(
    tmp_path: Path, split: list[str], expected_kwargs: dict[str, object]
) -> None:
    source, data = _write_responses(tmp_path)
    scores = _json_scores(["screen", str(source), "--indices", "irv", *split])
    expected = irv(data, split=True, **expected_kwargs)  # type: ignore[arg-type]
    np.testing.assert_allclose(scores["irv"], expected, rtol=0, atol=1e-15)
    assert scores["irv"] != irv(data).tolist()


def test_cli_scores_average_run_lengths(tmp_path: Path) -> None:
    source, data = _write_responses(tmp_path)
    scores = _json_scores(["screen", str(source), "--indices", "avgstr", "longstring"])
    assert scores["avgstr"] == longstring_scores(data, avg=True).tolist() == [2.0, 1.2, 1.2, 1.5]


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--longstring-max-pattern-length", "1", "must be an integer of at least 2"),
        ("--longstring-max-pattern-length", "2.5", "must be an integer of at least 2"),
        ("--irv-num-split", "0", "must be an integer of at least 1"),
        ("--irv-num-split", "two", "must be an integer of at least 1"),
    ],
)
def test_cli_rejects_out_of_range_integer_options(
    option: str, value: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["screen", "responses.csv", option, value])
    assert error.value.code == 2
    assert f"argument {option}: {message}" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["screen", "composite"])
def test_every_index_option_help_shows_its_default(command: str) -> None:
    # Includes --missing-applicable-mask, whose file becomes IndexOptions' mask.
    actions = [action for action in _subparsers()[command]._actions if action.dest in _FIELD_NAMES]
    assert {action.dest for action in actions} == set(_FIELD_NAMES)
    hidden = [
        action.option_strings[0] for action in actions if "(default:" not in (action.help or "")
    ]
    assert not hidden, f"--help omits the default for {hidden}"


def test_screen_help_renders_none_defaults_for_optional_lists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["screen", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    for fragment in [
        "e.g. '5,5' (default: None)",
        "positively worded item indices (default: None)",
        "of 0-based item indices (default: None)",
        "for missing-rate scoring (default: None)",
        "columns in order (default: None)",
    ]:
        assert fragment in help_text


@pytest.mark.parametrize(
    ("option", "noun"),
    [("--irv-split-points", "split point"), ("--reliability-factors", "scale length")],
)
@pytest.mark.parametrize("value", ["", ",", " , "])
def test_explicitly_empty_scorer_lists_are_rejected(option: str, noun: str, value: str) -> None:
    # An empty value must not silently fall back to unsplit IRV or legacy reliability.
    args = _build_parser().parse_args(["screen", "responses.csv", option, value])
    with pytest.raises(ValueError, match=f"^{option} must include at least one {noun}$"):
        _options_from_args(args)


@pytest.mark.parametrize(
    ("option", "index"),
    [("--irv-split-points", "irv"), ("--reliability-factors", "individual_reliability")],
)
def test_cli_reports_explicitly_empty_scorer_lists(tmp_path: Path, option: str, index: str) -> None:
    source, _ = _write_responses(tmp_path)
    stdout, stderr = StringIO(), StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        code = main(["screen", str(source), "--indices", index, option, "", "--format", "json"])
    assert code == 1
    assert stdout.getvalue() == ""
    assert f"{option} must include at least one" in stderr.getvalue()


def test_guttman_normalize_help_names_the_na_rm_denominators(tmp_path: Path) -> None:
    help_text = next(
        action.help or ""
        for action in _subparsers()["screen"]._actions
        if action.dest == "guttman_normalize"
    )
    assert "answered pairs with --na-rm, all pairs with --no-na-rm" in help_text
    data = np.array(
        [[1, 1, 0, 0, np.nan], [1, 0, 1, 0, 0], [0, 1, np.nan, np.nan, 1], [1, 1, 1, 0, 0]]
    )
    source = tmp_path / "guttman.csv"
    rows = ["a,b,c,d,e"]
    rows.extend(",".join("" if np.isnan(v) else str(int(v)) for v in row) for row in data)
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    errors = guttman(data, normalize=False)
    answered = np.sum(~np.isnan(data), axis=1)
    documented = {
        "--na-rm": errors / (answered * (answered - 1) / 2),
        "--no-na-rm": errors / (5 * 4 / 2),
    }
    for flag, expected in documented.items():
        scores = _json_scores(["screen", str(source), "--indices", "guttman", flag])
        np.testing.assert_allclose(
            np.array(scores["guttman"], dtype=float), expected, rtol=0, atol=1e-15
        )
        assert scores["guttman"] == guttman(data, na_rm=flag == "--na-rm").tolist()


def test_infrequency_help_shows_the_spelling_for_leading_minus_values() -> None:
    helps = {action.dest: action.help or "" for action in _subparsers()["screen"]._actions}
    assert "--infrequency-acceptable-ranges=-3:-1" in helps["infrequency_acceptable_ranges"]
    assert "empty for an open end" in helps["infrequency_acceptable_ranges"]
    assert "':-1'" in helps["infrequency_acceptable_ranges"]
    assert "--infrequency-expected-responses=-1,2" in helps["infrequency_expected_responses"]


@pytest.mark.parametrize(
    ("arguments", "kwargs"),
    [
        (
            ["--infrequency-acceptable-ranges=-3:-1,1:"],
            {"acceptable_ranges": [(-3, -1), (1, math.inf)]},
        ),
        (
            ["--infrequency-acceptable-ranges=-inf:-1,:3"],
            {"acceptable_ranges": [(-math.inf, -1), (-math.inf, 3)]},
        ),
        (
            ["--infrequency-acceptable-ranges", ":-1,0:"],
            {"acceptable_ranges": [(-math.inf, -1), (0, math.inf)]},
        ),
        (["--infrequency-expected-responses=-1,2"], {"expected_responses": [-1, 2]}),
    ],
)
def test_documented_leading_minus_spellings_score_attention_checks(
    tmp_path: Path, arguments: list[str], kwargs: dict[str, object]
) -> None:
    data = np.array([[-3, 0, 1], [2, 1, -1], [-1, 3, 3], [0, 2, 2]], dtype=float)
    source = tmp_path / "bipolar.csv"
    rows = ["q1,q2,q3", *(",".join(str(int(v)) for v in row) for row in data)]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    scores = _json_scores(
        ["screen", str(source), "--indices", "infrequency", "--infrequency-item-indices", "0,2"]
        + arguments
    )
    expected = infrequency(data, [0, 2], **kwargs)  # type: ignore[arg-type]
    assert scores["infrequency"] == expected.tolist()
    assert 0 < sum(expected) < 2 * len(data)
