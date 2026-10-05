"""TOML configuration files supply option values that the command line can override."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from ier import save_response_time_archive, save_score_archive
from ier._cli_config import _config_actions, _subparsers
from ier.cli import _build_parser, main

if TYPE_CHECKING:
    import argparse

_SURVEY = (
    "participant,q1,q2,q3,q4,q5,q6,notes\n"
    "a,1,2,3,4,5,1,x\n"
    "b,3,3,3,3,3,3,y\n"
    "c,5,4,NA,2,1,5,z\n"
    "d,2,5,1,4,2,3,w\n"
    "e,4,4,4,4,4,4,v\n"
)
_CONFIGURABLE_COMMANDS = (
    "screen",
    "composite",
    "screen-scores",
    "composite-scores",
    "response-time",
    "response-time-scores",
)


@pytest.fixture
def survey(tmp_path: Path) -> Path:
    path = tmp_path / "survey.csv"
    path.write_text(_SURVEY, encoding="utf-8")
    return path


def _config(tmp_path: Path, text: str, name: str = "ier.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _parsed(argv: list[str]) -> argparse.Namespace:
    """Run main() and return the namespace its command handler receives."""
    received: list[argparse.Namespace] = []

    def handler(args: argparse.Namespace) -> int:
        received.append(args)
        return 0

    with patch.multiple(
        "ier.cli",
        _run_screen=handler,
        _run_composite=handler,
        _run_response_time=handler,
    ):
        assert main(argv) == 0
    return received[0]


def _failure(argv: list[str]) -> str:
    """Run a failing command and return its standard error."""
    stderr = StringIO()
    with patch("sys.stderr", stderr):
        assert main(argv) == 1
    return stderr.getvalue()


def _json_output(argv: list[str]) -> dict[str, Any]:
    stdout = StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", StringIO()):
        assert main(argv) == 0
    payload: dict[str, Any] = json.loads(stdout.getvalue())
    return payload


def test_flat_file_matches_the_equivalent_command_line(survey: Path, tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        """
indices = ["irv", "longstring", "midpoint"]
id-column = "participant"
exclude_column = ["notes"]
missing_value = ["NA"]
scale_min = 1
scale_max = 5
min_flags = 1
threshold = { irv = 1.2 }
index_percentile = { longstring = 80 }
format = "json"
""",
    )

    configured = _json_output(["screen", str(survey), "--config", str(config)])
    explicit = _json_output(
        [
            "screen",
            str(survey),
            "--indices",
            "irv",
            "longstring",
            "midpoint",
            "--id-column",
            "participant",
            "--exclude-column",
            "notes",
            "--missing-value",
            "NA",
            "--scale-min",
            "1",
            "--scale-max",
            "5",
            "--min-flags",
            "1",
            "--threshold",
            "irv=1.2",
            "--index-percentile",
            "longstring=80",
            "--format",
            "json",
        ]
    )

    assert configured == explicit
    assert configured["respondent_ids"] == ["a", "b", "c", "d", "e"]
    assert configured["threshold_sources"]["irv"] == "fixed"
    assert configured["percentiles"]["longstring"] == 80.0


def test_command_sections_apply_to_their_own_command(survey: Path, tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        """
[screen]
indices = ["irv"]
percentile = 90

[composite]
indices = ["irv", "longstring"]
method = "max"
standardize = false
""",
    )

    screen_args = _parsed(["screen", str(survey), "--config", str(config)])
    composite_args = _parsed(["composite", str(survey), "--config", str(config)])

    assert screen_args.indices == ["irv"]
    assert screen_args.percentile == 90.0
    assert composite_args.indices == ["irv", "longstring"]
    assert composite_args.method == "max"
    assert composite_args.standardize is False
    assert composite_args.percentile is None


@pytest.mark.parametrize(
    ("entry", "dest", "expected"),
    [
        ("scale_min = 1", "scale_min", 1.0),
        ("scale-max = 7.5", "scale_max", 7.5),
        ("top = 3", "top", 3),
        ('header = "present"', "header", "present"),
        ('delimiter = ";"', "delimiter", ";"),
        ('evenodd_method = "halves"', "evenodd_method", "halves"),
        ("acquiescence_positive_items = [0, 2]", "acquiescence_positive_items", "0,2"),
        ('semantic_item_pairs = "0,1;2,3"', "semantic_item_pairs", "0,1;2,3"),
        (
            'infrequency_acceptable_ranges = ["1:2", "4:"]',
            "infrequency_acceptable_ranges",
            "1:2,4:",
        ),
        ("strict = true", "strict", True),
        ("strict = false", "strict", False),
        ("na_rm = false", "na_rm", False),
        ("guttman-normalize = true", "guttman_normalize", True),
        ('indices = ["irv", "longstring"]', "indices", ["irv", "longstring"]),
        ('indices = "irv"', "indices", ["irv"]),
        ('missing_value = ["NA", "-99"]', "missing_values", ["NA", "-99"]),
        ('missing_values = "."', "missing_values", ["."]),
        ('exclude_column = "notes"', "exclude_columns", ["notes"]),
        ("threshold = { irv = 0.25, longstring = 8 }", "threshold", ["irv=0.25", "longstring=8"]),
        ('index_percentile = ["irv=90"]', "index_percentile", ["irv=90"]),
        ('item_columns = ["q1,q2", "q3"]', "item_columns", ["q1", "q2", "q3"]),
        ('item_column = ["Q1, agreement"]', "item_columns", ["Q1, agreement"]),
        ('item_pattern = ["q*"]', "item_patterns", ["q*"]),
        ('missing_applicable_mask = "mask.csv"', "missing_applicable_mask", Path("mask.csv")),
        ("compress_level = 9", "compress_level", 9),
    ],
)
def test_every_action_kind_is_converted(
    survey: Path, tmp_path: Path, entry: str, dest: str, expected: object
) -> None:
    config = _config(tmp_path, f"[screen]\n{entry}\n")

    args = _parsed(["screen", str(survey), "--config", str(config)])

    assert getattr(args, dest) == expected


def test_unset_options_keep_their_defaults(survey: Path, tmp_path: Path) -> None:
    config = _config(tmp_path, "indices = ['irv']\n")

    configured = vars(_parsed(["screen", str(survey), "--config", str(config)]))
    plain = vars(_parsed(["screen", str(survey), "--indices", "irv"]))

    assert configured.pop("config") == config
    assert plain.pop("config") is None
    del configured["handler"], plain["handler"]
    assert configured == plain


def test_command_line_values_override_configured_values(survey: Path, tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        """
[screen]
percentile = 99
min_flags = 3
threshold = { irv = 0.25 }
missing_value = ["NA"]
na_rm = false
""",
    )

    args = _parsed(
        [
            "screen",
            str(survey),
            "--config",
            str(config),
            "--percentile",
            "95",
            "--threshold",
            "longstring=8",
            "--na-rm",
        ]
    )

    assert args.percentile == 95.0
    assert args.min_flags == 3
    # A repeatable option on the command line replaces the configured list.
    assert args.threshold == ["longstring=8"]
    assert args.missing_values == ["NA"]
    assert args.na_rm is True


@pytest.mark.parametrize(
    ("command", "entry", "argv", "expected"),
    [
        ("composite", "threshold = 1.5", ["--percentile", "90"], {"threshold": None}),
        ("composite", "percentile = 90", ["--threshold", "1.5"], {"percentile": None}),
        ("response-time", "threshold = 1.5", ["--percentile", "10"], {"threshold": None}),
        (
            "screen",
            'infrequency_expected_responses = "5"',
            ["--infrequency-acceptable-ranges", "4:5"],
            {"infrequency_expected_responses": None},
        ),
        ("screen", 'item_columns = ["q1"]', ["--item-pattern", "q*"], {"item_columns": None}),
        ("screen", 'item_pattern = ["q*"]', ["--item-column", "q1"], {"item_patterns": None}),
    ],
)
def test_explicit_options_displace_conflicting_configured_options(
    survey: Path,
    tmp_path: Path,
    command: str,
    entry: str,
    argv: list[str],
    expected: dict[str, object],
) -> None:
    config = _config(tmp_path, f"[{command}]\n{entry}\n")

    args = _parsed([command, str(survey), "--config", str(config), *argv])

    for dest, value in expected.items():
        assert getattr(args, dest) == value


def test_configured_flags_combine_with_command_line_flags(survey: Path, tmp_path: Path) -> None:
    config = _config(tmp_path, "[composite]\ninclude_components = true\nweight = { irv = 2 }\n")

    args = _parsed(
        ["composite", str(survey), "--config", str(config), "--include-probability", "--strict"]
    )

    assert args.include_components is True
    assert args.include_probability is True
    assert args.strict is True
    assert args.weight == ["irv=2"]


def test_saved_score_and_timing_commands_accept_config(tmp_path: Path) -> None:
    scores = tmp_path / "scores.npz"
    save_score_archive(scores, {"irv": [0.1, 0.5, 0.9], "longstring": [1.0, 2.0, 6.0]})
    timing = tmp_path / "timing.npz"
    save_response_time_archive(timing, [0.5, 1.0, 2.0], [True, False, False], threshold=0.5)
    config = _config(
        tmp_path,
        """
[screen-scores]
indices = ["irv"]
min_flags = 1
threshold = { irv = 0.5 }
format = "json"

[composite-scores]
indices = ["irv", "longstring"]
weight = { irv = 2 }
include_components = true
format = "json"

[response-time-scores]
threshold = 1.0
format = "json"
""",
    )

    screened = _json_output(["screen-scores", str(scores), "--config", str(config)])
    combined = _json_output(["composite-scores", str(scores), "--config", str(config)])
    reflagged = _json_output(["response-time-scores", str(timing), "--config", str(config)])

    assert screened["indices_used"] == ["irv"]
    # Low IRV is suspicious, so the fixed cutoff flags the low tail inclusively.
    assert screened["flags"]["irv"] == [True, True, False]
    assert combined["weights"] == {"irv": 2.0}
    assert combined["indices_used"] == ["irv", "longstring"]
    assert reflagged["threshold"] == 1.0
    assert reflagged["flags"] == [True, True, False]


def test_response_time_options_come_from_config(tmp_path: Path) -> None:
    timings = tmp_path / "timings.csv"
    timings.write_text("t1,t2\n1.0,2.0\n0.2,0.3\n3.0,4.0\n", encoding="utf-8")
    config = _config(tmp_path, '[response-time]\nmetric = "mean"\nthreshold = 1.0\n')

    payload = _json_output(
        ["response-time", str(timings), "--config", str(config), "--format", "json"]
    )

    assert payload["metric"] == "mean"
    assert payload["threshold"] == 1.0
    assert payload["flags"] == [False, True, False]


def test_configured_output_settings_write_files(survey: Path, tmp_path: Path) -> None:
    destination = tmp_path / "screen.npz"
    config = _config(
        tmp_path,
        f"""
[screen]
indices = ["irv"]
missing_value = ["NA"]
format = "npz"
compress = true
output = {json.dumps(str(destination))}
""",
    )

    assert main(["screen", str(survey), "--config", str(config), "--item-pattern", "q*"]) == 0

    with np.load(destination, allow_pickle=False) as archive:
        assert archive["result_type"].item() == "screen"
        assert archive["score__irv"].shape == (5,)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[screen]\nscale_mn = 1\n", "unknown option 'scale_mn' in [screen] of {path}"),
        ("scale_mn = 1\n", "unknown option 'scale_mn' in {path} for 'ier screen'"),
        ("data = 'other.csv'\n", "unknown option 'data' in {path} for 'ier screen'"),
        ("config = 'other.toml'\n", "unknown option 'config' in {path} for 'ier screen'"),
        ("help = true\n", "unknown option 'help' in {path} for 'ier screen'"),
        ("no_na_rm = true\n", "unknown option 'no_na_rm' in {path} for 'ier screen'"),
        ("[composite]\nindices = ['irv']\n", "no [screen] section in {path}"),
        (
            "indices = ['irv']\n[composite]\nmethod = 'max'\n",
            "unknown section 'indices' in {path}; a file with command sections must place "
            "every option in one, such as [screen]",
        ),
    ],
)
def test_unknown_keys_and_sections_name_the_file(
    survey: Path, tmp_path: Path, text: str, message: str
) -> None:
    config = _config(tmp_path, text)

    errors = _failure(["screen", str(survey), "--config", str(config)])

    assert errors == f"error: {message.format(path=config)}\n"


@pytest.mark.parametrize(
    ("entry", "detail"),
    [
        ('scale_min = "abc"', "argument --scale-min: invalid float value: 'abc'"),
        ("top = 1.5", "argument --top: invalid int value: '1.5'"),
        ('header = "maybe"', "argument --header: invalid choice: "),
        ("compress_level = 10", "argument --compress-level: invalid choice: "),
        ("indices = []", "argument --indices: expected at least one argument"),
        (
            'item_columns = ["q1"]\nitem_pattern = ["q*"]',
            "argument --item-pattern: not allowed with argument --item-columns/--item-column",
        ),
        (
            'infrequency_expected_responses = "5"\ninfrequency_acceptable_ranges = "4:5"',
            "argument --infrequency-acceptable-ranges: not allowed with argument "
            "--infrequency-expected-responses",
        ),
    ],
)
def test_invalid_values_are_rejected_by_argparse_with_file_context(
    survey: Path, tmp_path: Path, entry: str, detail: str
) -> None:
    config = _config(tmp_path, f"[screen]\n{entry}\n")

    errors = _failure(["screen", str(survey), "--config", str(config)])

    assert errors.startswith(f"error: invalid option in [screen] of {config}: {detail}")


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ('strict = "yes"', "option 'strict' in [screen] of {path} must be true or false"),
        ("na_rm = 1", "option 'na_rm' in [screen] of {path} must be true or false"),
        (
            "scale_min = true",
            "option 'scale_min' in [screen] of {path} expects strings or numbers, not a boolean",
        ),
        (
            "threshold = { irv = [1] }",
            "option 'threshold' in [screen] of {path} expects strings or numbers, not an array",
        ),
        (
            "indices = [{ name = 'irv' }]",
            "option 'indices' in [screen] of {path} expects strings or numbers, not a table",
        ),
        (
            "delimiter = 2024-01-01",
            "option 'delimiter' in [screen] of {path} expects strings or numbers, "
            "not a date or time",
        ),
    ],
)
def test_values_without_an_argument_form_are_rejected(
    survey: Path, tmp_path: Path, entry: str, message: str
) -> None:
    config = _config(tmp_path, f"[screen]\n{entry}\n")

    errors = _failure(["screen", str(survey), "--config", str(config)])

    assert errors == f"error: {message.format(path=config)}\n"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "[screen]\nscale-min = 1\nscale_min = 0\n",
            "options 'scale-min' and 'scale_min' in [screen] of {path} both set --scale-min",
        ),
        (
            "scale_max = 5\nscale-max = 7\n",
            "options 'scale_max' and 'scale-max' in {path} for 'ier screen' both set --scale-max",
        ),
        (
            "[screen]\nmissing_value = ['NA']\nmissing_values = ['-99']\n",
            "options 'missing_value' and 'missing_values' in [screen] of {path} both set "
            "--missing-value",
        ),
        (
            "[screen]\nna-rm = true\nna_rm = false\n",
            "options 'na-rm' and 'na_rm' in [screen] of {path} both set --na-rm",
        ),
    ],
)
def test_keys_that_set_one_option_twice_are_rejected(
    survey: Path, tmp_path: Path, text: str, message: str
) -> None:
    config = _config(tmp_path, text)

    errors = _failure(["screen", str(survey), "--config", str(config)])

    assert errors == f"error: {message.format(path=config)}\n"


def test_exact_and_comma_separated_item_columns_combine(survey: Path, tmp_path: Path) -> None:
    # --item-columns and --item-column are distinct options that share one list.
    config = _config(tmp_path, '[screen]\nitem_columns = "q1,q2"\nitem_column = ["q3"]\n')

    args = _parsed(["screen", str(survey), "--config", str(config)])

    assert args.item_columns == ["q1", "q2", "q3"]


@pytest.mark.parametrize(
    ("command", "entry", "detail"),
    [
        (
            "screen",
            'acquiescence_positive_items = ["x"]',
            "option 'acquiescence_positive_items' in [screen] of {path}: invalid literal for "
            "int() with base 10: 'x'",
        ),
        (
            "screen",
            "acquiescence_negative_items = []",
            "option 'acquiescence_negative_items' in [screen] of {path}: "
            "--acquiescence-negative-items must include at least one item index",
        ),
        (
            "composite",
            'evenodd-factors = "5,five"',
            "option 'evenodd-factors' in [composite] of {path}: invalid literal for int() "
            "with base 10: 'five'",
        ),
        (
            "screen",
            'semantic_item_pairs = "0;1"',
            "option 'semantic_item_pairs' in [screen] of {path}: invalid pair '0'; expected "
            "'i,j' pairs separated by ';'",
        ),
        (
            "screen",
            'infrequency_acceptable_ranges = ["1-2"]',
            "option 'infrequency_acceptable_ranges' in [screen] of {path}: invalid acceptable "
            "range '1-2'; expected LOW:HIGH, with an empty side for an open end",
        ),
        (
            "screen",
            'infrequency_expected_responses = ["yes"]',
            "option 'infrequency_expected_responses' in [screen] of {path}: could not convert "
            "string to float: 'yes'",
        ),
        (
            "response-time",
            "skip_rows = -1",
            "option 'skip_rows' in [response-time] of {path}: must be an integer of at least 0",
        ),
        (
            "screen",
            'item_columns = ""',
            "option 'item_columns' in [screen] of {path}: --item-columns must include at least "
            "one column name",
        ),
        (
            "screen",
            'item_columns = ["q1"]\nitem_column = [" "]',
            "options 'item_columns' and 'item_column' in [screen] of {path}: --item-columns "
            "must include at least one column name",
        ),
        (
            "screen",
            'threshold = { irv = "high" }',
            "option 'threshold' in [screen] of {path}: invalid threshold value for irv: high",
        ),
        (
            "screen-scores",
            'threshold = { irv = "nan" }',
            "option 'threshold' in [screen-scores] of {path}: threshold for irv must be a "
            "finite number",
        ),
        (
            "screen-scores",
            'index_percentile = ["irv"]',
            "option 'index_percentile' in [screen-scores] of {path}: invalid percentile "
            "'irv'; expected INDEX=VALUE",
        ),
        (
            "composite",
            "weight = { irv = 0 }",
            "option 'weight' in [composite] of {path}: weight for irv must be a positive "
            "finite number",
        ),
        (
            "composite-scores",
            'weight = { irv = "heavy" }',
            "option 'weight' in [composite-scores] of {path}: invalid weight value for irv: heavy",
        ),
    ],
)
def test_values_the_cli_parses_itself_fail_with_file_context_before_reading_data(
    tmp_path: Path, command: str, entry: str, detail: str
) -> None:
    config = _config(tmp_path, f"[{command}]\n{entry}\n")
    # The data file does not exist, so a load attempt would report it instead.
    absent = tmp_path / "absent"

    errors = _failure([command, str(absent), "--config", str(config)])

    assert errors == f"error: invalid {detail.format(path=config)}\n"


def test_numeric_flag_thresholds_are_not_parsed_as_index_tables(
    survey: Path, tmp_path: Path
) -> None:
    config = _config(tmp_path, "[composite]\nthreshold = 1.5\n[response-time]\nthreshold = 2\n")

    composite_args = _parsed(["composite", str(survey), "--config", str(config)])
    timing_args = _parsed(["response-time", str(survey), "--config", str(config)])

    assert composite_args.threshold == 1.5
    assert timing_args.threshold == 2.0


def test_composite_flagging_conflicts_inside_one_file_are_rejected(
    survey: Path, tmp_path: Path
) -> None:
    config = _config(tmp_path, "[composite]\nthreshold = 1.5\npercentile = 90\n")

    errors = _failure(["composite", str(survey), "--config", str(config)])

    assert "invalid option in [composite] of" in errors
    assert "not allowed with argument --threshold" in errors


def test_invalid_toml_and_missing_files_are_reported(survey: Path, tmp_path: Path) -> None:
    broken = _config(tmp_path, "[screen\nindices = ['irv']\n")
    undecodable = tmp_path / "latin1.toml"
    undecodable.write_bytes(b"indices = ['\xe9']\n")
    missing = tmp_path / "missing.toml"

    broken_errors = _failure(["screen", str(survey), "--config", str(broken)])
    undecodable_errors = _failure(["screen", str(survey), "--config", str(undecodable)])
    missing_errors = _failure(["screen", str(survey), "--config", str(missing)])

    assert broken_errors.startswith(f"error: invalid TOML in {broken}: ")
    assert undecodable_errors.startswith(f"error: invalid TOML in {undecodable}: ")
    assert missing_errors.startswith("error: ")
    assert "missing.toml" in missing_errors


@pytest.mark.parametrize(
    "value",
    ["[" * 5000 + "]" * 5000, "{ a = " * 5000 + "1" + " }" * 5000],
    ids=["arrays", "inline-tables"],
)
def test_deeply_nested_toml_is_reported_as_invalid(
    survey: Path, tmp_path: Path, value: str
) -> None:
    config = _config(tmp_path, f"indices = {value}\n")

    errors = _failure(["screen", str(survey), "--config", str(config)])

    assert errors == f"error: invalid TOML in {config}: arrays or tables are nested too deeply\n"


def test_command_line_usage_errors_still_exit_through_argparse(
    survey: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path, "indices = ['irv']\n")

    with pytest.raises(SystemExit) as raised:
        main(["screen", str(survey), "--config", str(config), "--top", "many"])

    assert raised.value.code == 2
    assert "argument --top: invalid int value: 'many'" in capsys.readouterr().err


@pytest.mark.parametrize("command", _CONFIGURABLE_COMMANDS)
def test_every_command_line_option_has_a_configuration_key(command: str) -> None:
    subparser = _subparsers(_build_parser())[command]
    actions = _config_actions(subparser)

    for action in subparser._actions:
        if not action.option_strings or action.dest in {"help", "config"}:
            continue
        for option in action.option_strings:
            if option.startswith("--no-"):
                assert option[5:].replace("-", "_") in actions
                continue
            assert actions[option[2:].replace("-", "_")] is action
        assert action.dest in actions
    assert "config" not in actions


@pytest.mark.parametrize("command", ["inspect", "indices"])
def test_reporting_commands_do_not_take_config(command: str) -> None:
    subparser = _subparsers(_build_parser())[command]

    assert all("--config" not in action.option_strings for action in subparser._actions)
