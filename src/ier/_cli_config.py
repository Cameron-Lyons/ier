"""TOML configuration files that supply command-line option values."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
    from pathlib import Path

# Actions that a configuration file cannot set.
_UNCONFIGURABLE_DESTS = frozenset({"help", "config", argparse.SUPPRESS})


class _UsageError(Exception):
    """An argparse usage error raised instead of exiting the process."""


class _CliArgumentParser(argparse.ArgumentParser):
    """Argument parser that can raise usage errors for configuration-file context."""

    raise_usage_errors = False

    def error(self, message: str) -> NoReturn:
        if self.raise_usage_errors:
            raise _UsageError(message)
        super().error(message)


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return a parser's subcommand parsers by command name."""
    subcommands = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return dict(subcommands.choices)


def _config_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Map configuration keys to optional actions by destination and long option name.

    Option names win over destinations, so ``item_column`` selects the exact-name
    form of the ``item_columns`` destination while ``missing_values`` and
    ``missing_value`` both reach ``--missing-value``.
    """
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest in _UNCONFIGURABLE_DESTS:
            continue
        actions.setdefault(action.dest, action)
        for option in action.option_strings:
            negated = isinstance(action, argparse.BooleanOptionalAction) and option.startswith(
                "--no-"
            )
            if option.startswith("--") and not negated:
                actions[option[2:].replace("-", "_")] = action
    return actions


def _read_document(path: Path) -> dict[str, Any]:
    """Load one TOML document, reporting syntax errors with the file path."""
    # Only runs that read a configuration file pay for importing the parser.
    import tomllib  # noqa: PLC0415

    with path.open("rb") as handle:
        try:
            return tomllib.load(handle)
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as err:
            raise ValueError(f"invalid TOML in {path}: {err}") from err
        except RecursionError as err:
            # The parser recurses into each nested array and inline table.
            raise ValueError(
                f"invalid TOML in {path}: arrays or tables are nested too deeply"
            ) from err


def _config_table(
    document: dict[str, Any], command: str, commands: Collection[str], path: Path
) -> tuple[Mapping[str, Any], str]:
    """Select the command's section, or the whole document when it has none."""
    sections = {
        key for key, value in document.items() if key in commands and isinstance(value, dict)
    }
    if not sections:
        return document, f"{path} for 'ier {command}'"
    for key in document:
        if key not in sections:
            raise ValueError(
                f"unknown section '{key}' in {path}; a file with command sections "
                f"must place every option in one, such as [{command}]"
            )
    if command not in sections:
        raise ValueError(f"no [{command}] section in {path}")
    return document[command], f"[{command}] of {path}"


def _toml_type(value: object) -> str:
    """Name a TOML value's type for error messages."""
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, dict):
        return "a table"
    if isinstance(value, list):
        return "an array"
    return "a date or time"


def _scalar_token(value: object, key: str, where: str) -> str:
    """Return one string or number as an argument string."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(
            f"option '{key}' in {where} expects strings or numbers, not {_toml_type(value)}"
        )
    return str(value)


def _flag_value(value: object, key: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"option '{key}' in {where} must be true or false")
    return value


def _long_option(action: argparse.Action) -> str:
    """Return an action's first long option name, such as ``--scale-min``."""
    return next(name for name in action.option_strings if name.startswith("--"))


def _option_tokens(action: argparse.Action, key: str, value: object, where: str) -> list[str]:
    """Convert one configured value into the arguments its action accepts.

    Single-value options use ``--name=value`` so values beginning with a dash
    stay values. Arrays of a single-value option join with commas, the CLI's list
    syntax. Repeatable options take one value per array element, and a table
    becomes repeated ``INDEX=VALUE`` entries.
    """
    option = _long_option(action)
    if isinstance(action, argparse.BooleanOptionalAction):
        return [option if _flag_value(value, key, where) else f"--no-{option[2:]}"]
    if action.nargs == 0:
        return [option] if _flag_value(value, key, where) else []
    if action.nargs in ("+", "*"):
        items = value if isinstance(value, list) else [value]
        return [option, *(_scalar_token(item, key, where) for item in items)]
    if isinstance(action, argparse._StoreAction):
        if isinstance(value, list):
            text = ",".join(_scalar_token(item, key, where) for item in value)
        else:
            text = _scalar_token(value, key, where)
        return [f"{option}={text}"]
    if isinstance(value, dict):
        entries = [f"{name}={_scalar_token(item, key, where)}" for name, item in value.items()]
    elif isinstance(value, list):
        entries = [_scalar_token(item, key, where) for item in value]
    else:
        entries = [_scalar_token(value, key, where)]
    return [f"{option}={entry}" for entry in entries]


def _apply_config_file(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    build_parser: Callable[[], argparse.ArgumentParser],
    *,
    exclusive: Iterable[frozenset[str]] = (),
    value_checks: Mapping[str, Callable[[Any], object]] | None = None,
) -> None:
    """Fill options that the command line left unset from the ``--config`` file.

    Configured values pass through the command's own argparse actions, so types,
    choices, and mutually exclusive groups are validated as on the command line,
    with errors naming the file. ``value_checks`` maps destinations that the
    command would otherwise check only while loading data or later, such as item
    lists and ``INDEX=VALUE`` tables, to a parser that raises for invalid
    values, so configured values fail before any data is read and name their
    keys. Two keys that set the same option, such as ``scale-min`` and
    ``scale_min``, are rejected. Command-line options take precedence: a
    repeatable option given there replaces the configured list rather than
    extending it, and a configured option that conflicts with an explicit one,
    through an argparse group or an ``exclusive`` destination set, is ignored.
    """
    path: Path = args.config
    command: str = args.command
    parser = build_parser()
    commands = _subparsers(parser)
    subparser = commands[command]
    assert isinstance(subparser, _CliArgumentParser)
    actions = _config_actions(subparser)
    table, where = _config_table(_read_document(path), command, commands, path)
    tokens: list[str] = []
    first_keys: dict[argparse.Action, str] = {}
    keys_by_dest: dict[str, list[str]] = {}
    for key, value in table.items():
        action = actions.get(key.replace("-", "_"))
        if action is None:
            raise ValueError(f"unknown option '{key}' in {where}")
        # Spellings and destination names alias one option; TOML only rejects exact repeats.
        first = first_keys.setdefault(action, key)
        if first != key:
            raise ValueError(
                f"options '{first}' and '{key}' in {where} both set {_long_option(action)}"
            )
        keys_by_dest.setdefault(action.dest, []).append(key)
        tokens.extend(_option_tokens(action, key, value, where))

    # Without defaults, each parse returns only the options its source supplied.
    for action in subparser._actions:
        action.default = argparse.SUPPRESS
    given = set(vars(parser.parse_args(argv)))
    subparser.raise_usage_errors = True
    try:
        configured = vars(subparser.parse_args([*tokens, "--", str(args.data)]))
    except _UsageError as err:
        raise ValueError(f"invalid option in {where}: {err}") from err
    checks = value_checks or {}
    for dest, keys in keys_by_dest.items():
        check = checks.get(dest)
        # False switches and empty repeatable arrays supply no arguments to check.
        if check is None or dest not in configured:
            continue
        try:
            check(configured[dest])
        except (ValueError, argparse.ArgumentTypeError) as err:
            noun = "option" if len(keys) == 1 else "options"
            names = " and ".join(f"'{key}'" for key in keys)
            raise ValueError(f"invalid {noun} {names} in {where}: {err}") from err

    blocked = set(given)
    groups = [
        frozenset(action.dest for action in group._group_actions)
        for group in subparser._mutually_exclusive_groups
    ]
    for dests in (*groups, *exclusive):
        if dests & given:
            blocked |= dests
    configurable = {action.dest for action in actions.values()}
    for dest, value in configured.items():
        if dest in configurable and dest not in blocked:
            setattr(args, dest, value)
