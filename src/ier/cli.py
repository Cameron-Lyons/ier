"""Command-line interface for IER screening."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import fields
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

import ier
from ier import (
    IndexOptions,
    composite,
    composite_scores,
    composite_scores_summary,
    composite_summary,
    index_catalog,
    load_response_time_archive,
    load_score_archive,
    response_time,
    response_time_consistency,
    response_time_effort,
    response_time_mixture,
    screen,
    screen_scores,
)
from ier._cli_composite import CompositeReport, ResponseTimeReport
from ier._cli_config import _apply_config_file, _CliArgumentParser
from ier._cli_input import _input_label, _InputReport, _load_applicable_mask, _load_input
from ier._cli_inspect import _emit_inspection_text, _inspect_matrix, _write_inspection_json
from ier._cli_npz import (
    _require_npz_output_path,
    _write_composite_npz,
    _write_response_time_npz,
    _write_screen_npz,
)
from ier._cli_output import (
    _emit_composite_text,
    _emit_index_catalog_json,
    _emit_index_catalog_text,
    _emit_response_time_text,
    _emit_screen_text,
    _output_stream,
    _write_composite_csv,
    _write_composite_json,
    _write_index_catalog_csv,
    _write_output,
    _write_response_time_csv,
    _write_response_time_json,
    _write_screen_csv,
    _write_screen_json,
    _write_stream_output,
)
from ier._flagging import resolve_threshold, threshold_flags
from ier._registry import composite_index_names, validate_index_names, validate_worker_count
from ier._statistics import logistic_transform
from ier.types import ResponseTimeMetric

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TextIO


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    return [int(part) for part in parts]


def _parse_configured_int_list(
    raw: str | None, option: str, noun: str = "item index"
) -> list[int] | None:
    """Parse an optional integer list while rejecting an explicitly empty value."""
    parsed = _parse_int_list(raw)
    if raw is not None and parsed is None:
        raise ValueError(f"{option} must include at least one {noun}")
    return parsed


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    values: list[float] = []
    for part in parts:
        try:
            value: float = int(part)
        except ValueError:
            value = float(part)
        values.append(value)
    return values


def _parse_range_bound(raw: str, open_bound: float) -> float:
    """Parse one range endpoint, keeping integers exact and empty sides open."""
    text = raw.strip()
    if not text:
        return open_bound
    try:
        return int(text)
    except ValueError:
        return float(text)


def _parse_range_list(raw: str | None) -> list[tuple[float, float]] | None:
    """Parse comma-separated inclusive LOW:HIGH ranges with optional open sides."""
    if raw is None:
        return None
    ranges: list[tuple[float, float]] = []
    for raw_chunk in raw.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        message = (
            f"invalid acceptable range '{chunk}'; expected LOW:HIGH, with an empty side "
            "for an open end"
        )
        low, separator, high = chunk.partition(":")
        if not separator or ":" in high:
            raise ValueError(message)
        try:
            bounds = (_parse_range_bound(low, -math.inf), _parse_range_bound(high, math.inf))
        except ValueError as err:
            raise ValueError(message) from err
        ranges.append(bounds)
    return ranges or None


def _integer_at_least(minimum: int) -> Callable[[str], int]:
    """Return an argparse type for integers no smaller than ``minimum``."""

    message = f"must be an integer of at least {minimum}"

    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as err:
            raise argparse.ArgumentTypeError(message) from err
        if value < minimum:
            raise argparse.ArgumentTypeError(message)
        return value

    return parse


def _optional_integer_at_least(minimum: int) -> Callable[[str], int | None]:
    """Return an argparse type for ``none`` or integers no smaller than ``minimum``."""

    parse_integer = _integer_at_least(minimum)
    message = f"must be 'none' or an integer of at least {minimum}"

    def parse(raw: str) -> int | None:
        if raw.strip().lower() == "none":
            return None
        try:
            return parse_integer(raw)
        except argparse.ArgumentTypeError as err:
            raise argparse.ArgumentTypeError(message) from err

    return parse


def _positive_number(*, maximum: float = math.inf) -> Callable[[str], float]:
    """Return an argparse type for finite numbers above 0 and at most ``maximum``."""

    message = (
        "must be a positive finite number"
        if maximum == math.inf
        else f"must be a number greater than 0 and at most {maximum:g}"
    )

    def parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError as err:
            raise argparse.ArgumentTypeError(message) from err
        if not (math.isfinite(value) and 0.0 < value <= maximum):
            raise argparse.ArgumentTypeError(message)
        return value

    return parse


class _VersionAction(argparse.Action):
    """Print the package version, reading distribution metadata only when requested."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = argparse.SUPPRESS,
        default: str = argparse.SUPPRESS,
        help: str = "show program's version number and exit",
    ) -> None:
        super().__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0, help=help
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        print(f"{parser.prog} {ier.__version__}")
        parser.exit()


class _ItemColumnAction(argparse.Action):
    """Append exact names or comma-separated lists in command-line order."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        assert isinstance(values, str)
        if getattr(namespace, "item_patterns", None) is not None:
            raise argparse.ArgumentError(self, "not allowed with argument --item-pattern")
        previous: list[str] | None = getattr(namespace, self.dest, None)
        names = (
            [name.strip() for name in values.split(",") if name.strip()]
            if option_string == "--item-columns"
            else [values.strip()]
        )
        setattr(namespace, self.dest, [*(previous or []), *names])


class _ItemPatternAction(argparse.Action):
    """Append glob patterns, which replace rather than extend exact column names."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        assert isinstance(values, str)
        if getattr(namespace, "item_columns", None) is not None:
            raise argparse.ArgumentError(
                self, "not allowed with argument --item-columns/--item-column"
            )
        previous: list[str] | None = getattr(namespace, self.dest, None)
        setattr(namespace, self.dest, [*(previous or []), values])


class _EffortThresholdAction(argparse._StoreAction):
    """Store a shared effort threshold or a cap, rejecting the other one.

    The shared threshold already excludes ``--effort-fraction`` through an
    argparse group, and an option cannot join a second group, so the threshold
    and the cap check each other's value, which defaults to None. Deriving from
    the store action keeps each a single-value option for configuration files,
    which reject an array instead of repeating the option once per element.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        shared = self.dest == "effort_threshold"
        other = "effort_max_threshold" if shared else "effort_threshold"
        if getattr(namespace, other, None) is not None:
            option = "--effort-max-threshold" if shared else "--effort-threshold"
            raise argparse.ArgumentError(self, f"not allowed with argument {option}")
        setattr(namespace, self.dest, values)


# The item and effort-threshold actions above reject each other without an
# argparse group, so a configured value must also yield to the other kind on
# the command line.
_EXCLUSIVE_DESTINATIONS = (
    frozenset({"item_columns", "item_patterns"}),
    frozenset({"effort_threshold", "effort_max_threshold"}),
)

# Wise and Kong's screening rule flags response time effort strictly below 0.90,
# the default of response_time_effort_flag().
_EFFORT_CUTOFF = 0.90


def _parse_name_list(raw: list[str] | None) -> list[str] | None:
    """Validate names collected by the exact and shorthand column options."""
    if raw is None:
        return None
    names = [name.strip() for name in raw]
    if not names or any(not name for name in names):
        raise ValueError("--item-columns must include at least one column name")
    return names


def _parse_pair_list(raw: str | None) -> list[tuple[int, int]] | None:
    if raw is None:
        return None
    pairs: list[tuple[int, int]] = []
    for raw_chunk in raw.split(";"):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        left, sep, right = chunk.partition(",")
        if not sep:
            raise ValueError(f"invalid pair '{chunk}'; expected 'i,j' pairs separated by ';'")
        pairs.append((int(left.strip()), int(right.strip())))
    return pairs or None


def _parse_named_floats(
    raw: list[str] | None,
    noun: str,
    *,
    positive: bool = False,
) -> dict[str, float] | None:
    if raw is None:
        return None

    values: dict[str, float] = {}
    for entry in raw:
        name, separator, value = entry.partition("=")
        name = name.strip()
        if not separator or not name or not value.strip():
            raise ValueError(f"invalid {noun} '{entry}'; expected INDEX=VALUE")
        if name in values:
            raise ValueError(f"duplicate {noun} for index: {name}")
        try:
            number = float(value)
        except ValueError as err:
            raise ValueError(f"invalid {noun} value for {name}: {value.strip()}") from err
        if not np.isfinite(number) or (positive and number <= 0):
            requirement = "a positive finite number" if positive else "a finite number"
            raise ValueError(f"{noun} for {name} must be {requirement}")
        values[name] = number
    return values


def _parse_thresholds(raw: list[str] | None) -> dict[str, float] | None:
    return _parse_named_floats(raw, "threshold")


def _parse_percentiles(raw: list[str] | None) -> dict[str, float] | None:
    return _parse_named_floats(raw, "percentile")


def _parse_weights(raw: list[str] | None) -> dict[str, float] | None:
    return _parse_named_floats(raw, "weight", positive=True)


def _report_soft_errors(errors: dict[str, str]) -> None:
    """Report skipped indices without corrupting structured standard output."""
    for name, message in errors.items():
        print(f"warning: index '{name}' was skipped: {message}", file=sys.stderr)


# CLI defaults come from IndexOptions so the two cannot drift apart.
_INDEX_DEFAULTS = IndexOptions()

# IndexOptions fields whose same-named CLI value needs parsing before use.
_OPTION_CONVERTERS: dict[str, Callable[[Any], Any]] = {
    "evenodd_factors": _parse_int_list,
    "mad_positive_items": _parse_int_list,
    "mad_negative_items": _parse_int_list,
    "acquiescence_positive_items": partial(
        _parse_configured_int_list, option="--acquiescence-positive-items"
    ),
    "acquiescence_negative_items": partial(
        _parse_configured_int_list, option="--acquiescence-negative-items"
    ),
    "semantic_item_pairs": _parse_pair_list,
    "infrequency_item_indices": _parse_int_list,
    "infrequency_expected_responses": _parse_float_list,
    "missing_item_indices": _parse_int_list,
    "infrequency_acceptable_ranges": _parse_range_list,
    # An explicitly empty list must not silently select the unsplit or legacy scorer.
    "irv_split_points": partial(
        _parse_configured_int_list, option="--irv-split-points", noun="split point"
    ),
    "reliability_factors": partial(
        _parse_configured_int_list, option="--reliability-factors", noun="scale length"
    ),
    "reverse_keyed_items": partial(_parse_configured_int_list, option="--reverse-keyed-items"),
}

# IndexOptions fields that are not copied from a same-named CLI destination.
# The applicability mask option names a file that is loaded after the matrix.
_CLI_EXCLUDED_FIELDS = frozenset({"missing_applicable_mask"})

# Values that handlers check only while loading data or later; configuration files
# check them first so errors name the key. Only screening thresholds are INDEX=VALUE
# entries; composite and timing thresholds are numbers that argparse already parsed.
_CONFIG_VALUE_CHECKS: dict[str, Callable[[Any], object]] = {
    **_OPTION_CONVERTERS,
    "item_columns": _parse_name_list,
    "skip_rows": _integer_at_least(0),
    "threshold": lambda raw: _parse_thresholds(raw) if isinstance(raw, list) else raw,
    "index_percentile": _parse_percentiles,
    "weight": _parse_weights,
}


def _options_from_args(
    args: argparse.Namespace, *, missing_applicable_mask: np.ndarray | None = None
) -> IndexOptions:
    """Build IndexOptions from same-named CLI destinations, parsing list values."""
    values: dict[str, Any] = {}
    for field in fields(IndexOptions):
        if field.name in _CLI_EXCLUDED_FIELDS:
            continue
        value = getattr(args, field.name)
        convert = _OPTION_CONVERTERS.get(field.name)
        values[field.name] = value if convert is None else convert(value)
    return IndexOptions(**values, missing_applicable_mask=missing_applicable_mask)


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    """Add the TOML option file accepted by scoring and saved-score commands."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "TOML file of option values, flat or in [command] sections; options given "
            "on the command line take precedence"
        ),
    )


def _add_matrix_input_options(parser: argparse.ArgumentParser) -> None:
    """Add matrix selection options shared by scoring commands."""
    group = parser.add_argument_group("input")
    group.add_argument(
        "--delimiter",
        default=None,
        help="Input delimiter (auto-detect comma, tab, semicolon, or whitespace if omitted)",
    )
    group.add_argument(
        "--header",
        choices=["auto", "present", "absent"],
        default="auto",
        help="Delimited input header mode (default: auto)",
    )
    group.add_argument(
        "--missing-value",
        action="append",
        dest="missing_values",
        default=None,
        metavar="TOKEN",
        help="Delimited missing-value token; repeat for multiple exact tokens",
    )
    group.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        metavar="N",
        help="Physical input lines to discard before delimiter and header detection",
    )
    group.add_argument(
        "--id-column",
        default=None,
        metavar="NAME",
        help="Named header column to preserve as respondent identifiers",
    )
    group.add_argument(
        "--item-columns",
        action=_ItemColumnAction,
        default=None,
        metavar="NAME[,NAME...]",
        help="Named header columns to score, in order; comma-separate or repeat",
    )
    group.add_argument(
        "--item-column",
        action=_ItemColumnAction,
        dest="item_columns",
        default=None,
        metavar="NAME",
        help="One exact header name to score, including commas; repeat to select in order",
    )
    group.add_argument(
        "--item-pattern",
        action=_ItemPatternAction,
        dest="item_patterns",
        default=None,
        metavar="GLOB",
        help=(
            "Score header columns matching a case-sensitive glob such as 'Q*', in header "
            "order; repeat to combine patterns (not with --item-columns)"
        ),
    )
    group.add_argument(
        "--exclude-column",
        action="append",
        dest="exclude_columns",
        default=None,
        metavar="NAME",
        help=(
            "Exact header name to leave unscored, after any item selection; without one, "
            "score every other column except --id-column; repeat for more"
        ),
    )


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output controls shared by scoring commands."""
    group = parser.add_argument_group("output")
    group.add_argument(
        "--format",
        choices=["text", "json", "csv", "npz"],
        default="text",
        help="Output format (default: text summary)",
    )
    group.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write to a path, optionally .gz, .bz2, or .xz; use '-' for stdout; "
            "NPZ requires a .npz path"
        ),
    )
    group.add_argument(
        "--compress",
        action="store_true",
        help="Compress NPZ members to reduce file size (requires --format npz)",
    )
    group.add_argument(
        "--compress-level",
        type=int,
        choices=range(1, 10),
        default=None,
        metavar="{1..9}",
        help=(
            "DEFLATE level for --compress, from 1 (fastest, default) to 9 (smallest); "
            "requires --compress"
        ),
    )
    group.add_argument(
        "--top",
        type=int,
        default=10,
        help="For text format: show the top N respondents (default: 10)",
    )


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    """Add matrix input options and the IndexOptions-backed scoring options."""
    _add_matrix_input_options(parser)
    group = parser.add_argument_group("index options")
    group.add_argument(
        "--scale-min",
        type=float,
        default=_INDEX_DEFAULTS.scale_min,
        help=(
            "Lowest response-scale value; inferred from the data if omitted (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--scale-max",
        type=float,
        default=_INDEX_DEFAULTS.scale_max,
        help=(
            "Highest response-scale value; inferred from the data if omitted (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--reverse-keyed-items",
        default=_INDEX_DEFAULTS.reverse_keyed_items,
        metavar="ITEMS",
        help=(
            "Comma-separated 0-based reverse-worded item indices, reverse-scored with the "
            "scale bounds only for indices that use keyed responses, such as evenodd and "
            "guttman (see 'ier indices'); other indices read responses as presented "
            "(default: %(default)s)"
        ),
    )
    group.add_argument(
        "--na-rm",
        action=argparse.BooleanOptionalAction,
        default=_INDEX_DEFAULTS.na_rm,
        help="Drop incomplete rows / pairwise NaNs where supported (default: %(default)s)",
    )
    group.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested index cannot be computed",
    )
    group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Score independent indices concurrently (default: 1)",
    )
    group.add_argument(
        "--psychsyn-critval",
        type=float,
        default=_INDEX_DEFAULTS.psychsyn_critval,
        help="Minimum item correlation for psychometric synonym pairs (default: %(default)s)",
    )
    group.add_argument(
        "--psychant-critval",
        type=float,
        default=_INDEX_DEFAULTS.psychant_critval,
        help="Maximum item correlation for psychometric antonym pairs (default: %(default)s)",
    )
    group.add_argument(
        "--psychsyn-item-correlations",
        choices=["complete", "pairwise"],
        default=_INDEX_DEFAULTS.psychsyn_item_correlations,
        help=(
            "Missing-data policy for psychsyn/psychant item pairs: 'pairwise' correlates "
            "items over shared responses, like careless (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--evenodd-factors",
        default=_INDEX_DEFAULTS.evenodd_factors,
        help="Comma-separated factor lengths, e.g. '5,5' (default: %(default)s)",
    )
    group.add_argument(
        "--acquiescence-positive-items",
        default=_INDEX_DEFAULTS.acquiescence_positive_items,
        help="Comma-separated 0-based positively worded item indices (default: %(default)s)",
    )
    group.add_argument(
        "--acquiescence-negative-items",
        default=_INDEX_DEFAULTS.acquiescence_negative_items,
        help="Comma-separated 0-based negatively worded item indices (default: %(default)s)",
    )
    group.add_argument(
        "--mad-positive-items",
        default=_INDEX_DEFAULTS.mad_positive_items,
        help="Comma-separated item indices (default: %(default)s)",
    )
    group.add_argument(
        "--mad-negative-items",
        default=_INDEX_DEFAULTS.mad_negative_items,
        help="Comma-separated item indices (default: %(default)s)",
    )
    group.add_argument(
        "--mad-scale-min",
        type=float,
        default=_INDEX_DEFAULTS.mad_scale_min,
        help=(
            "Lowest scale value for reverse-scoring MAD pairs; inferred from the data if "
            "omitted (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--mad-scale-max",
        type=float,
        default=_INDEX_DEFAULTS.mad_scale_max,
        help=(
            "Highest scale value for reverse-scoring MAD pairs; inferred from the data if "
            "omitted (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--longstring-max-pattern-length",
        type=_integer_at_least(2),
        default=_INDEX_DEFAULTS.longstring_max_pattern_length,
        metavar="N",
        help=(
            "Longest repeating sub-pattern for longstring_pattern, at least 2 "
            "(default: %(default)s)"
        ),
    )
    group.add_argument(
        "--midpoint-tolerance",
        type=float,
        default=_INDEX_DEFAULTS.midpoint_tolerance,
        help=(
            "Distance from the scale midpoint counted as midpoint responding (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--guttman-normalize",
        action=argparse.BooleanOptionalAction,
        default=_INDEX_DEFAULTS.guttman_normalize,
        help=(
            "Report Guttman errors as a proportion of item pairs (answered pairs with "
            "--na-rm, all pairs with --no-na-rm) instead of raw counts (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--onset-window-size",
        type=int,
        default=_INDEX_DEFAULTS.onset_window_size,
        help="Running-window length for onset changepoint detection (default: %(default)s)",
    )
    group.add_argument(
        "--onset-min-items",
        type=int,
        default=_INDEX_DEFAULTS.onset_min_items,
        help="Minimum observed responses required for onset detection (default: %(default)s)",
    )
    group.add_argument(
        "--reliability-n-splits",
        type=int,
        default=_INDEX_DEFAULTS.reliability_n_splits,
        help="Random split halves averaged by individual_reliability (default: %(default)s)",
    )
    group.add_argument(
        "--reliability-random-seed",
        type=int,
        default=_INDEX_DEFAULTS.reliability_random_seed,
        help="Seed for reproducible individual_reliability splits (default: %(default)s)",
    )
    group.add_argument(
        "--evenodd-method",
        choices=["item_pairs", "halves"],
        default=_INDEX_DEFAULTS.evenodd_method,
        help=(
            "Even-odd algorithm: 'halves' correlates per-factor odd/even half means "
            "across factors (careless::evenodd, recommended); 'item_pairs' pairs items "
            "within each factor (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--reliability-factors",
        default=_INDEX_DEFAULTS.reliability_factors,
        help=(
            "Comma-separated scale lengths for scale-aware split-half reliability, "
            "e.g. '8,8,8' (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--semantic-item-pairs",
        default=_INDEX_DEFAULTS.semantic_item_pairs,
        help="Pairs as 'i,j;i,j' of 0-based item indices (default: %(default)s)",
    )
    group.add_argument(
        "--infrequency-item-indices",
        default=_INDEX_DEFAULTS.infrequency_item_indices,
        help="Comma-separated 0-based attention-check item indices (default: %(default)s)",
    )
    infrequency_answers = group.add_mutually_exclusive_group()
    infrequency_answers.add_argument(
        "--infrequency-expected-responses",
        default=_INDEX_DEFAULTS.infrequency_expected_responses,
        help=(
            "Comma-separated correct response for each attention-check item; write a list "
            "starting with a negative value as --infrequency-expected-responses=-1,2 "
            "(default: %(default)s)"
        ),
    )
    infrequency_answers.add_argument(
        "--infrequency-acceptable-ranges",
        default=_INDEX_DEFAULTS.infrequency_acceptable_ranges,
        metavar="RANGES",
        help=(
            "Inclusive LOW:HIGH correct-response range for each attention-check item; "
            "leave a side empty for an open end, e.g. '1:2,4:' or ':-1'. Write ranges "
            "starting with '-' as --infrequency-acceptable-ranges=-3:-1 "
            "(default: %(default)s)"
        ),
    )
    group.add_argument(
        "--infrequency-proportion",
        action=argparse.BooleanOptionalAction,
        default=_INDEX_DEFAULTS.infrequency_proportion,
        help=(
            "Score the share of failed attention checks instead of the count (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--infrequency-missing",
        choices=["pass", "fail", "omit", "propagate"],
        default=_INDEX_DEFAULTS.infrequency_missing,
        help="Missing attention-check policy (default: %(default)s)",
    )
    group.add_argument(
        "--missing-item-indices",
        default=_INDEX_DEFAULTS.missing_item_indices,
        help=(
            "Comma-separated required item indices for missing-rate scoring (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--missing-applicable-mask",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Boolean .npy or headerless 0/1 text mask for missing-rate scoring; "
            "must match selected response rows and columns in order (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--irv-num-split",
        type=_integer_at_least(1),
        default=_INDEX_DEFAULTS.irv_num_split,
        metavar="N",
        help="Average IRV across N consecutive item sections (default: %(default)s)",
    )
    group.add_argument(
        "--irv-split-points",
        default=_INDEX_DEFAULTS.irv_split_points,
        metavar="POINTS",
        help=(
            "Comma-separated IRV section boundaries from 0 to the item count, e.g. "
            "'0,10,20'; overrides --irv-num-split (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--autocorrelation-max-lag",
        type=_optional_integer_at_least(1),
        default=_INDEX_DEFAULTS.autocorrelation_max_lag,
        metavar="N",
        help=(
            "Largest lag for the autocorrelation index, at least 1, or 'none' for each "
            "respondent's observed responses minus 3 (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--autocorrelation-statistic",
        choices=["max_abs", "sum_abs"],
        default=_INDEX_DEFAULTS.autocorrelation_statistic,
        help=(
            "Score the largest or the summed absolute lag autocorrelation (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--person-fit-ncat",
        type=_integer_at_least(2),
        default=_INDEX_DEFAULTS.person_fit_ncat,
        metavar="N",
        help=(
            "Response categories per item for the gpoly and u3poly_fit person-fit indices, "
            "at least 2; inferred from --scale-min/--scale-max or the data when omitted "
            "(default: %(default)s)"
        ),
    )


def _add_screen_decision_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("screening decision")
    group.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help=(
            "Tail percentile for per-index flags; low-direction indices use 100 minus "
            "this value (default: %(default)s)"
        ),
    )
    group.add_argument(
        "--threshold",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Fixed per-index cutoff; repeat for multiple indices",
    )
    group.add_argument(
        "--index-percentile",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Per-index tail percentile; repeat for multiple indices",
    )
    group.add_argument(
        "--min-flags",
        type=int,
        default=2,
        help="Minimum per-index flags for a consensus flag (default: 2)",
    )
    group.add_argument(
        "--min-valid-indices",
        type=int,
        default=None,
        help="Minimum available index scores required for consensus eligibility",
    )


def _add_composite_decision_options(
    parser: argparse.ArgumentParser, *, precomputed: bool = False
) -> None:
    group = parser.add_argument_group("composite decision")
    group.add_argument(
        "--method",
        choices=["mean", "sum", "max"] if precomputed else ["mean", "sum", "max", "best_subset"],
        default="mean",
        help="How to combine directed index scores (default: %(default)s)",
    )
    group.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize each directed component before combining (default: true)",
    )
    composite_flagging = group.add_mutually_exclusive_group()
    composite_flagging.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Flag scores at or above a fixed cutoff",
    )
    composite_flagging.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Flag scores strictly above a sample percentile",
    )
    group.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Positive index weight override; repeat for multiple indices",
    )
    group.add_argument(
        "--min-valid-indices",
        type=int,
        default=None,
        metavar="N",
        help="Require at least N available component scores per respondent",
    )
    group.add_argument(
        "--include-components",
        action="store_true",
        help="Include raw component scores and per-respondent availability counts",
    )
    group.add_argument(
        "--include-probability",
        action="store_true",
        help="Include uncalibrated logistic composite values alongside scores",
    )


def _add_response_time_decision_options(
    parser: argparse.ArgumentParser, *, precomputed: bool = False
) -> None:
    flagging = parser.add_argument_group("timing decision").add_mutually_exclusive_group()
    flagging.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Fixed flag cutoff in score units; inclusive, except that effort flags RTE "
            "strictly below it"
            + ("" if precomputed else f" (default for effort: {_EFFORT_CUTOFF:g})")
        ),
    )
    flagging.add_argument(
        "--percentile",
        type=float,
        default=None,
        help=(
            "Percentile cutoff (default: preserve saved decisions)"
            if precomputed
            else "Percentile cutoff (default: 5 for low scores, 95 for mixture, none for effort)"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = _CliArgumentParser(
        prog="ier",
        description="Detect insufficient effort / careless responding in survey matrices.",
    )
    parser.add_argument("--version", action=_VersionAction)

    sub = parser.add_subparsers(dest="command", required=True)

    screen_parser = sub.add_parser("screen", help="Run multi-index screening on a matrix.")
    screen_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace or .npy item scores; use '-' for standard input",
    )
    _add_config_option(screen_parser)
    screen_parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index names to compute (default: package screen defaults)",
    )
    _add_shared_options(screen_parser)
    _add_screen_decision_options(screen_parser)
    _add_output_options(screen_parser)
    screen_parser.set_defaults(handler=_run_screen)

    composite_parser = sub.add_parser(
        "composite", help="Compute a composite IER score for each respondent."
    )
    composite_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace or .npy item scores; use '-' for standard input",
    )
    _add_config_option(composite_parser)
    composite_parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index names to include (default: package composite defaults)",
    )
    _add_shared_options(composite_parser)
    _add_composite_decision_options(composite_parser)
    _add_output_options(composite_parser)
    composite_parser.set_defaults(handler=_run_composite)

    for command, help_text in (
        ("screen-scores", "Apply screening decisions to a saved score archive."),
        ("composite-scores", "Combine saved component scores without rescoring items."),
    ):
        saved_parser = sub.add_parser(command, help=help_text)
        saved_parser.add_argument("data", type=Path, help="Reusable .npz score archive")
        _add_config_option(saved_parser)
        saved_parser.add_argument(
            "--indices",
            nargs="+",
            default=None,
            help="Saved score names to select in order (default: all saved scores)",
        )
        saved_parser.add_argument(
            "--strict", action="store_true", help="Fail if the archive retains index failures"
        )
        if command == "screen-scores":
            _add_screen_decision_options(saved_parser)
            saved_parser.set_defaults(handler=_run_screen)
        else:
            saved_parser.add_argument(
                "--skip-unsupported",
                action="store_true",
                help="Omit saved scores for indices that are not composite-enabled",
            )
            _add_composite_decision_options(saved_parser, precomputed=True)
            saved_parser.set_defaults(handler=_run_composite)
        _add_output_options(saved_parser)

    response_time_parser = sub.add_parser(
        "response-time",
        help="Score and flag a response-time matrix.",
    )
    response_time_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace or .npy timing values; use '-' for standard input",
    )
    _add_config_option(response_time_parser)
    _add_matrix_input_options(response_time_parser)
    timing_options = response_time_parser.add_argument_group("timing options")
    timing_options.add_argument(
        "--metric",
        choices=list(get_args(ResponseTimeMetric)),
        default="median",
        help="Timing score to compute (default: median)",
    )
    timing_options.add_argument(
        "--components",
        type=int,
        default=2,
        help="Gaussian components for the mixture metric (default: 2)",
    )
    timing_options.add_argument(
        "--log-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log-transform median times for the mixture metric (default: true)",
    )
    timing_options.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for reproducible mixture-model initialization (default: %(default)s)",
    )
    effort_rule = timing_options.add_mutually_exclusive_group()
    effort_rule.add_argument(
        "--effort-fraction",
        type=_positive_number(maximum=1.0),
        default=0.10,
        metavar="FRACTION",
        help=(
            "Share of each item's mean time used as its normative threshold for the "
            "effort metric, greater than 0 and at most 1 (default: %(default)s)"
        ),
    )
    effort_rule.add_argument(
        "--effort-threshold",
        action=_EffortThresholdAction,
        type=_positive_number(),
        default=None,
        metavar="TIME",
        help=(
            "One item threshold shared by every item for the effort metric, in timing "
            "units; replaces normative thresholds (default: %(default)s)"
        ),
    )
    timing_options.add_argument(
        "--effort-max-threshold",
        action=_EffortThresholdAction,
        type=_positive_number(),
        default=None,
        metavar="TIME",
        help=(
            "Cap on every normative item threshold for the effort metric, in timing "
            "units, such as 10 seconds (default: %(default)s)"
        ),
    )
    _add_response_time_decision_options(response_time_parser)
    _add_output_options(response_time_parser)
    response_time_parser.set_defaults(handler=_run_response_time)

    timing_saved_parser = sub.add_parser(
        "response-time-scores",
        help="Reflag or export a saved timing archive without recomputing scores.",
    )
    timing_saved_parser.add_argument("data", type=Path, help="Response-time .npz archive")
    _add_config_option(timing_saved_parser)
    _add_response_time_decision_options(timing_saved_parser, precomputed=True)
    _add_output_options(timing_saved_parser)
    timing_saved_parser.set_defaults(handler=_run_response_time)

    inspect_parser = sub.add_parser(
        "inspect",
        help="Show how a matrix is parsed, its missing cells, and its inferred scale.",
    )
    inspect_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace or .npy item scores; use '-' for standard input",
    )
    _add_matrix_input_options(inspect_parser)
    inspect_output = inspect_parser.add_argument_group("output")
    inspect_output.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    inspect_output.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write output to a path, optionally .gz, .bz2, or .xz; use '-' for stdout",
    )
    inspect_parser.set_defaults(handler=_run_inspect)

    indices_parser = sub.add_parser(
        "indices", help="List registered indices and orchestration metadata."
    )
    indices_parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    indices_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write output to a path, optionally .gz, .bz2, or .xz; use '-' for stdout",
    )
    indices_parser.set_defaults(handler=_run_index_catalog)

    return parser


def _score_response_times(
    matrix: np.ndarray,
    metric: str,
    components: int,
    log_transform: bool,
    random_seed: int | None,
    *,
    effort_fraction: float,
    effort_threshold: float | None,
    effort_max_threshold: float | None,
) -> tuple[np.ndarray, Literal["high", "low"]]:
    """Compute one timing metric and its suspicious-score direction."""
    if metric == "consistency":
        return response_time_consistency(matrix), "low"
    if metric == "effort":
        return (
            response_time_effort(
                matrix,
                effort_threshold,
                normative_fraction=effort_fraction,
                max_threshold=effort_max_threshold,
            ),
            "low",
        )
    if metric == "mixture":
        return (
            response_time_mixture(
                matrix,
                n_components=components,
                log_transform=log_transform,
                random_seed=random_seed,
            ),
            "high",
        )
    return response_time(matrix, metric=metric), "low"


def _load_reusable_scores(
    path: Path, indices: list[str] | None, strict: bool
) -> tuple[dict[str, np.ndarray], dict[str, str], list[str] | None, int]:
    """Select saved raw scores while retaining relevant archive provenance."""
    archive = load_score_archive(path)
    scores = archive["scores"]
    errors = archive["errors"]
    if indices is not None:
        validate_index_names(indices)
        for name in indices:
            if name not in scores:
                detail = errors.get(name, "the index was not saved in this archive")
                raise ValueError(f"index '{name}' has no saved scores: {detail}")
        scores = {name: scores[name] for name in indices}
        errors = {}
    if strict and errors:
        name, message = next(iter(errors.items()))
        raise ValueError(f"archive index '{name}' failed: {message}")
    return scores, errors, archive["respondent_ids"], archive["n_respondents"]


def _check_output_options(args: argparse.Namespace) -> None:
    """Reject output option combinations before any input is read or scored."""
    if args.compress and args.format != "npz":
        raise ValueError("--compress requires --format npz")
    if args.compress_level is not None and not args.compress:
        raise ValueError("--compress-level requires --compress")
    if args.format == "npz":
        _require_npz_output_path(args.output)


def _emit_result(
    args: argparse.Namespace,
    *,
    text: Callable[[], str],
    json: Callable[[TextIO], object],
    csv: Callable[[TextIO], object] | None = None,
    npz: Callable[[Path | None, bool, int | None], object] | None = None,
) -> int:
    """Write one result in the requested format and return the exit status.

    ``npz`` receives the destination, ``--compress``, and ``--compress-level``.
    Commands without CSV or NPZ output do not offer those formats.
    """
    if args.format == "json":
        _write_stream_output(args.output, json)
    elif args.format == "csv":
        assert csv is not None
        with _output_stream(args.output) as handle:
            csv(handle)
    elif args.format == "npz":
        assert npz is not None
        npz(args.output, args.compress, args.compress_level)
    else:
        _write_output(text(), args.output)
    return 0


def _load_matrix_from_args(
    args: argparse.Namespace, *, report: _InputReport | None = None
) -> tuple[np.ndarray, list[str] | None]:
    """Load the selected numeric items and optional identifiers named by input options."""
    return _load_input(
        args.data,
        args.delimiter,
        args.id_column,
        _parse_name_list(args.item_columns),
        args.header,
        args.missing_values,
        args.skip_rows,
        item_patterns=args.item_patterns,
        exclude_columns=args.exclude_columns,
        report=report,
    )


def _load_scoring_inputs(
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str] | None, IndexOptions]:
    """Load responses, the optional applicability mask, and index options."""
    if args.data == Path("-") and args.missing_applicable_mask == Path("-"):
        raise ValueError("response data and --missing-applicable-mask cannot both use stdin")
    matrix, respondent_ids = _load_matrix_from_args(args)
    applicable_mask = None
    if args.missing_applicable_mask is not None:
        applicable_mask = _load_applicable_mask(
            args.missing_applicable_mask, (matrix.shape[0], matrix.shape[1])
        )
    options = _options_from_args(args, missing_applicable_mask=applicable_mask)
    return matrix, respondent_ids, options


def _flag_scores(
    scores: np.ndarray,
    threshold: float | None,
    percentile: float | None,
    *,
    direction: Literal["high", "low"],
    default_percentile: float,
    inclusive_threshold: bool = True,
) -> tuple[float, np.ndarray]:
    """Resolve one cutoff and flag its suspicious tail.

    Fixed thresholds flag inclusively unless ``inclusive_threshold`` is false;
    percentile cutoffs exclude ties.
    """
    comparison_percentile = default_percentile if percentile is None else percentile
    cutoff = resolve_threshold(scores, threshold, comparison_percentile)
    flags = threshold_flags(
        scores,
        threshold=cutoff,
        percentile=comparison_percentile,
        direction=direction,
        inclusive=inclusive_threshold and threshold is not None,
    )
    return cutoff, flags


def _check_effort_cutoff(metric: str, threshold: float | None) -> None:
    """Reject a fixed effort cutoff outside the range of response time effort."""
    if metric == "effort" and threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "--threshold for the effort metric must be an RTE proportion between 0 and 1, "
            "not an item time"
        )


def _run_index_catalog(args: argparse.Namespace) -> int:
    """List registered indices and their orchestration metadata."""
    catalog = index_catalog()
    return _emit_result(
        args,
        text=lambda: _emit_index_catalog_text(catalog),
        json=lambda handle: handle.write(_emit_index_catalog_json(catalog)),
        csv=lambda handle: _write_index_catalog_csv(handle, catalog),
    )


def _run_inspect(args: argparse.Namespace) -> int:
    """Report how one matrix was parsed and which response values it contains."""
    report = _InputReport()
    matrix, _ = _load_matrix_from_args(args, report=report)
    inspection = _inspect_matrix(
        matrix, report, source=_input_label(args.data), id_column=args.id_column
    )
    return _emit_result(
        args,
        text=lambda: _emit_inspection_text(inspection),
        json=lambda handle: _write_inspection_json(handle, inspection),
    )


def _run_screen(args: argparse.Namespace) -> int:
    """Screen a response matrix, or apply new decisions to saved scores."""
    saved = args.command == "screen-scores"
    if not saved:
        validate_worker_count(args.workers)
    _check_output_options(args)
    if saved:
        scores, errors, respondent_ids, n_respondents = _load_reusable_scores(
            args.data, args.indices, args.strict
        )
        result = screen_scores(
            scores,
            percentile=args.percentile,
            min_flags=args.min_flags,
            min_valid_indices=args.min_valid_indices,
            thresholds=_parse_thresholds(args.threshold),
            percentiles=_parse_percentiles(args.index_percentile),
            errors=errors,
            n_respondents=n_respondents,
        )
    else:
        matrix, respondent_ids, options = _load_scoring_inputs(args)
        result = screen(
            matrix,
            indices=args.indices,
            options=options,
            percentile=args.percentile,
            min_flags=args.min_flags,
            min_valid_indices=args.min_valid_indices,
            thresholds=_parse_thresholds(args.threshold),
            percentiles=_parse_percentiles(args.index_percentile),
            strict=args.strict,
            workers=args.workers,
        )
    _report_soft_errors(result["errors"])
    return _emit_result(
        args,
        text=lambda: _emit_screen_text(result, args.top, respondent_ids),
        json=lambda handle: _write_screen_json(handle, result, respondent_ids),
        csv=lambda handle: _write_screen_csv(handle, result, respondent_ids),
        npz=lambda path, compressed, level: _write_screen_npz(
            path, result, respondent_ids, compressed=compressed, compression_level=level
        ),
    )


def _run_composite(args: argparse.Namespace) -> int:
    """Combine indices from a response matrix, or recombine saved component scores."""
    saved = args.command == "composite-scores"
    if not saved:
        validate_worker_count(args.workers)
    _check_output_options(args)
    scores: np.ndarray
    errors: dict[str, str]
    component_scores: dict[str, np.ndarray] | None = None
    valid_index_counts: np.ndarray | None = None
    if saved:
        saved_scores, archive_errors, respondent_ids, _ = _load_reusable_scores(
            args.data, args.indices, args.strict
        )
        weights = _parse_weights(args.weight)
        # Error metadata in composite archives follows the composite registry.
        enabled = composite_index_names()
        errors = {name: message for name, message in archive_errors.items() if name in enabled}
        unsupported: Literal["error", "drop"] = "drop" if args.skip_unsupported else "error"
        skipped = [name for name in saved_scores if name not in enabled]
        if args.skip_unsupported and skipped:
            # Name dropped screen-only scores without corrupting structured standard output.
            print(
                f"warning: skipped indices that are not composite-enabled: {', '.join(skipped)}",
                file=sys.stderr,
            )
        if args.include_components:
            details = composite_scores_summary(
                saved_scores,
                method=args.method,
                standardize=args.standardize,
                weights=weights,
                min_valid_indices=args.min_valid_indices,
                errors=errors,
                unsupported=unsupported,
            )
            scores = details["composite"]
            component_scores = details["indices"]
            valid_index_counts = details["valid_index_counts"]
        else:
            scores = composite_scores(
                saved_scores,
                method=args.method,
                standardize=args.standardize,
                weights=weights,
                min_valid_indices=args.min_valid_indices,
                errors=errors,
                unsupported=unsupported,
            )
    else:
        indices = args.indices
        if args.method == "best_subset" and indices is not None:
            # Warn here: the library DeprecationWarning is hidden from console-script users.
            print(
                "warning: --indices is ignored with --method best_subset; "
                "this will become an error",
                file=sys.stderr,
            )
            indices = None
        matrix, respondent_ids, options = _load_scoring_inputs(args)
        weights = _parse_weights(args.weight)
        if args.include_components:
            details = composite_summary(
                matrix,
                indices=indices,
                method=args.method,
                standardize=args.standardize,
                options=options,
                weights=weights,
                min_valid_indices=args.min_valid_indices,
                strict=args.strict,
                workers=args.workers,
            )
            scores = details["composite"]
            errors = details["errors"]
            component_scores = details["indices"]
            valid_index_counts = details["valid_index_counts"]
        else:
            scores_result = composite(
                matrix,
                indices=indices,
                method=args.method,
                standardize=args.standardize,
                options=options,
                weights=weights,
                min_valid_indices=args.min_valid_indices,
                return_diagnostics=True,
                strict=args.strict,
                workers=args.workers,
            )
            scores, errors = scores_result
    _report_soft_errors(errors)

    flags: np.ndarray | None = None
    flag_threshold: float | None = None
    if args.threshold is not None or args.percentile is not None:
        flag_threshold, flags = _flag_scores(
            scores, args.threshold, args.percentile, direction="high", default_percentile=95.0
        )
    report = CompositeReport(
        scores,
        args.method,
        respondent_ids,
        weights,
        args.min_valid_indices,
        errors,
        component_scores,
        valid_index_counts,
        standardized=args.standardize,
        flags=flags,
        flag_threshold=flag_threshold,
        flag_percentile=args.percentile,
        probabilities=logistic_transform(scores) if args.include_probability else None,
    )
    return _emit_result(
        args,
        text=lambda: _emit_composite_text(report, args.top),
        json=lambda handle: _write_composite_json(handle, report),
        csv=lambda handle: _write_composite_csv(handle, report),
        npz=lambda path, compressed, level: _write_composite_npz(
            path, report, compressed=compressed, compression_level=level
        ),
    )


def _run_response_time(args: argparse.Namespace) -> int:
    """Score or reuse timing results through one decision and output path."""
    _check_output_options(args)
    if args.command == "response-time-scores":
        saved = load_response_time_archive(args.data)
        scores = saved["scores"]
        direction = saved["flag_direction"]
        metric = saved["metric"]
        respondent_ids = saved["respondent_ids"]
        cutoff = saved["threshold"]
        flags = saved["flags"]
        _check_effort_cutoff(metric, args.threshold)
    else:
        metric = args.metric
        # Reject an impossible effort cutoff before any input is read.
        _check_effort_cutoff(metric, args.threshold)
        matrix, respondent_ids = _load_matrix_from_args(args)
        scores, direction = _score_response_times(
            matrix,
            metric,
            args.components,
            args.log_transform,
            args.random_seed,
            effort_fraction=args.effort_fraction,
            effort_threshold=args.effort_threshold,
            effort_max_threshold=args.effort_max_threshold,
        )

    # Saved timing decisions are preserved unless a new cutoff is requested.
    if args.command == "response-time" or args.threshold is not None or args.percentile is not None:
        threshold = args.threshold
        if metric == "effort" and threshold is None and args.percentile is None:
            threshold = _EFFORT_CUTOFF
        cutoff, flags = _flag_scores(
            scores,
            threshold,
            args.percentile,
            direction=direction,
            default_percentile=95.0 if direction == "high" else 5.0,
            # Fixed RTE cutoffs are strict, like response_time_effort_flag().
            inclusive_threshold=metric != "effort",
        )
    report = ResponseTimeReport(scores, flags, metric, direction, cutoff, respondent_ids)
    return _emit_result(
        args,
        text=lambda: _emit_response_time_text(report, args.top),
        json=lambda handle: _write_response_time_json(handle, report),
        csv=lambda handle: _write_response_time_csv(handle, report),
        npz=lambda path, compressed, level: _write_response_time_npz(
            path, report, compressed=compressed, compression_level=level
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ier`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler

    try:
        if getattr(args, "config", None) is not None:
            _apply_config_file(
                args,
                argv,
                _build_parser,
                exclusive=_EXCLUSIVE_DESTINATIONS,
                value_checks=_CONFIG_VALUE_CHECKS,
            )
        return handler(args)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
