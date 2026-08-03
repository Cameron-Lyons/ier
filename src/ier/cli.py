"""Command-line interface for IER screening."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from ier import (
    IndexOptions,
    __version__,
    composite,
    composite_summary,
    index_catalog,
    response_time,
    response_time_consistency,
    response_time_mixture,
    screen,
)
from ier._cli_input import _load_input
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
    _write_json_output,
    _write_output,
    _write_response_time_csv,
    _write_response_time_json,
    _write_screen_csv,
    _write_screen_json,
)
from ier._flagging import resolve_threshold, threshold_flags
from ier._registry import validate_worker_count


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    return [int(part) for part in parts]


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    return [float(part) for part in parts]


def _parse_name_list(raw: list[str] | None) -> list[str] | None:
    """Parse repeated comma-separated column-name arguments."""
    if raw is None:
        return None
    names = [name.strip() for entry in raw for name in entry.split(",") if name.strip()]
    if not names:
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


def _parse_weights(raw: list[str] | None) -> dict[str, float] | None:
    return _parse_named_floats(raw, "weight", positive=True)


def _report_soft_errors(errors: dict[str, str]) -> None:
    """Report skipped indices without corrupting structured standard output."""
    for name, message in errors.items():
        print(f"warning: index '{name}' was skipped: {message}", file=sys.stderr)


def _options_from_args(args: argparse.Namespace) -> IndexOptions:
    return IndexOptions(
        na_rm=args.na_rm,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        psychsyn_critval=args.psychsyn_critval,
        psychant_critval=args.psychant_critval,
        evenodd_factors=_parse_int_list(args.evenodd_factors),
        mad_positive_items=_parse_int_list(args.mad_positive_items),
        mad_negative_items=_parse_int_list(args.mad_negative_items),
        mad_scale_max=args.mad_scale_max,
        longstring_max_pattern_length=args.longstring_max_pattern_length,
        midpoint_tolerance=args.midpoint_tolerance,
        guttman_normalize=args.guttman_normalize,
        onset_window_size=args.onset_window_size,
        onset_min_items=args.onset_min_items,
        reliability_n_splits=args.reliability_n_splits,
        reliability_random_seed=args.reliability_random_seed,
        semantic_item_pairs=_parse_pair_list(args.semantic_item_pairs),
        infrequency_item_indices=_parse_int_list(args.infrequency_item_indices),
        infrequency_expected_responses=_parse_float_list(args.infrequency_expected_responses),
        infrequency_proportion=args.infrequency_proportion,
    )


def _add_matrix_input_options(parser: argparse.ArgumentParser) -> None:
    """Add matrix selection options shared by scoring commands."""
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Input delimiter (auto-detect comma, tab, semicolon, or whitespace if omitted)",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        metavar="NAME",
        help="Named header column to preserve as respondent identifiers",
    )
    parser.add_argument(
        "--item-columns",
        action="append",
        default=None,
        metavar="NAME[,NAME...]",
        help="Named header columns to score, in order; comma-separate or repeat",
    )


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output controls shared by scoring commands."""
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv", "npz"],
        default="text",
        help="Output format (default: text summary)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write to a path, optionally .gz; use '-' for stdout; NPZ requires a .npz path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="For text format: show the top N respondents (default: 10)",
    )


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    _add_matrix_input_options(parser)
    parser.add_argument("--scale-min", type=float, default=None)
    parser.add_argument("--scale-max", type=float, default=None)
    parser.add_argument(
        "--na-rm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop incomplete rows / pairwise NaNs where supported (default: true)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested index cannot be computed",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Score independent indices concurrently (default: 1)",
    )
    parser.add_argument("--psychsyn-critval", type=float, default=0.6)
    parser.add_argument("--psychant-critval", type=float, default=-0.6)
    parser.add_argument(
        "--evenodd-factors",
        default=None,
        help="Comma-separated factor lengths, e.g. '5,5'",
    )
    parser.add_argument("--mad-positive-items", default=None, help="Comma-separated item indices")
    parser.add_argument("--mad-negative-items", default=None, help="Comma-separated item indices")
    parser.add_argument("--mad-scale-max", type=int, default=None)
    parser.add_argument("--longstring-max-pattern-length", type=int, default=5)
    parser.add_argument("--midpoint-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--guttman-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--onset-window-size", type=int, default=10)
    parser.add_argument("--onset-min-items", type=int, default=20)
    parser.add_argument("--reliability-n-splits", type=int, default=100)
    parser.add_argument("--reliability-random-seed", type=int, default=None)
    parser.add_argument(
        "--semantic-item-pairs",
        default=None,
        help="Pairs as 'i,j;i,j' (0-based item indices)",
    )
    parser.add_argument("--infrequency-item-indices", default=None)
    parser.add_argument("--infrequency-expected-responses", default=None)
    parser.add_argument(
        "--infrequency-proportion",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    _add_output_options(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ier",
        description="Detect insufficient effort / careless responding in survey matrices.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    screen_parser = sub.add_parser("screen", help="Run multi-index screening on a matrix.")
    screen_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace or .npy item scores; use '-' for standard input",
    )
    screen_parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index names to compute (default: package screen defaults)",
    )
    screen_parser.add_argument("--percentile", type=float, default=95.0)
    screen_parser.add_argument(
        "--threshold",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Fixed per-index cutoff; repeat for multiple indices",
    )
    screen_parser.add_argument(
        "--min-flags",
        type=int,
        default=2,
        help="Minimum per-index flags for a consensus flag (default: 2)",
    )
    _add_shared_options(screen_parser)

    composite_parser = sub.add_parser(
        "composite", help="Compute a composite IER score for each respondent."
    )
    composite_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace or .npy item scores; use '-' for standard input",
    )
    composite_parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index names to include (default: package composite defaults)",
    )
    composite_parser.add_argument(
        "--method",
        choices=["mean", "sum", "max", "best_subset"],
        default="mean",
    )
    composite_parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize each directed component before combining (default: true)",
    )
    composite_parser.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Positive index weight override; repeat for multiple indices",
    )
    composite_parser.add_argument(
        "--min-valid-indices",
        type=int,
        default=None,
        metavar="N",
        help="Require at least N available component scores per respondent",
    )
    composite_parser.add_argument(
        "--include-components",
        action="store_true",
        help="Include raw component scores and per-respondent availability counts",
    )
    _add_shared_options(composite_parser)

    response_time_parser = sub.add_parser(
        "response-time",
        help="Score and flag a response-time matrix.",
    )
    response_time_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace or .npy timing values; use '-' for standard input",
    )
    response_time_parser.add_argument(
        "--metric",
        choices=["mean", "median", "sd", "min", "consistency", "mixture"],
        default="median",
        help="Timing score to compute (default: median)",
    )
    response_time_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed inclusive flag cutoff in score units",
    )
    response_time_parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Percentile cutoff (default: 5 for low scores, 95 for mixture)",
    )
    response_time_parser.add_argument(
        "--components",
        type=int,
        default=2,
        help="Gaussian components for the mixture metric (default: 2)",
    )
    response_time_parser.add_argument(
        "--log-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log-transform median times for the mixture metric (default: true)",
    )
    response_time_parser.add_argument("--random-seed", type=int, default=None)
    _add_matrix_input_options(response_time_parser)
    _add_output_options(response_time_parser)

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
        help="Write output to a path, optionally .gz; use '-' for stdout",
    )

    return parser


def _score_response_times(
    matrix: np.ndarray,
    metric: str,
    components: int,
    log_transform: bool,
    random_seed: int | None,
) -> tuple[np.ndarray, Literal["high", "low"]]:
    """Compute one timing metric and its suspicious-score direction."""
    if metric == "consistency":
        return response_time_consistency(matrix), "low"
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


def _run_command(args: argparse.Namespace) -> int:
    """Execute one parsed CLI command, allowing user-facing failures to bubble to main()."""
    if args.command == "indices":
        catalog = index_catalog()
        if args.format == "json":
            text = _emit_index_catalog_json(catalog)
        elif args.format == "csv":
            with _output_stream(args.output) as handle:
                _write_index_catalog_csv(handle, catalog)
            return 0
        else:
            text = _emit_index_catalog_text(catalog)
        _write_output(text, args.output)
        return 0

    if args.command in {"screen", "composite"}:
        validate_worker_count(args.workers)

    if args.format == "npz":
        _require_npz_output_path(args.output)

    matrix, respondent_ids = _load_input(
        args.data,
        args.delimiter,
        args.id_column,
        _parse_name_list(args.item_columns),
    )
    if args.command == "response-time":
        scores, direction = _score_response_times(
            matrix,
            args.metric,
            args.components,
            args.log_transform,
            args.random_seed,
        )
        percentile = args.percentile
        if percentile is None:
            percentile = 95.0 if direction == "high" else 5.0
        cutoff = resolve_threshold(scores, args.threshold, percentile)
        flags = threshold_flags(
            scores,
            threshold=cutoff,
            percentile=percentile,
            direction=direction,
            inclusive=args.threshold is not None,
        )

        if args.format == "json":
            _write_json_output(
                args.output,
                lambda handle: _write_response_time_json(
                    handle,
                    scores,
                    flags,
                    args.metric,
                    direction,
                    cutoff,
                    respondent_ids,
                ),
            )
            return 0
        elif args.format == "csv":
            with _output_stream(args.output) as handle:
                _write_response_time_csv(handle, scores, flags, respondent_ids)
            return 0
        elif args.format == "npz":
            _write_response_time_npz(
                args.output,
                scores,
                flags,
                args.metric,
                direction,
                cutoff,
                respondent_ids,
            )
            return 0
        else:
            text = _emit_response_time_text(
                scores,
                flags,
                args.metric,
                direction,
                cutoff,
                args.top,
                respondent_ids,
            )
        _write_output(text, args.output)
        return 0

    options = _options_from_args(args)
    if args.command == "screen":
        result = screen(
            matrix,
            indices=args.indices,
            options=options,
            percentile=args.percentile,
            min_flags=args.min_flags,
            thresholds=_parse_thresholds(args.threshold),
            strict=args.strict,
            workers=args.workers,
        )
        _report_soft_errors(result["errors"])
        if args.format == "json":
            _write_json_output(
                args.output,
                lambda handle: _write_screen_json(handle, result, respondent_ids),
            )
            return 0
        elif args.format == "csv":
            with _output_stream(args.output) as handle:
                _write_screen_csv(handle, result, respondent_ids)
            return 0
        elif args.format == "npz":
            _write_screen_npz(args.output, result, respondent_ids)
            return 0
        else:
            text = _emit_screen_text(result, args.top, respondent_ids)
        _write_output(text, args.output)
        return 0

    weights = _parse_weights(args.weight)
    component_scores: dict[str, np.ndarray] | None = None
    valid_index_counts: np.ndarray | None = None
    if args.include_components:
        details = composite_summary(
            matrix,
            indices=args.indices,
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
            indices=args.indices,
            method=args.method,
            standardize=args.standardize,
            options=options,
            weights=weights,
            min_valid_indices=args.min_valid_indices,
            return_diagnostics=True,
            strict=args.strict,
            workers=args.workers,
        )
        if not isinstance(scores_result, tuple):
            print("error: unexpected composite return type", file=sys.stderr)
            return 1
        scores, errors = scores_result
    _report_soft_errors(errors)

    if args.format == "json":
        _write_json_output(
            args.output,
            lambda handle: _write_composite_json(
                handle,
                scores,
                args.method,
                respondent_ids,
                weights,
                args.min_valid_indices,
                errors,
                component_scores,
                valid_index_counts,
                standardized=args.standardize,
            ),
        )
        return 0
    elif args.format == "csv":
        with _output_stream(args.output) as handle:
            _write_composite_csv(
                handle,
                scores,
                respondent_ids,
                component_scores,
                valid_index_counts,
            )
        return 0
    elif args.format == "npz":
        _write_composite_npz(
            args.output,
            scores,
            args.method,
            respondent_ids,
            weights,
            args.min_valid_indices,
            errors,
            component_scores,
            valid_index_counts,
            standardized=args.standardize,
        )
        return 0
    else:
        text = _emit_composite_text(
            scores,
            args.method,
            args.top,
            respondent_ids,
            weights,
            args.min_valid_indices,
            errors,
            component_scores,
            valid_index_counts,
            standardized=args.standardize,
        )
    _write_output(text, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ier`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _run_command(args)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
