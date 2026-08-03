"""Command-line interface for IER screening."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from ier import (
    IndexOptions,
    IndexScoreMap,
    __version__,
    composite,
    composite_scores,
    composite_summary,
    index_catalog,
    load_archive,
    load_response_time_archive,
    load_score_archive,
    response_time,
    response_time_consistency,
    response_time_mixture,
    screen,
    screen_scores,
)
from ier._cli_input import _load_boolean_matrix, _load_input, _load_numeric_vector
from ier._cli_npz import _require_npz_output_path
from ier._cli_results import (
    flag_response_time_scores,
    valid_score_counts,
    write_archive_info_result,
    write_composite_result,
    write_index_catalog_result,
    write_response_time_result,
    write_screen_result,
)
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


def _parse_percentiles(raw: list[str] | None) -> dict[str, float] | None:
    return _parse_named_floats(raw, "percentile")


def _parse_weights(raw: list[str] | None) -> dict[str, float] | None:
    return _parse_named_floats(raw, "weight", positive=True)


def _report_soft_errors(errors: dict[str, str]) -> None:
    """Report skipped indices without corrupting structured standard output."""
    for name, message in errors.items():
        print(f"warning: index '{name}' was skipped: {message}", file=sys.stderr)


def _load_optional_numeric_vector(path: Path | None, label: str) -> np.ndarray | None:
    """Load an optional numeric parameter vector from a CLI path."""
    return None if path is None else _load_numeric_vector(path, label)


def _load_optional_applicability_mask(path: Path | None) -> np.ndarray | None:
    """Load an optional respondent-by-item skip-logic mask."""
    return None if path is None else _load_boolean_matrix(path, "missing applicability mask")


def _options_from_args(args: argparse.Namespace) -> IndexOptions:
    return IndexOptions(
        na_rm=args.na_rm,
        irv_num_split=args.irv_num_split,
        irv_split_points=_parse_int_list(args.irv_split_points),
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        psychsyn_critval=args.psychsyn_critval,
        psychsyn_random_seed=args.psychsyn_random_seed,
        psychant_critval=args.psychant_critval,
        psychant_random_seed=args.psychant_random_seed,
        evenodd_factors=_parse_int_list(args.evenodd_factors),
        mad_positive_items=_parse_int_list(args.mad_positive_items),
        mad_negative_items=_parse_int_list(args.mad_negative_items),
        mad_scale_min=args.mad_scale_min,
        mad_scale_max=args.mad_scale_max,
        longstring_max_pattern_length=args.longstring_max_pattern_length,
        midpoint_tolerance=args.midpoint_tolerance,
        guttman_normalize=args.guttman_normalize,
        onset_window_size=args.onset_window_size,
        onset_min_items=args.onset_min_items,
        reliability_n_splits=args.reliability_n_splits,
        reliability_random_seed=args.reliability_random_seed,
        lz_difficulty=_load_optional_numeric_vector(
            args.lz_difficulty,
            "LZ difficulty vector",
        ),
        lz_discrimination=_load_optional_numeric_vector(
            args.lz_discrimination,
            "LZ discrimination vector",
        ),
        lz_theta=_load_optional_numeric_vector(args.lz_theta, "LZ theta vector"),
        lz_model=args.lz_model,
        semantic_item_pairs=_parse_pair_list(args.semantic_item_pairs),
        infrequency_item_indices=_parse_int_list(args.infrequency_item_indices),
        infrequency_expected_responses=_parse_float_list(args.infrequency_expected_responses),
        infrequency_proportion=args.infrequency_proportion,
        infrequency_missing=args.infrequency_missing,
        missing_item_indices=_parse_int_list(args.missing_item_indices),
        missing_applicable_mask=_load_optional_applicability_mask(args.missing_applicable_mask),
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


def _add_metadata_output_options(parser: argparse.ArgumentParser) -> None:
    """Add compact text and JSON output controls for metadata commands."""
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text summary)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write to a path, optionally .gz; use '-' for stdout",
    )


def _add_screen_decision_options(
    parser: argparse.ArgumentParser,
    *,
    indices_help: str,
) -> None:
    """Add reusable screening cutoff and consensus controls."""
    parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help=indices_help,
    )
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument(
        "--threshold",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Fixed per-index cutoff; repeat for multiple indices",
    )
    parser.add_argument(
        "--index-percentile",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Per-index tail percentile; repeat for multiple indices",
    )
    parser.add_argument(
        "--min-flags",
        type=int,
        default=2,
        help="Minimum per-index flags for a consensus flag (default: 2)",
    )
    parser.add_argument(
        "--min-valid-indices",
        type=int,
        default=None,
        help="Minimum available index scores required for consensus eligibility",
    )


def _add_composite_decision_options(
    parser: argparse.ArgumentParser,
    *,
    indices_help: str,
    methods: tuple[str, ...],
) -> None:
    """Add reusable composite combination and decision controls."""
    parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help=indices_help,
    )
    parser.add_argument(
        "--method",
        choices=methods,
        default="mean",
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize each directed component before combining (default: true)",
    )
    composite_flagging = parser.add_mutually_exclusive_group()
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
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Positive index weight override; repeat for multiple indices",
    )
    parser.add_argument(
        "--min-valid-indices",
        type=int,
        default=None,
        metavar="N",
        help="Require at least N available component scores per respondent",
    )
    parser.add_argument(
        "--include-components",
        action="store_true",
        help="Include raw component scores and per-respondent availability counts",
    )
    parser.add_argument(
        "--include-probability",
        action="store_true",
        help="Include uncalibrated logistic composite values alongside scores",
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
        "--irv-num-split",
        type=int,
        default=None,
        metavar="N",
        help="Average IRV across N equal item sections instead of the full matrix",
    )
    parser.add_argument(
        "--irv-split-points",
        default=None,
        metavar="I,J,...",
        help="Average IRV across custom 0-based item boundaries, e.g. '0,10,20'",
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
    parser.add_argument(
        "--psychsyn-random-seed",
        type=int,
        default=None,
        help="Seed missing-response psychometric synonym retries",
    )
    parser.add_argument("--psychant-critval", type=float, default=-0.6)
    parser.add_argument(
        "--psychant-random-seed",
        type=int,
        default=None,
        help="Seed missing-response psychometric antonym retries",
    )
    parser.add_argument(
        "--evenodd-factors",
        default=None,
        help="Comma-separated factor lengths, e.g. '5,5'",
    )
    parser.add_argument("--mad-positive-items", default=None, help="Comma-separated item indices")
    parser.add_argument("--mad-negative-items", default=None, help="Comma-separated item indices")
    parser.add_argument("--mad-scale-min", type=float, default=None)
    parser.add_argument("--mad-scale-max", type=float, default=None)
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
        "--lz-difficulty",
        type=Path,
        default=None,
        metavar="PATH",
        help="Calibrated LZ item difficulties as a one-row, one-column, or .npy vector",
    )
    parser.add_argument(
        "--lz-discrimination",
        type=Path,
        default=None,
        metavar="PATH",
        help="Calibrated 2PL LZ item discriminations as a vector file",
    )
    parser.add_argument(
        "--lz-theta",
        type=Path,
        default=None,
        metavar="PATH",
        help="Calibrated LZ respondent abilities as a vector file",
    )
    parser.add_argument(
        "--lz-model",
        choices=["1pl", "2pl"],
        default="2pl",
        help="LZ IRT model when scoring through the registry (default: 2pl)",
    )
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
    parser.add_argument(
        "--infrequency-missing",
        choices=["pass", "fail", "omit", "propagate"],
        default="pass",
        help="Missing attention-check policy (default: pass)",
    )
    parser.add_argument(
        "--missing-item-indices",
        default=None,
        help="Comma-separated required item indices for missing-rate scoring",
    )
    parser.add_argument(
        "--missing-applicable-mask",
        type=Path,
        default=None,
        metavar="PATH",
        help="Respondent-by-item skip-logic mask as 0/1 text or a Boolean/numeric .npy file",
    )
    _add_output_options(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ier",
        description="Detect insufficient effort / careless responding in survey matrices.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    archive_info_parser = sub.add_parser(
        "archive-info",
        help="Inspect a validated result archive without choosing its type first.",
    )
    archive_info_parser.add_argument(
        "archive",
        type=Path,
        help="Validated screen, composite, or response-time .npz archive",
    )
    _add_metadata_output_options(archive_info_parser)

    screen_parser = sub.add_parser("screen", help="Run multi-index screening on a matrix.")
    screen_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace or .npy item scores; use '-' for standard input",
    )
    _add_screen_decision_options(
        screen_parser,
        indices_help="Index names to compute (default: package screen defaults)",
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
    _add_composite_decision_options(
        composite_parser,
        indices_help="Index names to include (default: package composite defaults)",
        methods=("mean", "sum", "max", "best_subset"),
    )
    _add_shared_options(composite_parser)

    composite_recombine_parser = sub.add_parser(
        "composite-recombine",
        help="Recombine stored component scores without rescoring.",
    )
    composite_recombine_parser.add_argument(
        "archive",
        type=Path,
        help="Validated score .npz archive to reuse",
    )
    _add_composite_decision_options(
        composite_recombine_parser,
        indices_help="Stored component names to reuse, in order (default: all)",
        methods=("mean", "sum", "max"),
    )
    _add_output_options(composite_recombine_parser)

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

    response_time_reflag_parser = sub.add_parser(
        "response-time-reflag",
        help="Reflag stored response-time scores without rescoring the timing matrix.",
    )
    response_time_reflag_parser.add_argument(
        "archive",
        type=Path,
        help="Validated response-time .npz archive to reuse",
    )
    response_time_reflagging = response_time_reflag_parser.add_mutually_exclusive_group(
        required=True
    )
    response_time_reflagging.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Flag scores inclusively at a fixed cutoff",
    )
    response_time_reflagging.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Flag scores beyond a sample percentile, excluding cutoff ties",
    )
    _add_output_options(response_time_reflag_parser)

    screen_reflag_parser = sub.add_parser(
        "screen-reflag",
        help="Reapply screening decisions to stored index scores without rescoring.",
    )
    screen_reflag_parser.add_argument(
        "archive",
        type=Path,
        help="Validated score .npz archive to reuse",
    )
    _add_screen_decision_options(
        screen_reflag_parser,
        indices_help="Stored index names to reuse, in order (default: all)",
    )
    _add_output_options(screen_reflag_parser)

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


def _select_archive_scores(
    scores: IndexScoreMap,
    indices: list[str] | None,
) -> IndexScoreMap:
    """Select ordered stored score vectors without copying their arrays."""
    if indices is None:
        return scores
    if len(indices) != len(set(indices)):
        raise ValueError("--indices must not contain duplicates")
    for name in indices:
        if name not in scores:
            raise ValueError(f"score archive does not contain selected index: {name}")
    return {name: scores[name] for name in indices}


def _run_command(args: argparse.Namespace) -> int:
    """Execute one parsed CLI command, allowing user-facing failures to bubble to main()."""
    if args.command == "indices":
        return write_index_catalog_result(args, index_catalog())

    if args.command == "archive-info":
        return write_archive_info_result(args, load_archive(args.archive))

    if args.command in {"screen", "composite"}:
        validate_worker_count(args.workers)

    if args.format == "npz":
        _require_npz_output_path(args.output)

    if args.command == "response-time-reflag":
        timing_archive = load_response_time_archive(args.archive)
        flags, cutoff, threshold_source, percentile = flag_response_time_scores(
            timing_archive["scores"],
            timing_archive["flag_direction"],
            args.threshold,
            args.percentile,
        )
        return write_response_time_result(
            args,
            timing_archive["scores"],
            flags,
            timing_archive["metric"],
            timing_archive["flag_direction"],
            cutoff,
            timing_archive["respondent_ids"],
            threshold_source,
            percentile,
        )

    if args.command == "screen-reflag":
        score_archive = load_score_archive(args.archive)
        selected_scores = _select_archive_scores(score_archive["scores"], args.indices)
        result = screen_scores(
            selected_scores,
            percentile=args.percentile,
            min_flags=args.min_flags,
            min_valid_indices=args.min_valid_indices,
            thresholds=_parse_thresholds(args.threshold),
            percentiles=_parse_percentiles(args.index_percentile),
        )
        result["errors"] = score_archive["errors"].copy()
        _report_soft_errors(result["errors"])
        return write_screen_result(args, result, score_archive["respondent_ids"])

    if args.command == "composite-recombine":
        score_archive = load_score_archive(args.archive)
        selected_scores = _select_archive_scores(score_archive["scores"], args.indices)
        if (
            args.format == "npz"
            and args.output is not None
            and args.output.resolve() == args.archive.resolve()
            and not args.include_components
        ):
            raise ValueError(
                "in-place composite archive output requires --include-components "
                "to preserve reusable scores"
            )
        weights = _parse_weights(args.weight)
        scores = composite_scores(
            selected_scores,
            method=args.method,
            standardize=args.standardize,
            weights=weights,
            min_valid_indices=args.min_valid_indices,
        )
        errors = score_archive["errors"].copy()
        _report_soft_errors(errors)
        return write_composite_result(
            args,
            scores,
            score_archive["respondent_ids"],
            weights,
            errors,
            selected_scores if args.include_components else None,
            valid_score_counts(selected_scores) if args.include_components else None,
        )

    options = None if args.command == "response-time" else _options_from_args(args)
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
        flags, cutoff, threshold_source, percentile = flag_response_time_scores(
            scores,
            direction,
            args.threshold,
            args.percentile,
        )
        return write_response_time_result(
            args,
            scores,
            flags,
            args.metric,
            direction,
            cutoff,
            respondent_ids,
            threshold_source,
            percentile,
        )

    assert options is not None
    if args.command == "screen":
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
        return write_screen_result(args, result, respondent_ids)

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
    return write_composite_result(
        args,
        scores,
        respondent_ids,
        weights,
        errors,
        component_scores,
        valid_index_counts,
    )


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
