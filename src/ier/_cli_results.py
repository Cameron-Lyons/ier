"""Shared result derivation and format dispatch for command-line workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ier._cli_npz import (
    _write_composite_npz,
    _write_response_time_npz,
    _write_screen_npz,
)
from ier._cli_output import (
    _emit_archive_info_text,
    _emit_composite_text,
    _emit_index_catalog_json,
    _emit_index_catalog_text,
    _emit_response_time_text,
    _emit_screen_text,
    _output_stream,
    _write_archive_info_json,
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
from ier._statistics import logistic_transform

if TYPE_CHECKING:
    import argparse

    from ier.types import (
        IndexCatalog,
        IndexScoreMap,
        InspectableArchive,
        ResponseTimeFlagDirection,
        ResponseTimeMetric,
        ResponseTimeThresholdSource,
        ScreenResult,
    )


def write_archive_info_result(args: argparse.Namespace, archive: InspectableArchive) -> int:
    """Write validated archive metadata through the selected CLI format."""
    if args.format == "json":
        _write_json_output(
            args.output,
            lambda handle: _write_archive_info_json(handle, archive),
        )
    else:
        _write_output(_emit_archive_info_text(archive), args.output)
    return 0


def write_index_catalog_result(args: argparse.Namespace, catalog: IndexCatalog) -> int:
    """Write the index catalog through the selected CLI format."""
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


def write_screen_result(
    args: argparse.Namespace,
    result: ScreenResult,
    respondent_ids: list[str] | None,
) -> int:
    """Write one screening result through the selected CLI format."""
    if args.format == "json":
        _write_json_output(
            args.output,
            lambda handle: _write_screen_json(handle, result, respondent_ids),
        )
    elif args.format == "csv":
        with _output_stream(args.output) as handle:
            _write_screen_csv(handle, result, respondent_ids)
    elif args.format == "npz":
        _write_screen_npz(args.output, result, respondent_ids)
    else:
        _write_output(
            _emit_screen_text(result, args.top, respondent_ids),
            args.output,
        )
    return 0


def valid_score_counts(scores: IndexScoreMap) -> np.ndarray:
    """Count available stored components with one respondent-sized accumulator."""
    n_respondents = len(next(iter(scores.values())))
    counts = np.zeros(n_respondents, dtype=np.int_)
    for values in scores.values():
        counts += ~np.isnan(values)
    return counts


def write_composite_result(
    args: argparse.Namespace,
    scores: np.ndarray,
    respondent_ids: list[str] | None,
    weights: dict[str, float] | None,
    errors: dict[str, str],
    component_scores: IndexScoreMap | None,
    valid_index_counts: np.ndarray | None,
) -> int:
    """Derive optional composite fields and write the selected CLI format."""
    composite_flags: np.ndarray | None = None
    flag_threshold: float | None = None
    flag_percentile: float | None = args.percentile
    if args.threshold is not None or flag_percentile is not None:
        comparison_percentile = 95.0 if flag_percentile is None else flag_percentile
        flag_threshold = resolve_threshold(scores, args.threshold, comparison_percentile)
        composite_flags = threshold_flags(
            scores,
            threshold=flag_threshold,
            percentile=comparison_percentile,
            direction="high",
            inclusive=args.threshold is not None,
        )
    probabilities = logistic_transform(scores) if args.include_probability else None

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
                flags=composite_flags,
                flag_threshold=flag_threshold,
                flag_percentile=flag_percentile,
                probabilities=probabilities,
            ),
        )
    elif args.format == "csv":
        with _output_stream(args.output) as handle:
            _write_composite_csv(
                handle,
                scores,
                respondent_ids,
                component_scores,
                valid_index_counts,
                composite_flags,
                probabilities,
            )
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
            flags=composite_flags,
            flag_threshold=flag_threshold,
            flag_percentile=flag_percentile,
            probabilities=probabilities,
        )
    else:
        _write_output(
            _emit_composite_text(
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
                flags=composite_flags,
                flag_threshold=flag_threshold,
                flag_percentile=flag_percentile,
                probabilities=probabilities,
            ),
            args.output,
        )
    return 0


def flag_response_time_scores(
    scores: np.ndarray,
    direction: ResponseTimeFlagDirection,
    threshold: float | None,
    percentile: float | None,
) -> tuple[np.ndarray, float, ResponseTimeThresholdSource, float | None]:
    """Apply one fixed or percentile cutoff and retain its provenance."""
    comparison_percentile = (
        (95.0 if direction == "high" else 5.0) if percentile is None else percentile
    )
    cutoff = resolve_threshold(scores, threshold, comparison_percentile)
    threshold_source: ResponseTimeThresholdSource = (
        "fixed" if threshold is not None else "percentile"
    )
    requested_percentile = None if threshold is not None else comparison_percentile
    flags = threshold_flags(
        scores,
        threshold=cutoff,
        percentile=comparison_percentile,
        direction=direction,
        inclusive=threshold is not None,
    )
    return flags, cutoff, threshold_source, requested_percentile


def write_response_time_result(
    args: argparse.Namespace,
    scores: np.ndarray,
    flags: np.ndarray,
    metric: ResponseTimeMetric,
    direction: ResponseTimeFlagDirection,
    cutoff: float,
    respondent_ids: list[str] | None,
    threshold_source: ResponseTimeThresholdSource,
    percentile: float | None,
) -> int:
    """Write one response-time result through the selected CLI format."""
    if args.format == "json":
        _write_json_output(
            args.output,
            lambda handle: _write_response_time_json(
                handle,
                scores,
                flags,
                metric,
                direction,
                cutoff,
                respondent_ids,
                threshold_source=threshold_source,
                percentile=percentile,
            ),
        )
    elif args.format == "csv":
        with _output_stream(args.output) as handle:
            _write_response_time_csv(handle, scores, flags, respondent_ids)
    elif args.format == "npz":
        _write_response_time_npz(
            args.output,
            scores,
            flags,
            metric,
            direction,
            cutoff,
            respondent_ids,
            threshold_source=threshold_source,
            percentile=percentile,
        )
    else:
        _write_output(
            _emit_response_time_text(
                scores,
                flags,
                metric,
                direction,
                cutoff,
                args.top,
                respondent_ids,
                threshold_source=threshold_source,
                percentile=percentile,
            ),
            args.output,
        )
    return 0
