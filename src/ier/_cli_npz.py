"""Versioned, pickle-free NumPy archive serializers for CLI results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ier.archive import (
    _archive_header,
    _respondent_ids_member,
    _validate_archive_strings,
    _write_npz_atomically,
    save_response_time_archive,
    save_screen_archive,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ier._cli_composite import CompositeReport, ResponseTimeReport
    from ier.types import ScreenResult


def _add_errors(
    payload: dict[str, np.ndarray],
    errors: Mapping[str, str] | None,
) -> None:
    """Add aligned, pickle-free error metadata to a result payload."""
    items = list((errors or {}).items())
    _validate_archive_strings([message for _, message in items], name="archive error messages")
    payload["error_names"] = np.asarray([name for name, _ in items], dtype=np.str_)
    payload["error_messages"] = np.asarray([message for _, message in items], dtype=np.str_)


def _require_npz_output_path(path: Path | None) -> Path:
    """Validate and return the explicit file destination required by NPZ output."""
    if path is None or path == Path("-"):
        raise ValueError("--format npz requires --output with a .npz file path")
    if path.suffix.casefold() != ".npz":
        raise ValueError("--format npz requires an output path ending in .npz")
    return path


def _write_npz_archive(
    path: Path | None,
    payload: dict[str, np.ndarray],
    *,
    compressed: bool = False,
    compression_level: int | None = None,
) -> None:
    """Write one pickle-free NumPy result archive to an explicit file path."""
    destination = _require_npz_output_path(path)
    _write_npz_atomically(
        destination, payload, compressed=compressed, compression_level=compression_level
    )


def _write_screen_npz(
    path: Path | None,
    result: ScreenResult,
    respondent_ids: list[str] | None = None,
    *,
    compressed: bool = False,
    compression_level: int | None = None,
) -> None:
    """Write complete screening results as a versioned NumPy archive."""
    save_screen_archive(
        _require_npz_output_path(path),
        result,
        respondent_ids=respondent_ids,
        compressed=compressed,
        compression_level=compression_level,
    )


def _write_composite_npz(
    path: Path | None,
    report: CompositeReport,
    *,
    compressed: bool = False,
    compression_level: int | None = None,
) -> None:
    """Write composite results as a versioned NumPy archive."""
    scores = report.scores
    payload = _archive_header("composite", len(scores))
    payload.update(
        {
            "method": np.asarray(report.method, dtype=np.str_),
            "standardized": np.asarray(report.standardized, dtype=np.bool_),
            "scores": np.asarray(scores, dtype=np.float64),
        }
    )
    if report.weights:
        payload["weight_names"] = np.asarray(list(report.weights), dtype=np.str_)
        payload["weights"] = np.asarray(list(report.weights.values()), dtype=np.float64)
    if report.min_valid_indices is not None:
        payload["min_valid_indices"] = np.asarray(report.min_valid_indices, dtype=np.int64)
    if report.probabilities is not None:
        payload["probability_scale"] = np.asarray("uncalibrated_logistic", dtype=np.str_)
        payload["probabilities"] = np.asarray(report.probabilities, dtype=np.float64)
    if report.flags is not None:
        assert report.flag_threshold is not None
        payload["threshold"] = np.asarray(report.flag_threshold, dtype=np.float64)
        payload["threshold_source"] = np.asarray(
            "percentile" if report.flag_percentile is not None else "fixed",
            dtype=np.str_,
        )
        if report.flag_percentile is not None:
            payload["percentile"] = np.asarray(report.flag_percentile, dtype=np.float64)
        payload["flags"] = np.asarray(report.flags, dtype=np.bool_)
    if report.component_scores is not None:
        assert report.valid_index_counts is not None
        payload["index_names"] = np.asarray(list(report.component_scores), dtype=np.str_)
        payload["valid_index_counts"] = np.asarray(report.valid_index_counts, dtype=np.int64)
        for name, values in report.component_scores.items():
            payload[f"score__{name}"] = np.asarray(values, dtype=np.float64)
    _add_errors(payload, report.errors)
    payload.update(_respondent_ids_member(report.respondent_ids, len(scores)))
    _write_npz_archive(path, payload, compressed=compressed, compression_level=compression_level)


def _write_response_time_npz(
    path: Path | None,
    report: ResponseTimeReport,
    *,
    compressed: bool = False,
    compression_level: int | None = None,
) -> None:
    """Write response-time results as a versioned NumPy archive."""
    destination = _require_npz_output_path(path)
    save_response_time_archive(
        destination,
        report.scores,
        report.flags,
        threshold=report.cutoff,
        metric=report.metric,
        flag_direction=report.direction,
        respondent_ids=report.respondent_ids,
        compressed=compressed,
        compression_level=compression_level,
    )
