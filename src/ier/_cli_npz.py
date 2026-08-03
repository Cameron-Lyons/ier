"""Versioned, pickle-free NumPy archive serializers for CLI results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from ier._cli_composite import (
    validate_composite_components,
    validate_composite_flags,
    validate_composite_probabilities,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ier.types import ScreenResult


def _metadata(result_type: str, n_respondents: int) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "result_type": np.asarray(result_type, dtype=np.str_),
        "n_respondents": np.asarray(n_respondents, dtype=np.int64),
    }


def _add_respondent_ids(
    payload: dict[str, np.ndarray],
    n_respondents: int,
    respondent_ids: list[str] | None,
) -> None:
    if respondent_ids is None:
        return
    if len(respondent_ids) != n_respondents:
        raise ValueError("respondent ID count must match result length")
    payload["respondent_ids"] = np.asarray(respondent_ids, dtype=np.str_)


def _add_errors(
    payload: dict[str, np.ndarray],
    errors: Mapping[str, str] | None,
) -> None:
    """Add aligned, pickle-free error metadata to a result payload."""
    items = list((errors or {}).items())
    payload["error_names"] = np.asarray([name for name, _ in items], dtype=np.str_)
    payload["error_messages"] = np.asarray([message for _, message in items], dtype=np.str_)


def _require_npz_output_path(path: Path | None) -> Path:
    """Validate and return the explicit file destination required by NPZ output."""
    if path is None or path == Path("-"):
        raise ValueError("--format npz requires --output with a .npz file path")
    if path.suffix.casefold() != ".npz":
        raise ValueError("--format npz requires an output path ending in .npz")
    return path


def _write_npz_archive(path: Path | None, payload: dict[str, np.ndarray]) -> None:
    """Write one pickle-free NumPy result archive to an explicit file path."""
    destination = _require_npz_output_path(path)
    if any(value.dtype.hasobject for value in payload.values()):
        raise ValueError("NPZ output cannot contain object arrays")
    with ZipFile(destination, mode="w", compression=ZIP_STORED, allowZip64=True) as archive:
        for name, value in payload.items():
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                np.save(member, value, allow_pickle=False)


def _write_screen_npz(
    path: Path | None,
    result: ScreenResult,
    respondent_ids: list[str] | None = None,
) -> None:
    """Write complete screening results as a versioned NumPy archive."""
    names = result["indices_used"]
    summary_columns = ["mean", "std", "min", "max"]
    summary_statistics = np.asarray(
        [
            [
                result["summary"][name]["mean"],
                result["summary"][name]["std"],
                result["summary"][name]["min"],
                result["summary"][name]["max"],
            ]
            for name in names
        ],
        dtype=np.float64,
    ).reshape(len(names), len(summary_columns))
    payload = _metadata("screen", result["n_respondents"])
    payload.update(
        {
            "n_indices": np.asarray(result["n_indices"], dtype=np.int64),
            "min_flags": np.asarray(result["min_flags"], dtype=np.int64),
            "index_names": np.asarray(names, dtype=np.str_),
            "thresholds": np.asarray(
                [
                    np.nan if result["thresholds"][name] is None else result["thresholds"][name]
                    for name in names
                ],
                dtype=np.float64,
            ),
            "flag_counts": np.asarray(result["flag_counts"], dtype=np.int64),
            "valid_index_counts": np.asarray(
                result["valid_index_counts"],
                dtype=np.int64,
            ),
            "consensus_eligible": np.asarray(
                result["consensus_eligible"],
                dtype=np.bool_,
            ),
            "consensus_flags": np.asarray(result["consensus_flags"], dtype=np.bool_),
            "summary_columns": np.asarray(summary_columns, dtype=np.str_),
            "summary_statistics": summary_statistics,
            "summary_n_flagged": np.asarray(
                [result["summary"][name]["n_flagged"] for name in names],
                dtype=np.int64,
            ),
        }
    )
    if result["min_valid_indices"] is not None:
        payload["min_valid_indices"] = np.asarray(
            result["min_valid_indices"],
            dtype=np.int64,
        )
    _add_errors(payload, result["errors"])
    for name in names:
        payload[f"score__{name}"] = np.asarray(result["scores"][name], dtype=np.float64)
        payload[f"flag__{name}"] = np.asarray(result["flags"][name], dtype=np.bool_)
    _add_respondent_ids(payload, result["n_respondents"], respondent_ids)
    _write_npz_archive(path, payload)


def _write_composite_npz(
    path: Path | None,
    scores: np.ndarray,
    method: str,
    respondent_ids: list[str] | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    errors: Mapping[str, str] | None = None,
    component_scores: Mapping[str, np.ndarray] | None = None,
    valid_index_counts: np.ndarray | None = None,
    standardized: bool = True,
    flags: np.ndarray | None = None,
    flag_threshold: float | None = None,
    flag_percentile: float | None = None,
    probabilities: np.ndarray | None = None,
) -> None:
    """Write composite results as a versioned NumPy archive."""
    validate_composite_components(len(scores), component_scores, valid_index_counts)
    validate_composite_flags(len(scores), flags, flag_threshold, flag_percentile)
    validate_composite_probabilities(len(scores), probabilities)

    payload = _metadata("composite", len(scores))
    payload.update(
        {
            "method": np.asarray(method, dtype=np.str_),
            "standardized": np.asarray(standardized, dtype=np.bool_),
            "scores": np.asarray(scores, dtype=np.float64),
        }
    )
    if weights:
        payload["weight_names"] = np.asarray(list(weights), dtype=np.str_)
        payload["weights"] = np.asarray(list(weights.values()), dtype=np.float64)
    if min_valid_indices is not None:
        payload["min_valid_indices"] = np.asarray(min_valid_indices, dtype=np.int64)
    if probabilities is not None:
        payload["probability_scale"] = np.asarray("uncalibrated_logistic", dtype=np.str_)
        payload["probabilities"] = np.asarray(probabilities, dtype=np.float64)
    if flags is not None:
        assert flag_threshold is not None
        payload["threshold"] = np.asarray(flag_threshold, dtype=np.float64)
        payload["threshold_source"] = np.asarray(
            "percentile" if flag_percentile is not None else "fixed",
            dtype=np.str_,
        )
        if flag_percentile is not None:
            payload["percentile"] = np.asarray(flag_percentile, dtype=np.float64)
        payload["flags"] = np.asarray(flags, dtype=np.bool_)
    if component_scores is not None:
        assert valid_index_counts is not None
        payload["index_names"] = np.asarray(list(component_scores), dtype=np.str_)
        payload["valid_index_counts"] = np.asarray(valid_index_counts, dtype=np.int64)
        for name, values in component_scores.items():
            payload[f"score__{name}"] = np.asarray(values, dtype=np.float64)
    _add_errors(payload, errors)
    _add_respondent_ids(payload, len(scores), respondent_ids)
    _write_npz_archive(path, payload)


def _write_response_time_npz(
    path: Path | None,
    scores: np.ndarray,
    flags: np.ndarray,
    metric: str,
    direction: Literal["high", "low"],
    cutoff: float,
    respondent_ids: list[str] | None = None,
) -> None:
    """Write response-time results as a versioned NumPy archive."""
    payload = _metadata("response_time", len(scores))
    payload.update(
        {
            "metric": np.asarray(metric, dtype=np.str_),
            "flag_direction": np.asarray(direction, dtype=np.str_),
            "threshold": np.asarray(cutoff, dtype=np.float64),
            "scores": np.asarray(scores, dtype=np.float64),
            "flags": np.asarray(flags, dtype=np.bool_),
        }
    )
    _add_respondent_ids(payload, len(scores), respondent_ids)
    _write_npz_archive(path, payload)
