"""
Composite index combining multiple IER detection indices.

Research suggests combining multiple indices improves detection accuracy. The "Best Subset"
approach (Curran, 2016; Meade & Craig, 2012) recommends combining indices that capture
different types of careless responding: consistency-based, pattern-based, and outlier-based.

References:
- Curran, P. G. (2016). Methods for the detection of carelessly invalid responses in
  survey data. Journal of Experimental Social Psychology, 66, 4-19.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from ier._flagging import threshold_flags
from ier._registry import (
    INDEX_REGISTRY,
    IndexOptions,
    composite_index_names,
    default_composite_indices,
    resolve_index_options,
    score_registered_indices,
    validate_index_names,
    validate_worker_count,
)
from ier._statistics import logistic_transform
from ier._validation import MatrixLike, validate_matrix_input

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ier.types import CompositeMethod, CompositeSummary


def _resolve_composite_indices(
    indices: list[str] | None,
    method: Literal["mean", "sum", "max", "best_subset"],
    options: IndexOptions,
) -> list[str]:
    if method == "best_subset":
        if options.mad_positive_items is not None and options.mad_negative_items is not None:
            return ["mad", "irv", "longstring", "lz"]
        return ["irv", "longstring", "lz"]

    if indices is None:
        return default_composite_indices()
    return indices


def _validate_composite_request(
    indices: list[str],
    method: Literal["mean", "sum", "max", "best_subset"],
) -> Literal["mean", "sum", "max"]:
    validate_index_names(indices, composite_index_names())

    combine_method = "mean" if method == "best_subset" else method
    if combine_method not in ["mean", "sum", "max"]:
        raise ValueError("method must be 'mean', 'sum', 'max', or 'best_subset'")
    return combine_method


def _resolve_composite_weights(
    weights: Mapping[str, float] | None,
    indices: list[str],
) -> dict[str, float]:
    """Validate partial weight overrides and return every effective index weight."""
    resolved = dict.fromkeys(indices, 1.0)
    if weights is None:
        return resolved

    for name, value in weights.items():
        if name not in resolved:
            raise ValueError(f"weight index is not selected: {name}")
        if isinstance(value, bool):
            raise ValueError(f"weight for {name} must be a positive finite number")
        try:
            weight = float(value)
        except (TypeError, ValueError) as err:
            raise ValueError(f"weight for {name} must be a positive finite number") from err
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight for {name} must be a positive finite number")
        resolved[name] = weight
    return resolved


def _validate_min_valid_indices(
    min_valid_indices: int | None,
    n_selected_indices: int,
) -> int | None:
    """Validate an optional respondent-level composite completeness requirement."""
    if min_valid_indices is None:
        return None
    if (
        isinstance(min_valid_indices, bool)
        or not isinstance(min_valid_indices, int)
        or min_valid_indices < 1
    ):
        raise ValueError("min_valid_indices must be a positive integer or None")
    if min_valid_indices > n_selected_indices:
        raise ValueError(
            f"min_valid_indices cannot exceed the number of selected indices ({n_selected_indices})"
        )
    return min_valid_indices


def _standardize_index_scores(scores: np.ndarray) -> np.ndarray:
    """Return z-scores while retaining established sparse and constant behavior."""
    valid_mask = ~np.isnan(scores)
    if np.sum(valid_mask) <= 1:
        return scores

    mean_val = float(np.nanmean(scores))
    std_val = float(np.nanstd(scores))
    if std_val > 0:
        return (scores - mean_val) / std_val

    standardized = np.zeros_like(scores)
    standardized[np.isnan(scores)] = np.nan
    return standardized


def _combine_scores(
    index_scores: dict[str, np.ndarray],
    diagnostics: dict[str, str],
    method: Literal["mean", "sum", "max"],
    standardize: bool,
    weights: Mapping[str, float] | None = None,
    *,
    min_valid_indices: int | None = None,
    valid_counts_out: np.ndarray | None = None,
) -> np.ndarray:
    if len(index_scores) == 0:
        failed = "; ".join(f"{name}: {msg}" for name, msg in sorted(diagnostics.items()))
        raise ValueError(f"no valid indices could be computed from the data. failures: {failed}")
    if method not in {"mean", "sum", "max"}:
        raise ValueError("method must be 'mean', 'sum', or 'max'")

    n_respondents = len(next(iter(index_scores.values())))
    if valid_counts_out is not None:
        if valid_counts_out.shape != (n_respondents,) or valid_counts_out.dtype.kind not in "iu":
            raise ValueError("valid_counts_out must be a respondent-length integer array")
        valid_counts_out.fill(0)

    needs_separate_counts = min_valid_indices is not None and not (
        method == "mean" and weights is None
    )
    tracked_counts = valid_counts_out
    if tracked_counts is None and needs_separate_counts:
        tracked_counts = np.zeros(n_respondents, dtype=np.int_)

    if method == "max":
        combined = np.full(n_respondents, np.nan)
        denominators = None
    else:
        combined = np.zeros(n_respondents, dtype=float)
        denominators = (
            np.zeros(
                n_respondents,
                dtype=float if weights is not None else np.int_,
            )
            if method == "mean"
            else None
        )

    for name, scores in index_scores.items():
        values = _standardize_index_scores(scores) if standardize else scores
        weight = weights.get(name, 1.0) if weights is not None else 1.0
        weighted_values = values * weight if weight != 1.0 else values
        valid_mask = ~np.isnan(weighted_values)
        if tracked_counts is not None:
            tracked_counts += valid_mask

        if method == "max":
            np.fmax(combined, weighted_values, out=combined)
            continue

        np.add(combined, weighted_values, out=combined, where=valid_mask)
        if denominators is not None:
            if weights is None:
                denominators += valid_mask
            else:
                np.add(denominators, weight, out=denominators, where=valid_mask)

    if method == "mean":
        assert denominators is not None
        np.divide(combined, denominators, out=combined, where=denominators > 0)
        combined[denominators == 0] = np.nan

    if min_valid_indices is not None:
        available_counts = (
            denominators
            if method == "mean" and weights is None and valid_counts_out is None
            else tracked_counts
        )
        assert available_counts is not None
        combined[available_counts < min_valid_indices] = np.nan

    return combined


def composite(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    standardize: bool = True,
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: bool = False,
    strict: bool = False,
    workers: int = 1,
) -> np.ndarray | tuple[np.ndarray, dict[str, str]]:
    """
    Calculate a composite IER index combining multiple detection methods.

    This function computes multiple IER indices, standardizes them to z-scores,
    and combines them into a single composite score. Higher composite scores
    indicate greater likelihood of careless responding.

    Configure indices with a single ``IndexOptions`` via ``options=``. By default,
    missing required config is recorded in diagnostics without aborting other
    indices. Set ``strict=True`` to require every selected index to succeed.

    The composite score is a sample-relative signal, not a calibrated probability
    of careless responding. Prefer multi-index agreement and substantive review
    over any single cutoff.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include. Options include: "irv", "longstring",
              "mahad", "psychsyn", "psychant", "evenodd", "person_total", "lz",
              "mad", "markov", "longstring_pattern", "guttman",
              "individual_reliability", "semantic_syn", "semantic_ant",
              "infrequency", "missing_rate". Default includes NumPy-safe indices that do not
              require extra config.
    - method: How to combine indices. "mean" (default), "sum", "max", or
              "best_subset" (overrides indices to ["mad", "irv", "longstring", "lz"],
              falling back to ["irv", "longstring", "lz"] if MAD item info not provided).
    - standardize: If True (default), standardize each index to z-scores before combining.
    - options: Shared index configuration (``IndexOptions``).
    - weights: Optional positive finite per-index weight overrides. Unspecified
               selected indices retain weight 1. Weighting is applied after
               direction correction and optional standardization.
    - min_valid_indices: Optional minimum number of available index scores required
                         per respondent. Respondents below the minimum receive NaN.
    - return_diagnostics: If True, also return per-index soft-failure messages.
    - strict: If True, raise when any selected index fails instead of collecting
              diagnostics (default False).
    - workers: Number of indices to score concurrently. The default of 1 preserves
               sequential execution; values above 1 trade additional peak memory
               for throughput on larger matrices.

    Returns:
    - A numpy array of composite scores for each individual. Higher scores indicate
      greater likelihood of careless responding.

    Raises:
    - ValueError: If invalid indices are specified, or no index succeeds.

    Example:
        >>> from ier import IndexOptions, composite
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> scores = composite(data, options=IndexOptions())
        >>> print(scores)
    """
    workers = validate_worker_count(workers)
    x_array = validate_matrix_input(x, check_type=False)
    resolved = resolve_index_options(options)
    selected_indices = _resolve_composite_indices(indices, method, resolved)
    combine_method = _validate_composite_request(selected_indices, method)
    resolved_weights = _resolve_composite_weights(weights, selected_indices)
    min_valid_indices = _validate_min_valid_indices(min_valid_indices, len(selected_indices))

    index_scores, diagnostics = score_registered_indices(
        x_array,
        selected_indices,
        resolved,
        apply_composite_direction=True,
        strict=strict,
        workers=workers,
    )
    result = _combine_scores(
        index_scores,
        diagnostics,
        combine_method,
        standardize,
        resolved_weights if weights is not None else None,
        min_valid_indices=min_valid_indices,
    )

    if return_diagnostics:
        return result, diagnostics
    return result


def composite_flag(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    threshold: float | None = None,
    percentile: float = 95.0,
    standardize: bool = True,
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: bool = False,
    strict: bool = False,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """
    Calculate composite IER scores and flag potential careless responders.

    Configure with ``options=IndexOptions(...)``. Missing index config soft-fails
    by default; set ``strict=True`` to require every selected index to succeed.
    Optional ``weights`` follow the same validation and combination semantics as
    ``composite()``.
    Set ``min_valid_indices`` to suppress scores based on too few available indices.
    Set ``workers`` above 1 to score independent indices concurrently.
    Explicit thresholds include scores equal to the cutoff; percentile cutoffs
    flag only scores strictly above the sample percentile.

    Returns:
    - Tuple of (composite_scores, flags) where flags is True for suspected
      careless responders.
    """
    composite_result = composite(
        x,
        indices=indices,
        method=method,
        standardize=standardize,
        options=options,
        weights=weights,
        min_valid_indices=min_valid_indices,
        return_diagnostics=return_diagnostics,
        strict=strict,
        workers=workers,
    )
    if return_diagnostics:
        if not isinstance(composite_result, tuple):
            raise TypeError("expected (scores, diagnostics) when return_diagnostics=True")
        scores, diagnostics = composite_result
    else:
        if isinstance(composite_result, tuple):
            raise TypeError("unexpected diagnostics tuple when return_diagnostics=False")
        scores = composite_result

    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="high")

    if return_diagnostics:
        return scores, flags, diagnostics
    return scores, flags


def composite_summary(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    standardize: bool = True,
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    strict: bool = False,
    workers: int = 1,
) -> CompositeSummary:
    """
    Calculate composite scores with detailed summary statistics.

    Configure with ``options=IndexOptions(...)``. Set ``strict=True`` to require
    every selected index to succeed. Set ``workers`` above 1 to score independent
    indices concurrently. The returned ``weights`` mapping contains every
    resolved selected-index weight. ``valid_index_counts`` reports the available
    component count for each respondent before applying ``min_valid_indices``.
    """
    workers = validate_worker_count(workers)
    x_array = validate_matrix_input(x, check_type=False)
    resolved = resolve_index_options(options)
    selected_indices = _resolve_composite_indices(indices, method, resolved)
    combine_method = _validate_composite_request(selected_indices, method)
    resolved_weights = _resolve_composite_weights(weights, selected_indices)
    min_valid_indices = _validate_min_valid_indices(min_valid_indices, len(selected_indices))

    individual_scores, diagnostics = score_registered_indices(
        x_array,
        selected_indices,
        resolved,
        strict=strict,
        workers=workers,
    )
    composite_inputs = {
        name: INDEX_REGISTRY[name].composite_multiplier * scores
        for name, scores in individual_scores.items()
    }
    valid_index_counts = np.zeros(len(x_array), dtype=np.int_)
    composite_scores = _combine_scores(
        composite_inputs,
        diagnostics,
        combine_method,
        standardize,
        resolved_weights if weights is not None else None,
        min_valid_indices=min_valid_indices,
        valid_counts_out=valid_index_counts,
    )

    valid_composite = composite_scores[~np.isnan(composite_scores)]

    return {
        "composite": composite_scores,
        "indices": individual_scores,
        "indices_used": list(individual_scores.keys()),
        "errors": diagnostics,
        "method": method,
        "standardized": standardize,
        "weights": resolved_weights,
        "min_valid_indices": min_valid_indices,
        "valid_index_counts": valid_index_counts,
        "mean": float(np.nanmean(composite_scores)) if len(valid_composite) > 0 else float("nan"),
        "std": float(np.nanstd(composite_scores)) if len(valid_composite) > 0 else float("nan"),
        "min": float(np.nanmin(composite_scores)) if len(valid_composite) > 0 else float("nan"),
        "max": float(np.nanmax(composite_scores)) if len(valid_composite) > 0 else float("nan"),
        "n_total": len(composite_scores),
        "n_valid": int(np.sum(~np.isnan(composite_scores))),
    }


@overload
def composite_probability(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: Literal[False] = False,
    strict: bool = False,
    workers: int = 1,
) -> np.ndarray: ...


@overload
def composite_probability(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: Literal[True],
    strict: bool = False,
    workers: int = 1,
) -> tuple[np.ndarray, dict[str, str]]: ...


def composite_probability(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: bool = False,
    strict: bool = False,
    workers: int = 1,
) -> np.ndarray | tuple[np.ndarray, dict[str, str]]:
    """
    Compute an uncalibrated logistic composite IER score.

    This function computes the standardized composite score and applies a
    logistic transformation to map it into the interval [0, 1]. The returned
    values are sample-relative scores, not calibrated probabilities of IER.

    Configure with ``options=IndexOptions(...)``. Set ``strict=True`` to require
    every selected index to succeed. Set ``workers`` above 1 to score independent
    indices concurrently. Optional ``weights`` are applied before the logistic
    transform. ``min_valid_indices`` applies the same completeness rule as
    ``composite()`` before transformation. Set ``return_diagnostics=True`` to
    also receive ordered per-index soft-failure messages.
    """
    z_scores_result = composite(
        x,
        indices=indices,
        method=method,
        standardize=True,
        options=options,
        weights=weights,
        min_valid_indices=min_valid_indices,
        return_diagnostics=return_diagnostics,
        strict=strict,
        workers=workers,
    )
    if return_diagnostics:
        if not isinstance(z_scores_result, tuple):
            raise TypeError("expected (scores, diagnostics) when return_diagnostics=True")
        z_scores, diagnostics = z_scores_result
    else:
        if isinstance(z_scores_result, tuple):
            raise TypeError("unexpected diagnostics tuple when return_diagnostics=False")
        z_scores = z_scores_result

    result = logistic_transform(z_scores)
    if return_diagnostics:
        return result, diagnostics
    return result
