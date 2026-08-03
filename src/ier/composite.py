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

from typing import Literal

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
from ier._validation import MatrixLike, validate_matrix_input
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


def _standardize_index_scores(scores: np.ndarray) -> np.ndarray:
    """Return z-scores while retaining established sparse and constant behavior."""
    valid_mask = ~np.isnan(scores)
    if np.sum(valid_mask) <= 1:
        return scores

    mean_val = np.nanmean(scores)
    std_val = np.nanstd(scores)
    return (scores - mean_val) / std_val if std_val > 0 else np.zeros_like(scores)


def _combine_scores(
    index_scores: dict[str, np.ndarray],
    diagnostics: dict[str, str],
    method: Literal["mean", "sum", "max"],
    standardize: bool,
) -> np.ndarray:
    if len(index_scores) == 0:
        failed = "; ".join(f"{name}: {msg}" for name, msg in sorted(diagnostics.items()))
        raise ValueError(f"no valid indices could be computed from the data. failures: {failed}")
    if method not in {"mean", "sum", "max"}:
        raise ValueError("method must be 'mean', 'sum', or 'max'")

    n_respondents = len(next(iter(index_scores.values())))
    if method == "max":
        combined = np.full(n_respondents, np.nan)
        valid_counts = None
    else:
        combined = np.zeros(n_respondents, dtype=float)
        valid_counts = np.zeros(n_respondents, dtype=np.int_) if method == "mean" else None

    for scores in index_scores.values():
        values = _standardize_index_scores(scores) if standardize else scores

        if method == "max":
            np.fmax(combined, values, out=combined)
            continue

        valid_mask = ~np.isnan(values)
        np.add(combined, values, out=combined, where=valid_mask)
        if valid_counts is not None:
            valid_counts += valid_mask

    if method == "mean":
        assert valid_counts is not None
        np.divide(combined, valid_counts, out=combined, where=valid_counts > 0)
        combined[valid_counts == 0] = np.nan

    return combined


def composite(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    standardize: bool = True,
    *,
    options: IndexOptions | None = None,
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

    index_scores, diagnostics = score_registered_indices(
        x_array,
        selected_indices,
        resolved,
        apply_composite_direction=True,
        strict=strict,
        workers=workers,
    )
    result = _combine_scores(index_scores, diagnostics, combine_method, standardize)

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
    return_diagnostics: bool = False,
    strict: bool = False,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """
    Calculate composite IER scores and flag potential careless responders.

    Configure with ``options=IndexOptions(...)``. Missing index config soft-fails
    by default; set ``strict=True`` to require every selected index to succeed.
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
    strict: bool = False,
    workers: int = 1,
) -> CompositeSummary:
    """
    Calculate composite scores with detailed summary statistics.

    Configure with ``options=IndexOptions(...)``. Set ``strict=True`` to require
    every selected index to succeed. Set ``workers`` above 1 to score independent
    indices concurrently.
    """
    workers = validate_worker_count(workers)
    x_array = validate_matrix_input(x, check_type=False)
    resolved = resolve_index_options(options)
    selected_indices = _resolve_composite_indices(indices, method, resolved)
    combine_method = _validate_composite_request(selected_indices, method)

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
    composite_scores = _combine_scores(composite_inputs, diagnostics, combine_method, standardize)

    valid_composite = composite_scores[~np.isnan(composite_scores)]

    return {
        "composite": composite_scores,
        "indices": individual_scores,
        "indices_used": list(individual_scores.keys()),
        "errors": diagnostics,
        "method": method,
        "standardized": standardize,
        "mean": float(np.nanmean(composite_scores)) if len(valid_composite) > 0 else float("nan"),
        "std": float(np.nanstd(composite_scores)) if len(valid_composite) > 0 else float("nan"),
        "min": float(np.nanmin(composite_scores)) if len(valid_composite) > 0 else float("nan"),
        "max": float(np.nanmax(composite_scores)) if len(valid_composite) > 0 else float("nan"),
        "n_total": len(composite_scores),
        "n_valid": int(np.sum(~np.isnan(composite_scores))),
    }


def composite_probability(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    *,
    options: IndexOptions | None = None,
    strict: bool = False,
    workers: int = 1,
) -> np.ndarray:
    """
    Compute an uncalibrated logistic composite IER score.

    This function computes the standardized composite score and applies a
    logistic transformation to map it into the interval [0, 1]. The returned
    values are sample-relative scores, not calibrated probabilities of IER.

    Configure with ``options=IndexOptions(...)``. Set ``strict=True`` to require
    every selected index to succeed. Set ``workers`` above 1 to score independent
    indices concurrently.
    """
    z_scores_result = composite(
        x,
        indices=indices,
        method=method,
        standardize=True,
        options=options,
        strict=strict,
        workers=workers,
    )
    z_scores = z_scores_result[0] if isinstance(z_scores_result, tuple) else z_scores_result

    result: np.ndarray = 1.0 / (1.0 + np.exp(-z_scores))
    return result
