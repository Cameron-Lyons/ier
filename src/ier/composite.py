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

import warnings
from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from ier._composite_reductions import combine_mean_scores, combine_sum_max_scores
from ier._flagging import threshold_flags, validate_percentile, validate_threshold
from ier._registry import (
    INDEX_REGISTRY,
    IndexOptions,
    composite_index_names,
    default_composite_indices,
    numeric_override,
    resolve_index_options,
    resolve_index_overrides,
    score_registered_indices,
    validate_index_errors,
    validate_index_names,
    validate_min_valid_indices,
    validate_worker_count,
)
from ier._statistics import logistic_transform
from ier._summary import observed_summary_stats
from ier._validation import MatrixLike, validate_matrix_input, validate_score_vectors

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike

    from ier.types import CombineMethod, CompositeMethod, CompositeSummary, FloatArray, IntArray


@dataclass(frozen=True)
class _CompositeRun:
    """One reduced composite with the validated settings that produced it."""

    combined: FloatArray
    index_scores: dict[str, np.ndarray]
    diagnostics: dict[str, str]
    method: CompositeMethod
    standardize: bool
    weights: dict[str, float]
    min_valid_indices: int | None
    valid_index_counts: IntArray | None


def _select_composite_indices(
    indices: list[str] | None, method: CompositeMethod, options: IndexOptions
) -> list[str]:
    """Resolve and validate the indices one raw-matrix composite scores."""
    if method == "best_subset":
        if indices is not None:
            # Level 4 attributes the warning to the caller of the public entry point.
            warnings.warn(
                "indices is ignored when method='best_subset'; this will raise in a future release",
                DeprecationWarning,
                stacklevel=4,
            )
        mad_items = (
            options.mad_positive_items is not None and options.mad_negative_items is not None
        )
        indices = ["mad", "irv", "longstring", "lz"] if mad_items else ["irv", "longstring", "lz"]
    elif indices is None:
        indices = default_composite_indices()
    validate_index_names(indices, composite_index_names())
    # Accept array-like selections (NumPy arrays, pandas Index) as screen() does.
    selected = [str(name) for name in indices]
    if len(selected) == 0:
        raise ValueError("indices must name at least one registered index")
    # Tuple membership keeps unhashable values on the documented ValueError path.
    if method not in ("mean", "sum", "max", "best_subset"):
        raise ValueError("method must be 'mean', 'sum', 'max', or 'best_subset'")
    return selected


_positive_weight = numeric_override(
    "weight", "a positive finite number", lambda value: 0 < value < inf
)


def _resolve_composite_weights(
    weights: Mapping[str, float] | None,
    indices: list[str],
) -> dict[str, float]:
    """Validate partial weight overrides and return every effective index weight."""
    overrides = resolve_index_overrides(weights, indices, label="weight", convert=_positive_weight)
    return {**dict.fromkeys(indices, 1.0), **overrides}


def _validate_standardize(standardize: bool) -> bool:
    """Return a validated score-standardization control."""
    if not isinstance(standardize, bool):
        raise ValueError("standardize must be a boolean")
    return standardize


def _combine_scores(
    index_scores: dict[str, np.ndarray],
    diagnostics: dict[str, str],
    method: CombineMethod,
    standardize: bool,
    weights: Mapping[str, float] | None = None,
    *,
    min_valid_indices: int | None = None,
    valid_counts_out: np.ndarray | None = None,
    multipliers: Mapping[str, float] | None = None,
) -> FloatArray:
    if len(index_scores) == 0:
        failed = "; ".join(f"{name}: {msg}" for name, msg in sorted(diagnostics.items()))
        raise ValueError(f"no valid indices could be computed from the data. failures: {failed}")

    n_respondents = len(next(iter(index_scores.values())))
    if valid_counts_out is not None:
        if valid_counts_out.shape != (n_respondents,) or valid_counts_out.dtype.kind not in "iu":
            raise ValueError("valid_counts_out must be a respondent-length integer array")
        valid_counts_out.fill(0)

    if method == "mean":
        return combine_mean_scores(
            index_scores, standardize, weights, min_valid_indices, valid_counts_out, multipliers
        )

    return combine_sum_max_scores(
        index_scores, method, standardize, weights, min_valid_indices, valid_counts_out, multipliers
    )


def _reduce_prepared(
    scores: dict[str, np.ndarray],
    errors: dict[str, str],
    n_respondents: int,
    method: CompositeMethod,
    standardize: bool,
    weights: dict[str, float],
    weighted: bool,
    min_valid_indices: int | None,
    count_valid: bool,
) -> _CompositeRun:
    """Reduce validated components once, applying registered directions."""
    counts = np.zeros(n_respondents, dtype=np.int_) if count_valid else None
    combined = _combine_scores(
        scores,
        errors,
        "mean" if method == "best_subset" else method,
        standardize,
        # Unweighted requests keep the faster equal-weight reductions.
        weights if weighted else None,
        min_valid_indices=min_valid_indices,
        valid_counts_out=counts,
        multipliers={name: INDEX_REGISTRY[name].composite_multiplier for name in scores},
    )
    return _CompositeRun(
        combined, scores, errors, method, standardize, weights, min_valid_indices, counts
    )


def _run_composite(
    x: MatrixLike,
    indices: list[str] | None,
    method: CompositeMethod,
    standardize: bool,
    options: IndexOptions | None,
    weights: Mapping[str, float] | None,
    min_valid_indices: int | None,
    strict: bool,
    workers: int,
    *,
    count_valid: bool = False,
) -> _CompositeRun:
    """Validate one raw-matrix request, score its indices, and reduce them once."""
    workers = validate_worker_count(workers)
    standardize = _validate_standardize(standardize)
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    resolved = resolve_index_options(options)
    selected = _select_composite_indices(indices, method, resolved)
    resolved_weights = _resolve_composite_weights(weights, selected)
    min_valid_indices = validate_min_valid_indices(min_valid_indices, len(selected))
    x_array = validate_matrix_input(x)
    index_scores, diagnostics = score_registered_indices(
        x_array, selected, resolved, strict=strict, workers=workers, validated=True
    )
    return _reduce_prepared(
        index_scores,
        diagnostics,
        len(x_array),
        method,
        standardize,
        resolved_weights,
        weights is not None,
        min_valid_indices,
        count_valid,
    )


def composite_scores(
    scores: Mapping[str, ArrayLike],
    method: CombineMethod = "mean",
    standardize: bool = True,
    *,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    errors: Mapping[str, str] | None = None,
    unsupported: Literal["error", "drop"] = "error",
) -> FloatArray:
    """
    Combine already-computed registered-index score vectors.

    This reusable counterpart to :func:`composite` supports fast weight,
    standardization, completeness, and reduction-method sensitivity analysis
    without calculating any index again. Input scores use their original public
    directions; low-is-suspicious indices are reversed automatically so higher
    composite values consistently represent more evidence of careless responding.

    Score mappings preserve insertion order. Every vector must be non-empty,
    one-dimensional, respondent-aligned, and contain only finite values or
    ``NaN``. The function does not mutate input arrays.

    Parameters:
    - scores: Mapping from composite-enabled registered index names to raw score vectors.
    - method: Reduction across available directed scores: ``"mean"``, ``"sum"``,
              or ``"max"``.
    - standardize: Standardize each index before applying direction and weights.
    - weights: Optional positive finite per-index weight overrides.
    - min_valid_indices: Optional minimum available component count per respondent.
    - errors: Optional retained per-index soft failures. Failed indices remain
              selected for weight and completeness validation, with no available scores.
    - unsupported: ``"error"`` (default) rejects registered indices that are not
                   composite-enabled. ``"drop"`` omits them from ``scores`` and
                   ``errors`` so default :func:`screen` results can be reused
                   directly. Unknown index names raise in both modes.

    Returns:
    - A respondent-aligned NumPy array of composite scores.

    Raises:
    - ValueError: If a final weighted sum or maximum exceeds the finite float
                  range. Reduce weights or choose ``method="mean"``.

    Example:
        >>> import numpy as np
        >>> from ier import composite_scores, composite_summary, screen
        >>> data = [
        ...     [1, 2, 3, 4, 5, 4],
        ...     [3, 3, 3, 3, 3, 3],
        ...     [5, 4, 3, 2, 1, 2],
        ...     [2, 5, 1, 4, 3, 2],
        ...     [4, 4, 5, 4, 4, 5],
        ...     [1, 5, 1, 5, 1, 5],
        ... ]
        >>> initial = composite_summary(data, indices=["irv", "longstring"])
        >>> weighted = composite_scores(
        ...     initial["indices"],
        ...     weights={"irv": 2.0, "longstring": 0.5},
        ... )
        >>> np.round(weighted, 2).tolist()
        [-0.43, 1.76, -0.43, -0.43, 0.75, -1.22]
        >>> screened = screen(data)  # includes screen-only u3_poly, midpoint, and acquiescence
        >>> reused = composite_scores(
        ...     screened["scores"], errors=screened["errors"], unsupported="drop"
        ... )
        >>> reused.shape
        (6,)
    """
    return _precomputed_run(
        scores, method, standardize, weights, min_valid_indices, errors, unsupported
    ).combined


def _precomputed_run(
    scores: Mapping[str, ArrayLike],
    method: CombineMethod,
    standardize: bool,
    weights: Mapping[str, float] | None,
    min_valid_indices: int | None,
    errors: Mapping[str, str] | None,
    unsupported: Literal["error", "drop"],
    *,
    count_valid: bool = False,
) -> _CompositeRun:
    """Validate and reduce one precomputed request without scoring any index."""
    if unsupported not in ("error", "drop"):
        raise ValueError("unsupported must be 'error' or 'drop'")
    validated_scores, n_respondents = validate_score_vectors(scores)
    allowed = composite_index_names()
    # Dropping still rejects unknown names; it omits only registered screen-only indices.
    checked = None if unsupported == "drop" else allowed
    validate_index_names(list(validated_scores), checked)
    retained_errors = validate_index_errors(errors, list(validated_scores), checked)
    if unsupported == "drop":
        dropped = [name for name in validated_scores if name not in allowed]
        validated_scores = {name: arr for name, arr in validated_scores.items() if name in allowed}
        retained_errors = {name: msg for name, msg in retained_errors.items() if name in allowed}
        if not validated_scores:
            raise ValueError(
                "scores must contain at least one composite-enabled index after dropping "
                f"unsupported indices: {', '.join(dropped)}"
            )
    indices = [*validated_scores, *retained_errors]
    if method not in ("mean", "sum", "max"):
        raise ValueError("method must be 'mean', 'sum', or 'max' for precomputed scores")
    standardize = _validate_standardize(standardize)
    resolved_weights = _resolve_composite_weights(weights, indices)
    min_valid_indices = validate_min_valid_indices(min_valid_indices, len(indices))
    return _reduce_prepared(
        validated_scores,
        retained_errors,
        n_respondents,
        method,
        standardize,
        resolved_weights,
        weights is not None,
        min_valid_indices,
        count_valid,
    )


def composite_scores_summary(
    scores: Mapping[str, ArrayLike],
    method: CombineMethod = "mean",
    standardize: bool = True,
    *,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    errors: Mapping[str, str] | None = None,
    unsupported: Literal["error", "drop"] = "error",
) -> CompositeSummary:
    """Combine reusable scores with component coverage and summary statistics.

    This is the post-scoring counterpart to :func:`composite_summary`. It uses
    the same directions, calibration, weights, and completeness rules as
    :func:`composite_scores`, and reports availability after calibration. Input
    arrays are never mutated; compatible arrays are retained in ``indices``.
    Optional ``errors`` retains original soft failures without recalculating
    indices. Failed indices remain selected for weight and completeness
    validation but never contribute to coverage or scores. Set
    ``unsupported="drop"`` to omit registered indices that are not
    composite-enabled, such as screen-only defaults.
    """
    run = _precomputed_run(
        scores,
        method,
        standardize,
        weights,
        min_valid_indices,
        errors,
        unsupported,
        count_valid=True,
    )
    return _summarize_composite_result(run)


@overload
def composite(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    standardize: bool = True,
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: Literal[False] = False,
    strict: bool = False,
    workers: int = 1,
) -> FloatArray: ...


@overload
def composite(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    standardize: bool = True,
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: Literal[True],
    strict: bool = False,
    workers: int = 1,
) -> tuple[FloatArray, dict[str, str]]: ...


@overload
def composite(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    standardize: bool = True,
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: bool,
    strict: bool = False,
    workers: int = 1,
) -> FloatArray | tuple[FloatArray, dict[str, str]]: ...


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
) -> FloatArray | tuple[FloatArray, dict[str, str]]:
    """
    Calculate a composite IER index combining multiple detection methods.

    This function computes multiple IER indices, standardizes them to z-scores,
    and combines them into a single composite score. Higher composite scores
    indicate greater likelihood of careless responding.

    Configure indices with a single ``IndexOptions`` via ``options=``. By default,
    missing required config is recorded in diagnostics without aborting other
    indices. Set ``strict=True`` to require every selected index to succeed.
    ``IndexOptions.reverse_keyed_items`` are reverse-scored only for components
    that use keyed responses, such as ``lz``; the others read ``x``.

    The composite score is a sample-relative signal, not a calibrated probability
    of careless responding. Prefer multi-index agreement and substantive review
    over any single cutoff.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include. Options include: "irv", "longstring",
              "mahad", "psychsyn", "psychant", "evenodd", "person_total", "lz",
              "mad", "markov", "longstring_pattern", "guttman",
              "individual_reliability", "semantic_syn", "semantic_ant",
              "infrequency", "missing_rate", "avgstr", "autocorrelation". Default
              includes NumPy-safe indices that do not require extra config.
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
    - ValueError: If invalid or no indices are specified, no index succeeds, or a
                  final weighted sum or maximum exceeds the finite float range,
                  including an infinite score returned by an index.
    - TypeError: If ``weights`` is neither a mapping nor a mapping-like object
                 with ``items()``, such as a pandas Series.

    Passing ``indices`` with ``method="best_subset"`` is deprecated: the explicit
    indices are ignored with a ``DeprecationWarning`` and will raise in a future release.

    Example:
        >>> import numpy as np
        >>> from ier import IndexOptions, composite
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> scores, errors = composite(data, options=IndexOptions(), return_diagnostics=True)
        >>> np.round(scores, 2).tolist()
        [-0.71, 1.41, -0.71]
        >>> list(errors)  # three respondents cannot support a five-item covariance
        ['mahad']
    """
    run = _run_composite(
        x, indices, method, standardize, options, weights, min_valid_indices, strict, workers
    )
    if return_diagnostics:
        return run.combined, run.diagnostics
    return run.combined


@overload
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
    return_diagnostics: Literal[False] = False,
    strict: bool = False,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
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
    return_diagnostics: Literal[True],
    strict: bool = False,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]: ...


@overload
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
    return_diagnostics: bool,
    strict: bool = False,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, str]]: ...


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
    ``composite()``, including range errors for unrepresentable sums or maxima.
    Set ``min_valid_indices`` to suppress scores based on too few available indices.
    Set ``workers`` above 1 to score independent indices concurrently.
    Explicit thresholds include scores equal to the cutoff; percentile cutoffs
    flag only scores strictly above the sample percentile.

    Returns:
    - Tuple of (composite_scores, flags) where flags is True for suspected
      careless responders.
    """
    # Decision arguments fail before the matrix is converted or any index is scored.
    threshold = validate_threshold(threshold)
    percentile = validate_percentile(percentile)
    run = _run_composite(
        x, indices, method, standardize, options, weights, min_valid_indices, strict, workers
    )
    flags = threshold_flags(
        run.combined, threshold=threshold, percentile=percentile, direction="high"
    )
    if return_diagnostics:
        return run.combined, flags, run.diagnostics
    return run.combined, flags


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
    Unrepresentable weighted sums or maxima raise ``ValueError``, as in ``composite()``.
    """
    run = _run_composite(
        x,
        indices,
        method,
        standardize,
        options,
        weights,
        min_valid_indices,
        strict,
        workers,
        count_valid=True,
    )
    return _summarize_composite_result(run)


def _summarize_composite_result(run: _CompositeRun) -> CompositeSummary:
    """Describe one reduced composite without scoring or calibrating it again."""
    combined_scores = run.combined
    assert run.valid_index_counts is not None
    available = ~np.isnan(combined_scores)
    n_valid = int(np.count_nonzero(available))
    valid_composite = (
        combined_scores if n_valid == len(combined_scores) else combined_scores[available]
    )
    stats = observed_summary_stats(valid_composite)

    return {
        "composite": combined_scores,
        "indices": run.index_scores,
        "indices_used": list(run.index_scores),
        "errors": run.diagnostics,
        "method": run.method,
        "standardized": run.standardize,
        "weights": run.weights,
        "min_valid_indices": run.min_valid_indices,
        "valid_index_counts": run.valid_index_counts,
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "n_total": len(combined_scores),
        "n_valid": n_valid,
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
) -> FloatArray: ...


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
) -> tuple[FloatArray, dict[str, str]]: ...


@overload
def composite_probability(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    *,
    options: IndexOptions | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    return_diagnostics: bool,
    strict: bool = False,
    workers: int = 1,
) -> FloatArray | tuple[FloatArray, dict[str, str]]: ...


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
) -> FloatArray | tuple[FloatArray, dict[str, str]]:
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
    Unrepresentable weighted sums or maxima raise ``ValueError`` before transformation.
    """
    run = _run_composite(
        x, indices, method, True, options, weights, min_valid_indices, strict, workers
    )
    result = logistic_transform(run.combined)
    if return_diagnostics:
        return result, run.diagnostics
    return result
