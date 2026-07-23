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
    options: IndexOptions,
) -> Literal["mean", "sum", "max"]:
    validate_index_names(indices, composite_index_names())

    if "evenodd" in indices and options.evenodd_factors is None:
        raise ValueError("evenodd_factors must be provided when using evenodd index")

    combine_method = "mean" if method == "best_subset" else method
    if combine_method not in ["mean", "sum", "max"]:
        raise ValueError("method must be 'mean', 'sum', 'max', or 'best_subset'")
    return combine_method


def _combine_scores(
    index_scores: dict[str, np.ndarray],
    diagnostics: dict[str, str],
    method: Literal["mean", "sum", "max"],
    standardize: bool,
) -> np.ndarray:
    if len(index_scores) == 0:
        failed = "; ".join(f"{name}: {msg}" for name, msg in sorted(diagnostics.items()))
        raise ValueError(f"no valid indices could be computed from the data. failures: {failed}")

    if standardize:
        standardized_scores = {}
        for name, scores in index_scores.items():
            valid_mask = ~np.isnan(scores)
            if np.sum(valid_mask) > 1:
                mean_val = np.nanmean(scores)
                std_val = np.nanstd(scores)
                if std_val > 0:
                    standardized_scores[name] = (scores - mean_val) / std_val
                else:
                    standardized_scores[name] = np.zeros_like(scores)
            else:
                standardized_scores[name] = scores
        index_scores = standardized_scores

    score_matrix = np.column_stack(list(index_scores.values()))

    match method:
        case "mean":
            mean_result: np.ndarray = np.nanmean(score_matrix, axis=1)
            return mean_result
        case "sum":
            sum_result: np.ndarray = np.nansum(score_matrix, axis=1)
            return sum_result
        case "max":
            max_result: np.ndarray = np.nanmax(score_matrix, axis=1)
            return max_result

    raise ValueError("method must be 'mean', 'sum', or 'max'")


def _resolve_options(
    options: IndexOptions | None,
    *,
    na_rm: bool,
    psychsyn_critval: float,
    psychant_critval: float,
    evenodd_factors: list[int] | None,
    mad_positive_items: list[int] | None,
    mad_negative_items: list[int] | None,
    mad_scale_max: int | None,
    scale_min: float | None,
    scale_max: float | None,
    acquiescence_positive_items: list[int] | None,
    acquiescence_negative_items: list[int] | None,
    longstring_max_pattern_length: int,
    midpoint_tolerance: float,
    guttman_normalize: bool,
    onset_window_size: int,
    onset_min_items: int,
    reliability_n_splits: int,
    reliability_random_seed: int | None,
    semantic_item_pairs: list[tuple[int, int]] | None,
    infrequency_item_indices: list[int] | None,
    infrequency_expected_responses: list[float] | None,
    infrequency_proportion: bool,
) -> IndexOptions:
    return resolve_index_options(
        options,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        psychant_critval=psychant_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
        scale_min=scale_min,
        scale_max=scale_max,
        acquiescence_positive_items=acquiescence_positive_items,
        acquiescence_negative_items=acquiescence_negative_items,
        longstring_max_pattern_length=longstring_max_pattern_length,
        midpoint_tolerance=midpoint_tolerance,
        guttman_normalize=guttman_normalize,
        onset_window_size=onset_window_size,
        onset_min_items=onset_min_items,
        reliability_n_splits=reliability_n_splits,
        reliability_random_seed=reliability_random_seed,
        semantic_item_pairs=semantic_item_pairs,
        infrequency_item_indices=infrequency_item_indices,
        infrequency_expected_responses=infrequency_expected_responses,
        infrequency_proportion=infrequency_proportion,
    )


def composite(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: CompositeMethod = "mean",
    standardize: bool = True,
    *,
    options: IndexOptions | None = None,
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    psychant_critval: float = -0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    acquiescence_positive_items: list[int] | None = None,
    acquiescence_negative_items: list[int] | None = None,
    longstring_max_pattern_length: int = 5,
    midpoint_tolerance: float = 0.0,
    guttman_normalize: bool = True,
    onset_window_size: int = 10,
    onset_min_items: int = 20,
    reliability_n_splits: int = 100,
    reliability_random_seed: int | None = None,
    semantic_item_pairs: list[tuple[int, int]] | None = None,
    infrequency_item_indices: list[int] | None = None,
    infrequency_expected_responses: list[float] | None = None,
    infrequency_proportion: bool = False,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, str]]:
    """
    Calculate a composite IER index combining multiple detection methods.

    This function computes multiple IER indices, standardizes them to z-scores,
    and combines them into a single composite score. Higher composite scores
    indicate greater likelihood of careless responding.

    Prefer passing a single ``IndexOptions`` via ``options=``. Legacy keyword
    arguments remain supported and are used only when ``options`` is omitted.

    The composite score is a sample-relative signal, not a calibrated probability
    of careless responding. Prefer multi-index agreement and substantive review
    over any single cutoff.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include. Options include: "irv", "longstring",
              "mahad", "psychsyn", "psychant", "evenodd", "person_total", "lz",
              "mad", "markov", "longstring_pattern", "guttman",
              "individual_reliability", "semantic_syn", "semantic_ant",
              "infrequency". Default includes NumPy-safe indices that do not
              require extra config.
    - method: How to combine indices. "mean" (default), "sum", "max", or
              "best_subset" (overrides indices to ["mad", "irv", "longstring", "lz"],
              falling back to ["irv", "longstring", "lz"] if MAD item info not provided).
    - standardize: If True (default), standardize each index to z-scores before combining.
    - options: Shared index configuration. When set, overrides individual kwargs.
    - na_rm / other kwargs: Legacy per-index configuration (see ``IndexOptions``).

    Returns:
    - A numpy array of composite scores for each individual. Higher scores indicate
      greater likelihood of careless responding.

    Raises:
    - ValueError: If invalid indices specified or evenodd requested without factors.

    Example:
        >>> from ier import IndexOptions, composite
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> scores = composite(data, options=IndexOptions())
        >>> print(scores)
    """
    x_array = validate_matrix_input(x, check_type=False)
    resolved = _resolve_options(
        options,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        psychant_critval=psychant_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
        scale_min=scale_min,
        scale_max=scale_max,
        acquiescence_positive_items=acquiescence_positive_items,
        acquiescence_negative_items=acquiescence_negative_items,
        longstring_max_pattern_length=longstring_max_pattern_length,
        midpoint_tolerance=midpoint_tolerance,
        guttman_normalize=guttman_normalize,
        onset_window_size=onset_window_size,
        onset_min_items=onset_min_items,
        reliability_n_splits=reliability_n_splits,
        reliability_random_seed=reliability_random_seed,
        semantic_item_pairs=semantic_item_pairs,
        infrequency_item_indices=infrequency_item_indices,
        infrequency_expected_responses=infrequency_expected_responses,
        infrequency_proportion=infrequency_proportion,
    )
    selected_indices = _resolve_composite_indices(indices, method, resolved)
    combine_method = _validate_composite_request(selected_indices, method, resolved)

    index_scores, diagnostics = score_registered_indices(
        x_array,
        selected_indices,
        resolved,
        apply_composite_direction=True,
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
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    psychant_critval: float = -0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    acquiescence_positive_items: list[int] | None = None,
    acquiescence_negative_items: list[int] | None = None,
    longstring_max_pattern_length: int = 5,
    midpoint_tolerance: float = 0.0,
    guttman_normalize: bool = True,
    onset_window_size: int = 10,
    onset_min_items: int = 20,
    reliability_n_splits: int = 100,
    reliability_random_seed: int | None = None,
    semantic_item_pairs: list[tuple[int, int]] | None = None,
    infrequency_item_indices: list[int] | None = None,
    infrequency_expected_responses: list[float] | None = None,
    infrequency_proportion: bool = False,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """
    Calculate composite IER scores and flag potential careless responders.

    Prefer ``options=IndexOptions(...)``. Legacy kwargs remain supported.

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
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        psychant_critval=psychant_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
        scale_min=scale_min,
        scale_max=scale_max,
        acquiescence_positive_items=acquiescence_positive_items,
        acquiescence_negative_items=acquiescence_negative_items,
        longstring_max_pattern_length=longstring_max_pattern_length,
        midpoint_tolerance=midpoint_tolerance,
        guttman_normalize=guttman_normalize,
        onset_window_size=onset_window_size,
        onset_min_items=onset_min_items,
        reliability_n_splits=reliability_n_splits,
        reliability_random_seed=reliability_random_seed,
        semantic_item_pairs=semantic_item_pairs,
        infrequency_item_indices=infrequency_item_indices,
        infrequency_expected_responses=infrequency_expected_responses,
        infrequency_proportion=infrequency_proportion,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        scores, diagnostics = composite_result
        assert isinstance(scores, np.ndarray)
    else:
        scores = composite_result
        assert isinstance(scores, np.ndarray)

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
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    psychant_critval: float = -0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    acquiescence_positive_items: list[int] | None = None,
    acquiescence_negative_items: list[int] | None = None,
    longstring_max_pattern_length: int = 5,
    midpoint_tolerance: float = 0.0,
    guttman_normalize: bool = True,
    onset_window_size: int = 10,
    onset_min_items: int = 20,
    reliability_n_splits: int = 100,
    reliability_random_seed: int | None = None,
    semantic_item_pairs: list[tuple[int, int]] | None = None,
    infrequency_item_indices: list[int] | None = None,
    infrequency_expected_responses: list[float] | None = None,
    infrequency_proportion: bool = False,
) -> CompositeSummary:
    """
    Calculate composite scores with detailed summary statistics.

    Prefer ``options=IndexOptions(...)``. Legacy kwargs remain supported.
    """
    x_array = validate_matrix_input(x, check_type=False)
    resolved = _resolve_options(
        options,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        psychant_critval=psychant_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
        scale_min=scale_min,
        scale_max=scale_max,
        acquiescence_positive_items=acquiescence_positive_items,
        acquiescence_negative_items=acquiescence_negative_items,
        longstring_max_pattern_length=longstring_max_pattern_length,
        midpoint_tolerance=midpoint_tolerance,
        guttman_normalize=guttman_normalize,
        onset_window_size=onset_window_size,
        onset_min_items=onset_min_items,
        reliability_n_splits=reliability_n_splits,
        reliability_random_seed=reliability_random_seed,
        semantic_item_pairs=semantic_item_pairs,
        infrequency_item_indices=infrequency_item_indices,
        infrequency_expected_responses=infrequency_expected_responses,
        infrequency_proportion=infrequency_proportion,
    )
    selected_indices = _resolve_composite_indices(indices, method, resolved)
    combine_method = _validate_composite_request(selected_indices, method, resolved)

    individual_scores, diagnostics = score_registered_indices(x_array, selected_indices, resolved)
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
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    psychant_critval: float = -0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    acquiescence_positive_items: list[int] | None = None,
    acquiescence_negative_items: list[int] | None = None,
    longstring_max_pattern_length: int = 5,
    midpoint_tolerance: float = 0.0,
    guttman_normalize: bool = True,
    onset_window_size: int = 10,
    onset_min_items: int = 20,
    reliability_n_splits: int = 100,
    reliability_random_seed: int | None = None,
    semantic_item_pairs: list[tuple[int, int]] | None = None,
    infrequency_item_indices: list[int] | None = None,
    infrequency_expected_responses: list[float] | None = None,
    infrequency_proportion: bool = False,
) -> np.ndarray:
    """
    Compute an uncalibrated logistic composite IER score.

    This function computes the standardized composite score and applies a
    logistic transformation to map it into the interval [0, 1]. The returned
    values are sample-relative scores, not calibrated probabilities of IER.

    Prefer ``options=IndexOptions(...)``. Legacy kwargs remain supported.
    """
    z_scores_result = composite(
        x,
        indices=indices,
        method=method,
        standardize=True,
        options=options,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        psychant_critval=psychant_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
        scale_min=scale_min,
        scale_max=scale_max,
        acquiescence_positive_items=acquiescence_positive_items,
        acquiescence_negative_items=acquiescence_negative_items,
        longstring_max_pattern_length=longstring_max_pattern_length,
        midpoint_tolerance=midpoint_tolerance,
        guttman_normalize=guttman_normalize,
        onset_window_size=onset_window_size,
        onset_min_items=onset_min_items,
        reliability_n_splits=reliability_n_splits,
        reliability_random_seed=reliability_random_seed,
        semantic_item_pairs=semantic_item_pairs,
        infrequency_item_indices=infrequency_item_indices,
        infrequency_expected_responses=infrequency_expected_responses,
        infrequency_proportion=infrequency_proportion,
    )
    z_scores = z_scores_result[0] if isinstance(z_scores_result, tuple) else z_scores_result

    result: np.ndarray = 1.0 / (1.0 + np.exp(-z_scores))
    return result
