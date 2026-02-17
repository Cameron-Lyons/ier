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

from typing import Any, Literal

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input
from ier.evenodd import evenodd
from ier.irv import irv
from ier.longstring import longstring_pattern, longstring_scores
from ier.lz import lz
from ier.mad import mad
from ier.mahad import mahad
from ier.markov import markov
from ier.person_total import person_total
from ier.psychsyn import psychsyn


def _add_error(errors: dict[str, str], index_name: str, error: Exception) -> None:
    errors[index_name] = str(error)


def composite(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: Literal["mean", "sum", "max", "best_subset"] = "mean",
    standardize: bool = True,
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, str]]:
    """
    Calculate a composite IER index combining multiple detection methods.

    This function computes multiple IER indices, standardizes them to z-scores,
    and combines them into a single composite score. Higher composite scores
    indicate greater likelihood of careless responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include. Options: "irv", "longstring", "mahad",
              "psychsyn", "evenodd", "person_total", "lz", "mad", "markov",
              "longstring_pattern". Default includes all except evenodd (which
              requires factor specification) and mad (which requires item info).
    - method: How to combine indices. "mean" (default), "sum", "max", or
              "best_subset" (overrides indices to ["mad", "irv", "longstring", "lz"],
              falling back to ["irv", "longstring", "lz"] if MAD item info not provided).
    - standardize: If True (default), standardize each index to z-scores before combining.
    - na_rm: Handle missing values in individual indices.
    - psychsyn_critval: Critical correlation value for psychometric synonyms (default 0.6).
    - evenodd_factors: Factor lengths for even-odd consistency. Required if "evenodd"
                      is in indices. List of integers where each integer is the number
                      of items in that factor (e.g., [5, 5, 5] for three 5-item scales).
    - mad_positive_items: Column indices for positively-worded items (for MAD index).
    - mad_negative_items: Column indices for negatively-worded items (for MAD index).
    - mad_scale_max: Maximum value of the response scale (for MAD index).

    Returns:
    - A numpy array of composite scores for each individual. Higher scores indicate
      greater likelihood of careless responding.

    Raises:
    - ValueError: If invalid indices specified or evenodd requested without factors.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> scores = composite(data)
        >>> print(scores)
        [-0.5, 1.2, -0.3]

        >>> scores = composite(data, indices=["irv", "longstring"])
        >>> print(scores)
        [-0.7, 1.5, -0.2]
    """
    x_array = validate_matrix_input(x, check_type=False)

    valid_indices = {
        "irv",
        "longstring",
        "mahad",
        "psychsyn",
        "evenodd",
        "person_total",
        "lz",
        "mad",
        "markov",
        "longstring_pattern",
    }

    if method == "best_subset":
        if mad_positive_items is not None and mad_negative_items is not None:
            indices = ["mad", "irv", "longstring", "lz"]
        else:
            indices = ["irv", "longstring", "lz"]

    if indices is None:
        indices = ["irv", "longstring", "mahad", "psychsyn", "person_total"]

    for idx in indices:
        if idx not in valid_indices:
            raise ValueError(f"invalid index '{idx}'. Valid options: {valid_indices}")

    if "evenodd" in indices and evenodd_factors is None:
        raise ValueError("evenodd_factors must be provided when using evenodd index")

    combine_method = "mean" if method == "best_subset" else method
    if combine_method not in ["mean", "sum", "max"]:
        raise ValueError("method must be 'mean', 'sum', 'max', or 'best_subset'")

    index_scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, str] = {}

    if "irv" in indices:
        try:
            index_scores["irv"] = -irv(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "irv", err)

    if "longstring" in indices:
        try:
            index_scores["longstring"] = longstring_scores(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "longstring", err)

    if "longstring_pattern" in indices:
        try:
            index_scores["longstring_pattern"] = longstring_pattern(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "longstring_pattern", err)

    if "mahad" in indices:
        try:
            mahad_result = mahad(x_array, na_rm=na_rm)
            if isinstance(mahad_result, np.ndarray):
                index_scores["mahad"] = mahad_result
        except ValueError as err:
            _add_error(diagnostics, "mahad", err)

    if "psychsyn" in indices:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                psyn_result = psychsyn(x_array, critval=psychsyn_critval, resample_na=na_rm)
            if isinstance(psyn_result, np.ndarray):
                index_scores["psychsyn"] = -psyn_result
        except ValueError as err:
            _add_error(diagnostics, "psychsyn", err)

    if "evenodd" in indices and evenodd_factors is not None:
        try:
            eo_result = evenodd(x_array, factors=evenodd_factors)
            if isinstance(eo_result, np.ndarray):
                index_scores["evenodd"] = -eo_result
        except ValueError as err:
            _add_error(diagnostics, "evenodd", err)

    if "person_total" in indices:
        try:
            index_scores["person_total"] = -person_total(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "person_total", err)

    if "lz" in indices:
        try:
            index_scores["lz"] = -lz(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "lz", err)

    if "mad" in indices:
        if mad_positive_items is None or mad_negative_items is None:
            diagnostics["mad"] = (
                "mad_positive_items and mad_negative_items must be provided when using mad index"
            )
        else:
            try:
                index_scores["mad"] = mad(
                    x_array,
                    positive_items=mad_positive_items,
                    negative_items=mad_negative_items,
                    scale_max=mad_scale_max,
                    na_rm=na_rm,
                )
            except ValueError as err:
                _add_error(diagnostics, "mad", err)

    if "markov" in indices:
        try:
            index_scores["markov"] = -markov(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "markov", err)

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

    if combine_method == "mean":
        result: np.ndarray = np.nanmean(score_matrix, axis=1)
    elif combine_method == "sum":
        result = np.nansum(score_matrix, axis=1)
    else:
        result = np.nanmax(score_matrix, axis=1)

    if return_diagnostics:
        return result, diagnostics
    return result


def composite_flag(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: Literal["mean", "sum", "max", "best_subset"] = "mean",
    threshold: float | None = None,
    percentile: float = 95.0,
    standardize: bool = True,
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """
    Calculate composite IER scores and flag potential careless responders.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include (see composite() for options).
    - method: How to combine indices ("mean", "sum", "max", or "best_subset").
    - threshold: Absolute threshold above which to flag. If None, uses percentile.
    - percentile: Percentile cutoff for flagging (default 95th percentile).
    - standardize: Standardize indices to z-scores before combining.
    - na_rm: Handle missing values.
    - psychsyn_critval: Critical value for psychometric synonyms.
    - evenodd_factors: Factor structure for even-odd consistency.
    - mad_positive_items: Column indices for positively-worded items (for MAD index).
    - mad_negative_items: Column indices for negatively-worded items (for MAD index).
    - mad_scale_max: Maximum value of the response scale (for MAD index).

    Returns:
    - Tuple of (composite_scores, flags) where flags is True for suspected
      careless responders.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> scores, flags = composite_flag(data)
        >>> print(flags)
        [False, True, False]
    """
    composite_result = composite(
        x,
        indices=indices,
        method=method,
        standardize=standardize,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        scores, diagnostics = composite_result
        assert isinstance(scores, np.ndarray)
    else:
        scores = composite_result
        assert isinstance(scores, np.ndarray)

    valid_scores = scores[~np.isnan(scores)]

    if threshold is None:
        if len(valid_scores) == 0:
            threshold = 0.0
        else:
            threshold = float(np.percentile(valid_scores, percentile))

    flags = np.zeros(len(scores), dtype=bool)
    valid_mask = ~np.isnan(scores)
    flags[valid_mask] = scores[valid_mask] > threshold

    if return_diagnostics:
        return scores, flags, diagnostics
    return scores, flags


def composite_summary(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: Literal["mean", "sum", "max", "best_subset"] = "mean",
    standardize: bool = True,
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
) -> dict[str, Any]:
    """
    Calculate composite scores with detailed summary statistics.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include.
    - method: Combination method ("mean", "sum", "max", or "best_subset").
    - standardize: Standardize before combining.
    - na_rm: Handle missing values.
    - psychsyn_critval: Critical value for psychometric synonyms.
    - evenodd_factors: Factor structure for even-odd consistency.
    - mad_positive_items: Column indices for positively-worded items (for MAD index).
    - mad_negative_items: Column indices for negatively-worded items (for MAD index).
    - mad_scale_max: Maximum value of the response scale (for MAD index).

    Returns:
    - Dictionary with composite scores, individual index scores, and statistics.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> summary = composite_summary(data)
        >>> print(summary.keys())
        dict_keys(['composite', 'indices', 'mean', 'std', 'min', 'max', ...])
    """
    x_array = validate_matrix_input(x, check_type=False)

    if method == "best_subset":
        if mad_positive_items is not None and mad_negative_items is not None:
            indices = ["mad", "irv", "longstring", "lz"]
        else:
            indices = ["irv", "longstring", "lz"]

    if indices is None:
        indices = ["irv", "longstring", "mahad", "psychsyn", "person_total"]

    if "evenodd" in indices and evenodd_factors is None:
        raise ValueError("evenodd_factors must be provided when using evenodd index")

    individual_scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, str] = {}

    if "irv" in indices:
        try:
            individual_scores["irv"] = irv(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "irv", err)

    if "longstring" in indices:
        try:
            individual_scores["longstring"] = longstring_scores(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "longstring", err)

    if "longstring_pattern" in indices:
        try:
            individual_scores["longstring_pattern"] = longstring_pattern(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "longstring_pattern", err)

    if "mahad" in indices:
        try:
            mahad_result = mahad(x_array, na_rm=na_rm)
            if isinstance(mahad_result, np.ndarray):
                individual_scores["mahad"] = mahad_result
        except ValueError as err:
            _add_error(diagnostics, "mahad", err)

    if "psychsyn" in indices:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                psyn_result = psychsyn(x_array, critval=psychsyn_critval, resample_na=na_rm)
            if isinstance(psyn_result, np.ndarray):
                individual_scores["psychsyn"] = psyn_result
        except ValueError as err:
            _add_error(diagnostics, "psychsyn", err)

    if "evenodd" in indices and evenodd_factors is not None:
        try:
            eo_result = evenodd(x_array, factors=evenodd_factors)
            if isinstance(eo_result, np.ndarray):
                individual_scores["evenodd"] = eo_result
        except ValueError as err:
            _add_error(diagnostics, "evenodd", err)

    if "person_total" in indices:
        try:
            individual_scores["person_total"] = person_total(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "person_total", err)

    if "lz" in indices:
        try:
            individual_scores["lz"] = lz(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "lz", err)

    if "mad" in indices and mad_positive_items is not None and mad_negative_items is not None:
        try:
            individual_scores["mad"] = mad(
                x_array,
                positive_items=mad_positive_items,
                negative_items=mad_negative_items,
                scale_max=mad_scale_max,
                na_rm=na_rm,
            )
        except ValueError as err:
            _add_error(diagnostics, "mad", err)
    elif "mad" in indices:
        diagnostics["mad"] = (
            "mad_positive_items and mad_negative_items must be provided when using mad index"
        )

    if "markov" in indices:
        try:
            individual_scores["markov"] = markov(x_array, na_rm=na_rm)
        except ValueError as err:
            _add_error(diagnostics, "markov", err)

    composite_scores_result = composite(
        x_array,
        indices=indices,
        method=method,
        standardize=standardize,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
    )
    composite_scores = (
        composite_scores_result[0]
        if isinstance(composite_scores_result, tuple)
        else composite_scores_result
    )

    valid_composite = composite_scores[~np.isnan(composite_scores)]

    return {
        "composite": composite_scores,
        "indices": individual_scores,
        "indices_used": list(individual_scores.keys()),
        "errors": diagnostics,
        "method": method,
        "standardized": standardize,
        "mean": float(np.nanmean(composite_scores)) if len(valid_composite) > 0 else np.nan,
        "std": float(np.nanstd(composite_scores)) if len(valid_composite) > 0 else np.nan,
        "min": float(np.nanmin(composite_scores)) if len(valid_composite) > 0 else np.nan,
        "max": float(np.nanmax(composite_scores)) if len(valid_composite) > 0 else np.nan,
        "n_total": len(composite_scores),
        "n_valid": int(np.sum(~np.isnan(composite_scores))),
    }


def composite_probability(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: Literal["mean", "sum", "max", "best_subset"] = "mean",
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
) -> np.ndarray:
    """
    Compute a probabilistic composite IER score using logistic transformation.

    Computes the standardized composite score and applies a logistic function
    to produce P(IER) between 0 and 1 per respondent.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include (see composite() for options).
    - method: How to combine indices ("mean", "sum", "max", or "best_subset").
    - na_rm: Handle missing values.
    - psychsyn_critval: Critical value for psychometric synonyms.
    - evenodd_factors: Factor structure for even-odd consistency.
    - mad_positive_items: Column indices for positively-worded items (for MAD index).
    - mad_negative_items: Column indices for negatively-worded items (for MAD index).
    - mad_scale_max: Maximum value of the response scale (for MAD index).

    Returns:
    - A numpy array of P(IER) values between 0 and 1 per respondent.
      Higher values indicate greater probability of careless responding.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> probs = composite_probability(data)
    """
    z_scores_result = composite(
        x,
        indices=indices,
        method=method,
        standardize=True,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
    )
    z_scores = z_scores_result[0] if isinstance(z_scores_result, tuple) else z_scores_result

    result: np.ndarray = 1.0 / (1.0 + np.exp(-z_scores))
    return result
