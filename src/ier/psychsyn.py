"""
Takes a matrix of item responses and identifies item pairs that are highly correlated within the
overall dataset. What defines "highly correlated" is set by the critical value (e.g., r > .60). Each
respondents' psychometric synonym score is then computed as the within-person correlation be-
tween the identified item-pairs. Alternatively computes the psychometric antonym score which is a
variant that uses item pairs that are highly negatively correlated.

This module provides functions for detecting careless responding patterns by analyzing how
individuals respond to psychometrically similar (synonym) or opposite (antonym) items.
"""

from typing import Any, Literal, overload

import numpy as np

from ier._summary import calculate_summary_stats
from ier._validation import MatrixLike, validate_matrix_input


def get_highly_correlated_pairs(
    item_correlations: np.ndarray, critval: float, anto: bool
) -> np.ndarray:
    """
    Identify item pairs that meet the correlation threshold.

    Parameters:
    - item_correlations: Correlation matrix between items
    - critval: Critical value for correlation threshold
    - anto: If True, find negatively correlated pairs; if False, find positively correlated pairs

    Returns:
    - Array of item pair indices (i, j) that meet the threshold
    """
    if anto:
        return np.argwhere(np.tril(item_correlations, -1) <= critval)
    else:
        return np.argwhere(np.tril(item_correlations, -1) >= critval)


def compute_person_correlations(response_i: np.ndarray, response_j: np.ndarray) -> np.ndarray:
    """
    Compute within-person correlations between item pairs.

    Parameters:
    - response_i: Responses to first item in each pair
    - response_j: Responses to second item in each pair

    Returns:
    - Array of within-person correlations for each item pair
    """
    if response_i.shape[0] == 0 or response_j.shape[0] == 0:
        return np.array([])

    mean_i = response_i.mean(axis=1, keepdims=True)
    mean_j = response_j.mean(axis=1, keepdims=True)
    std_i = response_i.std(axis=1, keepdims=True)
    std_j = response_j.std(axis=1, keepdims=True)

    std_i[std_i == 0] = 1
    std_j[std_j == 0] = 1

    numerator = (response_i - mean_i) * (response_j - mean_j)
    denominator = std_i * std_j

    result: np.ndarray = numerator / denominator
    return result


@overload
def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: Literal[False] = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: Literal[False] = False,
) -> np.ndarray: ...


@overload
def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: Literal[True] = True,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: Literal[True] = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate psychometric synonym (or antonym) scores based on the provided item response matrix.

    Psychometric synonyms are item pairs that are highly correlated across the sample.
    This function identifies such pairs and computes within-person correlations between them.
    High scores indicate consistent responding to psychometrically similar items.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
          Can be a 2D list or numpy array.
    - critval: Minimum magnitude of correlation for items to be considered synonyms/antonyms.
               Default is 0.60 for synonyms, typically -0.60 for antonyms.
    - anto: Boolean indicating whether to compute antonym scores
            (highly negatively correlated items).
    - diag: Boolean to optionally return the number of item pairs available for each observation.
    - resample_na: Boolean to indicate resampling when encountering NA for a respondent.
    - random_seed: Optional seed for random number generation when resample_na=True.

    Returns:
    - A numpy array of psychometric synonym/antonym scores, or
    - A tuple of (scores, diagnostic_values) if diag=True.

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid critval, etc.)
    - TypeError: If input is not a list or numpy array

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 1, 1, 4, 5, 6]]
        >>> scores = psychsyn(data, critval=0.5)
        >>> print(scores)
        [0.87, 0.92, 0.45]

        >>> scores, diag = psychsyn(data, critval=0.5, diag=True)
        >>> print(f"Scores: {scores}, Pairs per person: {diag}")
    """

    x_array = validate_matrix_input(x, min_columns=2)

    if not isinstance(critval, (int, float)):
        raise ValueError("critval must be a number")

    if anto and critval > 0:
        raise ValueError("critval should be negative for antonym analysis")

    if not anto and critval < 0:
        raise ValueError("critval should be positive for synonym analysis")

    rng = np.random.default_rng(random_seed)

    item_correlations = np.corrcoef(x_array, rowvar=False)

    item_correlations[np.isnan(item_correlations)] = 0

    item_pairs = get_highly_correlated_pairs(item_correlations, critval, anto)

    if len(item_pairs) == 0:
        empty_scores = np.full(x_array.shape[0], np.nan)
        empty_diag = np.zeros(x_array.shape[0], dtype=int)
        if _return_item_info:
            return empty_scores, empty_diag, item_pairs
        elif diag:
            return empty_scores, empty_diag
        else:
            return empty_scores

    response_i = x_array[:, item_pairs[:, 0]]
    response_j = x_array[:, item_pairs[:, 1]]

    person_corrs = compute_person_correlations(response_i, response_j)

    invalid_pairs = np.isnan(response_i) | np.isnan(response_j)
    person_corrs[invalid_pairs] = np.nan

    if resample_na:
        person_corrs = _resample_missing_correlations(person_corrs, rng)

    scores = np.nanmean(person_corrs, axis=1)

    if np.any(np.isnan(scores)) and len(item_pairs) > 0:
        scores = np.nan_to_num(scores, nan=0.0)

    diag_values: np.ndarray = np.sum(~np.isnan(person_corrs), axis=1)

    if _return_item_info:
        return (scores, diag_values, item_pairs)
    if diag:
        return (scores, diag_values)
    result: np.ndarray = scores
    return result


def _resample_missing_correlations(
    person_corrs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Resample missing correlations based on available data.

    Parameters:
    - person_corrs: Array of person correlations with potential NaN values

    Returns:
    - Array with resampled values replacing NaN
    """
    missing_mask = np.isnan(person_corrs)
    total_missing = missing_mask.sum()

    if total_missing == 0:
        return person_corrs

    result = person_corrs.copy()
    n_pairs = result.shape[1]

    all_nan_rows = missing_mask.all(axis=1)

    with np.errstate(invalid="ignore"):
        row_means = np.abs(np.nanmean(result, axis=1))

    overall_mean = np.abs(np.nanmean(result))
    row_means[all_nan_rows] = overall_mean if not np.isnan(overall_mean) else 0.0

    random_signs = rng.choice([-1, 1], size=total_missing)

    if all_nan_rows.any():
        all_nan_count = all_nan_rows.sum() * n_pairs
        result[all_nan_rows] = (
            random_signs[:all_nan_count].reshape(-1, n_pairs) * row_means[all_nan_rows, np.newaxis]
        )
        remaining_signs = random_signs[all_nan_count:]
    else:
        remaining_signs = random_signs

    has_some_valid = ~all_nan_rows & missing_mask.any(axis=1)
    if has_some_valid.any():
        partial_missing_mask = missing_mask & has_some_valid[:, np.newaxis]
        row_indices = np.broadcast_to(np.arange(result.shape[0])[:, np.newaxis], result.shape)[
            partial_missing_mask
        ]
        result[partial_missing_mask] = (
            remaining_signs[: partial_missing_mask.sum()] * row_means[row_indices]
        )

    return result


def psychsyn_critval(
    x: MatrixLike, anto: bool = False, min_correlation: float = 0.0
) -> list[tuple[int, int, float]]:
    """
    Calculate and order pairwise correlations for all items in the provided item response matrix.

    This function helps identify appropriate critical values for psychsyn analysis by showing
    the distribution of item correlations.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - anto: Boolean indicating whether to order correlations by largest negative values.
    - min_correlation: Minimum correlation magnitude to include in results.

    Returns:
    - A list of tuples containing (item_i, item_j, correlation), ordered by magnitude.

    Example:
        >>> data = [[1, 2, 3, 4], [2, 3, 4, 5], [1, 1, 1, 4]]
        >>> pairs = psychsyn_critval(data, min_correlation=0.3)
        >>> print(pairs[:3])
        [(0, 1, 0.87), (1, 2, 0.82), (0, 2, 0.65)]
    """

    x_array = validate_matrix_input(x, min_columns=2)

    item_correlations = np.corrcoef(x_array, rowvar=False)
    n_items = item_correlations.shape[0]

    i_indices, j_indices = np.triu_indices(n_items, k=1)
    corr_values = item_correlations[i_indices, j_indices]

    valid_mask = ~np.isnan(corr_values) & (np.abs(corr_values) >= min_correlation)
    i_filtered = i_indices[valid_mask]
    j_filtered = j_indices[valid_mask]
    corr_filtered = corr_values[valid_mask]

    sort_indices = np.argsort(corr_filtered) if anto else np.argsort(-corr_filtered)

    correlation_list: list[tuple[int, int, float]] = [
        (int(i_filtered[idx]), int(j_filtered[idx]), float(corr_filtered[idx]))
        for idx in sort_indices
    ]

    return correlation_list


@overload
def psychant(
    x: MatrixLike,
    critval: float = -0.60,
    diag: Literal[False] = False,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> np.ndarray: ...


@overload
def psychant(
    x: MatrixLike,
    critval: float = -0.60,
    diag: Literal[True] = True,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...


def psychant(
    x: MatrixLike,
    critval: float = -0.60,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate the psychometric antonym score.

    Psychometric antonyms are item pairs that are highly negatively correlated across the sample.
    This function is a convenience wrapper around psychsyn with antonym settings.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Minimum magnitude of negative correlation for items to be considered antonyms.
               Default is -0.60.
    - diag: Boolean to optionally return the number of item pairs available for each observation.
    - resample_na: Boolean to indicate resampling when encountering NA for a respondent.
    - random_seed: Optional seed for random number generation when resample_na=True.

    Returns:
    - A numpy array of psychometric antonym scores, or
    - A tuple of (scores, diagnostic_values) if diag=True.

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 1, 1, 4, 5, 6]]
        >>> scores = psychant(data, critval=-0.5)
        >>> print(scores)
        [0.23, 0.18, 0.45]
    """
    if diag:
        return psychsyn(
            x,
            critval=critval,
            anto=True,
            diag=True,
            resample_na=resample_na,
            random_seed=random_seed,
            _return_item_info=False,
        )

    return psychsyn(
        x,
        critval=critval,
        anto=True,
        diag=False,
        resample_na=resample_na,
        random_seed=random_seed,
        _return_item_info=False,
    )


def psychsyn_summary(x: MatrixLike, critval: float = 0.60, anto: bool = False) -> dict[str, Any]:
    """
    Calculate summary statistics for psychometric synonym/antonym analysis.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Critical value for correlation threshold.
    - anto: If True, analyze antonyms; if False, analyze synonyms.

    Returns:
    - Dictionary with summary statistics and item pair information.

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 1, 1, 4, 5, 6]]
        >>> summary = psychsyn_summary(data, critval=0.5)
        >>> print(summary)
        {'mean_score': 0.75, 'std_score': 0.24, 'item_pairs': 3, ...}
    """

    scores, _, item_pairs = psychsyn(
        x, critval=critval, anto=anto, diag=True, _return_item_info=True
    )

    valid_count = int(np.sum(~np.isnan(scores)))
    summary = calculate_summary_stats(scores, suffix="_score")
    summary.update(
        {
            "item_pairs": len(item_pairs),
            "total_individuals": len(scores),
            "valid_individuals": valid_count,
            "missing_individuals": len(scores) - valid_count,
        }
    )
    return summary
