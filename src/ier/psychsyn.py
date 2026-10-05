"""
Takes a matrix of item responses and identifies item pairs that are highly correlated within the
overall dataset. What defines "highly correlated" is set by the critical value (e.g., r > .60). Each
respondents' psychometric synonym score is then computed as the within-person correlation be-
tween the identified item-pairs. Alternatively computes the psychometric antonym score which is a
variant that uses item pairs that are highly negatively correlated.

This module provides functions for detecting careless responding patterns by analyzing how
individuals respond to psychometrically similar (synonym) or opposite (antonym) items.
"""

from numbers import Real
from typing import Any, Literal, overload

import numpy as np

from ier._column_statistics import column_correlations, pairwise_column_correlations
from ier._correlation import selected_row_correlations
from ier._flagging import threshold_flags
from ier._summary import calculate_summary_stats
from ier._validation import MatrixLike, validate_matrix_input
from ier.types import ItemCorrelationMode

_PSYCHSYN_BATCH_ELEMENTS = 262_144


def _validate_correlation_cutoff(value: object, *, name: str) -> float:
    """Validate finite real cutoffs while permitting values beyond the correlation range."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        cutoff = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(cutoff):
        raise ValueError(f"{name} must be a finite number")
    return cutoff


def _item_correlations(x: np.ndarray, item_correlations: object) -> tuple[np.ndarray, bool]:
    """Correlate items in the requested missing-data mode, reporting pairwise scoring."""
    if not isinstance(item_correlations, str) or item_correlations not in ("complete", "pairwise"):
        raise ValueError("item_correlations must be 'complete' or 'pairwise'")
    if item_correlations == "pairwise":
        return pairwise_column_correlations(x), True
    return column_correlations(x), False


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
    row_indices, column_indices = np.tril_indices(item_correlations.shape[0], k=-1)
    pair_correlations = item_correlations[row_indices, column_indices]
    selected = pair_correlations <= critval if anto else pair_correlations >= critval
    selected &= np.isfinite(pair_correlations)
    return np.stack((row_indices[selected], column_indices[selected]), axis=1)


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
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> np.ndarray:
    pass


@overload
def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: Literal[True] = True,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: Literal[False] = False,
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> tuple[np.ndarray, np.ndarray]:
    pass


@overload
def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: Literal[False] = False,
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    pass


@overload
def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: Literal[True] = True,
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass


def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: bool = False,
    *,
    item_correlations: ItemCorrelationMode = "complete",
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
               Must be a finite real number. Default is 0.60 for synonyms,
               typically -0.60 for antonyms.
    - anto: Boolean indicating whether to compute antonym scores
            (highly negatively correlated items).
    - diag: Boolean to optionally return the number of item pairs available for each observation.
            A single available pair still cannot provide a respondent correlation.
    - resample_na: Accepted for compatibility with ``careless::psychsyn``; it has no
                   effect on scores. Complete discovery selects only fully observed
                   items, pairwise mode leaves respondents with fewer than two
                   answered pairs unavailable, and zero within-pair variance scores
                   0.0 (careless returns ``NA``), so no score is left to resample.
    - random_seed: Accepted for compatibility; it has no effect on scores.
    - item_correlations: Missing-data policy for item pair discovery. ``"complete"``
                         (the default) treats an item with any missing response as
                         undefined, as in ``np.corrcoef``. ``"pairwise"`` correlates
                         each item pair over the respondents who answered both, as
                         ``careless::psychsyn`` does, and scores each respondent over
                         the selected pairs they answered. Pairs need at least three
                         shared respondents, so with fewer than three respondents
                         pairwise mode finds no pairs. Otherwise, without missing
                         responses both modes discover the same pairs.

    Returns:
    - A numpy array of psychometric synonym/antonym scores. Fewer than two selected
      (or, in pairwise mode, answered) pairs produce unavailable (``NaN``) scores.
      Undefined item correlations do not qualify as pairs, including when critval=0.
    - A tuple of (scores, diagnostic_values) if diag=True. In pairwise mode the
      diagnostics count each respondent's answered pairs.

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid critval, etc.)
    - TypeError: If input is not array-like (for example, a scalar or string)

    Example:
        >>> data = [
        ...     [1, 1, 2, 3, 5, 5],
        ...     [1, 1, 4, 4, 1, 1],
        ...     [5, 5, 1, 1, 2, 1],
        ...     [2, 2, 2, 1, 5, 3],
        ...     [1, 1, 1, 2, 5, 5],
        ...     [2, 4, 4, 3, 5, 3],
        ... ]
        >>> scores, pairs = psychsyn(data, diag=True)
        >>> np.round(scores, 2).tolist()
        [0.96, 1.0, 0.97, 0.87, 0.97, -0.94]
        >>> pairs.tolist()
        [3, 3, 3, 3, 3, 3]

        One missing response removes its whole item from complete discovery,
        while pairwise discovery keeps the pair and skips only that response:

        >>> data[4][3] = float("nan")
        >>> np.round(psychsyn(data), 2).tolist()
        [1.0, 0.0, 1.0, 1.0, 1.0, -1.0]
        >>> np.round(psychsyn(data, item_correlations="pairwise"), 2).tolist()
        [0.96, 1.0, 0.97, 0.87, 1.0, -0.94]
    """

    x_array = validate_matrix_input(x, min_columns=2)

    critval = _validate_correlation_cutoff(critval, name="critval")

    if anto and critval > 0:
        raise ValueError("critval should be negative for antonym analysis")

    if not anto and critval < 0:
        raise ValueError("critval should be positive for synonym analysis")

    item_corr, pairwise = _item_correlations(x_array, item_correlations)

    item_pairs = get_highly_correlated_pairs(item_corr, critval, anto)

    if len(item_pairs) == 0:
        empty_scores = np.full(x_array.shape[0], np.nan)
        empty_diag = np.zeros(x_array.shape[0], dtype=int)
        if _return_item_info:
            return empty_scores, empty_diag, item_pairs
        elif diag:
            return empty_scores, empty_diag
        else:
            return empty_scores

    scores, diag_values = _compute_person_scores(x_array, item_pairs, pairwise=pairwise)

    if _return_item_info:
        return (scores, diag_values, item_pairs)
    if diag:
        return (scores, diag_values)
    result: np.ndarray = scores
    return result


def _compute_person_scores(
    x: np.ndarray, item_pairs: np.ndarray, *, pairwise: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Score selected pairs in bounded batches, including missing-response inputs.

    Complete scoring requires every selected response. Pairwise scoring
    correlates each respondent's answered pairs instead, and respondents with
    fewer than two answered pairs stay unavailable.
    """
    n_rows = len(x)
    n_pairs = len(item_pairs)
    scores = np.full(n_rows, np.nan)
    diag_values = np.zeros(n_rows, dtype=int)
    if n_pairs == 0:
        return scores, diag_values

    # Bound both the selected pairs and any row selection from the original matrix.
    batch_rows = max(1, _PSYCHSYN_BATCH_ELEMENTS // max(x.shape[1], 2 * n_pairs, 16))
    selected_items = np.unique(item_pairs)
    left_indices, right_indices = item_pairs[:, 0], item_pairs[:, 1]
    # Pair positions among the selected items count answered pairs from one mask.
    left_selected, right_selected = np.searchsorted(selected_items, item_pairs.T)

    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        block = x[start:stop]
        finite = np.isfinite(block[:, selected_items])
        finite_rows = finite.all(axis=1)
        if pairwise and not np.all(finite_rows):
            pair_counts = np.count_nonzero(
                finite[:, left_selected] & finite[:, right_selected], axis=1
            )
            diag_values[start:stop] = pair_counts
            if n_pairs < 2:
                continue
            batch_scores = selected_row_correlations(
                block,
                left_indices,
                right_indices,
                has_missing=True,
                zero_variance=0.0,
            )
            # Fewer than two answered pairs cannot supply a respondent correlation.
            batch_scores[pair_counts < 2] = np.nan
            scores[start:stop] = batch_scores
            continue
        diag_values[start:stop] = finite_rows * n_pairs
        if n_pairs < 2 or not np.any(finite_rows):
            continue
        complete = block if np.all(finite_rows) else block[finite_rows]
        batch_scores = scores[start:stop]
        batch_scores[finite_rows] = selected_row_correlations(
            complete,
            left_indices,
            right_indices,
            has_missing=False,
            zero_variance=0.0,
        )
    return scores, diag_values


def psychsyn_critval(
    x: MatrixLike,
    anto: bool = False,
    min_correlation: float = 0.0,
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> list[tuple[int, int, float]]:
    """
    Calculate and order pairwise correlations for all items in the provided item response matrix.

    This function helps identify appropriate critical values for psychsyn analysis by showing
    the distribution of item correlations.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - anto: Boolean indicating whether to order correlations by largest negative values.
    - min_correlation: Finite, nonnegative correlation magnitude to include in results.
    - item_correlations: ``"complete"`` (default) or ``"pairwise"`` missing-data
                         policy for item correlations, as in ``psychsyn``.

    Returns:
    - A list of tuples containing (item_i, item_j, correlation), ordered by magnitude.

    Example:
        >>> data = [
        ...     [1, 1, 2, 3, 5, 5],
        ...     [1, 1, 4, 4, 1, 1],
        ...     [5, 5, 1, 1, 2, 1],
        ...     [2, 2, 2, 1, 5, 3],
        ...     [1, 1, 1, 2, 5, 5],
        ...     [2, 4, 4, 3, 5, 3],
        ... ]
        >>> pairs = psychsyn_critval(data, min_correlation=0.6)
        >>> items = [(i, j) for i, j, _ in pairs]
        >>> items
        [(0, 1), (4, 5), (2, 3), (0, 3)]
        >>> np.round([correlation for _, _, correlation in pairs], 2).tolist()
        [0.88, 0.85, 0.77, -0.64]
    """

    x_array = validate_matrix_input(x, min_columns=2)

    min_correlation = _validate_correlation_cutoff(min_correlation, name="min_correlation")
    if min_correlation < 0:
        raise ValueError("min_correlation must be nonnegative")

    item_corr, _ = _item_correlations(x_array, item_correlations)
    n_items = item_corr.shape[0]

    i_indices, j_indices = np.triu_indices(n_items, k=1)
    corr_values = item_corr[i_indices, j_indices]

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
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> np.ndarray:
    pass


@overload
def psychant(
    x: MatrixLike,
    critval: float = -0.60,
    diag: Literal[True] = True,
    resample_na: bool = False,
    random_seed: int | None = None,
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> tuple[np.ndarray, np.ndarray]:
    pass


@overload
def psychant(
    x: MatrixLike,
    critval: float = -0.60,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    pass


def psychant(
    x: MatrixLike,
    critval: float = -0.60,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    *,
    item_correlations: ItemCorrelationMode = "complete",
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
    - resample_na: Accepted for compatibility; it has no effect on scores (see ``psychsyn``).
    - random_seed: Accepted for compatibility; it has no effect on scores.
    - item_correlations: ``"complete"`` (default) or ``"pairwise"`` missing-data
                         policy for pair discovery and scoring, as in ``psychsyn``.

    Returns:
    - A numpy array of psychometric antonym scores, or
    - A tuple of (scores, diagnostic_values) if diag=True.

    Example:
        >>> data = [
        ...     [1, 5, 2, 3, 5, 1],
        ...     [1, 5, 4, 2, 1, 5],
        ...     [5, 1, 1, 5, 2, 5],
        ...     [2, 4, 2, 5, 5, 3],
        ...     [1, 5, 1, 4, 5, 1],
        ...     [2, 2, 4, 3, 5, 3],
        ... ]
        >>> np.round(psychant(data), 2).tolist()
        [-0.96, -1.0, -0.97, -0.87, -0.97, 0.94]
    """
    return psychsyn(
        x,
        critval=critval,
        anto=True,
        diag=diag,
        resample_na=resample_na,
        random_seed=random_seed,
        item_correlations=item_correlations,
    )


def psychsyn_summary(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    *,
    item_correlations: ItemCorrelationMode = "complete",
) -> dict[str, Any]:
    """
    Calculate summary statistics for psychometric synonym/antonym analysis.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Critical value for correlation threshold.
    - anto: If True, analyze antonyms; if False, analyze synonyms.
    - item_correlations: ``"complete"`` (default) or ``"pairwise"`` missing-data
                         policy for pair discovery and scoring, as in ``psychsyn``.

    Returns:
    - Dictionary with score summary statistics (``mean_score``, ``std_score``,
      ``min_score``, ``max_score``, ``median_score``), the number of selected
      ``item_pairs``, and respondent counts ``n_total``, ``n_valid`` and
      ``n_missing``, as in ``mahad_summary()`` and ``markov_summary()``. The same
      counts remain available as ``total_individuals``, ``valid_individuals``
      and ``missing_individuals`` for compatibility.

    Example:
        >>> data = [
        ...     [1, 1, 2, 3, 5, 5],
        ...     [1, 1, 4, 4, 1, 1],
        ...     [5, 5, 1, 1, 2, 1],
        ...     [2, 2, 2, 1, 5, 3],
        ...     [1, 1, 1, 2, 5, 5],
        ...     [2, 4, 4, 3, 5, 3],
        ... ]
        >>> summary = psychsyn_summary(data)
        >>> summary["item_pairs"], summary["n_total"], summary["n_valid"], summary["n_missing"]
        (3, 6, 6, 0)
        >>> round(summary["median_score"], 2)
        0.97
    """

    scores, _, item_pairs = psychsyn(
        x,
        critval=critval,
        anto=anto,
        diag=True,
        _return_item_info=True,
        item_correlations=item_correlations,
    )

    valid_count = int(np.sum(~np.isnan(scores)))
    missing_count = len(scores) - valid_count
    summary = calculate_summary_stats(scores, suffix="_score")
    summary.update(
        {
            "item_pairs": len(item_pairs),
            "total_individuals": len(scores),
            "valid_individuals": valid_count,
            "missing_individuals": missing_count,
            "n_total": len(scores),
            "n_valid": valid_count,
            "n_missing": missing_count,
        }
    )
    return summary


def psychsyn_flag(
    x: MatrixLike,
    critval: float = 0.60,
    threshold: float | None = None,
    percentile: float = 5.0,
    *,
    resample_na: bool = False,
    random_seed: int | None = None,
    item_correlations: ItemCorrelationMode = "complete",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute psychometric synonym scores and flag respondents with low consistency.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Minimum item correlation for synonym pairs (default 0.60).
    - threshold: Absolute score threshold at or below which to flag. If None, uses percentile.
    - percentile: Percentile below which to flag (default 5th percentile).
    - resample_na: Accepted for compatibility; it has no effect on scores (see ``psychsyn``).
    - random_seed: Accepted for compatibility; it has no effect on scores.
    - item_correlations: ``"complete"`` (default) or ``"pairwise"`` missing-data
                         policy, as in ``psychsyn``.

    Returns:
    - Tuple of (scores, flags) where flags is True for flagged respondents.
      Unavailable (``NaN``) scores are never flagged.

    Example:
        >>> data = [
        ...     [1, 1, 2, 3, 5, 5],
        ...     [1, 1, 4, 4, 1, 1],
        ...     [5, 5, 1, 1, 2, 1],
        ...     [2, 2, 2, 1, 5, 3],
        ...     [1, 1, 1, 2, 5, 5],
        ...     [2, 4, 4, 3, 5, 3],
        ... ]
        >>> scores, flags = psychsyn_flag(data)
        >>> flags.tolist()
        [False, False, False, False, False, True]
    """
    scores = psychsyn(
        x,
        critval=critval,
        resample_na=resample_na,
        random_seed=random_seed,
        item_correlations=item_correlations,
    )

    # Mirrors INDEX_REGISTRY["psychsyn"].flag_direction; importing it here is circular.
    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="low")

    return scores, flags


def psychant_flag(
    x: MatrixLike,
    critval: float = -0.60,
    threshold: float | None = None,
    percentile: float = 95.0,
    *,
    resample_na: bool = False,
    random_seed: int | None = None,
    item_correlations: ItemCorrelationMode = "complete",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute psychometric antonym scores and flag the highest scores.

    Attentive respondents answer antonym pairs in opposite directions, giving
    strongly negative scores. Scores near zero or above are suspicious, so flags
    follow the high direction that ``screen()`` applies to ``psychant``.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Maximum (negative) item correlation for antonym pairs (default -0.60).
    - threshold: Absolute score threshold at or above which to flag. If None, uses percentile.
    - percentile: Percentile above which to flag (default 95th percentile).
    - resample_na: Accepted for compatibility; it has no effect on scores (see ``psychsyn``).
    - random_seed: Accepted for compatibility; it has no effect on scores.
    - item_correlations: ``"complete"`` (default) or ``"pairwise"`` missing-data
                         policy, as in ``psychsyn``.

    Returns:
    - Tuple of (scores, flags) where flags is True for flagged respondents.
      Unavailable (``NaN``) scores are never flagged.

    Example:
        >>> data = [
        ...     [1, 5, 2, 3, 5, 1],
        ...     [1, 5, 4, 2, 1, 5],
        ...     [5, 1, 1, 5, 2, 5],
        ...     [2, 4, 2, 5, 5, 3],
        ...     [1, 5, 1, 4, 5, 1],
        ...     [2, 2, 4, 3, 5, 3],
        ... ]
        >>> scores, flags = psychant_flag(data, threshold=0.0)
        >>> np.round(scores[flags], 2).tolist()
        [0.94]
    """
    scores = psychant(
        x,
        critval=critval,
        resample_na=resample_na,
        random_seed=random_seed,
        item_correlations=item_correlations,
    )

    # Mirrors INDEX_REGISTRY["psychant"].flag_direction; importing it here is circular.
    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="high")

    return scores, flags
