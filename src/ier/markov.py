"""
Markov chain index for detecting patterned insufficient effort responding.

Builds a first-order transition matrix from each respondent's response sequence and
computes the Shannon entropy of transitions. Low entropy indicates highly predictable
(patterned) responding, which may reflect careless strategies such as alternating
or cycling through response options.

References:
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

from typing import Any

import numpy as np

from ier._flagging import threshold_flags
from ier._summary import calculate_summary_stats
from ier._validation import MatrixLike, iter_rows, validate_matrix_input


def markov(
    x: MatrixLike,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Compute Markov chain transition entropy for each respondent.

    Builds a first-order transition matrix from each respondent's response sequence
    and computes the Shannon entropy of the transition probabilities, weighted by
    row marginals. Low entropy indicates predictable, patterned responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - na_rm: If True, removes NaN values before analysis.

    Returns:
    - A numpy array of transition entropy values per respondent.
      Lower values indicate more predictable (potentially careless) patterns.

    Raises:
    - ValueError: If data has fewer than 3 columns.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1]]
        >>> markov(data)
        array([0.  , 1.56])
    """
    x_array = validate_matrix_input(x, min_columns=3, check_type=False)

    if not na_rm and np.isnan(x_array).any():
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    all_valid = x_array[~np.isnan(x_array)]
    if len(all_valid) == 0:
        return np.full(x_array.shape[0], np.nan)

    categories = np.sort(np.unique(all_valid))
    cat_map = {v: i for i, v in enumerate(categories)}
    k = len(categories)

    n_rows = x_array.shape[0]
    result = np.zeros(n_rows, dtype=float)

    for i, row in enumerate(iter_rows(x_array, na_rm)):
        if len(row) < 2:
            result[i] = np.nan
            continue

        trans = _build_transition_matrix(row, cat_map, k)
        result[i] = _transition_entropy(trans)

    return result


def markov_flag(
    x: MatrixLike,
    threshold: float | None = None,
    percentile: float = 5.0,
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Markov chain entropy and flag respondents with low entropy.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - threshold: Absolute entropy threshold below which to flag. If None, uses percentile.
    - percentile: Percentile below which to flag (default 5th percentile).
    - na_rm: If True, removes NaN values before analysis.

    Returns:
    - Tuple of (entropy_scores, flags) where flags is True for flagged respondents.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1]]
        >>> scores, flags = markov_flag(data)
    """
    scores = markov(x, na_rm=na_rm)

    flags = threshold_flags(
        scores, threshold=threshold, percentile=percentile, direction="low", inclusive=True
    )

    return scores, flags


def markov_summary(
    x: MatrixLike,
    na_rm: bool = True,
) -> dict[str, Any]:
    """
    Calculate summary statistics for Markov chain entropy scores.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - na_rm: If True, removes NaN values before analysis.

    Returns:
    - Dictionary with summary statistics.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1]]
        >>> markov_summary(data)
    """
    scores = markov(x, na_rm=na_rm)

    summary = calculate_summary_stats(scores)
    summary.update(
        {
            "n_total": len(scores),
            "n_valid": int(np.sum(~np.isnan(scores))),
            "n_missing": int(np.sum(np.isnan(scores))),
        }
    )
    return summary


def _build_transition_matrix(row: np.ndarray, cat_map: dict[float, int], k: int) -> np.ndarray:
    """Build a first-order transition count matrix from a response sequence."""
    trans = np.zeros((k, k), dtype=float)
    for t in range(len(row) - 1):
        from_idx = cat_map[row[t]]
        to_idx = cat_map[row[t + 1]]
        trans[from_idx, to_idx] += 1
    return trans


def _transition_entropy(trans: np.ndarray) -> float:
    """Compute Shannon entropy of transition matrix, weighted by row marginals."""
    row_sums = trans.sum(axis=1)
    total = row_sums.sum()

    if total == 0:
        return 0.0

    entropy = 0.0
    for i in range(trans.shape[0]):
        if row_sums[i] == 0:
            continue
        probs = trans[i, :] / row_sums[i]
        weight = row_sums[i] / total
        for p in probs:
            if p > 0:
                entropy -= weight * p * np.log2(p)

    return float(entropy)
