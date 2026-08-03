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

_MAX_DENSE_STATES = 64
_TRANSITION_BATCH_WORKSPACE_BYTES = 64 * 1024 * 1024
_CATEGORY_DISCOVERY_ROWS = 4096


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

    missing = np.isnan(x_array)
    has_missing = bool(missing.any())
    if not na_rm and has_missing:
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    n_rows = x_array.shape[0]
    if not has_missing:
        return _markov_complete(x_array)

    if not np.any(~missing):
        return np.full(x_array.shape[0], np.nan)

    result = np.zeros(n_rows, dtype=float)
    for i, row in enumerate(iter_rows(x_array, na_rm=True)):
        if len(row) < 2:
            result[i] = np.nan
            continue

        result[i] = _transition_entropy_row(row)

    return result


def _markov_complete(x: np.ndarray) -> np.ndarray:
    """Score complete rows with bounded dense batches or a sparse fallback."""
    encoder = _dense_state_encoder(x)
    if encoder is None:
        return _transition_entropies_sparse(x)

    minimum, states = encoder
    n_states = len(states) if minimum is None else int(np.max(states)) + 1
    n_items = x.shape[1]
    integer_bytes = np.dtype(np.intp).itemsize
    float_bytes = np.dtype(float).itemsize
    bytes_per_row = integer_bytes * (2 * n_items + n_states * n_states + n_states) + float_bytes * (
        n_states * n_states + n_states
    )
    batch_rows = max(1, _TRANSITION_BATCH_WORKSPACE_BYTES // bytes_per_row)

    result = np.empty(len(x), dtype=float)
    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        encoded = _encode_state_batch(x[start:stop], minimum, states)
        transition_counts = _dense_transition_counts(encoded, n_states)
        result[start:stop] = _transition_entropy_batch(transition_counts)
    return result


def _dense_state_encoder(x: np.ndarray) -> tuple[float | None, np.ndarray] | None:
    """Return a bounded dense encoder, or None for high-cardinality data."""
    minimum = float(np.min(x))
    maximum = float(np.max(x))
    if np.isfinite(minimum) and np.isfinite(maximum):
        span = maximum - minimum
        if span < _MAX_DENSE_STATES:
            n_states = int(span) + 1
            mapping = _direct_state_mapping(x, n_states)
            if mapping is not None:
                return minimum, mapping

    categories = np.array([], dtype=x.dtype)
    for start in range(0, len(x), _CATEGORY_DISCOVERY_ROWS):
        found = np.unique(x[start : start + _CATEGORY_DISCOVERY_ROWS])
        if len(found) > _MAX_DENSE_STATES:
            return None
        categories = np.union1d(categories, found)
        if len(categories) > _MAX_DENSE_STATES:
            return None
    return None, categories


def _direct_state_mapping(x: np.ndarray, n_states: int) -> np.ndarray | None:
    """Build a compact mapping for a bounded integral response range."""
    bytes_per_row = x.shape[1] * (np.dtype(float).itemsize + np.dtype(np.intp).itemsize)
    batch_rows = max(1, _TRANSITION_BATCH_WORKSPACE_BYTES // bytes_per_row)
    present = np.zeros(n_states, dtype=bool)
    minimum = float(np.min(x))

    for start in range(0, len(x), batch_rows):
        batch = x[start : start + batch_rows]
        if not np.all(batch == np.floor(batch)):
            return None
        encoded = np.empty(batch.shape, dtype=np.intp)
        np.subtract(batch, minimum, out=encoded, casting="unsafe")
        present |= np.bincount(encoded.ravel(), minlength=n_states) > 0

    return np.cumsum(present, dtype=np.intp) - 1


def _encode_state_batch(
    batch: np.ndarray,
    minimum: float | None,
    states: np.ndarray,
) -> np.ndarray:
    """Encode one response batch using a direct mapping or sorted categories."""
    if minimum is None:
        encoded: np.ndarray = np.searchsorted(states, batch)
        return encoded

    encoded = np.empty(batch.shape, dtype=np.intp)
    np.subtract(batch, minimum, out=encoded, casting="unsafe")
    np.take(states, encoded, out=encoded)
    return encoded


def _dense_transition_counts(encoded: np.ndarray, n_states: int) -> np.ndarray:
    """Count transition pairs for one encoded batch without repeated row IDs."""
    n_rows, n_items = encoded.shape
    pair_ids = np.empty((n_rows, n_items - 1), dtype=np.intp)
    np.multiply(encoded[:, :-1], n_states, out=pair_ids)
    np.add(pair_ids, encoded[:, 1:], out=pair_ids)
    row_offsets = np.arange(n_rows, dtype=np.intp) * (n_states * n_states)
    pair_ids += row_offsets[:, None]

    counts = np.bincount(pair_ids.ravel(), minlength=n_rows * n_states * n_states)
    return counts.reshape(n_rows, n_states, n_states)


def _transition_entropies_sparse(x: np.ndarray) -> np.ndarray:
    """Score high-cardinality complete rows without dense state-square arrays."""
    result = np.empty(len(x), dtype=float)
    for row_index, row in enumerate(x):
        result[row_index] = _transition_entropy_row(row)
    return result


def _transition_entropy_row(row: np.ndarray) -> float:
    """Compute transition entropy from the observed counts in one row."""
    _, encoded = np.unique(row, return_inverse=True)
    n_states = int(np.max(encoded)) + 1
    from_counts = np.bincount(encoded[:-1])
    pair_ids = encoded[:-1] * n_states + encoded[1:]
    _, pair_counts = np.unique(pair_ids, return_counts=True)
    return _conditional_entropy_from_counts(from_counts, pair_counts, len(row) - 1)


def _conditional_entropy_from_counts(
    from_counts: np.ndarray,
    pair_counts: np.ndarray,
    total: int | float,
) -> float:
    """Compute conditional entropy using only positive transition counts."""
    if total == 0:
        return 0.0

    positive_from = from_counts[from_counts > 0]
    positive_pairs = pair_counts[pair_counts > 0]
    from_terms = positive_from @ np.log2(positive_from)
    pair_terms = positive_pairs @ np.log2(positive_pairs)
    return float((from_terms - pair_terms) / total)


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
    - threshold: Absolute entropy threshold at or below which to flag. If None, uses percentile.
    - percentile: Percentile below which to flag (default 5th percentile).
    - na_rm: If True, removes NaN values before analysis.

    Returns:
    - Tuple of (entropy_scores, flags) where flags is True for flagged respondents.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1]]
        >>> scores, flags = markov_flag(data)
    """
    scores = markov(x, na_rm=na_rm)

    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="low")

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


def _transition_entropy(trans: np.ndarray) -> float:
    """Compute Shannon entropy of one transition matrix, weighted by row marginals."""
    row_sums = trans.sum(axis=1)
    total = float(row_sums.sum())
    return _conditional_entropy_from_counts(row_sums, trans.ravel(), total)


def _transition_entropy_batch(transitions: np.ndarray) -> np.ndarray:
    """Vectorized Shannon entropy for a batch of transition matrices."""
    row_sums = transitions.sum(axis=2)
    totals = row_sums.sum(axis=1)

    transition_terms = np.zeros(transitions.shape, dtype=float)
    np.log2(transitions, where=transitions > 0, out=transition_terms)
    transition_terms *= transitions

    row_terms = np.zeros(row_sums.shape, dtype=float)
    np.log2(row_sums, where=row_sums > 0, out=row_terms)
    row_terms *= row_sums

    numerators = np.sum(row_terms, axis=1) - np.sum(transition_terms, axis=(1, 2))
    result: np.ndarray = np.divide(
        numerators,
        totals,
        out=np.zeros(len(transitions), dtype=float),
        where=totals > 0,
    )
    return result
