"""
Person-total correlation for detecting careless responding.

The person-total correlation measures how similar an individual's response pattern
is to the overall sample mean response pattern. Low correlations may indicate
careless or random responding.
"""

import numpy as np

from ier._column_statistics import _exact_mean_profile, column_mean_profile
from ier._correlation import row_correlations
from ier._validation import MatrixLike, validate_matrix_input

_PERSON_TOTAL_BATCH_ELEMENTS = 262_144
_MAX_PROFILE_CACHE = 32


def person_total(
    x: MatrixLike,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Calculate person-total correlation for each individual.

    The person-total correlation (also called "personal biserial") measures
    the correlation between each individual's responses and the mean response
    across all individuals for each item. Low values suggest responses that
    deviate substantially from typical patterns.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - na_rm: If True, use pairwise complete observations for correlations.

    Returns:
    - A numpy array of person-total correlations for each individual.

    Raises:
    - ValueError: If inputs are invalid

    Example:
        >>> data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        >>> scores = person_total(data)
        >>> print(scores)
        [1.0, -1.0, 1.0]
    """
    x_array = validate_matrix_input(x, min_columns=2)
    n_rows, n_items = x_array.shape
    batch_rows = max(1, _PERSON_TOTAL_BATCH_ELEMENTS // n_items)

    if not na_rm and x_array.dtype.kind not in "iub":
        for start in range(0, n_rows, batch_rows):
            if np.isnan(x_array[start : start + batch_rows]).any():
                return np.full(n_rows, np.nan)

    item_means = column_mean_profile(x_array, ignore_nan=na_rm)
    correlations = np.empty(n_rows)
    profile_cache: dict[bytes, np.ndarray] = {}
    repair_pairwise = False
    if na_rm and x_array.dtype.kind == "f" and x_array.dtype.itemsize <= 8:
        ordered = np.sort(item_means[np.isfinite(item_means)])
        with np.errstate(over="ignore"):
            gaps = np.diff(ordered)
            magnitude = np.maximum(np.abs(ordered[:-1]), np.abs(ordered[1:]))
            repair_pairwise = bool(np.any(gaps <= magnitude * np.sqrt(np.finfo(float).eps)))
    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        block = x_array[start:stop]
        correlations[start:stop] = row_correlations(
            block,
            np.broadcast_to(item_means, block.shape),
            zero_variance=np.nan,
        )
        if repair_pairwise:
            _repair_pairwise_profiles(
                x_array, block, item_means, correlations[start:stop], profile_cache
            )

    return correlations


def _repair_pairwise_profiles(
    x: np.ndarray,
    block: np.ndarray,
    profile: np.ndarray,
    scores: np.ndarray,
    cache: dict[bytes, np.ndarray],
) -> None:
    """Resolve close selected item means before a respondent's missing mask rounds them.

    A wide global profile can contain a narrowly separated item group. A respondent
    observing only that group needs its means centered before float conversion.
    Exact means retain each item's denominator from the complete original sample.
    """
    if not np.isnan(block).any():
        return
    paired = ~(np.isnan(block) | np.isnan(profile))
    profiles = np.broadcast_to(profile, block.shape)
    lower = np.min(profiles, axis=1, where=paired, initial=np.inf)
    upper = np.max(profiles, axis=1, where=paired, initial=-np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        span = upper - lower
        magnitude = np.maximum(np.abs(lower), np.abs(upper))
        candidates = span <= magnitude * np.sqrt(np.finfo(float).eps)
    candidates &= np.isfinite(lower) & np.isfinite(upper)
    candidates &= np.count_nonzero(paired, axis=1) >= 2
    if not np.any(candidates):
        return
    # Constant or nonfinite observed responses cannot supply a Pearson score.
    lower = np.min(block, axis=1, where=paired, initial=np.inf)
    upper = np.max(block, axis=1, where=paired, initial=-np.inf)
    candidates &= np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
    groups: dict[bytes, list[int]] = {}
    for row in np.flatnonzero(candidates):
        groups.setdefault(paired[row].tobytes(), []).append(int(row))
    for signature, positions in groups.items():
        exact = cache.get(signature)
        if exact is None:
            columns = np.flatnonzero(paired[positions[0]])
            exact = np.full(len(profile), np.nan)
            _exact_mean_profile(x, exact, columns, ignore_nan=True)
            if len(exact) <= _PERSON_TOTAL_BATCH_ELEMENTS:
                while cache and (
                    len(cache) >= _MAX_PROFILE_CACHE
                    or sum(value.size for value in cache.values()) + len(exact)
                    > _PERSON_TOTAL_BATCH_ELEMENTS
                ):
                    del cache[next(iter(cache))]
                cache[signature] = exact
        selected = block[positions]
        scores[positions] = row_correlations(
            selected, np.broadcast_to(exact, selected.shape), zero_variance=np.nan
        )
