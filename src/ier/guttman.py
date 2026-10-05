"""
Guttman errors for person-fit analysis in detecting careless responding.

Guttman errors count the number of response reversals relative to item
difficulty ordering. High error counts suggest inconsistent or careless responding.
"""

import warnings

import numpy as np

from ier._column_statistics import column_mean_order
from ier._flagging import validate_threshold
from ier._row_statistics import row_slices
from ier._validation import MatrixLike, validate_matrix_input

_MAX_CATEGORIES = 64
_GUTTMAN_BATCH_CELLS = 1_000_000


def guttman(
    x: MatrixLike,
    na_rm: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Calculate Guttman errors for each individual.

    Guttman errors measure the number of times a person's responses violate
    the expected ordering based on item difficulty (mean endorsement).
    Items are ordered easiest first (highest sample mean first). An error is
    an item pair where the respondent scores strictly higher on the harder,
    less endorsed item than on the easier one, so a perfect cumulative pattern
    scores 0. Items with tied difficulties are ordered by column position.
    Item means always use the available (non-missing) responses, whatever
    ``na_rm`` is; items without any observed response are ordered last. Pairs
    involving a missing response never count as errors.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - na_rm: Selects the normalization denominator. If True, normalized scores
             divide by the pairs among the respondent's answered items; if False,
             they divide by every item pair. Raw counts do not depend on it.
    - normalize: If True, return proportion of errors (0-1 scale).
                 If False, return raw error counts.

    Returns:
    - A numpy array of Guttman error scores for each individual.
      Higher values indicate more inconsistent responding.

    Raises:
    - ValueError: If inputs are invalid

    Example:
        >>> import numpy as np
        >>> # Items get harder from left to right; the last row reverses [1, 1, 0, 0].
        >>> data = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
        >>> guttman(data, normalize=False).tolist()
        [0.0, 0.0, 0.0, 0.0, 4.0]
        >>> np.round(guttman(data), 2).tolist()
        [0.0, 0.0, 0.0, 0.0, 0.67]
    """
    x_array = validate_matrix_input(x, min_columns=2)
    n_persons = x_array.shape[0]
    n_items = x_array.shape[1]

    # One missing response must not make its item the hardest, so na_rm sets
    # only the denominator below.
    difficulty_order = column_mean_order(x_array, ignore_nan=True)
    categories = _small_categorical_values(x_array)
    errors, valid_counts = _count_guttman_errors(
        x_array,
        difficulty_order,
        categories,
        count_valid=na_rm,
    )

    comparisons: np.ndarray
    if valid_counts is not None:
        comparisons = valid_counts * (valid_counts - 1.0) / 2.0
    else:
        comparisons = np.full(n_persons, n_items * (n_items - 1) / 2.0, dtype=float)

    result: np.ndarray
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = errors / comparisons
        result = np.where(comparisons == 0, np.nan, result)
    else:
        result = errors

    return result


def _small_categorical_values(x: np.ndarray) -> np.ndarray | None:
    """Return up to 64 ordered categories using only bounded scan workspaces."""
    if x.dtype.kind in "iu":
        lower, upper = int(np.min(x)), int(np.max(x))
        if upper - lower < _MAX_CATEGORIES:
            return np.arange(upper - lower + 1, dtype=x.dtype) + lower
        category_dtype = x.dtype
    else:
        category_dtype = np.dtype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            minimum = float(np.nanmin(x))
            maximum = float(np.nanmax(x))
        if np.isnan(minimum) or np.isnan(maximum):
            return np.array([], dtype=float)

        span = maximum - minimum
        # Every consecutive integer must remain distinct in the float64 grid.
        if span < _MAX_CATEGORIES and max(abs(minimum), abs(maximum)) <= 2**53:
            for start, stop in row_slices(len(x), x.shape[1]):
                block = x[start:stop]
                values = block[~np.isnan(block)]
                if np.any(values != np.floor(values)):
                    break
            else:
                return minimum + np.arange(int(span) + 1, dtype=float)

    categories = np.array([], dtype=category_dtype)
    for start, stop in row_slices(len(x), x.shape[1]):
        block = x[start:stop]
        values = block[~np.isnan(block)]
        block_categories = np.unique(values)
        if len(block_categories) > _MAX_CATEGORIES:
            return None
        categories = np.union1d(categories, block_categories)
        if len(categories) > _MAX_CATEGORIES:
            return None
    return categories


def _count_guttman_errors(
    x: np.ndarray,
    difficulty_order: np.ndarray,
    categories: np.ndarray | None,
    *,
    count_valid: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Count Guttman errors in bounded row batches of easiest-first items.

    With the easiest item first, every strictly increasing response pair is a
    response that favors a harder item over an easier one.
    """
    n_people, n_items = x.shape
    use_merges = categories is None and n_items >= 768 and x.dtype.kind in "iuf"
    # Merge levels retain sorted values, permutations and counting workspaces.
    # Include power-of-two padding in the budget, even just above a boundary.
    cells_per_row = 8 * (1 << (n_items - 1).bit_length()) if use_merges else n_items
    batch_rows = max(1, _GUTTMAN_BATCH_CELLS // cells_per_row)
    errors = np.zeros(n_people, dtype=np.int64)
    valid_counts = np.empty(n_people) if count_valid else None

    for start in range(0, n_people, batch_rows):
        stop = min(start + batch_rows, n_people)
        block = x[start:stop, difficulty_order]
        if valid_counts is not None:
            valid_counts[start:stop] = np.count_nonzero(~np.isnan(block), axis=1)
        if use_merges:
            errors[start:stop] = _count_merge_errors(block)
        elif categories is None:
            errors[start:stop] = _count_pairwise_errors(block)
        else:
            errors[start:stop] = _count_categorical_errors(block, categories)

    return errors.astype(float), valid_counts


def _count_categorical_errors(
    x_sorted: np.ndarray,
    categories: np.ndarray,
) -> np.ndarray:
    """Count increasing pairs of easiest-first items by grouping response categories."""
    errors = np.zeros(x_sorted.shape[0], dtype=np.int64)
    lower_categories = np.zeros(x_sorted.shape, dtype=bool)

    for category in range(1, len(categories)):
        # Array operands keep NumPy 1.x from rounding a float64 category to
        # the response dtype before comparing float16/float32 observations.
        lower_categories |= x_sorted == categories[category - 1 : category]
        prior_lower = np.cumsum(lower_categories, axis=1, dtype=np.int32)
        errors += np.einsum(
            "ij,ij->i",
            prior_lower,
            x_sorted == categories[category : category + 1],
            dtype=np.int64,
        )

    return errors


def _count_pairwise_errors(x_sorted: np.ndarray) -> np.ndarray:
    """Count increasing pairs of easiest-first items in one high-cardinality block."""
    n_people, n_items = x_sorted.shape
    errors = np.zeros(n_people, dtype=np.int64)

    for column in range(1, n_items):
        errors += np.count_nonzero(
            x_sorted[:, :column] < x_sorted[:, column, np.newaxis],
            axis=1,
        )

    return errors


def _count_merge_errors(x_sorted: np.ndarray) -> np.ndarray:
    """Count increasing pairs of easiest-first items across sorted runs, skipping ties and NaNs.

    Small runs use direct comparisons. Each subsequent level counts only pairs
    crossing its two halves, then retains their sorted merge for the next level.
    """
    n_people, n_items = x_sorted.shape
    width = 1 << (n_items - 1).bit_length()
    integer = x_sorted.dtype.kind in "iu"
    minimum = np.iinfo(x_sorted.dtype).min if integer else -np.inf
    values = np.full((n_people, width), minimum, dtype=x_sorted.dtype)
    values[:, :n_items] = x_sorted
    # Trailing minimum values cannot create an increasing pair, so padding
    # needs no mask and never changes the input dtype or large integer values.
    half = min(width, 32)
    runs = values.reshape(-1, half)
    errors: np.ndarray = _count_pairwise_errors(runs).reshape(n_people, -1).sum(axis=1)
    runs.sort(axis=1, kind="stable")
    del runs

    while half < width:
        groups = values.reshape(n_people, -1, 2, half)
        # Put the right half first so stable sorting places right-hand ties
        # before left-hand ties. Only strictly smaller left values move a
        # right value beyond its original position within that sorted half.
        swapped = groups[:, :, ::-1, :].reshape(n_people, -1, 2 * half)
        order = np.argsort(swapped, axis=-1, kind="stable")
        values = np.take_along_axis(swapped, order, axis=-1)
        right = order < half
        if not integer:
            right &= ~np.isnan(values)
        errors += np.sum(np.arange(2 * half) - order, axis=(1, 2), where=right)
        del groups, swapped, order, right
        half *= 2
    return errors


def guttman_flag(
    x: MatrixLike,
    threshold: float = 0.5,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Flag individuals with high Guttman error rates.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - threshold: Finite error-rate threshold for flagging (default 0.5).
                 Respondents with normalized scores strictly above it are flagged.
                 Like the other ``*_flag`` helpers, it is converted with
                 ``float()``, so numeric strings are accepted; ``True`` and
                 ``False`` are rejected.
    - na_rm: Normalization denominator passed to :func:`guttman`. If True, use
             the pairs among answered items; if False, use every item pair.

    Returns:
    - Boolean array where True indicates potentially careless responding.

    Raises:
    - ValueError: If ``threshold`` is ``True`` or ``False``, cannot be
                  converted with ``float()``, or is not finite, or if inputs
                  are invalid.

    Example:
        >>> data = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
        >>> guttman_flag(data, threshold=0.5).tolist()
        [False, False, False, False, True]
    """
    validated_threshold = validate_threshold(threshold)
    if validated_threshold is None:
        raise ValueError("threshold must be a finite number")
    scores = guttman(x, na_rm=na_rm, normalize=True)
    return scores > validated_threshold
