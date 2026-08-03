"""
Takes a matrix of item responses and identifies item pairs that are highly correlated within the
overall dataset. What defines "highly correlated" is set by the critical value (e.g., r > .60). Each
respondents' psychometric synonym score is then computed as the within-person correlation be-
tween the identified item-pairs. Alternatively computes the psychometric antonym score which is a
variant that uses item pairs that are highly negatively correlated.

This module provides functions for detecting careless responding patterns by analyzing how
individuals respond to psychometrically similar (synonym) or opposite (antonym) items.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from math import isqrt
from typing import Literal, overload

import numpy as np

from ier._correlation import row_correlations
from ier._summary import calculate_summary_stats
from ier._validation import MatrixLike, validate_matrix_input
from ier.types import PsychsynSummary

_PSYCHSYN_BATCH_ELEMENTS = 262_144
_PSYCHSYN_CORRELATION_BLOCK_ELEMENTS = 262_144


def _readonly_item_pairs(values: np.ndarray, n_items: int) -> np.ndarray:
    """Return validated, independently owned psychometric item pairs."""
    try:
        raw = np.asarray(values)
        if raw.dtype.kind == "c":
            raise ValueError
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("item_pairs must be a numeric two-column array") from error
    if numeric.ndim != 2 or numeric.shape[1] != 2:
        raise ValueError("item_pairs must be a two-column array")
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError("item_pairs must contain only finite integer indices")

    if np.any((numeric < 0) | (numeric >= n_items)):
        raise ValueError("item_pairs contains indices outside the fitted item range")
    pairs = np.asarray(numeric, dtype=np.intp)
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("item_pairs cannot pair an item with itself")
    if len(pairs):
        undirected = np.sort(pairs, axis=1)
        if len(np.unique(undirected, axis=0)) != len(pairs):
            raise ValueError("item_pairs cannot contain duplicate pairs")

    result = np.array(pairs, dtype=np.intp, copy=True)
    result.setflags(write=False)
    return result


def _validate_psychsyn_options(critval: float, anto: bool) -> float:
    """Validate shared synonym/antonym discovery options."""
    if not isinstance(critval, (int, float)) or isinstance(critval, bool):
        raise ValueError("critval must be a number")
    value = float(critval)
    if not np.isfinite(value):
        raise ValueError("critval must be finite")
    if not isinstance(anto, bool):
        raise ValueError("anto must be a boolean")
    if anto and value > 0:
        raise ValueError("critval should be negative for antonym analysis")
    if not anto and value < 0:
        raise ValueError("critval should be positive for synonym analysis")
    return value


@dataclass(frozen=True, eq=False)
class PsychsynModel:
    """Immutable item-pair calibration for psychometric scoring."""

    item_pairs: np.ndarray
    n_items: int
    critval: float = 0.60
    anto: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.n_items, int) or isinstance(self.n_items, bool):
            raise ValueError("n_items must be an integer")
        if self.n_items < 2:
            raise ValueError("n_items must be at least 2")
        if self.n_items > np.iinfo(np.intp).max:
            raise ValueError("n_items must fit in the platform index range")
        critval = _validate_psychsyn_options(self.critval, self.anto)
        item_pairs = _readonly_item_pairs(self.item_pairs, self.n_items)
        object.__setattr__(self, "critval", critval)
        object.__setattr__(self, "item_pairs", item_pairs)

    @property
    def n_pairs(self) -> int:
        """Number of calibrated psychometric item pairs."""
        return len(self.item_pairs)


def _complete_item_normalization(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return column offsets, norms, and usability without retaining a matrix copy."""
    n_rows, n_items = x.shape
    item_offsets = np.empty(n_items)
    item_norms = np.empty(n_items)
    valid_columns = np.zeros(n_items, dtype=bool)
    block_columns = max(1, _PSYCHSYN_CORRELATION_BLOCK_ELEMENTS // max(1, n_rows))

    for start in range(0, n_items, block_columns):
        stop = min(start + block_columns, n_items)
        block = np.array(x[:, start:stop], dtype=float, copy=True)
        finite = np.isfinite(block).all(axis=0)
        block[:, ~finite] = 0.0
        block_offsets = np.mean(block, axis=0)
        block -= block_offsets
        norms = np.sqrt(np.einsum("ij,ij->j", block, block))
        usable = finite & (norms > 0.0)
        item_offsets[start:stop] = block_offsets
        item_norms[start:stop] = norms
        valid_columns[start:stop] = usable

    return item_offsets, item_norms, valid_columns


def _normalized_complete_item_block(
    x: np.ndarray,
    item_offsets: np.ndarray,
    item_norms: np.ndarray,
    valid_columns: np.ndarray,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
) -> np.ndarray:
    """Copy and normalize one bounded complete-data item block."""
    block = np.array(
        x[row_start:row_stop, column_start:column_stop],
        dtype=float,
        copy=True,
    )
    block -= item_offsets[column_start:column_stop]
    usable = valid_columns[column_start:column_stop]
    np.divide(
        block,
        item_norms[column_start:column_stop],
        out=block,
        where=usable[np.newaxis, :],
    )
    block[:, ~usable] = 0.0
    return block


def _iter_item_correlation_tiles(
    x: np.ndarray,
    item_offsets: np.ndarray,
    item_norms: np.ndarray,
    valid_columns: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield complete-data item correlations from bounded triangular tiles."""
    n_rows, n_items = x.shape
    tile_width = max(1, isqrt(_PSYCHSYN_CORRELATION_BLOCK_ELEMENTS))
    all_indices = np.arange(n_items, dtype=np.intp)

    for row_start in range(0, n_items, tile_width):
        row_stop = min(row_start + tile_width, n_items)
        row_indices = all_indices[row_start:row_stop]
        for column_start in range(0, row_stop, tile_width):
            column_stop = min(column_start + tile_width, row_stop)
            column_indices = all_indices[column_start:column_stop]
            row_width = row_stop - row_start
            column_width = column_stop - column_start
            same_tile = row_start == column_start and row_stop == column_stop
            workspace_width = row_width if same_tile else row_width + column_width
            batch_rows = max(
                1,
                _PSYCHSYN_CORRELATION_BLOCK_ELEMENTS // workspace_width,
            )
            correlations = np.zeros((row_width, column_width))

            for start in range(0, n_rows, batch_rows):
                stop = min(start + batch_rows, n_rows)
                row_values = _normalized_complete_item_block(
                    x,
                    item_offsets,
                    item_norms,
                    valid_columns,
                    start,
                    stop,
                    row_start,
                    row_stop,
                )
                if same_tile:
                    column_values = row_values
                else:
                    column_values = _normalized_complete_item_block(
                        x,
                        item_offsets,
                        item_norms,
                        valid_columns,
                        start,
                        stop,
                        column_start,
                        column_stop,
                    )
                correlations += row_values.T @ column_values
            np.clip(correlations, -1.0, 1.0, out=correlations)

            lower_rows, lower_columns = np.nonzero(
                row_indices[:, np.newaxis] > column_indices[np.newaxis, :]
            )
            if len(lower_rows) == 0:
                continue
            values = correlations[lower_rows, lower_columns]
            usable = (
                valid_columns[row_indices[lower_rows]]
                & valid_columns[column_indices[lower_columns]]
            )
            values[~usable] = np.nan
            yield (
                row_indices[lower_rows],
                column_indices[lower_columns],
                values,
            )


def _pairwise_item_correlation_tile(
    x: np.ndarray,
    item_offsets: np.ndarray,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
) -> np.ndarray:
    """Correlate two item blocks from pairwise-complete raw moments."""
    row_width = row_stop - row_start
    column_width = column_stop - column_start
    shape = (row_width, column_width)
    counts = np.zeros(shape, dtype=np.intp)
    sums_row = np.zeros(shape)
    sums_column = np.zeros(shape)
    sums_squares_row = np.zeros(shape)
    sums_squares_column = np.zeros(shape)
    correlations = np.zeros(shape)
    same_tile = row_start == column_start and row_stop == column_stop
    batch_rows = max(
        1,
        _PSYCHSYN_CORRELATION_BLOCK_ELEMENTS // max(1, row_width + column_width),
    )

    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        row_values = np.array(x[start:stop, row_start:row_stop], dtype=float, copy=True)
        row_valid = np.isfinite(row_values)
        np.subtract(
            row_values,
            item_offsets[row_start:row_stop],
            out=row_values,
            where=row_valid,
        )
        row_values[~row_valid] = 0.0

        if same_tile:
            column_values = row_values
            column_valid = row_valid
        else:
            column_values = np.array(
                x[start:stop, column_start:column_stop],
                dtype=float,
                copy=True,
            )
            column_valid = np.isfinite(column_values)
            np.subtract(
                column_values,
                item_offsets[column_start:column_stop],
                out=column_values,
                where=column_valid,
            )
            column_values[~column_valid] = 0.0

        counts += np.matmul(row_valid.T, column_valid, dtype=np.intp)
        correlations += row_values.T @ column_values
        sums_row += row_values.T @ column_valid
        row_squares = np.square(row_values)
        sums_squares_row += row_squares.T @ column_valid

        if not same_tile:
            sums_column += row_valid.T @ column_values
            sums_squares_column += row_valid.T @ np.square(column_values)

    if same_tile:
        sums_column = sums_row.T.copy()
        sums_squares_column = sums_squares_row.T.copy()

    usable = counts >= 2
    count_values = counts.astype(float)
    adjustment = np.zeros(shape)

    np.multiply(sums_row, sums_column, out=adjustment)
    np.divide(adjustment, count_values, out=adjustment, where=usable)
    correlations -= adjustment

    np.square(sums_row, out=sums_row)
    np.divide(sums_row, count_values, out=sums_row, where=usable)
    sums_squares_row -= sums_row
    np.square(sums_column, out=sums_column)
    np.divide(sums_column, count_values, out=sums_column, where=usable)
    sums_squares_column -= sums_column

    np.maximum(sums_squares_row, 0.0, out=sums_squares_row)
    np.maximum(sums_squares_column, 0.0, out=sums_squares_column)
    np.multiply(
        sums_squares_row,
        sums_squares_column,
        out=sums_squares_row,
    )
    np.sqrt(sums_squares_row, out=sums_squares_row)
    usable &= sums_squares_row > 0.0

    result = np.full(shape, np.nan)
    np.divide(
        correlations,
        sums_squares_row,
        out=result,
        where=usable,
    )
    np.clip(result, -1.0, 1.0, out=result)
    return result


def _finite_item_offsets(x: np.ndarray) -> np.ndarray:
    """Choose finite column origins without cancellation or a matrix-sized temporary."""
    n_rows, n_items = x.shape
    offsets = np.zeros(n_items)
    found = np.zeros(n_items, dtype=bool)
    batch_rows = max(1, _PSYCHSYN_CORRELATION_BLOCK_ELEMENTS // n_items)

    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        block = x[start:stop]
        valid = np.isfinite(block)
        newly_found = ~found & np.any(valid, axis=0)
        if np.any(newly_found):
            columns = np.flatnonzero(newly_found)
            rows = np.argmax(valid[:, columns], axis=0)
            offsets[columns] = block[rows, columns]
            found[columns] = True
        if np.all(found):
            break

    return offsets


def _iter_pairwise_item_correlation_tiles(
    x: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield pairwise-complete item correlations in bounded triangular tiles."""
    n_items = x.shape[1]
    tile_width = max(1, isqrt(_PSYCHSYN_CORRELATION_BLOCK_ELEMENTS))
    all_indices = np.arange(n_items, dtype=np.intp)
    item_offsets = _finite_item_offsets(x)

    for row_start in range(0, n_items, tile_width):
        row_stop = min(row_start + tile_width, n_items)
        row_indices = all_indices[row_start:row_stop]
        for column_start in range(0, row_stop, tile_width):
            column_stop = min(column_start + tile_width, row_stop)
            column_indices = all_indices[column_start:column_stop]
            correlations = _pairwise_item_correlation_tile(
                x,
                item_offsets,
                row_start,
                row_stop,
                column_start,
                column_stop,
            )

            lower_rows, lower_columns = np.nonzero(
                row_indices[:, np.newaxis] > column_indices[np.newaxis, :]
            )
            if len(lower_rows) == 0:
                continue
            yield (
                row_indices[lower_rows],
                column_indices[lower_columns],
                correlations[lower_rows, lower_columns],
            )


def _iter_discovery_correlation_tiles(
    x: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Use the fast complete-data kernel or pairwise-complete missing-data kernel."""
    if np.isfinite(x).all():
        item_offsets, item_norms, valid_columns = _complete_item_normalization(x)
        yield from _iter_item_correlation_tiles(
            x,
            item_offsets,
            item_norms,
            valid_columns,
        )
        return

    yield from _iter_pairwise_item_correlation_tiles(x)


def _discover_item_pairs(x: np.ndarray, critval: float, anto: bool) -> np.ndarray:
    """Discover threshold-matching item pairs without a complete square matrix."""
    selected_blocks: list[np.ndarray] = []
    for row_indices, column_indices, correlations in _iter_discovery_correlation_tiles(x):
        selected = correlations <= critval if anto else correlations >= critval
        if np.any(selected):
            selected_blocks.append(
                np.column_stack((row_indices[selected], column_indices[selected]))
            )

    if not selected_blocks:
        return np.empty((0, 2), dtype=np.intp)
    pairs = np.concatenate(selected_blocks)
    ordered_pairs: np.ndarray = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return ordered_pairs


def fit_psychsyn_model(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
) -> PsychsynModel:
    """Discover and retain psychometric item pairs for later cohort scoring.

    The fitted model records the reference matrix's item count, threshold, mode,
    and an immutable copy of every selected pair. Later matrices must use the
    same item count and column order.

    Parameters:
    - x: Reference response matrix used for pair discovery.
    - critval: Correlation threshold for synonym or antonym discovery.
    - anto: Whether to discover negatively correlated antonym pairs.

    Returns:
    - An immutable psychometric item-pair calibration.
    """
    x_array = validate_matrix_input(x, min_columns=2)
    resolved_critval = _validate_psychsyn_options(critval, anto)
    return PsychsynModel(
        item_pairs=_discover_item_pairs(x_array, resolved_critval, anto),
        n_items=x_array.shape[1],
        critval=resolved_critval,
        anto=anto,
    )


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
def psychsyn_model_scores(
    x: MatrixLike,
    model: PsychsynModel,
    *,
    diag: Literal[False] = False,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> np.ndarray:
    pass


@overload
def psychsyn_model_scores(
    x: MatrixLike,
    model: PsychsynModel,
    *,
    diag: Literal[True],
    resample_na: bool = False,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pass


def psychsyn_model_scores(
    x: MatrixLike,
    model: PsychsynModel,
    *,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Score a cohort with a fitted psychometric item-pair calibration.

    Unlike :func:`psychsyn`, this function never rediscovers item pairs. The
    scoring matrix must retain the fitted item count and column order.

    Parameters:
    - x: Response matrix to score.
    - model: Calibration returned by :func:`fit_psychsyn_model`.
    - diag: Whether to return each respondent's usable pair count.
    - resample_na: Whether to retry undefined missing-response correlations.
    - random_seed: Optional seed for reproducible retries.

    Returns:
    - Scores, or ``(scores, usable_pair_counts)`` when ``diag=True``.
    """
    if not isinstance(model, PsychsynModel):
        raise TypeError("model must be a PsychsynModel")
    x_array = validate_matrix_input(x, min_columns=2)
    if x_array.shape[1] != model.n_items:
        raise ValueError(
            f"scoring data has {x_array.shape[1]} items; model requires {model.n_items}"
        )
    scores, diag_values = _score_psychsyn_pairs(
        x_array,
        model.item_pairs,
        resample_na=resample_na,
        rng=np.random.default_rng(random_seed),
    )
    if diag:
        return scores, diag_values
    return scores


@overload
def psychsyn(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
    diag: Literal[False] = False,
    resample_na: bool = False,
    random_seed: int | None = None,
    _return_item_info: Literal[False] = False,
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
    _return_item_info: Literal[True] = True,
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

    resolved_critval = _validate_psychsyn_options(critval, anto)

    rng = np.random.default_rng(random_seed)

    item_pairs = _discover_item_pairs(x_array, resolved_critval, anto)
    scores, diag_values = _score_psychsyn_pairs(
        x_array,
        item_pairs,
        resample_na=resample_na,
        rng=rng,
    )

    if _return_item_info:
        return (scores, diag_values, item_pairs)
    if diag:
        return (scores, diag_values)
    result: np.ndarray = scores
    return result


def _compute_complete_person_scores(
    x: np.ndarray,
    item_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score finite responses in bounded respondent batches."""
    scores = np.empty(len(x))
    n_pairs = len(item_pairs)
    if n_pairs < 3:
        return np.full(len(x), np.nan), np.full(len(x), n_pairs, dtype=int)
    batch_rows = max(1, _PSYCHSYN_BATCH_ELEMENTS // n_pairs)

    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        response_i = x[start:stop, item_pairs[:, 0]]
        response_j = x[start:stop, item_pairs[:, 1]]
        scores[start:stop] = row_correlations(response_i, response_j)

    diag_values = np.full(len(x), n_pairs, dtype=int)
    return scores, diag_values


def _compute_person_scores(
    x: np.ndarray,
    item_pairs: np.ndarray,
    *,
    resample_na: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Score selected pairs in bounded batches, including missing-response inputs."""
    if np.isfinite(x).all():
        return _compute_complete_person_scores(x, item_pairs)

    n_rows = len(x)
    n_pairs = len(item_pairs)
    batch_rows = max(1, _PSYCHSYN_BATCH_ELEMENTS // n_pairs)
    scores = np.full(n_rows, np.nan)
    diag_values = np.zeros(n_rows, dtype=int)

    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        response_i = x[start:stop, item_pairs[:, 0]]
        response_j = x[start:stop, item_pairs[:, 1]]
        valid = np.isfinite(response_i) & np.isfinite(response_j)
        valid_counts = np.sum(valid, axis=1, dtype=np.intp)
        response_i[~valid] = np.nan
        response_j[~valid] = np.nan

        batch_scores = row_correlations(
            response_i,
            response_j,
            zero_variance=np.nan,
        )
        batch_scores[valid_counts < 3] = np.nan
        scores[start:stop] = batch_scores
        diag_values[start:stop] = valid_counts

    missing_rows = np.isnan(scores)
    if not resample_na or not np.any(missing_rows):
        return scores, diag_values

    _resample_undefined_person_scores(
        x,
        item_pairs,
        scores,
        diag_values,
        rng,
        batch_rows,
    )
    return scores, diag_values


def _score_psychsyn_pairs(
    x: np.ndarray,
    item_pairs: np.ndarray,
    *,
    resample_na: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply fixed item pairs and preserve the established unavailable-score policy."""
    if len(item_pairs) == 0:
        return np.full(len(x), np.nan), np.zeros(len(x), dtype=int)

    scores, diag_values = _compute_person_scores(
        x,
        item_pairs,
        resample_na=resample_na,
        rng=rng,
    )
    if np.any(np.isnan(scores)):
        scores = np.nan_to_num(scores, nan=0.0)
    return scores, diag_values


def _resample_undefined_person_scores(
    x: np.ndarray,
    item_pairs: np.ndarray,
    scores: np.ndarray,
    pair_counts: np.ndarray,
    rng: np.random.Generator,
    batch_rows: int,
) -> None:
    """Retry undefined correlations by randomly swapping available pair directions."""
    unresolved = np.flatnonzero(np.isnan(scores) & (pair_counts >= 3))

    for start in range(0, len(unresolved), batch_rows):
        row_indices = unresolved[start : start + batch_rows]
        left = x[row_indices[:, np.newaxis], item_pairs[:, 0]]
        right = x[row_indices[:, np.newaxis], item_pairs[:, 1]]
        valid = np.isfinite(left) & np.isfinite(right)
        left[~valid] = np.nan
        right[~valid] = np.nan

        pending = np.arange(len(row_indices))
        for _ in range(10):
            swap = rng.integers(0, 2, size=left.shape, dtype=np.int8).astype(bool)
            swap &= valid
            held = left[swap].copy()
            left[swap] = right[swap]
            right[swap] = held

            candidates = row_correlations(
                left[pending],
                right[pending],
                zero_variance=np.nan,
            )
            resolved = np.isfinite(candidates)
            if np.any(resolved):
                scores[row_indices[pending[resolved]]] = candidates[resolved]
                pending = pending[~resolved]
            if len(pending) == 0:
                break


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

    pair_blocks: list[np.ndarray] = []
    correlation_blocks: list[np.ndarray] = []
    for row_indices, column_indices, correlations in _iter_discovery_correlation_tiles(x_array):
        selected = ~np.isnan(correlations) & (np.abs(correlations) >= min_correlation)
        if np.any(selected):
            pair_blocks.append(np.column_stack((column_indices[selected], row_indices[selected])))
            correlation_blocks.append(correlations[selected])

    if not pair_blocks:
        return []
    pairs = np.concatenate(pair_blocks)
    corr_filtered = np.concatenate(correlation_blocks)

    sort_indices = np.argsort(corr_filtered) if anto else np.argsort(-corr_filtered)

    correlation_list: list[tuple[int, int, float]] = [
        (int(pairs[idx, 0]), int(pairs[idx, 1]), float(corr_filtered[idx])) for idx in sort_indices
    ]

    return correlation_list


@overload
def psychant(
    x: MatrixLike,
    critval: float = -0.60,
    diag: Literal[False] = False,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> np.ndarray:
    pass


@overload
def psychant(
    x: MatrixLike,
    critval: float = -0.60,
    diag: Literal[True] = True,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pass


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


def psychsyn_summary(
    x: MatrixLike,
    critval: float = 0.60,
    anto: bool = False,
) -> PsychsynSummary:
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
    stats = calculate_summary_stats(scores, suffix="_score")
    return {
        "mean_score": stats["mean_score"],
        "std_score": stats["std_score"],
        "min_score": stats["min_score"],
        "max_score": stats["max_score"],
        "median_score": stats["median_score"],
        "item_pairs": len(item_pairs),
        "total_individuals": len(scores),
        "valid_individuals": valid_count,
        "missing_individuals": len(scores) - valid_count,
    }
