"""
Lagged autocorrelation index for detecting cyclic and repetitive response patterns.

Correlates each respondent's response sequence with lagged copies of itself.
Careless strategies such as zigzags (1-2-3-4-5-4-3-2), seesaws (1-5-1-5), and
short cycles (2-3-4-2-3-4) produce strong positive or negative autocorrelations
at their period, even after some responses are perturbed, while attentive
responses to substantive items show weak serial dependence.

References:
- Gottfried, J., Ježek, S., Králová, M., & Řiháček, T. (2022). Autocorrelation
  screening: A potentially efficient method for detecting repetitive response
  patterns in questionnaire data. Practical Assessment, Research, and Evaluation,
  27, Article 2.
"""

from typing import Literal, TypeAlias, overload

import numpy as np

from ier._flagging import threshold_flags
from ier._response_sequences import sequence_batches
from ier._row_statistics import _integer_offsets
from ier._validation import MatrixLike, validate_integer, validate_matrix_input

AutocorrelationStatistic: TypeAlias = Literal["max_abs", "sum_abs"]

# Each lag correlation needs at least three paired responses; two always correlate perfectly.
_MIN_PAIRS = 3
# A window variance below this fraction of the row's centered sum of squares is
# recomputed with a two-pass formula, bounding one-pass cancellation error in the
# correlation near eps / fraction.
_STABLE_VARIANCE_FRACTION = 2.0**-10
# Rows with larger responses are rescaled before centering so totals stay finite.
_SAFE_MAGNITUDE = 2.0**960
# The one-pass rounding bound above, used as the relative tolerance within which
# correlations tie: magnitudes this close to 1 are exactly 1, and lags this close
# to a respondent's largest magnitude tie with it.
_TIE_TOLERANCE = float(np.finfo(float).eps) / _STABLE_VARIANCE_FRACTION
_PERFECT_CORRELATION = 1.0 - _TIE_TOLERANCE


@overload
def autocorrelation(
    x: MatrixLike,
    max_lag: int | None = 10,
    min_lag: int = 1,
    statistic: AutocorrelationStatistic = "max_abs",
    na_rm: bool = True,
    return_lags: Literal[False] = False,
) -> np.ndarray: ...
@overload
def autocorrelation(
    x: MatrixLike,
    max_lag: int | None = 10,
    min_lag: int = 1,
    statistic: AutocorrelationStatistic = "max_abs",
    na_rm: bool = True,
    *,
    return_lags: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def autocorrelation(
    x: MatrixLike,
    max_lag: int | None,
    min_lag: int,
    statistic: AutocorrelationStatistic,
    na_rm: bool,
    return_lags: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def autocorrelation(
    x: MatrixLike,
    max_lag: int | None = 10,
    min_lag: int = 1,
    statistic: AutocorrelationStatistic = "max_abs",
    na_rm: bool = True,
    return_lags: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...


def autocorrelation(
    x: MatrixLike,
    max_lag: int | None = 10,
    min_lag: int = 1,
    statistic: AutocorrelationStatistic = "max_abs",
    na_rm: bool = True,
    return_lags: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute the strongest or summed lagged autocorrelation for each respondent.

    For each lag ``k`` from ``min_lag`` to ``max_lag``, the respondent's
    responses ``r[0:n-k]`` are correlated (Pearson) with ``r[k:n]``, following
    ``responsePatterns::rp.acors`` (Gottfried et al., 2022). The score is the
    largest absolute lag correlation (``"max_abs"``) or the sum of absolute lag
    correlations (``"sum_abs"``). Higher values indicate more repetitive, cyclic
    responding. Items must be in presentation order and share one response scale.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - max_lag: Largest lag, an integer of at least ``min_lag``, or None for each
               respondent's number of observed responses minus 3 (the
               ``rp.acors`` default). The default of 10 departs from that R
               default deliberately: long lags pair few responses and dilute
               the signal of short cycles.
    - min_lag: Smallest lag, an integer of at least 1 (default 1).
    - statistic: ``"max_abs"`` (default) or ``"sum_abs"``.
    - na_rm: If True, removes missing responses while preserving sequence order.
             If False, respondents with any missing response are unavailable.
    - return_lags: If True, also return the lag with the largest absolute
                   correlation per respondent.

    Returns:
    - A numpy array of autocorrelation scores per respondent, or a tuple of
      (scores, lags) when ``return_lags`` is True. Lags are floats, the smallest
      lag wins ties, and unavailable respondents have ``NaN`` lags. Correlations
      within rounding of 1 score exactly 1, and lags within rounding of the
      largest correlation tie with it, so a perfect cycle reports its period.

      A lag is used when it pairs at least three responses, so respondents with
      fewer than ``min_lag + 3`` observed responses are unavailable (``NaN``).
      As in ``rp.acors``, a lag whose window ``r[0:n-k]`` or ``r[k:n]`` does not
      vary scores a correlation of 1. Respondents whose responses are all
      identical therefore score 1 at every usable lag, and so do lags pairing a
      straight-lined run at either end of the sequence, so straight-lining
      overlaps with ``longstring``. Rows containing infinite responses are
      unavailable.

    Raises:
    - ValueError: If lags are not integers in range, ``statistic`` is unknown,
                  ``return_lags`` is not a boolean, or the data has fewer than
                  ``min_lag + 3`` columns.

    Example:
        >>> seesaw = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
        >>> cycle = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]
        >>> attentive = [4, 2, 5, 3, 3, 1, 4, 4, 2, 5, 3, 2]
        >>> data = [seesaw, cycle, attentive]
        >>> np.round(autocorrelation(data, max_lag=3), 2).tolist()
        [1.0, 1.0, 0.44]
        >>> scores, lags = autocorrelation(data, max_lag=3, return_lags=True)
        >>> lags.tolist()
        [1.0, 3.0, 1.0]
    """
    min_lag = validate_integer(
        min_lag, message="min_lag must be an integer of at least 1", minimum=1
    )
    if max_lag is not None:
        max_lag = validate_integer(
            max_lag,
            message="max_lag must be None or an integer of at least min_lag",
            minimum=min_lag,
        )
    if not isinstance(statistic, str) or statistic not in ("max_abs", "sum_abs"):
        raise ValueError("statistic must be 'max_abs' or 'sum_abs'")
    if not isinstance(return_lags, bool):
        raise ValueError("return_lags must be a boolean")
    x_array = validate_matrix_input(x, min_columns=min_lag + _MIN_PAIRS)
    width = x_array.shape[1]
    top = width - _MIN_PAIRS if max_lag is None else min(max_lag, width - _MIN_PAIRS)

    scores = np.full(len(x_array), np.nan)
    lags = np.full(len(x_array), np.nan) if return_lags else None
    for start, stop, block, observed in sequence_batches(x_array, na_rm=True):
        counts = observed
        if counts is not None and not na_rm:
            # Missing responses propagate: incomplete rows have no usable lags.
            counts = np.where(counts == width, counts, 0)
        correlations = _lag_correlations(block, counts, width, min_lag, top)
        if correlations is None:
            continue
        magnitudes = np.abs(correlations, out=correlations)
        # Exact correlations of +/-1 can round slightly either side of 1;
        # snapping them keeps perfect cycles at exactly 1.
        np.copyto(magnitudes, 1.0, where=magnitudes >= _PERFECT_CORRELATION)
        if statistic == "max_abs":
            scores[start:stop] = np.fmax.reduce(magnitudes, axis=1)
        else:
            defined = ~np.isnan(magnitudes)
            block_scores = np.sum(magnitudes, axis=1, where=defined)
            block_scores[~np.any(defined, axis=1)] = np.nan
            scores[start:stop] = block_scores
        if lags is not None:
            # Exactly tied correlations can differ by rounding, so every lag
            # within the tie tolerance of the best competes and the smallest wins.
            best = np.fmax.reduce(magnitudes, axis=1)
            near_best = magnitudes >= (best * (1.0 - _TIE_TOLERANCE))[:, None]
            block_lags = np.argmax(near_best, axis=1) + float(min_lag)
            block_lags[np.isnan(best)] = np.nan
            lags[start:stop] = block_lags

    if lags is not None:
        return scores, lags
    return scores


def autocorrelation_flag(
    x: MatrixLike,
    threshold: float | None = None,
    percentile: float = 95.0,
    max_lag: int | None = 10,
    min_lag: int = 1,
    statistic: AutocorrelationStatistic = "max_abs",
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute lagged autocorrelation scores and flag strongly repetitive respondents.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - threshold: Absolute score at or above which to flag. If None, uses percentile.
    - percentile: Percentile above which to flag (default 95th percentile).
    - max_lag: Largest lag passed to ``autocorrelation()`` (default 10).
    - min_lag: Smallest lag passed to ``autocorrelation()`` (default 1).
    - statistic: ``"max_abs"`` (default) or ``"sum_abs"``.
    - na_rm: Missing-response policy passed to ``autocorrelation()``.

    Returns:
    - Tuple of (scores, flags) where flags is True for flagged respondents.
      Unavailable (``NaN``) scores are never flagged.

    Example:
        >>> cycle = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]
        >>> attentive = [4, 2, 5, 3, 3, 1, 4, 4, 2, 5, 3, 2]
        >>> scores, flags = autocorrelation_flag([cycle, attentive], threshold=0.9, max_lag=3)
        >>> flags.tolist()
        [True, False]
    """
    scores = autocorrelation(x, max_lag=max_lag, min_lag=min_lag, statistic=statistic, na_rm=na_rm)

    # Mirrors INDEX_REGISTRY["autocorrelation"].flag_direction; importing it here is circular.
    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="high")

    return scores, flags


def _lag_correlations(
    block: np.ndarray,
    counts: np.ndarray | None,
    width: int,
    min_lag: int,
    top: int,
) -> np.ndarray | None:
    """Correlate one bounded batch with its lags, returning ``NaN`` for unusable lags.

    Rows are left-aligned with trailing padding when ``counts`` is given. The
    result has one column per lag from ``min_lag`` through ``top``, or is None
    when no row has a usable lag. Compacted rows are padded back to the full
    ``width``, so every batch reduces a respondent's responses in the same order.
    Correlations of +/-1 may round slightly beyond 1 in magnitude.
    """
    n_rows = len(block)
    longest = min(top, block.shape[1] - _MIN_PAIRS)
    if longest < min_lag:
        return None
    lengths = np.full(n_rows, width, dtype=np.intp) if counts is None else counts
    values, finite, leading, trailing = _centered_unit_rows(block, lengths, width)

    # Window sums and sums of squares come from prefix sums. Padding is zero, so
    # trailing windows and lagged cross products need no per-row truncation.
    first = np.zeros((n_rows, width + 1))
    np.cumsum(values, axis=1, out=first[:, 1:])
    second = np.zeros((n_rows, width + 1))
    np.cumsum(np.square(values), axis=1, out=second[:, 1:])
    # Prefix sums carry rounding relative to the row's whole sum of squares.
    limits = _STABLE_VARIANCE_FRACTION * second[:, width]
    rows = np.arange(n_rows)

    correlations = np.full((n_rows, top - min_lag + 1), np.nan)
    for column, lag in enumerate(range(min_lag, longest + 1)):
        sizes = lengths - lag
        if counts is None:
            sum_a = first[:, width - lag]
            squares_a = second[:, width - lag]
        else:
            ends = np.maximum(sizes, 0)
            sum_a = first[rows, ends]
            squares_a = second[rows, ends]
        sum_b = first[:, width] - first[:, lag]
        squares_b = second[:, width] - second[:, lag]
        cross = np.einsum("ij,ij->i", values[:, : width - lag], values[:, lag:])
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_b = sum_b / sizes
            variance_a = squares_a - sum_a * (sum_a / sizes)
            variance_b = squares_b - sum_b * mean_b
            covariance = cross - sum_a * mean_b
            lag_correlations = covariance / np.sqrt(variance_a * variance_b)
        paired = (sizes >= _MIN_PAIRS) & finite
        # Window r[0:n-k] is constant when the leading run covers it, r[k:n] when
        # the trailing run does. As in rp.acors, a zero-variance window scores 1.
        flat = paired & ((leading >= sizes) | (trailing >= sizes))
        unstable = paired & ~flat & ((variance_a <= limits) | (variance_b <= limits))
        if np.any(unstable):
            lag_correlations[unstable] = _two_pass_correlations(
                _response_rows(block, unstable, width), lag, sizes[unstable]
            )
        lag_correlations[~paired] = np.nan
        lag_correlations[flat] = 1.0
        correlations[:, column] = lag_correlations
    return correlations


def _centered_unit_rows(
    block: np.ndarray, lengths: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center observed responses per row and rescale each row by a power of two.

    Returns ``width`` float64 columns with zero padding and zeroed unavailable
    rows, a mask of rows whose observed responses are all finite, and the
    lengths of the runs of identical responses that open and close each row.
    Centered values span less than one unit, which keeps sums of squares away
    from overflow and underflow.
    """
    complete = block.shape[1] == width and bool(np.all(lengths == width))
    values = _response_rows(block, slice(None), width)
    leading, trailing = _edge_runs(values, lengths, complete)
    if complete:
        lower = np.min(values, axis=1)
        upper = np.max(values, axis=1)
    else:
        observed = np.arange(width) < lengths[:, None]
        lower = np.min(values, axis=1, where=observed, initial=np.inf)
        upper = np.max(values, axis=1, where=observed, initial=-np.inf)
        np.copyto(values, 0.0, where=~observed)
    finite = np.isfinite(lower) & np.isfinite(upper)
    if not np.all(finite):
        values[~finite] = 0.0
        lower[~finite] = upper[~finite] = 0.0

    magnitudes = np.maximum(-lower, upper)
    huge = magnitudes > _SAFE_MAGNITUDE
    if np.any(huge):
        # Rescale before summing so totals and centered ranges stay finite.
        _, exponents = np.frexp(magnitudes[huge])
        values[huge] = np.ldexp(values[huge], -exponents[:, None])
        lower[huge] = np.ldexp(lower[huge], -exponents)
        upper[huge] = np.ldexp(upper[huge], -exponents)

    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.sum(values, axis=1) / lengths
    np.copyto(means, 0.0, where=lengths == 0)
    values -= means[:, None]
    if not complete:
        np.copyto(values, 0.0, where=~observed)

    # Centered responses lie within the observed range, so one power of two
    # derived from that range brings every row's deviations below one.
    spread = upper - lower
    _, exponents = np.frexp(np.where(spread > 0, spread, 1.0))
    with np.errstate(under="ignore"):
        np.ldexp(values, -exponents[:, None], out=values)
    return values, finite, leading, trailing


def _edge_runs(
    values: np.ndarray, lengths: np.ndarray, complete: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Count the identical responses that open and close each NaN-padded row.

    Rows whose observed responses are all identical have both runs equal to
    their length.
    """
    n_rows, width = values.shape
    size = n_rows * width
    # One contiguous comparison marks every response that starts a run; each
    # row's first response always does, and so does a final sentinel.
    marks = np.ones(size + 1, dtype=bool)
    flat = values.reshape(-1)
    np.not_equal(flat[1:], flat[:-1], out=marks[1:size])
    starts = marks[:size].reshape(n_rows, width)
    starts[:, 0] = True
    if not complete:
        starts[:, 1:] &= np.arange(1, width) < lengths[:, None]
    # Shifted by one response, each row ends with the next row's first start.
    leading = np.argmax(marks[1:].reshape(n_rows, width), axis=1) + 1
    np.minimum(leading, lengths, out=leading)
    trailing = lengths - (width - 1 - np.argmax(starts[:, ::-1], axis=1))
    return leading, trailing


def _response_rows(block: np.ndarray, rows: np.ndarray | slice, width: int) -> np.ndarray:
    """Copy selected rows as C-ordered float64 responses padded with NaN to ``width``."""
    selected = block[rows]
    if selected.dtype.kind in "iu" and _beyond_exact_integers(selected):
        # Exact integer distances from each row minimum keep adjacent large values.
        selected = _integer_offsets(selected)
    if selected.shape[1] == width:
        return np.array(selected, dtype=float, order="C")
    responses = np.full((len(selected), width), np.nan)
    responses[:, : selected.shape[1]] = selected
    return responses


def _beyond_exact_integers(x: np.ndarray) -> bool:
    """Return whether integer responses could round when converted to float64."""
    if x.dtype.itemsize < 8:
        return False
    return bool(int(np.max(x)) > 2**53 or int(np.min(x)) < -(2**53))


def _two_pass_correlations(responses: np.ndarray, lag: int, sizes: np.ndarray) -> np.ndarray:
    """Recompute lag correlations from each window's own centered responses.

    ``responses`` holds uncentered rows: centering a whole row first would round
    each response relative to the row's spread rather than its window's. Windows
    whose observed responses are identical have an undefined (``NaN``) correlation.
    """
    width = responses.shape[1]
    magnitudes = np.fmax.reduce(np.abs(responses), axis=1)
    huge = magnitudes > _SAFE_MAGNITUDE
    if np.any(huge):
        # Power-of-two scaling of extreme rows keeps every window range finite.
        _, exponents = np.frexp(magnitudes[huge])
        responses[huge] = np.ldexp(responses[huge], -exponents[:, None])

    observed = np.arange(width - lag) < sizes[:, None]
    first = responses[:, : width - lag]
    second = responses[:, lag:]
    first_lower = np.min(first, axis=1, where=observed, initial=np.inf)
    first_upper = np.max(first, axis=1, where=observed, initial=-np.inf)
    second_lower = np.min(second, axis=1, where=observed, initial=np.inf)
    second_upper = np.max(second, axis=1, where=observed, initial=-np.inf)
    varying = (first_lower < first_upper) & (second_lower < second_upper)
    correlations = np.full(len(responses), np.nan)
    if not np.any(varying):
        return correlations

    observed = observed[varying]
    sizes = sizes[varying]
    first_deviations, first_residuals = _window_deviations(
        first[varying], first_lower[varying], first_upper[varying], observed, sizes
    )
    second_deviations, second_residuals = _window_deviations(
        second[varying], second_lower[varying], second_upper[varying], observed, sizes
    )
    # Corrected two-pass sums remove the rounding left in each window mean.
    covariance = np.einsum("ij,ij->i", first_deviations, second_deviations)
    covariance -= first_residuals * second_residuals / sizes
    first_variance = np.einsum("ij,ij->i", first_deviations, first_deviations)
    first_variance -= first_residuals * first_residuals / sizes
    second_variance = np.einsum("ij,ij->i", second_deviations, second_deviations)
    second_variance -= second_residuals * second_residuals / sizes
    correlations[varying] = covariance / np.sqrt(first_variance) / np.sqrt(second_variance)
    return correlations


def _window_deviations(
    window: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    observed: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center varying windows on their own means after scaling their ranges below one.

    Returns deviations with zero padding and their rounding residual sums.
    """
    _, exponents = np.frexp(upper - lower)
    # Distances from the window minimum round relative to the window's own spread.
    scaled = window - lower[:, None]
    with np.errstate(under="ignore"):
        np.ldexp(scaled, -exponents[:, None], out=scaled)
    scaled -= (np.sum(scaled, axis=1, where=observed) / sizes)[:, None]
    np.copyto(scaled, 0.0, where=~observed)
    residuals: np.ndarray = np.sum(scaled, axis=1)
    return scaled, residuals
