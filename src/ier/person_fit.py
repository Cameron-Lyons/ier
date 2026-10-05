"""
Item-step person-fit statistics: polytomous Guttman errors, U3, and Ht.

Nonparametric person-fit statistics compare each respondent's answers with the
item ordering of the whole sample. An item with ``M + 1`` ordered categories is
split into ``M`` item steps ``X_j >= h`` (``h = 1..M``), and a step's popularity
is the proportion of respondents who pass it. Attentive respondents tend to
pass popular steps before unpopular ones; random or careless responding passes
unpopular steps while failing popular ones.

- ``gpoly`` counts polytomous Guttman errors (Molenaar, 1991): pairs of item
  steps where the respondent fails the more popular step but passes the less
  popular one. ``normalize=True`` divides by the largest count possible for the
  respondent's total score (Emons, 2008), as PerFit's ``Gnormed.poly`` does;
  ``normalize=False`` returns the raw count of PerFit's ``Gpoly``.
- ``u3poly`` is the polytomous U3 statistic (van der Flier, 1982; Emons, 2008),
  which weights every passed step by the log-odds of its popularity, as in
  PerFit's ``U3poly``.
- ``ht`` is the transposed scalability coefficient for dichotomous items
  (Sijtsma & Meijer, 1992), as in PerFit's ``Ht``: the covariance of a
  respondent's item scores with everyone else's, relative to its maximum.

Higher ``gpoly`` and ``u3poly`` values and lower ``ht`` values indicate
aberrant responding. Niessen, Meijer, and Tendeiro (2016) found polytomous
Guttman errors effective for detecting careless respondents, and Ht was among
the best-performing statistics in Karabatsos (2003).

References:
- Emons, W. H. M. (2008). Nonparametric person-fit analysis of polytomous item
  scores. Applied Psychological Measurement, 32(3), 224-247.
- Karabatsos, G. (2003). Comparing the aberrant response detection performance
  of thirty-six person-fit statistics. Applied Measurement in Education, 16(4),
  277-298.
- Molenaar, I. W. (1991). A weighted Loevinger H-coefficient extending Mokken
  scaling to multicategory items. Kwantitatieve Methoden, 12(37), 97-117.
- Niessen, A. S. M., Meijer, R. R., & Tendeiro, J. N. (2016). Detecting careless
  respondents in web-based questionnaires: Which method to use? Journal of
  Research in Personality, 63, 1-11.
- Sijtsma, K., & Meijer, R. R. (1992). A method for investigating the
  intersection of item response functions in Mokken's nonparametric IRT model.
  Applied Psychological Measurement, 16(2), 149-157.
- Tendeiro, J. N., Meijer, R. R., & Niessen, A. S. M. (2016). PerFit: An R
  package for person-fit analysis in IRT. Journal of Statistical Software,
  74(5), 1-27.
- van der Flier, H. (1982). Deviant response patterns and comparability of test
  scores. Journal of Cross-Cultural Psychology, 13(3), 267-298.
"""

import math
import numbers
from typing import NamedTuple, SupportsFloat

import numpy as np

from ier._flagging import threshold_flags
from ier._row_statistics import row_slices
from ier._validation import (
    MatrixLike,
    resolve_scale_bounds,
    validate_integer,
    validate_matrix_input,
)

# Item steps describe ordinal rating scales. Wider integer ranges, such as
# identifiers or timings, are rejected before per-category tables are allocated.
_MAX_CATEGORIES = 1024
# U3 weights become integers whose absolute sum stays below 2**61, so every
# partial sum and difference of two sums is exact in int64.
_WEIGHT_SUM_BITS = 61
# Ht products reach items**2 * respondents; larger samples use Python integers.
_HT_INT64_LIMIT = 2**62
_MISSING_MESSAGE = "data contains missing values. Set na_rm=True to handle them"


class _ItemSteps(NamedTuple):
    """Validated responses as zero-based categories with their counts."""

    codes: np.ndarray
    """int16 categories ``0..M`` per respondent and item; ``-1`` marks a missing response."""
    counts: np.ndarray
    """Observed responses per item (rows) and category (columns), as int64."""
    complete: np.ndarray
    """Boolean mask of respondents without missing responses."""


def gpoly(
    x: MatrixLike,
    ncat: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    normalize: bool = True,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Count polytomous Guttman errors over item steps for each respondent.

    Responses on a scale with ``M + 1`` categories are split into the item steps
    ``X_j >= h`` for ``h = 1..M``. Steps are ordered by popularity, the share of
    respondents passing them, most popular first. Ties keep item order and then
    step order, like ``rank(ties.method = "first")`` in PerFit, so the steps of
    one item stay nested. A Guttman error is a pair of steps where the
    respondent fails the more popular step but passes the less popular one
    (Molenaar, 1991), which equals PerFit's ``Gpoly``. With ``normalize=True``
    the count is divided by the largest count any response pattern with the
    same total score can reach (Emons, 2008), PerFit's ``Gnormed.poly``. For
    dichotomous items this is ``G / (r * (J - r))`` with ``r`` correct answers.

    Parameters:
    - x: A matrix of integer category responses where rows are individuals and
         columns are items. All items share one response scale.
    - ncat: Number of response categories per item, at least 2. If None, the
            categories run from ``scale_min`` to ``scale_max``.
    - scale_min: Lowest response category, an integer. If None, inferred from
                 ``scale_max`` and ``ncat`` or from the observed minimum. Pass
                 it when the lowest category may be unobserved.
    - scale_max: Highest response category, an integer. If None, inferred from
                 ``scale_min`` and ``ncat`` or from the observed maximum.
    - normalize: If True (default), divide by the maximum error count for the
                 respondent's total score. If False, return raw counts.
    - na_rm: If True, step popularities use each item's observed responses and
             raw counts include only pairs of observed steps, while normalized
             scores of respondents with missing responses are ``NaN``. If False,
             missing responses raise ``ValueError``.

    Returns:
    - A numpy array of Guttman error counts or proportions. Higher values
      indicate less consistent responding. Normalized scores are ``NaN`` when
      no error is possible for the total score (for example, every step failed
      or passed); respondents without observed responses are ``NaN``.

    Raises:
    - ValueError: If responses are not integer categories within the scale,
                  a given scale bound is not a finite integer, ``ncat`` is not
                  an integer of at least 2 or disagrees with both scale bounds,
                  the scale has more than 1024 categories, or ``na_rm=False``
                  and responses are missing.

    Example:
        >>> data = [[2, 2, 1, 0], [2, 1, 1, 0], [1, 1, 0, 0],
        ...         [2, 2, 2, 1], [2, 0, 2, 0], [0, 0, 1, 2]]
        >>> gpoly(data, normalize=False).tolist()  # the last respondent reverses the order
        [0.0, 0.0, 2.0, 0.0, 2.0, 11.0]
        >>> np.round(gpoly(data), 2).tolist()
        [0.0, 0.0, 0.17, 0.0, 0.14, 0.85]
    """
    x_array = validate_matrix_input(x, min_columns=2)
    scores = np.full(len(x_array), np.nan)
    steps = _item_steps(x_array, ncat, scale_min, scale_max, na_rm=na_rm)
    if steps is None:
        return scores

    ranks = _step_ranks(steps.counts)
    table = _step_prefix_sums(ranks)
    codes, totals = _complete_rows(steps)
    # The k-th passed step at rank t follows t - k failed steps, so the errors
    # are the passed ranks minus the smallest possible rank sum k(k+1)/2.
    errors = _row_table_sums(codes, table) - _triangular(totals)
    if normalize:
        all_totals = np.arange(ranks.size + 1, dtype=np.int64)
        maxima = _nested_step_extremes(table, maximize=True) - _triangular(all_totals)
        scores[steps.complete] = _ratio(errors, maxima[totals])
        return scores

    scores[steps.complete] = errors
    incomplete = ~steps.complete
    if np.any(incomplete):
        partial = steps.codes[incomplete]
        answered = np.any(partial >= 0, axis=1)
        raw = np.full(len(partial), np.nan)
        raw[answered] = _observed_step_errors(partial[answered], ranks)
        scores[incomplete] = raw
    return scores


def u3poly(
    x: MatrixLike,
    ncat: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Calculate the polytomous U3 person-fit statistic over item steps.

    Item steps and their popularities are defined as in :func:`gpoly`. Each
    step is weighted by the log-odds of its popularity, ``log(p / (1 - p))``,
    so passing popular steps adds the most. With ``W`` the summed weight of a
    respondent's passed steps and ``W_max`` and ``W_min`` its largest and
    smallest values over all nested response patterns with the same total
    score, ``U3 = (W_max - W) / (W_max - W_min)`` (van der Flier, 1982; Emons,
    2008), as in PerFit's ``U3poly``. A step passed by every respondent, or by
    none, is weighted as if half a response went the other way, which keeps its
    weight finite. Weights share one fixed-point scale with 61 bits for their
    summed magnitude, so the extremes, perfect patterns (0), and least likely
    patterns (1) are exact.

    This is the person-fit statistic. :func:`ier.u3_poly` is unrelated: it
    measures the proportion of extreme responses.

    Parameters:
    - x: A matrix of integer category responses where rows are individuals and
         columns are items. All items share one response scale.
    - ncat: Number of response categories per item, at least 2. If None, the
            categories run from ``scale_min`` to ``scale_max``.
    - scale_min: Lowest response category, an integer. If None, inferred from
                 ``scale_max`` and ``ncat`` or from the observed minimum.
    - scale_max: Highest response category, an integer. If None, inferred from
                 ``scale_min`` and ``ncat`` or from the observed maximum.
    - na_rm: If True, step popularities use each item's observed responses and
             respondents with missing responses are ``NaN``. If False, missing
             responses raise ``ValueError``.

    Returns:
    - A numpy array of U3 values from 0 (a perfect Guttman pattern) to 1 (the
      lowest weight possible for the total score). Higher values indicate less
      consistent responding. Respondents whose total score allows only one
      pattern weight (for example, every step failed or passed) are ``NaN``.

    Raises:
    - ValueError: If responses are not integer categories within the scale,
                  a given scale bound is not a finite integer, ``ncat`` is not
                  an integer of at least 2 or disagrees with both scale bounds,
                  the scale has more than 1024 categories, or ``na_rm=False``
                  and responses are missing.

    Example:
        >>> data = [[2, 2, 1, 0], [2, 1, 1, 0], [1, 1, 0, 0],
        ...         [2, 2, 2, 1], [2, 0, 2, 0], [0, 0, 1, 2]]
        >>> np.round(u3poly(data), 2).tolist()
        [0.0, 0.0, 0.17, 0.0, 0.2, 0.83]
    """
    x_array = validate_matrix_input(x, min_columns=2)
    scores = np.full(len(x_array), np.nan)
    steps = _item_steps(x_array, ncat, scale_min, scale_max, na_rm=na_rm)
    if steps is None:
        return scores

    weights = _step_weights(steps.counts)
    table = _step_prefix_sums(weights)
    # The most popular steps form a nested pattern, so the unconstrained
    # largest weights give the maximum for every total score.
    largest = np.zeros(weights.size + 1, dtype=np.int64)
    np.cumsum(np.sort(weights, axis=None)[::-1], out=largest[1:])
    spreads = largest - _nested_step_extremes(table, maximize=False)
    codes, totals = _complete_rows(steps)
    shortfalls = largest[totals] - _row_table_sums(codes, table)
    scores[steps.complete] = _ratio(shortfalls, spreads[totals])
    return scores


def ht(x: MatrixLike, na_rm: bool = True) -> np.ndarray:
    """
    Calculate the transposed scalability coefficient Ht for dichotomous items.

    For respondent ``n``, ``Ht`` sums the covariances between ``n``'s item
    scores and those of every other respondent, taken across items, and divides
    by the sum of their maxima given each pair's numbers of correct answers
    (Sijtsma & Meijer, 1992), as in PerFit's ``Ht``. Respondents whose answers
    follow the sample's item ordering score high, up to 1; values near or below
    0 indicate aberrant responding. With ``r`` the respondents' numbers of
    correct answers, ``c`` the item totals, and ``R`` the sum of ``r``, the
    numerator is ``J * (x_n . c - r_n) - r_n * (R - r_n)`` and the denominator
    is ``J * (sum_m min(r_n, r_m) - r_n) - r_n * (R - r_n)``. Both are computed
    in exact integer arithmetic in ``O(N * J)`` time.

    Parameters:
    - x: A matrix of dichotomous responses coded 0 and 1 where rows are
         individuals and columns are items.
    - na_rm: If True, respondents with missing responses are ``NaN`` and are
             left out of everyone else's comparisons. If False, missing
             responses raise ``ValueError``.

    Returns:
    - A numpy array of Ht values of at most 1. Lower values indicate less
      consistent responding. Respondents who answer every item or no item
      correctly, or whose comparisons have no positive maximum covariance,
      are ``NaN``.

    Raises:
    - ValueError: If responses are not 0 or 1, or ``na_rm=False`` and responses
                  are missing.

    Example:
        >>> data = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0],
        ...         [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
        >>> np.round(ht(data), 2).tolist()  # the last respondent reverses the order
        [0.6, 0.43, 0.5, 0.6, 0.43, -1.0]
    """
    x_array = validate_matrix_input(x, min_columns=2)
    steps = _categorize(x_array, 0, 1, "ht requires dichotomous responses coded 0 and 1")
    if not na_rm and not np.all(steps.complete):
        raise ValueError(_MISSING_MESSAGE)
    scores = np.full(len(x_array), np.nan)
    codes, totals = _complete_rows(steps)
    n_rows, n_items = codes.shape
    item_totals = np.stack(
        [np.zeros(n_items, dtype=np.int64), np.sum(codes, axis=0, dtype=np.int64)], axis=1
    )
    # Items both answered correctly, summed over every respondent m: x_n . c.
    shared = _row_table_sums(codes, item_totals)
    # min(r_n, r_m) summed over every respondent m, from the number-correct counts.
    frequencies = np.bincount(totals, minlength=n_items + 1)
    levels = np.arange(n_items + 1, dtype=np.int64)
    by_level = np.cumsum(levels * frequencies) + levels * (n_rows - np.cumsum(frequencies))
    smaller: np.ndarray = by_level[totals]
    if n_rows * n_items * n_items >= _HT_INT64_LIMIT:
        totals, shared, smaller = (values.astype(object) for values in (totals, shared, smaller))
    # Subtracting each respondent's pairing with itself leaves the m != n sums.
    products = totals * (np.sum(totals) - totals)
    covariances = n_items * (shared - totals) - products
    maxima = n_items * (smaller - totals) - products
    scores[steps.complete] = _ratio(covariances, maxima)
    return scores


def gpoly_flag(
    x: MatrixLike,
    threshold: float | None = None,
    percentile: float = 95.0,
    *,
    ncat: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    normalize: bool = True,
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute polytomous Guttman errors and flag unusually many.

    Parameters:
    - x: A matrix of integer category responses where rows are individuals and
         columns are items.
    - threshold: Absolute score at or above which to flag. If None, uses percentile.
    - percentile: Percentile above which to flag (default 95th percentile).
    - ncat: Number of response categories per item, as in :func:`gpoly`.
    - scale_min: Lowest response category, as in :func:`gpoly`.
    - scale_max: Highest response category, as in :func:`gpoly`.
    - normalize: If True (default), flag normalized scores; otherwise raw counts.
    - na_rm: Missing-response policy passed to :func:`gpoly`.

    Returns:
    - Tuple of (scores, flags) where flags is True for flagged respondents.
      Unavailable (``NaN``) scores are never flagged.

    Example:
        >>> data = [[2, 2, 1, 0], [2, 1, 1, 0], [1, 1, 0, 0],
        ...         [2, 2, 2, 1], [2, 0, 2, 0], [0, 0, 1, 2]]
        >>> scores, flags = gpoly_flag(data, threshold=0.5)
        >>> flags.tolist()
        [False, False, False, False, False, True]
    """
    scores = gpoly(
        x,
        ncat=ncat,
        scale_min=scale_min,
        scale_max=scale_max,
        normalize=normalize,
        na_rm=na_rm,
    )

    # Mirrors INDEX_REGISTRY["gpoly"].flag_direction; importing it here is circular.
    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="high")

    return scores, flags


def u3poly_flag(
    x: MatrixLike,
    threshold: float | None = None,
    percentile: float = 95.0,
    *,
    ncat: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute polytomous U3 person-fit values and flag unusually high ones.

    Parameters:
    - x: A matrix of integer category responses where rows are individuals and
         columns are items.
    - threshold: Absolute U3 value at or above which to flag. If None, uses percentile.
    - percentile: Percentile above which to flag (default 95th percentile).
    - ncat: Number of response categories per item, as in :func:`u3poly`.
    - scale_min: Lowest response category, as in :func:`u3poly`.
    - scale_max: Highest response category, as in :func:`u3poly`.
    - na_rm: Missing-response policy passed to :func:`u3poly`.

    Returns:
    - Tuple of (scores, flags) where flags is True for flagged respondents.
      Unavailable (``NaN``) scores are never flagged.

    Example:
        >>> data = [[2, 2, 1, 0], [2, 1, 1, 0], [1, 1, 0, 0],
        ...         [2, 2, 2, 1], [2, 0, 2, 0], [0, 0, 1, 2]]
        >>> scores, flags = u3poly_flag(data, threshold=0.5)
        >>> flags.tolist()
        [False, False, False, False, False, True]
    """
    scores = u3poly(x, ncat=ncat, scale_min=scale_min, scale_max=scale_max, na_rm=na_rm)

    # Mirrors INDEX_REGISTRY["u3poly_fit"].flag_direction; importing it here is circular.
    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="high")

    return scores, flags


def ht_flag(
    x: MatrixLike,
    threshold: float | None = None,
    percentile: float = 5.0,
    *,
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Ht person-fit values and flag unusually low ones.

    Parameters:
    - x: A matrix of dichotomous responses coded 0 and 1.
    - threshold: Absolute Ht value at or below which to flag. If None, uses percentile.
    - percentile: Percentile below which to flag (default 5th percentile).
    - na_rm: Missing-response policy passed to :func:`ht`.

    Returns:
    - Tuple of (scores, flags) where flags is True for flagged respondents.
      Unavailable (``NaN``) scores are never flagged.

    Example:
        >>> data = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0],
        ...         [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
        >>> scores, flags = ht_flag(data, threshold=0.0)
        >>> flags.tolist()
        [False, False, False, False, False, True]
    """
    scores = ht(x, na_rm=na_rm)

    # Mirrors INDEX_REGISTRY["ht"].flag_direction; importing it here is circular.
    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="low")

    return scores, flags


def _item_steps(
    x: np.ndarray,
    ncat: int | None,
    scale_min: float | None,
    scale_max: float | None,
    *,
    na_rm: bool,
) -> _ItemSteps | None:
    """Validate polytomous responses, returning None when none are observed."""
    bounds = _category_bounds(x, ncat, scale_min, scale_max)
    if bounds is None:
        if not na_rm:
            raise ValueError(_MISSING_MESSAGE)
        return None
    lower, upper = bounds
    steps = _categorize(
        x, lower, upper, f"responses must be integer categories from {lower} to {upper}"
    )
    if not na_rm and not np.all(steps.complete):
        raise ValueError(_MISSING_MESSAGE)
    return steps


def _category_bounds(
    x: np.ndarray,
    ncat: int | None,
    scale_min: float | None,
    scale_max: float | None,
) -> tuple[int, int] | None:
    """Resolve the integer scale from ``ncat`` and given or observed endpoints.

    ``ncat`` and one given endpoint fix the other before any response is read,
    so responses outside that scale are reported as such. With neither endpoint
    given, ``ncat`` extends the observed minimum upward. Registry recoding of
    reverse-keyed items for ``gpoly`` and ``u3poly_fit`` uses this scale too.
    """
    if ncat is not None:
        ncat = validate_integer(ncat, message="ncat must be an integer of at least 2", minimum=2)
    given_min = None if scale_min is None else _integer_bound(scale_min, name="scale_min")
    given_max = None if scale_max is None else _integer_bound(scale_max, name="scale_max")
    if ncat is not None:
        if given_max is None and given_min is not None:
            given_max = given_min + ncat - 1
        elif given_min is None and given_max is not None:
            given_min = given_max - ncat + 1
    bounds = resolve_scale_bounds(x, scale_min=given_min, scale_max=given_max)
    if bounds is None:
        return None
    # Given endpoints are already integers; inferred ones are observed responses.
    lower = _integer_bound(bounds[0], name=None)
    upper = _integer_bound(bounds[1], name=None)
    if ncat is not None:
        if given_min is None and given_max is None:
            upper = lower + ncat - 1
        elif upper - lower != ncat - 1:
            raise ValueError("ncat must equal scale_max - scale_min + 1")
    if upper - lower >= _MAX_CATEGORIES:
        raise ValueError(
            f"item-step person-fit statistics support at most {_MAX_CATEGORIES} response "
            f"categories (got {upper - lower + 1})"
        )
    return lower, upper


def _integer_bound(value: SupportsFloat, *, name: str | None) -> int:
    """Return a finite integral scale endpoint, naming a given endpoint in errors.

    NumPy floats are tested in their own precision, because ``float()`` could
    round an extended-precision fraction to an integer.
    """
    if isinstance(value, numbers.Integral):
        return int(value)
    number = value if isinstance(value, np.floating) else float(value)
    if np.isfinite(number) and np.floor(number) == number:
        return int(number)
    if name is None:
        raise ValueError("responses must be finite integer categories")
    raise ValueError(f"{name} must be a finite integer")


def _categorize(x: np.ndarray, lower: int, upper: int, message: str) -> _ItemSteps:
    """Convert integer responses from ``lower`` to ``upper`` to zero-based categories."""
    n_rows, n_items = x.shape
    width = upper - lower + 1
    codes = np.empty(x.shape, dtype=np.int16)
    counts = np.zeros(n_items * width, dtype=np.int64)
    complete = np.ones(n_rows, dtype=bool)
    offsets = np.arange(0, n_items * width, width, dtype=np.intp)
    for start, stop in row_slices(n_rows, n_items):
        categories, missing = _block_categories(x[start:stop], lower, upper, message)
        codes[start:stop] = categories
        cells = codes[start:stop] + offsets
        if missing is not None:
            complete[start:stop] = ~np.any(missing, axis=1)
            cells = cells[~missing]
        counts += np.bincount(cells.reshape(-1), minlength=counts.size)
    return _ItemSteps(codes, counts.reshape(n_items, width), complete)


def _block_categories(
    block: np.ndarray, lower: int, upper: int, message: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate one row batch, returning its categories and any missing-response mask.

    Missing responses become category -1; the mask is None without them.
    """
    if block.dtype.kind == "f":
        # Check responses in their own precision: converting extended-precision
        # values to float64 first could round a fraction to an integer.
        # fmin and fmax skip missing responses, and NaN never compares below its floor.
        smallest = np.fmin.reduce(block, axis=None)
        largest = np.fmax.reduce(block, axis=None)
        missing = np.isnan(block)
        if np.isnan(smallest):
            return np.full(block.shape, -1, dtype=np.int16), missing
        if (
            not (np.isfinite(smallest) and np.isfinite(largest))
            or int(np.floor(smallest)) < lower
            or int(np.ceil(largest)) > upper
            or np.any(np.floor(block) < block)
        ):
            raise ValueError(message)
        # In-range integers differ by less than 1024, so subtracting the batch
        # minimum is exact in every floating dtype, as is the remaining shift.
        categories = block - smallest
        categories += int(smallest) - lower
        if not np.any(missing):
            return categories, None
        categories[missing] = -1
        return categories, missing

    # Integer and Boolean responses: compare as Python integers, then shift
    # by the batch minimum, which the response dtype can always represent.
    smallest, largest = int(np.min(block)), int(np.max(block))
    if smallest < lower or largest > upper:
        raise ValueError(message)
    if block.dtype.kind == "u" and block.dtype.itemsize == 8:
        shifted = (block - block.dtype.type(smallest)).astype(np.int64)
    else:
        shifted = block.astype(np.int64) - smallest
    return shifted + (smallest - lower), None


def _complete_rows(steps: _ItemSteps) -> tuple[np.ndarray, np.ndarray]:
    """Return the complete respondents' categories and their total scores."""
    codes = steps.codes if np.all(steps.complete) else steps.codes[steps.complete]
    totals = np.asarray(np.sum(codes, axis=1, dtype=np.int64))
    return codes, totals


def _passing_counts(counts: np.ndarray) -> np.ndarray:
    """Count respondents passing each step ``X_j >= h`` for ``h = 1..M``."""
    return np.cumsum(counts[:, :0:-1], axis=1)[:, ::-1]


def _step_ranks(counts: np.ndarray) -> np.ndarray:
    """Rank item steps by descending popularity, starting from 1.

    Ties keep item-major order, so one item's steps stay in level order and a
    step never ranks before an easier step of its own item. Items without
    observed responses rank last.
    """
    observed = np.sum(counts, axis=1, keepdims=True)
    popularity = np.divide(
        _passing_counts(counts),
        observed,
        out=np.full((len(counts), counts.shape[1] - 1), -1.0),
        where=observed > 0,
    )
    order = np.argsort(-popularity, axis=None, kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1)
    return ranks.reshape(popularity.shape)


def _step_weights(counts: np.ndarray) -> np.ndarray:
    """Return each step's log-odds popularity as an int64 fixed-point value.

    Clipping the passing count to half a response from either end keeps every
    weight finite. Within an item the weights cannot increase with the step,
    which ``_nested_step_extremes`` relies on.
    """
    passing = _passing_counts(counts).astype(float)
    observed = np.sum(counts, axis=1).astype(float)
    weights = np.zeros(passing.shape)
    answered = observed > 0
    totals = observed[answered, np.newaxis]
    passed = np.clip(passing[answered], 0.5, totals - 0.5)
    weights[answered] = np.log(passed) - np.log(totals - passed)
    magnitude = float(np.sum(np.abs(weights)))
    if magnitude == 0.0:
        return np.zeros(weights.shape, dtype=np.int64)
    exponent = _WEIGHT_SUM_BITS - math.frexp(magnitude)[1]
    quantized: np.ndarray = np.rint(np.ldexp(weights, exponent)).astype(np.int64)
    return quantized


def _step_prefix_sums(values: np.ndarray) -> np.ndarray:
    """Return ``table[j, k]``, the sum of item ``j``'s first ``k`` step values."""
    table = np.zeros((values.shape[0], values.shape[1] + 1), dtype=np.int64)
    np.cumsum(values, axis=1, out=table[:, 1:])
    return table


def _triangular(values: np.ndarray) -> np.ndarray:
    """Return ``k * (k + 1) / 2``, the rank sum of the ``k`` most popular steps."""
    result: np.ndarray = values * (values + 1) // 2
    return result


def _row_table_sums(codes: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Sum ``table[j, codes[:, j]]`` across items for complete respondents."""
    n_items, width = table.shape
    flat = table.reshape(-1)
    offsets = np.arange(0, n_items * width, width, dtype=np.intp)
    sums = np.empty(len(codes), dtype=np.int64)
    for start, stop in row_slices(len(codes), n_items):
        sums[start:stop] = np.sum(flat[codes[start:stop] + offsets], axis=1)
    return sums


def _nested_step_extremes(table: np.ndarray, *, maximize: bool) -> np.ndarray:
    """Return the largest or smallest nested step sum for every total score.

    A nested pattern passes a prefix of each item's steps, and ``table[j, k]``
    sums item ``j``'s first ``k`` step values. Maximizing requires rows with
    increasing increments (convex in ``k``); minimizing requires decreasing
    increments. The objective is then convex on the polytope of fractional step
    counts with a fixed total, so an optimum lies at a vertex: ``q = total // M``
    items pass every step, one more item passes ``total % M`` steps, and the
    rest pass none. The complete items are the ``q`` best, unless the partial
    item is one of them, when the next best complete item takes its place.
    This needs ``O(J * M)`` work instead of a dynamic program over totals.
    """
    n_items, width = table.shape
    n_steps = width - 1
    if n_steps == 0:
        return np.zeros(1, dtype=table.dtype)
    values = table if maximize else -table
    ranked = values[np.argsort(-values[:, -1], kind="stable")]
    best_complete = np.zeros(n_items + 1, dtype=values.dtype)
    np.cumsum(ranked[:, -1], out=best_complete[1:])

    extremes = np.empty(n_items * n_steps + 1, dtype=values.dtype)
    extremes[-1] = best_complete[-1]
    by_complete = extremes[:-1].reshape(n_items, n_steps)
    by_complete[:, 0] = best_complete[:-1]
    if n_steps > 1:
        partial = ranked[:, 1:-1]
        # Row q: the partial item ranks after the q best complete items ...
        by_complete[:, 1:] = (
            best_complete[:-1, np.newaxis] + np.maximum.accumulate(partial[::-1], axis=0)[::-1]
        )
        # ... or ranks among them, and the (q + 1)-th best item completes the set.
        exchanged = np.maximum.accumulate(partial - ranked[:, -1:], axis=0)[:-1]
        np.maximum(
            by_complete[1:, 1:],
            best_complete[2:, np.newaxis] + exchanged,
            out=by_complete[1:, 1:],
        )
    return extremes if maximize else -extremes


def _observed_step_errors(codes: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Count Guttman errors among each respondent's observed steps."""
    n_steps = ranks.shape[1]
    order = np.argsort(ranks, axis=None)
    items, levels = np.divmod(order, n_steps)
    levels = (levels + 1).astype(np.int16)
    errors = np.empty(len(codes), dtype=np.int64)
    # Each batch holds step categories, two masks, and running failure counts.
    for start, stop in row_slices(len(codes), 8 * order.size):
        categories = codes[start:stop][:, items]
        passed = categories >= levels
        failed = (categories >= 0) & ~passed
        prior_failures = np.cumsum(failed, axis=1, dtype=np.int32)
        errors[start:stop] = np.sum(prior_failures, axis=1, where=passed, dtype=np.int64)
    return errors


def _ratio(numerators: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    """Divide exact integer statistics, leaving nonpositive denominators unavailable."""
    valid = np.asarray(denominators > 0, dtype=bool)
    ratios = np.full(len(numerators), np.nan)
    ratios[valid] = np.asarray(numerators[valid] / denominators[valid], dtype=float)
    return ratios
