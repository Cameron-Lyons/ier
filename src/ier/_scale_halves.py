"""Half-scale means correlated across scales, shared by even-odd and split-half reliability."""

from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from ier._row_statistics import row_mean
from ier._validation import validate_integer

# Veltkamp's constant splits a double into halves of at most 26 significant bits.
_SPLITTER = 2.0**27 + 1
# Group totals of split halves are exact while a row's nonzero responses span at most
# this many binary exponents, less the bits of the group size.
_EXACT_SPAN = 26
_SMALLEST_NORMAL = float(np.finfo(float).smallest_normal)


def validate_factors(factors: Sequence[int]) -> list[int]:
    """Return factor sizes as positive Python integers."""
    if len(factors) == 0:
        raise ValueError("factors cannot be empty")

    return [
        validate_integer(factor_size, message="factors must contain positive integers", minimum=1)
        for factor_size in factors
    ]


def validate_factor_columns(factor_sizes: Sequence[int], n_columns: int) -> None:
    """Require factor sizes to partition every response column."""
    expected_cols = sum(factor_sizes)
    if n_columns != expected_cols:
        raise ValueError(
            f"sum of factors ({expected_cols}) must equal number of columns ({n_columns})"
        )


def factor_bounds(sizes: Sequence[int]) -> list[tuple[int, int]]:
    """Return consecutive ``(start, stop)`` column bounds for each factor."""
    bounds = []
    start = 0
    for size in sizes:
        bounds.append((start, start + size))
        start += size
    return bounds


def group_batches(groups: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batch equally sized column groups as ``(group positions, position-major columns)``."""
    sizes = np.array([len(group) for group in groups], dtype=np.intp)
    batches = []
    for size in np.unique(sizes[sizes > 0]):
        members = np.flatnonzero(sizes == size)
        columns = np.stack([groups[member] for member in members], axis=1).ravel()
        batches.append((members, columns))
    return batches


class HalfMeans:
    """Correctly rounded NaN-aware means of column groups within one block of rows.

    R's ``mean()`` rounds each half mean once in practice, so equal exact means give
    equal half means and a half-mean vector that does not vary has zero variance.
    Integer totals are already exact. Float responses are split into two halves of
    at most 26 significant bits, whose group totals are exact, and each total is then
    divided by its count with a single rounding. The split is prepared once per block
    so repeated random splits reuse it.
    """

    def __init__(self, block: np.ndarray) -> None:
        self._block = block
        self._high: np.ndarray | None = None
        if block.dtype.kind in "biu":
            return
        # Item-major rows make every group gather a run of contiguous row copies.
        values = np.array(block.T, dtype=float, order="C")
        missing = np.isnan(values)
        self._missing: np.ndarray | None = None
        if missing.any():
            np.copyto(values, 0.0, where=missing)
            self._missing = missing
        del missing
        magnitudes = np.abs(values)
        largest = magnitudes.max(axis=0)
        np.copyto(magnitudes, np.inf, where=magnitudes == 0)
        smallest = magnitudes.min(axis=0)
        del magnitudes
        _, exponents = np.frexp(largest)
        _, lowest = np.frexp(smallest)
        nonzero = np.isfinite(smallest)
        finite = np.isfinite(largest)
        # A row's nonzero responses span these binary exponents; rows with infinite
        # responses always take the exact fallback.
        self._span = np.where(nonzero, exponents - lowest, 0)
        if not finite.all():
            self._span[~finite] = _EXACT_SPAN + 1
            values[:, ~finite] = 0.0
        self._widest = int(self._span.max())
        self._exponents: np.ndarray | None = None
        if not (finite.all() and largest.max() < 2.0**900 and smallest.min() >= 2.0**-900):
            # A power-of-two scale per row keeps every split and product inside the
            # normal range; means that still fall below it are rounded exactly.
            exponents[~(nonzero & finite)] = 0
            self._exponents = exponents
            with np.errstate(under="ignore"):
                np.ldexp(values, -exponents, out=values)
        high = values * _SPLITTER
        high -= high - values
        values -= high
        self._high = high
        self._low = values if values.any() else None

    def __call__(
        self, batches: Sequence[tuple[np.ndarray, np.ndarray]], n_groups: int
    ) -> np.ndarray:
        """Return each row's mean of every column group; empty groups are NaN."""
        n_rows = len(self._block)
        means = np.full((n_rows, n_groups), np.nan)
        for members, columns in batches:
            size = len(columns) // len(members)
            if self._high is None:
                # Every (group, row) pair becomes one row of a column-major matrix, so one
                # exact reduction accumulates whole item positions with long vector operations.
                values = self._block.T[columns].reshape(size, -1).T
                means[:, members] = row_mean(values, ignore_nan=True).reshape(len(members), -1).T
                continue
            totals = self._high[columns].reshape(size, -1).sum(axis=0)
            counts: np.ndarray | int = size
            if self._missing is not None:
                counts = size - np.count_nonzero(self._missing[columns].reshape(size, -1), axis=0)
            if self._low is not None:
                low_totals = self._low[columns].reshape(size, -1).sum(axis=0)
                totals = _rounded_quotients(totals, low_totals, counts)
            else:
                # Totals of at most 26-bit halves are exact, so one division rounds once.
                with np.errstate(invalid="ignore"):
                    totals /= counts
            batch_means = totals.reshape(len(members), n_rows).T
            unsafe = self._inexact_means(batch_means, size)
            if unsafe is not None:
                for row, member in zip(*np.nonzero(unsafe), strict=True):
                    group = columns[member :: len(members)]
                    batch_means[row, member] = _exact_mean(self._block[row, group])
            means[:, members] = batch_means
        return means

    def _inexact_means(self, means: np.ndarray, size: int) -> np.ndarray | None:
        """Restore scaled means in place, flagging those that need exact rounding.

        Rows too wide for exact split totals of ``size`` responses, and means below
        the normal range, are flagged; None means that every mean is exact.
        """
        unsafe = None
        if self._exponents is not None:
            with np.errstate(under="ignore"):
                np.ldexp(means, self._exponents[:, np.newaxis], out=means)
            unsafe = (np.abs(means) < _SMALLEST_NORMAL) & (means != 0)
        limit = _EXACT_SPAN - (size - 1).bit_length()
        if self._widest > limit:
            if unsafe is None:
                unsafe = np.zeros(means.shape, dtype=bool)
            unsafe[self._span > limit] = True
        return unsafe


def _rounded_quotients(high: np.ndarray, low: np.ndarray, counts: np.ndarray | int) -> np.ndarray:
    """Divide exact totals ``high + low`` by their counts with one rounding.

    Counts have at most 26 bits and totals span few binary exponents, so Dekker's
    product gives the quotient's exact remainder. The corrected quotient then rounds
    to the same value as the exact mean: their distance is far below that mean's
    distance to any rounding midpoint, unless the mean is one, which is represented.
    """
    divisors = np.maximum(counts, 1).astype(float) if isinstance(counts, np.ndarray) else counts
    # Knuth's TwoSum: the exact total is total + residual. The inputs are owned
    # temporaries, so they hold the residual's parts.
    total = high + low
    low_part = total - high
    high -= total - low_part
    low -= low_part
    residual = high
    residual += low
    quotient: np.ndarray = total / divisors
    # Dekker's product: quotient * divisors is exactly product + product_error.
    product = quotient * divisors
    quotient_high = quotient * _SPLITTER
    quotient_high -= quotient_high - quotient
    quotient_low = quotient - quotient_high
    product_error = quotient_high
    product_error *= divisors
    product_error -= product
    quotient_low *= divisors
    product_error += quotient_low
    # The exact remainder, total + residual - quotient * divisors, corrects the quotient.
    remainder = total
    remainder -= product
    remainder -= product_error
    remainder += residual
    remainder /= divisors
    quotient += remainder
    if isinstance(counts, np.ndarray):
        quotient[counts == 0] = np.nan
    return quotient


def _exact_mean(values: np.ndarray) -> float:
    """Round the exact mean of observed responses once; infinities follow IEEE sums."""
    observed = np.asarray(values, dtype=float)
    observed = observed[~np.isnan(observed)]
    if not len(observed):
        return np.nan
    infinite = observed[np.isinf(observed)]
    if len(infinite):
        return float(infinite[0]) if np.all(infinite == infinite[0]) else np.nan
    return float(sum(map(Fraction, observed.tolist()), Fraction(0)) / len(observed))


def constant_rows(block: np.ndarray, missing: np.ndarray | None) -> np.ndarray:
    """Flag rows whose observed responses are identical, including all-missing rows.

    ``missing`` is the block's NaN mask, or None when the block has no missing values.
    """
    if missing is not None:
        first = np.argmax(~missing, axis=1)
        anchors = block[np.arange(len(block)), first, None]
        constant: np.ndarray = np.all((block == anchors) | missing, axis=1)
    else:
        constant = np.all(block == block[:, :1], axis=1)
    return constant


def spearman_brown(r: np.ndarray) -> np.ndarray:
    """Step up half correlations to full length, clamped below at -1 as in careless."""
    corrected = np.full_like(r, -1.0)
    np.divide(2 * r, 1 + r, out=corrected, where=r > -1)
    np.maximum(corrected, -1.0, out=corrected)
    corrected[np.isnan(r)] = np.nan
    return corrected
