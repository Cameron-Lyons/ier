"""Reverse-score reverse-keyed items before consistency and person-fit scoring."""

import math
import numbers
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from ier._pair_statistics import _reflection_parameters
from ier._row_statistics import row_slices
from ier._validation import (
    MatrixLike,
    resolve_scale_bounds,
    validate_item_indices,
    validate_matrix_input,
)


def reverse_score(
    x: MatrixLike,
    items: Sequence[int],
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> np.ndarray:
    """
    Reverse-score selected items, returning a recoded copy of the responses.

    Each selected response ``v`` becomes ``scale_min + scale_max - v``, so a 1 on a
    1-5 scale becomes 5. Recode reverse-worded items before indices that assume all
    items are keyed in the same direction: ``evenodd``, ``individual_reliability``,
    ``guttman``, ``lz``, ``gpoly``, ``u3poly_fit``, and ``ht``, the indices whose
    ``index_catalog()`` entry sets ``uses_keyed_responses``. Sequence and
    response-style indices such as ``longstring``, ``markov``, ``irv``, and
    ``acquiescence`` should keep the responses as presented.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - items: Distinct zero-based column indices of the reverse-keyed items.
    - scale_min: Minimum response-scale value. If None, the observed minimum of
                 the whole matrix is used.
    - scale_max: Maximum response-scale value. If None, the observed maximum of
                 the whole matrix is used. Pass explicit bounds whenever an
                 endpoint might be unobserved.

    Returns:
    - A new array with the selected columns reverse-scored; the input is never
      modified and missing responses stay ``NaN``. Integer and Boolean responses
      with integral bounds are recoded exactly in an integer dtype: the input
      dtype when it holds both bounds, otherwise the smallest integer dtype that
      holds the input dtype's range and both bounds. Unsigned 64-bit responses
      with a negative bound have no such dtype; they, integer responses with a
      non-integral bound, and other inputs return float64 with each exact
      reflection rounded once, including integer responses beyond ``2**53``.
      Apart from converting non-array input, the returned copy is the only
      full-size allocation: the selected columns are checked and recoded in
      bounded row batches.

    Raises:
    - ValueError: If inputs are invalid, items are empty, duplicated, or out of
                  range, bounds are not finite real numbers or are reversed, or a
                  selected response lies outside the bounds. Responses are compared
                  with the bounds exactly, whatever their precision.

    Example:
        >>> data = [[1, 2, 5], [4, 5, 1], [3, 3, 2]]
        >>> print(reverse_score(data, [2], scale_min=1, scale_max=5).tolist())
        [[1, 2, 1], [4, 5, 5], [3, 3, 4]]
        >>> print(reverse_score([[1.5, float("nan")], [4.0, 2.0]], [0, 1]).tolist())
        [[4.0, nan], [1.5, 3.5]]
    """
    x_array = validate_matrix_input(x)
    selected = validate_item_indices(items, x_array.shape[1], name="items")
    bounds = resolve_scale_bounds(
        x_array,
        scale_min=_validate_bound(scale_min, "scale_min"),
        scale_max=_validate_bound(scale_max, "scale_max"),
    )
    if bounds is None:
        # Every response is missing, so there is nothing to recode.
        return np.array(x_array, dtype=float)
    lower, upper = bounds
    if not (math.isfinite(lower) and math.isfinite(upper)):
        # Only an inferred endpoint can be infinite here: the data contain infinity.
        raise ValueError("scale bounds must be finite; infinite responses cannot be reversed")

    observed_min, observed_max = _selected_extremes(x_array, selected)
    if observed_min < _exact_bound(lower) or observed_max > _exact_bound(upper):
        raise ValueError("reverse-scored responses must lie within scale_min and scale_max")

    if x_array.dtype.kind in "biu" and float(lower).is_integer() and float(upper).is_integer():
        recoded = _reverse_integers(x_array, selected, int(lower), int(upper))
        if recoded is not None:
            return recoded

    result = np.array(x_array, dtype=float)
    _reverse_floats(result, selected, (float(lower), float(upper)))
    if x_array.dtype.kind in "biu":
        _repair_integer_reflections(result, x_array, selected, lower, upper)
    return result


def _selected_extremes(
    x: np.ndarray, selected: np.ndarray
) -> tuple[float | Fraction, float | Fraction]:
    """Return the exact extremes of the selected columns from bounded row batches."""
    minima = []
    maxima = []
    for start, stop in row_slices(len(x), len(selected)):
        block = x[start:stop, selected]
        # NaN-ignoring reductions; an all-missing selection has nothing to check.
        minima.append(np.fmin.reduce(block, axis=None))
        maxima.append(np.fmax.reduce(block, axis=None))
    if x.dtype.kind in "biu":
        # Compare exact integer extremes before any conversion can round them.
        return min(int(value) for value in minima), max(int(value) for value in maxima)
    # Exact Python values keep the bounds from rounding to the input precision.
    return (
        _exact_value(np.fmin.reduce(np.array(minima))),
        _exact_value(np.fmax.reduce(np.array(maxima))),
    )


def _exact_value(value: np.floating) -> float | Fraction:
    """Return a NumPy floating scalar as an exactly equal Python number."""
    as_float = float(value)
    if not math.isfinite(as_float) or as_float == value:
        # Wider values beyond float64 range compare like the infinity they round to.
        return as_float
    return Fraction(*value.as_integer_ratio())


def _exact_bound(bound: float) -> float | Fraction:
    """Return an endpoint exactly; extended-precision data infer extended endpoints."""
    endpoint: object = bound
    return _exact_value(endpoint) if isinstance(endpoint, np.floating) else bound


def _validate_bound(value: object, name: str) -> float | None:
    """Return None or a finite real endpoint, keeping integer endpoints exact."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number or None")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite real number or None")
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def _reverse_integers(
    x: np.ndarray, selected: np.ndarray, lower: int, upper: int
) -> np.ndarray | None:
    """Recode integer responses exactly in a dtype that holds the data and both bounds."""
    if x.dtype.kind == "b":
        data_min, data_max = 0, 1
    else:
        data_min, data_max = int(np.iinfo(x.dtype).min), int(np.iinfo(x.dtype).max)
    candidates = [x.dtype] if x.dtype.kind in "iu" else []
    # Signed data excludes the unsigned candidates, whose minimum is zero.
    candidates += map(np.dtype, ("u1", "i1", "u2", "i2", "u4", "i4", "u8", "i8"))
    dtype = next(
        (
            dtype
            for dtype in candidates
            if np.iinfo(dtype).min <= min(lower, data_min)
            and max(upper, data_max) <= np.iinfo(dtype).max
        ),
        None,
    )
    if dtype is None:
        # Unsigned 64-bit responses and a negative bound share no integer dtype.
        return None
    result = x.astype(dtype)
    # Responses lie within the bounds, so each recoded value fits the result dtype.
    work_dtype: type = np.int64
    width = len(selected)
    if abs(lower) + abs(upper) > 2**62:
        # Python integers cost several times an int64, so their batches are narrower.
        work_dtype, width = object, 8 * width
    for start, stop in row_slices(len(x), width):
        values: np.ndarray = x[start:stop, selected].astype(work_dtype, copy=False)
        result[start:stop, selected] = (lower + upper) - values
    return result


def _repair_integer_reflections(
    result: np.ndarray, x: np.ndarray, selected: np.ndarray, lower: float, upper: float
) -> None:
    """Round exact reflections of integer responses that float64 cannot hold exactly."""
    exact_bounds = float(lower) == lower and float(upper) == upper
    if exact_bounds and x.dtype.itemsize < 8:
        return
    # Both bounds are dyadic, so a power-of-two denominator puts each reflection
    # over an exact integer numerator; integer true division rounds it once.
    endpoints = Fraction(lower) + Fraction(upper)
    # Python numbers cost several times a float64, so these batches are narrower.
    for start, stop in row_slices(len(x), 8 * len(selected)):
        values = x[start:stop, selected]
        if exact_bounds:
            # Integers up to 2**53 convert exactly, so their float reflections are
            # already rounded once; only larger responses were rounded beforehand.
            inexact = values > 2**53
            if x.dtype.kind == "i":
                inexact |= values < -(2**53)
            rows, columns = np.nonzero(inexact)
        else:
            # The float reflection used rounded bounds, so recompute every response.
            rows, columns = np.indices(values.shape).reshape(2, -1)
        if len(rows):
            responses = values[rows, columns].astype(object)
            reflected = (
                endpoints.numerator - responses * endpoints.denominator
            ) / endpoints.denominator
            result[start + rows, selected[columns]] = reflected.astype(float)


def _two_sum(left: float | np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rounded sum and its exact rounding error (Knuth's TwoSum)."""
    total = left + right
    right_part = total - left
    error = (left - (total - right_part)) + (right - right_part)
    return total, error


def _reverse_floats(result: np.ndarray, selected: np.ndarray, bounds: tuple[float, float]) -> None:
    """Reflect in-bounds columns in place, rounding each exact reflection once."""
    reflection, remainder, exact = _reflection_parameters(bounds)
    assert exact is not None
    halved = not math.isfinite(reflection)
    if halved:
        # Same-signed endpoints near the float64 limit: halving is exact at these
        # magnitudes and keeps the endpoint sum finite.
        reflection, remainder, _ = _reflection_parameters((bounds[0] / 2, bounds[1] / 2))
    # Bounded row batches keep the working arrays small whatever the matrix size;
    # the error-free sums hold about eight batch-sized arrays at once.
    width = len(selected) if remainder == 0 else 8 * len(selected)
    for start, stop in row_slices(len(result), width):
        values = result[start:stop, selected]
        if halved:
            values *= 0.5
        np.negative(values, out=values)
        recoded, unsafe = _reflect_negated(values, reflection, remainder)
        if halved:
            recoded *= 2
        if unsafe is not None:
            for row, column in zip(*np.nonzero(unsafe), strict=True):
                response = Fraction(float(result[start + row, selected[column]]))
                recoded[row, column] = float(exact - response)
        result[start:stop, selected] = recoded


def _reflect_negated(
    values: np.ndarray, reflection: float, remainder: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Round each ``reflection + remainder + value`` once, marking sums to recompute.

    Returns the rounded sums, reusing ``values`` when the endpoint sum is exact, and
    a mask of the rare sums that must be rounded exactly instead, or None.
    """
    if remainder == 0:
        # An exactly representable endpoint sum leaves one IEEE addition, which
        # already rounds each exact reflection once.
        values += reflection
        return values, None
    # Error-free sums give reflection + remainder - value exactly as
    # recoded + residual + correction_error, with recoded rounded once.
    reflected, error = _two_sum(reflection, values)
    correction, correction_error = _two_sum(remainder, error)
    del error
    recoded, residual = _two_sum(reflected, correction)
    del reflected, correction
    # A rounded correction matters only when it can carry the exact value across a
    # rounding midpoint of recoded; those rare values are rounded exactly instead.
    # The gap toward zero is the narrower one, so half of it bounds both midpoints.
    magnitude = np.abs(recoded)
    half_gap = magnitude - np.nextafter(magnitude, 0)
    del magnitude
    half_gap /= 2
    np.abs(correction_error, out=correction_error)
    np.abs(residual, out=residual)
    residual += correction_error
    unsafe = (correction_error > 0) & (residual >= half_gap)
    return recoded, unsafe
