"""Batched reverse scoring matches the whole-matrix recoding bit for bit in bounded memory."""

from __future__ import annotations

import math
import re
import tracemalloc
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from ier import IndexOptions, reverse_score, screen
from ier._pair_statistics import _reflection_parameters
from ier._validation import resolve_scale_bounds, validate_item_indices, validate_matrix_input
from ier.keying import _exact_bound, _exact_value, _validate_bound

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ier._validation import MatrixLike


# The whole-matrix implementation that bounded batches replaced, kept as the oracle.
def _reference_reverse_score(
    x: MatrixLike,
    items: Sequence[int],
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> np.ndarray:
    x_array = validate_matrix_input(x)
    selected = validate_item_indices(items, x_array.shape[1], name="items")
    bounds = resolve_scale_bounds(
        x_array,
        scale_min=_validate_bound(scale_min, "scale_min"),
        scale_max=_validate_bound(scale_max, "scale_max"),
    )
    if bounds is None:
        return np.array(x_array, dtype=float)
    lower, upper = bounds
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError("scale bounds must be finite; infinite responses cannot be reversed")

    observed = x_array[:, selected]
    observed_min: float | Fraction
    observed_max: float | Fraction
    if x_array.dtype.kind in "biu":
        observed_min, observed_max = int(observed.min()), int(observed.max())
    else:
        observed_min = _exact_value(np.fmin.reduce(observed, axis=None))
        observed_max = _exact_value(np.fmax.reduce(observed, axis=None))
    if observed_min < _exact_bound(lower) or observed_max > _exact_bound(upper):
        raise ValueError("reverse-scored responses must lie within scale_min and scale_max")
    del observed

    if x_array.dtype.kind in "biu" and float(lower).is_integer() and float(upper).is_integer():
        recoded = _reference_reverse_integers(x_array, selected, int(lower), int(upper))
        if recoded is not None:
            return recoded

    result = np.array(x_array, dtype=float)
    _reference_reverse_floats(result, selected, (float(lower), float(upper)))
    if x_array.dtype.kind in "biu":
        _reference_repair_integer_reflections(result, x_array, selected, lower, upper)
    return result


def _reference_reverse_integers(
    x: np.ndarray, selected: np.ndarray, lower: int, upper: int
) -> np.ndarray | None:
    if x.dtype.kind == "b":
        data_min, data_max = 0, 1
    else:
        data_min, data_max = int(np.iinfo(x.dtype).min), int(np.iinfo(x.dtype).max)
    candidates = [x.dtype] if x.dtype.kind in "iu" else []
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
        return None
    result = x.astype(dtype)
    values = x[:, selected]
    if abs(lower) + abs(upper) <= 2**62:
        result[:, selected] = (lower + upper) - values.astype(np.int64)
    else:
        result[:, selected] = (lower + upper) - values.astype(object)
    return result


def _reference_repair_integer_reflections(
    result: np.ndarray, x: np.ndarray, selected: np.ndarray, lower: float, upper: float
) -> None:
    values = x[:, selected]
    if float(lower) == lower and float(upper) == upper:
        if x.dtype.itemsize < 8:
            return
        inexact = values > 2**53
        if x.dtype.kind == "i":
            inexact |= values < -(2**53)
        rows, columns = np.nonzero(inexact)
    else:
        rows, columns = np.indices(values.shape).reshape(2, -1)
    if not len(rows):
        return
    endpoints = Fraction(lower) + Fraction(upper)
    responses = values[rows, columns].astype(object)
    reflected = (endpoints.numerator - responses * endpoints.denominator) / endpoints.denominator
    result[rows, selected[columns]] = reflected.astype(float)


def _reference_two_sum(
    left: float | np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = left + right
    right_part = total - left
    error = (left - (total - right_part)) + (right - right_part)
    return total, error


def _reference_reverse_floats(
    result: np.ndarray, selected: np.ndarray, bounds: tuple[float, float]
) -> None:
    values = result[:, selected]
    reflection, remainder, exact = _reflection_parameters(bounds)
    assert exact is not None
    halved = not math.isfinite(reflection)
    if halved:
        values *= 0.5
        reflection, remainder, _ = _reflection_parameters((bounds[0] / 2, bounds[1] / 2))
    np.negative(values, out=values)
    reflected, error = _reference_two_sum(reflection, values)
    correction, correction_error = _reference_two_sum(remainder, error)
    recoded, residual = _reference_two_sum(reflected, correction)
    magnitude = np.abs(recoded)
    half_gap = magnitude - np.nextafter(magnitude, 0)
    half_gap /= 2
    np.abs(correction_error, out=correction_error)
    unsafe = (correction_error > 0) & (np.abs(residual) + correction_error >= half_gap)
    if halved:
        recoded *= 2
    for row, column in zip(*np.nonzero(unsafe), strict=True):
        recoded[row, column] = float(exact - Fraction(float(result[row, selected[column]])))
    result[:, selected] = recoded


def _assert_identical(actual: np.ndarray, expected: np.ndarray) -> None:
    """Require the same dtype, layout, and bits, treating every NaN as missing."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    if expected.dtype.kind == "f":
        missing = np.isnan(expected)
        np.testing.assert_array_equal(np.isnan(actual), missing)
        # Every other value matches bit for bit, signed zeros included.
        assert np.where(missing, 0, actual).tobytes() == np.where(missing, 0, expected).tobytes()
    else:
        assert actual.tobytes() == expected.tobytes()


_INTEGER_DTYPES = ["bool", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"]
_FLOAT_DTYPES = ["f2", "f4", "f8"]


def _integer_cases(dtype: np.dtype[Any], rng: np.random.Generator) -> Iterator[Any]:
    """Yield integer matrices with inferred, exact, wider, fractional, and negative bounds."""
    if dtype.kind == "b":
        low, high = 0, 1
    else:
        low, high = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
    shape = (int(rng.integers(1, 40)), int(rng.integers(1, 7)))
    for data_low, data_high, bounds in [
        (low, high, (None, None)),
        (low, high, (low, high)),
        (max(low, 1), min(high, 5), (1, 5)),
        (low, high, (low - 3, high + 2)),
        (low, high, (low - 0.5, high + 0.5)),
        (max(low, 1), min(high, 5), (0.1, 5.300000000000001)),
        (0, min(high, 2**53 + 3), (-1, min(high, 2**53 + 3))),
        (low, high, (low - 0.25, float(high))),
    ]:
        # Exact Python integers span even the full uint64 range.
        steps = [int(rng.integers(0, 2**62)) for _ in range(math.prod(shape))]
        values = [data_low + step * (data_high - data_low) // 2**62 for step in steps]
        data = np.array(values, dtype=object).reshape(shape)
        data.flat[0], data.flat[-1] = data_low, data_high
        yield data.astype(dtype), bounds


def _float_cases(dtype: np.dtype[Any], rng: np.random.Generator) -> Iterator[Any]:
    """Yield floating matrices with exact, residual, midpoint-sensitive, and extreme bounds."""
    largest = float(np.finfo(dtype).max)
    bounds_list: list[tuple[float, float]] = [
        (1.0, 5.0),
        (0.1, 0.30000000000000004),
        (-0.1, 0.30000000000000004),
        (-2.0, 2.0),
        (5.905306427913862e-18, 1.1655705045063483e17),
        (-2.789278090258446e-18, 1.943487081852115e16),
        (-largest, largest),
        (largest / 2, largest),
    ]
    if dtype == np.float64:
        bounds_list += [
            (-1e308, 1e308),
            (1e308, 1.7e308),
            (-1.7e308, -1e308),
            (-1.7976931348623157e308, 1e308),
        ]
    shape = (int(rng.integers(1, 40)), int(rng.integers(1, 7)))
    for low, high in bounds_list:
        data_low, data_high = max(low, -largest), min(high, largest)
        # Halves keep the span finite at the float64 limit.
        span = rng.random(shape) * (data_high / 2 - data_low / 2)
        values = 2 * (data_low / 2 + span)
        values.flat[0] = data_low
        data = values.astype(dtype)
        # Responses that round past a bound in the input precision become missing.
        data[(data.astype(float) < low) | (data.astype(float) > high)] = np.nan
        data[rng.random(shape) < 0.15] = np.nan
        if low <= 0.0 <= high:
            data.flat[-1] = -0.0
            data.flat[len(data.flat) // 2] = 0.0
        yield data, (low, high)
    yield np.full((3, 2), np.nan, dtype=dtype), (None, None)
    yield np.full((3, 2), np.nan, dtype=dtype), (1.0, 5.0)
    inferred = np.array([[1.5, np.nan, 2.25], [4.0, 2.0, -0.0]], dtype=dtype)
    yield inferred, (None, None)
    yield inferred, (None, 4.5)


def _differential_cases() -> Iterator[Any]:
    rng = np.random.default_rng(2024)
    for name in _INTEGER_DTYPES:
        for data, bounds in _integer_cases(np.dtype(name), rng):
            yield pytest.param(data, bounds, id=f"{name}-{bounds}")
    for name in _FLOAT_DTYPES:
        for data, bounds in _float_cases(np.dtype(name), rng):
            yield pytest.param(data, bounds, id=f"{name}-{bounds}")
    # Responses beyond the bounds fail in both implementations with one message.
    yield pytest.param(np.array([[0, 3], [2, 6]]), (1, 5), id="outside-int")
    yield pytest.param(np.array([[0.5, 3.0]], dtype="f4"), (1.0, 5.0), id="outside-float")
    yield pytest.param(np.array([[1.0, np.inf]]), (None, None), id="infinite")


@pytest.mark.parametrize("batch_elements", [1, 5, 64, 262_144])
@pytest.mark.parametrize(("data", "bounds"), list(_differential_cases()))
def test_batched_recoding_matches_whole_matrix_reference(
    data: np.ndarray, bounds: tuple[float | None, float | None], batch_elements: int
) -> None:
    rng = np.random.default_rng(data.size)
    for layout in (data, np.asfortranarray(data)):
        n_items = int(rng.integers(1, data.shape[1] + 1))
        items = [int(item) for item in rng.permutation(data.shape[1])[:n_items]]
        try:
            expected = _reference_reverse_score(layout, items, *bounds)
        except (ValueError, TypeError) as error:
            with (
                patch("ier._row_statistics._ROW_BATCH_ELEMENTS", batch_elements),
                pytest.raises(type(error), match=re.escape(str(error))),
            ):
                reverse_score(layout, items, *bounds)
            continue
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", batch_elements):
            actual = reverse_score(layout, items, *bounds)
        _assert_identical(actual, expected)
        assert not np.shares_memory(actual, layout)


@pytest.mark.parametrize("seed", range(40))
def test_random_float_reflections_match_the_reference_across_batches(seed: int) -> None:
    rng = np.random.default_rng(seed)
    low = float(rng.uniform(-1, 1) * 10.0 ** rng.integers(-20, 20))
    high = low + abs(float(rng.uniform(0, 1) * 10.0 ** rng.integers(-20, 20)))
    data = rng.uniform(low, high, size=(int(rng.integers(1, 60)), int(rng.integers(1, 9))))
    data = np.clip(data, low, high)
    data[rng.random(data.shape) < 0.1] = np.nan
    items = [int(item) for item in np.flatnonzero(rng.random(data.shape[1]) < 0.6)] or [0]

    expected = _reference_reverse_score(data, items, low, high)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", int(rng.integers(1, 50))):
        actual = reverse_score(data, items, low, high)

    _assert_identical(actual, expected)


def test_midpoint_repairs_land_in_their_own_batch_rows() -> None:
    # These reflections need exact rounding; batches of one row offset each repair.
    values = [3.260500540269927e16, 1.0, 2.5e16, 1.1655705045063483e17]
    data = np.array([values, values[::-1], values[1:] + values[:1]])
    bounds = (5.905306427913862e-18, 1.1655705045063483e17)

    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 1):
        actual = reverse_score(data, [0, 1, 2, 3], *bounds)

    endpoints = Fraction(bounds[0]) + Fraction(bounds[1])
    expected = [[float(endpoints - Fraction(value)) for value in row] for row in data.tolist()]
    assert actual.tolist() == expected
    _assert_identical(actual, _reference_reverse_score(data, [0, 1, 2, 3], *bounds))


def test_exact_endpoint_sums_skip_the_error_free_sums() -> None:
    data = np.array([[1.0, 2.5, np.nan], [5.0, -0.0, 3.25]])

    with patch("ier.keying._two_sum", side_effect=AssertionError("TwoSum is not needed")):
        actual = reverse_score(data, [0, 1, 2], scale_min=-1, scale_max=5)
        halved = reverse_score(np.array([[1e308, 1.5e308]]), [0, 1], 1e308, 1.5e308)

    _assert_identical(actual, _reference_reverse_score(data, [0, 1, 2], -1, 5))
    assert halved.tolist() == [[1.5e308, 1e308]]


def _traced_peak(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, int]:
    tracemalloc.start()
    try:
        result = function(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def _survey() -> np.ndarray:
    """Return 40,000 five-point responses to 60 items (19.2 MB as float64)."""
    return np.random.default_rng(5).integers(1, 6, size=(40_000, 60))


@pytest.mark.parametrize(
    ("kind", "bounds"),
    [
        ("float64", (1, 5)),
        ("residual", (0.1, 0.30000000000000004)),
        ("float32", (1, 5)),
        ("int64", (1, 5)),
        ("int64", (0.5, 5.5)),
        ("uint64", (-1, 2**60 + 5)),
        ("int64", (0, 2**63 - 1)),
    ],
    ids=["float-exact", "float-residual", "float32", "int", "int-fractional", "uint64", "object"],
)
def test_working_memory_beyond_the_returned_copy_is_bounded(
    kind: str, bounds: tuple[float, float]
) -> None:
    data = _survey()
    if kind == "residual":
        data = 0.1 + (data - 1) * 0.05
    elif kind == "uint64":
        data = data.astype(np.uint64) + np.uint64(2**60)
    else:
        data = data.astype(kind)

    result, peak = _traced_peak(reverse_score, data, list(range(60)), *bounds)

    # The whole-matrix version kept several full selected-column copies (over 70 MB).
    assert peak - result.nbytes < 8 * 2**20


def test_screen_peak_grows_by_one_recoded_matrix() -> None:
    data = _survey().astype(float)
    options = IndexOptions(scale_min=1, scale_max=5, evenodd_factors=[20, 20, 20])
    keyed = replace(options, reverse_keyed_items=list(range(0, 60, 3)))

    _, unset_peak = _traced_peak(screen, data, indices=["evenodd"], options=options)
    _, keyed_peak = _traced_peak(screen, data, indices=["evenodd"], options=keyed)

    assert keyed_peak - unset_peak < data.nbytes + 4 * 2**20
