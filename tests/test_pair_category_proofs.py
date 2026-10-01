"""Compact category checks preserve paired arithmetic at cast and float limits."""

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ier import mad
from ier._pair_statistics import paired_mean_absolute_difference

_MIN = float(np.nextafter(0.0, 1.0))
_MAX = float(np.finfo(float).max)
_VALUES = [
    -float(2**51),
    -float(2**40),
    float(np.nextafter(-float(2**31), -np.inf)),
    -float(2**31),
    float(np.nextafter(-float(2**31), np.inf)),
    float(2**31 - 1),
    float(np.nextafter(float(2**31), -np.inf)),
    float(2**31),
    float(np.nextafter(float(2**31), np.inf)),
    float(2**40),
    float(2**51),
    float(np.nextafter(float(2**51), np.inf)),
    -0.5,
    0.5,
    -_MIN,
    _MIN,
    -_MAX,
    _MAX,
]


def _reference(row: np.ndarray, ignore_nan: bool, divisor: float | None = None) -> float:
    differences = []
    for left, right in zip(row[::2], row[1::2], strict=True):
        if np.isnan(left) or np.isnan(right):
            if ignore_nan:
                continue
            return math.nan
        differences.append(abs(Fraction(float(left)) + Fraction(float(right)) - 6))
    if not differences:
        return math.nan
    total = sum(differences) / len(differences)
    if divisor is not None:
        total /= Fraction(divisor)
    try:
        return float(total)
    except OverflowError:
        return math.inf


@pytest.mark.parametrize("right", _VALUES)
@pytest.mark.parametrize("ignore_nan", [False, True])
def test_category_probe_preserves_fractional_wide_and_extreme_responses(
    right: float, ignore_nan: bool
) -> None:
    data = np.array([[6.0, right, 1.0, 5.0], [-right, right, np.nan, 1.0]])
    data.flags.writeable = False
    original = data.copy()
    expected = [_reference(row, ignore_nan) for row in data]
    with np.errstate(all="raise"):
        actual = mad(data, item_pairs=[(0, 1), (2, 3)], scale_min=1, scale_max=5, na_rm=ignore_nan)
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=0)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_category_probe_keeps_missing_policies_and_input_ownership(dtype: type, order: str) -> None:
    data = np.array([[1, 5, 4, 2], [3, 2, np.nan, 4], [np.nan] * 4], dtype=dtype)
    if order == "strided":
        data = np.repeat(data, 2, axis=1)[:, ::2]
    else:
        data = np.asarray(data, order=order)
    original = data.copy()
    data.flags.writeable = False
    for ignore_nan in [False, True]:
        with np.errstate(all="raise"):
            actual = mad(
                data, item_pairs=[(0, 1), (2, 3)], scale_min=1, scale_max=5, na_rm=ignore_nan
            )
        expected = [_reference(row, ignore_nan) for row in data]
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_category_probe_handles_an_entire_missing_reverse_buffer(ignore_nan: bool) -> None:
    data = np.array([[1.0, np.nan, 2.0, np.nan], [3.0, np.nan, 4.0, np.nan]])
    with np.errstate(all="raise"):
        actual = mad(data, item_pairs=[(0, 1), (2, 3)], scale_min=1, scale_max=5, na_rm=ignore_nan)
    np.testing.assert_array_equal(actual, [np.nan, np.nan])


@settings(max_examples=300, deadline=None, derandomize=True)
@given(
    values=st.tuples(*[st.sampled_from([*_VALUES, 0.0, 1.0, 5.0, np.nan])] * 4),
    ignore_nan=st.booleans(),
    divisor=st.sampled_from([None, _MIN, 1e-300, 1.0, 1e300, _MAX]),
)
def test_category_scale_pairs_match_fraction_before_final_rounding(
    values: tuple[float, float, float, float], ignore_nan: bool, divisor: float | None
) -> None:
    data = np.array([values])
    expected = _reference(data[0], ignore_nan, divisor)
    with np.errstate(all="raise"):
        actual = paired_mean_absolute_difference(
            data,
            np.array([0, 2]),
            np.array([1, 3]),
            right_bounds=(1.0, 5.0),
            ignore_nan=ignore_nan,
            normalizers=None if divisor is None else np.array([divisor]),
        )[0]
    assert (
        actual == expected
        or (math.isnan(actual) and math.isnan(expected))
        or math.isclose(actual, expected, rel_tol=5e-14, abs_tol=0)
    )
