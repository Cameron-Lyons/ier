"""Overflow repairs preserve finite paired residuals and normalized differences."""

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ier import mad, semantic_ant
from ier._pair_statistics import _rescaled_pair_difference

_MAX = float(np.finfo(float).max)
_MIN = float(np.nextafter(0.0, 1.0))


def _reference(
    row: np.ndarray, bounds: tuple[float, float], ignore_nan: bool, divisor: float | None
) -> float:
    reflection = Fraction(bounds[0]) + Fraction(bounds[1])
    differences = []
    for left, right in zip(row[::2], row[1::2], strict=True):
        if np.isnan(left) or np.isnan(right):
            if ignore_nan:
                continue
            return math.nan
        differences.append(abs(Fraction(float(left)) + Fraction(float(right)) - reflection))
    if not differences:
        return math.nan
    result = sum(differences) / len(differences)
    if divisor is not None:
        result /= Fraction(divisor)
    try:
        return float(result)
    except OverflowError:
        return math.inf


@pytest.mark.parametrize("value", [1.0, 1e-300, _MIN])
@pytest.mark.parametrize("strided", [False, True])
def test_overflowing_neighbors_preserve_small_mad_and_antonym_scores(
    value: float, strided: bool
) -> None:
    data = np.array([[_MAX, _MAX], [value, 2 * value], [-_MAX, -_MAX], [0.0, 0.0]])
    if strided:
        data = np.repeat(data, 2, axis=0)[::2]
    original = data.copy()
    data.flags.writeable = False
    with np.errstate(all="raise"):
        scores = mad(data, item_pairs=[(0, 1)], scale_min=-_MAX, scale_max=_MAX)
        antonyms = semantic_ant(data, [(0, 1)], scale_min=-_MAX, scale_max=_MAX)
        isolated = semantic_ant([[value, 2 * value]], [(0, 1)], scale_min=-_MAX, scale_max=_MAX)
    expected = float(Fraction(value) + Fraction(2 * value))
    np.testing.assert_array_equal(scores, [np.inf, expected, np.inf, 0.0])
    np.testing.assert_array_equal(antonyms[1:2], isolated)
    if value != _MIN:
        assert antonyms[1] == -1
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("swap", [False, True])
def test_overflowing_reflection_preserves_near_endpoint_differences(
    sign: float, swap: bool
) -> None:
    small = 1e292
    values = [sign * _MAX, sign * np.nextafter(small, np.inf)]
    bounds = (-_MAX, -small) if sign < 0 else (small, _MAX)
    if swap:
        values.reverse()
    expected = _reference(np.array(values), bounds, True, None)
    with np.errstate(all="raise"):
        actual = mad([values], item_pairs=[(0, 1)], scale_min=bounds[0], scale_max=bounds[1])
    np.testing.assert_array_equal(actual, [expected])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_reflection_and_distance_retain_finite_semantic_ratio(sign: float) -> None:
    bounds = (-_MAX, -_MAX / 2) if sign < 0 else (_MAX / 2, _MAX)
    data = np.array([[-_MAX, _MAX]])
    with np.errstate(all="raise"):
        raw = mad(data, item_pairs=[(0, 1)], scale_min=bounds[0], scale_max=bounds[1])
        normalized = semantic_ant(data, [(0, 1)], scale_min=bounds[0], scale_max=bounds[1])
    np.testing.assert_array_equal(raw, [np.inf])
    np.testing.assert_array_equal(normalized, [-0.5])


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_rescaled_pairs_match_exact_totals_with_missing_and_subnormal_values(
    ignore_nan: bool, normalized: bool
) -> None:
    data = np.array(
        [
            [_MAX, _MAX, 1.0, -1.0],
            [1.0, 2.0, np.nan, 1.0],
            [np.nan, np.nan, np.nan, np.nan],
            [_MAX, 0.0, 0.0, 0.0],
            [_MIN, 0.0, _MIN, 0.0],
        ]
    )
    divisors = np.array([1.0, 0.5, 1.0, 2.0, 0.5]) if normalized else None
    bounds = (-_MAX, _MAX)
    expected = [
        _reference(row, bounds, ignore_nan, None if divisors is None else float(divisors[i]))
        for i, row in enumerate(data)
    ]
    actual = _rescaled_pair_difference(
        data, np.array([0, 2]), np.array([1, 3]), bounds, ignore_nan, divisors
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=0)


@pytest.mark.parametrize("small", [_MIN, 1e-300, 1.0])
@pytest.mark.parametrize("normalized", [False, True])
def test_rescaled_reflection_retains_rounded_away_bound(small: float, normalized: bool) -> None:
    data = np.array([[_MAX, 0.0]])
    divisors = np.array([small]) if normalized else None
    actual = _rescaled_pair_difference(
        data, np.array([0]), np.array([1]), (small, _MAX), True, divisors
    )
    np.testing.assert_array_equal(actual, [1.0 if normalized else small])


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    values=st.tuples(*[st.floats(allow_nan=False, allow_infinity=False)] * 2),
    bounds=st.tuples(*[st.floats(allow_nan=False, allow_infinity=False)] * 2),
    divisor=st.one_of(st.none(), st.floats(min_value=_MIN, allow_nan=False, allow_infinity=False)),
)
def test_rescaled_finite_pairs_match_fraction(
    values: tuple[float, float], bounds: tuple[float, float], divisor: float | None
) -> None:
    data = np.array([values])
    ordered = tuple(sorted(bounds))
    expected = _reference(data[0], ordered, True, divisor)
    actual = _rescaled_pair_difference(
        data,
        np.array([0]),
        np.array([1]),
        ordered,
        True,
        None if divisor is None else np.array([divisor]),
    )[0]
    assert actual == expected or math.isclose(actual, expected, rel_tol=5e-14, abs_tol=0)
