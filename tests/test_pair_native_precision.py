"""Native paired arithmetic retains residuals before mean and ratio rounding."""

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ier import mad, semantic_syn
from ier._pair_statistics import paired_mean_absolute_difference

_MAX = float(np.finfo(float).max)
_MIN = float(np.nextafter(0.0, 1.0))


def _reference(
    row: np.ndarray, bounds: tuple[float, float] | None, divisor: float | None, ignore_nan: bool
) -> float:
    reflection = None if bounds is None else Fraction(bounds[0]) + Fraction(bounds[1])
    differences = []
    for left, right in zip(row[::2], row[1::2], strict=True):
        if np.isnan(left) or np.isnan(right):
            if ignore_nan:
                continue
            return math.nan
        a, b = Fraction(float(left)), Fraction(float(right))
        differences.append(abs(a - b if reflection is None else a + b - reflection))
    if not differences:
        return math.nan
    result = sum(differences) / len(differences)
    if divisor is not None:
        result /= Fraction(divisor)
    try:
        return float(result)
    except OverflowError:
        return math.inf


@pytest.mark.parametrize("large", [1e16, _MAX])
@pytest.mark.parametrize("small", [_MIN, 1e-300, 1.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_reflected_pairs_are_symmetric_and_preserve_small_residuals(
    large: float, small: float, sign: float
) -> None:
    data = np.array([[sign * large, sign * small], [sign * small, sign * large]])
    bounds = (-large, 0.0) if sign < 0 else (0.0, large)
    with np.errstate(all="raise"):
        actual = mad(data, item_pairs=[(0, 1)], scale_min=bounds[0], scale_max=bounds[1])
    np.testing.assert_array_equal(actual, [small, small])


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_native_pair_means_round_after_normalization(
    normalized: bool, ignore_nan: bool, strided: bool
) -> None:
    data = np.array([[3 * _MIN, 2 * _MIN, 0.0, 0.0], [_MIN, 0.0, np.nan, np.nan]])
    if strided:
        data = np.repeat(data, 2, axis=1)[:, ::2]
    original = data.copy()
    data.flags.writeable = False
    divisors = np.full(len(data), _MIN) if normalized else None
    with np.errstate(all="raise"):
        actual = paired_mean_absolute_difference(
            data,
            np.array([0, 2]),
            np.array([1, 3]),
            right_bounds=None,
            ignore_nan=ignore_nan,
            normalizers=divisors,
        )
    expected = [_reference(row, None, _MIN if normalized else None, ignore_nan) for row in data]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)


def test_semantic_synonyms_preserve_subnormal_normalized_mean() -> None:
    # In smallest-float units, the exact variance is 27/16 and pair mean is 1/2.
    observed = [Fraction(value) for value in (3, 2, 0, 0)]
    mean = sum(observed) / len(observed)
    variance = sum((value - mean) ** 2 for value in observed) / len(observed)
    pair_mean = (abs(observed[0] - observed[1]) + abs(observed[2] - observed[3])) / 2
    with localcontext() as context:
        context.prec = 80
        deviation = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        normalized = (Decimal(pair_mean.numerator) / Decimal(pair_mean.denominator)) / deviation
        expected = float(Decimal(1) - normalized)
    with np.errstate(all="raise"):
        actual = semantic_syn([[3 * _MIN, 2 * _MIN, 0.0, 0.0]], [(0, 1), (2, 3)])
    np.testing.assert_array_equal(actual, [expected])


@pytest.mark.parametrize("swap", [False, True])
def test_float_pairs_preserve_exact_integer_scale_endpoints(swap: bool) -> None:
    lower, upper = 2**63 + 1, 2**63 + 5
    data = np.array([[float(2**63), float(2**63)]])
    with np.errstate(all="raise"):
        actual = mad(
            data, item_pairs=[(1, 0) if swap else (0, 1)], scale_min=lower, scale_max=upper
        )
    np.testing.assert_array_equal(actual, [6.0])


@pytest.mark.parametrize("swap", [False, True])
def test_native_reflection_correction_and_bound_remainder_cancel_exactly(swap: bool) -> None:
    data = np.array([[1e16, -1.0], [1e16, 0.0]])
    with np.errstate(all="raise"):
        actual = mad(data, item_pairs=[(1, 0) if swap else (0, 1)], scale_min=-1.0, scale_max=1e16)
    np.testing.assert_array_equal(actual, [0.0, 1.0])


@pytest.mark.parametrize("divisor", [1e16, _MAX, _MIN])
def test_native_normalized_ratios_match_final_fraction_rounding(divisor: float) -> None:
    data = np.array([[1e-300, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    with np.errstate(all="raise"):
        actual = paired_mean_absolute_difference(
            data,
            np.array([0, 2]),
            np.array([1, 3]),
            right_bounds=None,
            ignore_nan=True,
            normalizers=np.full(len(data), divisor),
        )
    expected = [_reference(row, None, divisor, True) for row in data]
    np.testing.assert_array_equal(actual, expected)


@settings(max_examples=400, deadline=None, derandomize=True)
@given(
    values=st.tuples(*[st.floats(allow_nan=False, allow_infinity=False)] * 4),
    bounds=st.one_of(st.none(), st.tuples(*[st.floats(allow_nan=False, allow_infinity=False)] * 2)),
    divisor=st.one_of(st.none(), st.floats(min_value=_MIN, allow_nan=False, allow_infinity=False)),
)
def test_native_pairs_match_fraction(
    values: tuple[float, float, float, float],
    bounds: tuple[float, float] | None,
    divisor: float | None,
) -> None:
    data = np.array([values])
    ordered = None if bounds is None else tuple(sorted(bounds))
    expected = _reference(data[0], ordered, divisor, True)
    with np.errstate(all="raise"):
        actual = paired_mean_absolute_difference(
            data,
            np.array([0, 2]),
            np.array([1, 3]),
            right_bounds=ordered,
            ignore_nan=True,
            normalizers=None if divisor is None else np.array([divisor]),
        )[0]
    assert actual == expected or math.isclose(actual, expected, rel_tol=5e-14, abs_tol=0)
