"""Reverse scoring recodes reverse-keyed items exactly and leaves the input untouched."""

import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import ier
from ier import evenodd, reverse_score


def _exact_reflection(values: np.ndarray, scale_min: float, scale_max: float) -> np.ndarray:
    """Round each exact reflection once."""
    endpoints = Fraction(scale_min) + Fraction(scale_max)
    return np.array(
        [
            [math.nan if math.isnan(value) else float(endpoints - Fraction(value)) for value in row]
            for row in values.tolist()
        ]
    )


def test_integer_items_are_reflected_exactly_without_mutating_input() -> None:
    data = np.array([[1, 2, 5, 4], [5, 5, 1, 3], [3, 1, 2, 2]])
    data.flags.writeable = False
    original = data.copy()

    result = reverse_score(data, [2, 0], scale_min=1, scale_max=5)

    np.testing.assert_array_equal(result, [[5, 2, 1, 4], [1, 5, 5, 3], [3, 1, 4, 2]])
    assert result.dtype == np.int64
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(reverse_score(result, [0, 2], 1, 5), original)


def test_reverse_score_is_exported() -> None:
    assert "reverse_score" in ier.__all__
    assert ier.reverse_score is reverse_score


@pytest.mark.parametrize(
    ("dtype", "values", "bounds", "expected_dtype"),
    [
        (np.uint8, [1, 5], (1, 5), np.uint8),
        (np.uint8, [0, 255], (0, 300), np.uint16),
        (np.uint8, [0, 250], (-2, 250), np.int16),
        (np.int8, [-128, 127], (-128, 127), np.int8),
        (np.int8, [1, 5], (1, 5), np.int8),
        (np.int8, [-3, 100], (-3, 200), np.int16),
        (np.int16, [1, 5], (1, 5), np.int16),
        (np.uint16, [0, 5], (-1, 5), np.int32),
        (np.int32, [1, 7], (1, 7), np.int32),
        (np.int8, [0, 7], (0, 2**40), np.int64),
        (np.bool_, [False, True], (0, 1), np.uint8),
        (np.bool_, [False, True], (-1, 1), np.int8),
    ],
)
def test_integer_dtypes_hold_both_bounds(
    dtype: type, values: list[int], bounds: tuple[int, int], expected_dtype: type
) -> None:
    data = np.array([values, values[::-1]], dtype=dtype)

    result = reverse_score(data, [0], *bounds)

    assert result.dtype == expected_dtype
    reflected = [sum(bounds) - int(value) for value in data[:, 0].tolist()]
    assert result[:, 0].tolist() == reflected
    assert result[:, 1].tolist() == data[:, 1].astype(int).tolist()


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (np.int64, [np.iinfo(np.int64).min, -1, 0, np.iinfo(np.int64).max]),
        (np.uint64, [0, 1, 2**63, 2**64 - 1]),
    ],
)
def test_full_width_integers_are_reflected_exactly(dtype: type, values: list[int]) -> None:
    data = np.array([values], dtype=dtype)

    result = reverse_score(data, [0, 1, 2, 3])

    assert result.dtype == dtype
    lower, upper = min(values), max(values)
    assert result.tolist() == [[lower + upper - value for value in values]]


def test_unsigned_responses_with_negative_bound_fall_back_to_float() -> None:
    data = np.array([[2**53, 0]], dtype=np.uint64)

    result = reverse_score(data, [1], scale_min=-1, scale_max=2**53)

    assert result.dtype == np.float64
    assert result.tolist() == [[2.0**53, 2.0**53 - 1]]


@pytest.mark.parametrize(
    ("dtype", "values", "bounds"),
    [
        (np.uint64, [5, 2**53 + 1, 7], (-1, 2**53 + 3)),
        (np.uint64, [2**64 - 1, 2**53 + 1, 0], (-1, 2**64 - 1)),
        (np.int64, [5, 2**53 + 1, -7], (-7.5, 2.0**53 + 2)),
        (np.int64, [-(2**53) - 1, 0, -(2**63)], (-(2.0**63), 0.5)),
        (np.int64, [2**63 - 1, 2**62 + 1, 1], (0, 2**63 + 5)),
    ],
)
def test_integer_reflections_without_an_integer_dtype_are_rounded_once(
    dtype: type, values: list[int], bounds: tuple[float, float]
) -> None:
    data = np.array([values, values[::-1]], dtype=dtype)

    result = reverse_score(data, [0, 1], *bounds)

    endpoints = Fraction(bounds[0]) + Fraction(bounds[1])
    expected = [
        [float(endpoints - value) for value in row[:2]] + [float(row[2])] for row in data.tolist()
    ]
    assert result.dtype == np.float64
    assert result.tolist() == expected


def test_non_integral_bounds_reverse_integer_responses_as_floats() -> None:
    for dtype in (np.int64, np.int16, np.bool_):
        data = np.array([[1, 5], [2, 3]]).astype(dtype)

        result = reverse_score(data, [1], scale_min=0.5, scale_max=5.5)

        assert result.dtype == np.float64
        expected = data.astype(float)
        expected[:, 1] = 6.0 - expected[:, 1]
        np.testing.assert_array_equal(result, expected)


def test_float_items_round_each_exact_reflection_once() -> None:
    rng = np.random.default_rng(3)
    data = rng.uniform(0.1, 0.30000000000000004, size=(40, 6))
    data[rng.random(data.shape) < 0.1] = np.nan
    data[0, :3] = [0.1, 0.2, 0.30000000000000004]

    result = reverse_score(data, [0, 2, 5], scale_min=0.1, scale_max=0.30000000000000004)

    expected = _exact_reflection(data[:, [0, 2, 5]], 0.1, 0.30000000000000004)
    np.testing.assert_array_equal(result[:, [0, 2, 5]], expected)
    np.testing.assert_array_equal(result[:, [1, 3, 4]], data[:, [1, 3, 4]])


@pytest.mark.parametrize(
    ("scale_min", "scale_max", "values"),
    [
        # Cancellation leaves only the endpoint residual.
        (-0.1, 0.30000000000000004, [0.2, 0.1, 0.25, 0.0, 0.05, -0.1]),
        # A rounded correction near a rounding midpoint is recomputed exactly.
        (5.905306427913862e-18, 1.1655705045063483e17, [3.260500540269927e16, 1.0, 2.5e16]),
        (-2.789278090258446e-18, 1.943487081852115e16, [7144510713290445.0, 5859577342029433.0]),
    ],
)
def test_float_reflections_are_correctly_rounded(
    scale_min: float, scale_max: float, values: list[float]
) -> None:
    data = np.array([values])

    result = reverse_score(data, list(range(len(values))), scale_min, scale_max)

    np.testing.assert_array_equal(result, _exact_reflection(data, scale_min, scale_max))


def test_reflected_zero_stays_positive() -> None:
    result = reverse_score(np.array([[5.0, 2.5, 0.0]]), [0, 1, 2], scale_min=0, scale_max=5)

    np.testing.assert_array_equal(result, [[0.0, 2.5, 5.0]])
    assert not np.signbit(result).any()


@pytest.mark.parametrize(
    "bounds",
    [(-1e308, 1e308), (1e308, 1.7e308), (-1.7e308, -1e308), (-1.7976931348623157e308, 1e308)],
)
def test_extreme_bounds_stay_finite(bounds: tuple[float, float]) -> None:
    scale_min, scale_max = bounds
    data = np.array([[scale_min, scale_max, (scale_min / 2) + (scale_max / 2)]])

    result = reverse_score(data, [0, 1, 2], scale_min=scale_min, scale_max=scale_max)

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0, :2], [scale_max, scale_min], rtol=1e-15)
    np.testing.assert_allclose(result[0, 2], data[0, 2], rtol=1e-15)


def test_missing_bounds_are_inferred_from_the_whole_matrix() -> None:
    data = np.array([[2.0, 1.0, np.nan], [3.0, 7.0, 4.0]])

    result = reverse_score(data, [0])

    np.testing.assert_array_equal(result[:, 0], [6.0, 5.0])
    np.testing.assert_array_equal(reverse_score(data, [0], scale_max=4.0)[:, 0], [3.0, 2.0])
    np.testing.assert_array_equal(reverse_score(data, [2], scale_min=0.0)[:, 2], [np.nan, 3.0])


def test_missing_responses_are_preserved() -> None:
    data = np.array([[1.0, np.nan], [np.nan, np.nan]])

    result = reverse_score(data, [0, 1], scale_min=1, scale_max=5)

    np.testing.assert_array_equal(result, [[5.0, np.nan], [np.nan, np.nan]])
    all_missing = reverse_score(np.full((2, 2), np.nan), [1])
    assert all_missing.dtype == np.float64
    assert np.isnan(all_missing).all()
    assert np.isnan(reverse_score(np.full((2, 2), np.nan), [0], 1, 5)).all()


@pytest.mark.parametrize(
    ("data", "bounds"),
    [
        ([[0, 3]], (1, 5)),
        ([[6, 3]], (1, 5)),
        ([[0.5, 3.0]], (1, 5)),
        ([[np.inf, 3.0]], (1, 5)),
        ([[2**63 - 1, 3]], (1, 2**62)),
    ],
)
def test_responses_outside_the_bounds_are_rejected(
    data: list[list[float]], bounds: tuple[float, float]
) -> None:
    with pytest.raises(ValueError, match="within scale_min and scale_max"):
        reverse_score(np.array(data), [0], *bounds)


@pytest.mark.parametrize(
    ("data", "bounds"),
    [
        (np.array([[1.0, 3.0]], dtype=np.float32), (1.00000001, 5.0)),
        (np.array([[4.6, 3.0]], dtype=np.float16), (1.0, 4.6)),
        (np.array([[5.0, 3.0]], dtype=np.float32), (1.0, 4.9999999)),
    ],
)
def test_low_precision_responses_are_checked_against_exact_bounds(
    data: np.ndarray, bounds: tuple[float, float]
) -> None:
    # The bound rounds to the stored response in the input's precision.
    assert data.dtype.type(bounds[0]) <= data[0, 0] <= data.dtype.type(bounds[1])
    with pytest.raises(ValueError, match="within scale_min and scale_max"):
        reverse_score(data, [0], *bounds)


@pytest.mark.parametrize(
    ("dtype", "bounds"),
    [(np.float16, (1.0, 70000.0)), (np.float32, (-1e39, 5.0)), (np.float32, (-1.0, 1e300))],
)
def test_bounds_beyond_the_input_range_are_compared_exactly(
    dtype: type, bounds: tuple[float, float]
) -> None:
    data = np.array([[1.0, 3.0], [2.5, np.nan]], dtype=dtype)

    result = reverse_score(data, [0], *bounds)

    expected = _exact_reflection(data[:, :1].astype(float), *bounds)
    np.testing.assert_array_equal(result[:, :1], expected)
    with pytest.raises(ValueError, match="within scale_min and scale_max"):
        reverse_score(data, [0], 1.5, bounds[1])


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
    reason="long double is no wider than float64",
)
def test_extended_precision_responses_are_checked_exactly() -> None:
    data = np.array([[1.0, 3.0]], dtype=np.longdouble)
    data[0, 0] += np.longdouble(2) ** -60

    with pytest.raises(ValueError, match="within scale_min and scale_max"):
        reverse_score(data, [0], 0.0, 1.0)
    reverse_score(data, [0], 0.0, 1.0000000000000002)
    # Inferred extended-precision endpoints hold the extended response exactly.
    assert reverse_score(data, [0, 1]).dtype == np.float64


def test_infinite_responses_cannot_supply_inferred_bounds() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        reverse_score(np.array([[1.0, np.inf]]), [0])


@pytest.mark.parametrize(
    ("scale_min", "scale_max", "message"),
    [
        (True, 5, "scale_min must be a finite real number"),
        (1, "5", "scale_max must be a finite real number"),
        (np.nan, 5, "scale_min must be a finite real number"),
        (1, np.inf, "scale_max must be a finite real number"),
        (-(10**400), 5, "scale_min must be a finite real number"),
        (5, 1, "scale_max must be greater than or equal to scale_min"),
    ],
)
def test_invalid_bounds_raise_value_error(scale_min: Any, scale_max: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        reverse_score([[1, 2], [3, 4]], [0], scale_min=scale_min, scale_max=scale_max)


def test_numpy_and_fraction_bounds_are_accepted() -> None:
    data = [[1, 2], [3, 4]]

    np.testing.assert_array_equal(
        reverse_score(data, [0], np.int64(1), np.float64(4)), [[4, 2], [2, 4]]
    )
    np.testing.assert_array_equal(
        reverse_score(data, [0], Fraction(1, 2), Fraction(9, 2)), [[4.0, 2.0], [2.0, 4.0]]
    )


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "items cannot be empty"),
        ([0, 0], "items cannot contain duplicates"),
        ([2], "out of bounds"),
        ([-1], "out of bounds"),
        ([1.0], "items must contain integer column indices"),
    ],
)
def test_invalid_items_raise_value_error(items: list[Any], message: str) -> None:
    with pytest.raises(ValueError, match=message) as caught:
        reverse_score([[1, 2], [3, 4]], items)
    assert "item_indices" not in str(caught.value)


def test_reverse_scored_items_restore_even_odd_consistency() -> None:
    rng = np.random.default_rng(10)
    traits = rng.normal(size=(200, 4))
    latent = np.repeat(traits, 6, axis=1) + rng.normal(scale=0.7, size=(200, 24))
    keyed = np.clip(np.rint(3 + 1.2 * latent), 1, 5).astype(np.int64)
    reverse_items = [column for column in range(24) if column % 6 in (1, 4)]
    presented = reverse_score(keyed, reverse_items, scale_min=1, scale_max=5)

    recoded = reverse_score(presented, reverse_items, scale_min=1, scale_max=5)

    np.testing.assert_array_equal(recoded, keyed)
    raw = evenodd(presented, [6] * 4, method="halves")
    rescored = evenodd(recoded, [6] * 4, method="halves")
    assert np.nanmean(rescored) > np.nanmean(raw) + 0.3
