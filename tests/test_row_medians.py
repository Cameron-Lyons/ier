"""Bounded medians preserve missing policies and extreme middle values."""

from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

from ier._row_statistics import row_median
from ier.response_time import response_time, response_time_flag, response_time_mixture


def _median_reference(row: np.ndarray, *, ignore_nan: bool) -> float:
    values = row.tolist()
    if not ignore_nan and any(value != value for value in values):
        return np.nan
    observed = sorted(value for value in values if value == value)
    if not observed:
        return np.nan
    middle = len(observed) // 2
    if len(observed) % 2:
        return float(observed[middle])
    lower, upper = observed[middle - 1 : middle + 1]
    if lower == -np.inf and upper == np.inf:
        return np.nan
    with localcontext() as context:
        context.prec = 800
        return float((Decimal(lower) + Decimal(upper)) / 2)


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("width", [1, 2, 7, 80, 600, 601])
def test_medians_match_independent_sorted_reference(
    ignore_nan: bool, layout: str, dtype: type, width: int
) -> None:
    rng = np.random.default_rng(721)
    if np.issubdtype(dtype, np.integer):
        data = rng.integers(0, 16, size=(19, width), dtype=dtype)
    else:
        data = rng.normal(size=(19, width)).astype(dtype)
        data[rng.random(data.shape) < 0.2] = np.nan
        data[0] = np.nan
        data[1] = 3
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = [_median_reference(row, ignore_nan=ignore_nan) for row in data]
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100):
        actual = row_median(data, ignore_nan=ignore_nan)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_extreme_floating_medians_remain_representable(ignore_nan: bool, layout: str) -> None:
    largest = np.finfo(float).max
    smallest = np.nextafter(0.0, 1.0)
    data = np.array(
        [
            [largest, largest, largest, largest],
            [-largest, -largest, -largest, -largest],
            [-largest, -largest / 2, largest / 2, largest],
            [smallest, smallest, smallest, smallest],
            [smallest, 2 * smallest, 3 * smallest, 4 * smallest],
            [largest, np.nan, 1e308, np.nan],
            [largest, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [-np.inf, np.inf, np.nan, np.nan],
            [-np.inf, -np.inf, 0, np.inf],
            [0, np.inf, np.inf, np.nan],
            [-0.0, -0.0, -0.0, -0.0],
        ]
    )
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = [_median_reference(row, ignore_nan=ignore_nan) for row in data]
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 17):
        actual = row_median(data, ignore_nan=ignore_nan)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_single_precision_midpoints_use_double_precision(ignore_nan: bool) -> None:
    largest = np.finfo(np.float32).max
    data = np.array([[1e8, 1e8 + 8], [largest, largest]], dtype=np.float32)
    expected = [_median_reference(row, ignore_nan=ignore_nan) for row in data]
    np.testing.assert_array_equal(row_median(data, ignore_nan=ignore_nan), expected)


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_middle_pairs_are_averaged_before_rounding(ignore_nan: bool, dtype: type) -> None:
    largest = np.iinfo(dtype).max
    values = [[largest - 1, largest], [2**53, 2**53 + 3], [2**53 - 1, 2**53]]
    if dtype == np.int64:
        values += [
            [np.iinfo(dtype).min, largest],
            [-(2**53) - 3, 2**53 + 1],
            [-(2**53) - 3, -(2**53)],
        ]
    data = np.array(values, dtype=dtype)
    data.flags.writeable = False
    expected = [_median_reference(row, ignore_nan=ignore_nan) for row in data]
    np.testing.assert_array_equal(row_median(data, ignore_nan=ignore_nan), expected)


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_internal_median_shapes(ignore_nan: bool, shape: tuple[int, int]) -> None:
    np.testing.assert_array_equal(
        row_median(np.empty(shape), ignore_nan=ignore_nan), np.full(shape[0], np.nan)
    )


def test_large_finite_medians_reach_timing_flags_and_mixture() -> None:
    times = np.array([[1e308, 1e308], [1.6e308, 1.6e308], [np.nan, np.nan]])
    np.testing.assert_array_equal(response_time(times), [1e308, 1.6e308, np.nan])
    np.testing.assert_array_equal(
        response_time_flag(times, threshold=1.3e308), [True, False, False]
    )
    probabilities = response_time_mixture(times, random_seed=42)
    assert np.isfinite(probabilities[:2]).all()
    assert np.all((probabilities[:2] >= 0) & (probabilities[:2] <= 1))
    assert probabilities[0] > probabilities[1]
    assert np.isnan(probabilities[2])
