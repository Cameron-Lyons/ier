"""Tests for bounded row-wise statistical reductions."""

from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

import ier._row_statistics as row_statistics
from ier import irv, response_pattern, response_time, response_time_consistency


def _decimal_moments(row: np.ndarray, *, ignore_nan: bool) -> tuple[float, float]:
    """Independent population moments without binary overflow or underflow."""
    if not ignore_nan and np.isnan(row).any():
        return np.nan, np.nan
    observed = [
        Decimal(int(value)) if row.dtype.kind in "iu" else Decimal.from_float(float(value))
        for value in row
        if not np.isnan(value)
    ]
    if not observed:
        return np.nan, np.nan
    with localcontext() as context:
        context.prec = 800
        mean = sum(observed) / len(observed)
        variance = sum((value - mean) ** 2 for value in observed) / len(observed)
        return float(mean), float(variance.sqrt())


def test_missing_aware_reductions_match_numpy() -> None:
    """Bounded missing-aware reductions preserve NumPy's row results."""
    rng = np.random.default_rng(20260803)
    data = rng.normal(size=(513, 31))
    data[rng.random(data.shape) < 0.15] = np.nan
    original = data.copy()

    expected_mean = np.nanmean(data, axis=1)
    expected_median = np.nanmedian(data, axis=1)
    expected_std = np.nanstd(data, axis=1)
    means, deviations = row_statistics.row_mean_std(data, ignore_nan=True)

    np.testing.assert_allclose(means, expected_mean, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(deviations, expected_std, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        row_statistics.row_mean(data, ignore_nan=True),
        expected_mean,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        row_statistics.row_median(data, ignore_nan=True),
        expected_median,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        row_statistics.row_std(data, ignore_nan=True),
        expected_std,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(data, original)


def test_complete_reductions_match_numpy() -> None:
    """Strict bounded reductions preserve complete-data results."""
    data = np.arange(77, dtype=float).reshape(11, 7)

    means, deviations = row_statistics.row_mean_std(data, ignore_nan=False)

    np.testing.assert_array_equal(means, np.mean(data, axis=1))
    np.testing.assert_array_equal(deviations, np.std(data, axis=1))
    np.testing.assert_array_equal(
        row_statistics.row_mean(data, ignore_nan=False),
        np.mean(data, axis=1),
    )
    np.testing.assert_array_equal(
        row_statistics.row_median(data, ignore_nan=False),
        np.median(data, axis=1),
    )
    np.testing.assert_array_equal(
        row_statistics.row_std(data, ignore_nan=False),
        np.std(data, axis=1),
    )


def test_strict_reductions_propagate_missing_values() -> None:
    """Strict reductions leave every contaminated row unavailable."""
    data = np.array([[1.0, 2.0, 3.0], [1.0, np.nan, 3.0]])

    means, deviations = row_statistics.row_mean_std(data, ignore_nan=False)
    medians = row_statistics.row_median(data, ignore_nan=False)

    np.testing.assert_allclose(means[0], 2.0)
    np.testing.assert_allclose(deviations[0], np.std(data[0]))
    assert np.isnan(means[1])
    assert np.isnan(deviations[1])
    assert np.isnan(medians[1])


def test_all_missing_rows_are_unavailable_without_warning() -> None:
    """Missing-aware reductions return NaN for rows with no observations."""
    data = np.array([[np.nan, np.nan, np.nan], [1.0, np.nan, 5.0]])

    means, deviations = row_statistics.row_mean_std(data, ignore_nan=True)
    medians = row_statistics.row_median(data, ignore_nan=True)

    assert np.isnan(means[0])
    assert np.isnan(deviations[0])
    assert np.isnan(medians[0])
    np.testing.assert_allclose(means[1], 3.0)
    np.testing.assert_allclose(deviations[1], 2.0)
    np.testing.assert_allclose(medians[1], 3.0)


def test_reductions_obey_shared_element_budget() -> None:
    """Mean and standard-deviation work is split into bounded row blocks."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(17, 7))
    data[2, 3] = np.nan

    with (
        patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 20),
        patch.object(
            row_statistics,
            "_row_mean_block",
            wraps=row_statistics._row_mean_block,
        ) as mean_blocks,
        patch.object(
            row_statistics,
            "_row_mean_std_block",
            wraps=row_statistics._row_mean_std_block,
        ) as mean_std_blocks,
        patch.object(
            row_statistics,
            "_row_median_block",
            wraps=row_statistics._row_median_block,
        ) as median_blocks,
    ):
        means = row_statistics.row_mean(data, ignore_nan=True)
        medians = row_statistics.row_median(data, ignore_nan=True)
        combined_means, deviations = row_statistics.row_mean_std(data, ignore_nan=True)

    assert mean_blocks.call_count > 1
    assert mean_std_blocks.call_count > 1
    assert median_blocks.call_count > 1
    assert all(call.args[0].size <= 20 for call in mean_blocks.call_args_list)
    assert all(call.args[0].size <= 20 for call in mean_std_blocks.call_args_list)
    assert all(call.args[0].size <= 20 for call in median_blocks.call_args_list)
    np.testing.assert_allclose(means, np.nanmean(data, axis=1), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(medians, np.nanmedian(data, axis=1), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(combined_means, means, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        deviations,
        np.nanstd(data, axis=1),
        rtol=0.0,
        atol=1e-15,
    )


def test_wide_rows_still_make_progress() -> None:
    """A row wider than the budget is emitted as a one-row block."""
    with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 4):
        assert list(row_statistics.row_slices(3, 10)) == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_row_totals_match_numpy(ignore_nan: bool, layout: str) -> None:
    rng = np.random.default_rng(413)
    data = rng.normal(size=(17, 31))
    data[rng.random(data.shape) < 0.2] = np.nan
    data[0] = np.nan
    data = data[::-2, ::2] if layout == "strided" else np.array(data, order=layout)
    data.flags.writeable = False
    expected = np.nansum(data, axis=1) if ignore_nan else np.sum(data, axis=1)
    with (
        patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 20),
        patch.object(row_statistics.np, "isnan", wraps=np.isnan) as masks,
    ):
        actual = row_statistics.row_sum(data, ignore_nan=ignore_nan)
    assert all(call.args[0].size <= max(20, data.shape[1]) for call in masks.call_args_list)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_integer_row_totals_do_not_overflow() -> None:
    data = np.full((3, 7), np.iinfo(np.int64).max)
    np.testing.assert_array_equal(
        row_statistics.row_sum(data, ignore_nan=True), np.sum(data, axis=1, dtype=float)
    )


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_moments_preserve_exact_means_and_small_variation(
    ignore_nan: bool, layout: str, dtype: type
) -> None:
    largest = int(np.iinfo(dtype).max)
    smallest = int(np.iinfo(dtype).min)
    rows = [
        [2**60 + value for value in [0, 1, 2, 3, 4, 5]],
        [largest - value for value in [1, 0, 0, 3, 4, 5]],
        [smallest + value for value in [0, 1, 2, 3, 4, 5]],
        [largest, smallest, 2, largest, smallest, 2],
        [largest] * 6,
        [1, 2, 3, 1, 2, 3],
    ]
    data = np.array(rows, dtype=dtype)
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected_mean, expected_std = np.array(
        [_decimal_moments(row, ignore_nan=ignore_nan) for row in data]
    ).T
    expected_split = np.array(
        [
            sum(_decimal_moments(section, ignore_nan=ignore_nan)[1] for section in np.split(row, 2))
            / 2
            for row in data
        ]
    )
    with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 25):
        means, deviations = row_statistics.row_mean_std(data, ignore_nan=ignore_nan)
        separate_means = row_statistics.row_mean(data, ignore_nan=ignore_nan)
        separate_deviations = row_statistics.row_std(data, ignore_nan=ignore_nan)
        combined = response_pattern(data)
        split_deviations = irv(data, split=True, num_split=2, na_rm=ignore_nan)
    for actual in (means, separate_means, combined["acquiescence"]):
        np.testing.assert_array_equal(actual, expected_mean)
    for actual in (deviations, separate_deviations, combined["variability"]):
        np.testing.assert_allclose(actual, expected_std, rtol=3e-15, atol=0)
    np.testing.assert_allclose(split_deviations, expected_split, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("width", [1, 3, 19, 601])
def test_integer_means_match_python_totals_across_the_dtype_range(dtype: type, width: int) -> None:
    data = np.random.default_rng(451).bit_generator.random_raw((17, width)).view(dtype)
    expected = np.array([sum(int(value) for value in row) / width for row in data])
    np.testing.assert_array_equal(row_statistics.row_mean(data, ignore_nan=True), expected)


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_integer_means_keep_small_terms_during_large_cancellation(ignore_nan: bool) -> None:
    limit = 2**53
    data = np.array([[limit, 1, -limit], [-limit, 1, limit], [limit, limit - 1, limit - 2]])
    expected = np.array([sum(int(value) for value in row) / len(row) for row in data])
    np.testing.assert_array_equal(row_statistics.row_mean(data, ignore_nan=ignore_nan), expected)
    np.testing.assert_array_equal(
        row_statistics.row_mean_std(data, ignore_nan=ignore_nan)[0], expected
    )


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_extreme_finite_moments_match_decimal(ignore_nan: bool, missing: bool, layout: str) -> None:
    largest = np.finfo(float).max
    smallest = np.nextafter(0.0, 1.0)
    data = np.array(
        [
            [1e308, 1e308, 1e308, 1e308],
            [largest, largest, largest, largest],
            [largest, -largest, largest, -largest],
            [largest, largest / 2, largest / 4, -largest / 2],
            [1e200, -1e200, 2e200, -2e200],
            [1e-200, -1e-200, 2e-200, -2e-200],
            [1e-160, -1e-160, 2e-160, -2e-160],
            [smallest, 2 * smallest, 3 * smallest, 4 * smallest],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    if missing:
        data[::2, 1] = np.nan
        data[-1] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected_mean, expected_std = np.array(
        [_decimal_moments(row, ignore_nan=ignore_nan) for row in data]
    ).T
    with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 12):
        means, deviations = row_statistics.row_mean_std(data, ignore_nan=ignore_nan)
        separate_means = row_statistics.row_mean(data, ignore_nan=ignore_nan)
        separate_deviations = row_statistics.row_std(data, ignore_nan=ignore_nan)
    for actual, expected in (
        (means, expected_mean),
        (separate_means, expected_mean),
        (deviations, expected_std),
        (separate_deviations, expected_std),
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_single_precision_accumulates_in_double_precision(ignore_nan: bool, layout: str) -> None:
    data = np.array([[1e8, 1e8 + 8], [1e8 + 8, 1e8 + 24]], dtype=np.float32)
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected_mean, expected_std = np.array(
        [_decimal_moments(row, ignore_nan=ignore_nan) for row in data]
    ).T
    means, deviations = row_statistics.row_mean_std(data, ignore_nan=ignore_nan)
    np.testing.assert_array_equal(means, expected_mean)
    np.testing.assert_array_equal(deviations, expected_std)
    np.testing.assert_array_equal(
        row_statistics.row_mean(data, ignore_nan=ignore_nan), expected_mean
    )
    np.testing.assert_array_equal(irv(data, na_rm=ignore_nan), expected_std)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_infinite_observations_keep_their_unavailable_deviations(ignore_nan: bool) -> None:
    data = np.array(
        [[1, np.inf], [1, -np.inf], [np.inf, -np.inf], [np.inf, np.nan], [np.nan, np.nan]]
    )
    expected = [np.inf, -np.inf, np.nan, np.inf if ignore_nan else np.nan, np.nan]
    means, deviations = row_statistics.row_mean_std(data, ignore_nan=ignore_nan)
    np.testing.assert_array_equal(means, expected)
    np.testing.assert_array_equal(deviations, np.full(5, np.nan))
    np.testing.assert_array_equal(row_statistics.row_mean(data, ignore_nan=ignore_nan), expected)


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_reductions_have_consistent_shapes(ignore_nan: bool, shape: tuple[int, int]) -> None:
    data = np.empty(shape)
    means, deviations = row_statistics.row_mean_std(data, ignore_nan=ignore_nan)
    expected = np.full(shape[0], np.nan)
    np.testing.assert_array_equal(means, expected)
    np.testing.assert_array_equal(deviations, expected)
    np.testing.assert_array_equal(row_statistics.row_mean(data, ignore_nan=ignore_nan), expected)


def test_public_summaries_preserve_extreme_finite_moments() -> None:
    data = np.array([[1e308, 5e307, 1e308, 5e307], [1e-200, 2e-200, 1e-200, 2e-200]])
    expected_mean, expected_std = np.array(
        [_decimal_moments(row, ignore_nan=True) for row in data]
    ).T
    patterns = response_pattern(data)
    for actual in (patterns["acquiescence"], response_time(data, metric="mean")):
        np.testing.assert_allclose(actual, expected_mean, rtol=1e-15, atol=0)
    for actual in (patterns["variability"], response_time(data, metric="sd"), irv(data)):
        np.testing.assert_allclose(actual, expected_std, rtol=1e-15, atol=0)
    np.testing.assert_allclose(
        response_time_consistency(data), expected_std / expected_mean, rtol=1e-15, atol=0
    )


@pytest.mark.parametrize("offset", [1.1, -1.1, 1e100, 1e-100, 1e308])
@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_nearly_constant_moments_match_decimal(
    offset: float, ignore_nan: bool, layout: str
) -> None:
    steps = np.random.default_rng(44).integers(0, 9, size=(9, 17))
    steps[0] = 3
    data = offset + steps * np.spacing(offset)
    data[1, :3] = np.nan
    data[2] = np.nan
    data[3, 1:] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected_mean, expected_std = np.array(
        [_decimal_moments(row, ignore_nan=ignore_nan) for row in data]
    ).T
    with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 40):
        means, deviations = row_statistics.row_mean_std(data, ignore_nan=ignore_nan)
    np.testing.assert_allclose(means, expected_mean, rtol=3e-15, atol=0)
    np.testing.assert_allclose(deviations, expected_std, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("ignore_nan", [False, True])
def test_split_variability_preserves_nearly_constant_sections(
    layout: str, ignore_nan: bool
) -> None:
    data = 1.1 + np.random.default_rng(3).integers(0, 9, size=(7, 21)) * np.spacing(1.1)
    data[0] = 1.1
    data[1, 0] = np.nan
    data[2, 7:14] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = [
        np.mean(
            [_decimal_moments(section, ignore_nan=ignore_nan)[1] for section in row.reshape(3, 7)]
        )
        for row in data
    ]
    with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 50):
        actual = irv(data, na_rm=ignore_nan, split=True, num_split=3)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(data, original)


def test_constant_decimal_responses_have_zero_variability() -> None:
    data = np.full((7, 14), 1.1)
    np.testing.assert_array_equal(irv(data), np.zeros(7))
    np.testing.assert_array_equal(irv(data, split=True, num_split=2), np.zeros(7))
    np.testing.assert_array_equal(response_pattern(data)["variability"], np.zeros(7))
    np.testing.assert_array_equal(response_time(data, metric="sd"), np.zeros(7))
    np.testing.assert_array_equal(response_time_consistency(data), np.zeros(7))


@pytest.mark.parametrize("scale", [1.2345e306, 1e-200])
@pytest.mark.parametrize("ignore_nan", [False, True])
def test_extreme_moment_repairs_preserve_cancellation(scale: float, ignore_nan: bool) -> None:
    data = np.zeros((4, 101))
    data[:3, 0] = scale
    data[0, -1] = -scale
    data[2] *= -1
    data[3] = data[0]
    data[3, 1] = np.nan
    expected_mean, expected_std = np.array(
        [_decimal_moments(row, ignore_nan=ignore_nan) for row in data]
    ).T
    means, deviations = row_statistics.row_mean_std(data, ignore_nan=ignore_nan)
    np.testing.assert_allclose(means, expected_mean, rtol=3e-15, atol=0)
    np.testing.assert_allclose(deviations, expected_std, rtol=3e-15, atol=0)
