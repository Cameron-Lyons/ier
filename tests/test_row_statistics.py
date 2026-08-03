"""Tests for bounded row-wise statistical reductions."""

from unittest.mock import patch

import numpy as np

import ier._row_statistics as row_statistics


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


def test_compressed_row_groups_preserve_order_and_budget() -> None:
    """Missing responses are removed in row order within bounded equal-length groups."""
    rng = np.random.default_rng(20260803)
    data = rng.integers(1, 6, size=(37, 11)).astype(float)
    data[rng.random(data.shape) < 0.3] = np.nan
    data[0] = np.nan
    data[1, 2:] = np.nan
    original = data.copy()
    expected_rows = np.flatnonzero(np.count_nonzero(~np.isnan(data), axis=1) >= 2)
    observed_rows: list[int] = []

    for rows, compressed in row_statistics.compressed_row_groups(
        data,
        min_columns=2,
        max_elements=30,
    ):
        assert compressed.size <= 30
        assert not np.isnan(compressed).any()
        for row_index, packed in zip(rows, compressed, strict=True):
            observed_rows.append(int(row_index))
            np.testing.assert_array_equal(packed, data[row_index][~np.isnan(data[row_index])])

    np.testing.assert_array_equal(np.sort(observed_rows), expected_rows)
    np.testing.assert_array_equal(data, original)
