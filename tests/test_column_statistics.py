"""Regression coverage for the shared bounded item-mean reduction."""

from unittest.mock import patch

import numpy as np
import pytest

from ier._column_statistics import column_mean


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_column_means_match_numpy(ignore_nan: bool, dtype: type, layout: str) -> None:
    rng = np.random.default_rng(723)
    data = rng.normal(size=(157, 31)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        data[rng.random(data.shape) < 0.2] = np.nan
    data = data[::-2, ::2] if layout == "strided" else np.array(data, order=layout)
    data.flags.writeable = False
    expected = np.nanmean(data, axis=0, dtype=float) if ignore_nan else np.mean(data, axis=0)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100):
        actual = column_mean(data, ignore_nan=ignore_nan)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_column_means_leave_empty_items_unavailable_without_warning() -> None:
    data = np.array([[np.nan, 1.0, np.nan], [np.nan, 3.0, 5.0]])
    np.testing.assert_array_equal(column_mean(data, ignore_nan=True), [np.nan, 2.0, 5.0])
    np.testing.assert_array_equal(column_mean(data, ignore_nan=False), [np.nan, 2.0, np.nan])


def test_column_mean_masks_are_bounded() -> None:
    data = np.zeros((513, 31))
    data[3, 4] = np.nan
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100),
        patch("ier._column_statistics.np.isnan", wraps=np.isnan) as masks,
    ):
        column_mean(data, ignore_nan=True)
    assert masks.call_count > 1
    assert all(call.args[0].size <= 100 for call in masks.call_args_list)
