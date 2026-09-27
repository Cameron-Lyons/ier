"""Mahalanobis regressions for symmetric covariance and bounded complete-case work."""

import tracemalloc
from unittest.mock import patch

import numpy as np
import pytest

from ier.mahad import mahad


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("rank", [1, 5, 13])
@pytest.mark.parametrize("missing", [False, True])
def test_distances_match_direct_covariance_reference(layout: str, rank: int, missing: bool) -> None:
    rng = np.random.default_rng(193)
    data = rng.normal(size=(157, rank)) @ rng.normal(size=(rank, 13))
    if missing:
        data[5:15] = np.nan
        data[::7, 3] = np.nan
    data = data[::-2, ::2] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    valid = ~np.isnan(data).any(axis=1)
    complete = data[valid]
    centered = complete - np.mean(complete, axis=0)
    inverse = np.linalg.pinv(np.cov(complete, rowvar=False))
    expected = np.full(len(data), np.nan)
    expected[valid] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inverse, centered), 0))

    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 31):
        actual = mahad(data, na_rm=True)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("method", ["chi2", "iqr", "zscore"])
def test_missing_rows_keep_distance_and_flag_alignment(method: str) -> None:
    rng = np.random.default_rng(110)
    data = rng.normal(size=(500, 17))
    data[2] *= 10
    data[13:29] = np.nan
    data[::11, 8] = np.nan
    valid = ~np.isnan(data).any(axis=1)
    expected_distances, expected_flags = mahad(data[valid], flag=True, method=method)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 51):
        distances, flags = mahad(data, na_rm=True, flag=True, method=method)
    np.testing.assert_allclose(distances[valid], expected_distances, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(flags[valid], expected_flags)
    assert np.isnan(distances[~valid]).all()
    assert not flags[~valid].any()


def test_covariance_condition_check_does_not_overflow() -> None:
    rng = np.random.default_rng(129)
    data = rng.normal(size=(100, 2)) * [1e100, 1e-100]
    expected = np.abs((data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0], ddof=1))
    actual = mahad(data)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("missing", [False, True])
def test_single_complete_observation_is_rejected(missing: bool) -> None:
    data = [[3.0], [np.nan]] if missing else [[3.0]]
    with pytest.raises(ValueError, match="at least two complete observations"):
        mahad(data, na_rm=missing)


def test_single_variable_matches_sample_standard_scores() -> None:
    data = np.array([[1.0], [3.0], [7.0], [np.nan]])
    values = data[:3, 0]
    expected = np.abs((values - np.mean(values)) / np.std(values, ddof=1))
    actual = mahad(data, na_rm=True)
    np.testing.assert_allclose(actual[:3], expected, atol=1e-14, rtol=1e-14)
    assert np.isnan(actual[-1])


@pytest.mark.parametrize("missing", [False, True])
def test_complete_case_workspaces_do_not_copy_the_input(missing: bool) -> None:
    rng = np.random.default_rng(212)
    data = rng.normal(size=(5000, 80))
    if missing:
        data[::10, 0] = np.nan
    data.flags.writeable = False
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 8000):
        tracemalloc.start()
        try:
            mahad(data, na_rm=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    # The output, row mask, covariance, and bounded workspaces fit well below
    # even one complete-case response copy at this shape and batch size.
    assert peak < data.nbytes / 2
