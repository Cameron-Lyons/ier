"""Item discovery parity, undefined correlations, and bounded workspace checks."""

from unittest.mock import patch

import numpy as np
import pytest

from ier._column_statistics import column_correlations
from ier.psychsyn import psychant, psychsyn, psychsyn_critval


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64, np.complex128])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("missing", [False, True])
def test_item_correlations_match_numpy(dtype: type, layout: str, missing: bool) -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(size=(157, 17)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        data += 1j * rng.normal(size=data.shape)
    if missing and not np.issubdtype(dtype, np.integer):
        data[-1, -1] = np.nan
        data[2, 4] = np.nan
    data = data[::-2, ::2] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.corrcoef(data, rowvar=False)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100):
        actual = column_correlations(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(data, original)


def test_item_correlations_center_large_offsets() -> None:
    rng = np.random.default_rng(45)
    data = 1e9 + rng.normal(size=(513, 9))
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100):
        actual = column_correlations(data)
    np.testing.assert_allclose(actual, np.corrcoef(data, rowvar=False), atol=1e-14)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 4.0])
def test_undefined_items_leave_other_correlations_intact(value: float) -> None:
    rng = np.random.default_rng(67)
    data = rng.normal(size=(43, 6))
    data[:, 1] = value
    data[-1, 3] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.corrcoef(data, rowvar=False)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 50):
        actual = column_correlations(data)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert np.isnan(actual[[1, 3]]).all()
    assert np.isnan(actual[:, [1, 3]]).all()


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (2, 5), (5, 5)])
@pytest.mark.parametrize("value", [np.nan, 0.0])
def test_undefined_correlations_keep_item_matrix_shape(
    shape: tuple[int, int], value: float
) -> None:
    actual = column_correlations(np.full(shape, value))
    assert actual.shape == (shape[1], shape[1])
    assert np.isnan(actual).all()


def test_single_nonconstant_item_is_self_correlated() -> None:
    np.testing.assert_allclose(column_correlations(np.arange(5)[:, None]), [[1.0]])


@pytest.mark.parametrize("missing", [False, True])
def test_centering_copies_stay_bounded(missing: bool) -> None:
    data = np.random.default_rng(23).normal(size=(513, 31))
    if missing:
        data[-1, 4] = np.nan
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100),
        patch("ier._column_statistics.np.array", wraps=np.array) as copies,
        patch("ier._column_statistics.np.asarray", wraps=np.asarray) as conversions,
    ):
        column_correlations(data)
    calls = copies.call_args_list + conversions.call_args_list
    assert len(calls) > 1
    assert all(call.args[0].size <= 100 for call in calls)


@pytest.mark.parametrize("anto", [False, True])
@pytest.mark.parametrize("critval", [0.0, 0.6, 0.95])
@pytest.mark.parametrize("missing", ["none", "item", "scattered"])
@pytest.mark.parametrize("resample", [False, True])
def test_scoring_and_discovery_match_original_correlations(
    anto: bool, critval: float, missing: str, resample: bool
) -> None:
    rng = np.random.default_rng(91)
    data = rng.normal(size=(51, 8))
    data += rng.normal(scale=3, size=(51, 1))
    data[:, 1::2] *= -1
    if missing == "item":
        data[3, 0] = np.nan
    elif missing == "scattered":
        data[rng.random(data.shape) < 0.1] = np.nan
    correlation = np.corrcoef(data, rowvar=False)
    threshold = -critval if anto else critval
    scorer = psychant if anto else psychsyn
    with patch("ier.psychsyn.column_correlations", return_value=correlation.copy()):
        expected, expected_counts = scorer(
            data, critval=threshold, diag=True, resample_na=resample, random_seed=7
        )
    with patch("ier.psychsyn.column_correlations", return_value=correlation.copy()):
        expected_pairs = psychsyn_critval(data, anto=anto, min_correlation=critval)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100):
        actual, actual_counts = scorer(
            data, critval=threshold, diag=True, resample_na=resample, random_seed=7
        )
        actual_pairs = psychsyn_critval(data, anto=anto, min_correlation=critval)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(actual_counts, expected_counts)
    np.testing.assert_allclose(actual_pairs, expected_pairs, atol=1e-14)


@pytest.mark.parametrize("data", [[[1.0, 2.0, 3.0]], [[2.0] * 3] * 4, [[np.nan] * 3] * 4])
def test_undefined_discovery_returns_no_pairs_without_warnings(data: list[list[float]]) -> None:
    scores, counts = psychsyn(data, diag=True)
    assert np.isnan(scores).all()
    np.testing.assert_array_equal(counts, np.zeros(len(data)))
    assert psychsyn_critval(data) == []
