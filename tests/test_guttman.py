"""Exact pair-count regressions for wide, high-cardinality Guttman scoring."""

from unittest.mock import patch

import numpy as np
import pytest

from ier._column_statistics import column_mean_order
from ier.guttman import _count_merge_errors, guttman


def _pair_counts(data: np.ndarray) -> np.ndarray:
    """Use the public definition without sorting or merging response values."""
    counts = np.zeros(len(data), dtype=np.int64)
    for right in range(1, data.shape[1]):
        counts += (data[:, :right] < data[:, right, None]).sum(axis=1)
    return counts


@pytest.mark.parametrize("width", [2, 31, 32, 33, 127, 767, 768, 769, 1024, 1025])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
def test_merge_counts_preserve_ties_extremes_and_padding(width: int, dtype: type) -> None:
    rng = np.random.default_rng(163)
    data = rng.integers(0, 200, size=(7, width)).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        data += np.iinfo(dtype).max - 200
        data[0, ::3] = np.iinfo(dtype).min
        data[1, ::3] = np.iinfo(dtype).max
    else:
        data[0, ::3] = np.nan
        data[1, ::3] = -np.inf
        data[2, ::3] = np.inf
        data[3] = np.nan
        data[4] = 0.1
    original = data.copy()
    data.flags.writeable = False
    np.testing.assert_array_equal(_count_merge_errors(data), _pair_counts(data))
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_wide_scores_match_direct_definition(layout: str, na_rm: bool, normalize: bool) -> None:
    rng = np.random.default_rng(622)
    data = rng.normal(size=(9, 769))
    data[rng.random(data.shape) < 0.2] = np.nan
    data[0] = np.nan
    data[1] = np.nan
    data[1, 0] = 0
    data[:, 1] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    order = column_mean_order(data, ignore_nan=na_rm)
    expected = _pair_counts(data[:, order]).astype(float)
    if normalize:
        counts = (~np.isnan(data)).sum(axis=1) if na_rm else np.full(len(data), data.shape[1])
        comparisons = counts * (counts - 1) / 2
        expected = np.divide(
            expected, comparisons, out=np.full(len(data), np.nan), where=comparisons > 0
        )
    with (
        patch("ier.guttman._GUTTMAN_BATCH_CELLS", 20_000),
        patch("ier.guttman._count_merge_errors", wraps=_count_merge_errors) as counter,
    ):
        actual = guttman(data, na_rm=na_rm, normalize=normalize)
    assert counter.call_count == 5
    assert all(call.args[0].shape[0] <= 2 for call in counter.call_args_list)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_wide_integer_scores_keep_exact_response_order(dtype: type) -> None:
    base = np.random.default_rng(661).integers(0, 1000, size=(8, 769)).astype(dtype)
    shifted = base + (np.iinfo(dtype).max - 1000)
    order = column_mean_order(base, ignore_nan=True)
    expected = _pair_counts(base[:, order])
    np.testing.assert_array_equal(guttman(shifted, normalize=False), expected)


@pytest.mark.parametrize("width", [70, 513, 767])
def test_narrow_continuous_data_keeps_direct_counter(width: int) -> None:
    data = np.random.default_rng(816).normal(size=(5, width))
    expected = _pair_counts(data[:, column_mean_order(data, ignore_nan=True)])
    with patch("ier.guttman._count_merge_errors", side_effect=AssertionError("unexpected merge")):
        np.testing.assert_array_equal(guttman(data, normalize=False), expected)
