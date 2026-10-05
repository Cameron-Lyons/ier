"""Pairwise-complete item correlations against a brute-force shared-row oracle."""

import tracemalloc
from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest

from ier._column_statistics import (
    _shared_row_correlation,
    column_correlations,
    pairwise_column_correlations,
)


def _shared_row_oracle(x: np.ndarray, min_pairs: int = 3) -> np.ndarray:
    """Correlate every item pair over its complete rows with ``np.corrcoef``."""
    n_items = x.shape[1]
    expected = np.full((n_items, n_items), np.nan)
    finite_items = ~np.isinf(x).any(axis=0)
    for first in range(n_items):
        for second in range(n_items):
            shared = ~np.isnan(x[:, first]) & ~np.isnan(x[:, second])
            if not (finite_items[first] and finite_items[second]) or shared.sum() < min_pairs:
                continue
            pair = np.array(x[shared][:, [first, second]], dtype=float)
            # A sample shift and power-of-two scaling keep np.corrcoef exact enough
            # for large offsets and extreme magnitudes; constant items stay zero.
            pair -= pair[0]
            magnitude = np.max(np.abs(pair), axis=0)
            pair = np.ldexp(pair, -np.frexp(magnitude)[1])
            with np.errstate(invalid="ignore", divide="ignore"):
                expected[first, second] = np.corrcoef(pair, rowvar=False)[0, 1]
    return expected


def _likert(rows: int, items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rows, 1))
    return np.clip(np.rint(3 + latent + rng.normal(scale=0.8, size=(rows, items))), 1, 5)


def _assert_matches_oracle(x: np.ndarray, *, atol: float = 1e-13) -> np.ndarray:
    actual = pairwise_column_correlations(x)
    expected = _shared_row_oracle(x)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True)
    np.testing.assert_array_equal(actual, actual.T)
    return actual


def _scattered(rate: float) -> np.ndarray:
    data = _likert(400, 12, seed=int(rate * 1000))
    data[np.random.default_rng(7).random(data.shape) < rate] = np.nan
    return data


def _itemwise() -> np.ndarray:
    data = _likert(300, 9, seed=11)
    data[:120, 1] = np.nan
    data[200:, 4] = np.nan
    data[::3, 6] = np.nan
    data[:, 8] = np.nan
    data[5:, 7] = np.nan  # Five observed rows.
    return data


def _offset() -> np.ndarray:
    rng = np.random.default_rng(13)
    data = 1e9 + rng.normal(size=(500, 6)) + rng.normal(size=(500, 1))
    data[rng.random(data.shape) < 0.1] = np.nan
    return data


def _extreme_scales() -> np.ndarray:
    rng = np.random.default_rng(17)
    data = rng.normal(size=(300, 6)) + rng.normal(size=(300, 1))
    data[:, 0] *= 1e-300
    data[:, 1] *= 1e300
    data[:, 2] *= 2.0**-1060  # Subnormal responses.
    data[:, 3] = 1e300  # Constant at the edge of the finite range.
    data[rng.random(data.shape) < 0.05] = np.nan
    return data


def _constant_within_pairs() -> np.ndarray:
    rng = np.random.default_rng(19)
    data = rng.normal(size=(80, 5))
    data[:40, 1] = 0.1  # Constant on the rows item 2 observes.
    data[40:, 2] = np.nan
    data[:, 3] = 0.3
    data[::2, 3] = np.nan
    data[60:, 4] = 4.0
    data[:60, 0] = np.nan
    return data


def _sparse_overlap() -> np.ndarray:
    rng = np.random.default_rng(23)
    data = np.full((30, 5), np.nan)
    data[:3, 0] = rng.normal(size=3)  # Shares three rows with item 2.
    data[27:, 1] = rng.normal(size=3)
    data[:5, 2] = rng.normal(size=5)
    data[[2, 28, 29], 3] = rng.normal(size=3)  # Shares two rows with item 1.
    data[:, 4] = rng.normal(size=30)
    data[[4, 9], 4] = np.inf
    data[7, 4] = np.nan
    return data


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: _scattered(0.01), id="mcar-1pct"),
        pytest.param(lambda: _scattered(0.05), id="mcar-5pct"),
        pytest.param(lambda: _scattered(0.20), id="mcar-20pct"),
        pytest.param(_itemwise, id="itemwise"),
        pytest.param(_offset, id="offset-1e9"),
        pytest.param(_extreme_scales, id="extreme-scales"),
        pytest.param(_constant_within_pairs, id="constant-within-pairs"),
        pytest.param(_sparse_overlap, id="sparse-overlap-and-infinite"),
    ],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_pairwise_correlations_match_shared_row_oracle(
    factory: Callable[[], np.ndarray], layout: str
) -> None:
    data = factory()
    if layout == "strided":
        doubled = np.repeat(data, 2, axis=1)
        data = doubled[::-1, ::2]
    else:
        data = np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    _assert_matches_oracle(data)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 50):
        _assert_matches_oracle(data)
    np.testing.assert_array_equal(data, original)


def test_too_few_shared_rows_and_infinite_items_are_undefined() -> None:
    actual = _assert_matches_oracle(_sparse_overlap())
    assert np.isnan(actual[1, 3]) and np.isnan(actual[0, 1])
    assert np.isnan(actual[4]).all() and np.isnan(actual[:, 4]).all()
    assert np.isfinite(actual[0, 2]) and np.isfinite(np.diagonal(actual)[:4]).all()


def test_constant_pairs_and_diagonal_follow_complete_semantics() -> None:
    actual = _assert_matches_oracle(_constant_within_pairs())
    assert np.isnan(actual[1, 2])  # Item 1 is constant where item 2 is observed.
    assert np.isfinite(actual[1, 4])
    assert np.isnan(actual[3]).all()  # Constant wherever it is observed.
    np.testing.assert_allclose(np.diagonal(actual)[[0, 1, 2, 4]], 1.0, atol=1e-15)


@pytest.mark.parametrize("min_pairs", [2, 5])
def test_min_pairs_sets_the_shared_row_requirement(min_pairs: int) -> None:
    data = _sparse_overlap()
    actual = pairwise_column_correlations(data, min_pairs=min_pairs)
    np.testing.assert_allclose(
        actual, _shared_row_oracle(data, min_pairs=min_pairs), atol=1e-13, equal_nan=True
    )
    # Two shared rows correlate perfectly; only an explicit R-style minimum accepts them.
    assert bool(np.isclose(abs(actual[1, 3]), 1.0)) is (min_pairs == 2)


@pytest.mark.parametrize("value", [1, 0, -3, 2.5, True, "3", None])
def test_min_pairs_must_be_an_integer_of_at_least_two(value: object) -> None:
    with pytest.raises(ValueError, match="min_pairs must be an integer of at least 2"):
        pairwise_column_correlations(_scattered(0.05), min_pairs=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(_likert(200, 8, seed=3), id="float64"),
        pytest.param(_likert(200, 8, seed=3).astype(np.float32), id="float32"),
        pytest.param(_likert(200, 8, seed=3).astype(np.int64), id="int64"),
        pytest.param(_likert(200, 8, seed=3).astype(np.uint8), id="uint8"),
        pytest.param(_likert(200, 8, seed=3) > 3, id="bool"),
        pytest.param(np.ones((2, 3)), id="two-rows"),
        pytest.param(np.where(np.eye(4, dtype=bool), np.inf, _likert(4, 4, seed=5)), id="infinite"),
    ],
)
def test_complete_inputs_reuse_complete_correlations(data: np.ndarray) -> None:
    with patch(
        "ier._column_statistics._shared_row_correlation",
        side_effect=AssertionError("complete inputs need no shared-row repair"),
    ):
        actual = pairwise_column_correlations(data)
    np.testing.assert_array_equal(actual, column_correlations(data))


def test_unobserved_and_unusable_items_leave_no_columns() -> None:
    data = np.full((6, 3), np.nan)
    data[0, 0] = 1.0
    data[:, 1] = np.inf
    data[1, 1] = np.nan
    actual = pairwise_column_correlations(data)
    assert actual.shape == (3, 3)
    assert np.isnan(actual).all()


def test_shift_cancellation_is_repaired_from_shared_rows() -> None:
    rng = np.random.default_rng(29)
    data = rng.normal(size=(600, 4)) + rng.normal(size=(600, 1))
    # Item 0's typical response is enormous, but the rows shared with item 1 are
    # ordinary, so the shift cancels nearly all of that pair's moments.
    data[:500, 0] = 1e9 + rng.normal(size=500)
    data[:500, 1] = np.nan
    data[rng.random(data.shape) < 0.02] = np.nan
    with patch(
        "ier._column_statistics._shared_row_correlation", wraps=_shared_row_correlation
    ) as repair:
        _assert_matches_oracle(data)
    repaired = {tuple(sorted(call.args[1:])) for call in repair.call_args_list}
    assert (0, 1) in repaired
    assert all(0 in pair or 1 in pair for pair in repaired)


def test_ordinary_responses_need_no_repairs() -> None:
    data = _scattered(0.05)
    with patch(
        "ier._column_statistics._shared_row_correlation",
        side_effect=AssertionError("well-conditioned pairs were recomputed"),
    ):
        _assert_matches_oracle(data)


def test_outlying_first_responses_keep_full_precision() -> None:
    rng = np.random.default_rng(31)
    data = rng.normal(size=(5000, 6)) + rng.normal(size=(5000, 1))
    data[rng.random(data.shape) < 0.05] = np.nan
    data[0] = 1e5  # Shifting by the first observation would cancel ~4 digits.
    _assert_matches_oracle(data, atol=2e-15 * 8)


def test_integer_valued_responses_stay_exact() -> None:
    data = _likert(1000, 6, seed=37)
    data[np.random.default_rng(41).random(data.shape) < 0.1] = np.nan
    shifted = pairwise_column_correlations(data + 2.0**40)
    np.testing.assert_array_equal(shifted, pairwise_column_correlations(data))


def _unshared_extreme(outlier: float, *, symmetric: bool) -> np.ndarray:
    rng = np.random.default_rng(43)
    latent = rng.normal(size=200)
    data = np.column_stack(
        [latent + rng.normal(scale=0.5, size=200), latent + rng.normal(scale=0.5, size=200)]
    )
    if symmetric:
        # Equally many shared 1s and 3s around the shift 2 sum to exactly zero.
        data[1:, 0] = np.resize([2.0, 1.0, 3.0], 199)
    data[0] = [outlier, np.nan]  # The outlier sets item 0's scale but is never shared.
    return data


@pytest.mark.parametrize("outlier", [1e158, 1e160, 1e165, 1e200, 1e300, -1.7e308])
@pytest.mark.parametrize("symmetric", [False, True])
def test_unshared_extreme_responses_keep_shared_row_precision(
    outlier: float, symmetric: bool
) -> None:
    data = _unshared_extreme(outlier, symmetric=symmetric)
    shared = ~np.isnan(data).any(axis=1)
    expected = np.corrcoef(data[shared], rowvar=False)[0, 1]
    with patch(
        "ier._column_statistics._shared_row_correlation", wraps=_shared_row_correlation
    ) as repair:
        actual = pairwise_column_correlations(data)
    # Scaling by the outlier leaves the shared squares subnormal or zero.
    assert repair.call_count == 1
    np.testing.assert_allclose(actual[[0, 1], [1, 0]], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(actual, actual.T)


@pytest.mark.parametrize("offset", [5.0, 10.0, 30.0, 1e3, 3e3, 6e3, 8e3])
def test_unshared_outlying_shift_keeps_full_precision(offset: float) -> None:
    rng = np.random.default_rng(47)
    latent = rng.normal(size=4000)
    data = np.column_stack(
        [latent + rng.normal(scale=0.3, size=4000), latent + rng.normal(scale=0.3, size=4000)]
    )
    data = np.round(data, 3)
    block_rows = 10  # Rows per accumulation block for two items at the budget below.
    data[:block_rows, 0] = np.nan
    data[0] = [offset, np.nan]  # Item 0's only first-block response becomes its shift.
    first, second = data[~np.isnan(data).any(axis=1)].astype(np.longdouble).T
    first -= first.mean()
    second -= second.mean()
    expected = float(first @ second / np.sqrt((first @ first) * (second @ second)))
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 6 * block_rows):
        actual = pairwise_column_correlations(data)
    np.testing.assert_allclose(actual[[0, 1], [1, 0]], expected, rtol=0.0, atol=1e-12)


def _skip_logic() -> np.ndarray:
    data = _likert(600, 12, seed=53)
    data[:300, 4:8] = np.nan
    data[300:, 8:] = np.nan
    data[np.random.default_rng(59).random(data.shape) < 0.03] = np.nan
    return data


def _with_constant_item() -> np.ndarray:
    data = _likert(400, 8, seed=61)
    data[:, 2] = 5.0  # Exactly zero shifted moments are checked, not recomputed.
    data[np.random.default_rng(67).random(data.shape) < 0.05] = np.nan
    return data


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: _scattered(0.20), id="mcar-20pct"),
        pytest.param(_offset, id="offset-1e9"),
        pytest.param(_skip_logic, id="skip-logic"),
        pytest.param(_with_constant_item, id="constant-item"),
    ],
)
def test_typical_responses_take_the_fast_path(factory: Callable[[], np.ndarray]) -> None:
    data = factory()
    with patch(
        "ier._column_statistics._shared_row_correlation",
        side_effect=AssertionError("well-conditioned pairs were recomputed"),
    ):
        _assert_matches_oracle(data)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.bool_])
def test_min_pairs_applies_without_missing_responses(dtype: type) -> None:
    two = np.array([[1, 2, 4, 5], [3, 5, 1, 2]]).astype(dtype)
    assert np.isnan(pairwise_column_correlations(two)).all()
    four = np.array([[1, 2, 4], [3, 5, 1], [2, 2, 7], [5, 1, 3]]).astype(dtype)
    assert np.isnan(pairwise_column_correlations(four, min_pairs=5)).all()
    np.testing.assert_array_equal(
        pairwise_column_correlations(four, min_pairs=4), column_correlations(four)
    )


def test_blank_respondents_do_not_change_pairwise_correlations() -> None:
    two = np.array([[1.0, 2.0, 4.0, 5.0], [3.0, 5.0, 1.0, 2.0]])
    three = np.vstack([two, [2.0, 2.0, 7.0, 1.0]])
    for data in (two, three):
        with_blank = np.vstack([data, np.full(4, np.nan)])
        np.testing.assert_allclose(
            pairwise_column_correlations(with_blank),
            pairwise_column_correlations(data),
            rtol=0.0,
            atol=1e-15,
            equal_nan=True,
        )
    assert np.isnan(pairwise_column_correlations(two)).all()
    assert np.isfinite(pairwise_column_correlations(three)).all()


@pytest.mark.parametrize("all_usable", [True, False])
def test_peak_workspace_stays_near_four_item_by_item_accumulators(all_usable: bool) -> None:
    rng = np.random.default_rng(71)
    data = rng.normal(size=(3000, 200)) + rng.normal(size=(3000, 1))
    data[rng.random(data.shape) < 0.05] = np.nan
    if not all_usable:
        data[2:, 0] = np.nan  # Too few responses, so the result is assembled separately.
    matrix_bytes = data.shape[1] ** 2 * 8
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 4096):
        tracemalloc.start()
        try:
            actual = pairwise_column_correlations(data)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert bool(np.isnan(actual[0, 1])) is not all_usable
    # Four accumulators, Boolean pair masks, and bounded blocks; finishing the
    # moments out of place previously held about ten such matrices at once.
    assert peak < 5 * matrix_bytes
    # Sliced cross-product updates agree with one product per block.
    np.testing.assert_allclose(
        actual, pairwise_column_correlations(data), rtol=0.0, atol=1e-14, equal_nan=True
    )
    np.testing.assert_array_equal(actual, actual.T)
