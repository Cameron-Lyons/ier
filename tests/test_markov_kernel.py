"""Sorted-multiplicity transition entropy matches the previous per-row scorer."""

from collections import Counter
from decimal import Decimal, localcontext
from functools import cache
from unittest.mock import patch

import numpy as np
import pytest

from ier import markov
from ier.markov import (
    _encode_states,
    _markov_complete,
    _sum_xlogx_multiplicities,
    _transition_entropies_from_codes,
)


def _reference_row_entropy(row: np.ndarray) -> float:
    """Compute transition entropy from the observed counts in one row."""
    _, encoded = np.unique(row, return_inverse=True)
    n_states = int(np.max(encoded)) + 1
    from_counts = np.bincount(encoded[:-1])
    pair_ids = encoded[:-1] * n_states + encoded[1:]
    _, pair_counts = np.unique(pair_ids, return_counts=True)
    positive_from = from_counts[from_counts > 0]
    from_terms = positive_from @ np.log2(positive_from)
    pair_terms = pair_counts @ np.log2(pair_counts)
    return float((from_terms - pair_terms) / (len(row) - 1))


def _reference_markov(x: np.ndarray) -> np.ndarray:
    """Score each row's observed responses independently."""
    scores = []
    for row in np.asarray(x):
        observed = row[~np.isnan(row)] if row.dtype.kind == "f" else row
        scores.append(np.nan if len(observed) < 2 else _reference_row_entropy(observed))
    return np.asarray(scores)


def _assert_matches_reference(x: np.ndarray) -> np.ndarray:
    actual = markov(x)
    expected = _reference_markov(x)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13, equal_nan=True)
    return actual


def _with_missing(data: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    missing = data.astype(float)
    missing[rng.random(missing.shape) < rate] = np.nan
    return missing


@cache
def _exact_xlog2x(count: int) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return Decimal(count) * Decimal(count).ln() / Decimal(2).ln()


def _exact_row_entropy(row: np.ndarray) -> float:
    """Evaluate the count form of conditional entropy with 50-digit logarithms."""
    values = row.tolist()
    sources = Counter(values[:-1])
    pairs = Counter(zip(values[:-1], values[1:], strict=True))
    with localcontext() as context:
        context.prec = 50
        numerator = sum(map(_exact_xlog2x, sources.values()), Decimal(0))
        numerator -= sum(map(_exact_xlog2x, pairs.values()), Decimal(0))
        return float(numerator / (len(values) - 1))


@pytest.mark.parametrize("missing_rate", [0.0, 0.15])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_integer_slider_matches_previous_scorer(missing_rate: float, layout: str) -> None:
    rng = np.random.default_rng(100)
    data = rng.integers(0, 101, size=(300, 50)).astype(float)
    # Repeated segments create states with several distinct successors.
    data[::4, 25:] = data[::4, :25]
    data[1::5, ::3] = 50.0
    data = np.array(_with_missing(data, rng, missing_rate), order=layout)
    _assert_matches_reference(data)


@pytest.mark.parametrize("missing_rate", [0.0, 0.1])
def test_continuous_responses_match_previous_scorer(missing_rate: float) -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(200, 40))
    data[::3, 20:] = data[::3, :20]
    data[1::7] = np.round(data[1::7], 1)
    _assert_matches_reference(_with_missing(data, rng, missing_rate))


def test_large_integer_labels_remain_exact() -> None:
    rng = np.random.default_rng(60)
    labels = 2**60 + rng.integers(-500, 500, size=(120, 30), dtype=np.int64)
    labels[::2, 15:] = labels[::2, :15]
    labels[1::3, ::2] = 2**60
    # Neighboring labels above 2**53 collapse if converted to floating point.
    labels[5, :] = 2**60 + np.arange(30) % 2
    scores = _assert_matches_reference(labels)
    assert scores[5] == 0.0


def test_infinite_categories_match_previous_scorer() -> None:
    rng = np.random.default_rng(11)
    data = rng.integers(0, 90, size=(60, 25)).astype(float)
    data[rng.random(data.shape) < 0.2] = np.inf
    data[rng.random(data.shape) < 0.2] = -np.inf
    data[0] = [np.inf, -np.inf] * 12 + [np.inf]
    data[1] = np.inf
    _assert_matches_reference(data)
    _assert_matches_reference(_with_missing(data, rng, 0.1))


def test_padded_rows_cover_every_observed_count() -> None:
    rng = np.random.default_rng(3)
    data = rng.integers(0, 200, size=(9, 24)).astype(float)
    data[0] = np.nan
    data[1, 1:] = np.nan
    data[2, 2:] = np.nan
    data[3, [0, 5, 9, 23]] = np.nan
    data[4, :20] = np.nan
    data[5, ::2] = np.nan
    data[6] = 4.0
    data[6, ::5] = np.nan
    scores = _assert_matches_reference(data)
    assert np.isnan(scores[:2]).all()
    assert scores[2] == 0.0
    assert scores[6] == 0.0


def test_padding_uses_distinct_sentinels() -> None:
    codes = np.array([[3, 3, 3, 3, 3], [1, 2, 1, 2, 0], [0, 0, 0, 0, 0], [4, 0, 4, 0, 4]])
    counts = np.array([0, 4, 2, 5])
    actual = _transition_entropies_from_codes(codes, 5, counts=counts)
    expected = [np.nan, _reference_row_entropy(codes[1, :4]), 0.0, 0.0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    unpadded = _transition_entropies_from_codes(codes[1:2, :4], 5)
    np.testing.assert_array_equal(actual[1:2], unpadded)


def test_long_row_matches_previous_scorer() -> None:
    rng = np.random.default_rng(20_000)
    row = rng.integers(0, 400, size=20_000).astype(float)
    row[5000:5600] = 7.0
    row[9000:9300] = np.tile([1.0, 2.0, 3.0], 100)
    _assert_matches_reference(row[None, :])
    _assert_matches_reference(np.arange(20_000, dtype=float)[None, :])
    assert abs(markov(row[None, :])[0] - _exact_row_entropy(row)) <= 4e-15


@pytest.mark.parametrize("states", [100, 400])
def test_long_sorted_rows_keep_pairwise_summation_accuracy(states: int) -> None:
    # A masked reduction over sparse run ends adds one term at a time, which
    # left 60,000-response rows about 2e-13 from the exact entropy.
    rng = np.random.default_rng(states)
    data = rng.integers(0, states, size=(3, 60_000)).astype(float)
    data[1, 10_000:40_000] = np.tile(np.arange(10.0), 3000)
    actual = markov(data)
    expected = [_exact_row_entropy(row) for row in data]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-15)


def test_forced_sorted_kernel_matches_exact_entropy_on_long_rows() -> None:
    rng = np.random.default_rng(64)
    data = rng.integers(0, 64, size=(2, 50_000)).astype(float)
    with patch("ier.markov._DENSE_CELLS_PER_SORT_STEP", 0):
        sorted_scores = markov(data)
    with patch("ier.markov._DENSE_CELLS_PER_SORT_STEP", np.inf):
        dense = markov(data)
    expected = [_exact_row_entropy(row) for row in data]
    np.testing.assert_allclose(sorted_scores, expected, rtol=0, atol=4e-15)
    np.testing.assert_allclose(dense, expected, rtol=0, atol=4e-15)


@pytest.mark.parametrize("run", [254, 255, 256, 257, 300])
def test_multiplicities_cross_the_small_dtype_boundary(run: int) -> None:
    # A distinct-valued row keeps the block above the dense state limit.
    data = np.vstack((np.arange(run + 2, dtype=float), np.full((2, run + 2), 5.0)))
    data[2, -1] = 6.0
    scores = _assert_matches_reference(data)
    assert scores[1] == 0.0
    # One departure after a long run: (n log2 n - (n - 1) log2 (n - 1)) / n.
    n = run + 1
    np.testing.assert_allclose(
        scores[2], (n * np.log2(n) - (n - 1) * np.log2(n - 1)) / n, rtol=1e-14
    )
    ids = np.zeros((1, run), dtype=np.int64)
    expected = run * np.log2(run)
    np.testing.assert_allclose(_sum_xlogx_multiplicities(ids), [expected], rtol=1e-15)


def test_small_row_batches_match_previous_scorer() -> None:
    rng = np.random.default_rng(19)
    data = _with_missing(rng.integers(0, 150, size=(97, 31)).astype(float), rng, 0.25)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 31 * 4 + 3):
        _assert_matches_reference(data)


@pytest.mark.parametrize("states", [8, 21, 40, 64])
@pytest.mark.parametrize("n_items", [6, 50, 400])
@pytest.mark.parametrize("missing", [False, True])
def test_sorted_and_dense_kernels_agree(states: int, n_items: int, missing: bool) -> None:
    rng = np.random.default_rng(states * n_items)
    data = rng.integers(0, states, size=(150, n_items)).astype(float)
    data[::5, n_items // 2 :] = data[::5, : n_items - n_items // 2]
    data[:, 0] = np.arange(150) % states
    if missing:
        data = _with_missing(data, rng, 0.2)
    with patch("ier.markov._DENSE_CELLS_PER_SORT_STEP", np.inf):
        dense = markov(data)
    with patch("ier.markov._DENSE_CELLS_PER_SORT_STEP", 0):
        sorted_scores = markov(data)
    np.testing.assert_array_equal(np.isnan(sorted_scores), np.isnan(dense))
    np.testing.assert_allclose(sorted_scores, dense, rtol=0, atol=4e-15, equal_nan=True)
    np.testing.assert_allclose(markov(data), dense, rtol=0, atol=4e-15, equal_nan=True)


def _uses_sorted_kernel(states: int, n_items: int) -> bool:
    data = np.tile(np.arange(n_items, dtype=float) % states, (3, 1))
    with patch(
        "ier.markov._transition_entropies_from_codes", wraps=_transition_entropies_from_codes
    ) as sorted_kernel:
        _markov_complete(data)
    return bool(sorted_kernel.called)


@pytest.mark.parametrize(
    ("cells_per_step", "states", "n_items", "expected_sorted"),
    [
        # 0.5 * 50 * log2(50) = 141.1 and 0.5 * 800 * log2(800) = 3857.7.
        (0.5, 11, 50, False),
        (0.5, 12, 50, True),
        (0.5, 62, 800, False),
        (0.5, 63, 800, True),
        # 1.5 * 50 * log2(50) = 423.3 and 1.5 * 200 * log2(200) = 2293.2.
        (1.5, 20, 50, False),
        (1.5, 21, 50, True),
        (1.5, 47, 200, False),
        (1.5, 48, 200, True),
        (np.inf, 64, 3, False),
        (np.inf, 65, 10_000, True),
    ],
)
def test_dispatch_compares_state_square_with_sort_steps(
    cells_per_step: float, states: int, n_items: int, expected_sorted: bool
) -> None:
    with patch("ier.markov._DENSE_CELLS_PER_SORT_STEP", cells_per_step):
        assert _uses_sorted_kernel(states, n_items) is expected_sorted


@pytest.mark.parametrize(
    ("states", "n_items", "sorted_numpy_1", "sorted_numpy_2"),
    [
        (5, 50, False, False),
        (7, 80, False, False),
        (21, 50, True, True),
        (64, 50, True, True),
        (29, 200, False, True),
        # The previous rule sorted these long rows, up to 2x slower than dense
        # tables with NumPy 1.26 and no faster with NumPy 2.
        (45, 500, False, False),
        (64, 1000, False, False),
        (65, 10_000, True, True),
    ],
)
def test_default_dispatch_depends_on_numpy_sort_speed(
    states: int, n_items: int, sorted_numpy_1: bool, sorted_numpy_2: bool
) -> None:
    numpy_2 = np.lib.NumpyVersion(np.__version__) >= "2.0.0"
    expected_sorted = sorted_numpy_2 if numpy_2 else sorted_numpy_1
    assert _uses_sorted_kernel(states, n_items) is expected_sorted


def test_noninteger_codes_keep_sorted_category_order() -> None:
    rng = np.random.default_rng(5)
    data = rng.choice([-np.inf, -2.5, -0.0, 0.0, 0.25, 9.75, np.inf], size=(40, 12))
    codes, n_states = _encode_states(data)
    categories = np.unique(data)
    assert n_states == len(categories) == 6
    assert codes.dtype == np.intp
    assert codes.shape == data.shape
    np.testing.assert_array_equal(codes, np.searchsorted(categories, data))
    many = rng.normal(size=(30, 20))
    codes, n_states = _encode_states(many)
    np.testing.assert_array_equal(codes, np.searchsorted(np.unique(many), many))
    assert n_states == many.size
