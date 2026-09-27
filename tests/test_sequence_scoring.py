"""Independent parity checks for batched, missing-aware response sequences."""

import math
from collections import Counter
from itertools import groupby, product
from unittest.mock import patch

import numpy as np
import pytest

from ier import longstring_pattern, longstring_scores, markov


def _longest_run(row: list[float]) -> float:
    return float(max((sum(1 for _ in group) for _, group in groupby(row)), default=0))


def _longest_pattern(row: list[float], max_k: int) -> float:
    """Enumerate every candidate and compare responses directly."""
    best = 0
    for k in range(2, min(max_k, len(row) // 2) + 1):
        for start in range(len(row) - k):
            pattern = row[start : start + k]
            if len(set(pattern)) < 2:
                continue
            end = start + k
            while end < len(row) and row[end] == pattern[(end - start) % k]:
                end += 1
            if end > start + k:
                best = max(best, end - start)
    return float(best)


def _entropy(row: list[float]) -> float:
    """Use the conditional probability definition, independently of encoding."""
    if len(row) < 2:
        return math.nan
    sources = Counter(row[:-1])
    pairs = Counter(zip(row[:-1], row[1:], strict=True))
    return -sum(
        count / (len(row) - 1) * math.log2(count / sources[source])
        for (source, _), count in pairs.items()
    )


@pytest.mark.parametrize("items", [4, 17, 80])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
def test_missing_sequences_match_scalar_definitions(items: int, layout: str) -> None:
    rng = np.random.default_rng(20260927)
    data = rng.choice([-2.5, 0.0, 0.25, 9.75], size=(83, items))
    data[rng.random(data.shape) < np.linspace(0, 1, len(data))[:, None]] = np.nan
    if layout == "fortran":
        data = np.asfortranarray(data)
    elif layout == "strided":
        data = np.repeat(data, 2, axis=1)[:, ::-2]
    original = data.copy()
    data.flags.writeable = False
    rows = [row[~np.isnan(row)].tolist() for row in data]

    # Tiny batches exercise cross-batch state, partial final batches, and rows
    # with different valid lengths, even within the same batch.
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", items * 7),
        patch("ier.markov._TRANSITION_BATCH_WORKSPACE_BYTES", 512),
    ):
        np.testing.assert_array_equal(longstring_scores(data), [_longest_run(row) for row in rows])
        for max_k in [1, 2, 5, 12]:
            np.testing.assert_array_equal(
                longstring_pattern(data, max_pattern_length=max_k),
                [_longest_pattern(row, max_k) for row in rows],
            )
        np.testing.assert_allclose(markov(data), [_entropy(row) for row in rows], atol=1e-12)
    np.testing.assert_array_equal(data, original)


def test_patterns_exhaustively_match_binary_sequences_with_omissions() -> None:
    data = np.array(list(product([0.0, 1.0, np.nan], repeat=6)))
    rows = [row[~np.isnan(row)].tolist() for row in data]
    for max_k in [2, 3, 5]:
        np.testing.assert_array_equal(
            longstring_pattern(data, max_pattern_length=max_k),
            [_longest_pattern(row, max_k) for row in rows],
        )


@pytest.mark.parametrize("rows", [2, 40])
def test_empty_and_single_response_batches(rows: int) -> None:
    data = np.full((rows, 12), np.nan)
    np.testing.assert_array_equal(longstring_scores(data), np.zeros(rows))
    np.testing.assert_array_equal(longstring_pattern(data), np.zeros(rows))
    assert np.isnan(markov(data)).all()

    data[1::2, 7] = 3.0
    expected = np.zeros(rows)
    expected[1::2] = 1.0
    np.testing.assert_array_equal(longstring_scores(data), expected)
    np.testing.assert_array_equal(longstring_pattern(data), np.zeros(rows))
    assert np.isnan(markov(data)).all()


@pytest.mark.parametrize("states", [5, 64, 65, 200])
def test_missing_markov_dense_and_sparse_categories(states: int) -> None:
    rng = np.random.default_rng(8)
    data = rng.integers(0, states, size=(73, 211)).astype(float)
    data[rng.random(data.shape) < 0.2] = np.nan
    data[0] = np.nan
    data[1] = np.nan
    data[1, 3] = 7.0
    rows = [row[~np.isnan(row)].tolist() for row in data]
    np.testing.assert_allclose(markov(data), [_entropy(row) for row in rows], atol=1e-12)


@pytest.mark.parametrize("rows", [2, 40])
def test_extreme_categories_keep_exact_sequence_equality(rows: int) -> None:
    data = np.tile([np.inf, np.nan, np.inf, -np.inf, 1e308, -1e308, np.nan], (rows, 1))
    np.testing.assert_array_equal(longstring_scores(data), np.full(rows, 2.0))
    expected_rows = [row[~np.isnan(row)].tolist() for row in data]
    np.testing.assert_array_equal(
        longstring_pattern(data), [_longest_pattern(row, 5) for row in expected_rows]
    )
    np.testing.assert_allclose(markov(data), [_entropy(row) for row in expected_rows], atol=1e-12)


def test_wide_missing_sequences_do_not_overflow_run_counts() -> None:
    data = np.tile([1.0, np.nan, 2.0], (40, 300))
    data[1] = 3.0
    np.testing.assert_array_equal(longstring_pattern(data), [600.0, 0.0] + [600.0] * 38)
    np.testing.assert_array_equal(longstring_scores(data), [1.0, 900.0] + [1.0] * 38)


def test_missing_sequences_reject_strict_policy() -> None:
    data = np.ones((40, 8))
    data[-1, -1] = np.nan
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 32):
        for score in (longstring_scores, longstring_pattern, markov):
            with pytest.raises(ValueError, match="data contains missing values"):
                score(data, na_rm=False)
