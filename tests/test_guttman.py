"""Exact Guttman error definitions, item ordering, and pair-count regressions."""

from fractions import Fraction
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from ier._column_statistics import column_mean_order
from ier.guttman import (
    _count_categorical_errors,
    _count_merge_errors,
    _count_pairwise_errors,
    guttman,
    guttman_flag,
)

_KERNELS = {
    "_count_categorical_errors": _count_categorical_errors,
    "_count_pairwise_errors": _count_pairwise_errors,
    "_count_merge_errors": _count_merge_errors,
}


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
    # Item means use available responses for both missing-data policies.
    order = column_mean_order(data, ignore_nan=True)
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


def _textbook_errors(data: np.ndarray) -> np.ndarray:
    """Count Guttman errors from exact item means without production kernels.

    Items run easiest first (largest mean of the available responses first, ties
    by column, items without responses last); an error is a pair whose later,
    harder item has the higher response. Missing responses never form errors.
    """
    keys = []
    for column in data.T:
        observed = column[~np.isnan(column)]
        if len(observed) == 0:
            keys.append((1, Fraction(0)))
        else:
            keys.append((0, -sum(map(Fraction, observed.tolist())) / len(observed)))
    order = sorted(range(data.shape[1]), key=keys.__getitem__)
    errors = np.zeros(len(data), dtype=np.int64)
    for row, values in enumerate(data[:, order]):
        errors[row] = np.count_nonzero(np.triu(values[:, None] < values[None, :], k=1))
    return errors


@pytest.mark.parametrize("n_items", [2, 5, 9])
def test_cumulative_patterns_score_zero_and_reversed_patterns_score_maximum(n_items: int) -> None:
    # Row r endorses exactly the r easiest items. Three copies of each cumulative
    # pattern plus one reversed copy give item j a total of 3J + 1 - 2j.
    cumulative = np.tri(n_items + 1, n_items, k=-1, dtype=np.int64)
    reversed_patterns = cumulative[:, ::-1]
    data = np.vstack([cumulative, cumulative, cumulative, reversed_patterns])
    endorsed = np.arange(n_items + 1)
    expected = np.concatenate([np.zeros(3 * (n_items + 1)), endorsed * (n_items - endorsed)])
    permutation = np.random.default_rng(n_items).permutation(n_items)

    for columns in (np.arange(n_items), permutation):
        np.testing.assert_array_equal(guttman(data[:, columns], normalize=False), expected)
        np.testing.assert_array_equal(
            guttman(data[:, columns]), expected / (n_items * (n_items - 1) / 2)
        )


@pytest.mark.parametrize(
    ("kernel", "shape", "categorical"),
    [
        ("_count_categorical_errors", (40, 15), True),
        ("_count_pairwise_errors", (40, 50), False),
        ("_count_merge_errors", (2, 800), False),
    ],
)
@pytest.mark.parametrize("na_rm", [False, True])
def test_scores_match_textbook_oracle_on_every_kernel(
    kernel: str, shape: tuple[int, int], categorical: bool, na_rm: bool
) -> None:
    rng = np.random.default_rng(2026)
    data = rng.integers(1, 6, size=shape).astype(float) if categorical else rng.normal(size=shape)
    missing = rng.random(shape) < 0.1
    # Mix complete even items with partially observed odd items.
    missing[:, ::2] = False
    data[missing] = np.nan
    with patch(f"ier.guttman.{kernel}", wraps=_KERNELS[kernel]) as counter:
        actual = guttman(data, na_rm=na_rm, normalize=False)
    assert counter.call_count > 0
    # Raw counts depend only on available-response item means, never on na_rm.
    np.testing.assert_array_equal(actual, _textbook_errors(data))


@pytest.mark.parametrize("shape", [(80, 12), (60, 40), (3, 800)])
def test_scores_are_invariant_to_column_order_when_item_means_differ(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(4)
    n_items = shape[1]
    if n_items == 12:
        data = rng.integers(0, 5, size=shape)
    else:
        data = rng.normal(size=shape)
        data[rng.random(shape) < 0.05] = np.nan
    assert len(np.unique(np.nanmean(data, axis=0))) == n_items
    expected = guttman(data, normalize=False)
    for _ in range(3):
        permutation = rng.permutation(n_items)
        np.testing.assert_array_equal(guttman(data[:, permutation], normalize=False), expected)


@pytest.mark.parametrize("na_rm", [False, True])
def test_one_missing_response_leaves_item_order_and_other_scores_unchanged(na_rm: bool) -> None:
    # Perfectly cumulative patterns; item 0 is the easiest item.
    patterns = [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
    complete = np.array((patterns + [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]) * 3, dtype=float)
    data = complete.copy()
    data[5, 0] = np.nan  # a nonendorsing respondent skips the easiest item

    raw = guttman(data, na_rm=na_rm, normalize=False)

    np.testing.assert_array_equal(raw, np.zeros(len(data)))
    np.testing.assert_array_equal(raw, guttman(complete, na_rm=na_rm, normalize=False))


@pytest.mark.parametrize("na_rm", [False, True])
def test_scores_are_invariant_to_column_order_when_every_item_has_missing_responses(
    na_rm: bool,
) -> None:
    rng = np.random.default_rng(0)
    ability = rng.normal(size=(400, 1))
    difficulty = np.linspace(-1.5, 1.5, 8)
    data = (rng.random((400, 8)) < 1 / (1 + np.exp(-2 * (ability - difficulty)))).astype(float)
    data[np.arange(8), np.arange(8)] = np.nan
    assert len(np.unique(np.nanmean(data, axis=0))) == 8

    expected = guttman(data, na_rm=na_rm)
    for permutation in (np.arange(8)[::-1], rng.permutation(8)):
        np.testing.assert_array_equal(guttman(data[:, permutation], na_rm=na_rm), expected)


def test_na_rm_selects_only_the_normalization_denominator() -> None:
    data = np.array(
        [[1, 1, 0, np.nan], [0, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [np.nan, 1, 0, 1]]
    )
    raw = guttman(data, normalize=False)
    answered = np.array([3, 4, 4, 4, 3])

    np.testing.assert_array_equal(raw, [0.0, 2.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(guttman(data, na_rm=False, normalize=False), raw)
    np.testing.assert_array_equal(guttman(data, na_rm=False), raw / 6)
    np.testing.assert_array_equal(guttman(data), raw / (answered * (answered - 1) / 2))


@pytest.mark.parametrize(("n_rows", "n_items", "high"), [(30, 6, 5), (4, 400, 1000)])
@pytest.mark.parametrize("missing", [False, True])
def test_tied_item_means_follow_column_order_on_every_kernel(
    n_rows: int, n_items: int, high: int, missing: bool
) -> None:
    rng = np.random.default_rng(31)
    base = rng.integers(0, high, size=(n_rows, n_items)).astype(float)
    if missing:
        base[rng.random(base.shape) < 0.1] = np.nan
    # Reversing respondents keeps each copied item's mean exactly but changes its
    # responses, so every copied pair is a tie whose order changes the count.
    data = np.hstack([base, base[::-1]])
    if not missing:
        data = data.astype(np.int64)
    expected = _textbook_errors(data)
    assert len(np.unique(np.nanmean(data, axis=0))) <= n_items

    np.testing.assert_array_equal(guttman(data, normalize=False), expected)
    with patch("ier.guttman._MAX_CATEGORIES", 0):
        np.testing.assert_array_equal(guttman(data, normalize=False), expected)
    ordered = data[:, column_mean_order(data, ignore_nan=True)]
    categories = np.unique(ordered[~np.isnan(ordered)])
    for counts in (
        _count_categorical_errors(ordered, categories),
        _count_pairwise_errors(ordered),
        _count_merge_errors(ordered),
    ):
        np.testing.assert_array_equal(counts, expected)


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Return the Mann-Whitney probability that a positive outranks a negative."""
    above = scores[positive][:, None] - scores[~positive][None, :]
    return float(np.mean(above > 0) + 0.5 * np.mean(above == 0))


@pytest.mark.parametrize(
    ("missing_rate", "na_rm"), [(0.0, True), (0.01, True), (0.01, False), (0.05, False)]
)
def test_random_responders_score_higher_than_attentive_two_pl_responders(
    missing_rate: float, na_rm: bool
) -> None:
    rng = np.random.default_rng(20261004)
    n_items = 20
    ability = rng.normal(size=(1000, 1))
    discrimination = rng.uniform(0.8, 2.0, size=n_items)
    difficulty = rng.normal(size=n_items)
    endorse = 1.0 / (1.0 + np.exp(-discrimination * (ability - difficulty)))
    attentive = (rng.random(endorse.shape) < endorse).astype(float)
    random_responders = (rng.random((100, n_items)) < 0.5).astype(float)
    data = np.vstack([attentive, random_responders])
    careless = np.arange(len(data)) >= len(attentive)
    # Sparse missingness leaves a missing response in every item.
    data[rng.random(data.shape) < missing_rate] = np.nan

    assert _auc(guttman(data, na_rm=na_rm), careless) >= 0.90


@pytest.mark.parametrize("threshold", [None, np.nan, np.inf, -np.inf, True, "high"])
def test_flag_rejects_non_finite_thresholds(threshold: object) -> None:
    data = [[1, 1, 0], [1, 0, 0], [0, 1, 1]]
    with pytest.raises(ValueError, match="threshold must be a finite number"):
        guttman_flag(data, threshold=cast("Any", threshold))


def test_flag_compares_validated_thresholds_strictly() -> None:
    data = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
    scores = guttman(data)
    expected = [False, False, False, False, True]
    np.testing.assert_array_equal(guttman_flag(data, threshold=float(scores[-1])), [False] * 5)
    for threshold in (0, np.float32(0.5), np.int64(0), cast("Any", "0.5")):
        np.testing.assert_array_equal(guttman_flag(data, threshold=threshold), expected)
