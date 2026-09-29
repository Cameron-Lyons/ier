"""Psychometric pair availability, bounded scoring, and cutoff validation."""

from unittest.mock import patch

import numpy as np
import pytest

from ier import IndexOptions, composite, psychant, psychsyn, screen
from ier.psychsyn import _compute_person_scores, psychsyn_critval, psychsyn_summary


@pytest.mark.parametrize("anto", [False, True])
@pytest.mark.parametrize("resample", [False, True])
@pytest.mark.parametrize("value", [np.nan, 3.0, np.inf])
def test_undefined_item_correlations_never_become_pairs(
    anto: bool, resample: bool, value: float
) -> None:
    data = np.full((5, 4), value)
    scores, counts, pairs = psychsyn(
        data, critval=0, anto=anto, resample_na=resample, _return_item_info=True
    )
    assert pairs.shape == (0, 2)
    assert np.isnan(scores).all()
    np.testing.assert_array_equal(counts, np.zeros(5))


@pytest.mark.parametrize("anto", [False, True])
@pytest.mark.parametrize("resample", [False, True])
def test_one_selected_pair_is_insufficient_for_a_respondent_correlation(
    anto: bool, resample: bool
) -> None:
    data = np.arange(10, dtype=float).reshape(5, 2)
    if anto:
        data[:, 1] *= -1
    scorer = psychant if anto else psychsyn
    scores, counts = scorer(data, diag=True, resample_na=resample)
    assert np.isnan(scores).all()
    # Diagnostics count observed pairs, even when they cannot support a correlation.
    np.testing.assert_array_equal(counts, np.ones(5))


@pytest.mark.parametrize(
    ("data", "pair_count"),
    [(np.full((3, 4), np.nan), 0), (np.arange(6).reshape(3, 2), 1)],
    ids=["undefined-items", "single-pair"],
)
def test_missing_scores_remain_unavailable_in_workflows_and_summary(
    data: np.ndarray, pair_count: int
) -> None:
    result = screen(
        data, indices=["psychsyn"], options=IndexOptions(psychsyn_critval=0), strict=True
    )
    np.testing.assert_array_equal(result["valid_index_counts"], [0, 0, 0])
    np.testing.assert_array_equal(result["flag_counts"], [0, 0, 0])
    assert np.isnan(
        composite(data, indices=["psychsyn"], options=IndexOptions(psychsyn_critval=0))
    ).all()
    summary = psychsyn_summary(data, critval=0)
    assert summary["item_pairs"] == pair_count
    assert summary["valid_individuals"] == 0
    assert summary["missing_individuals"] == 3
    assert all(
        np.isnan(summary[f"{name}_score"]) for name in ("mean", "std", "min", "max", "median")
    )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(True),
        "0.6",
        0.6j,
        pytest.param(10**400, id="overflow"),
    ],
)
def test_cutoffs_must_be_finite_real_numbers(value: object) -> None:
    data = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    with pytest.raises(ValueError, match="critval.*finite"):
        psychsyn(data, critval=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_correlation.*finite"):
        psychsyn_critval(data, min_correlation=value)  # type: ignore[arg-type]


def test_cutoffs_accept_numpy_numbers_and_discovery_requires_nonnegative_magnitude() -> None:
    data = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    np.testing.assert_array_equal(
        psychsyn(data, critval=np.float32(0.5)), psychsyn(data, critval=0.5)
    )
    assert psychsyn_critval(data, min_correlation=np.int64(0)) == psychsyn_critval(data)
    with pytest.raises(ValueError, match="nonnegative"):
        psychsyn_critval(data, min_correlation=-0.1)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("resample", [False, True])
def test_selected_pairs_preserve_missing_and_resampling_policies(
    layout: str, resample: bool
) -> None:
    rng = np.random.default_rng(111)
    data = rng.normal(size=(19, 7))
    data[0] = np.nan
    data[1, 0] = np.nan
    data[2, 1] = np.inf
    data[3, 5] = np.nan  # An unselected item must not invalidate this row.
    data[4] = 3
    data = data[::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    pairs = np.array([[0, 1], [2, 3], [4, 6], [1, 4]])
    reference = np.full(len(data), np.nan)
    counts = np.zeros(len(data), dtype=int)
    for row_index, row in enumerate(data):
        left, right = row[pairs[:, 0]], row[pairs[:, 1]]
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            continue
        reference[row_index] = (
            0.0 if np.std(left) == 0 or np.std(right) == 0 else np.corrcoef(left, right)[0, 1]
        )
        counts[row_index] = len(pairs)
    if resample:
        missing = np.isnan(reference)
        average = abs(np.mean(reference[~missing]))
        signs = np.random.default_rng(7).choice([-1, 1], size=(int(missing.sum()), len(pairs)))
        reference[missing] = signs.mean(axis=1) * average
        counts[missing] = len(pairs)
    for budget in (3, 71):
        with patch("ier.psychsyn._PSYCHSYN_BATCH_ELEMENTS", budget):
            actual, actual_counts = _compute_person_scores(
                data, pairs, resample_na=resample, rng=np.random.default_rng(7)
            )
        np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(actual_counts, counts)
    np.testing.assert_array_equal(data, original)


def test_pair_scoring_never_scans_the_full_input_for_finiteness() -> None:
    data = np.random.default_rng(13).normal(size=(50, 17))
    data[0, 15] = np.nan  # Not selected for scoring.
    pairs = np.array([[0, 1], [2, 3], [4, 5]])
    with (
        patch("ier.psychsyn._PSYCHSYN_BATCH_ELEMENTS", 100),
        patch("ier.psychsyn.np.isfinite", wraps=np.isfinite) as checks,
    ):
        _compute_person_scores(data, pairs, resample_na=False, rng=np.random.default_rng(7))
    checked_matrices = [
        call.args[0]
        for call in checks.call_args_list
        if isinstance(call.args[0], np.ndarray) and call.args[0].ndim == 2
    ]
    assert checked_matrices
    assert all(matrix.size <= 100 for matrix in checked_matrices)
