"""Psychometric pair availability, bounded scoring, and cutoff validation."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ier import (
    IndexOptions,
    composite,
    psychant,
    psychant_flag,
    psychsyn,
    psychsyn_flag,
    screen,
)
from ier._column_statistics import pairwise_column_correlations
from ier.cli import main
from ier.psychsyn import (
    _compute_person_scores,
    get_highly_correlated_pairs,
    psychsyn_critval,
    psychsyn_summary,
)


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
def test_selected_pairs_preserve_the_missing_response_policy(layout: str) -> None:
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
    for budget in (3, 71):
        with patch("ier.psychsyn._PSYCHSYN_BATCH_ELEMENTS", budget):
            actual, actual_counts = _compute_person_scores(data, pairs, pairwise=False)
        np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(actual_counts, counts)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("pairwise", [False, True])
def test_pair_scoring_never_scans_the_full_input_for_finiteness(pairwise: bool) -> None:
    data = np.random.default_rng(13).normal(size=(50, 17))
    data[0, 15] = np.nan  # Not selected for scoring.
    data[1, 2] = np.nan  # Selected, so pairwise scoring counts the answered pairs.
    pairs = np.array([[0, 1], [2, 3], [4, 5]])
    with (
        patch("ier.psychsyn._PSYCHSYN_BATCH_ELEMENTS", 100),
        patch("ier.psychsyn.np.isfinite", wraps=np.isfinite) as checks,
    ):
        _compute_person_scores(data, pairs, pairwise=pairwise)
    checked_matrices = [
        call.args[0]
        for call in checks.call_args_list
        if isinstance(call.args[0], np.ndarray) and call.args[0].ndim == 2
    ]
    assert checked_matrices
    assert all(matrix.size <= 100 for matrix in checked_matrices)


def _one_factor_likert(rows: int, items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rows, 1))
    return np.clip(np.rint(3 + latent + rng.normal(scale=0.7, size=(rows, items))), 1, 5)


def _with_missing(data: np.ndarray, rate: float, seed: int) -> np.ndarray:
    missing = np.array(data, dtype=float)
    missing[np.random.default_rng(seed).random(missing.shape) < rate] = np.nan
    return missing


def _answered_pair_oracle(data: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correlate each respondent's answered pairs, as careless::psychsyn does."""
    scores = np.full(len(data), np.nan)
    counts = np.zeros(len(data), dtype=int)
    for row_index, row in enumerate(data):
        left, right = row[pairs[:, 0]], row[pairs[:, 1]]
        answered = np.isfinite(left) & np.isfinite(right)
        counts[row_index] = np.count_nonzero(answered)
        if counts[row_index] < 2:
            continue
        left, right = left[answered], right[answered]
        scores[row_index] = (
            0.0 if np.ptp(left) == 0 or np.ptp(right) == 0 else np.corrcoef(left, right)[0, 1]
        )
    return scores, counts


def test_complete_item_correlations_remain_the_default() -> None:
    data = _with_missing(_one_factor_likert(300, 10, seed=1), 0.01, seed=2)
    np.testing.assert_array_equal(psychsyn(data), psychsyn(data, item_correlations="complete"))
    assert psychsyn_critval(data) == psychsyn_critval(data, item_correlations="complete")


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("anto", [False, True])
def test_pairwise_mode_changes_nothing_without_missing_responses(dtype: type, anto: bool) -> None:
    data = _one_factor_likert(500, 12, seed=3).astype(dtype)
    if anto:
        data[:, ::2] = 6 - data[:, ::2]
    critval = -0.3 if anto else 0.3
    complete = psychsyn(data, critval=critval, anto=anto, _return_item_info=True)
    pairwise = psychsyn(
        data, critval=critval, anto=anto, _return_item_info=True, item_correlations="pairwise"
    )
    assert len(complete[2])
    for expected, actual in zip(complete, pairwise, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_pairwise_discovery_scores_respondents_with_sparse_missingness() -> None:
    data = _with_missing(_one_factor_likert(2000, 20, seed=4), 0.01, seed=5)
    assert np.isnan(psychsyn(data)).all()
    scores, counts = psychsyn(data, diag=True, item_correlations="pairwise")
    assert np.isfinite(scores).mean() >= 0.95
    assert np.median(counts) > 100


def test_pairwise_scores_use_each_respondents_answered_pairs() -> None:
    data = _with_missing(_one_factor_likert(400, 12, seed=6), 0.15, seed=7)
    data[0] = np.nan
    data[1, 1:] = np.nan
    data[2] = 3.0
    scores, counts, pairs = psychsyn(
        data, critval=0.4, _return_item_info=True, item_correlations="pairwise"
    )
    np.testing.assert_array_equal(
        pairs, get_highly_correlated_pairs(pairwise_column_correlations(data), 0.4, anto=False)
    )
    expected_scores, expected_counts = _answered_pair_oracle(data, pairs)
    np.testing.assert_allclose(scores, expected_scores, rtol=0.0, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(counts, expected_counts)
    assert np.isnan(scores[:2]).all()
    assert scores[2] == 0.0
    assert np.isfinite(scores[3:]).mean() > 0.95


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_pairwise_batches_match_the_answered_pair_oracle(layout: str) -> None:
    rng = np.random.default_rng(112)
    data = rng.normal(size=(23, 7))
    data[rng.random(data.shape) < 0.25] = np.nan
    data[0] = np.nan
    data[1, 0:3] = [0.5, -0.5, np.inf]  # Infinite responses leave a row unavailable.
    data[4] = 3
    data = data[::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    pairs = np.array([[0, 1], [2, 3], [4, 6], [1, 4], [5, 6]])
    reference, counts = _answered_pair_oracle(data, pairs)
    reference[np.isinf(data).any(axis=1)] = np.nan
    for budget in (3, 71, 10_000):
        with patch("ier.psychsyn._PSYCHSYN_BATCH_ELEMENTS", budget):
            actual, actual_counts = _compute_person_scores(data, pairs, pairwise=True)
        np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(actual_counts, counts)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("n_pairs", [1, 2])
def test_too_few_answered_pairs_leave_pairwise_scores_unavailable(n_pairs: int) -> None:
    data = np.array(
        [
            [1.0, 2.0, 3.0, 5.0],
            [1.0, np.nan, 3.0, 5.0],
            [4.0, 1.0, 1.0, 2.0],
            [np.nan, np.nan, 2.0, 2.0],
            [1.0, 2.0, np.inf, 5.0],
        ]
    )
    pairs = np.array([[0, 1], [2, 3]])[:n_pairs]
    scores, counts = _compute_person_scores(data, pairs, pairwise=True)
    answered = np.isfinite(data[:, pairs[:, 0]]) & np.isfinite(data[:, pairs[:, 1]])
    np.testing.assert_array_equal(counts, answered.sum(axis=1))
    expected = [1.0, np.nan, -1.0, np.nan, np.nan] if n_pairs == 2 else [np.nan] * 5
    np.testing.assert_array_equal(scores, expected)


def test_resampling_never_invents_scores_for_unanswered_pairwise_rows() -> None:
    data = _with_missing(_one_factor_likert(300, 10, seed=8), 0.05, seed=9)
    data[:5, 1:] = np.nan
    data[5] = np.nan
    plain, plain_counts = psychsyn(data, diag=True, item_correlations="pairwise")
    assert np.isnan(plain[:6]).all()
    assert (plain_counts[:6] < 2).all()

    resampled, resampled_counts = psychsyn(
        data, diag=True, resample_na=True, random_seed=3, item_correlations="pairwise"
    )

    # Respondents without two answered pairs supplied nothing to resample, so
    # they stay unavailable, as in careless; answered rows are unchanged.
    np.testing.assert_array_equal(resampled, plain)
    np.testing.assert_array_equal(resampled_counts, plain_counts)
    result = screen(
        data, indices=["psychsyn"], options=IndexOptions(psychsyn_item_correlations="pairwise")
    )
    assert np.isnan(result["scores"]["psychsyn"][:6]).all()
    assert not result["flags"]["psychsyn"][:6].any()


def test_antonym_and_cutoff_helpers_accept_pairwise_correlations() -> None:
    data = _with_missing(_one_factor_likert(500, 10, seed=10), 0.02, seed=11)
    data[:, ::2] = 6 - data[:, ::2]
    expected = psychsyn(data, critval=-0.5, anto=True, diag=True, item_correlations="pairwise")
    actual = psychant(data, critval=-0.5, diag=True, item_correlations="pairwise")
    for actual_values, expected_values in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_values, expected_values)
    assert np.isfinite(actual[0]).mean() > 0.95
    assert np.isnan(psychant(data, critval=-0.5)).all()

    listed = psychsyn_critval(data, anto=True, min_correlation=0.5, item_correlations="pairwise")
    correlations = pairwise_column_correlations(data)
    assert listed and all(correlations[i, j] == r and abs(r) >= 0.5 for i, j, r in listed)
    assert [r for _, _, r in listed] == sorted(r for _, _, r in listed)
    assert psychsyn_critval(data, anto=True, min_correlation=0.5) == []

    summary = psychsyn_summary(data, critval=-0.5, anto=True, item_correlations="pairwise")
    assert summary["item_pairs"] == sum(r <= -0.5 for _, _, r in listed)
    assert summary["valid_individuals"] == np.count_nonzero(np.isfinite(actual[0]))
    assert psychsyn_summary(data, critval=-0.5, anto=True)["item_pairs"] == 0


@pytest.mark.parametrize("mode", ["Pairwise", "pairwise ", "", None, 1, ("pairwise",)])
def test_item_correlations_must_name_a_mode(mode: object) -> None:
    data = [[1, 2, 3], [2, 3, 4], [3, 4, 6], [5, 5, 5]]
    calls = [
        lambda: psychsyn(data, item_correlations=mode),  # type: ignore[arg-type]
        lambda: psychant(data, item_correlations=mode),  # type: ignore[arg-type]
        lambda: psychsyn_critval(data, item_correlations=mode),  # type: ignore[arg-type]
        lambda: psychsyn_summary(data, item_correlations=mode),  # type: ignore[arg-type]
    ]
    for call in calls:
        with pytest.raises(ValueError, match="item_correlations must be 'complete' or 'pairwise'"):
            call()


@pytest.mark.parametrize("na_rm", [True, False])
def test_workflows_score_missing_responses_with_the_pairwise_option(na_rm: bool) -> None:
    data = _with_missing(_one_factor_likert(1000, 30, seed=12), 0.02, seed=13)
    assert np.isnan(screen(data, indices=["psychsyn"])["scores"]["psychsyn"]).all()

    options = IndexOptions(na_rm=na_rm, psychsyn_item_correlations="pairwise")
    result = screen(data, indices=["psychsyn"], options=options, strict=True)
    assert np.isfinite(result["scores"]["psychsyn"]).mean() >= 0.95
    assert np.isfinite(composite(data, indices=["psychsyn"], options=options)).mean() >= 0.95

    antonyms = data.copy()
    antonyms[:, ::2] = 6 - antonyms[:, ::2]
    scores = screen(antonyms, indices=["psychant"], options=options, strict=True)["scores"]
    assert np.isfinite(scores["psychant"]).mean() >= 0.95


def test_invalid_registry_option_is_a_soft_index_error() -> None:
    data = _one_factor_likert(50, 6, seed=14)
    options = IndexOptions(psychsyn_item_correlations="both")  # type: ignore[arg-type]
    result = screen(data, indices=["psychsyn", "irv"], options=options)
    assert result["errors"] == {"psychsyn": "item_correlations must be 'complete' or 'pairwise'"}


def test_cli_selects_pairwise_item_correlations(tmp_path: Path) -> None:
    data = _with_missing(_one_factor_likert(200, 12, seed=15), 0.02, seed=16)
    path = tmp_path / "responses.npy"
    np.save(path, data)
    available = {}
    for mode in (None, "complete", "pairwise"):
        output = tmp_path / f"{mode}.json"
        option = [] if mode is None else ["--psychsyn-item-correlations", mode]
        arguments = ["screen", str(path), "--indices", "psychsyn", "--format", "json"]
        assert main([*arguments, *option, "--output", str(output)]) == 0
        scores = json.loads(output.read_text(encoding="utf-8"))["scores"]["psychsyn"]
        available[mode] = sum(score is not None for score in scores)
    assert available == {None: 0, "complete": 0, "pairwise": 200}


def test_screen_flags_inconsistent_antonym_responders() -> None:
    rng = np.random.default_rng(21)
    trait = rng.normal(size=(600, 1))
    loadings = np.where(np.arange(20) % 2 == 0, 1.0, -1.0)
    latent = trait * loadings + rng.normal(scale=0.6, size=(600, 20))
    attentive = np.clip(np.round(3 + 1.2 * latent), 1, 5)
    careless = rng.integers(1, 6, size=(60, 20)).astype(float)
    data = np.vstack([attentive, careless])

    flags = screen(data, indices=["psychant"], strict=True)["flags"]["psychant"]
    scores = psychant(data)

    assert np.nanmean(scores[:600]) < np.nanmean(scores[600:])
    # Careless respondents are flagged at more than twice the attentive rate;
    # the former low direction flagged attentive respondents only.
    assert flags[600:].mean() > 2 * flags[:600].mean()


def _mixed_respondents(mode: str = "pairwise") -> np.ndarray:
    data = _one_factor_likert(400, 12, seed=17)
    data[:, 6:] = 6 - data[:, 6:]  # Antonym pairs for psychant.
    data[:20] = 3.0  # Straight-liners have zero within-pair variance.
    if mode == "complete":
        # Complete discovery skips items with gaps, but their rows are still scored.
        data[20:40, [0, 6]] = np.nan
        return data
    data = _with_missing(data, 0.05, seed=18)
    data[20:25, 1:] = np.nan  # Fewer than two answered pairs.
    data[25] = np.nan
    data[26:30, ::2] = np.nan
    return data


@pytest.mark.parametrize("mode", ["complete", "pairwise"])
def test_resample_na_and_random_seed_never_change_scores(mode: str) -> None:
    data = _mixed_respondents(mode)
    calls = [
        (psychsyn, {"critval": 0.4, "diag": True}),
        (psychant, {"critval": -0.4, "diag": True}),
        (psychsyn_flag, {"critval": 0.4}),
        (psychant_flag, {"critval": -0.4}),
    ]
    for function, arguments in calls:
        plain = function(data, item_correlations=mode, **arguments)
        # No generator is built, because no score is ever left to resample.
        with patch("ier.psychsyn.np.random.default_rng", side_effect=AssertionError):
            for seed in (None, 0, 3):
                resampled = function(
                    data, item_correlations=mode, resample_na=True, random_seed=seed, **arguments
                )
                for expected, actual in zip(plain, resampled, strict=True):
                    np.testing.assert_array_equal(actual, expected)
    scores, counts = psychsyn(data, critval=0.4, diag=True, item_correlations=mode)
    assert counts.max() > 2
    np.testing.assert_array_equal(scores[:20], 0.0)
    if mode == "complete":
        assert np.isfinite(scores).all()
    else:
        assert np.isnan(scores[20:26]).all()


def test_na_rm_does_not_change_psychometric_scores_in_workflows() -> None:
    data = _mixed_respondents()
    results = [
        screen(
            data,
            indices=["psychsyn", "psychant"],
            options=IndexOptions(na_rm=na_rm, psychsyn_item_correlations="pairwise"),
            strict=True,
        )["scores"]
        for na_rm in (True, False)
    ]
    for name in ("psychsyn", "psychant"):
        np.testing.assert_array_equal(results[0][name], results[1][name])
        assert np.isfinite(results[0][name]).mean() > 0.9


def test_summary_reports_shared_respondent_count_keys() -> None:
    data = _mixed_respondents()
    summary = psychsyn_summary(data, critval=0.4, item_correlations="pairwise")
    scores = psychsyn(data, critval=0.4, item_correlations="pairwise")
    valid = int(np.count_nonzero(np.isfinite(scores)))
    assert (summary["n_total"], summary["n_valid"], summary["n_missing"]) == (
        len(data),
        valid,
        len(data) - valid,
    )
    assert summary["n_missing"] == summary["missing_individuals"] > 0
    assert summary["n_total"] == summary["total_individuals"]
    assert summary["n_valid"] == summary["valid_individuals"]


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_pairwise_discovery_needs_three_respondents_without_missing_responses(
    dtype: type,
) -> None:
    two = np.array([[1, 2, 4, 5], [3, 5, 1, 2]]).astype(dtype)
    assert psychsyn_critval(two, item_correlations="pairwise") == []
    with_blank = np.vstack([two, np.full(4, np.nan)])
    assert psychsyn_critval(with_blank, item_correlations="pairwise") == []
    scores, counts = psychsyn(two, critval=0.0, diag=True, item_correlations="pairwise")
    assert np.isnan(scores).all()
    np.testing.assert_array_equal(counts, [0, 0])
