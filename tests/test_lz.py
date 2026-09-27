"""Person-fit regression tests with independent scalar reference calculations."""

import unittest
from unittest.mock import patch

import numpy as np
import pytest

from ier._statistics import logistic_transform
from ier.lz import _compute_lz, _compute_lz_batch, _estimate_theta, _ml_theta_batch, lz, lz_flag


def _reference_theta(responses: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Estimate theta with safeguarded Newton iterations on the score equation."""
    lower = -4.0
    upper = 4.0
    proportion = float(np.clip(np.mean(responses), 0.01, 0.99))
    theta = float(np.clip(np.log(proportion / (1.0 - proportion)), lower, upper))

    for _ in range(64):
        linear_predictor = a * (theta - b)
        probabilities = logistic_transform(linear_predictor)

        score = float(np.sum(a * (responses - probabilities)))
        if abs(score) <= 1e-12:
            return theta

        if score > 0.0:
            lower = theta
        else:
            upper = theta

        information = float(np.sum(a**2 * probabilities * (1.0 - probabilities)))
        candidate = theta + score / information if information > 0.0 else np.nan
        if not np.isfinite(candidate) or not lower < candidate < upper:
            candidate = (lower + upper) / 2.0

        if abs(candidate - theta) <= 1e-12:
            return float(candidate)
        theta = float(candidate)

    return theta


def _reference_lz_row(
    responses: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    theta: float,
) -> float:
    """Compute one lz score for the missing-data fallback path."""
    prob = logistic_transform(a * (theta - b))
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    log_l = np.sum(responses * np.log(prob) + (1 - responses) * np.log(1 - prob))
    expected_l = np.sum(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))
    log_odds = np.log(prob / (1 - prob))
    var_l = np.sum(prob * (1 - prob) * log_odds**2)
    return 0.0 if var_l <= 0 else float((log_l - expected_l) / np.sqrt(var_l))


class TestLz(unittest.TestCase):
    """Tests for standardized log-likelihood (lz) functions."""

    def test_basic_functionality(self) -> None:
        """Test basic lz calculation."""
        data = [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        result = lz(data)
        self.assertEqual(len(result), 3)

    def test_normal_pattern_positive_lz(self) -> None:
        """Test that normal response patterns get non-negative lz."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]
        result = lz(data)
        for score in result:
            self.assertGreater(score, -2.0)

    def test_aberrant_pattern_negative_lz(self) -> None:
        """Test that aberrant patterns tend toward negative lz."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        lz_scores = lz(data)
        self.assertGreater(lz_scores[0], lz_scores[3])

    def test_1pl_model(self) -> None:
        """Test lz with 1PL (Rasch) model."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        result = lz(data, model="1pl")
        self.assertEqual(len(result), 2)

    def test_2pl_model(self) -> None:
        """Test lz with 2PL model (default)."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        result = lz(data, model="2pl")
        self.assertEqual(len(result), 2)

    def test_custom_parameters(self) -> None:
        """Test lz with user-specified item parameters."""
        data = [[1, 1, 0, 0], [0, 1, 1, 0]]
        difficulty = [-1.0, -0.5, 0.5, 1.0]
        discrimination = [1.0, 1.0, 1.0, 1.0]
        result = lz(data, difficulty=difficulty, discrimination=discrimination)
        self.assertEqual(len(result), 2)

    def test_custom_theta(self) -> None:
        """Test lz with user-specified theta values."""
        data = [[1, 1, 0, 0], [0, 1, 1, 0]]
        theta = [0.0, 0.5]
        result = lz(data, theta=theta)
        self.assertEqual(len(result), 2)

    def test_extreme_item_parameters_do_not_overflow(self) -> None:
        data = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        difficulty = np.array([-1000.0, 1000.0, -1000.0, 1000.0])
        discrimination = np.ones(4)
        theta = np.zeros(2)

        complete = lz(
            data,
            difficulty=difficulty,
            discrimination=discrimination,
            theta=theta,
        )
        data[0, 0] = np.nan
        missing = lz(
            data,
            difficulty=difficulty,
            discrimination=discrimination,
            theta=theta,
        )

        self.assertTrue(np.isfinite(complete).all())
        self.assertTrue(np.isfinite(missing).all())

    def test_local_theta_solver_matches_scipy_reference_values(self) -> None:
        cases = [
            ([1, 1, 0, 0], [1, 1, 1, 1], [-1, -0.5, 0.5, 1]),
            ([1, 0, 1, 0], [0.5, 1, 1.5, 2], [-2, -0.5, 0.5, 2]),
            ([1, 1, 1, 0, 0], [3, 0.2, 1.2, 2, 0.7], [-3, -1, 0, 1, 3]),
            ([0, 1, 0, 1, 1], [1, -0.5, 2, 1.5, 0.2], [-2, -1, 0, 1, 2]),
        ]
        expected = [0.0, 0.5359047366047544, 0.3859716606867468, -0.6358889853624815]
        actual = [
            _reference_theta(
                np.asarray(responses), np.asarray(discrimination), np.asarray(difficulty)
            )
            for responses, discrimination, difficulty in cases
        ]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6)

        for theta, (responses, discrimination, difficulty) in zip(actual, cases, strict=True):
            response_array = np.asarray(responses)
            discrimination_array = np.asarray(discrimination)
            difficulty_array = np.asarray(difficulty)
            probabilities = 1.0 / (1.0 + np.exp(-discrimination_array * (theta - difficulty_array)))
            score = np.sum(discrimination_array * (response_array - probabilities))
            self.assertAlmostEqual(float(score), 0.0, places=11)

    def test_flag_function(self) -> None:
        """Test lz flagging."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        scores, flags = lz_flag(data, threshold=-1.5)
        self.assertEqual(len(flags), 2)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )

    def test_flag_with_custom_threshold(self) -> None:
        """Test lz flagging with custom threshold."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        scores, flags = lz_flag(data, threshold=0.0)
        self.assertEqual(len(flags), 2)

    def test_with_nan(self) -> None:
        """Test handling of missing values."""
        data = [[1, 1, np.nan, 0], [1, 0, 1, 0]]
        result = lz(data, na_rm=True)
        self.assertEqual(len(result), 2)

    def test_complete_batch_kernels_match_scalar_rows(self) -> None:
        """Batched complete-data kernels preserve exact scalar results."""
        rng = np.random.default_rng(29)
        for n_items in (4, 5, 17, 80):
            with self.subTest(n_items=n_items):
                data = rng.integers(0, 2, size=(257, n_items)).astype(float)
                data[0] = 0.0
                data[1] = 1.0
                discrimination = rng.uniform(0.2, 3.0, n_items)
                difficulty = rng.uniform(-3.0, 3.0, n_items)

                with patch("ier.lz._LZ_BATCH_ELEMENTS", 512):
                    theta = _estimate_theta(data, discrimination, difficulty)
                    scores = _compute_lz(
                        data,
                        discrimination,
                        difficulty,
                        theta,
                    )

                expected_theta = np.array(
                    [
                        -3.0
                        if np.all(row == 0)
                        else 3.0
                        if np.all(row == 1)
                        else _reference_theta(row, discrimination, difficulty)
                        for row in data
                    ]
                )
                expected_scores = np.array(
                    [
                        _reference_lz_row(row, discrimination, difficulty, row_theta)
                        for row, row_theta in zip(data, expected_theta, strict=True)
                    ]
                )
                np.testing.assert_array_equal(theta, expected_theta)
                np.testing.assert_array_equal(scores, expected_scores)

    def test_all_correct_responses(self) -> None:
        """Test handling of all correct responses."""
        data = [[1, 1, 1, 1], [0, 0, 0, 0]]
        result = lz(data)
        self.assertEqual(len(result), 2)
        self.assertFalse(np.isnan(result[0]))

    def test_polytomous_dichotomization(self) -> None:
        """Test that polytomous data is dichotomized."""
        data = [[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        result = lz(data)
        self.assertEqual(len(result), 2)

    def test_invalid_model_raises(self) -> None:
        """Test that invalid model raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, model="invalid")

    def test_mismatched_difficulty_raises(self) -> None:
        """Test that mismatched difficulty length raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, difficulty=[1.0, 2.0])

    def test_mismatched_discrimination_raises(self) -> None:
        """Test that mismatched discrimination length raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, discrimination=[1.0, 2.0])

    def test_mismatched_theta_raises(self) -> None:
        """Test that mismatched theta length raises ValueError."""
        data = [[1, 1, 0, 0], [0, 0, 1, 1]]
        with self.assertRaises(ValueError):
            lz(data, theta=[0.0])


def _reference_rows(
    data: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    na_rm: bool,
    theta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score each respondent independently with the original scalar equations."""
    abilities = np.full(len(data), np.nan)
    scores = np.full(len(data), np.nan)
    for i, row in enumerate(data):
        valid = ~np.isnan(row) if na_rm else np.ones(len(row), dtype=bool)
        responses = row[valid]
        if len(responses) == 0:
            continue
        row_a = a[valid]
        row_b = b[valid]
        if theta is not None:
            abilities[i] = theta[i]
        elif np.all(responses == 1):
            abilities[i] = 3.0
        elif np.all(responses == 0):
            abilities[i] = -3.0
        else:
            abilities[i] = _reference_theta(responses, row_a, row_b)
        if not np.isnan(abilities[i]):
            scores[i] = _reference_lz_row(responses, row_a, row_b, abilities[i])
    return abilities, scores


@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("n_items", [1, 5, 17, 80])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_missing_batches_match_scalar_rows(na_rm: bool, n_items: int, layout: str) -> None:
    rng = np.random.default_rng(731)
    data = rng.integers(0, 2, (259, n_items)).astype(float)
    data[rng.random(data.shape) < 0.3] = np.nan
    data[0] = 0.0
    data[1] = 1.0
    data[2] = np.nan
    data[3, :-1] = np.nan
    data[3, -1] = 0.0
    data[4, :-1] = np.nan
    data[4, -1] = 1.0
    if layout == "strided":
        backing = np.empty((2 * len(data), 2 * n_items))
        backing[::2, ::2] = data
        data = backing[::2, ::2]
    else:
        data = np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    a = rng.uniform(0.2, 3.0, n_items)
    b = rng.uniform(-3.0, 3.0, n_items)
    expected_theta, expected_scores = _reference_rows(data, a, b, na_rm=na_rm)

    # Cross many batch boundaries, including inputs wider than the element budget.
    with patch("ier.lz._LZ_BATCH_ELEMENTS", 53):
        actual_theta = _estimate_theta(data, a, b, na_rm=na_rm)
        actual_scores = _compute_lz(data, a, b, actual_theta, na_rm=na_rm)
    np.testing.assert_allclose(actual_theta, expected_theta, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(actual_scores, expected_scores, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("model", ["1pl", "2pl"])
@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("provided_theta", [False, True])
def test_public_missing_scoring_and_flags(model: str, na_rm: bool, provided_theta: bool) -> None:
    rng = np.random.default_rng(301)
    data = rng.integers(0, 2, (91, 13)).astype(float)
    data[rng.random(data.shape) < 0.15] = np.nan
    data[0] = np.nan
    a = rng.uniform(0.2, 3.0, data.shape[1])
    b = rng.normal(size=data.shape[1])
    theta = rng.normal(size=len(data)) if provided_theta else None
    if theta is not None:
        theta[1] = np.nan
    _, expected = _reference_rows(
        data, a if model == "2pl" else np.ones(len(a)), b, na_rm=na_rm, theta=theta
    )
    scores, flags = lz_flag(
        data, difficulty=b, discrimination=a, theta=theta, model=model, na_rm=na_rm
    )
    np.testing.assert_allclose(scores, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(flags, expected < -1.96)


@pytest.mark.parametrize("na_rm", [False, True])
def test_zero_likelihood_variance_and_unavailable_rows(na_rm: bool) -> None:
    data = np.array([[1, 0, 1], [0, np.nan, 1], [np.nan, np.nan, np.nan]])
    expected = [0.0, 0.0, np.nan if na_rm else 0.0]
    scores = lz(data, difficulty=np.zeros(3), theta=np.zeros(3), model="1pl", na_rm=na_rm)
    np.testing.assert_array_equal(scores, expected)


def test_missing_item_parameters_only_affect_observed_items() -> None:
    data = np.array([[1, 0, np.nan], [0, 1, 1], [np.nan, np.nan, np.nan]])
    a = np.array([1.0, 1.0, np.nan])
    b = np.array([-1.0, 1.0, np.nan])
    theta = np.zeros(3)
    _, expected = _reference_rows(data, a, b, na_rm=True, theta=theta)
    actual = lz(data, difficulty=b, discrimination=a, theta=theta)
    np.testing.assert_allclose(actual, expected)


def test_unavailable_parameters_propagate_for_complete_rows() -> None:
    data = np.array([[1, 0], [0, 1]])
    scores, flags = lz_flag(data, difficulty=[0, np.nan], theta=[0, 0], model="1pl")
    assert np.isnan(scores).all()
    assert not flags.any()


def test_missing_theta_solver_workspace_is_bounded() -> None:
    rng = np.random.default_rng(513)
    data = rng.integers(0, 2, (300, 13)).astype(float)
    data[::3, 0] = np.nan
    with (
        patch("ier.lz._LZ_BATCH_ELEMENTS", 53),
        patch("ier.lz._ml_theta_batch", wraps=_ml_theta_batch) as solve,
        patch("ier.lz._compute_lz_batch", wraps=_compute_lz_batch) as score,
    ):
        theta = _estimate_theta(data, np.ones(13), np.zeros(13))
        _compute_lz(data, np.ones(13), np.zeros(13), theta)
    assert solve.call_count == 75
    assert score.call_count == 75
    assert all(call.args[0].size <= 53 for call in solve.call_args_list)
    assert all(call.args[0].size <= 53 for call in score.call_args_list)


@pytest.mark.parametrize("a_value", [0.0, 1e-15, 1.0, -1.0])
@pytest.mark.parametrize("b_value", [0.0, -1000.0, 1000.0])
def test_missing_solver_handles_extreme_parameters(a_value: float, b_value: float) -> None:
    data = np.array([[1, np.nan, 0, 1], [0, 1, np.nan, 0], [0, 1, 0, 1]], dtype=float)
    a = np.full(4, a_value)
    b = np.full(4, b_value)
    expected_theta, expected_scores = _reference_rows(data, a, b, na_rm=True)
    actual_theta = _estimate_theta(data, a, b)
    actual_scores = _compute_lz(data, a, b, actual_theta)
    np.testing.assert_allclose(actual_theta, expected_theta, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(actual_scores, expected_scores, atol=2e-11, rtol=2e-11)
