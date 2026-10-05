"""Active-row compaction in the lz ability solver preserves every estimate exactly."""

import builtins
import math
from unittest.mock import patch

import numpy as np
import pytest

from ier import lz
from ier._statistics import logistic_transform
from ier.lz import (
    _dichotomize,
    _estimate_theta,
    _initial_scaled_theta,
    _ml_theta_batch,
    _scaled_theta_steps,
)

_PREVIOUS_THETA_BATCH_ELEMENTS = 10_240


def _reference_ml_theta_batch(
    responses: np.ndarray, a: np.ndarray, b: np.ndarray, *, na_rm: bool = True
) -> np.ndarray:
    """Apply safeguarded Newton iterations to one response batch."""
    theta = np.full(len(responses), np.nan)
    missing = np.isnan(responses)
    observed = ~missing if na_rm and np.any(missing) else None
    if observed is None:
        available = ~np.any(missing, axis=1)
        all_correct = np.all(responses == 1, axis=1)
        all_incorrect = np.all(responses == 0, axis=1)
    else:
        available = np.any(observed, axis=1)
        all_correct = available & np.all((responses == 1) | missing, axis=1)
        all_incorrect = available & np.all((responses == 0) | missing, axis=1)
    theta[all_correct] = 3.0
    theta[all_incorrect] = -3.0

    interior = available & ~(all_correct | all_incorrect)
    active_responses = responses[interior]
    if len(active_responses) == 0:
        return theta

    valid = True if observed is None else observed[interior]
    proportion = np.clip(np.mean(active_responses, axis=1, where=valid), 0.01, 0.99)
    estimates = np.clip(np.log(proportion / (1.0 - proportion)), -4.0, 4.0)
    lower = np.full(len(active_responses), -4.0)
    upper = np.full(len(active_responses), 4.0)
    active = np.ones(len(active_responses), dtype=bool)
    magnitude = np.max(np.abs(a), where=~np.isnan(a), initial=0.0)
    scaled = magnitude > math.sqrt(np.finfo(float).max / len(a)) / 4
    if scaled:
        estimates = _initial_scaled_theta(active_responses, a, b, estimates, valid)

    # Saturated predictors and tiny moments are valid finite-model limits.
    # Unrepresentable Newton steps fall back to the safeguarded bracket.
    with np.errstate(over="ignore", under="ignore"):
        a_squared = None if scaled else a**2
        for _ in range(64):
            linear_predictor = a * (estimates[:, None] - b)
            probabilities = logistic_transform(linear_predictor)

            if scaled:
                score, steps, step_scales = _scaled_theta_steps(
                    active_responses, probabilities, a, valid
                )
                with np.errstate(divide="ignore"):
                    tolerance = 1e-12 / np.minimum(step_scales, 1.0)
            else:
                score = np.sum(a * (active_responses - probabilities), axis=1, where=valid)
                tolerance = 1e-12
            score_converged = active & (np.abs(score) <= tolerance)
            active[score_converged] = False
            if not np.any(active):
                break

            positive = active & (score > 0.0)
            negative = active & ~positive
            lower[positive] = estimates[positive]
            upper[negative] = estimates[negative]

            if not scaled:
                assert a_squared is not None
                information = np.sum(
                    a_squared * probabilities * (1.0 - probabilities), axis=1, where=valid
                )
                steps = np.divide(
                    score,
                    information,
                    out=np.full(len(active_responses), np.nan),
                    where=information > 0.0,
                )
            candidates = estimates + steps
            invalid = ~np.isfinite(candidates) | (candidates <= lower) | (candidates >= upper)
            candidates[invalid] = (lower[invalid] + upper[invalid]) / 2.0

            changes = np.abs(candidates - estimates)
            if scaled:
                changes *= np.maximum(step_scales, 1.0)
            step_converged = active & (changes <= 1e-12)
            estimates[active] = candidates[active]
            active[step_converged] = False
            if not np.any(active):
                break

    theta[interior] = estimates
    return theta


def _reference_estimate_theta(
    data: np.ndarray, a: np.ndarray, b: np.ndarray, *, na_rm: bool = True
) -> np.ndarray:
    """Run the previous solver at its previous batch size."""
    with (
        patch("ier.lz._ml_theta_batch", _reference_ml_theta_batch),
        patch("ier.lz._LZ_THETA_BATCH_ELEMENTS", _PREVIOUS_THETA_BATCH_ELEMENTS),
    ):
        return _estimate_theta(data, a, b, na_rm=na_rm)


def _two_parameter_data(
    rng: np.random.Generator, n_rows: int, n_items: int, *, missing_rate: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 2PL responses, with boundary and saturated rows in every block."""
    a = rng.uniform(0.2, 2.0, n_items)
    b = rng.normal(size=n_items)
    ability = rng.normal(size=n_rows)
    data = (rng.random((n_rows, n_items)) < logistic_transform(a * (ability[:, None] - b))).astype(
        float
    )
    # One in twenty responses against the trend puts the root outside [-4, 4]
    # for modest slopes, so these rows bisect toward a bracket bound.
    for row in range(0, n_rows, 97):
        data[row] = 0.0
        data[row, rng.choice(n_items, size=max(1, n_items // 20), replace=False)] = 1.0
    for row in range(1, n_rows, 89):
        data[row] = 1.0
        data[row, rng.choice(n_items, size=max(1, n_items // 20), replace=False)] = 0.0
    data[2::211] = 1.0
    data[3::223] = 0.0
    if missing_rate:
        data[rng.random(data.shape) < missing_rate] = np.nan
        data[4] = np.nan
    return data, a, b


def _layout(data: np.ndarray, layout: str) -> np.ndarray:
    if layout == "strided":
        backing = np.zeros((2 * len(data), 2 * data.shape[1]))
        backing[::2, ::2] = data
        return backing[::2, ::2]
    return np.array(data, order=layout)


@pytest.mark.parametrize("missing_rate", [0.0, 0.05])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("na_rm", [False, True])
def test_two_parameter_estimates_are_bit_identical(
    missing_rate: float, layout: str, na_rm: bool
) -> None:
    rng = np.random.default_rng(4093)
    data, a, b = _two_parameter_data(rng, 1500, 60, missing_rate=missing_rate)
    data = _layout(data, layout)
    expected = _reference_estimate_theta(data, a, b, na_rm=na_rm)
    actual = _estimate_theta(data, a, b, na_rm=na_rm)
    assert np.array_equal(actual, expected, equal_nan=True)
    if na_rm or not missing_rate:
        # The fixture must include roots that bisect toward a bracket bound.
        assert np.count_nonzero(np.abs(np.abs(actual) - 4.0) < 1e-9) >= 10


def test_batches_with_and_without_missing_responses_are_bit_identical() -> None:
    # Only the first previous batch had missing responses; the rest used
    # unmasked reductions, while every new batch mixes both kinds of rows.
    rng = np.random.default_rng(613)
    data, a, b = _two_parameter_data(rng, 6000, 60)
    data[:60][rng.random((60, 60)) < 0.1] = np.nan
    expected = _reference_estimate_theta(data, a, b)
    actual = _estimate_theta(data, a, b)
    assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("correct", [3, 57])
def test_roots_outside_the_bracket_are_bit_identical(correct: int) -> None:
    rng = np.random.default_rng(correct)
    data = np.zeros((40, 60))
    for row in data:
        row[rng.choice(60, size=correct, replace=False)] = 1.0
    data[::7] = rng.integers(0, 2, (len(data[::7]), 60))
    a = rng.uniform(0.3, 0.6, 60)
    b = rng.normal(scale=0.5, size=60)
    expected = _reference_ml_theta_batch(data, a, b)
    actual = _ml_theta_batch(data, a, b)
    assert np.array_equal(actual, expected, equal_nan=True)
    bound = 4.0 if correct > 30 else -4.0
    assert np.count_nonzero(np.abs(actual - bound) < 1e-9) >= 30


@pytest.mark.parametrize("na_rm", [False, True])
def test_saturated_and_unavailable_rows_are_bit_identical(na_rm: bool) -> None:
    data = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, np.nan, 1, 1, 1],
            [0, 0, np.nan, 0, 0],
            [np.nan] * 5,
            [1, 0, 1, 0, 1],
            [1, 0, np.nan, 0, 0],
        ]
    )
    a = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    b = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    expected = _reference_ml_theta_batch(data, a, b, na_rm=na_rm)
    actual = _ml_theta_batch(data, a, b, na_rm=na_rm)
    assert np.array_equal(actual, expected, equal_nan=True)
    only_saturated = data[:5]
    assert np.array_equal(
        _ml_theta_batch(only_saturated, a, b, na_rm=na_rm),
        _reference_ml_theta_batch(only_saturated, a, b, na_rm=na_rm),
        equal_nan=True,
    )


@pytest.mark.parametrize("scale", [1e155, 1e200, np.finfo(float).max])
@pytest.mark.parametrize("missing_rate", [0.0, 0.1])
@pytest.mark.parametrize("negative", [False, True])
def test_scaled_path_is_bit_identical(scale: float, missing_rate: float, negative: bool) -> None:
    rng = np.random.default_rng(77)
    data, coefficients, b = _two_parameter_data(rng, 400, 12, missing_rate=missing_rate)
    if negative:
        coefficients[::3] *= -1.0
    a = coefficients / np.max(np.abs(coefficients)) * scale
    b = b / scale
    expected = _reference_estimate_theta(data, a, b)
    with np.errstate(all="raise"):
        actual = _estimate_theta(data, a, b)
    assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("missing_rate", [0.0, 0.05])
def test_one_parameter_estimates_are_bit_identical(missing_rate: float) -> None:
    rng = np.random.default_rng(1031)
    data, _, b = _two_parameter_data(rng, 1200, 60, missing_rate=missing_rate)
    a = np.ones(60)
    expected = _reference_estimate_theta(data, a, b)
    actual = _estimate_theta(data, a, b)
    assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("layout", ["C", "F"])
def test_dichotomized_likert_estimates_are_bit_identical(layout: str) -> None:
    rng = np.random.default_rng(509)
    likert = rng.integers(1, 6, (2000, 40)).astype(float)
    likert[rng.random(likert.shape) < 0.1] = np.nan
    data = _dichotomize(_layout(likert, layout))
    a = rng.uniform(0.2, 3.0, 40)
    b = rng.normal(size=40)
    expected = _reference_estimate_theta(data, a, b)
    actual = _estimate_theta(data, a, b)
    assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("missing_rate", [0.0, 0.05])
@pytest.mark.parametrize("na_rm", [False, True])
def test_reduced_precision_estimates_are_bit_identical(
    dtype: type[np.floating], missing_rate: float, na_rm: bool
) -> None:
    # Binary float32/float16 inputs start from estimates in their own precision,
    # and the previous solver rounded every Newton iterate back to it.
    rng = np.random.default_rng(8191)
    data, a, b = _two_parameter_data(rng, 1500, 60, missing_rate=missing_rate)
    data = data.astype(dtype)
    expected = _reference_estimate_theta(data, a, b, na_rm=na_rm)
    actual = _estimate_theta(data, a, b, na_rm=na_rm)
    assert np.array_equal(actual, expected, equal_nan=True)
    rounded = actual[np.isfinite(actual)]
    assert rounded.size
    assert np.array_equal(rounded.astype(dtype).astype(float), rounded)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_reduced_precision_batches_with_and_without_missing_are_bit_identical(
    dtype: type[np.floating],
) -> None:
    rng = np.random.default_rng(6143)
    data, a, b = _two_parameter_data(rng, 6000, 60)
    data[:60][rng.random((60, 60)) < 0.1] = np.nan
    data = data.astype(dtype)
    expected = _reference_estimate_theta(data, a, b)
    actual = _estimate_theta(data, a, b)
    assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("model", ["1pl", "2pl"])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_reduced_precision_public_scores_are_bit_identical(
    dtype: type[np.floating], model: str, layout: str
) -> None:
    rng = np.random.default_rng(3571)
    data, _, _ = _two_parameter_data(rng, 2000, 40, missing_rate=0.05)
    data = _layout(data, layout).astype(dtype)
    with (
        patch("ier.lz._ml_theta_batch", _reference_ml_theta_batch),
        patch("ier.lz._LZ_THETA_BATCH_ELEMENTS", _PREVIOUS_THETA_BATCH_ELEMENTS),
    ):
        expected = lz(data, model=model)
    actual = lz(data, model=model)
    assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_reduced_precision_scaled_path_is_bit_identical(dtype: type[np.floating]) -> None:
    rng = np.random.default_rng(97)
    data, coefficients, b = _two_parameter_data(rng, 400, 12, missing_rate=0.1)
    a = coefficients / np.max(np.abs(coefficients)) * 1e200
    data = data.astype(dtype)
    expected = _reference_estimate_theta(data, a, b / 1e200)
    actual = _estimate_theta(data, a, b / 1e200)
    assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("batch_elements", [7, 53])
@pytest.mark.parametrize("missing_rate", [0.0, 0.2])
def test_theta_batch_boundaries_are_bit_identical(batch_elements: int, missing_rate: float) -> None:
    rng = np.random.default_rng(batch_elements)
    data, a, b = _two_parameter_data(rng, 457, 13, missing_rate=missing_rate)
    expected = _reference_estimate_theta(data, a, b)
    with (
        patch("ier.lz._LZ_THETA_BATCH_ELEMENTS", batch_elements),
        patch("ier.lz._ml_theta_batch", wraps=_ml_theta_batch) as solve,
    ):
        actual = _estimate_theta(data, a, b)
    assert np.array_equal(actual, expected, equal_nan=True)
    assert all(call.args[0].size <= max(batch_elements, 13) for call in solve.call_args_list)


def test_theta_batches_use_their_own_element_budget() -> None:
    data = np.tile([1.0, 0.0, 1.0, 0.0], (300, 25))
    with patch("ier.lz._ml_theta_batch", wraps=_ml_theta_batch) as solve:
        _estimate_theta(data, np.ones(100), np.zeros(100))
    assert solve.call_count == 1
    with (
        patch("ier.lz._LZ_THETA_BATCH_ELEMENTS", 1000),
        patch("ier.lz._ml_theta_batch", wraps=_ml_theta_batch) as solve,
    ):
        _estimate_theta(data, np.ones(100), np.zeros(100))
    assert solve.call_count == 30


def test_resolved_rows_leave_the_newton_workspace() -> None:
    rng = np.random.default_rng(4093)
    data, a, b = _two_parameter_data(rng, 1000, 60)
    evaluated_rows = []

    def counting_transform(values: np.ndarray) -> np.ndarray:
        evaluated_rows.append(len(values))
        return logistic_transform(values)

    with patch("ier.lz.logistic_transform", counting_transform):
        theta = _ml_theta_batch(data, a, b)
    interior = np.count_nonzero(np.abs(theta) != 3.0)
    # Bracket-bound rows bisect for dozens of iterations, but only they remain.
    assert len(evaluated_rows) > 30
    assert evaluated_rows[-1] < 10
    assert sum(evaluated_rows) < 8 * interior


@pytest.mark.parametrize("model", ["1pl", "2pl"])
@pytest.mark.parametrize("missing_rate", [0.0, 0.05])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_public_scores_are_bit_identical(model: str, missing_rate: float, layout: str) -> None:
    rng = np.random.default_rng(2718)
    data, _, _ = _two_parameter_data(rng, 3000, 60, missing_rate=missing_rate)
    data = _layout(data, layout)
    with (
        patch("ier.lz._ml_theta_batch", _reference_ml_theta_batch),
        patch("ier.lz._LZ_THETA_BATCH_ELEMENTS", _PREVIOUS_THETA_BATCH_ELEMENTS),
    ):
        expected = lz(data, model=model)
    actual = lz(data, model=model)
    assert np.array_equal(actual, expected, equal_nan=True)


def test_iteration_budget_keeps_last_candidates() -> None:
    data = np.array([[1, 0, 1, 0, 1], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=float)
    a = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    b = np.zeros(5)

    def two_iterations(_: int) -> range:
        return builtins.range(2)

    # Rows still moving when the iteration budget ends keep their last candidates.
    with patch("ier.lz.range", two_iterations, create=True):
        actual = _ml_theta_batch(data, a, b)
    with patch(f"{__name__}.range", two_iterations, create=True):
        expected = _reference_ml_theta_batch(data, a, b)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, _ml_theta_batch(data, a, b))
