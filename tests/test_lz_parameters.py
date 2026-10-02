"""Public calibrated-model validation prevents item/respondent axis corruption."""

import math
from unittest.mock import patch

import numpy as np
import pytest

from ier import lz, lz_flag

DATA = np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
PARAMETERS = {
    "difficulty": [-1.0, 0.0, 1.0],
    "discrimination": [0.5, 1.0, 1.5],
    "theta": [-0.3, 0.0, 0.7],
}


@pytest.mark.parametrize("name", ["difficulty", "discrimination", "theta"])
@pytest.mark.parametrize(
    "invalid",
    [0.0, np.zeros((3, 1)), np.zeros((1, 3)), np.zeros((3, 3)), np.zeros((3, 1, 1))],
    ids=["scalar", "column-vector", "row-vector", "matrix", "three-dimensional"],
)
def test_parameter_dimensions_are_rejected_before_numerical_broadcast(
    name: str, invalid: np.ndarray | float
) -> None:
    parameters = dict(PARAMETERS, **{name: invalid})
    with (
        patch("ier.lz._estimate_theta") as estimate,
        patch("ier.lz._compute_lz") as compute,
        pytest.raises(ValueError, match=f"{name} must be one-dimensional"),
    ):
        lz(DATA, **parameters)
    estimate.assert_not_called()
    compute.assert_not_called()


@pytest.mark.parametrize("name", ["difficulty", "discrimination", "theta"])
@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([True, False, True]),
        np.asarray(["-1", "0", "1"]),
        np.asarray([b"-1", b"0", b"1"]),
        np.asarray([1 + 0j, 2 + 0j, 3 + 0j]),
        np.asarray([1 + 1j, 2 + 0j, 3 + 0j]),
        np.asarray([-1, 0, 1], dtype=object),
        np.asarray([1, 2, 3], dtype="datetime64[s]"),
        np.asarray([1, 2, 3], dtype="timedelta64[s]"),
    ],
    ids=["boolean", "unicode", "bytes", "complex-real", "complex", "object", "date", "duration"],
)
def test_parameters_reject_lossy_or_non_real_types(name: str, invalid: np.ndarray) -> None:
    parameters = dict(PARAMETERS, **{name: invalid})
    with pytest.raises(ValueError, match=f"{name} must be a one-dimensional real numeric array"):
        lz(DATA, **parameters)


@pytest.mark.parametrize("name", ["difficulty", "discrimination", "theta"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nonfinite_parameters_require_nan_instead_of_infinity(name: str, value: float) -> None:
    parameters = dict(PARAMETERS, **{name: [0.0, value, 1.0]})
    with pytest.raises(ValueError, match=f"{name} must contain only finite values or NaN"):
        lz(DATA, **parameters)


@pytest.mark.parametrize("name", ["difficulty", "discrimination", "theta"])
def test_parameter_count_must_match_the_correct_axis(name: str) -> None:
    parameters = dict(PARAMETERS, **{name: [0.0, 1.0]})
    axis = "respondents" if name == "theta" else "items"
    with pytest.raises(ValueError, match=f"{name} length must match number of {axis}"):
        lz(DATA, **parameters)


@pytest.mark.parametrize("name", ["difficulty", "discrimination", "theta"])
@pytest.mark.parametrize("invalid", [[], [[0.0], [0.0, 1.0]]], ids=["empty", "ragged"])
def test_empty_and_ragged_parameters_have_public_validation_errors(
    name: str, invalid: list
) -> None:
    parameters = dict(PARAMETERS, **{name: invalid})
    with pytest.raises(ValueError, match=name):
        lz(DATA, **parameters)


def _expected_scores(parameters: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate the calibrated Bernoulli likelihood independently, item by item."""
    scores = []
    for row, ability in zip(DATA, parameters["theta"], strict=True):
        centered = 0.0
        variance = 0.0
        for response, difficulty, discrimination in zip(
            row, parameters["difficulty"], parameters["discrimination"], strict=True
        ):
            odds = float(discrimination) * (float(ability) - float(difficulty))
            probability = 1 / (1 + math.exp(-odds))
            centered += (float(response) - probability) * odds
            variance += probability * (1 - probability) * odds * odds
        scores.append(centered / math.sqrt(variance))
    return np.asarray(scores)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("container", ["list", "tuple", "array", "strided"])
def test_real_parameter_vectors_preserve_calibrated_scores_and_inputs(
    dtype: type, container: str
) -> None:
    if dtype is np.uint64:
        parameters = {name: np.arange(1, 4, dtype=dtype) for name in PARAMETERS}
    else:
        parameters = {name: np.asarray(values, dtype=dtype) for name, values in PARAMETERS.items()}
    expected = _expected_scores(parameters)
    originals = {name: values.copy() for name, values in parameters.items()}
    supplied = {}
    for name, values in parameters.items():
        if container == "list":
            supplied[name] = values.tolist()
        elif container == "tuple":
            supplied[name] = tuple(values.tolist())
        else:
            prepared = values
            if container == "strided":
                backing = np.zeros(2 * len(values), dtype=dtype)
                backing[::2] = values
                prepared = backing[::2]
            prepared.flags.writeable = False
            supplied[name] = prepared
    scores, flags = lz_flag(DATA, **supplied, threshold=-0.3)
    np.testing.assert_allclose(scores, expected, rtol=3e-15, atol=1e-15)
    np.testing.assert_array_equal(flags, expected < -0.3)
    for name, values in supplied.items():
        np.testing.assert_array_equal(values, originals[name])


def test_nan_parameters_retain_item_and_respondent_availability() -> None:
    data = np.asarray([[1, np.nan, 0], [1, 0, 1], [0, np.nan, 1], [np.nan] * 3])
    scores, flags = lz_flag(
        data,
        difficulty=[-1, np.nan, 1],
        discrimination=[1, np.nan, 1],
        theta=[0.0, 0.0, np.nan, 0.0],
    )
    assert math.isfinite(scores[0])
    assert np.isnan(scores[1:]).all()
    assert not flags.any()
    expected = lz([[1, 0]], difficulty=[-1, 1], discrimination=[1, 1], theta=[0.0])
    np.testing.assert_array_equal(scores[:1], expected)


def test_rasch_model_keeps_discrimination_ignored() -> None:
    expected = lz(DATA, model="1pl", difficulty=PARAMETERS["difficulty"], theta=PARAMETERS["theta"])
    actual = lz(
        DATA,
        model="1pl",
        difficulty=PARAMETERS["difficulty"],
        theta=PARAMETERS["theta"],
        discrimination=np.ones((3, 1), dtype=complex),
    )
    np.testing.assert_array_equal(actual, expected)
