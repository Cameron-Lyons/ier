"""Masked responses, temporal missing values, and probability options in the input contract."""

import re
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from ier import IndexOptions, irv, missing_rate, screen
from ier._registry import INDEX_REGISTRY
from ier._validation import validate_matrix_input, validate_probability

REAL_NUMERIC_MESSAGE = r"input data must contain real numeric responses \(got dtype "


def _sentinel_responses() -> tuple[np.ndarray, np.ndarray]:
    """Return integer responses coded -99 when missing and their NaN-coded twin."""
    rng = np.random.default_rng(11)
    raw = rng.integers(1, 6, size=(60, 12))
    raw[rng.random(raw.shape) < 0.08] = -99
    return raw, np.where(raw == -99, np.nan, raw.astype(float))


def test_masked_sentinel_cells_become_missing_responses() -> None:
    raw = np.array([[1, 2, 3, 4, 5, 4], [3, -99, 3, -99, 3, 3], [5, 4, 3, 2, 1, 2]])
    masked = np.ma.masked_equal(raw, -99)
    original = raw.copy()

    result = validate_matrix_input(masked)

    assert type(result) is np.ndarray
    assert result.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(result), raw == -99)
    np.testing.assert_array_equal(result[raw != -99], raw[raw != -99])
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(missing_rate(masked), [0.0, 1 / 3, 0.0])
    assert irv(masked)[1] == 0.0


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_masked_floating_responses_keep_their_precision(dtype: type) -> None:
    data = np.arange(1, 13, dtype=dtype).reshape(3, 4) / 3
    masked = np.ma.masked_array(data, mask=np.eye(3, 4, dtype=bool))

    result = validate_matrix_input(masked)

    assert result.dtype == data.dtype
    assert not np.shares_memory(result, data)
    expected = data.copy()
    expected[np.eye(3, 4, dtype=bool)] = np.nan
    np.testing.assert_array_equal(result, expected)
    assert not np.isnan(data).any()


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1, 2], [3, -99]], dtype=object),
        np.array([["1", "2"], ["3", "N/A"]]),
        np.array([[b"1", b"2"], [b"3", b"N/A"]]),
        np.array([[True, False], [True, True]]),
        np.array([[1, 2], [3, 2**62]], dtype=np.uint64),
    ],
    ids=["object", "text", "bytes", "bool", "uint64"],
)
def test_masked_object_text_and_integer_responses_become_float64(data: np.ndarray) -> None:
    mask = np.array([[False, False], [False, True]])

    result = validate_matrix_input(np.ma.masked_array(data, mask=mask))

    expected = np.asarray(np.where(mask, None, data).tolist(), dtype=np.float64)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)
    assert np.isnan(result[1, 1])


@pytest.mark.parametrize(
    "mask",
    [np.ma.nomask, np.zeros((3, 4), dtype=bool)],
    ids=["nomask", "no-masked-cells"],
)
@pytest.mark.parametrize("dtype", [np.int64, np.bool_, np.float32])
def test_masked_arrays_without_masked_cells_are_used_without_a_copy(mask: Any, dtype: type) -> None:
    data = (np.arange(12).reshape(3, 4) % 3).astype(dtype)

    result = validate_matrix_input(np.ma.masked_array(data, mask=mask))

    assert type(result) is np.ndarray
    assert result.dtype == data.dtype
    assert np.shares_memory(result, data)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1 + 1j, 2], [3, 4]]),
        np.zeros((2, 2), dtype="datetime64[D]"),
        np.zeros((2, 2), dtype="timedelta64[s]"),
        np.zeros((2, 2), dtype=[("a", np.float64)]),
        np.array([["one", "2"], ["3", "4"]]),
    ],
    ids=["complex", "datetime64", "timedelta64", "structured", "text"],
)
def test_masked_non_response_data_is_still_rejected(data: np.ndarray) -> None:
    mask = np.zeros(data.shape, dtype=bool)
    mask[1, 1] = True

    with pytest.raises(ValueError, match=REAL_NUMERIC_MESSAGE):
        validate_matrix_input(np.ma.masked_array(data, mask=mask))


def test_masked_responses_screen_every_index_like_nan_coded_responses() -> None:
    raw, nan_coded = _sentinel_responses()
    masked = np.ma.masked_equal(raw, -99)
    options = IndexOptions(
        evenodd_factors=[6, 6],
        mad_positive_items=[0, 1, 2],
        mad_negative_items=[3, 4, 5],
        semantic_item_pairs=[(0, 1), (2, 3)],
        infrequency_item_indices=[0, 1],
        infrequency_expected_responses=[1, 5],
        reliability_random_seed=0,
        scale_min=1,
        scale_max=5,
        onset_window_size=3,
        onset_min_items=6,
    )
    # Ht needs dichotomous items; tests/test_person_fit.py covers its masked input.
    names = sorted(name for name in INDEX_REGISTRY if name != "ht")

    result = screen(masked, names, options=options)
    reference = screen(nan_coded, names, options=options)

    assert result["errors"] == {}
    assert result["indices_used"] == reference["indices_used"] == names
    for name in names:
        np.testing.assert_array_equal(result["scores"][name], reference["scores"][name])
        np.testing.assert_array_equal(result["flags"][name], reference["flags"][name])
    np.testing.assert_array_equal(result["consensus_flags"], reference["consensus_flags"])
    assert not np.array_equal(
        screen(raw, ["missing_rate"])["scores"]["missing_rate"],
        result["scores"]["missing_rate"],
    )


def test_pandas_not_a_time_is_rejected_as_temporal_data() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1, 2, 3], "b": pd.Series([4, pd.NaT, 6], dtype=object)})

    for data in ([[1.0, pd.NaT], [2.0, 3.0]], frame):
        with pytest.raises(ValueError, match=REAL_NUMERIC_MESSAGE):
            validate_matrix_input(data)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (1, 1.0),
        (0.95, 0.95),
        (np.int64(1), 1.0),
        (np.uint8(0), 0.0),
        (np.float16(0.5), 0.5),
        (np.float32(0.25), 0.25),
        (np.longdouble(0.75), 0.75),
        (np.array(0.95), 0.95),
        (np.array(1), 1.0),
        (np.array(0.5, dtype=object), 0.5),
        (Decimal("0.95"), 0.95),
        (Fraction(19, 20), 0.95),
    ],
)
def test_probabilities_accept_real_numbers_and_return_python_floats(
    value: object, expected: float
) -> None:
    result = validate_probability(value, name="level")

    assert type(result) is float
    assert result == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.False_,
        np.array(True),
        "0.5",
        b"0.5",
        np.str_("0.5"),
        np.array("0.5"),
        None,
        [0.5],
        np.array([0.5]),
        0.5j,
        np.complex64(0.5),
        np.datetime64("2020-01-01"),
        np.timedelta64(1, "D"),
        object(),
    ],
)
def test_probabilities_reject_non_real_values_with_the_reason(value: object) -> None:
    prefix = re.escape("level must be between 0 and 1 (expected a real number, got ")

    with pytest.raises(ValueError, match=f"^{prefix}"):
        validate_probability(value, name="level")


@pytest.mark.parametrize(
    "value",
    [
        -1e-300,
        1 + 2**-52,
        np.nan,
        np.inf,
        -np.inf,
        np.float64(1.5),
        np.longdouble("1e400"),
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        10**400,
        Fraction(10**400, 7),
    ],
)
def test_probabilities_reject_non_finite_and_out_of_range_values(value: object) -> None:
    with pytest.raises(ValueError, match=r"^level must be between 0 and 1 \(got "):
        validate_probability(value, name="level")
