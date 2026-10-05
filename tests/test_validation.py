"""Tests for shared matrix input validation."""

import re
import unittest
from collections import UserList
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from ier import IndexOptions, composite, irv, missing_rate, psychsyn, screen
from ier._registry import INDEX_REGISTRY, validate_min_valid_indices, validate_worker_count
from ier._validation import (
    validate_column_index,
    validate_integer,
    validate_item_indices,
    validate_matrix_input,
    validate_score_vectors,
)

REAL_NUMERIC_MESSAGE = r"input data must contain real numeric responses \(got dtype "
ARRAY_LIKE_MESSAGE = "input data must be array-like (list, tuple, numpy array, or DataFrame)"


class TestValidateMatrixInput(unittest.TestCase):
    def test_matching_numpy_dtype_is_reused(self) -> None:
        data = np.arange(12, dtype=float).reshape(3, 4)

        result = validate_matrix_input(data, dtype=float)

        self.assertIs(result, data)

    def test_mismatched_numpy_dtype_is_converted(self) -> None:
        data = np.arange(12, dtype=np.int64).reshape(3, 4)

        result = validate_matrix_input(data, dtype=float)

        self.assertIsNot(result, data)
        self.assertEqual(result.dtype, np.dtype(float))
        np.testing.assert_array_equal(result, data)


def _assert_float_matrix(result: np.ndarray, expected: Any) -> None:
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.float64))


def _nullable_frames() -> dict[str, Any]:
    pd = pytest.importorskip("pandas")
    values = np.arange(1.0, 25.0).reshape(6, 4)
    missing = values.copy()
    missing[2, 1] = np.nan
    fractional = missing / 4
    with_na = pd.DataFrame(missing).astype("Int64")
    assert with_na.isna().to_numpy().sum() == 1
    return {
        "convert_dtypes": pd.DataFrame(values).convert_dtypes(),
        "Int64 with pd.NA": with_na,
        "Float64 with pd.NA": pd.DataFrame(fractional).astype("Float64"),
        "boolean with pd.NA": pd.DataFrame(
            {
                "a": pd.array([True, None, False, True, False, True], dtype="boolean"),
                "b": [1, 2, 3, 4, 5, 6],
            }
        ),
        "object column": pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": ["1", "2.5", None, "4", "5", "6"]}
        ),
        "string column with pd.NA": pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": ["1", "2.5", None, "4", "5", "6"]}
        ).convert_dtypes(),
    }


@pytest.mark.parametrize(
    "name",
    [
        "convert_dtypes",
        "Int64 with pd.NA",
        "Float64 with pd.NA",
        "boolean with pd.NA",
        "object column",
        "string column with pd.NA",
    ],
)
def test_pandas_extension_frames_become_float64_responses(name: str) -> None:
    frame = _nullable_frames()[name]
    expected = frame.to_numpy(dtype=float, na_value=np.nan)
    assert np.asarray(frame).dtype == object

    result = validate_matrix_input(frame)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "string"])
def test_nullable_series_is_one_respondent_when_one_dimensional_input_is_allowed(
    dtype: str,
) -> None:
    pd = pytest.importorskip("pandas")
    series = pd.Series(["1", None, "3", "4"]).astype(dtype)

    result = validate_matrix_input(series, allow_1d=True)

    _assert_float_matrix(result, [[1.0, np.nan, 3.0, 4.0]])


@pytest.mark.parametrize(
    ("column", "dtype"),
    [
        (["2020-01-01", "2020-01-02", "2020-01-03"], "datetime64[ns]"),
        ([1, 2, 3], "timedelta64[ns]"),
        ([1 + 1j, 2, 3], "complex128"),
        (["one", "two", None], "object"),
    ],
)
def test_pandas_frames_with_non_response_columns_are_rejected(column: list, dtype: str) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1, 2, 3], "b": pd.Series(column).astype(dtype)})

    with pytest.raises(ValueError, match=REAL_NUMERIC_MESSAGE):
        validate_matrix_input(frame)


def test_lists_containing_pandas_missing_values_become_nan() -> None:
    pd = pytest.importorskip("pandas")

    result = validate_matrix_input([[1, pd.NA, 3], [None, 5, "6"]])

    _assert_float_matrix(result, [[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]])


class _FrameLike:
    """Frame-like object whose ``to_numpy`` does not follow the pandas keywords."""

    def __init__(self, values: np.ndarray, dtypes: Any, converted: Any = None) -> None:
        self.values = values
        self.dtypes = dtypes
        self.converted = converted

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self.values

    def to_numpy(self, **kwargs: Any) -> np.ndarray:
        if self.converted is None:
            raise TypeError("to_numpy() got an unexpected keyword argument 'dtype'")
        return np.asarray(self.converted)


@pytest.mark.parametrize(
    ("dtypes", "converted"),
    [
        ([np.dtype(np.int64)] * 2, None),
        ([np.dtype(np.int64)] * 2, np.zeros((3, 2))),
        ([], np.zeros((2, 2))),
        (None, np.zeros((2, 2))),
    ],
    ids=["incompatible-to-numpy", "mismatched-shape", "no-columns", "no-dtypes"],
)
def test_frame_like_objects_fall_back_to_elementwise_conversion(
    dtypes: Any, converted: Any
) -> None:
    values = np.array([[1, None], ["2", Decimal("2.5")]], dtype=object)

    result = validate_matrix_input(_FrameLike(values, dtypes, converted))

    _assert_float_matrix(result, [[1.0, np.nan], [2.0, 2.5]])


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([[1, None, 3], [4, 5, None]], [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]]),
        (
            np.array([[1, None], [np.float32(2.5), True]], dtype=object),
            [[1.0, np.nan], [2.5, 1.0]],
        ),
        (np.array([["1", "2.5"], ["nan", "-4"]]), [[1.0, 2.5], [np.nan, -4.0]]),
        (np.array([[b"1", b"2.5"], [b"3", b"4"]]), [[1.0, 2.5], [3.0, 4.0]]),
        (
            np.array([[Decimal("1.5"), Fraction(1, 4)], [np.int64(3), np.bool_(False)]]),
            [[1.5, 0.25], [3.0, 0.0]],
        ),
    ],
    ids=["nested-lists-with-none", "object-array", "numeric-text", "numeric-bytes", "rational"],
)
def test_python_object_and_text_responses_become_float64(data: Any, expected: Any) -> None:
    _assert_float_matrix(validate_matrix_input(data), expected)


@pytest.mark.skipif(not hasattr(np.dtypes, "StringDType"), reason="requires NumPy 2 strings")
def test_numpy_variable_width_strings_follow_the_text_rule() -> None:
    string_dtype = np.dtypes.StringDType()

    result = validate_matrix_input(np.array([["1", "2.5"], ["nan", "4"]], dtype=string_dtype))

    _assert_float_matrix(result, [[1.0, 2.5], [np.nan, 4.0]])
    with pytest.raises(ValueError, match=REAL_NUMERIC_MESSAGE):
        validate_matrix_input(np.array([["one", "2"]], dtype=string_dtype))


def test_object_inputs_are_converted_before_a_requested_dtype() -> None:
    result = validate_matrix_input([[1, None], [3, 4]], dtype=float)

    _assert_float_matrix(result, [[1.0, np.nan], [3.0, 4.0]])


@pytest.mark.parametrize("dtype", [np.int64, np.uint8, np.bool_, np.float32, np.float64])
def test_real_numeric_arrays_are_returned_without_a_copy(dtype: type) -> None:
    data = (np.arange(12).reshape(3, 4) % 2).astype(dtype)

    result = validate_matrix_input(data)

    assert result.dtype == data.dtype
    assert np.shares_memory(result, data)


def _object_matrix(value: object) -> np.ndarray:
    matrix = np.empty((2, 2), dtype=object)
    matrix[:] = 1.0
    matrix[1, 0] = value
    return matrix


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1 + 2j, 3]]),
        np.array([[1, 2]], dtype=np.complex64),
        np.zeros((2, 2), dtype="datetime64[D]"),
        np.zeros((2, 2), dtype="timedelta64[s]"),
        np.zeros((2, 2), dtype=[("a", np.float64)]),
        np.array([["one", "2"], ["3", "4"]]),
        [[1, "two"], [3, 4]],
        _object_matrix(1 + 2j),
        _object_matrix(np.complex128(1)),
        _object_matrix(np.datetime64("2020-01-01")),
        _object_matrix(np.timedelta64(3, "D")),
        _object_matrix(object()),
        _object_matrix(np.array(1.0)),
        _object_matrix(10**400),
        _object_matrix(Decimal("sNaN")),
    ],
    ids=[
        "complex",
        "complex64",
        "datetime64",
        "timedelta64",
        "structured",
        "text",
        "mixed-text",
        "object-complex",
        "object-numpy-complex",
        "object-datetime64",
        "object-timedelta64",
        "object-arbitrary",
        "object-array",
        "object-overflowing-integer",
        "object-signaling-nan",
    ],
)
def test_non_real_numeric_inputs_are_rejected_with_one_message(data: Any) -> None:
    with pytest.raises(ValueError, match=REAL_NUMERIC_MESSAGE):
        validate_matrix_input(data)


@pytest.mark.parametrize("scorer", [irv, psychsyn, screen, composite])
def test_complex_inputs_are_rejected_by_public_entry_points(scorer: Callable) -> None:
    data = np.arange(1, 13).reshape(3, 4) + 1j

    with pytest.raises(ValueError, match=REAL_NUMERIC_MESSAGE):
        scorer(data)


@pytest.mark.parametrize(
    "data",
    ["1,2,3", b"1,2,3", {"a": [1, 2]}, 5, 2.5, {1, 2}, (value for value in range(3))],
    ids=["str", "bytes", "dict", "int", "float", "set", "generator"],
)
@pytest.mark.parametrize("scorer", [validate_matrix_input, irv, psychsyn, screen])
def test_non_array_inputs_raise_type_error_everywhere(scorer: Callable, data: Any) -> None:
    with pytest.raises(TypeError, match=re.escape(ARRAY_LIKE_MESSAGE)) as caught:
        scorer(data)

    # Callers that previously caught ValueError for these inputs still do.
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (None, "input data cannot be None"),
        ([], "input data cannot be empty"),
        (np.empty((0, 3)), "input data cannot be empty"),
        ([[]], "input data cannot be empty"),
        ([1, 2, 3], "input data must be 2-dimensional"),
        (np.ones((2, 2, 2)), "input data must be 2-dimensional"),
        (np.array([[None, "1"], ["2", 3]], dtype=object)[:, :1], "at least 2 columns"),
    ],
)
def test_shape_failures_keep_their_value_errors(data: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message) as caught:
        validate_matrix_input(data, min_columns=2)

    assert not isinstance(caught.value, TypeError)


_USER_LIST_OPTIONS = IndexOptions(
    evenodd_factors=[4, 4],
    mad_positive_items=[0, 1],
    mad_negative_items=[2, 3],
    semantic_item_pairs=[(0, 1), (2, 3)],
    infrequency_item_indices=[0, 1],
    infrequency_expected_responses=[1, 5],
    reliability_random_seed=0,
    scale_min=1,
    scale_max=5,
)


@pytest.mark.parametrize("name", sorted(INDEX_REGISTRY))
def test_user_list_matrices_are_accepted_by_every_registered_index(name: str) -> None:
    data = np.random.default_rng(7).integers(1, 6, size=(30, 8))
    if name == "ht":  # Ht is defined for dichotomous items only.
        data = (data > 3).astype(np.int64)
    user_list = UserList(UserList(row) for row in data.tolist())
    scorer = INDEX_REGISTRY[name].scorer

    expected = scorer(data, _USER_LIST_OPTIONS)
    actual = scorer(user_list, _USER_LIST_OPTIONS)

    np.testing.assert_array_equal(actual, expected)


class _Index:
    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


@pytest.mark.parametrize("value", [np.int64(3), np.uint8(3), np.array(3), _Index(3), 3])
def test_validate_integer_accepts_integer_protocol_values(value: object) -> None:
    result = validate_integer(value, message="count must be an integer", minimum=1)

    assert result == 3
    assert type(result) is int


@pytest.mark.parametrize("value", [True, np.True_, np.array(True), 2.0, "2", None, [2]])
def test_validate_integer_rejects_booleans_and_non_integers(value: object) -> None:
    with pytest.raises(ValueError) as caught:
        validate_integer(value, message="count must be an integer")

    assert str(caught.value) == "count must be an integer"


def test_validate_integer_reports_minimum_failures() -> None:
    with pytest.raises(ValueError) as caught:
        validate_integer(np.int64(1), message="count must be an integer", minimum=2)
    assert str(caught.value) == "count must be an integer"

    with pytest.raises(ValueError) as caught:
        validate_integer(
            1,
            message="count must be an integer",
            minimum=2,
            minimum_message="count must be at least 2",
        )
    assert str(caught.value) == "count must be at least 2"


def test_validate_column_index_reports_type_and_bounds_failures() -> None:
    assert validate_column_index(np.array(2), 3, name="items") == 2

    with pytest.raises(ValueError) as caught:
        validate_column_index(np.True_, 3, name="items")
    assert str(caught.value) == "items must contain integer column indices"

    with pytest.raises(ValueError) as caught:
        validate_column_index(np.int64(3), 3, name="items")
    assert str(caught.value) == "item index 3 out of bounds for data with 3 columns"


def test_item_indices_accept_integer_protocol_values_and_detect_their_duplicates() -> None:
    result = validate_item_indices([np.array(2), _Index(0), np.int8(1)], 3)

    assert result.dtype == np.intp
    assert result.tolist() == [2, 0, 1]
    with pytest.raises(ValueError, match="item_indices cannot contain duplicates"):
        validate_item_indices([np.array(1), 1], 3)
    with pytest.raises(ValueError, match="item_indices must contain integer column indices"):
        validate_item_indices([1.0], 3)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "items cannot be empty"),
        ([1, 1], "items cannot contain duplicates"),
        ([1.0], "items must contain integer column indices"),
    ],
)
def test_item_index_errors_name_the_callers_parameter(items: list[object], message: str) -> None:
    with pytest.raises(ValueError) as caught:
        validate_item_indices(items, 3, name="items")  # type: ignore[arg-type]
    assert str(caught.value) == message
    with pytest.raises(ValueError, match=message.replace("items", "item_indices")):
        validate_item_indices(items, 3)  # type: ignore[arg-type]


def test_integer_options_accept_numpy_integers() -> None:
    data = np.random.default_rng(3).integers(1, 6, size=(20, 6)).astype(float)
    data[0, 1] = np.nan
    reference = screen(data, min_flags=2, workers=1, min_valid_indices=1)

    result = screen(
        data,
        min_flags=np.int64(2),
        workers=np.int64(1),
        min_valid_indices=np.int64(1),
    )

    assert result["min_flags"] == 2
    assert type(result["min_flags"]) is int
    assert type(result["min_valid_indices"]) is int
    np.testing.assert_array_equal(result["consensus_flags"], reference["consensus_flags"])
    np.testing.assert_array_equal(
        missing_rate(data, item_indices=[np.array(1)]), missing_rate(data, item_indices=[1])
    )
    assert validate_worker_count(np.int64(2)) == 2
    assert validate_min_valid_indices(np.array(2), 3) == 2
    _, n_respondents = validate_score_vectors({}, n_respondents=np.int64(4))
    assert n_respondents == 4
    assert type(n_respondents) is int


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: screen([[1, 2], [3, 4]], min_flags=np.True_), "min_flags must be a positive"),
        (lambda: screen([[1, 2], [3, 4]], workers=2.0), "workers must be a positive integer"),
        (
            lambda: screen([[1, 2], [3, 4]], indices=["irv"], min_valid_indices=np.int64(2)),
            r"min_valid_indices cannot exceed the number of selected indices \(1\)",
        ),
        (
            lambda: validate_score_vectors({}, n_respondents=np.int64(0)),
            "n_respondents must be a positive integer within the platform index range or None",
        ),
        (
            lambda: validate_score_vectors({}, n_respondents=np.iinfo(np.intp).max + 1),
            "n_respondents must be a positive integer within the platform index range or None",
        ),
    ],
)
def test_integer_options_keep_their_error_messages(
    call: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


if __name__ == "__main__":
    unittest.main()
