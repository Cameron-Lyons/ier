"""Mahalanobis regressions for symmetric covariance and bounded complete-case work."""

import re
import tracemalloc
from collections import UserList
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

import ier
from ier._statistics import normal_quantile
from ier._validation import validate_matrix_input
from ier.mahad import mahad, mahad_qqplot, mahad_summary


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("rank", [1, 5, 13])
@pytest.mark.parametrize("missing", [False, True])
def test_distances_match_direct_covariance_reference(layout: str, rank: int, missing: bool) -> None:
    rng = np.random.default_rng(193)
    data = rng.normal(size=(157, rank)) @ rng.normal(size=(rank, 13))
    if missing:
        data[5:15] = np.nan
        data[::7, 3] = np.nan
    data = data[::-2, ::2] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    valid = ~np.isnan(data).any(axis=1)
    complete = data[valid]
    centered = complete - np.mean(complete, axis=0)
    inverse = np.linalg.pinv(np.cov(complete, rowvar=False))
    expected = np.full(len(data), np.nan)
    expected[valid] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inverse, centered), 0))

    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 31):
        actual = mahad(data, na_rm=True)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("method", ["chi2", "iqr", "zscore"])
def test_missing_rows_keep_distance_and_flag_alignment(method: str) -> None:
    rng = np.random.default_rng(110)
    data = rng.normal(size=(500, 17))
    data[2] *= 10
    data[13:29] = np.nan
    data[::11, 8] = np.nan
    valid = ~np.isnan(data).any(axis=1)
    expected_distances, expected_flags = mahad(data[valid], flag=True, method=method)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 51):
        distances, flags = mahad(data, na_rm=True, flag=True, method=method)
    np.testing.assert_allclose(distances[valid], expected_distances, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(flags[valid], expected_flags)
    assert np.isnan(distances[~valid]).all()
    assert not flags[~valid].any()


def test_covariance_condition_check_does_not_overflow() -> None:
    rng = np.random.default_rng(129)
    data = rng.normal(size=(100, 2)) * [1e100, 1e-100]
    expected = np.abs((data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0], ddof=1))
    actual = mahad(data)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("missing", [False, True])
def test_single_complete_observation_is_rejected(missing: bool) -> None:
    data = [[3.0], [np.nan]] if missing else [[3.0]]
    with pytest.raises(ValueError, match="at least two complete observations"):
        mahad(data, na_rm=missing)


def test_single_variable_matches_sample_standard_scores() -> None:
    data = np.array([[1.0], [3.0], [7.0], [np.nan]])
    values = data[:3, 0]
    expected = np.abs((values - np.mean(values)) / np.std(values, ddof=1))
    actual = mahad(data, na_rm=True)
    np.testing.assert_allclose(actual[:3], expected, atol=1e-14, rtol=1e-14)
    assert np.isnan(actual[-1])


@pytest.mark.parametrize("missing", [False, True])
def test_complete_case_workspaces_do_not_copy_the_input(missing: bool) -> None:
    rng = np.random.default_rng(212)
    data = rng.normal(size=(5000, 80))
    if missing:
        data[::10, 0] = np.nan
    data.flags.writeable = False
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 8000):
        tracemalloc.start()
        try:
            mahad(data, na_rm=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    # The output, row mask, covariance, and bounded workspaces fit well below
    # even one complete-case response copy at this shape and batch size.
    assert peak < data.nbytes / 2


@pytest.fixture(scope="module")
def normal_responses() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(5000, 50))


@pytest.mark.parametrize("method", ["chi2", "iqr", "zscore"])
def test_flags_only_unusually_large_distances(normal_responses: np.ndarray, method: str) -> None:
    # With many items, distances concentrate away from zero; a lower fence would
    # flag the most typical respondents.
    distances, flags = mahad(normal_responses, flag=True, method=method)
    assert flags.any()
    assert np.all(distances[flags] > np.median(distances))


@pytest.mark.parametrize("confidence", [0.8, 0.95, 0.99])
def test_zscore_flags_use_the_two_sided_critical_value_in_the_upper_tail(
    normal_responses: np.ndarray, confidence: float
) -> None:
    distances, flags = mahad(normal_responses, flag=True, confidence=confidence, method="zscore")
    z_scores = (distances - distances.mean()) / distances.std()
    np.testing.assert_array_equal(flags, z_scores > normal_quantile(1 - (1 - confidence) / 2))


def test_iqr_flags_use_only_the_upper_fence(normal_responses: np.ndarray) -> None:
    distances, flags = mahad(normal_responses, flag=True, method="iqr")
    q1, q3 = np.percentile(distances, [25, 75])
    np.testing.assert_array_equal(flags, distances > q3 + 1.5 * (q3 - q1))
    assert np.any(distances < q1 - 1.5 * (q3 - q1))


@pytest.mark.parametrize(
    "confidence",
    [
        True,
        False,
        np.True_,
        np.array(True),
        np.nan,
        np.inf,
        -np.inf,
        "0.9",
        b"0.9",
        np.array("0.9"),
        1.5,
        -0.1,
        None,
        10**400,
        Fraction(10**400, 3),
        Decimal("1.5"),
        Decimal("NaN"),
        Decimal("sNaN"),
        np.array([0.95]),
        np.complex128(0.5),
        np.timedelta64(0),
    ],
)
@pytest.mark.parametrize("flag", [False, True])
def test_confidence_must_be_a_finite_real_probability(confidence: object, flag: bool) -> None:
    data = np.random.default_rng(5).normal(size=(20, 3))
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        mahad(data, flag=flag, confidence=cast("Any", confidence))


@pytest.mark.parametrize(
    ("confidence", "reason"),
    [
        ("0.95", "expected a real number, got str"),
        (True, "expected a real number, got bool"),
        (np.array([0.95]), "expected a real number, got ndarray"),
        (1.5, "got 1.5"),
        (np.nan, "got nan"),
        (10**400, "got a value with no float equivalent"),
    ],
)
def test_rejected_confidence_messages_state_the_reason(confidence: object, reason: str) -> None:
    data = np.random.default_rng(5).normal(size=(20, 3))
    expected = re.escape(f"confidence must be between 0 and 1 ({reason})")
    with pytest.raises(ValueError, match=f"^{expected}$"):
        mahad(data, flag=True, confidence=cast("Any", confidence))


@pytest.mark.parametrize(
    "confidence",
    [
        Fraction(19, 20),
        np.float64(0.95),
        0.95,
        np.array(0.95),
        np.array(0.95, dtype=object),
        Decimal("0.95"),
    ],
)
def test_real_confidence_values_are_accepted(confidence: Any) -> None:
    data = np.random.default_rng(6).normal(size=(200, 4))
    data[0] = 6.0
    distances, flags = mahad(data, flag=True, confidence=confidence)
    expected_distances, expected_flags = mahad(data, flag=True, confidence=0.95)
    np.testing.assert_array_equal(distances, expected_distances)
    np.testing.assert_array_equal(flags, expected_flags)
    assert flags[0]


def test_summary_reports_respondent_counts_with_shared_key_names() -> None:
    data = np.random.default_rng(7).normal(size=(40, 3))
    data[[3, 7], 1] = np.nan
    summary = mahad_summary(data, na_rm=True)
    _, flags = mahad(data, flag=True, na_rm=True)
    assert (summary["n_total"], summary["n_valid"], summary["n_missing"]) == (40, 38, 2)
    assert (summary["total"], summary["valid_count"], summary["missing_count"]) == (40, 38, 2)
    assert summary["outliers"] == int(np.sum(flags))
    assert ier.mahad_summary is mahad_summary
    assert "mahad_summary" in ier.__all__


@pytest.mark.parametrize("container", [tuple, UserList])
def test_array_like_inputs_are_accepted(container: type) -> None:
    rows = np.random.default_rng(9).normal(size=(30, 3))
    data = container(container(row) for row in rows.tolist())

    np.testing.assert_array_equal(mahad(data), mahad(rows))


@pytest.mark.parametrize(
    "data", [5, "1,2", {"a": [1, 2]}, {1, 2}], ids=["int", "str", "dict", "set"]
)
def test_non_array_inputs_raise_type_error_that_is_also_a_value_error(data: object) -> None:
    with pytest.raises(TypeError, match="input data must be array-like") as caught:
        mahad(cast("Any", data))

    assert isinstance(caught.value, ValueError)


def test_qqplot_converts_list_input_once() -> None:
    data = np.random.default_rng(8).normal(size=(30, 3)).tolist()
    with patch("ier.mahad.validate_matrix_input", wraps=validate_matrix_input) as validate:
        theoretical, observed = mahad_qqplot(data)
    assert [type(call.args[0]) for call in validate.call_args_list] == [list, np.ndarray]
    assert len(theoretical) == 30
    np.testing.assert_allclose(observed, np.sort(mahad(data) ** 2), rtol=0, atol=0)


def test_missing_values_in_every_row_leave_no_complete_cases() -> None:
    with pytest.raises(ValueError, match="no complete cases"):
        mahad([[np.nan, 1.0], [2.0, np.nan]], na_rm=True)


@pytest.mark.parametrize("method", ["chi2", "iqr", "zscore"])
def test_identical_distances_are_not_flagged(method: str) -> None:
    distances, flags = mahad(np.full((20, 4), 3.0), flag=True, method=method)
    np.testing.assert_array_equal(distances, np.zeros(20))
    assert not flags.any()
