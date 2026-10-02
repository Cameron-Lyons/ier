"""Reusable score boundaries reject values that would change their meaning."""

from pathlib import Path

import numpy as np
import pytest

from ier import (
    composite_scores,
    response_time_score_flags,
    save_response_time_archive,
    save_score_archive,
    screen_scores,
)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([True, False]),
        np.asarray(["0.1", "0.9"]),
        np.asarray([b"0.1", b"0.9"]),
        np.asarray([0.1 + 2j, 0.9 + 3j]),
        np.asarray([0.1, 0.9], dtype=object),
        np.asarray([1, 2], dtype="timedelta64[s]"),
    ],
    ids=["boolean", "unicode", "bytes", "complex", "object", "timedelta"],
)
def test_reusable_decision_apis_reject_non_real_numeric_scores(values: np.ndarray) -> None:
    before = values.copy()
    for reuse in (screen_scores, composite_scores):
        with pytest.raises(ValueError, match="real numeric array"):
            reuse({"irv": values})
    with pytest.raises(ValueError, match="real numeric array"):
        response_time_score_flags(values, threshold=0.5)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("writer", ["score", "response_time"])
@pytest.mark.parametrize(
    "values",
    [np.asarray([True, False]), np.asarray(["0.1", "0.9"]), np.asarray([0.1 + 2j, 0.9 + 3j])],
    ids=["boolean", "unicode", "complex"],
)
def test_archive_writers_reject_lossy_conversion_before_replacement(
    tmp_path: Path,
    writer: str,
    values: np.ndarray,
) -> None:
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"previous results")
    with pytest.raises(ValueError, match="real numeric array"):
        if writer == "score":
            save_score_archive(destination, {"irv": values})
        else:
            save_response_time_archive(destination, values, [True, False], threshold=0.5)
    assert destination.read_bytes() == b"previous results"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("dtype", [np.int16, np.uint64, np.float32, np.float64])
def test_real_numeric_views_are_converted_without_mutation(dtype: type) -> None:
    values = np.arange(6, dtype=dtype)[::-2]
    before = values.copy()
    result = screen_scores({"longstring": values}, thresholds={"longstring": 3}, min_flags=1)
    np.testing.assert_array_equal(result["scores"]["longstring"], [5.0, 3.0, 1.0])
    np.testing.assert_array_equal(result["flags"]["longstring"], [True, True, False])
    np.testing.assert_array_equal(values, before)
    assert result["scores"]["longstring"].dtype == np.float64
    if dtype is np.float64:
        assert result["scores"]["longstring"] is values


def test_out_of_range_wide_floats_raise_validation_error_without_warning() -> None:
    if np.finfo(np.longdouble).max == np.finfo(np.float64).max:
        pytest.skip("platform does not provide wider floating-point range")
    values = np.asarray([np.finfo(np.longdouble).max], dtype=np.longdouble)
    with pytest.raises(ValueError, match="finite values or NaN"):
        screen_scores({"irv": values})
