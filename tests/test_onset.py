"""Onset parameter validation and complete/missing sequence equivalence."""

from decimal import Decimal, localcontext
from itertools import accumulate
from unittest.mock import patch

import numpy as np
import pytest

from ier import onset, onset_flag


def _decimal_onset(row: np.ndarray, window: int) -> float:
    """Evaluate the established statistic using high-precision scalar arithmetic."""
    with localcontext() as context:
        context.prec = 800
        values = [
            Decimal(int(value)) if row.dtype.kind in "iu" else Decimal.from_float(float(value))
            for value in row
            if not np.isnan(value)
        ]
        series = []
        for start in range(len(values) - window + 1):
            subset = values[start : start + window]
            mean = sum(subset) / window
            series.append((sum((value - mean) ** 2 for value in subset) / window).sqrt())
        n = len(series)
        prefix = list(accumulate(series))
        squares = list(accumulate(value * value for value in series))
        trim = max(1, n // 10)
        candidates = range(trim, n - trim)
        statistics = [
            (prefix[k] - prefix[-1] * (k + 1) / n) ** 2
            / max(squares[k - 1] - prefix[k - 1] ** 2 / k, Decimal("1e-10"))
            for k in candidates
        ]
        best = max(range(len(statistics)), key=statistics.__getitem__)
        return float(trim + best + window - 1) if statistics[best] > Decimal("1.358") else np.nan


@pytest.mark.parametrize("window", [5, 32])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize(
    ("dtype", "offset"),
    [
        ("int64", 2**60),
        ("int64", -(2**63)),
        ("int64", 2**63 - 6),
        ("uint64", 2**64 - 6),
    ],
)
def test_large_integer_responses_preserve_onsets(
    window: int, layout: str, dtype: str, offset: int
) -> None:
    data = np.random.default_rng(3).integers(0, 5, size=(3, 61)).astype(dtype)
    data += offset
    data = data[:, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    expected = [_decimal_onset(row, window) for row in data]
    data.flags.writeable = False
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 400):
        np.testing.assert_array_equal(onset(data, window_size=window, min_items=window), expected)
        np.testing.assert_array_equal(
            onset_flag(data, window_size=window, min_items=window), ~np.isnan(expected)
        )
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("window", [16, 32, 59])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64, np.bool_])
def test_categorical_rolling_deviations_match_exact_window_moments(
    window: int, dtype: type
) -> None:
    from ier.onset import _running_inconsistency_complete

    data = np.random.default_rng(15).integers(0, 5, size=(4, 61)).astype(dtype)
    data[1, :32] = 1
    data[2] = 0
    data[3] *= 100_000 if dtype != np.bool_ else True
    data.flags.writeable = False
    expected = []
    for row in data:
        deviations = []
        for start in range(len(row) - window + 1):
            values = [int(value) for value in row[start : start + window]]
            variance = (window * sum(value * value for value in values) - sum(values) ** 2) / (
                window * window
            )
            deviations.append(np.sqrt(variance))
        expected.append(deviations)
    np.testing.assert_array_equal(_running_inconsistency_complete(data, window), expected)


@pytest.mark.parametrize("window", [16, 32])
@pytest.mark.parametrize("step", [1, 7])
def test_repeated_window_variability_has_no_onset(window: int, step: int) -> None:
    data = np.tile(np.arange(window) * step, (3, 4)).astype(float)
    assert np.isnan(onset(data, window_size=window, min_items=window)).all()


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("structure", ["categorical", "continuous", "large", "fractional"])
def test_wide_windows_match_high_precision_reference(missing: bool, structure: str) -> None:
    rng = np.random.default_rng(37)
    data = rng.integers(1, 6, size=(4, 71)).astype(float)
    if structure == "continuous":
        data = rng.uniform(-3.5, 7.25, size=data.shape)
    elif structure == "large":
        data[1] *= 2**30
        data[2] *= 1e200
    elif structure == "fractional":
        data *= 0.1
        data += 1.1
    data[3] = data[3, 0]
    if missing:
        data[:, [3, 17, 26, 45]] = np.nan
    expected = [_decimal_onset(row, 32) for row in data]
    np.testing.assert_array_equal(onset(data, window_size=32, min_items=32), expected)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 200):
        np.testing.assert_array_equal(onset(data, window_size=32, min_items=32), expected)


@pytest.mark.parametrize("exponent", [490, 700, 1023])
@pytest.mark.parametrize("missing", [False, True])
def test_large_finite_responses_match_high_precision_reference(
    exponent: int, missing: bool
) -> None:
    data = np.random.default_rng(3).uniform(-0.75, 0.75, size=(4, 31))
    data[1:3] = np.ldexp(data[1:3], exponent)
    data[3] = np.ldexp(0.25, exponent)
    if missing:
        data[:, [3, 17]] = np.nan
    expected = [_decimal_onset(row, 5) for row in data]
    data.flags.writeable = False
    np.testing.assert_array_equal(onset(data, window_size=5, min_items=10), expected)
    np.testing.assert_array_equal(onset(data[1:3], window_size=5, min_items=10), expected[1:3])


@pytest.mark.parametrize("value", [1.0, 1e200, np.finfo(float).max])
def test_constant_running_variability_has_no_changepoint(value: float) -> None:
    data = np.tile([value, -value], (1, 20))
    assert np.isnan(onset(data, window_size=2, min_items=10)).all()


@pytest.mark.parametrize("name", ["window_size", "min_items"])
@pytest.mark.parametrize("value", [2.5, 3.0, True, np.bool_(True), np.nan, np.inf, "3", None])
def test_onset_size_parameters_require_integers(name: str, value: object) -> None:
    options = {"window_size": 3, "min_items": 5, name: value}
    with pytest.raises(ValueError, match=f"{name}.*integer"):
        onset([[1, 2, 3, 4, 5, 6]], **options)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_numpy_integer_parameters_are_safe(dtype: type) -> None:
    data = np.random.default_rng(3).integers(1, 6, size=(7, 30))
    np.testing.assert_array_equal(
        onset(data, window_size=dtype(3), min_items=dtype(10)),
        onset(data, window_size=3, min_items=10),
    )


@pytest.mark.parametrize(
    ("window", "minimum", "message"), [(1, 5, "at least 2"), (5, 4, "at least as large")]
)
def test_size_bounds_are_enforced(window: int, minimum: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        onset([[1, 2, 3, 4, 5, 6]], window_size=window, min_items=minimum)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("budget", [20, 420, 10000])
def test_mixed_sequences_match_independent_complete_row_scoring(layout: str, budget: int) -> None:
    rng = np.random.default_rng(31)
    data = rng.integers(1, 6, size=(31, 41)).astype(float)
    data[rng.random(data.shape) < 0.2] = np.nan
    data[0] = np.nan
    data[1, :25] = np.nan
    data[2] = 4
    data[3] = rng.integers(1, 6, size=41)
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = np.array(
        [
            onset(row[~np.isnan(row)][None, :], window_size=7, min_items=20)[0]
            if np.any(~np.isnan(row))
            else np.nan
            for row in data
        ]
    )
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget):
        scores = onset(data, window_size=7, min_items=20)
        flags = onset_flag(data, window_size=7, min_items=20)
    np.testing.assert_array_equal(scores, expected)
    np.testing.assert_array_equal(flags, ~np.isnan(expected))
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("items", [2, 30])
def test_strict_missing_policy_applies_even_to_ineligible_rows(items: int) -> None:
    data = np.ones((9, items))
    data[-1, -1] = np.nan
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 20),
        pytest.raises(ValueError, match="missing values"),
    ):
        onset(data, window_size=3, min_items=5, na_rm=False)
    assert np.isnan(onset(data, window_size=3, min_items=5)).all()


def test_windows_need_three_running_observations() -> None:
    data = np.arange(18).reshape(3, 6)
    assert np.isnan(onset(data, window_size=5, min_items=5)).all()
    assert np.isnan(onset(data, window_size=10**100, min_items=10**101)).all()


@pytest.mark.parametrize("na_rm", [False, True])
def test_infinite_responses_leave_their_rows_unavailable(na_rm: bool) -> None:
    data = np.random.default_rng(3).integers(1, 6, size=(3, 30)).astype(float)
    expected = onset(data[1:2], window_size=5, min_items=10)[0]
    data[0, 3] = np.inf
    data[2, 7] = -np.inf
    np.testing.assert_array_equal(
        onset(data, window_size=5, min_items=10, na_rm=na_rm), [np.nan, expected, np.nan]
    )


def test_missing_batch_only_scores_eligible_rows() -> None:
    from ier.onset import _running_inconsistency_complete

    rng = np.random.default_rng(17)
    data = rng.integers(1, 6, size=(15, 40)).astype(float)
    for row, missing in enumerate(range(15)):
        data[row, :missing] = np.nan
    with patch(
        "ier.onset._running_inconsistency_complete", wraps=_running_inconsistency_complete
    ) as rolling:
        scores = onset(data, window_size=5, min_items=30)
    assert sum(call.args[0].shape[0] for call in rolling.call_args_list) == 11
    assert np.isnan(scores[11:]).all()
    assert scores.shape == (15,)
