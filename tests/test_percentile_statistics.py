"""Sample-relative cutoffs retain finite interpolation and input ownership."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from ier import screen_scores
from ier._flagging import resolve_threshold, threshold_flags


def _decimal_percentile(values: np.ndarray, percentile: float) -> float:
    observed = sorted(values[~np.isnan(values)])
    if not observed:
        return 0.0
    # The linear policy uses the floating-point virtual rank (n - 1) * q.
    position = (len(observed) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(observed) - 1)
    with localcontext() as context:
        context.prec = 800
        numbers = [
            Decimal(int(value)) if values.dtype.kind in "iu" else Decimal.from_float(float(value))
            for value in (observed[lower], observed[upper])
        ]
        fraction = Decimal.from_float(position) - lower
        return float(numbers[0] * (1 - fraction) + numbers[1] * fraction)


@pytest.mark.parametrize("percentile", [0, 1, 5, 25, 50, 75, 95, 99, 100])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        np.array([-np.finfo(float).max, np.finfo(float).max]),
        np.array([-np.finfo(float).max, -np.finfo(float).max, np.finfo(float).max]),
        np.array([np.finfo(float).max / 2, np.finfo(float).max]),
        np.full(37, 1.1),
        np.array([0.0, np.nextafter(0.0, 1.0)]),
        np.array([2, 3]) * np.nextafter(0.0, 1.0),
        np.array([-3, -2]) * np.nextafter(0.0, 1.0),
        np.array([-np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32),
    ],
)
def test_finite_percentiles_match_decimal(
    values: np.ndarray, percentile: float, missing: bool, strided: bool
) -> None:
    scores = values.copy()
    if missing:
        scores = np.concatenate(([np.nan], scores, [np.nan]))
    if strided:
        scores = np.repeat(scores, 2)[::-2]
    original = scores.copy()
    scores.flags.writeable = False
    cutoff = resolve_threshold(scores, None, percentile)
    expected = _decimal_percentile(scores, percentile)
    np.testing.assert_allclose(cutoff, expected, rtol=5e-15, atol=0)
    for direction in ("high", "low"):
        flags = threshold_flags(scores, None, percentile, direction)
        expected_flags = scores > expected if direction == "high" else scores < expected
        np.testing.assert_array_equal(flags, expected_flags)
    np.testing.assert_array_equal(scores, original)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("percentile", [0, 5, 25, 50, 75, 95, 100])
def test_integer_cutoffs_round_only_after_interpolation(dtype: type, percentile: float) -> None:
    info = np.iinfo(dtype)
    scores = np.array([info.min, info.min + 1, info.max - 1, info.max], dtype=dtype)
    assert resolve_threshold(scores, None, percentile) == _decimal_percentile(scores, percentile)


@pytest.mark.parametrize("percentile", [0, 25, 50, 95, 100])
def test_screen_percentiles_keep_finite_cutoffs_and_direction(percentile: float) -> None:
    maximum = np.finfo(float).max
    scores = np.array([-maximum, maximum, np.nan])
    result = screen_scores({"longstring": scores, "irv": scores}, percentile=percentile)
    high = _decimal_percentile(scores, percentile)
    low = _decimal_percentile(scores, 100 - percentile)
    assert result["thresholds"] == {"longstring": high, "irv": low}
    np.testing.assert_array_equal(result["flags"]["longstring"], scores > high)
    np.testing.assert_array_equal(result["flags"]["irv"], scores < low)
    np.testing.assert_array_equal(result["valid_index_counts"], [2, 2, 0])
    assert result["threshold_sources"] == {"longstring": "percentile", "irv": "percentile"}


@pytest.mark.parametrize("percentile", [0, 50, 100])
@pytest.mark.parametrize("scores", [np.array([]), np.full(7, np.nan), np.array([np.nan, 1.1])])
def test_empty_missing_and_singleton_percentiles(scores: np.ndarray, percentile: float) -> None:
    expected = 1.1 if np.any(~np.isnan(scores)) else 0.0
    assert resolve_threshold(scores, None, percentile) == expected
    assert not threshold_flags(scores, None, percentile, "high").any()
    assert not threshold_flags(scores, None, percentile, "low").any()


@pytest.mark.parametrize("percentile", [np.nextafter(0.0, 1.0), np.nextafter(100.0, 0.0)])
def test_percentiles_next_to_endpoints_remain_finite(percentile: float) -> None:
    scores = np.array([-np.finfo(float).max, np.finfo(float).max])
    assert np.isfinite(resolve_threshold(scores, None, percentile))


@pytest.mark.parametrize("missing_rate", [0, 0.1, 0.9])
def test_ordinary_float64_cutoffs_match_numpy(missing_rate: float) -> None:
    rng = np.random.default_rng(20260930)
    scores = rng.normal(size=1001)
    scores[rng.random(len(scores)) < missing_rate] = np.nan
    for percentile in [0, 1, 5, 25, 50, 75, 95, 99, 100, *rng.uniform(0, 100, size=32)]:
        assert resolve_threshold(scores, None, percentile) == np.nanpercentile(scores, percentile)


@pytest.mark.parametrize("direction", ["high", "low"])
@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize(
    ("scores", "cutoff"),
    [
        (np.array([2**60 - 1, 2**60, 2**60 + 1], dtype=np.int64), float(2**60)),
        (np.array([-(2**60) - 1, -(2**60), -(2**60) + 1], dtype=np.int64), float(-(2**60))),
        (np.array([0, 1, 2], dtype=np.uint64), -0.5),
        (np.array([0, 1, 2], dtype=np.uint64), 0.5),
        (np.array([np.iinfo(np.uint64).max - 1, np.iinfo(np.uint64).max]), float(2**64)),
        (np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max]), np.finfo(float).max),
        (np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max]), -np.finfo(float).max),
        (
            np.array([1.0, 1.0 + 2**-23, 1.0 + 2**-22, np.nan], dtype=np.float32),
            1.0 + 3 * 2**-24,
        ),
    ],
)
def test_flags_compare_original_values_to_reported_cutoff(
    scores: np.ndarray, cutoff: float, direction: str, inclusive: bool
) -> None:
    expected = []
    for value in scores:
        number = int(value) if scores.dtype.kind in "iu" else float(value)
        expected.append(
            (number >= cutoff if inclusive else number > cutoff)
            if direction == "high"
            else (number <= cutoff if inclusive else number < cutoff)
        )
    flags = threshold_flags(scores, cutoff, 95, direction, inclusive=inclusive)
    np.testing.assert_array_equal(flags, expected)


def test_float32_percentile_flags_use_double_precision_cutoff() -> None:
    scores = np.array([1.0 + 2**-23, 1.0 + 2**-22], dtype=np.float32)
    assert resolve_threshold(scores, None, 50) == 1.0 + 3 * 2**-24
    np.testing.assert_array_equal(threshold_flags(scores, None, 50, "high"), [False, True])
    np.testing.assert_array_equal(threshold_flags(scores, None, 50, "low"), [True, False])
