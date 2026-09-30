"""Finite score normalization and summaries preserve their mathematical definitions."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from ier import composite, composite_flag, composite_scores, composite_summary, screen_scores
from ier._summary import calculate_summary_stats


def _decimal_stats(values: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    observed = values[~np.isnan(values)]
    with localcontext() as context:
        context.prec = 800
        numbers = [
            Decimal(int(value)) if observed.dtype.kind in "iu" else Decimal.from_float(float(value))
            for value in observed
        ]
        mean = sum(numbers) / len(numbers)
        deviation = (sum((value - mean) ** 2 for value in numbers) / len(numbers)).sqrt()
        ordered = sorted(numbers)
        middle = len(numbers) // 2
        median = (
            ordered[middle] if len(numbers) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        )
        z_scores = np.full(len(values), np.nan)
        z_scores[~np.isnan(values)] = [
            float((value - mean) / deviation) if deviation else 0.0 for value in numbers
        ]
        return float(mean), float(deviation), float(median), z_scores


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        np.full(37, 1.1),
        np.full(37, np.finfo(float).max),
        np.full(37, np.nextafter(0.0, 1.0)),
        np.array([1e8, 1e8 + 8, 1e8 + 16, 1e8 + 24], dtype=np.float32),
        np.array([-np.finfo(float).max, np.finfo(float).max, np.finfo(float).max]),
        np.array([1e300, 2e300, 3e300, 4e300, 5e300]),
        np.array([1e-300, 2e-300, 3e-300, 4e-300, 5e-300]),
        np.array([-1e-300, 1e-300, 1e300, -1e300]),
        1.1 + np.arange(5) * np.spacing(1.1),
        -1.1 + np.arange(5) * np.spacing(1.1),
        np.array([0.0, np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0) * 2]),
        np.array([0.0, np.nextafter(0.0, 1.0)]),
    ],
)
def test_finite_score_statistics_match_decimal(
    values: np.ndarray, missing: bool, strided: bool
) -> None:
    data = values.copy()
    if missing:
        data = np.concatenate(([np.nan], data, [np.nan]))
    if strided:
        data = np.repeat(data, 2)[::-2]
    original = data.copy()
    data.flags.writeable = False
    mean, deviation, median, normalized = _decimal_stats(data)
    actual = composite_scores({"longstring": data})
    np.testing.assert_allclose(actual, normalized, rtol=5e-15, atol=5e-15)
    summary = calculate_summary_stats(data)
    np.testing.assert_allclose(summary["mean"], mean, rtol=5e-15, atol=0)
    np.testing.assert_allclose(summary["std"], deviation, rtol=5e-15, atol=0)
    np.testing.assert_allclose(summary["median"], median, rtol=5e-15, atol=0)
    screened = screen_scores({"longstring": data}, thresholds={"longstring": 0})
    details = screened["summary"]["longstring"]
    np.testing.assert_allclose(details["mean"], mean, rtol=5e-15, atol=0)
    np.testing.assert_allclose(details["std"], deviation, rtol=5e-15, atol=0)
    assert details["n_valid"] == np.count_nonzero(~np.isnan(data))
    np.testing.assert_array_equal(screened["flags"]["longstring"], data >= 0)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("values", [[np.nan] * 5, [np.nan, 1.1, np.nan]])
def test_sparse_standardization_preserves_established_scores(values: list[float]) -> None:
    np.testing.assert_array_equal(composite_scores({"longstring": values}), values)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_summaries_preserve_exact_means_and_variability(dtype: type) -> None:
    values = np.array([2**60, 2**60 + 1, 2**60 + 2, 2**60 + 3, 2**60 + 4], dtype=dtype)
    expected_mean, expected_deviation, expected_median, _ = _decimal_stats(values)
    summary = calculate_summary_stats(values)
    assert summary["mean"] == expected_mean
    assert summary["median"] == expected_median
    np.testing.assert_allclose(summary["std"], expected_deviation, rtol=5e-15, atol=0)


def test_constant_decimal_irv_does_not_create_composite_flags() -> None:
    data = np.tile([0, 0, 2.2, 2.2], (37, 1))
    scores, flags = composite_flag(data, indices=["irv"], threshold=0.5)
    np.testing.assert_array_equal(scores, np.zeros(37))
    assert not flags.any()
    np.testing.assert_array_equal(composite(data, indices=["irv"]), scores)


@pytest.mark.parametrize("scale", [1e-300, 1, 1e300])
def test_raw_composite_summaries_keep_finite_variability(scale: float) -> None:
    data = np.outer([1.0, 2.0, 3.0], [-1.0, 1.0]) * scale
    details = composite_summary(data, indices=["irv"], standardize=False)
    mean, deviation, _, _ = _decimal_stats(details["composite"])
    np.testing.assert_allclose(details["mean"], mean, rtol=5e-15, atol=0)
    np.testing.assert_allclose(details["std"], deviation, rtol=5e-15, atol=0)
    assert details["n_valid"] == 3
