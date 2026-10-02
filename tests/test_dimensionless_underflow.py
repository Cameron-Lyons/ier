"""Independent exact arithmetic checks for dimensionless score normalization."""

from decimal import Decimal, localcontext
from fractions import Fraction
from importlib import import_module

import numpy as np
import pytest

from ier import response_time_consistency, semantic_ant, semantic_syn


def _exact_moments(row: np.ndarray) -> tuple[Fraction, Fraction]:
    observed = [Fraction(float(value)) for value in row if np.isfinite(value)]
    mean = sum(observed, start=Fraction()) / len(observed)
    variance = sum(((value - mean) ** 2 for value in observed), start=Fraction()) / len(observed)
    return mean, variance


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _expected_cv(row: np.ndarray) -> float:
    mean, variance = _exact_moments(row)
    with localcontext() as context:
        context.prec = 800
        return float(_decimal(variance).sqrt() / _decimal(mean))


def _expected_semantic(
    row: np.ndarray,
    pairs: list[tuple[int, int]],
    bounds: tuple[float, float] | None = None,
) -> float:
    differences = []
    for left, right in pairs:
        if not np.isfinite(row[left]) or not np.isfinite(row[right]):
            continue
        first, second = Fraction(float(row[left])), Fraction(float(row[right]))
        differences.append(
            abs(first - second)
            if bounds is None
            else abs(first + second - Fraction(bounds[0]) - Fraction(bounds[1]))
        )
    if not differences:
        return np.nan
    _, variance = _exact_moments(row)
    difference = sum(differences, start=Fraction()) / len(differences)
    with localcontext() as context:
        context.prec = 800
        normalized = _decimal(difference) / _decimal(variance).sqrt()
        return float(max(Decimal(-1), min(Decimal(1), 1 - normalized)))


def test_timing_consistency_keeps_subnormal_dimensionless_variation() -> None:
    smallest = np.nextafter(0.0, 1.0)
    tiny = np.finfo(float).tiny
    data = np.asarray(
        [
            [smallest, 2 * smallest, np.nan],
            [0.0, smallest, np.nan],
            [tiny, np.nextafter(tiny, np.inf), np.nan],
            [-smallest, smallest, 3 * smallest],
        ]
    )
    expected = np.asarray([_expected_cv(row) for row in data])
    expected[:2] = [1 / 3, 1.0]
    np.testing.assert_allclose(response_time_consistency(data), expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("antonym", [False, True])
def test_semantic_normalization_retains_subnormal_pair_differences(antonym: bool) -> None:
    smallest = np.nextafter(0.0, 1.0)
    data = smallest * np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, np.nan, 2.0, 1.0]])
    pairs = [(0, 1), (2, 3)]
    bounds = (0.0, 5 * smallest) if antonym else None
    expected = [_expected_semantic(row, pairs, bounds) for row in data]
    actual = (
        semantic_ant(data, pairs, scale_min=bounds[0], scale_max=bounds[1])
        if bounds is not None
        else semantic_syn(data, pairs)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize(
    "bounds",
    [(-np.finfo(float).max, np.finfo(float).max), (0.0, 1.0)],
)
def test_tiny_antonym_rows_preserve_large_scale_reflection_and_missing_pairs(
    bounds: tuple[float, float],
) -> None:
    smallest = np.nextafter(0.0, 1.0)
    data = smallest * np.asarray([[-2.0, 1.0, 3.0, 4.0], [np.nan, np.nan, 1.0, 2.0]])
    pairs = [(0, 1)]
    expected = [_expected_semantic(row, pairs, bounds) for row in data]
    actual = semantic_ant(data, pairs, scale_min=bounds[0], scale_max=bounds[1])
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_semantic_default_bounds_preserve_equivalent_response_units() -> None:
    data = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    pairs = [(0, 1), (2, 3)]
    smallest = np.nextafter(0.0, 1.0)
    for scorer in (semantic_syn, semantic_ant):
        np.testing.assert_allclose(scorer(data * smallest, pairs), scorer(data, pairs), atol=2e-15)


@pytest.mark.skipif(
    np.finfo(np.longdouble).tiny >= np.finfo(float).tiny,
    reason="longdouble has no extended exponent range on this platform",
)
def test_extended_precision_tiny_response_units_preserve_inferred_antonym_bounds() -> None:
    data = np.asarray([[1, 2, 3, 4]], dtype=np.longdouble) * np.longdouble("1e-400")
    pairs = [(0, 1), (2, 3)]
    np.testing.assert_allclose(response_time_consistency(data), [np.sqrt(5) / 5], rtol=2e-15)
    np.testing.assert_array_equal(response_time_consistency(np.full_like(data, data[0, 0])), [0.0])
    np.testing.assert_allclose(semantic_syn(data, pairs), [1 - 2 / np.sqrt(5)], atol=2e-15)
    np.testing.assert_allclose(semantic_ant(data, pairs), [1 - 4 / np.sqrt(5)], atol=2e-15)


def test_dimensionless_repairs_keep_ordinary_missing_nonfinite_and_constant_policies() -> None:
    data = np.asarray(
        [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [np.nan] * 3, [np.inf, 1.0, 2.0]]
    )
    original = data.copy()
    np.testing.assert_allclose(
        response_time_consistency(data), [np.sqrt(2 / 3) / 2, 0.0, np.nan, np.nan, np.nan]
    )
    np.testing.assert_allclose(
        semantic_syn(data, [(0, 1)]), [1 - 1 / np.sqrt(2 / 3), 1.0, 1.0, np.nan, np.nan]
    )
    np.testing.assert_array_equal(data, original)


def test_repairs_remain_bounded_for_mixed_scale_readonly_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ier._row_statistics as statistics
    import ier.semantic as semantic

    timing = import_module("ier.response_time")

    smallest = np.nextafter(0.0, 1.0)
    scales = [smallest, 1.0, np.finfo(float).tiny, 1e100] * 10
    data = np.asarray(scales)[:, None] * np.asarray([1.0, 2.0, 3.0, 4.0])
    data[::3, 1] = np.nan
    data.flags.writeable = False
    monkeypatch.setattr(statistics, "_ROW_BATCH_ELEMENTS", 8)
    helper = statistics._scaled_subnormal_moment_rows

    def bounded_helper(
        values: np.ndarray,
        means: np.ndarray,
        deviations: np.ndarray,
        *,
        repair_mean: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        assert values.size <= 8
        return helper(values, means, deviations, repair_mean=repair_mean)

    monkeypatch.setattr(timing, "_scaled_subnormal_moment_rows", bounded_helper)
    monkeypatch.setattr(semantic, "_scaled_subnormal_moment_rows", bounded_helper)
    expected_cv = [_expected_cv(row) for row in data]
    expected_semantic = [_expected_semantic(row, [(0, 1), (2, 3)]) for row in data]
    np.testing.assert_allclose(response_time_consistency(data), expected_cv, rtol=2e-15, atol=0)
    np.testing.assert_allclose(
        semantic_syn(data, [(0, 1), (2, 3)]), expected_semantic, rtol=2e-15, atol=2e-15
    )
