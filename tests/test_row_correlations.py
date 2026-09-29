"""Paired correlation accuracy, missingness, and bounded workspace checks."""

from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

from ier import evenodd, individual_reliability, person_total
from ier._correlation import row_correlations, selected_row_correlations


def _decimal_correlation(left: np.ndarray, right: np.ndarray, zero_variance: float) -> float:
    valid = ~(np.isnan(left) | np.isnan(right))
    left, right = left[valid], right[valid]
    if len(left) < 2 or not (np.isfinite(left).all() and np.isfinite(right).all()):
        return np.nan
    with localcontext() as context:
        context.prec = 800
        first = [
            Decimal(int(value)) if left.dtype.kind in "iu" else Decimal.from_float(float(value))
            for value in left
        ]
        second = [
            Decimal(int(value)) if right.dtype.kind in "iu" else Decimal.from_float(float(value))
            for value in right
        ]
        mean1, mean2 = sum(first) / len(first), sum(second) / len(second)
        first = [value - mean1 for value in first]
        second = [value - mean2 for value in second]
        squares1, squares2 = sum(value**2 for value in first), sum(value**2 for value in second)
        if not squares1 or not squares2:
            return zero_variance
        covariance = sum(a * b for a, b in zip(first, second, strict=True))
        return float(covariance / squares1.sqrt() / squares2.sqrt())


@pytest.mark.parametrize(
    ("dtype", "missing"),
    [
        (np.int64, False),
        (np.float32, False),
        (np.float64, False),
        (np.float32, True),
        (np.float64, True),
    ],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_correlations_match_paired_scalar_reference(
    dtype: type, layout: str, missing: bool
) -> None:
    rng = np.random.default_rng(248)
    left = rng.integers(-5, 6, size=(37, 19)).astype(dtype)
    right = rng.integers(-5, 6, size=(37, 21)).astype(dtype)
    if missing:
        left[rng.random(left.shape) < 0.15] = np.nan
        right[rng.random(right.shape) < 0.15] = np.nan
        left[0] = np.nan
        right[1, 1:] = np.nan
    left[2] = 3
    if layout == "strided":
        left, right = left[:, ::-2], right[:, ::-2]
    else:
        left, right = np.array(left, order=layout), np.array(right, order=layout)
    left.flags.writeable = right.flags.writeable = False
    originals = left.copy(), right.copy()
    expected = []
    for first, second in zip(left, right, strict=True):
        pairs = [
            (float(a), float(b))
            for a, b in zip(first, second, strict=False)
            if not np.isnan(a) and not np.isnan(b)
        ]
        if len(pairs) < 2:
            expected.append(np.nan)
            continue
        a, b = np.array(pairs).T
        a -= a.mean()
        b -= b.mean()
        norm = np.sqrt(sum(a * a)) * np.sqrt(sum(b * b))
        expected.append(sum(a * b) / norm if norm else 0)
    for budget in (7, 83):
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget):
            actual = row_correlations(left, right)
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(left, originals[0])
    np.testing.assert_array_equal(right, originals[1])


@pytest.mark.parametrize("left_scale", [1e-300, 1e-100, 1, 1e100, 1e300])
@pytest.mark.parametrize("right_scale", [1e-300, 1e-100, 1, 1e100, 1e300])
@pytest.mark.parametrize("missing", [False, True])
def test_correlations_are_invariant_to_scale(
    left_scale: float, right_scale: float, missing: bool
) -> None:
    left = np.array([[1, 2, 3, 4], [1, 3, 2, 4], [3, 3, 3, 3]], dtype=float)
    right = np.array([[4, 3, 2, 1], [2, 3, 1, 4], [1, 2, 3, 4]], dtype=float)
    if missing:
        left[1, 0] = np.nan
    expected = row_correlations(left, right)
    actual = row_correlations(left * left_scale, right * right_scale)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)


def test_large_finite_means_do_not_overflow() -> None:
    left = np.array([[1e308, 1.1e308, 1.2e308]])
    right = np.array([[3, 2, 1]])
    np.testing.assert_allclose(row_correlations(left, right), [-1])


@pytest.mark.parametrize("missing", [False, True])
def test_correlation_workspaces_stay_bounded(missing: bool) -> None:
    data = np.arange(350, dtype=float).reshape(50, 7)
    if missing:
        data[0, 1] = np.nan
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 42),
        patch("ier._correlation.np.isnan", wraps=np.isnan) as scans,
        patch("ier._correlation.np.einsum", wraps=np.einsum) as reductions,
    ):
        row_correlations(data, data)
    assert len(reductions.call_args_list) > 3
    assert all(call.args[0].size <= 42 for call in scans.call_args_list)
    assert all(call.args[1].size <= 42 for call in reductions.call_args_list)


def test_zero_variance_and_missing_pairs_keep_distinct_policies() -> None:
    left = np.array([[1, 1, 1], [1, np.nan, np.nan], [np.nan] * 3, [1, 2, 3]])
    right = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [3, 2, 1]])
    np.testing.assert_allclose(row_correlations(left, right), [0, np.nan, np.nan, -1])
    np.testing.assert_allclose(
        row_correlations(left, right, zero_variance=np.nan), [np.nan, np.nan, np.nan, -1]
    )


def test_infinite_observations_are_unavailable_unless_the_pair_is_missing() -> None:
    left = np.array([[np.inf, 1, 2], [np.inf, 1, 2], [-np.inf, 1, 2]])
    right = np.array([[0, 1, 2], [np.nan, 1, 2], [0, 1, 2]])
    np.testing.assert_allclose(row_correlations(left, right), [np.nan, 1, np.nan])


def test_evenodd_and_person_total_preserve_small_scale_correlations() -> None:
    data = np.array([[1, 1, 2, 2, 3, 3], [2, 2, 4, 4, 6, 6]], dtype=float)
    for scorer in (lambda x: evenodd(x, [6]), person_total):
        expected = scorer(data)
        np.testing.assert_allclose(scorer(data * 1e-100), expected)


def test_two_pair_correlations_keep_large_integers_exact() -> None:
    lower = 2**60
    left = np.array([[lower, lower + 1], [lower, lower], [lower + 1, lower]])
    right = np.array([[1, 2], [1, 2], [1, 2]])
    np.testing.assert_array_equal(row_correlations(left, right), [1, 0, -1])


@pytest.mark.parametrize(
    ("dtype", "baseline"), [(np.int64, 2**60), (np.int64, -(2**60)), (np.uint64, 2**64 - 32)]
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("zero_variance", [0.0, np.nan])
def test_large_integer_correlations_preserve_small_differences(
    dtype: type, baseline: int, layout: str, zero_variance: float
) -> None:
    rng = np.random.default_rng(483)
    left = rng.integers(0, 16, (7, 9)).astype(dtype) + baseline
    right = rng.integers(0, 16, (7, 9)).astype(dtype) + baseline
    left[0] = baseline
    if layout == "strided":
        left, right = left[::-1, ::-1], right[::-1, ::-1]
    else:
        left, right = np.array(left, order=layout), np.array(right, order=layout)
    originals = left.copy(), right.copy()
    left.flags.writeable = right.flags.writeable = False
    expected = [_decimal_correlation(a, b, zero_variance) for a, b in zip(left, right, strict=True)]
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 29):
        actual = row_correlations(left, right, zero_variance=zero_variance)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    joined = np.column_stack((left, right))
    selected = selected_row_correlations(
        joined, np.arange(9), np.arange(9, 18), has_missing=False, zero_variance=zero_variance
    )
    np.testing.assert_allclose(selected, expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(left, originals[0])
    np.testing.assert_array_equal(right, originals[1])


@pytest.mark.parametrize("integer_left", [False, True])
def test_integer_correlation_anchor_excludes_missing_pairs(integer_left: bool) -> None:
    integers = np.array([[0, 2**60, 2**60 + 1, 2**60 + 2, 2**60 + 3]] * 4)
    floats = np.array(
        [
            [np.nan, 0, 1, 2, 3],
            [np.nan] * 5,
            [np.nan, 1, np.nan, np.nan, np.nan],
            [np.nan, 1, 1, 1, 1],
        ]
    )
    left, right = (integers, floats) if integer_left else (floats, integers)
    np.testing.assert_allclose(row_correlations(left, right), [1, np.nan, np.nan, 0], rtol=1e-14)


@pytest.mark.parametrize("items", [4, 6, 18])
def test_consistency_indices_are_invariant_to_large_integer_translation(items: int) -> None:
    data = np.random.default_rng(601).integers(1, 6, size=(13, items))
    translated = data + 2**60
    np.testing.assert_allclose(
        evenodd(translated, [items]), evenodd(data, [items]), rtol=1e-14, atol=1e-15
    )
    np.testing.assert_allclose(
        individual_reliability(translated, n_splits=7, random_seed=29),
        individual_reliability(data, n_splits=7, random_seed=29),
        rtol=1e-13,
        atol=1e-14,
    )


@pytest.mark.parametrize("zero_variance", [0.0, np.nan])
def test_two_pair_correlations_respect_missing_and_constant_policies(zero_variance: float) -> None:
    left = np.array([[1, 2], [3, 3], [1, np.nan], [1, np.inf], [1e-300, 2e-300]])
    right = np.array([[2, 1], [1, 2], [1, 2], [1, 2], [1e300, 2e300]])
    np.testing.assert_array_equal(
        row_correlations(left, right, zero_variance=zero_variance),
        [-1, zero_variance, np.nan, np.nan, 1],
    )


@pytest.mark.parametrize("shape", [(3,), (2, 2, 2)])
def test_correlation_inputs_must_be_matrices(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        row_correlations(np.ones(shape), np.ones(shape))


def test_broadcast_profiles_and_empty_respondents_are_supported() -> None:
    data = np.array([[1, 2, 3], [3, 2, 1]], dtype=float)
    profile = np.broadcast_to(np.array([1, 2, 3]), data.shape)
    np.testing.assert_allclose(row_correlations(data, profile), [1, -1])
    assert row_correlations(np.empty((0, 3)), np.empty((0, 3))).shape == (0,)


@pytest.mark.parametrize("offsets", [(1.1, 1.1), (1e100, 1e-100), (-1e100, 1e308), (1e-100, -1.1)])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("zero_variance", [0.0, np.nan])
def test_nearly_constant_correlations_match_decimal(
    offsets: tuple[float, float], layout: str, zero_variance: float
) -> None:
    rng = np.random.default_rng(44)
    left = offsets[0] + rng.integers(0, 9, (9, 17)) * np.spacing(offsets[0])
    right = offsets[1] + rng.integers(0, 9, (9, 17)) * np.spacing(offsets[1])
    left[0] = offsets[0]
    right[1] = offsets[1]
    left[2, :3] = np.nan
    right[2, 3] = np.nan
    left[3] = np.nan
    left[4, 1:] = np.nan
    left[5, 2:] = np.nan
    left[6, 0] = np.inf
    left[7, 0], right[7, 0] = np.inf, np.nan
    if layout == "strided":
        left, right = left[::-1, ::-1], right[::-1, ::-1]
    else:
        left, right = np.array(left, order=layout), np.array(right, order=layout)
    originals = left.copy(), right.copy()
    left.flags.writeable = right.flags.writeable = False
    expected = [_decimal_correlation(a, b, zero_variance) for a, b in zip(left, right, strict=True)]
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 41):
        actual = row_correlations(left, right, zero_variance=zero_variance)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    combined = np.array(np.column_stack((left, right)), order="F" if layout == "F" else "C")
    original_combined = combined.copy()
    combined.flags.writeable = False
    selected = selected_row_correlations(
        combined,
        np.arange(17),
        np.arange(17, 34),
        has_missing=True,
        zero_variance=zero_variance,
    )
    np.testing.assert_allclose(selected, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(left, originals[0])
    np.testing.assert_array_equal(right, originals[1])
    np.testing.assert_array_equal(combined, original_combined)


@pytest.mark.parametrize("zero_variance", [0.0, np.nan])
def test_two_observed_pairs_have_exact_correlations_at_any_width(zero_variance: float) -> None:
    rng = np.random.default_rng(101)
    left = np.full((31, 19), np.nan)
    right = np.full((31, 19), np.nan)
    left[:, [1, 17]] = rng.normal(size=(31, 2))
    right[:, [1, 17]] = rng.normal(size=(31, 2))
    left[0, [1, 17]] = 1.1
    left[1, 1] = np.inf
    expected = [_decimal_correlation(a, b, zero_variance) for a, b in zip(left, right, strict=True)]
    np.testing.assert_array_equal(
        row_correlations(left, right, zero_variance=zero_variance), expected
    )
