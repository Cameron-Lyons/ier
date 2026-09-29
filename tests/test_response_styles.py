"""Response-style availability, scale inference, and bounded summary reductions."""

from collections.abc import Callable
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from ier import IndexOptions, acquiescence, midpoint_responding, response_pattern, screen, u3_poly


@pytest.mark.parametrize("bounds", [(None, None), (1, None), (None, 5), (1, 5), (3, 3)])
def test_all_missing_response_styles_are_unavailable(
    bounds: tuple[float | None, float | None],
) -> None:
    data = np.full((3, 4), np.nan)
    lower, upper = bounds
    for scorer in (acquiescence, midpoint_responding, u3_poly):
        assert np.isnan(scorer(data, scale_min=lower, scale_max=upper)).all()
    assert all(np.isnan(values).all() for values in response_pattern(data, lower, upper).values())


@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("balanced", [False, True])
def test_constant_scale_preserves_respondent_availability(na_rm: bool, balanced: bool) -> None:
    data = [[3, 3, 3, 3], [3, np.nan, 3, np.nan], [3, 3, np.nan, np.nan], [np.nan] * 4]
    actual = acquiescence(
        data,
        positive_items=[0, 2] if balanced else None,
        negative_items=[1, 3] if balanced else None,
        na_rm=na_rm,
    )
    expected = [0.5, np.nan if balanced else 0.5, 0.5, np.nan]
    if not na_rm:
        expected = [0.5, np.nan, np.nan, np.nan]
    np.testing.assert_array_equal(actual, expected)


def test_constant_scale_screening_does_not_count_missing_acquiescence() -> None:
    result = screen(
        [[3, 3], [np.nan, np.nan]],
        indices=["acquiescence", "u3_poly", "midpoint"],
        options=IndexOptions(scale_min=3, scale_max=3),
        thresholds={"acquiescence": 0.4, "u3_poly": 0.9, "midpoint": 0.9},
        strict=True,
    )
    np.testing.assert_array_equal(result["valid_index_counts"], [3, 0])
    np.testing.assert_array_equal(result["flag_counts"], [3, 0])


@pytest.mark.parametrize("scorer", [acquiescence, u3_poly, midpoint_responding, response_pattern])
def test_response_style_bounds_must_be_ordered(scorer: Callable) -> None:
    with pytest.raises(ValueError, match="scale_max"):
        scorer([[1, 3, 5]], scale_min=5, scale_max=1)


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf, -np.inf])
def test_midpoint_requires_finite_nonnegative_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        midpoint_responding([[1, 3, 5]], 1, 5, tolerance=tolerance)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
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
def test_response_patterns_match_scalar_reference(layout: str, dtype: type, missing: bool) -> None:
    rng = np.random.default_rng(27)
    data = rng.integers(1, 6, size=(37, 13)).astype(dtype)
    if missing:
        data[rng.random(data.shape) < 0.3] = np.nan
        data[0] = np.nan
    data = data[:, ::-2] if layout == "strided" else np.array(data, order=layout)
    data.flags.writeable = False
    original = data.copy()
    expected: dict[str, list[float]] = {
        name: [] for name in ("extreme", "midpoint", "acquiescence", "variability")
    }
    for row in data:
        values = [float(value) for value in row if not np.isnan(value)]
        if not values:
            for scores in expected.values():
                scores.append(np.nan)
            continue
        mean = sum(values) / len(values)
        expected["extreme"].append(sum(value in (1, 5) for value in values) / len(values))
        expected["midpoint"].append(values.count(3) / len(values))
        expected["acquiescence"].append(mean)
        expected["variability"].append((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5)

    for budget in (3, 47):
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget):
            actual = response_pattern(data, scale_min=1, scale_max=5)
        for name, values in expected.items():
            np.testing.assert_allclose(actual[name], values, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(data, original)


def test_inferred_integer_endpoints_remain_distinct() -> None:
    lower = 2**60
    data = np.array([[lower, lower + 1, lower + 2]], dtype=np.int64)
    np.testing.assert_array_equal(u3_poly(data), [2 / 3])


def test_midpoint_avoids_integer_endpoint_addition_overflow() -> None:
    bound = 2**62
    data = np.array([[bound, bound, bound]], dtype=np.int64)
    np.testing.assert_array_equal(midpoint_responding(data), [1.0])
    np.testing.assert_array_equal(response_pattern(data)["midpoint"], [1.0])


def test_midpoint_avoids_floating_endpoint_addition_overflow() -> None:
    np.testing.assert_array_equal(midpoint_responding([[1e308]]), [1.0])


def test_combined_response_patterns_share_one_bounded_missing_scan() -> None:
    data = np.arange(350, dtype=float).reshape(50, 7) % 5 + 1
    data[::3, 1] = np.nan
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 90),
        patch("ier._row_statistics.np.isnan", wraps=np.isnan) as missing_checks,
    ):
        response_pattern(data, scale_min=1, scale_max=5)
    scanned = [
        call.args[0].size
        for call in missing_checks.call_args_list
        if isinstance(call.args[0], np.ndarray)
    ]
    assert len(scanned) > 1
    assert max(scanned) <= 90
    assert sum(scanned) == data.size


def test_fractional_midpoint_tolerance_includes_boundary_values() -> None:
    data = [[0.5, 0.75, 1, 1.25, 1.5, np.nan], [np.nan] * 6]
    np.testing.assert_array_equal(midpoint_responding(data, tolerance=0.25), [3 / 5, np.nan])


def test_all_missing_screening_retains_unavailable_response_style_scores() -> None:
    result = screen(
        np.full((3, 4), np.nan),
        indices=["acquiescence", "u3_poly", "midpoint"],
        strict=True,
    )
    assert len(result["scores"]) == 3
    np.testing.assert_array_equal(result["valid_index_counts"], [0, 0, 0])
    np.testing.assert_array_equal(result["flag_counts"], [0, 0, 0])


@pytest.mark.parametrize(
    ("baseline", "dtype"),
    [(2**60, np.int64), (-(2**60), np.int64), (2**64 - 16, np.uint64)],
)
@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("tolerance", [0, 0.49, 0.5, 1.0, np.float32(1.5)])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_large_integer_midpoints_preserve_exact_interval(
    baseline: int, dtype: type, width: int, tolerance: float, layout: str
) -> None:
    data = np.array([[baseline + value for value in range(8)], [baseline + width] * 8], dtype=dtype)
    data = data[:, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    midpoint = Fraction(2 * baseline + width, 2)
    radius = Fraction(float(tolerance))
    expected = [
        sum(abs(Fraction(int(value)) - midpoint) <= radius for value in row) / len(row)
        for row in data
    ]
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 3):
        actual = midpoint_responding(data, baseline, baseline + width, tolerance)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_inferred_large_integer_midpoints_match_combined_summary(dtype: type) -> None:
    data = np.array([[2**60, 2**60 + 1, 2**60 + 2]], dtype=dtype)
    np.testing.assert_array_equal(midpoint_responding(data), [1 / 3])
    np.testing.assert_array_equal(response_pattern(data)["midpoint"], [1 / 3])


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int64, np.uint64])
@pytest.mark.parametrize("bounds", [(-1.5, 3.5), (-1e30, 1e30), (1e30, 1e31), (-np.inf, np.inf)])
def test_integer_style_bounds_outside_dtype_range(dtype: type, bounds: tuple[float, float]) -> None:
    data = np.array([[0, 1, 2]], dtype=dtype)
    expected_extreme = sum(int(value) in bounds for value in data[0]) / 3
    lo, hi = bounds
    expected_midpoint = (
        0
        if not np.isfinite(bounds).all()
        else sum(Fraction(int(value)) == (Fraction(lo) + Fraction(hi)) / 2 for value in data[0]) / 3
    )
    np.testing.assert_array_equal(u3_poly(data, lo, hi), [expected_extreme])
    np.testing.assert_array_equal(midpoint_responding(data, lo, hi), [expected_midpoint])


def test_float_bounds_do_not_collapse_adjacent_integer_responses() -> None:
    data = np.array([[2**60, 2**60 + 1, 2**60 + 2]], dtype=np.int64)
    np.testing.assert_array_equal(u3_poly(data, float(2**60), float(2**60 + 512)), [1 / 3])
    np.testing.assert_array_equal(
        response_pattern(data, float(2**60), float(2**60 + 512))["extreme"], [1 / 3]
    )


def test_float_responses_do_not_match_unrepresentable_integer_endpoint() -> None:
    data = np.array([[float(2**60), float(2**60 + 512)]])
    np.testing.assert_array_equal(u3_poly(data, 2**60 + 1, 2**60 + 511), [0])


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_style_bounds_do_not_round_to_response_dtype(dtype: type) -> None:
    data = np.array([[1, 1, 1]], dtype=dtype)
    lower = np.nextafter(1.0, 2.0)
    upper = np.nextafter(lower, 2.0)
    np.testing.assert_array_equal(u3_poly(data, lower, upper), [0])
    np.testing.assert_array_equal(midpoint_responding(data, lower, upper), [0])
    summary = response_pattern(data, lower, upper)
    np.testing.assert_array_equal(summary["extreme"], [0])
    np.testing.assert_array_equal(summary["midpoint"], [0])


@pytest.mark.parametrize("sign", [-1, 1])
def test_subnormal_constant_scale_keeps_its_midpoint(sign: int) -> None:
    tiny = sign * np.nextafter(0.0, 1.0)
    data = [[tiny, tiny, np.nan], [np.nan] * 3]
    np.testing.assert_array_equal(midpoint_responding(data), [1, np.nan])
    np.testing.assert_array_equal(response_pattern(data)["midpoint"], [1, np.nan])


def test_screening_preserves_large_integer_response_styles() -> None:
    lower = 2**60
    data = np.array([[lower, lower + 1, lower + 2], [lower + 1] * 3, [lower] * 3])
    result = screen(
        data,
        indices=["midpoint", "u3_poly", "irv", "acquiescence"],
        options=IndexOptions(scale_min=lower, scale_max=lower + 2),
        strict=True,
    )
    for name, expected in {
        "midpoint": [1 / 3, 1, 0],
        "u3_poly": [2 / 3, 0, 1],
        "irv": [np.sqrt(2 / 3), 0, 0],
        "acquiescence": [0.5, 0.5, 0],
    }.items():
        np.testing.assert_array_equal(result["scores"][name], expected)
