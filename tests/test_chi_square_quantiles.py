"""Chi-square lower tails, exact special cases, and array contracts."""

import math
import sys
from contextlib import AbstractContextManager, nullcontext
from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

from ier import mahad_qqplot
from ier._statistics import (
    _QUANTILE_ARRAY_MIN_ELEMENTS,
    _chi_square_quantiles_general,
    _chi_square_tail_pair,
    _regularized_gamma_pair,
    _regularized_gamma_pair_array,
    chi_square_quantile,
    chi_square_quantiles,
)

_DENSE_DEGREES = [1, 3, 4, 5, 10, 20, 40, 100, 120, 500, 1000]
_DENSE_PROBABILITIES = np.unique(
    np.concatenate(
        (np.logspace(-49, -1, 300), np.linspace(0.001, 0.999, 801), 1 - np.logspace(-16, -1, 100))
    )
)

# Independent 70-digit Decimal gamma-series evaluation with 110 bisections in
# log(quantile). Inputs are the exact represented binary float probabilities.
_LOWER_TAIL_REFERENCE = [
    (4, 1e-50, 2.8284271247461903e-25),
    (4, 1e-250, 2.8284271247461904e-125),
    (4, 1e-310, 2.828427124746186e-155),
    (4, 5e-324, 6.286911138810515e-162),
    (10, 1e-50, 5.210342169620935e-10),
    (10, 1e-250, 5.210342169394703e-50),
    (10, 1e-310, 5.2103421693947e-62),
    (10, 5e-324, 1.1366406302869085e-64),
    (100, 1e-50, 4.05460334364102),
    (100, 1e-250, 0.00038966657316455335),
    (100, 1e-310, 2.4586210624068186e-05),
    (100, 5e-324, 1.3321932933867034e-05),
    (1000, 1e-50, 471.3770861141663),
    (1000, 1e-250, 134.06014293366133),
    (1000, 1e-310, 98.1121748656284),
    (1000, 5e-324, 91.69128534930573),
]


def _force_array_solver() -> AbstractContextManager[object]:
    """Send even a single general probability through the array Newton solver."""
    return patch("ier._statistics._QUANTILE_ARRAY_MIN_ELEMENTS", 0)


@pytest.mark.parametrize(("df", "probability", "expected"), _LOWER_TAIL_REFERENCE)
def test_extreme_lower_tails_match_high_precision_reference(
    df: int, probability: float, expected: float
) -> None:
    np.testing.assert_allclose(chi_square_quantile(probability, df), expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("probability", [1e-150, 1e-155, 1.4e-162, 1e-200, 1e-320, 5e-324])
def test_one_degree_of_freedom_retains_subnormal_quantiles(probability: float) -> None:
    # The lower-tail correction to (pi/2)*p**2 is far below double precision.
    with localcontext() as context:
        context.prec = 60
        half_pi = Decimal("1.570796326794896619231321691639751442098584699687552910487472")
        expected = float(half_pi * Decimal.from_float(probability) ** 2)
    np.testing.assert_allclose(chi_square_quantile(probability, 1), expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_two_degree_quantiles_use_the_exponential_identity(layout: str) -> None:
    probabilities = np.array(
        [
            [0.0, 5e-324, 1e-320, 1e-310, 1e-50],
            [0.01, 0.5, 0.99, np.nextafter(1.0, 0.0), 1.0],
        ]
    )
    probabilities = (
        probabilities[::-1, ::-1] if layout == "strided" else np.array(probabilities, order=layout)
    )
    original = probabilities.copy()
    probabilities.flags.writeable = False
    expected = np.array(
        [-2 * math.log1p(-float(p)) if p < 1 else math.inf for p in probabilities.flat]
    ).reshape(probabilities.shape)
    actual = chi_square_quantiles(probabilities, np.uint64(2))
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
    for probability, quantile in zip(probabilities.flat, expected.flat, strict=True):
        assert chi_square_quantile(float(probability), 2) == quantile
    np.testing.assert_array_equal(probabilities, original)


@pytest.mark.parametrize("df", [1, 2, 3])
@pytest.mark.parametrize("probability", [np.float32(1e-30), np.float16(0.25), np.float32(0.75)])
def test_numpy_probabilities_use_double_precision_arithmetic(probability: float, df: int) -> None:
    actual = chi_square_quantile(probability, df)
    assert isinstance(actual, float)
    assert actual > 0
    assert actual == chi_square_quantile(float(probability), df)


@pytest.mark.parametrize("df", [1, 2, 3, 10, 1000])
def test_array_quantiles_match_scalar_values_across_the_domain(df: int) -> None:
    probabilities = np.concatenate(
        (
            [0.0, 5e-324],
            np.geomspace(1e-320, 1e-3, 101),
            np.linspace(0.002, 0.999, 101),
            [np.nextafter(1.0, 0.0), 1.0],
        )
    )
    actual = chi_square_quantiles(probabilities, df)
    expected = [chi_square_quantile(float(p), df) for p in probabilities]
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
    assert np.all(actual[1:] >= actual[:-1])


@pytest.mark.parametrize("df", [0, -1, True, np.bool_(True), 1.5, 2.0, np.nan, np.inf, "2", None])
def test_invalid_degrees_fail_even_at_boundaries_and_for_empty_arrays(df: object) -> None:
    for probability in (0.0, 0.5, 1.0):
        with pytest.raises(ValueError, match="degrees_of_freedom.*positive integer"):
            chi_square_quantile(probability, df)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="degrees_of_freedom.*positive integer"):
        chi_square_quantiles(np.array([]), df)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("df", [2, 10])
def test_array_validation_precedes_special_case_evaluation(value: float, df: int) -> None:
    with pytest.raises(ValueError, match="probability.*finite"):
        chi_square_quantiles(np.array([0.0, 0.5, value, 1.0]), df)


@pytest.mark.parametrize("df", [2, 3])
def test_empty_and_scalar_array_shapes_are_preserved(df: int) -> None:
    assert chi_square_quantiles(np.empty((2, 0, 3)), df).shape == (2, 0, 3)
    actual = chi_square_quantiles(np.array(0.5), df)
    assert actual.shape == ()
    np.testing.assert_allclose(actual, chi_square_quantile(0.5, df))


def test_two_item_mahalanobis_qq_quantiles_preserve_missing_response_alignment() -> None:
    data = np.random.default_rng(7).normal(size=(37, 2))
    data[[2, 9], 0] = np.nan
    theoretical, observed = mahad_qqplot(data, na_rm=True)
    probabilities = (np.arange(len(observed)) + 0.5) / len(observed)
    expected = [-2 * math.log1p(-float(p)) for p in probabilities]
    np.testing.assert_allclose(theoretical, expected, rtol=2e-15)
    assert len(observed) == 35
    assert np.all(np.diff(observed) >= 0)


@pytest.mark.parametrize(
    ("shape", "value"), [(0, 1), (np.nan, 1), (np.inf, 1), (1, -1), (1, np.nan)]
)
def test_gamma_domain_validation_rejects_undefined_arguments(shape: float, value: float) -> None:
    with pytest.raises(ValueError, match="shape.*positive.*value.*non-negative"):
        _regularized_gamma_pair(shape, value)


@pytest.mark.parametrize("df", _DENSE_DEGREES)
def test_dense_grid_array_quantiles_equal_scalar_quantiles(df: int) -> None:
    actual = chi_square_quantiles(_DENSE_PROBABILITIES, df)
    expected = [chi_square_quantile(p, df) for p in _DENSE_PROBABILITIES.tolist()]
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.diff(actual) > 0)


@pytest.mark.parametrize("df", _DENSE_DEGREES)
def test_dense_grid_quantiles_invert_the_independent_scalar_cdf(df: int) -> None:
    shape = df / 2.0
    probabilities = _DENSE_PROBABILITIES.tolist()
    quantiles = chi_square_quantiles(_DENSE_PROBABILITIES, df).tolist()
    for probability, quantile in zip(probabilities, quantiles, strict=True):
        cdf, survival = _chi_square_tail_pair(quantile, df)
        if probability <= 0.5:
            tail, residual = probability, cdf - probability
        else:
            tail, residual = 1.0 - probability, survival - (1.0 - probability)
        # Newton stops at a 5e-14 relative tail residual or a 5e-14 relative
        # step. The CDF also rounds its log-scale terms, which grow with df.
        half = quantile / 2.0
        log_terms = half + shape * abs(math.log(half)) + abs(math.lgamma(shape))
        tolerance = tail * (5e-14 + 2.0 * sys.float_info.epsilon * log_terms)
        assert abs(residual) <= tolerance, (probability, quantile)


@pytest.mark.parametrize("df", [1, 7])
def test_bounded_quantile_batches_preserve_values_and_order(df: int) -> None:
    probabilities = np.random.default_rng(df).permutation(_DENSE_PROBABILITIES).reshape(-1, 1)
    expected = chi_square_quantiles(probabilities, df)
    with patch("ier._statistics._QUANTILE_BATCH_ELEMENTS", 101):
        np.testing.assert_array_equal(chi_square_quantiles(probabilities, df), expected)


@pytest.mark.parametrize("df", [1, 3, 20, 1000])
def test_general_quantiles_ignore_strict_floating_point_policies(df: int) -> None:
    probabilities = np.concatenate(([0.0, 5e-324, 1e-300], _DENSE_PROBABILITIES, [1.0]))
    expected = chi_square_quantiles(probabilities, df)
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(chi_square_quantiles(probabilities, df), expected)
        assert chi_square_quantile(0.3, df) == chi_square_quantiles(np.array([0.3]), df)[0]
        with _force_array_solver():
            assert chi_square_quantile(0.3, df) == chi_square_quantiles(np.array([0.3]), df)[0]


@pytest.mark.parametrize("array_solver", [False, True])
@pytest.mark.parametrize("df", [1, 3, 7])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_mixed_special_and_general_quantiles_preserve_shapes(
    df: int, layout: str, array_solver: bool
) -> None:
    probabilities = np.array(
        [
            [0.0, 5e-324, 1e-60, 1e-50, 1e-8, 2e-8],
            [0.05, 0.1, 0.5, 0.75, np.nextafter(1.0, 0.0), 1.0],
        ]
    )
    probabilities = (
        probabilities[::-1, ::-2] if layout == "strided" else np.array(probabilities, order=layout)
    )
    probabilities.flags.writeable = False
    original = probabilities.copy()
    with _force_array_solver() if array_solver else nullcontext():
        actual = chi_square_quantiles(probabilities, df)
    expected = [chi_square_quantile(float(p), df) for p in probabilities.flat]
    assert actual.shape == probabilities.shape
    np.testing.assert_array_equal(actual, np.reshape(expected, probabilities.shape))
    np.testing.assert_array_equal(probabilities, original)


@pytest.mark.parametrize("df", [1, 5])
def test_general_empty_and_zero_dimensional_arrays(df: int) -> None:
    assert chi_square_quantiles(np.empty((3, 0)), df).shape == (3, 0)
    actual = chi_square_quantiles(np.array(0.05), df)
    assert actual.shape == ()
    assert actual == chi_square_quantile(0.05, df)


@pytest.mark.parametrize("array_solver", [False, True])
def test_newton_iteration_limit_fails_for_scalar_and_array_quantiles(array_solver: bool) -> None:
    with (
        patch("ier._statistics._QUANTILE_MAX_ITERATIONS", 1),
        _force_array_solver() if array_solver else nullcontext(),
        pytest.raises(ArithmeticError, match="chi-square quantile did not converge"),
    ):
        chi_square_quantiles(np.linspace(0.01, 0.99, 9), 5)
    with (
        patch("ier._statistics._QUANTILE_MAX_ITERATIONS", 1),
        pytest.raises(ArithmeticError, match="chi-square quantile did not converge"),
    ):
        chi_square_quantile(0.3, 5)


@pytest.mark.parametrize(("value", "message"), [(0.5, "series"), (40.0, "continued fraction")])
def test_array_gamma_iteration_limits_fail_like_the_scalar_recurrences(
    value: float, message: str
) -> None:
    values = np.array([value, value * 1.5])
    with patch("ier._statistics._GAMMA_MAX_ITERATIONS", 2):
        with pytest.raises(ArithmeticError, match=message):
            _regularized_gamma_pair_array(2.5, values)
        with pytest.raises(ArithmeticError, match=message):
            _regularized_gamma_pair(2.5, value)


@pytest.mark.parametrize("array_solver", [False, True])
@pytest.mark.parametrize("probability", [0.2, 0.7])
def test_unbracketed_quantiles_raise_instead_of_overflowing(
    probability: float, array_solver: bool
) -> None:
    def never_reaches_target(shape: float, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(values), np.ones_like(values)

    with (
        patch("ier._statistics._regularized_gamma_pair_array", never_reaches_target),
        patch("ier._statistics._chi_square_tail_pair", return_value=(0.0, 1.0)),
        _force_array_solver() if array_solver else nullcontext(),
        np.errstate(all="raise"),
    ):
        with pytest.raises(ArithmeticError, match="could not bracket"):
            chi_square_quantiles(np.array([probability, 0.5]), 4)
        with pytest.raises(ArithmeticError, match="could not bracket"):
            chi_square_quantile(probability, 4)


@pytest.mark.parametrize("shape", [0.5, 1.5, 10.0, 500.0])
def test_array_gamma_pairs_equal_scalar_tails_including_zero(shape: float) -> None:
    values = np.array([0.0, 5e-324, 1e-300, 1e-8, shape / 3, shape + 1, 2 * shape + 30])
    lower, upper = _regularized_gamma_pair_array(shape, values)
    expected = np.array([_regularized_gamma_pair(shape, float(value)) for value in values])
    assert (lower[0], upper[0]) == (0.0, 1.0)
    np.testing.assert_array_equal(lower, expected[:, 0])
    np.testing.assert_array_equal(upper, expected[:, 1])


@pytest.mark.parametrize("df", [3, 10, 50])
def test_small_general_sets_use_the_scalar_solver(df: int) -> None:
    # Array iterations cost more than scalar solves for a few probabilities.
    for count, array_calls in ((1, 0), (_QUANTILE_ARRAY_MIN_ELEMENTS - 1, 0)):
        probabilities = (np.arange(1, count + 1) - 0.5) / count
        with patch(
            "ier._statistics._chi_square_quantiles_general", wraps=_chi_square_quantiles_general
        ) as solver:
            actual = chi_square_quantiles(probabilities, df)
        assert solver.call_count == array_calls
        expected = [chi_square_quantile(p, df) for p in probabilities.tolist()]
        np.testing.assert_array_equal(actual, expected)
    probabilities = (np.arange(1, _QUANTILE_ARRAY_MIN_ELEMENTS + 1) - 0.5) / (
        _QUANTILE_ARRAY_MIN_ELEMENTS
    )
    with patch(
        "ier._statistics._chi_square_quantiles_general", wraps=_chi_square_quantiles_general
    ) as solver:
        actual = chi_square_quantiles(probabilities, df)
    assert solver.call_count == 1
    np.testing.assert_array_equal(actual, [chi_square_quantile(p, df) for p in probabilities])


def test_few_general_values_among_special_cases_use_the_scalar_solver() -> None:
    # One degree of freedom solves p <= 1e-8 and p >= 0.1 directly.
    probabilities = np.concatenate((np.linspace(0.1, 0.999, 5000), [1e-9, 0.01, 0.05], [0.0, 1.0]))
    with patch(
        "ier._statistics._chi_square_quantiles_general", wraps=_chi_square_quantiles_general
    ) as solver:
        actual = chi_square_quantiles(probabilities, 1)
    solver.assert_not_called()
    np.testing.assert_array_equal(actual, [chi_square_quantile(p, 1) for p in probabilities])


@pytest.mark.parametrize("count", [1, 5, 20, _QUANTILE_ARRAY_MIN_ELEMENTS - 1])
@pytest.mark.parametrize("df", [1, 3, 4, 10, 50, 1000])
def test_scalar_and_array_solvers_agree_for_small_sets(count: int, df: int) -> None:
    probabilities = (np.arange(1, count + 1) - 0.5) / count
    default = chi_square_quantiles(probabilities, df)
    with _force_array_solver():
        array = chi_square_quantiles(probabilities, df)
    np.testing.assert_array_equal(default, array)
    np.testing.assert_array_equal(default, [chi_square_quantile(p, df) for p in probabilities])
