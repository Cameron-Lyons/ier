"""Chi-square lower tails, exact special cases, and array contracts."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from ier import mahad_qqplot
from ier._statistics import _regularized_gamma_pair, chi_square_quantile, chi_square_quantiles

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
