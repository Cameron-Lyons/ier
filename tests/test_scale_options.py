"""Public response-scale options share one strict scalar contract."""

from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from ier import (
    acquiescence,
    mad,
    midpoint_responding,
    response_pattern,
    reverse_score,
    semantic_ant,
    u3_poly,
)


def _score(name: str, **bounds: Any) -> object:
    data = [[1, 5], [2, 4]]
    scorers = {
        "acquiescence": lambda: acquiescence(data, **bounds),
        "mad": lambda: mad(data, item_pairs=[(0, 1)], **bounds),
        "midpoint": lambda: midpoint_responding(data, **bounds),
        "response_pattern": lambda: response_pattern(data, **bounds),
        "reverse_score": lambda: reverse_score(data, [0], **bounds),
        "semantic_ant": lambda: semantic_ant(data, [(0, 1)], **bounds),
        "u3_poly": lambda: u3_poly(data, **bounds),
    }
    return scorers[name]()


@pytest.mark.parametrize(
    "name",
    [
        "acquiescence",
        "mad",
        "midpoint",
        "response_pattern",
        "reverse_score",
        "semantic_ant",
        "u3_poly",
    ],
)
@pytest.mark.parametrize("parameter", ["scale_min", "scale_max"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        "1",
        np.array([1.0]),
        np.nan,
        Decimal("NaN"),
        Decimal("sNaN"),
        Fraction(10**400),
    ],
)
def test_invalid_public_scale_bounds_raise_value_error(
    name: str, parameter: str, value: object
) -> None:
    bounds: dict[str, object] = {"scale_min": 1, "scale_max": 5, parameter: value}
    with pytest.raises(ValueError, match=parameter):
        _score(name, **bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        {"scale_min": np.array(1), "scale_max": np.array(5)},
        {"scale_min": Decimal("1"), "scale_max": Decimal("5")},
        {"scale_min": Fraction(1), "scale_max": Fraction(5)},
    ],
)
def test_numeric_scalar_bound_containers_remain_supported(bounds: dict[str, object]) -> None:
    expected = _score("u3_poly", scale_min=1, scale_max=5)
    np.testing.assert_array_equal(_score("u3_poly", **bounds), expected)
    np.testing.assert_array_equal(
        _score("reverse_score", **bounds), _score("reverse_score", scale_min=1, scale_max=5)
    )


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), "0.5", np.array([0.5]), np.nan, np.inf, Decimal("sNaN")],
)
def test_midpoint_tolerance_rejects_nonfinite_or_nonscalar_values(value: object) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        midpoint_responding([[1, 3, 5]], 1, 5, tolerance=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [np.array(0.5), Decimal("0.5"), Fraction(1, 2)])
def test_midpoint_tolerance_accepts_real_numeric_scalars(value: object) -> None:
    np.testing.assert_array_equal(
        midpoint_responding([[1, 2, 3, 4, 5]], 1, 5, tolerance=value),  # type: ignore[arg-type]
        [1 / 5],
    )
