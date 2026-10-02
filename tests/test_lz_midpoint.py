"""Exact category and scalar likelihood proofs for polytomous person-fit scores."""

import math
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from ier import lz, lz_flag
from ier.lz import _dichotomize


def _exact_binary(data: np.ndarray) -> np.ndarray:
    """Compare each observed category with the rational midpoint of the extrema."""
    integral = data.dtype.kind in "iu"
    observed = data[~np.isnan(data)]
    values = [Fraction(int(value)) if integral else Fraction(float(value)) for value in observed]
    midpoint = (min(values) + max(values)) / 2
    result = np.full(data.shape, np.nan)
    for position in np.ndindex(data.shape):
        value = data[position]
        if not np.isnan(value):
            rational = Fraction(int(value)) if integral else Fraction(float(value))
            result[position] = rational > midpoint
    return result


def _scalar_scores(binary: np.ndarray, *, na_rm: bool) -> np.ndarray:
    """Evaluate the Bernoulli log-likelihood moments without library kernels."""
    difficulties = [-1.3, 0.1, -0.7, 1.4, 0.8, -0.3]
    discriminations = [0.6, 1.1, 1.5, 0.8, 1.9, 1.2]
    abilities = [0.3, -0.4, 0.8]
    scores = []
    for row, ability in zip(binary, abilities, strict=True):
        if not na_rm and np.isnan(row).any():
            scores.append(math.nan)
            continue
        numerator = 0.0
        variance = 0.0
        for response, difficulty, discrimination in zip(
            row, difficulties, discriminations, strict=True
        ):
            if np.isnan(response):
                continue
            probability = 1 / (1 + math.exp(-discrimination * (ability - difficulty)))
            log_odds = math.log(probability) - math.log1p(-probability)
            numerator += (response - probability) * log_odds
            variance += probability * (1 - probability) * log_odds * log_odds
        scores.append(numerator / math.sqrt(variance))
    return np.asarray(scores)


def _data(categories: np.ndarray, layout: str, *, missing: bool) -> np.ndarray:
    positions = [[0, 1, 2, 3, 1, 2], [3, 2, 1, 0, 2, 1], [0, 3, 0, 3, 2, 1]]
    data = categories[positions]
    if missing:
        data[0, 4] = np.nan
        data[1, 2] = np.nan
    if layout == "strided":
        backing = np.zeros((6, 12), dtype=data.dtype)
        backing[::2, ::2] = data
        data = backing[::2, ::2]
    else:
        data = np.array(data, order=layout)
    data.flags.writeable = False
    return data


INTEGER_CATEGORIES = [
    np.array([124, 125, 126, 127], dtype=np.int8),
    np.array([-128, -127, -126, -125], dtype=np.int8),
    np.array([252, 253, 254, 255], dtype=np.uint8),
    np.array([2**53, 2**53 + 1, 2**53 + 2, 2**53 + 3], dtype=np.int64),
    np.array([-(2**53) - 3, -(2**53) - 2, -(2**53) - 1, -(2**53)], dtype=np.int64),
    np.arange(4, dtype=np.int64) + (2**63 - 4),
    np.arange(4, dtype=np.int64) + np.int64(-(2**63)),
    np.array([-(2**63), -1, 0, 2**63 - 1], dtype=np.int64),
    np.array([0, 1, 2**64 - 2, 2**64 - 1], dtype=np.uint64),
    np.arange(4, dtype=np.uint64) + (2**64 - 4),
]


@pytest.mark.parametrize("categories", INTEGER_CATEGORIES)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_integer_labels_preserve_exact_midpoint_and_public_scores(
    categories: np.ndarray, layout: str
) -> None:
    data = _data(categories, layout, missing=False)
    original = data.copy()
    binary = _exact_binary(data)
    expected = _scalar_scores(binary, na_rm=True)
    with np.errstate(all="raise"), patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 13):
        prepared = _dichotomize(data)
        scores, flags = lz_flag(
            data,
            difficulty=[-1.3, 0.1, -0.7, 1.4, 0.8, -0.3],
            discrimination=[0.6, 1.1, 1.5, 0.8, 1.9, 1.2],
            theta=[0.3, -0.4, 0.8],
            threshold=-0.3,
        )
        calibrated = lz(data)
    np.testing.assert_array_equal(prepared, binary)
    np.testing.assert_allclose(scores, expected, rtol=3e-15, atol=1e-15)
    np.testing.assert_array_equal(flags, expected < -0.3)
    np.testing.assert_allclose(calibrated, lz(binary), rtol=0, atol=0)
    np.testing.assert_array_equal(data, original)


def _float_categories(dtype: type) -> list[np.ndarray]:
    limit = np.finfo(dtype).max
    tiny = np.finfo(dtype).smallest_subnormal
    # Both signs of overflow, cancellation, subnormal midpoint rounding, and
    # adjacent categories whose midpoint rounds onto the upper category.
    return [
        np.asarray([limit / 2, limit * 0.6, limit * 0.8, limit], dtype=dtype),
        np.asarray([-limit, -limit * 0.8, -limit * 0.6, -limit / 2], dtype=dtype),
        np.asarray([-limit, -1, 1, limit], dtype=dtype),
        np.asarray([0, tiny, 2 * tiny, 3 * tiny], dtype=dtype),
        np.asarray([-3 * tiny, -2 * tiny, -tiny, 0], dtype=dtype),
        np.asarray(
            [1, 1, np.nextafter(dtype(1), dtype(2)), np.nextafter(dtype(1), dtype(2))], dtype=dtype
        ),
    ]


FLOAT_CATEGORIES = [
    categories
    for dtype in [np.float16, np.float32, np.float64]
    for categories in _float_categories(dtype)
]


@pytest.mark.parametrize("categories", FLOAT_CATEGORIES)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("na_rm", [False, True])
def test_floating_labels_preserve_exact_midpoint_and_public_scores(
    categories: np.ndarray, layout: str, na_rm: bool
) -> None:
    data = _data(categories, layout, missing=True)
    original = data.copy()
    binary = _exact_binary(data)
    expected = _scalar_scores(binary, na_rm=na_rm)
    with np.errstate(all="raise"), patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 13):
        prepared = _dichotomize(data)
        scores, flags = lz_flag(
            data,
            difficulty=[-1.3, 0.1, -0.7, 1.4, 0.8, -0.3],
            discrimination=[0.6, 1.1, 1.5, 0.8, 1.9, 1.2],
            theta=[0.3, -0.4, 0.8],
            threshold=-0.3,
            na_rm=na_rm,
        )
    np.testing.assert_array_equal(prepared, binary)
    np.testing.assert_allclose(scores, expected, rtol=3e-15, atol=1e-15)
    np.testing.assert_array_equal(flags, expected < -0.3)
    np.testing.assert_array_equal(data, original)
