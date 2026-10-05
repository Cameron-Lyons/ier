"""Half means are correctly rounded, so equal exact means are equal floating-point means."""

import math
from fractions import Fraction

import numpy as np
import pytest

from ier._scale_halves import HalfMeans, group_batches


def _exact_means(data: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Round every group's exact mean once; infinities follow IEEE sums."""
    means = np.full((len(data), len(groups)), np.nan)
    # Integers stay exact; floating responses widen exactly to float64.
    rows = data if data.dtype.kind in "biu" else np.asarray(data, dtype=float)
    for row_number, row in enumerate(rows):
        for group_number, columns in enumerate(groups):
            observed = [value for value in row[columns].tolist() if not math.isnan(value)]
            infinite = [value for value in observed if math.isinf(value)]
            if infinite:
                same = all(value == infinite[0] for value in infinite)
                means[row_number, group_number] = infinite[0] if same else np.nan
            elif observed:
                exact = sum(map(Fraction, observed), Fraction(0)) / len(observed)
                means[row_number, group_number] = float(exact)
    return means


def _assert_correctly_rounded(data: np.ndarray, groups: list[np.ndarray]) -> None:
    actual = HalfMeans(data)(group_batches(groups), len(groups))

    np.testing.assert_array_equal(actual, _exact_means(data, groups))


def _mixed_binade_integers(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Scaled integers whose sums need more than 53 bits, so many means are exact ties."""
    significands = rng.integers(2**52, 2**53, size=shape).astype(float)
    values = np.ldexp(significands, rng.integers(-1, 3, size=shape))
    return np.ldexp(values, rng.integers(-80, 80, size=(shape[0], 1)) - 54)


@pytest.mark.parametrize(
    "kind",
    ["tenths", "sevenths", "normal", "ties", "signed", "huge", "tiny", "offset", "float32"],
)
def test_half_means_round_exact_means_once(kind: str) -> None:
    rng = np.random.default_rng(len(kind))
    shape = (120, 11)
    data = {
        "tenths": rng.integers(0, 10, size=shape) / 10,
        "sevenths": (rng.integers(1, 8, size=shape) - 1) / 6,
        "normal": rng.normal(size=shape),
        "ties": _mixed_binade_integers(rng, shape),
        "signed": rng.choice([0.1, 0.7, 1 / 3], size=shape) * rng.choice([-1, 1], size=shape),
        "huge": rng.uniform(-1, 1, size=shape) * 1.7e308,
        "tiny": rng.uniform(-1, 1, size=shape) * 1e-305,
        "offset": 1e15 + rng.integers(0, 10, size=shape) / 10,
        "float32": (rng.integers(0, 10, size=shape) / 10).astype(np.float32),
    }[kind]
    if data.dtype == np.float64:
        data[rng.random(shape) < 0.15] = np.nan
    groups = [np.arange(0, 11, 2), np.arange(1, 11, 2), np.array([3, 7, 9]), np.array([5])]

    _assert_correctly_rounded(data, groups)


def test_repeated_and_reordered_responses_have_identical_means() -> None:
    data = np.tile([0.7, 0.7, 0.7, 0.7, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1], (3, 1))
    groups = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

    means = HalfMeans(data)(group_batches(groups), 4)

    assert (0.7 + 0.7 + 0.7) / 3 != 0.7
    np.testing.assert_array_equal(means, np.tile([0.7, 0.7, means[0, 2], means[0, 2]], (3, 1)))


def test_exact_midpoints_round_to_even() -> None:
    rng = np.random.default_rng(1)
    data = _mixed_binade_integers(rng, (2000, 7))
    counts = rng.integers(2, 8, size=len(data))
    data[np.arange(7) >= counts[:, None]] = np.nan
    groups = [np.arange(7)]
    exact = _exact_means(data, groups)[:, 0]
    naive = np.nanmean(data, axis=1)

    _assert_correctly_rounded(data, groups)
    # The data exercise rounding: naive sums miss many correctly rounded means.
    assert np.count_nonzero(naive != exact) > 100


@pytest.mark.parametrize("span", [24, 30, 1100])
def test_rows_spanning_many_binades_are_rounded_exactly(span: int) -> None:
    rng = np.random.default_rng(span)
    significands = rng.integers(2**52, 2**53, size=(30, 8)).astype(float)
    exponents = rng.integers(-min(span, 1020), 1, size=(30, 8))
    exponents[:, 0] = 0
    data = np.ldexp(significands, exponents - 53) * rng.choice([-1.0, 1.0], size=(30, 8))
    if span > 1000:
        data[:, 1] = 5e-324
        data[:, 2] = 1e300

    _assert_correctly_rounded(data, [np.arange(0, 8, 2), np.arange(1, 8, 2), np.arange(8)])


def test_infinite_and_subnormal_responses() -> None:
    data = np.array(
        [
            [np.inf, 1.0, 0.1, 0.2, 0.3, 1e308],
            [np.inf, -np.inf, 0.7, 0.7, 0.7, 2.0],
            [5e-324, 1e-323, 0.0, -5e-324, 1.5e-323, 0.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [-np.inf, 0.5, np.nan, np.nan, np.nan, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    groups = [np.array([0, 1]), np.array([2, 3, 4]), np.array([0, 5]), np.array([1, 3, 5])]

    _assert_correctly_rounded(data, groups)


def test_integer_and_boolean_means_are_exact() -> None:
    rng = np.random.default_rng(3)
    integers = rng.integers(-(2**62), 2**62, size=(50, 6))
    groups = [np.array([0, 2, 4]), np.array([1, 3, 5])]

    _assert_correctly_rounded(integers, groups)
    _assert_correctly_rounded(integers % 2 == 0, groups)


def test_split_halves_are_reused_across_calls() -> None:
    rng = np.random.default_rng(4)
    data = rng.integers(0, 10, size=(40, 9)) / 10
    data[rng.random(data.shape) < 0.2] = np.nan
    half_means = HalfMeans(data)

    for _ in range(3):
        groups = list(np.split(rng.permutation(9), [2, 5, 6]))
        np.testing.assert_array_equal(
            half_means(group_batches(groups), len(groups)), _exact_means(data, groups)
        )
