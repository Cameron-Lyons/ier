"""Bounded run-length kernels match their previous selection formulation exactly."""

import numpy as np
import pytest

from ier._response_sequences import true_run_lengths


def _reference_true_run_lengths(matches: np.ndarray) -> np.ndarray:
    """Count consecutive true values ending at each position in bounded rows."""
    width = matches.shape[1]
    positions = np.arange(1, width + 1, dtype=np.min_scalar_type(width))
    lengths = np.where(matches, 0, positions)
    np.maximum.accumulate(lengths, axis=1, out=lengths)
    np.subtract(positions, lengths, out=lengths)
    return lengths


def _assert_identical(matches: np.ndarray) -> None:
    expected = _reference_true_run_lengths(matches)
    actual = true_run_lengths(matches)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("width", [1, 2, 255, 256, 300, 2000])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_run_lengths_match_previous_kernel(width: int, layout: str) -> None:
    rng = np.random.default_rng(width)
    # Mix sparse, balanced, and dense matches with constant rows in each batch.
    probabilities = np.linspace(0.05, 0.98, 37)[:, None]
    matches = rng.random((37, width)) < probabilities
    matches[3] = True
    matches[4] = False
    matches[5, : width // 2] = True
    _assert_identical(np.array(matches, order=layout))


@pytest.mark.parametrize("width", [255, 256, 65_535, 65_536])
def test_uniform_rows_reach_the_dtype_boundary(width: int) -> None:
    matches = np.zeros((2, width), dtype=bool)
    matches[0] = True
    _assert_identical(matches)
    lengths = true_run_lengths(matches)
    assert int(lengths[0, -1]) == width
    assert not lengths[1].any()


def test_strided_and_empty_inputs_match_previous_kernel() -> None:
    rng = np.random.default_rng(31)
    backing = rng.random((40, 600)) < 0.7
    _assert_identical(backing[::2, ::3])
    _assert_identical(backing[::-3, 1::2])
    _assert_identical(np.zeros((4, 0), dtype=bool))
    _assert_identical(np.zeros((0, 9), dtype=bool))
