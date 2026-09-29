"""Shared summaries preserve unavailable scores without repeated NaN reductions."""

import numpy as np
import pytest

from ier._summary import calculate_summary_stats
from ier.markov import markov_summary


@pytest.mark.parametrize("values", [np.array([]), np.full(5, np.nan)])
@pytest.mark.parametrize("suffix", ["", "_score"])
def test_empty_summaries_are_unavailable_without_warnings(values: np.ndarray, suffix: str) -> None:
    summary = calculate_summary_stats(values, suffix=suffix)
    assert set(summary) == {f"{name}{suffix}" for name in ("mean", "std", "min", "max", "median")}
    assert all(np.isnan(value) for value in summary.values())


@pytest.mark.parametrize(
    ("dtype", "missing"),
    [
        (np.int64, False),
        (np.float32, False),
        (np.float32, True),
        (np.float64, False),
        (np.float64, True),
    ],
)
def test_summaries_match_numpy_and_preserve_input(dtype: type, missing: bool) -> None:
    data = np.random.default_rng(13).integers(1, 6, size=101).astype(dtype)[::-2]
    if missing and dtype != np.int64:
        data[::3] = np.nan
    data.flags.writeable = False
    original = data.copy()
    actual = calculate_summary_stats(data)
    for name, reduction in (
        ("mean", np.nanmean),
        ("std", np.nanstd),
        ("min", np.nanmin),
        ("max", np.nanmax),
        ("median", np.nanmedian),
    ):
        np.testing.assert_allclose(actual[name], reduction(data), rtol=1e-7)
    np.testing.assert_array_equal(data, original)


def test_all_missing_markov_summary_is_unavailable() -> None:
    summary = markov_summary(np.full((3, 5), np.nan))
    for name in ("mean", "std", "min", "max", "median"):
        assert np.isnan(summary[name])
    assert summary["n_total"] == 3
