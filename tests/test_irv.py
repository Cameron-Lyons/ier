"""Split IRV validation, section weighting, and bounded reductions."""

from unittest.mock import patch

import numpy as np
import pytest

from ier import irv


@pytest.mark.parametrize(
    "value", [0, -1, 1.5, 2.0, True, np.bool_(True), np.nan, np.inf, "2", None]
)
def test_split_count_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="num_split.*positive integer"):
        irv([[1, 2, 3, 4]], split=True, num_split=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1.5, 1.0, True, np.bool_(True), np.nan, np.inf, "1", None])
@pytest.mark.parametrize("split", [False, True])
def test_split_points_require_integer_positions(value: object, split: bool) -> None:
    with pytest.raises(ValueError, match="split_points.*integer"):
        irv([[1, 2, 3, 4]], split=split, split_points=[0, value, 4])  # type: ignore[list-item]


@pytest.mark.parametrize("count", [np.int32(3), np.uint64(3), np.int64(100)])
def test_numpy_split_counts_preserve_unequal_sections(count: int) -> None:
    data = np.arange(34, dtype=float).reshape(2, 17)
    expected = np.mean(
        [np.std(chunk, axis=1) for chunk in np.array_split(data, count, axis=1) if chunk.size],
        axis=0,
    )
    np.testing.assert_allclose(irv(data, split=True, num_split=count), expected)


@pytest.mark.parametrize("points", [[0, 3, 2, 4], [0, 2, 2, 4], [0, -1, 4], [0, 5, 4]])
def test_split_points_must_be_strictly_increasing(points: list[int]) -> None:
    with pytest.raises(ValueError, match="ascending order"):
        irv([[1, 2, 3, 4]], split=True, split_points=points)


def test_unused_split_count_is_ignored() -> None:
    data = np.arange(12).reshape(2, 6)
    np.testing.assert_array_equal(irv(data, num_split=0), irv(data))
    np.testing.assert_allclose(
        irv(data, split=True, num_split=0, split_points=[np.int64(0), np.int64(2), np.int64(6)]),
        np.mean([np.std(data[:, :2], axis=1), np.std(data[:, 2:], axis=1)], axis=0),
    )


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sections", [1, 2, 3, 17, 20, [0, 1, 3, 5, 9, 13, 17]])
def test_split_scores_match_independent_section_reference(
    ignore_nan: bool, layout: str, dtype: type, sections: int | list[int]
) -> None:
    rng = np.random.default_rng(17)
    data = rng.normal(size=(23, 17)).astype(dtype)
    data[0] = np.nan
    data[1, 3] = np.nan
    data[2, :8] = np.nan
    data[3] = 4
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    if isinstance(sections, list):
        chunks = [
            data[:, first:last] for first, last in zip(sections[:-1], sections[1:], strict=True)
        ]
        options = {"split_points": sections}
    else:
        chunks = [chunk for chunk in np.array_split(data, sections, axis=1) if chunk.size]
        options = {"num_split": sections}
    expected_sections = []
    for chunk in chunks:
        scores = []
        for row in chunk:
            observed = row[~np.isnan(row)] if ignore_nan else row
            scores.append(np.std(observed, dtype=float) if observed.size else np.nan)
        expected_sections.append(scores)
    expected = np.mean(expected_sections, axis=0, dtype=float)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 101):
        actual = irv(data, split=True, na_rm=ignore_nan, **options)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(data, original)


def test_an_unavailable_section_keeps_the_split_score_unavailable() -> None:
    data = [[1, 3, np.nan, np.nan], [1, 3, np.nan, 4]]
    np.testing.assert_array_equal(irv(data, split=True, num_split=2), [np.nan, 0.5])
    assert np.isnan(irv(data, split=True, num_split=2, na_rm=False)).all()


def test_excess_splits_do_not_construct_empty_sections() -> None:
    with patch("ier.irv.np.array_split", side_effect=AssertionError("empty sections constructed")):
        scores = irv([[1, 2, 3], [1, np.nan, 3]], split=True, num_split=10**100)
    np.testing.assert_array_equal(scores, [0, np.nan])


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_singleton_sections_require_finite_observations(ignore_nan: bool) -> None:
    data = [[1, 2], [1, np.nan], [1, np.inf]]
    np.testing.assert_array_equal(
        irv(data, split=True, num_split=2, na_rm=ignore_nan), [0, np.nan, np.nan]
    )


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_split_workspaces_are_bounded_by_the_row_budget(layout: str) -> None:
    from ier._row_statistics import _row_mean_std_block

    data = np.random.default_rng(4).normal(size=(100, 17))
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 60),
        patch("ier.irv._row_mean_std_block", wraps=_row_mean_std_block) as blocks,
    ):
        irv(data, split=True, num_split=8)
    assert blocks.call_count > 1
    assert all(call.args[0].size <= 60 for call in blocks.call_args_list)


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("sections", [2, [0, 2, 6]])
def test_split_irv_preserves_extreme_section_deviations(
    ignore_nan: bool, layout: str, sections: int | list[int]
) -> None:
    largest = np.finfo(float).max
    data = np.array(
        [
            [largest, -largest] * 3,
            [1e-200, -1e-200] * 3,
            [1e308, 1e308] * 3,
            [np.nan, 1e308, np.nan, 1e308, 1e308, 1e308],
        ]
    )
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    if isinstance(sections, list):
        points = sections
        options = {"split_points": sections}
    else:
        points = [0, 3, 6]
        options = {"num_split": sections}
    # All nonconstant sections have entries +/- magnitude; compute their
    # normalized population deviations before restoring the magnitude.
    expected = []
    for row in data:
        if np.isnan(row).any():
            expected.append(0.0 if ignore_nan else np.nan)
            continue
        magnitude = np.max(np.abs(row))
        section_scores = [
            np.std(row[first:last] / magnitude)
            for first, last in zip(points[:-1], points[1:], strict=True)
        ]
        expected.append(np.mean(section_scores) * magnitude)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 13):
        actual = irv(data, na_rm=ignore_nan, split=True, **options)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(data, original)
