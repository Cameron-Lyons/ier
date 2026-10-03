"""Inspect rendered plotting semantics rather than only checking Figure types."""

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from ier import plot_flag_counts, plot_flagged_heatmap, screen_scores

if TYPE_CHECKING:
    from ier.types import ScreenResult


@pytest.fixture(autouse=True)
def _headless_plots() -> Iterator[None]:
    matplotlib.use("Agg")
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.mark.parametrize("cmap", ["Reds", "Blues", "viridis"])
def test_constant_cohorts_use_distinct_flagged_and_unflagged_colors(cmap: str) -> None:
    colors = []
    for score, expected_flag in [(6.0, True), (4.0, False)]:
        result = screen_scores({"longstring": [score] * 3}, thresholds={"longstring": 5.0})
        figure = plot_flagged_heatmap(result, cmap=cmap)
        image = figure.axes[0].images[0]
        assert image.get_clim() == (0, 1)
        assert not np.ma.getmaskarray(image.get_array()).any()
        rgba = image.to_rgba(image.get_array())
        expected_color = matplotlib.colormaps[cmap](float(expected_flag))
        np.testing.assert_allclose(rgba, np.broadcast_to(expected_color, rgba.shape))
        assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
            "Not flagged",
            "Flagged",
        ]
        colors.append(rgba[0, 0])
    assert not np.array_equal(colors[0], colors[1])


def test_missing_scores_are_gray_while_absent_onset_remains_unflagged() -> None:
    result = screen_scores(
        {
            "longstring": [2.0, np.nan, 7.0],
            "onset": [np.nan, 4.0, np.nan],
            "irv": [np.nan, 0.1, 1.5],
        },
        thresholds={"longstring": 5.0, "irv": 0.5},
    )
    original_bad_color = matplotlib.colormaps["Reds"].get_bad().copy()
    figure = plot_flagged_heatmap(result)
    axes = figure.axes[0]
    image = axes.images[0]
    expected_mask = np.asarray([[False, False, True], [True, False, False], [False] * 3])
    np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()), expected_mask)
    np.testing.assert_array_equal(
        np.ma.getdata(image.get_array()),
        [[False, False, False], [False, True, True], [True, False, False]],
    )
    rgba = image.to_rgba(image.get_array())
    np.testing.assert_allclose(rgba[expected_mask], [to_rgba("#b3b3b3")] * 2)
    np.testing.assert_allclose(rgba[[0, 2], 1], [image.cmap(0.0)] * 2)
    assert [text.get_text() for text in axes.get_xticklabels()] == ["longstring", "onset", "irv"]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Not flagged",
        "Flagged",
        "Unavailable",
    ]
    np.testing.assert_array_equal(matplotlib.colormaps["Reds"].get_bad(), original_bad_color)
    np.testing.assert_array_equal(result["scores"]["irv"], [np.nan, 0.1, 1.5])
    np.testing.assert_array_equal(result["flags"]["onset"], [False, True, False])


@pytest.mark.parametrize("figsize", [(9.0, 5.0), (22.0, 14.0)])
def test_heatmap_honors_explicit_figure_dimensions(figsize: tuple[float, float]) -> None:
    result = screen_scores({"irv": np.ones(100)})
    figure = plot_flagged_heatmap(result, figsize=figsize)
    np.testing.assert_array_equal(figure.get_size_inches(), figsize)


def test_wide_heatmap_bounds_width_and_masks_unregistered_numeric_score_labels() -> None:
    names = [f"custom-{index}" for index in range(30)]
    result = cast(
        "ScreenResult",
        {
            "scores": {name: np.asarray([0.1, np.nan]) for name in names},
            "flags": {name: np.asarray([True, False]) for name in names},
        },
    )
    figure = plot_flagged_heatmap(result)
    np.testing.assert_array_equal(figure.get_size_inches(), [18, 4])
    np.testing.assert_array_equal(
        np.ma.getmaskarray(figure.axes[0].images[0].get_array()),
        [[False] * 30, [True] * 30],
    )


def test_large_survey_heatmap_renders_and_saves_with_bounded_automatic_dimensions(
    tmp_path: Path,
) -> None:
    n_respondents = 100_000
    irv = np.linspace(0, 2, n_respondents)
    irv[::7] = np.nan
    longstring = np.arange(n_respondents, dtype=float) % 10
    onset = np.full(n_respondents, np.nan)
    onset[::11] = 20.0
    result = screen_scores(
        {"irv": irv, "longstring": longstring, "onset": onset},
        thresholds={"irv": 0.5, "longstring": 5.0},
    )

    figure = plot_flagged_heatmap(result)
    width, height = figure.get_size_inches()
    assert 6 <= width <= 18
    assert height == 12
    assert figure.axes[0].images[0].get_array().shape == (n_respondents, 3)
    figure.canvas.draw()
    destination = tmp_path / "large-survey.png"
    figure.savefig(destination)
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert destination.stat().st_size > 1000


def test_flag_count_histogram_matches_exact_respondent_decisions() -> None:
    result = screen_scores(
        {
            "irv": [0.1, 0.1, 1.0, 1.0, np.nan],
            "longstring": [8.0, 1.0, 8.0, 1.0, np.nan],
            "onset": [2.0, np.nan, np.nan, np.nan, 3.0],
        },
        thresholds={"irv": 0.5, "longstring": 5.0},
    )
    np.testing.assert_array_equal(result["flag_counts"], [3, 1, 1, 0, 1])
    figure = plot_flag_counts(result)
    axes = figure.axes[0]
    np.testing.assert_array_equal(axes.get_xticks(), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal([bar.get_height() for bar in axes.patches], [1, 3, 0, 1, 0])
    assert sum(bar.get_height() for bar in axes.patches) == result["n_respondents"]


def test_failed_only_screen_counts_every_respondent_in_zero_flag_bin() -> None:
    result = screen_scores({}, n_respondents=15, errors={"mad": "unconfigured item pairs"})
    figure = plot_flag_counts(result)
    axes = figure.axes[0]
    np.testing.assert_array_equal(axes.get_xticks(), [0, 1])
    np.testing.assert_array_equal([bar.get_height() for bar in axes.patches], [15, 0])


def test_empty_float_flag_counts_remain_supported() -> None:
    result = cast("ScreenResult", {"flag_counts": np.asarray([]), "n_indices": 0})
    figure = plot_flag_counts(result)
    axes = figure.axes[0]
    np.testing.assert_array_equal(axes.get_xticks(), [0, 1])
    np.testing.assert_array_equal([bar.get_height() for bar in axes.patches], [0, 0])
