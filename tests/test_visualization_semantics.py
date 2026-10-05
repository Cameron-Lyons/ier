"""Inspect rendered plotting semantics rather than only checking Figure types."""

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from ier import (
    index_agreement,
    plot_composite,
    plot_distributions,
    plot_flag_counts,
    plot_flagged_heatmap,
    plot_index_agreement,
    screen_scores,
)

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
    assert figure.axes[0].images[0].get_array().shape == (1000, 3)
    assert figure.axes[0].get_ylabel() == "Respondent block (100 per row)"
    figure.canvas.draw()
    destination = tmp_path / "large-survey.png"
    figure.savefig(destination)
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert destination.stat().st_size > 1000

    unaggregated = plot_flagged_heatmap(result, max_rows=None)
    np.testing.assert_array_equal(unaggregated.get_size_inches(), figure.get_size_inches())
    assert unaggregated.axes[0].images[0].get_array().shape == (n_respondents, 3)
    assert unaggregated.axes[0].get_ylabel() == "Respondent"


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


def _plotted_cells(figure: Any) -> tuple[np.ndarray, np.ndarray]:
    image_array = figure.axes[0].images[0].get_array()
    return np.ma.getdata(image_array), np.ma.getmaskarray(image_array)


def _rendered_column(figure: Any, column: int) -> np.ndarray:
    """Return the drawn RGB pixels down the center of one heatmap column."""
    figure.canvas.draw()
    axes = figure.axes[0]
    pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].astype(int)
    bbox = axes.get_window_extent()
    x_position = round(float(axes.transData.transform((column, 0))[0]))
    top = int(pixels.shape[0] - bbox.y1) + 1
    bottom = int(pixels.shape[0] - bbox.y0) - 1
    return pixels[top:bottom, x_position]


def test_single_flag_among_large_survey_remains_a_visible_flagged_cell() -> None:
    n_respondents = 100_000
    irv = np.ones(n_respondents)
    irv[54_321] = 0.0
    result = screen_scores(
        {"irv": irv, "longstring": np.zeros(n_respondents)},
        thresholds={"irv": 0.5, "longstring": 5.0},
    )

    figure = plot_flagged_heatmap(result)
    data, mask = _plotted_cells(figure)
    assert data.shape == (1000, 2)
    np.testing.assert_array_equal(np.argwhere(data), [[543, 0]])
    assert not mask.any()

    flagged_color = np.asarray(figure.axes[0].images[0].cmap(1.0)[:3]) * 255
    irv_pixels = _rendered_column(figure, 0)
    assert np.count_nonzero(np.abs(irv_pixels - flagged_color).max(axis=1) <= 2) >= 1
    longstring_pixels = _rendered_column(figure, 1)
    assert np.count_nonzero(np.abs(longstring_pixels - flagged_color).max(axis=1) <= 2) == 0


def test_aggregated_blocks_are_gray_only_when_every_respondent_is_unavailable() -> None:
    irv = np.ones(2000)
    irv[[0, 1]] = np.nan
    irv[2] = np.nan
    irv[[4, 5]] = [np.nan, 0.0]
    result = screen_scores(
        {"irv": irv, "onset": np.full(2000, np.nan)},
        thresholds={"irv": 0.5},
    )

    figure = plot_flagged_heatmap(result)
    axes = figure.axes[0]
    data, mask = _plotted_cells(figure)
    assert data.shape == (1000, 2)
    assert axes.get_ylabel() == "Respondent block (2 per row)"
    np.testing.assert_array_equal(mask[:3], [[True, False], [False, False], [False, False]])
    assert not mask[3:].any()
    np.testing.assert_array_equal(data[:3, 0], [False, False, True])
    assert not data[3:].any()
    image = axes.images[0]
    rgba = image.to_rgba(image.get_array())
    np.testing.assert_allclose(rgba[0, 0], to_rgba("#b3b3b3"))
    np.testing.assert_allclose(rgba[1, 0], image.cmap(0.0))
    np.testing.assert_allclose(rgba[2, 0], image.cmap(1.0))
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Not flagged",
        "Flagged",
        "Unavailable",
    ]


def test_partial_final_block_reduces_only_its_own_respondents() -> None:
    irv = np.ones(1001)
    irv[1000] = 0.0
    mahad = np.ones(1001)
    mahad[1000] = np.nan
    result = screen_scores(
        {"irv": irv, "longstring": np.zeros(1001), "mahad": mahad},
        thresholds={"irv": 0.5, "longstring": 5.0, "mahad": 5.0},
    )

    figure = plot_flagged_heatmap(result, max_rows=1000)
    data, mask = _plotted_cells(figure)
    assert data.shape == (501, 3)
    assert figure.axes[0].get_ylabel() == "Respondent block (2 per row)"
    np.testing.assert_array_equal(data[-1], [True, False, False])
    np.testing.assert_array_equal(mask[-1], [False, False, True])
    assert not data[:-1].any()
    assert not mask[:-1].any()


@pytest.mark.parametrize("n_respondents", [3, 1000])
def test_heatmap_within_row_limit_keeps_one_row_per_respondent(n_respondents: int) -> None:
    irv = np.linspace(0, 1, n_respondents)
    result = screen_scores({"irv": irv}, thresholds={"irv": 0.4})
    figure = plot_flagged_heatmap(result)
    data, _ = _plotted_cells(figure)
    np.testing.assert_array_equal(data[:, 0], irv <= 0.4)
    assert figure.axes[0].get_ylabel() == "Respondent"


def test_numpy_integer_row_limits_are_accepted() -> None:
    result = screen_scores({"irv": [0.1, 0.9, 0.8, 0.7, 0.2]}, thresholds={"irv": 0.5})
    data, _ = _plotted_cells(plot_flagged_heatmap(result, max_rows=np.int64(2)))
    np.testing.assert_array_equal(data[:, 0], [True, True])


def test_flag_count_order_moves_flagged_respondents_to_the_top() -> None:
    n_respondents = 5000
    irv = np.ones(n_respondents)
    irv[[17, 2500, 4999]] = 0.0
    irv[10] = np.nan
    longstring = np.zeros(n_respondents)
    longstring[[2500, 3000]] = 9.0
    result = screen_scores(
        {"irv": irv, "longstring": longstring},
        thresholds={"irv": 0.5, "longstring": 5.0},
    )

    blocks = plot_flagged_heatmap(result, order="flag_count")
    data, _ = _plotted_cells(blocks)
    assert data.shape == (1000, 2)
    assert blocks.axes[0].get_ylabel() == "Respondent block (5 per row, sorted by flag count)"
    np.testing.assert_array_equal(data[0], [True, True])
    assert not data[1:].any()

    rows = plot_flagged_heatmap(result, order="flag_count", max_rows=None)
    data, mask = _plotted_cells(rows)
    assert rows.axes[0].get_ylabel() == "Respondent (sorted by flag count)"
    # Respondent 2500 has two flags; 17, 3000, and 4999 keep input order.
    np.testing.assert_array_equal(
        data[:4], [[True, True], [True, False], [False, True], [True, False]]
    )
    assert not data[4:].any()
    np.testing.assert_array_equal(np.argwhere(mask), [[14, 0]])

    unchanged = plot_flagged_heatmap(result, order="input", max_rows=None)
    np.testing.assert_array_equal(np.argwhere(_plotted_cells(unchanged)[1]), [[10, 0]])


@pytest.mark.parametrize("max_rows", [0, -5, True, False, 2.5, "100", np.float64(10)])
def test_heatmap_rejects_invalid_row_limits(max_rows: object) -> None:
    result = screen_scores({"irv": [0.1, 0.9]})
    with pytest.raises(ValueError, match="max_rows must be a positive integer or None"):
        plot_flagged_heatmap(result, max_rows=max_rows)  # type: ignore[arg-type]


@pytest.mark.parametrize("order", ["descending", "", None, 1])
def test_heatmap_rejects_unknown_orders(order: object) -> None:
    result = screen_scores({"irv": [0.1, 0.9]})
    with pytest.raises(ValueError, match="order must be 'input' or 'flag_count'"):
        plot_flagged_heatmap(result, order=order)  # type: ignore[arg-type]


def _distribution_result() -> "ScreenResult":
    return screen_scores(
        {
            "irv": [0.1, 0.4, 0.9, 1.2, np.nan],
            "longstring": [8.0, 3.0, 2.0, 7.0, 1.0],
            "onset": [np.nan, 4.0, np.nan, np.nan, 2.0],
        },
        thresholds={"irv": 0.5, "longstring": 5.0},
    )


def _cutoff_lines(axes: Any) -> list[Any]:
    return [line for line in axes.lines if line.get_label() == "cutoff"]


def _labelled_patches(axes: Any, label: str) -> list[Any]:
    return [patch for patch in axes.patches if patch.get_label() == label]


def _data_extent(axes: Any, artist: Any) -> tuple[float, float]:
    corners = axes.transData.inverted().transform(artist.get_window_extent().get_points())
    return float(corners[0, 0]), float(corners[1, 0])


def test_distribution_cutoffs_shade_each_index_flagged_tail() -> None:
    result = _distribution_result()
    figure = plot_distributions(result)
    figure.canvas.draw()
    irv_axes, longstring_axes, onset_axes = figure.axes

    assert [axes.get_title() for axes in figure.axes] == [
        "irv (flagged: 2)",
        "longstring (flagged: 2)",
        "onset (flagged: 2)",
    ]
    assert [len(_cutoff_lines(axes)) for axes in figure.axes] == [1, 1, 0]
    assert [len(axes.lines) for axes in figure.axes] == [1, 1, 0]
    for axes, cutoff in ((irv_axes, 0.5), (longstring_axes, 5.0)):
        (line,) = _cutoff_lines(axes)
        np.testing.assert_array_equal(line.get_xdata(), [cutoff, cutoff])
        assert line.get_linestyle() == "--"
        assert to_rgba(line.get_color()) == to_rgba("C3")
        assert [text.get_text() for text in axes.get_legend().get_texts()] == [
            "cutoff",
            "flagged",
        ]

    (low_tail,) = _labelled_patches(irv_axes, "flagged")
    lower, upper = irv_axes.get_xlim()
    assert _data_extent(irv_axes, low_tail) == pytest.approx((lower, 0.5))
    (high_tail,) = _labelled_patches(longstring_axes, "flagged")
    lower, upper = longstring_axes.get_xlim()
    assert _data_extent(longstring_axes, high_tail) == pytest.approx((5.0, upper))
    assert lower < 1.0 and upper > 8.0
    assert _labelled_patches(onset_axes, "flagged") == []
    assert onset_axes.get_legend() is None


def test_distribution_cutoff_outside_observed_scores_remains_visible() -> None:
    result = screen_scores({"longstring": [1.0, 2.0, 3.0]}, thresholds={"longstring": 50.0})
    figure = plot_distributions(result)
    axes = figure.axes[0]
    lower, upper = axes.get_xlim()
    assert lower < 1.0 and upper > 50.0
    (tail,) = _labelled_patches(axes, "flagged")
    assert _data_extent(axes, tail) == pytest.approx((50.0, upper))
    assert axes.get_title() == "longstring (flagged: 0)"


def test_distribution_grid_hides_unused_panels() -> None:
    names = ["irv", "longstring", "mahad", "person_total"]
    result = screen_scores({name: [0.1, 0.4, 0.9, 1.2] for name in names}, percentile=90)
    figure = plot_distributions(result)
    assert [axes.get_visible() for axes in figure.axes] == [True] * 4 + [False] * 2
    assert [len(_cutoff_lines(axes)) for axes in figure.axes[:4]] == [1, 1, 1, 1]


def test_wholly_unavailable_scores_draw_no_placeholder_cutoff() -> None:
    result = screen_scores(
        {"irv": [0.1, 0.4, 0.9], "psychsyn": [np.nan, np.nan, np.nan]},
        percentile=90,
    )
    assert result["thresholds"]["psychsyn"] == 0.0
    irv_axes, psychsyn_axes = plot_distributions(result).axes
    assert len(_cutoff_lines(irv_axes)) == 1
    assert len(psychsyn_axes.lines) == 0
    assert psychsyn_axes.get_legend() is None
    assert psychsyn_axes.get_title() == "psychsyn (flagged: 0)"


def test_unregistered_cutoffs_draw_lines_without_guessing_a_tail() -> None:
    result = cast(
        "ScreenResult",
        {"scores": {"custom": np.asarray([1.0, 2.0, 3.0])}, "thresholds": {"custom": 2.5}},
    )
    axes = plot_distributions(result).axes[0]
    assert axes.get_title() == "custom"
    assert len(_cutoff_lines(axes)) == 1
    assert _labelled_patches(axes, "flagged") == []


def test_distributions_without_summary_or_threshold_keys_draw_plain_histograms() -> None:
    result = cast("ScreenResult", {"scores": {"irv": np.asarray([0.1, 0.5, np.nan])}})
    axes = plot_distributions(result).axes[0]
    assert axes.get_title() == "irv"
    assert len(axes.lines) == 0
    assert axes.get_legend() is None
    assert sum(bar.get_height() for bar in axes.containers[0]) == 2


def test_distribution_thresholds_can_be_hidden() -> None:
    figure = plot_distributions(_distribution_result(), show_thresholds=False)
    assert [len(axes.lines) for axes in figure.axes] == [0, 0, 0]
    assert all(axes.get_legend() is None for axes in figure.axes)
    assert figure.axes[0].get_title() == "irv (flagged: 2)"


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_distribution_threshold_option_must_be_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="show_thresholds must be a boolean"):
        plot_distributions(_distribution_result(), show_thresholds=value)  # type: ignore[arg-type]


def _legend_texts(axes: Any) -> list[str]:
    legend = axes.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def test_composite_plot_shades_high_tail_and_counts_scores_at_cutoff() -> None:
    scores = np.array([-1.2, -0.3, 0.0, 0.4, np.nan, 1.5, 2.0, 2.0])
    figure = plot_composite(scores, threshold=1.5, bins=5)
    figure.canvas.draw()
    axes = figure.axes[0]

    assert len(axes.containers) == 1
    assert sum(bar.get_height() for bar in axes.containers[0]) == 7
    (line,) = axes.lines
    np.testing.assert_array_equal(line.get_xdata(), [1.5, 1.5])
    assert line.get_linestyle() == "--"
    (tail,) = _labelled_patches(axes, "Flagged (n=3)")
    lower, upper = axes.get_xlim()
    assert _data_extent(axes, tail) == pytest.approx((1.5, upper))
    assert _legend_texts(axes) == ["Cutoff (1.5)", "Flagged (n=3)"]
    assert axes.get_xlabel() == "Composite score"


def test_composite_plot_counts_supplied_flags_and_overlays_their_scores() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9, np.nan, 1.1])
    flags = np.array([False, False, False, True, False, True])
    axes = plot_composite(scores, flags=flags, threshold=0.25, bins=4).axes[0]

    assert [sum(bar.get_height() for bar in bars) for bars in axes.containers] == [5, 2]
    np.testing.assert_array_equal(
        [bar.get_x() for bar in axes.containers[0]], [bar.get_x() for bar in axes.containers[1]]
    )
    assert _legend_texts(axes) == ["Flagged (n=2)", "Cutoff (0.25)"]

    flagged_only = plot_composite(scores, flags=flags).axes[0]
    assert len(flagged_only.lines) == 0
    assert _legend_texts(flagged_only) == ["Flagged (n=2)"]


def test_composite_plot_without_cutoff_or_flags_has_no_legend() -> None:
    axes = plot_composite([0.1, 0.5, np.nan], figsize=(5.0, 3.0)).axes[0]
    assert axes.get_legend() is None
    assert len(axes.lines) == 0
    np.testing.assert_array_equal(axes.figure.get_size_inches(), [5.0, 3.0])


def test_composite_plot_renders_entirely_missing_scores() -> None:
    figure = plot_composite([np.nan, np.nan], threshold=0.0)
    figure.canvas.draw()
    assert _legend_texts(figure.axes[0]) == ["Cutoff (0)", "Flagged (n=0)"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scores": []}, "scores"),
        ({"scores": [[1.0, 2.0]]}, "scores"),
        ({"scores": [1.0, np.inf]}, "scores"),
        ({"scores": ["a", "b"]}, "scores"),
        ({"scores": [1.0, 2.0], "threshold": np.nan}, "threshold must be a finite number"),
        ({"scores": [1.0, 2.0], "threshold": True}, "threshold must be a finite number"),
        ({"scores": [1.0, 2.0], "flags": [1, 0]}, "flags must be a one-dimensional boolean"),
        ({"scores": [1.0, 2.0], "flags": [[True, False]]}, "one-dimensional boolean"),
        ({"scores": [1.0, 2.0], "flags": [[True], [False, True]]}, "one-dimensional boolean"),
        ({"scores": [1.0, 2.0], "flags": [True]}, "one value per composite score"),
    ],
)
def test_composite_plot_validates_inputs(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        plot_composite(**kwargs)


def _agreement_result() -> "ScreenResult":
    return screen_scores(
        {
            "irv": [0.1, 0.2, 0.9, 0.3, 1.0, 1.1],
            "longstring": [6.0, 1.0, 7.0, 8.0, 2.0, 1.0],
            "mahad": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "onset": [np.nan, 3.0, np.nan, 5.0, np.nan, np.nan],
        },
        thresholds={"irv": 0.5, "longstring": 5.0, "mahad": 10.0},
    )


def test_agreement_plot_uses_fixed_jaccard_scale_and_gray_undefined_cells() -> None:
    result = _agreement_result()
    figure = plot_index_agreement(result)
    axes = figure.axes[0]
    image = axes.images[0]
    names, matrix = index_agreement(result)

    np.testing.assert_array_equal(np.ma.getdata(image.get_array()), matrix)
    assert image.get_clim() == (0.0, 1.0)
    assert image.cmap.name == "viridis"
    assert [text.get_text() for text in axes.get_xticklabels()] == names
    assert [text.get_text() for text in axes.get_yticklabels()] == names
    assert figure.axes[1].get_ylabel() == "Jaccard similarity of flags"
    texts = [[axes.texts[row * 4 + column] for column in range(4)] for row in range(4)]
    assert [text.get_text() for text in texts[0]] == ["1.00", "0.50", "0.00", "0.67"]
    assert texts[2][2].get_text() == "n/a"
    assert texts[0][0].get_color() == "black"
    assert texts[0][2].get_color() == "white"
    rgba = image.to_rgba(image.get_array())
    np.testing.assert_allclose(rgba[2, 2], to_rgba("#b3b3b3"))
    np.testing.assert_allclose(rgba[0, 0], image.cmap(1.0))


def test_agreement_plot_scales_overlap_by_respondents_and_annotates_counts() -> None:
    result = _agreement_result()
    axes = plot_index_agreement(result, "overlap").axes[0]
    image = axes.images[0]
    _, counts = index_agreement(result, "overlap")

    np.testing.assert_allclose(np.ma.getdata(image.get_array()), counts / 6)
    assert image.get_clim() == (0.0, 1.0)
    assert [text.get_text() for text in axes.texts[:4]] == ["3", "2", "0", "2"]


def test_agreement_plot_spearman_uses_diverging_scale_without_presence_indices() -> None:
    result = _agreement_result()
    figure = plot_index_agreement(result, "spearman", annotate=False, cmap="coolwarm")
    axes = figure.axes[0]
    image = axes.images[0]

    assert len(axes.texts) == 0
    assert image.get_clim() == (-1.0, 1.0)
    assert image.cmap.name == "coolwarm"
    assert [text.get_text() for text in axes.get_xticklabels()] == ["irv", "longstring", "mahad"]
    assert figure.axes[1].get_ylabel() == "Spearman rank correlation"
    default = plot_index_agreement(result, "spearman").axes[0]
    assert default.images[0].cmap.name == "RdBu_r"
    assert len(default.texts) == 9


def test_agreement_plot_spearman_cells_follow_suspiciousness_orientation() -> None:
    # Low irv and high longstring scores rank the same respondents as suspicious.
    result = screen_scores(
        {"irv": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "longstring": [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]},
        thresholds={"irv": 0.3, "longstring": 7.0},
    )
    axes = plot_index_agreement(screen_result=result, kind="spearman").axes[0]
    image = axes.images[0]

    assert [text.get_text() for text in axes.texts] == ["1.00"] * 4
    np.testing.assert_allclose(image.to_rgba(image.get_array())[0, 1], image.cmap(1.0))
    with pytest.raises(TypeError, match="result"):
        plot_index_agreement(result=result)  # type: ignore[call-arg]


def test_agreement_plot_validates_inputs_and_handles_empty_results() -> None:
    result = _agreement_result()
    with pytest.raises(ValueError, match="annotate must be a boolean"):
        plot_index_agreement(result, annotate=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be"):
        plot_index_agreement(result, "pearson")  # type: ignore[arg-type]

    empty = screen_scores({}, n_respondents=3, errors={"mad": "unconfigured"})
    figure = plot_index_agreement(empty)
    assert len(figure.axes[0].images) == 0
    sized = plot_index_agreement(result, figsize=(7.0, 6.0))
    np.testing.assert_array_equal(sized.get_size_inches(), [7.0, 6.0])
