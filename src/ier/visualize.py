"""
Visualization functions for IER screening results.

Provides plots for inspecting the output of screen(), including
score distributions, flag heatmaps, and flag count summaries.
"""

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ier._optional_imports import require_matplotlib_pyplot
from ier._registry import INDEX_REGISTRY
from ier.types import ScreenResult

if TYPE_CHECKING:
    from collections.abc import Callable


def plot_distributions(
    screen_result: ScreenResult,
    figsize: tuple[float, float] | None = None,
    bins: int = 30,
) -> Any:
    """
    Plot histograms of score distributions for each index.

    Parameters:
    - screen_result: Output dict from screen().
    - figsize: Figure size as (width, height). If None, auto-calculated.
    - bins: Number of histogram bins.

    Returns:
    - matplotlib Figure object.

    Raises:
    - RuntimeError: If matplotlib is not available.

    Example:
        >>> result = screen(data)
        >>> fig = plot_distributions(result)
        >>> fig.savefig("distributions.png")
    """
    plt = require_matplotlib_pyplot()

    scores = screen_result["scores"]
    n_indices = len(scores)

    if n_indices == 0:
        fig, _ = plt.subplots(1, 1, figsize=figsize or (6, 4))
        return fig

    n_cols = min(3, n_indices)
    n_rows = (n_indices + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i, (name, score_arr) in enumerate(scores.items()):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        valid = score_arr[~np.isnan(score_arr)]
        ax.hist(valid, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")

    for i in range(n_indices, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig


def plot_flagged_heatmap(
    screen_result: ScreenResult,
    figsize: tuple[float, float] | None = None,
    cmap: str = "Reds",
) -> Any:
    """
    Plot a heatmap of flag status per respondent and index.

    Rows are respondents, columns are indices. Flagged and unflagged cells use
    the same color scale across cohorts. Unavailable scores are gray; for a
    presence-based index such as onset, NaN means no detected event and remains
    unflagged. Automatic figure dimensions are bounded for large surveys.

    Parameters:
    - screen_result: Output dict from screen().
    - figsize: Figure size as (width, height). If None, dimensions are bounded
      at 18 inches wide and 12 inches high.
    - cmap: Matplotlib colormap name.

    Returns:
    - matplotlib Figure object.

    Raises:
    - RuntimeError: If matplotlib is not available.

    Example:
        >>> result = screen(data)
        >>> fig = plot_flagged_heatmap(result)
    """
    plt = require_matplotlib_pyplot()

    flags = screen_result["flags"]
    index_names = list(flags.keys())

    if len(index_names) == 0:
        fig, _ = plt.subplots(1, 1, figsize=figsize or (6, 4))
        return fig

    flag_matrix = np.column_stack([flags[name] for name in index_names])
    unavailable = np.zeros_like(flag_matrix, dtype=bool)
    for column, name in enumerate(index_names):
        scores = screen_result["scores"].get(name)
        spec = INDEX_REGISTRY.get(name)
        if scores is not None and (spec is None or spec.flag_mode != "present"):
            unavailable[:, column] = np.isnan(scores)
    # NumPy's masked-array constructor lacks complete type annotations.
    masked_array = cast("Callable[..., np.ndarray]", np.ma.array)
    plotted_flags = masked_array(flag_matrix, mask=unavailable, copy=False)

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#b3b3b3")

    if figsize is None:
        figsize = (
            min(18, max(6, len(index_names) * 0.8)),
            min(12, max(4, flag_matrix.shape[0] * 0.15)),
        )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(
        plotted_flags,
        aspect="auto",
        cmap=colormap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(len(index_names)))
    ax.set_xticklabels(index_names, rotation=45, ha="right")
    ax.set_xlabel("Index")
    ax.set_ylabel("Respondent")
    ax.set_title("IER Flag Heatmap")
    states = [("Not flagged", colormap(0.0)), ("Flagged", colormap(1.0))]
    if unavailable.any():
        states.append(("Unavailable", colormap.get_bad()))
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name) for name, color in states
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    fig.tight_layout()
    return fig


def plot_flag_counts(
    screen_result: ScreenResult,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """
    Plot a bar chart of flag counts across respondents.

    X-axis is the number of flags, y-axis is the count of respondents
    with that many flags.

    Parameters:
    - screen_result: Output dict from screen().
    - figsize: Figure size as (width, height).

    Returns:
    - matplotlib Figure object.

    Raises:
    - RuntimeError: If matplotlib is not available.

    Example:
        >>> result = screen(data)
        >>> fig = plot_flag_counts(result)
    """
    plt = require_matplotlib_pyplot()

    flag_counts = screen_result["flag_counts"]
    n_indices = screen_result["n_indices"]

    if figsize is None:
        figsize = (8, 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    max_flags = max(int(np.max(flag_counts)), 0) if len(flag_counts) > 0 else 0
    n_bins = max(max_flags, n_indices) + 2
    bins_range = np.arange(n_bins)
    counts_per_bin = (
        np.bincount(flag_counts, minlength=n_bins)
        if len(flag_counts)
        else np.zeros(n_bins, dtype=np.int_)
    )

    ax.bar(bins_range, counts_per_bin, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of Flags")
    ax.set_ylabel("Number of Respondents")
    ax.set_title("Distribution of IER Flag Counts")
    ax.set_xticks(bins_range)
    fig.tight_layout()
    return fig
