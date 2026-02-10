"""
Visualization functions for IER screening results.

Provides plots for inspecting the output of screen(), including
score distributions, flag heatmaps, and flag count summaries.
"""

from typing import Any

import numpy as np


def plot_distributions(
    screen_result: dict[str, Any],
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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. "
            "Install with: pip install insufficient-effort[plot]"
        ) from exc

    scores: dict[str, np.ndarray] = screen_result["scores"]
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
    screen_result: dict[str, Any],
    figsize: tuple[float, float] | None = None,
    cmap: str = "Reds",
) -> Any:
    """
    Plot a heatmap of flag status per respondent and index.

    Rows are respondents, columns are indices. Colored cells indicate flagged.

    Parameters:
    - screen_result: Output dict from screen().
    - figsize: Figure size as (width, height).
    - cmap: Matplotlib colormap name.

    Returns:
    - matplotlib Figure object.

    Raises:
    - RuntimeError: If matplotlib is not available.

    Example:
        >>> result = screen(data)
        >>> fig = plot_flagged_heatmap(result)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. "
            "Install with: pip install insufficient-effort[plot]"
        ) from exc

    flags: dict[str, np.ndarray] = screen_result["flags"]
    index_names = list(flags.keys())

    if len(index_names) == 0:
        fig, _ = plt.subplots(1, 1, figsize=figsize or (6, 4))
        return fig

    flag_matrix = np.column_stack([flags[name] for name in index_names]).astype(float)

    if figsize is None:
        figsize = (max(6, len(index_names) * 0.8), max(4, flag_matrix.shape[0] * 0.15))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(flag_matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(index_names)))
    ax.set_xticklabels(index_names, rotation=45, ha="right")
    ax.set_xlabel("Index")
    ax.set_ylabel("Respondent")
    ax.set_title("IER Flag Heatmap")
    fig.tight_layout()
    return fig


def plot_flag_counts(
    screen_result: dict[str, Any],
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
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. "
            "Install with: pip install insufficient-effort[plot]"
        ) from exc

    flag_counts: np.ndarray = screen_result["flag_counts"]
    n_indices: int = screen_result["n_indices"]

    if figsize is None:
        figsize = (8, 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    max_flags = max(int(np.max(flag_counts)), 0) if len(flag_counts) > 0 else 0
    bins_range = range(0, max(max_flags + 2, n_indices + 2))
    counts_per_bin = [int(np.sum(flag_counts == b)) for b in bins_range]

    ax.bar(list(bins_range), counts_per_bin, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of Flags")
    ax.set_ylabel("Number of Respondents")
    ax.set_title("Distribution of IER Flag Counts")
    ax.set_xticks(list(bins_range))
    fig.tight_layout()
    return fig
