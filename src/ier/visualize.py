"""
Visualization functions for IER screening results.

Provides plots for inspecting the output of screen(), including
score distributions with their cutoffs, flag heatmaps, flag count summaries,
composite score distributions, and index agreement matrices.
"""

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from ier._flagging import validate_threshold
from ier._optional_imports import require_matplotlib_pyplot
from ier._registry import INDEX_REGISTRY
from ier._validation import validate_score_array
from ier.tables import index_agreement
from ier.types import AgreementKind, ScreenResult

if TYPE_CHECKING:
    from collections.abc import Callable

_UNAVAILABLE_COLOR = "#b3b3b3"
_CUTOFF_COLOR = "C3"
_TAIL_ALPHA = 0.12
_AGREEMENT_LABELS = {
    "overlap": "Share of respondents flagged by both",
    "jaccard": "Jaccard similarity of flags",
    "spearman": "Spearman rank correlation",
}


def _draw_cutoff(
    ax: Any,
    threshold: float,
    direction: Literal["high", "low"] | None,
    *,
    label: str,
    tail_label: str | None,
) -> None:
    """Draw a cutoff line and shade its flagged tail without widening the axes."""
    ax.axvline(threshold, color=_CUTOFF_COLOR, linestyle="--", label=label)
    if direction is None:
        return
    lower, upper = ax.get_xlim()
    if direction == "high":
        ax.axvspan(threshold, upper, color=_CUTOFF_COLOR, alpha=_TAIL_ALPHA, label=tail_label)
    else:
        ax.axvspan(lower, threshold, color=_CUTOFF_COLOR, alpha=_TAIL_ALPHA, label=tail_label)
    ax.set_xlim(lower, upper)


def plot_distributions(
    screen_result: ScreenResult,
    figsize: tuple[float, float] | None = None,
    bins: int = 30,
    *,
    show_thresholds: bool = True,
) -> Any:
    """
    Plot histograms of score distributions for each index.

    Each applied cutoff is drawn as a dashed line, and the tail that the index
    flags is shaded according to its registered direction. When per-index
    summaries are available, panel titles report the number of flagged
    respondents. Presence-flagged indices such as onset, and indices without
    any available scores, have no cutoff line.

    Parameters:
    - screen_result: Output dict from screen().
    - figsize: Figure size as (width, height). If None, auto-calculated.
    - bins: Number of histogram bins.
    - show_thresholds: Draw applied cutoffs and shade flagged tails.

    Returns:
    - matplotlib Figure object.

    Raises:
    - ValueError: If show_thresholds is not a boolean.
    - RuntimeError: If matplotlib is not available.

    Example:
        >>> from ier import plot_distributions, screen_scores
        >>> result = screen_scores(
        ...     {"irv": [0.1, 0.4, 0.9, 1.2], "longstring": [8.0, 3.0, 2.0, 7.0]},
        ...     thresholds={"irv": 0.5, "longstring": 5.0},
        ... )
        >>> fig = plot_distributions(result)
        >>> titles = [ax.get_title() for ax in fig.axes]
        >>> titles
        ['irv (flagged: 2)', 'longstring (flagged: 2)']
    """
    if not isinstance(show_thresholds, bool):
        raise ValueError("show_thresholds must be a boolean")
    plt = require_matplotlib_pyplot()

    scores = screen_result["scores"]
    thresholds = screen_result.get("thresholds", {}) if show_thresholds else {}
    summary = screen_result.get("summary", {})
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
        index_summary = summary.get(name)
        ax.set_title(
            name if index_summary is None else f"{name} (flagged: {index_summary['n_flagged']})"
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        threshold = thresholds.get(name)
        # Wholly unavailable scores have no distribution to compare a cutoff with.
        if threshold is not None and valid.size:
            spec = INDEX_REGISTRY.get(name)
            _draw_cutoff(
                ax,
                float(threshold),
                None if spec is None else spec.flag_direction,
                label="cutoff",
                tail_label="flagged",
            )
            ax.legend(fontsize="small")

    for i in range(n_indices, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig


def _validate_max_rows(max_rows: int | None) -> int | None:
    """Return a validated optional respondent-row display limit."""
    if max_rows is None:
        return None
    if isinstance(max_rows, bool) or not isinstance(max_rows, (int, np.integer)) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer or None")
    return int(max_rows)


def _aggregate_flag_blocks(
    flag_matrix: np.ndarray,
    unavailable: np.ndarray,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce consecutive respondents to blocks in which any flag stays visible.

    A block is unavailable only when every respondent in it is unavailable and
    none is flagged. Padding rows are unflagged and unavailable, so a shorter
    final block follows the same rules.
    """
    n_respondents, n_indices = flag_matrix.shape
    block = -(-n_respondents // max_rows)
    n_blocks = -(-n_respondents // block)
    padding = n_blocks * block - n_respondents
    if padding:
        flag_matrix = np.concatenate((flag_matrix, np.zeros((padding, n_indices), dtype=bool)))
        unavailable = np.concatenate((unavailable, np.ones((padding, n_indices), dtype=bool)))
    flagged = np.asarray(flag_matrix.reshape(n_blocks, block, n_indices).any(axis=1))
    missing = np.asarray(unavailable.reshape(n_blocks, block, n_indices).all(axis=1) & ~flagged)
    return flagged, missing, block


def plot_flagged_heatmap(
    screen_result: ScreenResult,
    figsize: tuple[float, float] | None = None,
    cmap: str = "Reds",
    *,
    max_rows: int | None = 1000,
    order: Literal["input", "flag_count"] = "input",
) -> Any:
    """
    Plot a heatmap of flag status per respondent and index.

    Rows are respondents, columns are indices. Flagged and unflagged cells use
    the same color scale across cohorts. Unavailable scores are gray; for a
    presence-based index such as onset, NaN means no detected event and remains
    unflagged. Automatic figure dimensions are bounded for large surveys.

    Surveys with more than ``max_rows`` respondents are drawn as consecutive
    respondent blocks, so rare flags are not lost to image downsampling. A block
    is flagged when any of its respondents is flagged for that index, and gray
    only when every respondent in it is unavailable.

    Parameters:
    - screen_result: Output dict from screen().
    - figsize: Figure size as (width, height). If None, dimensions are bounded
      at 18 inches wide and 12 inches high.
    - cmap: Matplotlib colormap name.
    - max_rows: Maximum number of plotted rows. Larger surveys are aggregated
      into blocks of ``ceil(n_respondents / max_rows)`` respondents. None plots
      one row per respondent.
    - order: ``"input"`` keeps respondent order; ``"flag_count"`` sorts
      respondents by descending flag count, keeping input order among ties, so
      flagged respondents appear at the top.

    Returns:
    - matplotlib Figure object.

    Raises:
    - ValueError: If max_rows is not a positive integer or None, or order is
      not recognized.
    - RuntimeError: If matplotlib is not available.

    Example:
        >>> import numpy as np
        >>> from ier import plot_flagged_heatmap, screen_scores
        >>> irv = np.ones(5000)
        >>> irv[1234] = 0.0
        >>> result = screen_scores({"irv": irv}, thresholds={"irv": 0.5})
        >>> fig = plot_flagged_heatmap(result)
        >>> fig.axes[0].images[0].get_array().shape
        (1000, 1)
        >>> fig.axes[0].get_ylabel()
        'Respondent block (5 per row)'
    """
    max_rows = _validate_max_rows(max_rows)
    if not isinstance(order, str) or order not in ("input", "flag_count"):
        raise ValueError("order must be 'input' or 'flag_count'")
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
    row_details = []
    if order == "flag_count":
        ranking = np.argsort(-np.count_nonzero(flag_matrix, axis=1), kind="stable")
        flag_matrix = flag_matrix[ranking]
        unavailable = unavailable[ranking]
        row_details.append("sorted by flag count")
    row_label = "Respondent"
    if max_rows is not None and flag_matrix.shape[0] > max_rows:
        flag_matrix, unavailable, block = _aggregate_flag_blocks(flag_matrix, unavailable, max_rows)
        row_label = "Respondent block"
        row_details.insert(0, f"{block} per row")
    if row_details:
        row_label = f"{row_label} ({', '.join(row_details)})"
    # NumPy's masked-array constructor lacks complete type annotations.
    masked_array = cast("Callable[..., np.ndarray]", np.ma.array)
    plotted_flags = masked_array(flag_matrix, mask=unavailable, copy=False)

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(_UNAVAILABLE_COLOR)

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
    ax.set_ylabel(row_label)
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
        >>> from ier import plot_flag_counts, screen_scores
        >>> result = screen_scores(
        ...     {"irv": [0.1, 0.4, 0.9, 1.2], "longstring": [8.0, 3.0, 2.0, 7.0]},
        ...     thresholds={"irv": 0.5, "longstring": 5.0},
        ... )
        >>> fig = plot_flag_counts(result)
        >>> heights = [int(bar.get_height()) for bar in fig.axes[0].patches]
        >>> heights
        [1, 2, 1, 0]
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


def _validate_plot_flags(flags: ArrayLike, n_respondents: int) -> np.ndarray:
    """Validate respondent-aligned Boolean composite flags."""
    try:
        flag_arr = np.asarray(flags)
    except (TypeError, ValueError) as error:
        raise ValueError("flags must be a one-dimensional boolean array") from error
    if flag_arr.ndim != 1 or flag_arr.dtype != np.bool_:
        raise ValueError("flags must be a one-dimensional boolean array")
    if len(flag_arr) != n_respondents:
        raise ValueError("flags must contain one value per composite score")
    return flag_arr


def plot_composite(
    scores: ArrayLike,
    *,
    threshold: float | None = None,
    flags: ArrayLike | None = None,
    bins: int = 30,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """
    Plot the distribution of composite scores with an optional cutoff.

    Higher composite scores are more suspicious, so a cutoff shades the high
    tail, matching ``composite_flag()``. The legend reports the number of flagged
    respondents: from ``flags`` when given, otherwise from scores at or above
    ``threshold``. Flagged respondents' scores are overlaid when ``flags`` is
    given. Missing composite scores are omitted from the histogram.

    Parameters:
    - scores: Composite score vector, for example from composite() or
      composite_flag(). May contain NaN.
    - threshold: Optional finite cutoff to draw.
    - flags: Optional boolean flags aligned with scores.
    - bins: Number of histogram bins.
    - figsize: Figure size as (width, height). Defaults to (8, 5).

    Returns:
    - matplotlib Figure object.

    Raises:
    - ValueError: If scores are not a non-empty real vector of finite values or
      NaN, threshold is not finite, or flags are not aligned booleans.
    - RuntimeError: If matplotlib is not available.

    Example:
        >>> from ier import plot_composite
        >>> scores = [-1.0, -0.2, 0.1, 0.4, 2.5]
        >>> fig = plot_composite(scores, threshold=2.0)
        >>> labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        >>> labels
        ['Cutoff (2)', 'Flagged (n=1)']
    """
    values = validate_score_array(scores)
    cutoff = validate_threshold(threshold)
    flag_arr = None if flags is None else _validate_plot_flags(flags, len(values))
    plt = require_matplotlib_pyplot()

    fig, ax = plt.subplots(1, 1, figsize=figsize or (8, 5))
    available = ~np.isnan(values)
    valid = values[available]
    edges = np.histogram_bin_edges(valid, bins=bins)
    ax.hist(valid, bins=edges, edgecolor="black", alpha=0.7)
    flagged_label = None
    if flag_arr is not None:
        flagged_label = f"Flagged (n={int(np.count_nonzero(flag_arr))})"
        ax.hist(
            values[flag_arr & available],
            bins=edges,
            color=_CUTOFF_COLOR,
            edgecolor="black",
            alpha=0.7,
            label=flagged_label,
        )
    if cutoff is not None:
        _draw_cutoff(
            ax,
            cutoff,
            "high",
            label=f"Cutoff ({cutoff:g})",
            tail_label=(
                f"Flagged (n={int(np.count_nonzero(valid >= cutoff))})"
                if flagged_label is None
                else None
            ),
        )
    ax.set_xlabel("Composite score")
    ax.set_ylabel("Count")
    ax.set_title("Composite Score Distribution")
    if flag_arr is not None or cutoff is not None:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_index_agreement(
    screen_result: ScreenResult,
    kind: AgreementKind = "jaccard",
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str | None = None,
    annotate: bool = True,
) -> Any:
    """
    Plot an index-by-index agreement matrix from index_agreement().

    Color limits are fixed so plots are comparable across studies: overlap
    (as the share of respondents flagged by both indices) and Jaccard use
    ``[0, 1]`` with ``viridis``; Spearman correlations use ``[-1, 1]`` with
    ``RdBu_r``. Spearman scores are oriented by each index's flag direction, so
    positive (red) cells mean both indices rank the same respondents as more
    suspicious and negative (blue) cells mean they disagree. Undefined cells are
    gray. Annotations show co-flag counts for overlap and two-decimal values
    otherwise.

    Parameters:
    - screen_result: Output dict from screen().
    - kind: ``"overlap"``, ``"jaccard"``, or ``"spearman"``.
    - figsize: Figure size as (width, height). If None, auto-calculated.
    - cmap: Matplotlib colormap name overriding the kind's default.
    - annotate: Write each cell's value.

    Returns:
    - matplotlib Figure object.

    Raises:
    - ValueError: If kind is not recognized or annotate is not a boolean.
    - RuntimeError: If matplotlib is not available.

    Example:
        >>> from ier import plot_index_agreement, screen_scores
        >>> result = screen_scores(
        ...     {"irv": [0.1, 0.2, 0.9, 1.2], "longstring": [8.0, 3.0, 2.0, 7.0]},
        ...     thresholds={"irv": 0.5, "longstring": 5.0},
        ... )
        >>> fig = plot_index_agreement(result)
        >>> cells = [text.get_text() for text in fig.axes[0].texts]
        >>> cells
        ['1.00', '0.33', '0.33', '1.00']
    """
    if not isinstance(annotate, bool):
        raise ValueError("annotate must be a boolean")
    names, matrix = index_agreement(screen_result, kind)
    plt = require_matplotlib_pyplot()

    n_indices = len(names)
    if n_indices == 0:
        fig, _ = plt.subplots(1, 1, figsize=figsize or (6, 4))
        return fig

    correlation = kind == "spearman"
    values = matrix / max(screen_result["n_respondents"], 1) if kind == "overlap" else matrix
    colormap = plt.get_cmap(cmap or ("RdBu_r" if correlation else "viridis")).copy()
    colormap.set_bad(_UNAVAILABLE_COLOR)

    if figsize is None:
        side = min(12.0, max(4.0, 0.7 * n_indices + 2.0))
        figsize = (side + 1.5, side)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    image = ax.imshow(
        values,
        cmap=colormap,
        interpolation="nearest",
        vmin=-1.0 if correlation else 0.0,
        vmax=1.0,
    )
    ax.set_xticks(range(n_indices))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n_indices))
    ax.set_yticklabels(names)
    ax.set_title(f"IER Index Agreement ({kind})")
    fig.colorbar(image, ax=ax, label=_AGREEMENT_LABELS[kind])
    if annotate:
        for row in range(n_indices):
            for column in range(n_indices):
                value = values[row, column]
                if np.isnan(value):
                    text, color = "n/a", "black"
                else:
                    text = f"{matrix[row, column]:.0f}" if kind == "overlap" else f"{value:.2f}"
                    red, green, blue, _ = colormap(image.norm(value))
                    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                    color = "black" if luminance > 0.5 else "white"
                ax.text(column, row, text, ha="center", va="center", color=color)
    fig.tight_layout()
    return fig
