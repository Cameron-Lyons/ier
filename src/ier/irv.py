"""
The IRV is the "standard deviation of responses across a set of consecutive item responses for
an individual" (Dunn, Heggestad, Shanock, & Theilgard, 2018, p. 108). By default, the IRV is
calculated across all columns of the input data. Additionally it can be applied to different subsets
of the data. This can detect degraded response quality which occurs only in a certain section of the
questionnaire (usually the end). Whereas Dunn et al. (2018) propose to mark persons with low IRV
scores as outliers - reflecting straightlining responses, Marjanovic et al. (2015) propose to mark
persons with high IRV scores - reflecting highly random responses
"""

from itertools import pairwise
from operator import index

import numpy as np

from ier._row_statistics import _row_mean_std_block, row_slices, row_std
from ier._validation import MatrixLike, validate_matrix_input


def irv(
    x: MatrixLike,
    na_rm: bool = True,
    split: bool = False,
    num_split: int = 1,
    split_points: list[int] | None = None,
) -> np.ndarray:
    """
    Calculate intra-individual response variability (IRV) for each individual.

    IRV measures the standard deviation of responses across consecutive items for each individual.
    Low IRV scores may indicate straightlining (consistent responses), while high IRV scores
    may indicate random or inconsistent responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
          Can be a 2D list or numpy array.
    - na_rm: Boolean indicating whether to ignore missing values (np.nan) during computation.
             If False, missing values propagate. Both modes accumulate in double precision.
    - split: Boolean indicating whether to calculate IRV on subsets of columns and return the mean.
    - num_split: Positive integer number of subsets if 'split' is True. Earlier subsets
                 receive one extra item when lengths differ; empty subsets are ignored.
                 Ignored if split_points is provided.
    - split_points: Optional list of column indices to use as split points. If provided,
                   overrides num_split. For example, [0, 10, 20] would split into
                   [0:10] and [10:20]. Points must be strictly increasing integers,
                   starting at zero and ending at the number of columns.

    Returns:
    - A numpy array of population standard deviations for each individual. Split mode
      averages section scores with equal weight, regardless of section length. An entirely
      missing section leaves the respondent's split score unavailable (NaN).

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid split parameters, etc.)

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [1, 1, 1, 4, 5, 6]]
        >>> irv_scores = irv(data)
        >>> print(irv_scores)
        [1.70782513 2.081666  ]

        >>> irv_split = irv(data, split=True, num_split=2)
        >>> print(irv_split)
        [0.81649658 0.40824829]

        >>> irv_custom = irv(data, split=True, split_points=[0, 3, 6])
        >>> print(irv_custom)
        [0.81649658 0.40824829]
    """

    x_array = validate_matrix_input(x, check_type=False)

    groups = (
        _split_groups(x_array.shape[1], num_split, split_points)
        if split or split_points is not None
        else []
    )
    if not split:
        return row_std(x_array, ignore_nan=na_rm)

    if abs(x_array.strides[0]) < abs(x_array.strides[1]):
        # Column-contiguous data reduce faster one section at a time. Keep
        # singleton runs together because they only need finiteness checks.
        separate = []
        for first, last, width in groups:
            if width == 1:
                separate.append((first, last, width))
            else:
                separate.extend(
                    (begin, begin + width, width) for begin in range(first, last, width)
                )
        groups = separate
    return _split_irv_scores(x_array, groups, ignore_nan=na_rm)


def _split_irv_scores(
    x: np.ndarray,
    groups: list[tuple[int, int, int]],
    *,
    ignore_nan: bool,
    scaled: bool = False,
) -> np.ndarray:
    """Average section deviations, rescaling exceptional sums in bounded blocks."""
    scores = np.zeros(len(x))
    for first, last, width in groups:
        for start, stop in row_slices(len(x), last - first):
            block = x[start:stop, first:last]
            if width == 1:
                # A finite singleton has zero deviation; an unavailable section
                # propagates through the respondent's equally weighted mean.
                scores[start:stop][~np.isfinite(block).all(axis=1)] = np.nan
                continue
            # Splitting the column axis preserves views for both array layouts.
            sections = block.reshape(stop - start, -1, width)
            _, deviations = _row_mean_std_block(sections, ignore_nan=ignore_nan)
            with np.errstate(over="ignore", under="ignore"):
                if scaled:
                    np.ldexp(deviations, -1024, out=deviations)
                scores[start:stop] += deviations.sum(axis=1, dtype=float)
    scores /= sum((last - first) // width for first, last, width in groups)
    if scaled:
        # Rounding at the upper boundary must not turn a finite average infinite.
        np.minimum(scores, np.ldexp(np.finfo(float).max, -1024), out=scores)
        np.ldexp(scores, 1024, out=scores)
        return scores

    overflow = np.isinf(scores)
    if np.any(overflow):
        for start, stop in row_slices(len(x), x.shape[1]):
            affected = overflow[start:stop]
            if np.any(affected):
                scores[start:stop][affected] = _split_irv_scores(
                    x[start:stop][affected], groups, ignore_nan=ignore_nan, scaled=True
                )
    return scores


def _split_groups(
    n_columns: int, num_split: int, split_points: list[int] | None
) -> list[tuple[int, int, int]]:
    """Validate sections and group adjacent equal widths as (first, last, width)."""
    if split_points is not None:
        if not isinstance(split_points, list) or len(split_points) < 2:
            raise ValueError("split_points must be a list with at least 2 elements")
        if any(isinstance(point, (bool, np.bool_)) for point in split_points):
            raise ValueError("split_points must contain integer column positions")
        try:
            split_points = [index(point) for point in split_points]
        except TypeError as error:
            raise ValueError("split_points must contain integer column positions") from error
        if split_points[0] != 0:
            raise ValueError("first split point must be 0")
        if split_points[-1] != n_columns:
            raise ValueError(f"last split point must be {n_columns} (number of columns)")

        groups: list[tuple[int, int, int]] = []
        for first, last in pairwise(split_points):
            width = last - first
            if width <= 0:
                raise ValueError("split points must be in ascending order")
            if groups and groups[-1][2] == width:
                groups[-1] = (groups[-1][0], last, width)
            else:
                groups.append((first, last, width))
        return groups

    if isinstance(num_split, (bool, np.bool_)):
        raise ValueError("num_split must be a positive integer")
    try:
        num_split = index(num_split)
    except TypeError as error:
        raise ValueError("num_split must be a positive integer") from error
    if num_split < 1:
        raise ValueError("num_split must be a positive integer")
    # Extra empty sections do not contribute to IRV; avoid constructing them.
    num_split = min(num_split, n_columns)
    width, extra = divmod(n_columns, num_split)
    boundary = extra * (width + 1)
    groups = [(0, boundary, width + 1)] if extra else []
    groups.append((boundary, n_columns, width))
    return groups
