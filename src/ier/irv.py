"""
The IRV is the "standard deviation of responses across a set of consecutive item responses for
an individual" (Dunn, Heggestad, Shanock, & Theilgard, 2018, p. 108). By default, the IRV is
calculated across all columns of the input data. Additionally it can be applied to different subsets
of the data. This can detect degraded response quality which occurs only in a certain section of the
questionnaire (usually the end). Whereas Dunn et al. (2018) propose to mark persons with low IRV
scores as outliers - reflecting straightlining responses, Marjanovic et al. (2015) propose to mark
persons with high IRV scores - reflecting highly random responses
"""

import numpy as np

from ier._row_statistics import row_std
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
             If True, uses np.nanstd; if False, uses np.std.
    - split: Boolean indicating whether to calculate IRV on subsets of columns and return the mean.
    - num_split: Number of equal-sized subsets to split the data into if 'split' is True.
                 Ignored if split_points is provided.
    - split_points: Optional list of column indices to use as split points. If provided,
                   overrides num_split. For example, [0, 10, 20] would split into
                   [0:10] and [10:20].

    Returns:
    - A numpy array of IRV values for each individual.

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid split parameters, etc.)

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [1, 1, 1, 4, 5, 6]]
        >>> irv_scores = irv(data)
        >>> print(irv_scores)
        [1.87, 2.16]

        >>> irv_split = irv(data, split=True, num_split=2)
        >>> print(irv_split)
        [1.87, 2.16]

        >>> irv_custom = irv(data, split=True, split_points=[0, 3, 6])
        >>> print(irv_custom)
        [1.87, 2.16]
    """

    x_array = validate_matrix_input(x, check_type=False)

    if split_points is not None:
        if not isinstance(split_points, list) or len(split_points) < 2:
            raise ValueError("split_points must be a list with at least 2 elements")

        if split_points[0] != 0:
            raise ValueError("first split point must be 0")

        if split_points[-1] != x_array.shape[1]:
            raise ValueError(f"last split point must be {x_array.shape[1]} (number of columns)")

        for i in range(1, len(split_points)):
            if split_points[i] <= split_points[i - 1]:
                raise ValueError("split points must be in ascending order")
            if split_points[i] > x_array.shape[1]:
                raise ValueError(f"split point {split_points[i]} exceeds number of columns")

    elif split and num_split <= 0:
        raise ValueError("num_split must be greater than 0")

    if not split:
        return row_std(x_array, ignore_nan=na_rm)

    if split_points is not None:
        chunks = [
            x_array[:, split_points[i] : split_points[i + 1]]
            for i in range(len(split_points) - 1)
            if split_points[i + 1] > split_points[i]
        ]
    else:
        chunks = np.array_split(x_array, num_split, axis=1)

    split_result: np.ndarray | None = None
    n_chunks = 0
    for chunk in chunks:
        if chunk.size == 0:
            continue
        chunk_result = row_std(chunk, ignore_nan=na_rm)
        if split_result is None:
            split_result = chunk_result
        else:
            split_result += chunk_result
        n_chunks += 1

    if split_result is not None:
        np.divide(split_result, n_chunks, out=split_result)
        return split_result

    return row_std(x_array, ignore_nan=na_rm)
