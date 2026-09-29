"""Shared input validation utilities for careless detection functions."""

import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike


class SupportsArray(Protocol):
    """Protocol for objects convertible to numpy arrays (e.g., pandas/polars DataFrame)."""

    def __array__(self, dtype: Any | None = None) -> np.ndarray: ...


MatrixLike: TypeAlias = Sequence[Sequence[float | int]] | np.ndarray | SupportsArray | ArrayLike


def resolve_scale_bounds(
    x: np.ndarray,
    *,
    scale_min: float | None,
    scale_max: float | None,
) -> tuple[float, float] | None:
    """Resolve ordered endpoints, returning None when an endpoint is unavailable."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        resolved_min = np.nanmin(x) if scale_min is None else scale_min
        resolved_max = np.nanmax(x) if scale_max is None else scale_max

    # Keep exact integer endpoints while avoiding fixed-width scalar arithmetic.
    if isinstance(resolved_min, np.generic):
        resolved_min = resolved_min.item()
    if isinstance(resolved_max, np.generic):
        resolved_max = resolved_max.item()

    if np.isnan(resolved_min) or np.isnan(resolved_max):
        return None
    if resolved_max < resolved_min:
        raise ValueError("scale_max must be greater than or equal to scale_min")
    return resolved_min, resolved_max


def validate_item_indices(item_indices: Sequence[int], n_columns: int) -> np.ndarray:
    """Validate an ordered, nonempty selection of distinct matrix columns."""
    selected = list(item_indices)
    if not selected:
        raise ValueError("item_indices cannot be empty")
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("item_indices must contain integer column indices")
        if index < 0 or index >= n_columns:
            raise ValueError(f"item index {index} out of bounds for data with {n_columns} columns")
    if len(set(selected)) != len(selected):
        raise ValueError("item_indices cannot contain duplicates")
    return np.asarray(selected, dtype=np.intp)


def validate_score_array(values: ArrayLike, *, name: str = "scores") -> np.ndarray:
    """Validate and return one non-empty one-dimensional reusable score vector."""
    try:
        score_arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if score_arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(score_arr) == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.isinf(score_arr).any():
        raise ValueError(f"{name} must contain only finite values or NaN")
    return score_arr


def validate_score_vectors(
    scores: Mapping[str, ArrayLike],
) -> tuple[dict[str, np.ndarray], int]:
    """Validate a non-empty, respondent-aligned mapping of reusable score vectors."""
    if not isinstance(scores, Mapping):
        raise TypeError("scores must be a mapping of registered index names to score arrays")
    if not scores:
        raise ValueError("scores must contain at least one registered index")

    validated: dict[str, np.ndarray] = {}
    n_respondents: int | None = None
    for name, values in scores.items():
        score_arr = validate_score_array(values, name=f"scores for {name}")
        if n_respondents is None:
            n_respondents = len(score_arr)
        elif len(score_arr) != n_respondents:
            raise ValueError("all score arrays must have the same respondent count")
        validated[name] = score_arr

    assert n_respondents is not None
    return validated, n_respondents


def iter_rows(x_array: np.ndarray, na_rm: bool) -> Iterator[np.ndarray]:
    """Yield rows with optional NaN removal for repeated row-wise index routines."""
    for row in x_array:
        yield row[~np.isnan(row)] if na_rm else row


def validate_matrix_input(
    x: MatrixLike | None,
    allow_1d: bool = False,
    min_columns: int = 1,
    dtype: type | None = None,
    check_type: bool = True,
) -> np.ndarray:
    """
    Validate and convert input data to a 2D numpy array.

    Parameters:
    - x: Input data to validate (list or numpy array)
    - allow_1d: If True, reshape 1D arrays to 2D (1 row)
    - min_columns: Minimum number of columns required
    - dtype: Optional dtype to convert the array to (e.g., float)
    - check_type: If True, validate that input is a list or numpy array

    Returns:
    - Validated 2D numpy array

    Raises:
    - ValueError: If data is None, empty, or doesn't meet dimensional requirements
    - TypeError: If data is not a list or numpy array (when check_type=True)
    """
    if x is None:
        raise ValueError("input data cannot be None")

    if check_type and not isinstance(x, (list, tuple, np.ndarray)) and not hasattr(x, "__array__"):
        raise TypeError("input data must be array-like (list, tuple, numpy array, or DataFrame)")

    if isinstance(x, np.ndarray) and x.size == 0:
        raise ValueError("input data cannot be empty")

    if isinstance(x, (list, tuple)) and len(x) == 0:
        raise ValueError("input data cannot be empty")

    x_array = np.asarray(x, dtype=dtype)

    if allow_1d and x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)

    if x_array.ndim != 2:
        raise ValueError("input data must be 2-dimensional")

    if x_array.shape[0] == 0 or x_array.shape[1] == 0:
        raise ValueError("input data cannot be empty")

    if x_array.shape[1] < min_columns:
        raise ValueError(f"data must have at least {min_columns} columns")

    return x_array
