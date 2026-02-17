"""Shared input validation utilities for careless detection functions."""

from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike


class SupportsArray(Protocol):
    """Protocol for objects convertible to numpy arrays (e.g., pandas/polars DataFrame)."""

    def __array__(self, dtype: Any | None = None) -> np.ndarray: ...


MatrixLike: TypeAlias = Sequence[Sequence[float | int]] | np.ndarray | SupportsArray | ArrayLike


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

    x_array = np.array(x, dtype=dtype) if dtype is not None else np.asarray(x)

    if allow_1d and x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)

    if x_array.ndim != 2:
        raise ValueError("input data must be 2-dimensional")

    if x_array.shape[0] == 0 or x_array.shape[1] == 0:
        raise ValueError("input data cannot be empty")

    if x_array.shape[1] < min_columns:
        raise ValueError(f"data must have at least {min_columns} columns")

    return x_array
