"""Shared input validation utilities for careless detection functions."""

import math
import numbers
import warnings
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from operator import index
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike


class SupportsArray(Protocol):
    """Protocol for objects convertible to numpy arrays (e.g., pandas/polars DataFrame)."""

    def __array__(self, dtype: Any | None = None) -> np.ndarray: ...


MatrixLike: TypeAlias = Sequence[Sequence[float | int]] | np.ndarray | SupportsArray | ArrayLike

_ARRAY_LIKE_MESSAGE = "input data must be array-like (list, tuple, numpy array, or DataFrame)"


class _MatrixInputTypeError(TypeError, ValueError):
    """Reject a non-array input while remaining catchable as either legacy error type."""


def validate_integer(
    value: object,
    *,
    message: str,
    minimum: int | None = None,
    minimum_message: str | None = None,
) -> int:
    """Return an integer option as a Python int, rejecting booleans and non-integers."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        integer = index(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(message) from error
    if minimum is not None and integer < minimum:
        raise ValueError(minimum_message or message)
    return integer


def validate_probability(value: object, *, name: str) -> float:
    """Return a finite probability in ``[0, 1]`` as a Python float.

    Python and NumPy real scalars, 0-d real arrays, ``Decimal`` and ``Fraction``
    values are accepted. Booleans, strings, bytes, and other non-real values are
    rejected with the same ``"<name> must be between 0 and 1"`` message prefix as
    out-of-range and non-finite values, followed by the reason.
    """
    message = f"{name} must be between 0 and 1"
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    # NumPy registers timedelta64 as an integer; neither temporal type is a number.
    rejected = (bool, np.bool_, str, bytes, np.datetime64, np.timedelta64)
    if isinstance(value, rejected) or not isinstance(value, (numbers.Real, Decimal)):
        raise ValueError(f"{message} (expected a real number, got {type(value).__name__})")
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        # Huge integers and fractions overflow; signaling Decimal NaNs refuse conversion.
        raise ValueError(f"{message} (got a value with no float equivalent)") from error
    # NaN fails both comparisons, so only finite levels in range remain.
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{message} (got {probability!r})")
    return probability


def validate_column_index(value: object, n_columns: int, *, name: str) -> int:
    """Return one in-bounds zero-based column index as a Python int."""
    item_index = validate_integer(value, message=f"{name} must contain integer column indices")
    if item_index < 0 or item_index >= n_columns:
        raise ValueError(f"item index {item_index} out of bounds for data with {n_columns} columns")
    return item_index


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


def validate_item_indices(
    item_indices: Sequence[int], n_columns: int, *, name: str = "item_indices"
) -> np.ndarray:
    """Validate an ordered, nonempty selection of distinct matrix columns."""
    selected = [validate_column_index(value, n_columns, name=name) for value in item_indices]
    if not selected:
        raise ValueError(f"{name} cannot be empty")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{name} cannot contain duplicates")
    return np.asarray(selected, dtype=np.intp)


def validate_score_array(values: ArrayLike, *, name: str = "scores") -> np.ndarray:
    """Validate one real numeric score vector without silently coercing its type."""
    try:
        score_arr = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if score_arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(score_arr) == 0:
        raise ValueError(f"{name} cannot be empty")
    if score_arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    # Reject wider floating values outside float64 range through the same
    # finite-value check, without leaking a NumPy conversion warning.
    with np.errstate(over="ignore", invalid="ignore"):
        score_arr = np.asarray(score_arr, dtype=float)
    if np.isinf(score_arr).any():
        raise ValueError(f"{name} must contain only finite values or NaN")
    return score_arr


def validate_score_vectors(
    scores: Mapping[str, ArrayLike],
    *,
    n_respondents: int | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Validate aligned vectors, using an explicit count when none are available."""
    if not isinstance(scores, Mapping):
        raise TypeError("scores must be a mapping of registered index names to score arrays")
    if n_respondents is not None:
        message = "n_respondents must be a positive integer within the platform index range or None"
        n_respondents = validate_integer(n_respondents, message=message, minimum=1)
        if n_respondents > np.iinfo(np.intp).max:
            raise ValueError(message)
    if not scores and n_respondents is None:
        raise ValueError("scores must contain at least one registered index")

    validated: dict[str, np.ndarray] = {}
    actual_respondents: int | None = None
    for name, values in scores.items():
        score_arr = validate_score_array(values, name=f"scores for {name}")
        if actual_respondents is None:
            actual_respondents = len(score_arr)
        elif len(score_arr) != actual_respondents:
            raise ValueError("all score arrays must have the same respondent count")
        if n_respondents is not None and len(score_arr) != n_respondents:
            raise ValueError("score arrays must match n_respondents")
        validated[name] = score_arr

    if n_respondents is None:
        n_respondents = actual_respondents
    assert n_respondents is not None
    return validated, n_respondents


def _is_pandas_missing_type(value_type: type) -> bool:
    """Recognize ``pd.NA`` without importing pandas."""
    return value_type.__name__ == "NAType"


def _is_response_type(value_type: type) -> bool:
    """Return whether float() maps every value of one object type to a response or NaN."""
    if value_type is type(None) or _is_pandas_missing_type(value_type):
        return True
    # NumPy registers timedelta64 as an integer; neither temporal type is a response.
    if issubclass(value_type, (np.datetime64, np.timedelta64)):
        return False
    return issubclass(value_type, (bool, np.bool_, numbers.Real, Decimal, str, bytes))


def _pandas_missing_to_nan(value: object) -> object:
    """Replace ``pd.NA`` with NaN and leave other responses for float()."""
    return math.nan if _is_pandas_missing_type(type(value)) else value


def _real_frame_values(source: object, shape: tuple[int, ...]) -> np.ndarray | None:
    """Convert pandas-style frames whose columns all have real numeric dtypes."""
    to_numpy = getattr(source, "to_numpy", None)
    dtypes = getattr(source, "dtypes", None)
    if not callable(to_numpy) or not isinstance(dtypes, Iterable):
        return None
    kinds = {getattr(dtype, "kind", "O") for dtype in dtypes}
    if not kinds or not kinds.issubset("biuf"):
        return None
    try:
        values = np.asarray(to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return values if values.shape == shape else None


def _masked_responses(x: np.ma.MaskedArray) -> np.ndarray:
    """Return masked cells as missing responses, copying only when a cell is masked."""
    data = np.asarray(x)  # the underlying values without their mask or a copy
    kind = data.dtype.kind
    mask = np.ma.getmask(x)
    # Other dtypes, such as complex or datetime data, fail the shared dtype rules.
    if kind not in "biufOUST" or not np.any(mask):
        return data
    responses: np.ndarray
    if kind in "biuf":
        # NaN needs a floating type; floating inputs keep their own precision.
        responses = data.astype(data.dtype if kind == "f" else np.float64)
        responses[mask] = np.nan
    else:
        # The object rule converts None to NaN alongside the unmasked responses.
        responses = data.astype(object)
        responses[mask] = None
    return responses


def _real_numeric_matrix(source: object, x_array: np.ndarray) -> np.ndarray:
    """Return real numeric responses, converting object and text inputs to float64."""
    kind = x_array.dtype.kind
    if kind in "biuf":
        # Keep integer arrays exact and avoid copying numeric inputs.
        return x_array
    message = f"input data must contain real numeric responses (got dtype {x_array.dtype})"
    if kind == "O":
        # Nullable extension frames (Int64, Float64, boolean) become object
        # arrays holding pd.NA; their own conversion maps missing values to NaN.
        values = _real_frame_values(source, x_array.shape)
        if values is not None:
            return values
        value_types = set(map(type, x_array.flat))
        if not all(map(_is_response_type, value_types)):
            raise ValueError(message)
        if any(map(_is_pandas_missing_type, value_types)):
            x_array = np.frompyfunc(_pandas_missing_to_nan, 1, 1)(x_array)
    elif kind not in "UST":
        raise ValueError(message)
    try:
        # None becomes NaN; numeric strings, Decimal and Fraction values use float().
        return np.asarray(x_array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error


def validate_matrix_input(
    x: MatrixLike | None,
    allow_1d: bool = False,
    min_columns: int = 1,
    dtype: type | None = None,
) -> np.ndarray:
    """
    Validate and convert input data to a 2D numpy array.

    Every array-like input (lists, tuples, sequences, NumPy arrays, and objects
    implementing ``__array__`` such as pandas or polars DataFrames) follows the
    same rules. Boolean, integer, and floating arrays are returned without a copy,
    so integer responses keep exact integer reductions. Object, extension, and
    text inputs are converted to float64: ``None`` and ``pd.NA`` become NaN, and
    numeric strings, ``Decimal`` and ``Fraction`` values are converted with
    ``float()``. ``pd.NaT`` is a datetime value and is rejected like other
    temporal data. Integers above 2**53 in object inputs round during that
    conversion; pass an integer ndarray to keep them exact. Masked cells of a
    ``numpy.ma.MaskedArray`` become NaN: only an array with at least one masked
    cell is copied (boolean and integer data to float64), while arrays without
    masked cells follow the rules above.

    Parameters:
    - x: Input data to validate (array-like)
    - allow_1d: If True, reshape 1D arrays to 2D (1 row)
    - min_columns: Minimum number of columns required
    - dtype: Optional dtype to convert the array to (e.g., float)

    Returns:
    - Validated 2D numpy array

    Raises:
    - ValueError: If data is None, empty, not real numeric (complex, datetime,
                  timedelta, or non-numeric text), or doesn't meet dimensional
                  requirements
    - TypeError: If data is a scalar, string, mapping, or other non-array object.
                 The exception also derives from ValueError.
    """
    if x is None:
        raise ValueError("input data cannot be None")

    if isinstance(x, (str, bytes, Mapping)):
        raise _MatrixInputTypeError(_ARRAY_LIKE_MESSAGE)

    if isinstance(x, np.ma.MaskedArray):
        # np.asarray() would drop the mask and score masked sentinel values.
        x = _masked_responses(x)

    if isinstance(x, np.ndarray) and x.size == 0:
        raise ValueError("input data cannot be empty")

    if isinstance(x, (list, tuple)) and len(x) == 0:
        raise ValueError("input data cannot be empty")

    x_array = np.asarray(x)
    if x_array.ndim == 0:
        raise _MatrixInputTypeError(_ARRAY_LIKE_MESSAGE)
    x_array = _real_numeric_matrix(x, x_array)
    if dtype is not None:
        x_array = np.asarray(x_array, dtype=dtype)

    if allow_1d and x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)

    if x_array.ndim != 2:
        raise ValueError("input data must be 2-dimensional")

    if x_array.shape[0] == 0 or x_array.shape[1] == 0:
        raise ValueError("input data cannot be empty")

    if x_array.shape[1] < min_columns:
        raise ValueError(f"data must have at least {min_columns} columns")

    return x_array
