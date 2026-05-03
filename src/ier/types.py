"""Public result types for IER orchestration APIs."""

from typing import Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.int_]

IndexScoreMap: TypeAlias = dict[str, FloatArray]
IndexFlagMap: TypeAlias = dict[str, BoolArray]
IndexErrorMap: TypeAlias = dict[str, str]

CompositeMethod: TypeAlias = Literal["mean", "sum", "max", "best_subset"]


class ScreenIndexSummary(TypedDict):
    """Summary statistics for one index in screen()."""

    mean: float
    std: float
    min: float
    max: float
    n_flagged: int


class ScreenResult(TypedDict):
    """Return value for screen()."""

    scores: IndexScoreMap
    flags: IndexFlagMap
    flag_counts: IntArray
    n_indices: int
    indices_used: list[str]
    errors: IndexErrorMap
    n_respondents: int
    summary: dict[str, ScreenIndexSummary]


class CompositeSummary(TypedDict):
    """Return value for composite_summary()."""

    composite: FloatArray
    indices: IndexScoreMap
    indices_used: list[str]
    errors: IndexErrorMap
    method: CompositeMethod
    standardized: bool
    mean: float
    std: float
    min: float
    max: float
    n_total: int
    n_valid: int
