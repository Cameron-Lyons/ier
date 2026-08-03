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
IndexThresholdMap: TypeAlias = dict[str, float | None]
IndexPercentileMap: TypeAlias = dict[str, float | None]
IndexThresholdSourceMap: TypeAlias = dict[str, Literal["fixed", "percentile", "presence"]]

CompositeMethod: TypeAlias = Literal["mean", "sum", "max", "best_subset"]
InfrequencyMissingPolicy: TypeAlias = Literal["pass", "fail", "omit", "propagate"]


class IndexMetadata(TypedDict):
    """Public metadata for one registered IER index."""

    flag_direction: Literal["high", "low"]
    flag_mode: Literal["percentile", "present"]
    default_screen: bool
    default_composite: bool
    composite_enabled: bool
    required_options: tuple[str, ...]


IndexCatalog: TypeAlias = dict[str, IndexMetadata]


class ScreenIndexSummary(TypedDict):
    """Summary statistics for one index in screen()."""

    mean: float
    std: float
    min: float
    max: float
    n_valid: int
    n_unavailable: int
    n_flagged: int
    flag_rate: float


class ScreenResult(TypedDict):
    """Return value for screen()."""

    scores: IndexScoreMap
    flags: IndexFlagMap
    thresholds: IndexThresholdMap
    threshold_sources: IndexThresholdSourceMap
    percentiles: IndexPercentileMap
    flag_counts: IntArray
    valid_index_counts: IntArray
    consensus_eligible: BoolArray
    consensus_flags: BoolArray
    min_flags: int
    min_valid_indices: int | None
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
    weights: dict[str, float]
    min_valid_indices: int | None
    valid_index_counts: IntArray
    mean: float
    std: float
    min: float
    max: float
    n_total: int
    n_valid: int
