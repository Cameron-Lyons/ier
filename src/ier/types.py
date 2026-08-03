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
ScoreArchiveResultType: TypeAlias = Literal["screen", "composite"]

CompositeMethod: TypeAlias = Literal["mean", "sum", "max", "best_subset"]
InfrequencyMissingPolicy: TypeAlias = Literal["pass", "fail", "omit", "propagate"]
ResponseTimeFlagDirection: TypeAlias = Literal["high", "low"]
ResponseTimeMetric: TypeAlias = Literal["mean", "median", "sd", "min", "consistency", "mixture"]
ResponseTimeThresholdSource: TypeAlias = Literal["fixed", "percentile"]


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


class ScoreArchive(TypedDict):
    """Reusable registered-index scores loaded from a versioned NPZ archive."""

    schema_version: int
    result_type: ScoreArchiveResultType
    n_respondents: int
    scores: IndexScoreMap
    respondent_ids: list[str] | None
    errors: IndexErrorMap


class ResponseTimeArchive(TypedDict):
    """Validated response-time scores loaded from a versioned NPZ archive."""

    schema_version: int
    result_type: Literal["response_time"]
    n_respondents: int
    metric: ResponseTimeMetric
    flag_direction: ResponseTimeFlagDirection
    threshold: float
    threshold_source: ResponseTimeThresholdSource | None
    percentile: float | None
    scores: FloatArray
    flags: BoolArray
    respondent_ids: list[str] | None


ResultArchive: TypeAlias = ScoreArchive | ResponseTimeArchive


class ResponseTimeMixtureModelArchive(TypedDict):
    """Validated response-time mixture model loaded through generic detection."""

    schema_version: int
    result_type: Literal["response_time_mixture_model"]
    n_components: int
    fast_component: int
    log_transform: bool
    weights: FloatArray
    means: FloatArray
    variances: FloatArray


InspectableArchive: TypeAlias = ResultArchive | ResponseTimeMixtureModelArchive


class MahadSummary(TypedDict):
    """Summary statistics and outlier counts from ``mahad_summary()``."""

    mean: float
    std: float
    min: float
    max: float
    median: float
    outliers: int
    total: int
    valid_count: int
    missing_count: int


class MarkovSummary(TypedDict):
    """Summary statistics and coverage counts from ``markov_summary()``."""

    mean: float
    std: float
    min: float
    max: float
    median: float
    n_total: int
    n_valid: int
    n_missing: int


class PsychsynSummary(TypedDict):
    """Summary statistics and pair coverage from ``psychsyn_summary()``."""

    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    median_score: float
    item_pairs: int
    total_individuals: int
    valid_individuals: int
    missing_individuals: int


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
