"""IER: Python library for detecting Insufficient Effort Responding in survey data."""

from importlib.metadata import PackageNotFoundError, version

from ._registry import IndexOptions as IndexOptions
from ._registry import index_catalog as index_catalog
from ._validation import MatrixLike as MatrixLike
from .acquiescence import acquiescence as acquiescence
from .acquiescence import acquiescence_flag as acquiescence_flag
from .archive import flag_consensus_archives as flag_consensus_archives
from .archive import load_archive as load_archive
from .archive import load_flag_consensus_archive as load_flag_consensus_archive
from .archive import load_psychsyn_model as load_psychsyn_model
from .archive import load_response_time_archive as load_response_time_archive
from .archive import load_response_time_mixture_model as load_response_time_mixture_model
from .archive import load_score_archive as load_score_archive
from .archive import merge_flag_consensus_archives as merge_flag_consensus_archives
from .archive import merge_score_archives as merge_score_archives
from .archive import save_flag_consensus_archive as save_flag_consensus_archive
from .archive import save_psychsyn_model as save_psychsyn_model
from .archive import save_response_time_archive as save_response_time_archive
from .archive import save_response_time_mixture_model as save_response_time_mixture_model
from .archive import save_score_archive as save_score_archive
from .composite import composite as composite
from .composite import composite_flag as composite_flag
from .composite import composite_probability as composite_probability
from .composite import composite_scores as composite_scores
from .composite import composite_summary as composite_summary
from .evenodd import evenodd as evenodd
from .guttman import guttman as guttman
from .guttman import guttman_flag as guttman_flag
from .infrequency import infrequency as infrequency
from .infrequency import infrequency_flag as infrequency_flag
from .irv import irv as irv
from .longstring import longstring as longstring
from .longstring import longstring_pattern as longstring_pattern
from .longstring import longstring_scores as longstring_scores
from .lz import lz as lz
from .lz import lz_flag as lz_flag
from .mad import mad as mad
from .mad import mad_flag as mad_flag
from .mahad import mahad as mahad
from .mahad import mahad_qqplot as mahad_qqplot
from .mahad import mahad_summary as mahad_summary
from .markov import markov as markov
from .markov import markov_flag as markov_flag
from .markov import markov_summary as markov_summary
from .missing import missing_rate as missing_rate
from .missing import missing_rate_flag as missing_rate_flag
from .onset import onset as onset
from .onset import onset_flag as onset_flag
from .person_total import person_total as person_total
from .psychsyn import PsychsynModel as PsychsynModel
from .psychsyn import fit_psychsyn_model as fit_psychsyn_model
from .psychsyn import psychant as psychant
from .psychsyn import psychsyn as psychsyn
from .psychsyn import psychsyn_model_scores as psychsyn_model_scores
from .psychsyn import psychsyn_summary as psychsyn_summary
from .reliability import individual_reliability as individual_reliability
from .reliability import individual_reliability_flag as individual_reliability_flag
from .response_time import ResponseTimeMixtureModel as ResponseTimeMixtureModel
from .response_time import fit_response_time_mixture as fit_response_time_mixture
from .response_time import response_time as response_time
from .response_time import response_time_consistency as response_time_consistency
from .response_time import response_time_flag as response_time_flag
from .response_time import response_time_mixture as response_time_mixture
from .response_time import response_time_mixture_scores as response_time_mixture_scores
from .response_time import response_time_score_flags as response_time_score_flags
from .screen import flag_consensus as flag_consensus
from .screen import screen as screen
from .screen import screen_scores as screen_scores
from .semantic import semantic_ant as semantic_ant
from .semantic import semantic_ant_flag as semantic_ant_flag
from .semantic import semantic_syn as semantic_syn
from .semantic import semantic_syn_flag as semantic_syn_flag
from .types import BoolArray as BoolArray
from .types import CompositeMethod as CompositeMethod
from .types import CompositeSummary as CompositeSummary
from .types import FlagConsensusArchive as FlagConsensusArchive
from .types import FlagConsensusResult as FlagConsensusResult
from .types import FloatArray as FloatArray
from .types import IndexCatalog as IndexCatalog
from .types import IndexErrorMap as IndexErrorMap
from .types import IndexFlagMap as IndexFlagMap
from .types import IndexMetadata as IndexMetadata
from .types import IndexScoreMap as IndexScoreMap
from .types import InfrequencyMissingPolicy as InfrequencyMissingPolicy
from .types import InspectableArchive as InspectableArchive
from .types import IntArray as IntArray
from .types import MahadSummary as MahadSummary
from .types import MarkovSummary as MarkovSummary
from .types import PsychsynModelArchive as PsychsynModelArchive
from .types import PsychsynSummary as PsychsynSummary
from .types import ResponseTimeArchive as ResponseTimeArchive
from .types import ResponseTimeFlagDirection as ResponseTimeFlagDirection
from .types import ResponseTimeMetric as ResponseTimeMetric
from .types import ResponseTimeMixtureModelArchive as ResponseTimeMixtureModelArchive
from .types import ResponseTimeThresholdSource as ResponseTimeThresholdSource
from .types import ResultArchive as ResultArchive
from .types import ScoreArchive as ScoreArchive
from .types import ScoreArchiveResultType as ScoreArchiveResultType
from .types import ScreenIndexSummary as ScreenIndexSummary
from .types import ScreenResult as ScreenResult
from .u3_poly import midpoint_responding as midpoint_responding
from .u3_poly import response_pattern as response_pattern
from .u3_poly import u3_poly as u3_poly
from .visualize import plot_distributions as plot_distributions
from .visualize import plot_flag_counts as plot_flag_counts
from .visualize import plot_flagged_heatmap as plot_flagged_heatmap

try:
    __version__ = version("insufficient-effort")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "MatrixLike",
    "IndexOptions",
    "__version__",
    "acquiescence",
    "acquiescence_flag",
    "BoolArray",
    "composite",
    "composite_flag",
    "CompositeMethod",
    "composite_probability",
    "composite_scores",
    "composite_summary",
    "CompositeSummary",
    "evenodd",
    "FloatArray",
    "flag_consensus",
    "flag_consensus_archives",
    "FlagConsensusArchive",
    "FlagConsensusResult",
    "guttman",
    "guttman_flag",
    "individual_reliability",
    "individual_reliability_flag",
    "infrequency",
    "infrequency_flag",
    "InfrequencyMissingPolicy",
    "index_catalog",
    "IndexCatalog",
    "IndexErrorMap",
    "IndexFlagMap",
    "IndexMetadata",
    "IndexScoreMap",
    "InspectableArchive",
    "IntArray",
    "irv",
    "longstring",
    "longstring_pattern",
    "longstring_scores",
    "load_archive",
    "load_flag_consensus_archive",
    "load_psychsyn_model",
    "load_response_time_archive",
    "load_response_time_mixture_model",
    "load_score_archive",
    "lz",
    "lz_flag",
    "mad",
    "mad_flag",
    "mahad",
    "mahad_qqplot",
    "mahad_summary",
    "MahadSummary",
    "markov",
    "markov_flag",
    "markov_summary",
    "MarkovSummary",
    "merge_flag_consensus_archives",
    "merge_score_archives",
    "midpoint_responding",
    "missing_rate",
    "missing_rate_flag",
    "onset",
    "onset_flag",
    "person_total",
    "plot_distributions",
    "plot_flag_counts",
    "plot_flagged_heatmap",
    "psychant",
    "fit_psychsyn_model",
    "psychsyn",
    "PsychsynModel",
    "PsychsynModelArchive",
    "psychsyn_model_scores",
    "psychsyn_summary",
    "PsychsynSummary",
    "response_pattern",
    "fit_response_time_mixture",
    "response_time",
    "response_time_consistency",
    "response_time_flag",
    "ResponseTimeFlagDirection",
    "ResponseTimeArchive",
    "ResponseTimeMetric",
    "ResponseTimeMixtureModelArchive",
    "ResponseTimeThresholdSource",
    "ResultArchive",
    "response_time_mixture",
    "ResponseTimeMixtureModel",
    "response_time_mixture_scores",
    "response_time_score_flags",
    "screen",
    "screen_scores",
    "save_flag_consensus_archive",
    "save_psychsyn_model",
    "save_response_time_archive",
    "save_response_time_mixture_model",
    "save_score_archive",
    "ScreenIndexSummary",
    "ScreenResult",
    "ScoreArchive",
    "ScoreArchiveResultType",
    "semantic_ant",
    "semantic_ant_flag",
    "semantic_syn",
    "semantic_syn_flag",
    "u3_poly",
]
