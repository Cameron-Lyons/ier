"""IER: Python library for detecting Insufficient Effort Responding in survey data."""

import importlib
from typing import TYPE_CHECKING

from ._registry import IndexOptions as IndexOptions
from ._registry import index_catalog as index_catalog
from ._validation import MatrixLike as MatrixLike
from .acquiescence import acquiescence as acquiescence
from .acquiescence import acquiescence_flag as acquiescence_flag
from .autocorrelation import autocorrelation as autocorrelation
from .autocorrelation import autocorrelation_flag as autocorrelation_flag
from .composite import composite as composite
from .composite import composite_flag as composite_flag
from .composite import composite_probability as composite_probability
from .composite import composite_scores as composite_scores
from .composite import composite_scores_summary as composite_scores_summary
from .composite import composite_summary as composite_summary
from .evenodd import evenodd as evenodd
from .guttman import guttman as guttman
from .guttman import guttman_flag as guttman_flag
from .infrequency import infrequency as infrequency
from .infrequency import infrequency_flag as infrequency_flag
from .irv import irv as irv
from .keying import reverse_score as reverse_score
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
from .person_fit import gpoly as gpoly
from .person_fit import gpoly_flag as gpoly_flag
from .person_fit import ht as ht
from .person_fit import ht_flag as ht_flag
from .person_fit import u3poly as u3poly
from .person_fit import u3poly_flag as u3poly_flag
from .person_total import person_total as person_total
from .person_total import person_total_flag as person_total_flag
from .psychsyn import psychant as psychant
from .psychsyn import psychant_flag as psychant_flag
from .psychsyn import psychsyn as psychsyn
from .psychsyn import psychsyn_critval as psychsyn_critval
from .psychsyn import psychsyn_flag as psychsyn_flag
from .psychsyn import psychsyn_summary as psychsyn_summary
from .reliability import individual_reliability as individual_reliability
from .reliability import individual_reliability_flag as individual_reliability_flag
from .response_time import response_time as response_time
from .response_time import response_time_consistency as response_time_consistency
from .response_time import response_time_effort as response_time_effort
from .response_time import response_time_effort_flag as response_time_effort_flag
from .response_time import response_time_flag as response_time_flag
from .response_time import response_time_mixture as response_time_mixture
from .response_time import response_time_score_flags as response_time_score_flags
from .screen import screen as screen
from .screen import screen_scores as screen_scores
from .semantic import semantic_ant as semantic_ant
from .semantic import semantic_ant_flag as semantic_ant_flag
from .semantic import semantic_syn as semantic_syn
from .semantic import semantic_syn_flag as semantic_syn_flag
from .tables import composite_table as composite_table
from .tables import index_agreement as index_agreement
from .tables import screen_table as screen_table
from .types import AgreementKind as AgreementKind
from .types import BoolArray as BoolArray
from .types import CombineMethod as CombineMethod
from .types import CompositeMethod as CompositeMethod
from .types import CompositeSummary as CompositeSummary
from .types import EvenOddMethod as EvenOddMethod
from .types import FlagDirection as FlagDirection
from .types import FlagMode as FlagMode
from .types import FloatArray as FloatArray
from .types import IndexCatalog as IndexCatalog
from .types import IndexErrorMap as IndexErrorMap
from .types import IndexFlagMap as IndexFlagMap
from .types import IndexMetadata as IndexMetadata
from .types import IndexPercentileMap as IndexPercentileMap
from .types import IndexScoreMap as IndexScoreMap
from .types import IndexThresholdMap as IndexThresholdMap
from .types import IndexThresholdSource as IndexThresholdSource
from .types import IndexThresholdSourceMap as IndexThresholdSourceMap
from .types import InfrequencyMissingPolicy as InfrequencyMissingPolicy
from .types import IntArray as IntArray
from .types import ItemCorrelationMode as ItemCorrelationMode
from .types import ResponseTimeArchive as ResponseTimeArchive
from .types import ResponseTimeFlagDirection as ResponseTimeFlagDirection
from .types import ResponseTimeMetric as ResponseTimeMetric
from .types import ScoreArchive as ScoreArchive
from .types import ScoreArchiveResultType as ScoreArchiveResultType
from .types import ScreenArchive as ScreenArchive
from .types import ScreenIndexSummary as ScreenIndexSummary
from .types import ScreenResult as ScreenResult
from .u3_poly import midpoint_responding as midpoint_responding
from .u3_poly import midpoint_responding_flag as midpoint_responding_flag
from .u3_poly import response_pattern as response_pattern
from .u3_poly import u3_poly as u3_poly
from .u3_poly import u3_poly_flag as u3_poly_flag
from .visualize import plot_composite as plot_composite
from .visualize import plot_distributions as plot_distributions
from .visualize import plot_flag_counts as plot_flag_counts
from .visualize import plot_flagged_heatmap as plot_flagged_heatmap
from .visualize import plot_index_agreement as plot_index_agreement

if TYPE_CHECKING:
    from .archive import load_response_time_archive as load_response_time_archive
    from .archive import load_score_archive as load_score_archive
    from .archive import load_screen_archive as load_screen_archive
    from .archive import save_response_time_archive as save_response_time_archive
    from .archive import save_score_archive as save_score_archive
    from .archive import save_screen_archive as save_screen_archive

    __version__: str
    """Installed distribution version, read from package metadata on first access."""

__all__ = [
    "MatrixLike",
    "IndexOptions",
    "__version__",
    "acquiescence",
    "acquiescence_flag",
    "AgreementKind",
    "autocorrelation",
    "autocorrelation_flag",
    "BoolArray",
    "CombineMethod",
    "composite",
    "composite_flag",
    "CompositeMethod",
    "composite_probability",
    "composite_scores",
    "composite_scores_summary",
    "composite_summary",
    "CompositeSummary",
    "composite_table",
    "evenodd",
    "EvenOddMethod",
    "FlagDirection",
    "FlagMode",
    "FloatArray",
    "gpoly",
    "gpoly_flag",
    "guttman",
    "guttman_flag",
    "ht",
    "ht_flag",
    "individual_reliability",
    "individual_reliability_flag",
    "infrequency",
    "infrequency_flag",
    "InfrequencyMissingPolicy",
    "index_agreement",
    "index_catalog",
    "IndexCatalog",
    "IndexErrorMap",
    "IndexFlagMap",
    "IndexMetadata",
    "IndexPercentileMap",
    "IndexScoreMap",
    "IndexThresholdMap",
    "IndexThresholdSource",
    "IndexThresholdSourceMap",
    "IntArray",
    "ItemCorrelationMode",
    "irv",
    "longstring",
    "longstring_pattern",
    "longstring_scores",
    "load_score_archive",
    "load_response_time_archive",
    "load_screen_archive",
    "lz",
    "lz_flag",
    "mad",
    "mad_flag",
    "mahad",
    "mahad_qqplot",
    "mahad_summary",
    "markov",
    "markov_flag",
    "markov_summary",
    "midpoint_responding",
    "midpoint_responding_flag",
    "missing_rate",
    "missing_rate_flag",
    "onset",
    "onset_flag",
    "person_total",
    "person_total_flag",
    "plot_composite",
    "plot_distributions",
    "plot_flag_counts",
    "plot_flagged_heatmap",
    "plot_index_agreement",
    "psychant",
    "psychant_flag",
    "psychsyn",
    "psychsyn_critval",
    "psychsyn_flag",
    "psychsyn_summary",
    "response_pattern",
    "response_time",
    "response_time_consistency",
    "response_time_effort",
    "response_time_effort_flag",
    "response_time_flag",
    "ResponseTimeFlagDirection",
    "ResponseTimeArchive",
    "ResponseTimeMetric",
    "response_time_mixture",
    "response_time_score_flags",
    "reverse_score",
    "screen",
    "screen_scores",
    "screen_table",
    "save_score_archive",
    "save_response_time_archive",
    "save_screen_archive",
    "ScreenArchive",
    "ScreenIndexSummary",
    "ScreenResult",
    "ScoreArchive",
    "ScoreArchiveResultType",
    "semantic_ant",
    "semantic_ant_flag",
    "semantic_syn",
    "semantic_syn_flag",
    "u3_poly",
    "u3_poly_flag",
    "u3poly",
    "u3poly_flag",
]

# Archive I/O (zipfile and compression codecs) and distribution metadata load on
# first use rather than with every `import ier`.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "load_response_time_archive": ("ier.archive", "load_response_time_archive"),
    "load_score_archive": ("ier.archive", "load_score_archive"),
    "load_screen_archive": ("ier.archive", "load_screen_archive"),
    "save_response_time_archive": ("ier.archive", "save_response_time_archive"),
    "save_score_archive": ("ier.archive", "save_score_archive"),
    "save_screen_archive": ("ier.archive", "save_screen_archive"),
}
_LAZY_SUBMODULES = frozenset({"archive"})


def _lazy_attribute(name: str) -> object:
    """Resolve and cache one deferred public attribute (PEP 562)."""
    value: object
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            value = version("insufficient-effort")
        except PackageNotFoundError:
            value = "0.0.0"
    elif name in _LAZY_ATTRS:
        module_name, attribute = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attribute)
    elif name in _LAZY_SUBMODULES:
        # Eager archive exports previously bound `ier.archive` as a side effect.
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List loaded and deferred public names, including lazily imported submodules."""
    return sorted(set(globals()) | set(__all__) | _LAZY_SUBMODULES)


if not TYPE_CHECKING:
    # Type checkers see the declarations above instead, so misspelled names
    # remain errors rather than resolving through a module __getattr__.
    __getattr__ = _lazy_attribute
