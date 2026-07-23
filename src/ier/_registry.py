"""Central registry for IER index orchestration APIs."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ier.acquiescence import acquiescence
from ier.evenodd import evenodd
from ier.guttman import guttman
from ier.infrequency import infrequency
from ier.irv import irv
from ier.longstring import longstring_pattern, longstring_scores
from ier.lz import lz
from ier.mad import run_mad_index
from ier.mahad import mahad
from ier.markov import markov
from ier.onset import onset
from ier.person_total import person_total
from ier.psychsyn import psychant, psychsyn
from ier.reliability import individual_reliability
from ier.semantic import semantic_ant, semantic_syn
from ier.u3_poly import midpoint_responding, u3_poly

FlagDirection = Literal["high", "low"]
FlagMode = Literal["percentile", "present"]


@dataclass(frozen=True)
class IndexOptions:
    """Shared optional configuration for registered index scorers."""

    na_rm: bool = True
    psychsyn_critval: float = 0.6
    psychant_critval: float = -0.6
    evenodd_factors: list[int] | None = None
    mad_positive_items: list[int] | None = None
    mad_negative_items: list[int] | None = None
    mad_scale_max: int | None = None
    scale_min: float | None = None
    scale_max: float | None = None
    acquiescence_positive_items: list[int] | None = None
    acquiescence_negative_items: list[int] | None = None
    longstring_max_pattern_length: int = 5
    midpoint_tolerance: float = 0.0
    guttman_normalize: bool = True
    onset_window_size: int = 10
    onset_min_items: int = 20
    reliability_n_splits: int = 100
    reliability_random_seed: int | None = None
    semantic_item_pairs: list[tuple[int, int]] | None = None
    infrequency_item_indices: list[int] | None = None
    infrequency_expected_responses: list[float] | None = None
    infrequency_proportion: bool = False


def resolve_index_options(
    options: IndexOptions | None = None,
    *,
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    psychant_critval: float = -0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    acquiescence_positive_items: list[int] | None = None,
    acquiescence_negative_items: list[int] | None = None,
    longstring_max_pattern_length: int = 5,
    midpoint_tolerance: float = 0.0,
    guttman_normalize: bool = True,
    onset_window_size: int = 10,
    onset_min_items: int = 20,
    reliability_n_splits: int = 100,
    reliability_random_seed: int | None = None,
    semantic_item_pairs: list[tuple[int, int]] | None = None,
    infrequency_item_indices: list[int] | None = None,
    infrequency_expected_responses: list[float] | None = None,
    infrequency_proportion: bool = False,
) -> IndexOptions:
    """Return ``options`` if provided; otherwise build from keyword arguments.

    Prefer passing a single ``IndexOptions`` instance to ``screen()`` /
    ``composite()``. Keyword arguments remain supported for backwards
    compatibility and are ignored when ``options`` is set.
    """
    if options is not None:
        return options
    return IndexOptions(
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        psychant_critval=psychant_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
        scale_min=scale_min,
        scale_max=scale_max,
        acquiescence_positive_items=acquiescence_positive_items,
        acquiescence_negative_items=acquiescence_negative_items,
        longstring_max_pattern_length=longstring_max_pattern_length,
        midpoint_tolerance=midpoint_tolerance,
        guttman_normalize=guttman_normalize,
        onset_window_size=onset_window_size,
        onset_min_items=onset_min_items,
        reliability_n_splits=reliability_n_splits,
        reliability_random_seed=reliability_random_seed,
        semantic_item_pairs=semantic_item_pairs,
        infrequency_item_indices=infrequency_item_indices,
        infrequency_expected_responses=infrequency_expected_responses,
        infrequency_proportion=infrequency_proportion,
    )


# Backwards-compatible alias
def build_index_options(
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    *,
    scale_min: float | None = None,
    scale_max: float | None = None,
    psychant_critval: float = -0.6,
    acquiescence_positive_items: list[int] | None = None,
    acquiescence_negative_items: list[int] | None = None,
    longstring_max_pattern_length: int = 5,
    midpoint_tolerance: float = 0.0,
    guttman_normalize: bool = True,
    onset_window_size: int = 10,
    onset_min_items: int = 20,
    reliability_n_splits: int = 100,
    reliability_random_seed: int | None = None,
    semantic_item_pairs: list[tuple[int, int]] | None = None,
    infrequency_item_indices: list[int] | None = None,
    infrequency_expected_responses: list[float] | None = None,
    infrequency_proportion: bool = False,
) -> IndexOptions:
    """Build shared option state for registered index scorers.

    .. deprecated:: 1.8.0
        Prefer ``IndexOptions(...)`` or ``resolve_index_options(...)``.
    """
    return resolve_index_options(
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        psychant_critval=psychant_critval,
        evenodd_factors=evenodd_factors,
        mad_positive_items=mad_positive_items,
        mad_negative_items=mad_negative_items,
        mad_scale_max=mad_scale_max,
        scale_min=scale_min,
        scale_max=scale_max,
        acquiescence_positive_items=acquiescence_positive_items,
        acquiescence_negative_items=acquiescence_negative_items,
        longstring_max_pattern_length=longstring_max_pattern_length,
        midpoint_tolerance=midpoint_tolerance,
        guttman_normalize=guttman_normalize,
        onset_window_size=onset_window_size,
        onset_min_items=onset_min_items,
        reliability_n_splits=reliability_n_splits,
        reliability_random_seed=reliability_random_seed,
        semantic_item_pairs=semantic_item_pairs,
        infrequency_item_indices=infrequency_item_indices,
        infrequency_expected_responses=infrequency_expected_responses,
        infrequency_proportion=infrequency_proportion,
    )


@dataclass(frozen=True)
class IndexSpec:
    """Metadata and scorer for an IER index used by orchestration APIs."""

    name: str
    scorer: Callable[[np.ndarray, IndexOptions], np.ndarray]
    flag_direction: FlagDirection
    composite_multiplier: float = 1.0
    default_screen: bool = False
    default_composite: bool = False
    composite_enabled: bool = True
    flag_mode: FlagMode = "percentile"
    required_error: Callable[[IndexOptions], str | None] | None = None


def _require_evenodd_factors(options: IndexOptions) -> str | None:
    if options.evenodd_factors is None:
        return "evenodd_factors must be provided when using evenodd index"
    return None


def _require_mad_items(options: IndexOptions) -> str | None:
    if options.mad_positive_items is None or options.mad_negative_items is None:
        return "mad_positive_items and mad_negative_items must be provided when using mad index"
    return None


def _require_semantic_pairs(options: IndexOptions) -> str | None:
    if options.semantic_item_pairs is None:
        return "semantic_item_pairs must be provided when using semantic_syn or semantic_ant"
    return None


def _require_infrequency_config(options: IndexOptions) -> str | None:
    if options.infrequency_item_indices is None or options.infrequency_expected_responses is None:
        return (
            "infrequency_item_indices and infrequency_expected_responses "
            "must be provided when using infrequency"
        )
    return None


def _mahad_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    # Distances are NumPy-only; screen/composite apply their own flagging.
    result = mahad(x, na_rm=options.na_rm, method="iqr")
    if not isinstance(result, np.ndarray):
        raise ValueError("mahad returned non-array output")
    return result


def _psychsyn_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = psychsyn(x, critval=options.psychsyn_critval, resample_na=options.na_rm)
    if not isinstance(result, np.ndarray):
        raise ValueError("psychsyn returned non-array output")
    return result


def _psychant_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = psychant(x, critval=options.psychant_critval, resample_na=options.na_rm)
    if not isinstance(result, np.ndarray):
        raise ValueError("psychant returned non-array output")
    return result


def _evenodd_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    if options.evenodd_factors is None:
        raise ValueError("evenodd_factors must be provided when using evenodd index")
    result = evenodd(x, factors=options.evenodd_factors)
    if not isinstance(result, np.ndarray):
        raise ValueError("evenodd returned non-array output")
    return result


def _mad_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return run_mad_index(
        x,
        positive_items=options.mad_positive_items,
        negative_items=options.mad_negative_items,
        scale_max=options.mad_scale_max,
        na_rm=options.na_rm,
    )


def _acquiescence_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return acquiescence(
        x,
        scale_min=options.scale_min,
        scale_max=options.scale_max,
        positive_items=options.acquiescence_positive_items,
        negative_items=options.acquiescence_negative_items,
        na_rm=options.na_rm,
    )


def _guttman_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return guttman(x, na_rm=options.na_rm, normalize=options.guttman_normalize)


def _onset_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return onset(
        x,
        window_size=options.onset_window_size,
        min_items=options.onset_min_items,
        na_rm=options.na_rm,
    )


def _reliability_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return individual_reliability(
        x,
        n_splits=options.reliability_n_splits,
        random_seed=options.reliability_random_seed,
    )


def _semantic_syn_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    if options.semantic_item_pairs is None:
        raise ValueError("semantic_item_pairs must be provided when using semantic_syn")
    return semantic_syn(x, item_pairs=options.semantic_item_pairs, anto=False)


def _semantic_ant_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    if options.semantic_item_pairs is None:
        raise ValueError("semantic_item_pairs must be provided when using semantic_ant")
    return semantic_ant(x, item_pairs=options.semantic_item_pairs)


def _infrequency_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    if options.infrequency_item_indices is None or options.infrequency_expected_responses is None:
        raise ValueError(
            "infrequency_item_indices and infrequency_expected_responses "
            "must be provided when using infrequency"
        )
    return infrequency(
        x,
        item_indices=options.infrequency_item_indices,
        expected_responses=options.infrequency_expected_responses,
        proportion=options.infrequency_proportion,
    )


INDEX_REGISTRY: dict[str, IndexSpec] = {
    "irv": IndexSpec(
        name="irv",
        scorer=lambda x, options: irv(x, na_rm=options.na_rm),
        flag_direction="low",
        composite_multiplier=-1.0,
        default_screen=True,
        default_composite=True,
    ),
    "longstring": IndexSpec(
        name="longstring",
        scorer=lambda x, options: longstring_scores(x, na_rm=options.na_rm),
        flag_direction="high",
        default_screen=True,
        default_composite=True,
    ),
    "longstring_pattern": IndexSpec(
        name="longstring_pattern",
        scorer=lambda x, options: longstring_pattern(
            x,
            max_pattern_length=options.longstring_max_pattern_length,
            na_rm=options.na_rm,
        ),
        flag_direction="high",
        default_screen=True,
    ),
    "mahad": IndexSpec(
        name="mahad",
        scorer=_mahad_scores,
        flag_direction="high",
        default_screen=True,
        default_composite=True,
    ),
    "psychsyn": IndexSpec(
        name="psychsyn",
        scorer=_psychsyn_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
        default_screen=True,
        default_composite=True,
    ),
    "psychant": IndexSpec(
        name="psychant",
        scorer=_psychant_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
    ),
    "person_total": IndexSpec(
        name="person_total",
        scorer=lambda x, options: person_total(x, na_rm=options.na_rm),
        flag_direction="low",
        composite_multiplier=-1.0,
        default_screen=True,
        default_composite=True,
    ),
    "markov": IndexSpec(
        name="markov",
        scorer=lambda x, options: markov(x, na_rm=options.na_rm),
        flag_direction="low",
        composite_multiplier=-1.0,
        default_screen=True,
    ),
    "u3_poly": IndexSpec(
        name="u3_poly",
        scorer=lambda x, options: u3_poly(
            x, scale_min=options.scale_min, scale_max=options.scale_max
        ),
        flag_direction="high",
        default_screen=True,
        composite_enabled=False,
    ),
    "midpoint": IndexSpec(
        name="midpoint",
        scorer=lambda x, options: midpoint_responding(
            x,
            scale_min=options.scale_min,
            scale_max=options.scale_max,
            tolerance=options.midpoint_tolerance,
        ),
        flag_direction="high",
        default_screen=True,
        composite_enabled=False,
    ),
    "acquiescence": IndexSpec(
        name="acquiescence",
        scorer=_acquiescence_scores,
        flag_direction="high",
        default_screen=True,
        composite_enabled=False,
    ),
    "guttman": IndexSpec(
        name="guttman",
        scorer=_guttman_scores,
        flag_direction="high",
        default_screen=True,
        # Available in composite(), but not default: Guttman errors can dilute
        # pattern-based signals like straightlining in small samples.
        default_composite=False,
    ),
    "individual_reliability": IndexSpec(
        name="individual_reliability",
        scorer=_reliability_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
    ),
    "onset": IndexSpec(
        name="onset",
        scorer=_onset_scores,
        flag_direction="high",
        flag_mode="present",
        composite_enabled=False,
    ),
    "evenodd": IndexSpec(
        name="evenodd",
        scorer=_evenodd_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
        required_error=_require_evenodd_factors,
    ),
    "mad": IndexSpec(
        name="mad",
        scorer=_mad_scores,
        flag_direction="high",
        required_error=_require_mad_items,
    ),
    "lz": IndexSpec(
        name="lz",
        scorer=lambda x, options: lz(x, na_rm=options.na_rm),
        flag_direction="low",
        composite_multiplier=-1.0,
    ),
    "semantic_syn": IndexSpec(
        name="semantic_syn",
        scorer=_semantic_syn_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
        required_error=_require_semantic_pairs,
    ),
    "semantic_ant": IndexSpec(
        name="semantic_ant",
        scorer=_semantic_ant_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
        required_error=_require_semantic_pairs,
    ),
    "infrequency": IndexSpec(
        name="infrequency",
        scorer=_infrequency_scores,
        flag_direction="high",
        required_error=_require_infrequency_config,
    ),
}


def default_screen_indices() -> list[str]:
    """Return default index names for screen()."""
    return [name for name, spec in INDEX_REGISTRY.items() if spec.default_screen]


def default_composite_indices() -> list[str]:
    """Return default index names for composite()."""
    return [name for name, spec in INDEX_REGISTRY.items() if spec.default_composite]


def composite_index_names() -> set[str]:
    """Return index names allowed in composite APIs."""
    return {name for name, spec in INDEX_REGISTRY.items() if spec.composite_enabled}


def validate_index_names(indices: list[str], allowed: set[str] | None = None) -> None:
    """Validate requested index names against the registry or a registry subset."""
    valid = set(INDEX_REGISTRY) if allowed is None else allowed
    for name in indices:
        if name not in valid:
            raise ValueError(f"invalid index '{name}'. Valid options: {sorted(valid)}")


def score_registered_indices(
    x: np.ndarray,
    indices: list[str],
    options: IndexOptions,
    *,
    apply_composite_direction: bool = False,
    raise_missing_config: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute registered indices and collect per-index validation failures."""
    scores: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    for name in indices:
        spec = INDEX_REGISTRY[name]
        required_error = spec.required_error(options) if spec.required_error is not None else None
        if required_error is not None:
            if raise_missing_config:
                raise ValueError(required_error)
            errors[name] = required_error
            continue

        try:
            score = spec.scorer(x, options)
        except (ValueError, RuntimeError, TypeError) as err:
            errors[name] = str(err)
            continue

        if apply_composite_direction:
            score = spec.composite_multiplier * score
        scores[name] = score

    return scores, errors
