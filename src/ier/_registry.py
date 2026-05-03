"""Central registry for IER index orchestration APIs."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ier.acquiescence import acquiescence
from ier.evenodd import evenodd
from ier.irv import irv
from ier.longstring import longstring_pattern, longstring_scores
from ier.lz import lz
from ier.mad import run_mad_index
from ier.mahad import mahad
from ier.markov import markov
from ier.person_total import person_total
from ier.psychsyn import psychsyn
from ier.u3_poly import midpoint_responding, u3_poly

FlagDirection = Literal["high", "low"]


@dataclass(frozen=True)
class IndexOptions:
    """Shared optional configuration for registered index scorers."""

    na_rm: bool = True
    psychsyn_critval: float = 0.6
    evenodd_factors: list[int] | None = None
    mad_positive_items: list[int] | None = None
    mad_negative_items: list[int] | None = None
    mad_scale_max: int | None = None
    scale_min: float | None = None
    scale_max: float | None = None


def build_index_options(
    na_rm: bool,
    psychsyn_critval: float,
    evenodd_factors: list[int] | None,
    mad_positive_items: list[int] | None,
    mad_negative_items: list[int] | None,
    mad_scale_max: int | None,
    *,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> IndexOptions:
    """Build shared option state for registered index scorers."""
    return IndexOptions(
        na_rm,
        psychsyn_critval,
        evenodd_factors,
        mad_positive_items,
        mad_negative_items,
        mad_scale_max,
        scale_min,
        scale_max,
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
    required_error: Callable[[IndexOptions], str | None] | None = None


def _require_evenodd_factors(options: IndexOptions) -> str | None:
    if options.evenodd_factors is None:
        return "evenodd_factors must be provided when using evenodd index"
    return None


def _require_mad_items(options: IndexOptions) -> str | None:
    if options.mad_positive_items is None or options.mad_negative_items is None:
        return "mad_positive_items and mad_negative_items must be provided when using mad index"
    return None


def _mahad_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    result = mahad(x, na_rm=options.na_rm)
    if not isinstance(result, np.ndarray):
        raise ValueError("mahad returned non-array output")
    return result


def _psychsyn_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = psychsyn(x, critval=options.psychsyn_critval, resample_na=options.na_rm)
    if not isinstance(result, np.ndarray):
        raise ValueError("psychsyn returned non-array output")
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
        scorer=lambda x, options: longstring_pattern(x, na_rm=options.na_rm),
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
            x, scale_min=options.scale_min, scale_max=options.scale_max
        ),
        flag_direction="high",
        default_screen=True,
        composite_enabled=False,
    ),
    "acquiescence": IndexSpec(
        name="acquiescence",
        scorer=lambda x, options: acquiescence(
            x, scale_min=options.scale_min, scale_max=options.scale_max, na_rm=options.na_rm
        ),
        flag_direction="high",
        default_screen=True,
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
            raise ValueError(f"invalid index '{name}'. Valid options: {valid}")


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
        except ValueError as err:
            errors[name] = str(err)
            continue

        if apply_composite_direction:
            score = spec.composite_multiplier * score
        scores[name] = score

    return scores, errors
