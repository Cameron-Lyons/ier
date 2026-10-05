"""Central registry for IER index orchestration APIs."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ier._validation import validate_integer
from ier.acquiescence import acquiescence
from ier.autocorrelation import autocorrelation
from ier.evenodd import evenodd
from ier.guttman import guttman
from ier.infrequency import infrequency
from ier.irv import irv
from ier.keying import reverse_score
from ier.longstring import longstring_pattern, longstring_scores
from ier.lz import lz
from ier.mad import mad
from ier.mahad import mahad
from ier.markov import markov
from ier.missing import missing_rate
from ier.onset import onset
from ier.person_fit import _category_bounds, gpoly, ht, u3poly
from ier.person_total import person_total
from ier.psychsyn import psychant, psychsyn
from ier.reliability import individual_reliability
from ier.semantic import semantic_ant, semantic_syn
from ier.types import (
    EvenOddMethod,
    FlagDirection,
    FlagMode,
    IndexCatalog,
    InfrequencyMissingPolicy,
    ItemCorrelationMode,
)
from ier.u3_poly import midpoint_responding, u3_poly


@dataclass(frozen=True)
class IndexOptions:
    """Shared optional configuration for registered index scorers.

    ``reverse_keyed_items`` lists 0-based reverse-worded columns. They are
    reverse-scored with ``scale_min`` and ``scale_max``, inferred from the data
    when omitted, only for indices whose catalog entry sets
    ``uses_keyed_responses``; every other index reads the responses as presented.
    When ``person_fit_ncat`` is set, ``gpoly`` and ``u3poly_fit`` reflect items
    on the declared category scale instead.
    """

    na_rm: bool = True
    psychsyn_critval: float = 0.6
    psychant_critval: float = -0.6
    evenodd_factors: list[int] | None = None
    mad_positive_items: list[int] | None = None
    mad_negative_items: list[int] | None = None
    mad_scale_max: float | None = None
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
    infrequency_missing: InfrequencyMissingPolicy = "pass"
    mad_scale_min: float | None = None
    missing_item_indices: list[int] | None = None
    missing_applicable_mask: ArrayLike | None = None
    infrequency_acceptable_ranges: list[tuple[float, float]] | None = None
    irv_num_split: int = 1
    irv_split_points: list[int] | None = None
    psychsyn_item_correlations: ItemCorrelationMode = "complete"
    evenodd_method: EvenOddMethod = "item_pairs"
    reliability_factors: list[int] | None = None
    autocorrelation_max_lag: int | None = 10
    autocorrelation_statistic: Literal["max_abs", "sum_abs"] = "max_abs"
    reverse_keyed_items: list[int] | None = None
    person_fit_ncat: int | None = None


def resolve_index_options(options: IndexOptions | None = None) -> IndexOptions:
    """Return ``options`` if provided; otherwise default ``IndexOptions()``."""
    return options if options is not None else IndexOptions()


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
    required_options: tuple[str, ...] = ()
    # At least one option in each group must be set, e.g. interchangeable answer forms.
    alternative_options: tuple[tuple[str, ...], ...] = ()
    # Scores responses with IndexOptions.reverse_keyed_items reverse-scored.
    keyed_input: bool = False


def _missing_required_options(spec: IndexSpec, options: IndexOptions) -> str | None:
    """Derive the configuration error from the options an index advertises."""

    def is_set(option: str) -> bool:
        return getattr(options, option) is not None

    if all(map(is_set, spec.required_options)) and all(
        any(map(is_set, group)) for group in spec.alternative_options
    ):
        return None
    requirements = [
        *spec.required_options,
        *(
            "either " + " or ".join(group) if len(group) > 1 else group[0]
            for group in spec.alternative_options
        ),
    ]
    return " and ".join(requirements) + f" must be provided when using {spec.name} index"


def _irv_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    # Validate before choosing the mode: True and 1.0 equal 1 but are not section
    # counts, so they must fail rather than silently select the unsplit computation.
    num_split = validate_integer(
        options.irv_num_split, message="num_split must be a positive integer", minimum=1
    )
    split = num_split != 1 or options.irv_split_points is not None
    return irv(
        x,
        na_rm=options.na_rm,
        split=split,
        num_split=num_split,
        split_points=options.irv_split_points,
    )


def _psychsyn_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return psychsyn(
        x,
        critval=options.psychsyn_critval,
        item_correlations=options.psychsyn_item_correlations,
    )


def _psychant_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return psychant(
        x,
        critval=options.psychant_critval,
        item_correlations=options.psychsyn_item_correlations,
    )


def _evenodd_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    assert options.evenodd_factors is not None
    return evenodd(x, factors=options.evenodd_factors, method=options.evenodd_method)


def _mad_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return mad(
        x,
        positive_items=options.mad_positive_items,
        negative_items=options.mad_negative_items,
        scale_min=options.mad_scale_min,
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
        factors=options.reliability_factors,
    )


def _semantic_syn_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    assert options.semantic_item_pairs is not None
    return semantic_syn(x, item_pairs=options.semantic_item_pairs, anto=False)


def _semantic_ant_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    assert options.semantic_item_pairs is not None
    return semantic_ant(
        x,
        item_pairs=options.semantic_item_pairs,
        scale_min=options.scale_min,
        scale_max=options.scale_max,
    )


def _infrequency_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    assert options.infrequency_item_indices is not None
    return infrequency(
        x,
        item_indices=options.infrequency_item_indices,
        expected_responses=options.infrequency_expected_responses,
        proportion=options.infrequency_proportion,
        missing=options.infrequency_missing,
        acceptable_ranges=options.infrequency_acceptable_ranges,
    )


def _missing_rate_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return missing_rate(
        x,
        item_indices=options.missing_item_indices,
        applicable_mask=options.missing_applicable_mask,
    )


def _autocorrelation_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return autocorrelation(
        x,
        max_lag=options.autocorrelation_max_lag,
        statistic=options.autocorrelation_statistic,
        na_rm=options.na_rm,
    )


def _gpoly_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return gpoly(
        x,
        ncat=options.person_fit_ncat,
        scale_min=options.scale_min,
        scale_max=options.scale_max,
        na_rm=options.na_rm,
    )


def _u3poly_scores(x: np.ndarray, options: IndexOptions) -> np.ndarray:
    return u3poly(
        x,
        ncat=options.person_fit_ncat,
        scale_min=options.scale_min,
        scale_max=options.scale_max,
        na_rm=options.na_rm,
    )


INDEX_REGISTRY: dict[str, IndexSpec] = {
    "irv": IndexSpec(
        name="irv",
        scorer=_irv_scores,
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
        # Reverse scoring is affine per item, so distances need no keyed copy.
        scorer=lambda x, options: mahad(x, na_rm=options.na_rm, method="iqr"),
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
        # Attentive respondents answer antonym pairs in opposite directions,
        # so near-zero or positive correlations are the suspicious tail.
        flag_direction="high",
    ),
    "person_total": IndexSpec(
        name="person_total",
        # The reference item profile comes from the presented items' own means.
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
    "missing_rate": IndexSpec(
        name="missing_rate",
        scorer=_missing_rate_scores,
        flag_direction="high",
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
        scorer=lambda x, options: guttman(
            x, na_rm=options.na_rm, normalize=options.guttman_normalize
        ),
        flag_direction="high",
        default_screen=True,
        default_composite=False,
        # A cumulative pattern requires every item to order respondents one way.
        keyed_input=True,
    ),
    "individual_reliability": IndexSpec(
        name="individual_reliability",
        scorer=_reliability_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
        keyed_input=True,
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
        required_options=("evenodd_factors",),
        keyed_input=True,
    ),
    "mad": IndexSpec(
        name="mad",
        scorer=_mad_scores,
        flag_direction="high",
        required_options=("mad_positive_items", "mad_negative_items"),
    ),
    "lz": IndexSpec(
        name="lz",
        scorer=lambda x, options: lz(x, na_rm=options.na_rm),
        flag_direction="low",
        composite_multiplier=-1.0,
        # Item response models assume responses increase with one latent trait.
        keyed_input=True,
    ),
    "semantic_syn": IndexSpec(
        name="semantic_syn",
        scorer=_semantic_syn_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
        required_options=("semantic_item_pairs",),
    ),
    "semantic_ant": IndexSpec(
        name="semantic_ant",
        scorer=_semantic_ant_scores,
        flag_direction="low",
        composite_multiplier=-1.0,
        required_options=("semantic_item_pairs",),
    ),
    "infrequency": IndexSpec(
        name="infrequency",
        scorer=_infrequency_scores,
        flag_direction="high",
        required_options=("infrequency_item_indices",),
        # Acceptable ranges replace expected responses as the answer key.
        alternative_options=(("infrequency_expected_responses", "infrequency_acceptable_ranges"),),
    ),
    "avgstr": IndexSpec(
        name="avgstr",
        scorer=lambda x, options: longstring_scores(x, na_rm=options.na_rm, avg=True),
        flag_direction="high",
    ),
    "autocorrelation": IndexSpec(
        name="autocorrelation",
        scorer=_autocorrelation_scores,
        flag_direction="high",
    ),
    "gpoly": IndexSpec(
        name="gpoly",
        scorer=_gpoly_scores,
        flag_direction="high",
        keyed_input=True,
    ),
    "u3poly_fit": IndexSpec(
        name="u3poly_fit",
        scorer=_u3poly_scores,
        flag_direction="high",
        keyed_input=True,
    ),
    "ht": IndexSpec(
        name="ht",
        scorer=lambda x, options: ht(x, na_rm=options.na_rm),
        flag_direction="low",
        composite_multiplier=-1.0,
        keyed_input=True,
    ),
}


def index_catalog() -> IndexCatalog:
    """Return discoverable metadata for all registered orchestration indices."""
    return {
        name: {
            "flag_direction": spec.flag_direction,
            "flag_mode": spec.flag_mode,
            "default_screen": spec.default_screen,
            "default_composite": spec.default_composite,
            "composite_enabled": spec.composite_enabled,
            "required_options": spec.required_options,
            "alternative_options": spec.alternative_options,
            "uses_keyed_responses": spec.keyed_input,
        }
        for name, spec in INDEX_REGISTRY.items()
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


def validate_index_names(indices: Sequence[str], allowed: set[str] | None = None) -> None:
    """Validate requested index names against the registry or a registry subset."""
    if isinstance(indices, str):
        raise ValueError("indices must be a list of index names, not a string")
    valid = set(INDEX_REGISTRY) if allowed is None else allowed
    seen: set[str] = set()
    for name in indices:
        if name not in valid:
            raise ValueError(f"invalid index '{name}'. Valid options: {sorted(valid)}")
        if name in seen:
            raise ValueError(f"duplicate index '{name}' is not supported")
        seen.add(name)


def validate_index_errors(
    errors: Mapping[str, str] | None,
    score_names: list[str],
    allowed: set[str] | None = None,
) -> dict[str, str]:
    """Copy reusable failure provenance, keeping failed indices unavailable."""
    if errors is None:
        return {}
    if not isinstance(errors, Mapping):
        raise TypeError("errors must be a mapping of registered index names to messages")
    names = list(errors)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("error index names must be nonblank strings")
    validate_index_names(names, allowed)
    if set(score_names).intersection(names):
        raise ValueError("indices cannot contain both scores and errors")
    messages = list(errors.values())
    if any(not isinstance(message, str) for message in messages):
        raise ValueError("error messages must be strings")
    if any(not message.strip() for message in messages):
        raise ValueError("error messages must be nonblank")
    return dict(zip(names, messages, strict=True))


def resolve_index_overrides(
    values: Mapping[str, object] | None,
    indices: Sequence[str],
    *,
    label: str,
    convert: Callable[[str, object], float],
    accepts: Callable[[str], str | None] | None = None,
) -> dict[str, float]:
    """Validate ordered per-index overrides; ``accepts`` may return a rejection message.

    Besides mappings, objects with a mapping-style ``items()`` method, such as a
    pandas Series indexed by index name, are accepted.
    """
    if values is None:
        return {}
    items = getattr(values, "items", None)
    if not callable(items):
        raise TypeError(f"{label}s must be a mapping of registered index names to numbers")
    selected = set(indices)
    resolved: dict[str, float] = {}
    for name, value in items():
        if not isinstance(name, str) or name not in INDEX_REGISTRY:
            raise ValueError(f"unknown {label} index: {name}")
        if name in resolved:  # Mapping-like inputs such as a Series may repeat labels.
            raise ValueError(f"duplicate {label} index: {name}")
        if name not in selected:
            raise ValueError(f"{label} index is not selected: {name}")
        rejection = accepts(name) if accepts is not None else None
        if rejection is not None:
            raise ValueError(rejection)
        resolved[name] = convert(name, value)
    return resolved


def numeric_override(
    label: str,
    requirement: str = "a finite number",
    valid: Callable[[float], bool] = math.isfinite,
) -> Callable[[str, object], float]:
    """Build an override converter for non-Boolean numbers that satisfy ``valid``."""

    def convert(name: str, value: object) -> float:
        message = f"{label} for {name} must be {requirement}"
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(message) from error
        if not valid(number):
            raise ValueError(message)
        return number

    return convert


IndexScoreResult = tuple[np.ndarray | None, str | None, Exception | None]


def validate_worker_count(workers: int) -> int:
    """Return a validated positive number of index-scoring workers."""
    return validate_integer(workers, message="workers must be a positive integer", minimum=1)


def validate_min_valid_indices(
    min_valid_indices: int | None,
    n_selected_indices: int,
) -> int | None:
    """Validate an optional respondent-level index completeness requirement."""
    if min_valid_indices is None:
        return None
    min_valid_indices = validate_integer(
        min_valid_indices,
        message="min_valid_indices must be a positive integer or None",
        minimum=1,
    )
    if min_valid_indices > n_selected_indices:
        raise ValueError(
            f"min_valid_indices cannot exceed the number of selected indices ({n_selected_indices})"
        )
    return min_valid_indices


# The matrix one index reads, with any failure preparing it.
_IndexInput = tuple[np.ndarray, IndexScoreResult | None]
# Keyed scorers whose response scale IndexOptions.person_fit_ncat declares.
_PERSON_FIT_SCALE_INDICES = frozenset({"gpoly", "u3poly_fit"})


def _keyed_responses(
    x: np.ndarray, indices: Sequence[str], options: IndexOptions
) -> list[_IndexInput]:
    """Return the matrix each selected index reads and any failure preparing it.

    Keyed-input indices read ``x`` with ``options.reverse_keyed_items``
    reverse-scored between ``scale_min`` and ``scale_max``, inferred from the data
    when omitted. With ``person_fit_ncat`` set and an endpoint omitted, ``gpoly``
    and ``u3poly_fit`` instead read the items reflected on the scale ``ncat``
    declares. Each scale is recoded once and its copy, or its failure, is shared.
    Every other index, and every index without configured items, reads ``x``.
    """
    items = options.reverse_keyed_items
    inputs: list[_IndexInput] = [(x, None)] * len(indices)
    if items is None:
        return inputs
    # Two given endpoints already fix the declared scale, so one copy serves all.
    declares_scale = options.person_fit_ncat is not None and (
        options.scale_min is None or options.scale_max is None
    )
    recoded: dict[bool, _IndexInput] = {}
    for position, name in enumerate(indices):
        if not INDEX_REGISTRY[name].keyed_input:
            continue
        declared = declares_scale and name in _PERSON_FIT_SCALE_INDICES
        if declared not in recoded:
            recoded[declared] = _recoded_input(x, items, options, declared=declared)
        inputs[position] = recoded[declared]
    return inputs


def _recoded_input(
    x: np.ndarray, items: Sequence[int], options: IndexOptions, *, declared: bool
) -> _IndexInput:
    """Reverse-score ``items``, on the declared person-fit scale when ``declared``."""
    bounds: tuple[float | None, float | None] = (options.scale_min, options.scale_max)
    if declared:
        # Resolve the scale exactly as gpoly and u3poly_fit will.
        try:
            scale = _category_bounds(x, options.person_fit_ncat, *bounds)
        except (ValueError, TypeError) as error:
            # The scorers reject this scale on their own; report their message.
            return x, (None, str(error), error)
        if scale is not None:  # None: every response is missing, nothing to recode.
            bounds = scale
    try:
        return reverse_score(x, items, *bounds), None
    except (ValueError, TypeError) as error:
        return x, (None, f"reverse_keyed_items could not be applied: {error}", error)


def _score_registered_index(
    name: str,
    x: np.ndarray,
    options: IndexOptions,
    input_failure: IndexScoreResult | None = None,
) -> IndexScoreResult:
    """Compute one index and retain supported failures for ordered handling.

    ``input_failure`` explains why ``x`` could not be prepared; missing required
    options are still reported first.
    """
    spec = INDEX_REGISTRY[name]
    required_error = (
        spec.required_error(options)
        if spec.required_error is not None
        else _missing_required_options(spec, options)
    )
    if required_error is not None:
        return None, required_error, None
    if input_failure is not None:
        return input_failure

    try:
        score = spec.scorer(x, options)
    except (ValueError, RuntimeError, TypeError) as error:
        return None, str(error), error
    return score, None, None


def _record_index_result(
    name: str,
    result: IndexScoreResult,
    scores: dict[str, np.ndarray],
    errors: dict[str, str],
    *,
    strict: bool,
) -> None:
    """Record one result in selection order or raise its contextual failure."""
    score, error_message, cause = result
    if error_message is not None:
        if strict:
            failure = ValueError(f"index '{name}' failed: {error_message}")
            if cause is not None:
                raise failure from cause
            raise failure
        errors[name] = error_message
        return

    assert score is not None
    scores[name] = score


def score_registered_indices(
    x: np.ndarray,
    indices: list[str],
    options: IndexOptions,
    *,
    strict: bool = False,
    workers: int = 1,
    validated: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute registered indices, optionally in parallel, preserving selection order.

    Keyed-input indices read a shared copy with ``options.reverse_keyed_items``
    reverse-scored, made once per scale; every other index reads ``x``. A failed
    recoding is a failure of each keyed index only.
    """
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    if not validated:  # Orchestrators validate their request once, before scoring.
        workers = validate_worker_count(workers)
        validate_index_names(indices)

    scores: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    inputs = _keyed_responses(x, indices, options)

    if workers == 1 or len(indices) < 2:
        for name, (matrix, input_failure) in zip(indices, inputs, strict=True):
            _record_index_result(
                name,
                _score_registered_index(name, matrix, options, input_failure),
                scores,
                errors,
                strict=strict,
            )
        return scores, errors

    # Keep the default import path lean; concurrency is an explicit opt-in.
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    with ThreadPoolExecutor(max_workers=min(workers, len(indices))) as executor:
        futures = [
            executor.submit(_score_registered_index, name, matrix, options, input_failure)
            for name, (matrix, input_failure) in zip(indices, inputs, strict=True)
        ]
        for name, future in zip(indices, futures, strict=True):
            _record_index_result(
                name,
                future.result(),
                scores,
                errors,
                strict=strict,
            )

    return scores, errors
