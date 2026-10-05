"""Real consumer workflows whose public return types must remain precise."""

from typing import Literal, assert_type

import numpy as np

import ier
from ier import (
    CombineMethod,
    CompositeMethod,
    CompositeSummary,
    FloatArray,
    IndexOptions,
    MatrixLike,
    composite,
    composite_flag,
    composite_probability,
    composite_scores,
    composite_scores_summary,
    composite_summary,
    mahad,
    mahad_summary,
)


def combine_responses(responses: MatrixLike, method: CompositeMethod) -> np.ndarray:
    """Default calls can be assigned and used as arrays without casts."""
    scores: np.ndarray = composite(responses, method=method)
    assert_type(scores, np.ndarray)
    assert_type(composite(responses, return_diagnostics=False), FloatArray)
    assert_type(composite_probability(responses), FloatArray)
    assert_type(composite_probability(responses, return_diagnostics=False), FloatArray)
    scores_with_errors = composite(
        responses,
        indices=["irv", "longstring"],
        standardize=False,
        options=IndexOptions(),
        weights={"irv": 2.0},
        min_valid_indices=1,
        return_diagnostics=True,
        strict=False,
        workers=2,
    )
    assert_type(scores_with_errors, tuple[FloatArray, dict[str, str]])
    assert_type(
        composite_probability(responses, return_diagnostics=True),
        tuple[FloatArray, dict[str, str]],
    )
    return scores.reshape(-1)


def flag_responses(responses: MatrixLike) -> tuple[np.ndarray, np.ndarray]:
    """Flagging keeps its two- or three-element tuple contract."""
    result: tuple[np.ndarray, np.ndarray] = composite_flag(responses)
    assert_type(result, tuple[np.ndarray, np.ndarray])
    assert_type(
        composite_flag(responses, threshold=1.0, return_diagnostics=False),
        tuple[np.ndarray, np.ndarray],
    )
    assert_type(
        composite_flag(responses, percentile=99.0, return_diagnostics=True),
        tuple[np.ndarray, np.ndarray, dict[str, str]],
    )
    scores, flags = result
    return scores[flags], flags


def combine_with_runtime_setting(responses: MatrixLike, diagnostics: bool) -> None:
    """A Boolean chosen at runtime retains an honest union of return shapes."""
    assert_type(
        composite(responses, return_diagnostics=diagnostics),
        FloatArray | tuple[FloatArray, dict[str, str]],
    )
    assert_type(
        composite_flag(responses, return_diagnostics=diagnostics),
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, str]],
    )
    result = composite_probability(responses, return_diagnostics=diagnostics)
    assert_type(result, FloatArray | tuple[FloatArray, dict[str, str]])
    if isinstance(result, tuple):
        scores, errors = result
        assert_type(scores, FloatArray)
        assert_type(errors, dict[str, str])
    else:
        assert_type(result, FloatArray)


def reuse_components(responses: MatrixLike) -> np.ndarray:
    """The fresh-to-retained-score workflow remains usable with public types."""
    details = composite_summary(responses, indices=["irv", "longstring"])
    assert_type(details, CompositeSummary)
    result = composite_scores(details["indices"], errors=details["errors"])
    assert_type(result, FloatArray)
    assert_type(
        composite_scores_summary(details["indices"], errors=details["errors"]), CompositeSummary
    )
    return result


def reuse_screen_scores(responses: MatrixLike, method: CombineMethod) -> float:
    """Default screen output feeds composite reuse and exposes typed decision metadata."""
    c: ier.FloatArray = ier.composite(responses)
    result = ier.screen(responses)
    t: ier.IndexThresholdMap = result["thresholds"]
    percentiles: ier.IndexPercentileMap = result["percentiles"]
    sources: ier.IndexThresholdSourceMap = result["threshold_sources"]
    source: ier.IndexThresholdSource = sources["irv"]
    metadata = ier.index_catalog()["irv"]
    direction: ier.FlagDirection = metadata["flag_direction"]
    mode: ier.FlagMode = metadata["flag_mode"]
    reused = ier.composite_scores(result["scores"], method=method, unsupported="drop")
    assert_type(reused, FloatArray)
    summary = ier.composite_scores_summary(result["scores"], unsupported="drop")
    assert_type(summary["composite"], FloatArray)
    labels = [source, direction, mode]
    return float(c.mean() + reused.sum()) + len(t) + len(percentiles) + len(labels)


def screen_distances(responses: MatrixLike, flag: bool) -> float:
    """Mahalanobis calls return arrays or flag tuples according to ``flag``."""
    distances: np.ndarray = mahad(responses)
    assert_type(mahad(responses, flag=False, confidence=0.9), np.ndarray)
    flagged_distances, flags = mahad(responses, flag=True)
    assert_type(flagged_distances, np.ndarray)
    assert_type(flags, np.ndarray)
    method = "zscore"
    assert_type(mahad(responses, True, 0.99, True, method), tuple[np.ndarray, np.ndarray])
    assert_type(mahad(responses, flag=flag), np.ndarray | tuple[np.ndarray, np.ndarray])
    summary = mahad_summary(responses, na_rm=True)
    return (
        float(mahad(responses, method="iqr").mean())
        + float(flagged_distances[flags].sum() + distances.max())
        + float(summary["n_valid"])
    )


def score_psychometric_pairs(responses: MatrixLike, diag: bool) -> float:
    """Psychometric pair scores keep precise shapes in both item-correlation modes."""
    s: np.ndarray = ier.psychant(responses, diag=False)
    assert_type(ier.psychsyn(responses, item_correlations="pairwise"), np.ndarray)
    assert_type(
        ier.psychant(responses, diag=True, item_correlations="pairwise"),
        tuple[np.ndarray, np.ndarray],
    )
    assert_type(ier.psychsyn(responses, diag=diag), np.ndarray | tuple[np.ndarray, np.ndarray])
    assert_type(ier.psychant(responses, diag=diag), np.ndarray | tuple[np.ndarray, np.ndarray])
    assert_type(ier.psychsyn_flag(responses), tuple[np.ndarray, np.ndarray])
    assert_type(ier.psychant_flag(responses), tuple[np.ndarray, np.ndarray])
    assert_type(ier.person_total_flag(responses), tuple[np.ndarray, np.ndarray])
    assert_type(ier.u3_poly_flag(responses), tuple[np.ndarray, np.ndarray])
    assert_type(ier.midpoint_responding_flag(responses), tuple[np.ndarray, np.ndarray])
    pairs = ier.psychsyn_critval(responses, item_correlations="pairwise")
    assert_type(pairs, list[tuple[int, int, float]])
    options = IndexOptions(psychsyn_item_correlations="pairwise")
    assert_type(options.psychsyn_item_correlations, Literal["complete", "pairwise"])
    summary = ier.psychsyn_summary(responses, item_correlations="pairwise")
    return float(s.sum()) + len(pairs) + float(summary["mean_score"])


def score_consistency(responses: MatrixLike, diagnostics: bool) -> float:
    """Even-odd calls return arrays or diagnostic tuples according to ``diag``."""
    scores: np.ndarray = ier.evenodd(responses, [4, 4])
    assert_type(ier.evenodd(responses, [4, 4], method="halves"), np.ndarray)
    halves, available = ier.evenodd(responses, [4, 4], diag=True, method="halves")
    assert_type(halves, np.ndarray)
    assert_type(available, np.ndarray)
    assert_type(
        ier.evenodd(responses, [4, 4], diag=diagnostics),
        np.ndarray | tuple[np.ndarray, np.ndarray],
    )
    reliability = ier.individual_reliability(
        ier.reverse_score(responses, [1, 3], scale_min=1, scale_max=5),
        random_seed=np.random.default_rng(0),
        factors=[4, 4],
    )
    assert_type(reliability, np.ndarray)
    options = IndexOptions(evenodd_method="halves", reliability_factors=[4, 4])
    keyed = IndexOptions(scale_min=1, scale_max=5, reverse_keyed_items=[1, 3])
    assert_type(keyed.reverse_keyed_items, list[int] | None)
    assert_type(ier.index_catalog()["evenodd"]["uses_keyed_responses"], bool)
    return float(scores.mean() + halves[available > 1].sum() + reliability.max()) + len(
        options.reliability_factors or []
    )


def score_repetition_and_effort(responses: MatrixLike, times: MatrixLike, lags: bool) -> float:
    """Autocorrelation and response-time effort return arrays or tuples by their flags."""
    scores: np.ndarray = ier.autocorrelation(responses)
    assert_type(ier.autocorrelation(responses, max_lag=None, statistic="sum_abs"), np.ndarray)
    strongest, at_lag = ier.autocorrelation(responses, return_lags=True)
    assert_type(strongest, np.ndarray)
    assert_type(at_lag, np.ndarray)
    assert_type(
        ier.autocorrelation(responses, 10, 1, "max_abs", True, True),
        tuple[np.ndarray, np.ndarray],
    )
    assert_type(
        ier.autocorrelation(responses, return_lags=lags),
        np.ndarray | tuple[np.ndarray, np.ndarray],
    )
    assert_type(ier.autocorrelation_flag(responses, max_lag=5), tuple[np.ndarray, np.ndarray])
    effort: np.ndarray = ier.response_time_effort(times, max_threshold=10.0)
    assert_type(ier.response_time_effort(times, [1.0, 2.0], normative_fraction=0.2), np.ndarray)
    effort_scores, rapid = ier.response_time_effort(times, return_item_flags=True)
    assert_type(rapid, np.ndarray)
    assert_type(
        ier.response_time_effort(times, return_item_flags=lags),
        np.ndarray | tuple[np.ndarray, np.ndarray],
    )
    assert_type(
        ier.response_time_effort_flag(times, 0.9, thresholds=2.0), tuple[np.ndarray, np.ndarray]
    )
    ier.save_response_time_archive("rte.npz", effort, effort < 0.9, threshold=0.9, metric="effort")
    options = IndexOptions(autocorrelation_max_lag=None, autocorrelation_statistic="sum_abs")
    assert_type(options.autocorrelation_statistic, Literal["max_abs", "sum_abs"])
    assert_type(options.autocorrelation_max_lag, int | None)
    return float(scores.max() + strongest.sum() + at_lag.min() + effort.mean()) + float(
        effort_scores.sum() + rapid.sum()
    )


def type_option_variables(responses: MatrixLike, screened: ier.ScreenResult) -> float:
    """Exported option aliases type variables passed to the options they name."""
    mode: ier.ItemCorrelationMode = "pairwise"
    kind: ier.AgreementKind = "spearman"
    method: ier.EvenOddMethod = "halves"
    assert_type(ier.psychsyn(responses, item_correlations=mode), np.ndarray)
    pairs = ier.psychsyn_critval(responses, item_correlations=mode)
    assert_type(pairs, list[tuple[int, int, float]])
    names, matrix = ier.index_agreement(screened, kind=kind)
    assert_type(names, list[str])
    assert_type(matrix, FloatArray)
    assert_type(ier.evenodd(responses, [4, 4], method=method), np.ndarray)
    options = IndexOptions(psychsyn_item_correlations=mode, evenodd_method=method)
    assert_type(options.psychsyn_item_correlations, ier.ItemCorrelationMode)
    assert_type(options.evenodd_method, ier.EvenOddMethod)
    return float(matrix.sum()) + len(names) + len(pairs)


def score_person_fit(responses: MatrixLike, binary: MatrixLike) -> float:
    """Item-step person-fit statistics return arrays; their flag helpers return tuples."""
    errors: np.ndarray = ier.gpoly(responses, ncat=5, scale_min=1)
    assert_type(ier.gpoly(responses, 5, 1, 5, False, True), np.ndarray)
    assert_type(ier.u3poly(responses, scale_max=5, na_rm=False), np.ndarray)
    assert_type(ier.ht(binary, na_rm=True), np.ndarray)
    assert_type(
        ier.gpoly_flag(responses, 0.5, ncat=5, normalize=False), tuple[np.ndarray, np.ndarray]
    )
    assert_type(
        ier.u3poly_flag(responses, percentile=90.0, scale_min=1), tuple[np.ndarray, np.ndarray]
    )
    fit, flagged = ier.ht_flag(binary, threshold=0.0)
    assert_type(flagged, np.ndarray)
    options = IndexOptions(person_fit_ncat=5, scale_min=1)
    assert_type(options.person_fit_ncat, int | None)
    return float(errors.mean() + fit.min()) + (options.person_fit_ncat or 0)


def count_psychometric_respondents(responses: MatrixLike) -> int:
    """Psychometric summaries share the respondent count keys of other summaries."""
    summary = ier.psychsyn_summary(responses, critval=0.5, item_correlations="pairwise")
    total: int = summary["n_total"]
    valid: int = summary["n_valid"]
    missing: int = summary["n_missing"]
    return total + valid + missing + int(mahad_summary(responses)["n_valid"])
