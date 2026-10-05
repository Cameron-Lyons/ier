"""Respondent-aligned result tables and index agreement matrices."""

import csv
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from fractions import Fraction
from io import StringIO
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from ier import (
    composite_scores_summary,
    composite_table,
    index_agreement,
    screen_scores,
    screen_table,
)
from ier._cli_composite import CompositeReport
from ier._cli_output import _write_composite_csv, _write_screen_csv

if TYPE_CHECKING:
    from pathlib import Path

    from ier.types import CompositeSummary, ScreenResult


def _screen_result() -> "ScreenResult":
    return screen_scores(
        {
            "irv": [0.1, np.nan, 0.9, 1.2, 0.3],
            "longstring": [8.0, 3.0, np.nan, 7.0, 6.0],
            "onset": [np.nan, 4.0, np.nan, np.nan, 9.0],
        },
        thresholds={"irv": 0.5, "longstring": 5.0},
        min_valid_indices=2,
    )


def _header(text: str) -> list[str]:
    return next(csv.reader(StringIO(text)))


def test_screen_table_columns_follow_cli_schema_order_and_dtypes() -> None:
    result = _screen_result()
    table = screen_table(result)

    assert list(table) == [
        "flag_count",
        "valid_index_count",
        "consensus_eligible",
        "consensus_flag",
        "irv_score",
        "irv_flag",
        "longstring_score",
        "longstring_flag",
        "onset_score",
        "onset_flag",
    ]
    for name in ("flag_count", "valid_index_count"):
        assert table[name].dtype.kind == "i"
    for name in ("consensus_eligible", "consensus_flag", "irv_flag", "onset_flag"):
        assert table[name].dtype == np.bool_
    for name in ("irv_score", "longstring_score", "onset_score"):
        assert table[name].dtype == np.float64
    assert all(column.shape == (5,) for column in table.values())
    np.testing.assert_array_equal(table["flag_count"], [2, 1, 0, 1, 3])
    np.testing.assert_array_equal(table["valid_index_count"], [2, 2, 1, 2, 3])
    np.testing.assert_array_equal(table["consensus_flag"], [True, False, False, False, True])


def test_screen_table_returns_result_arrays_by_reference() -> None:
    result = _screen_result()
    table = screen_table(result, respondent_ids=["a", "b", "c", "d", "e"])

    assert table["flag_count"] is result["flag_counts"]
    assert table["valid_index_count"] is result["valid_index_counts"]
    assert table["consensus_eligible"] is result["consensus_eligible"]
    assert table["consensus_flag"] is result["consensus_flags"]
    for name in result["indices_used"]:
        assert table[f"{name}_score"] is result["scores"][name]
        assert table[f"{name}_flag"] is result["flags"][name]


@pytest.mark.parametrize("respondent_ids", [None, ["r1", "r2", "r3", "r4", "r5"]])
def test_screen_table_matches_cli_csv_header(respondent_ids: list[str] | None) -> None:
    result = _screen_result()
    handle = StringIO()
    _write_screen_csv(handle, result, respondent_ids)

    expected = _header(handle.getvalue())
    if respondent_ids is None:
        assert ["respondent", *screen_table(result)] == expected
    else:
        assert list(screen_table(result, respondent_ids=respondent_ids)) == expected


def test_screen_table_matches_cli_csv_header_after_soft_failures() -> None:
    result = screen_scores(
        {"irv": [0.2, 0.8, 1.1]}, errors={"mad": "missing item pairs"}, min_flags=1
    )
    handle = StringIO()
    _write_screen_csv(handle, result)
    assert ["respondent", *screen_table(result)] == _header(handle.getvalue())
    assert "mad_score" not in screen_table(result)


def test_screen_table_respondent_column_uses_numpy_strings() -> None:
    result = _screen_result()
    ids = ["resp-0", "resp-1", "resp-22", "r3", "r4"]
    table = screen_table(result, respondent_ids=ids)

    assert next(iter(table)) == "respondent"
    assert table["respondent"].dtype.kind == "U"
    assert table["respondent"].tolist() == ids
    from_array = screen_table(result, respondent_ids=np.asarray(ids))  # type: ignore[arg-type]
    np.testing.assert_array_equal(from_array["respondent"], ids)
    from_tuple = screen_table(result, respondent_ids=tuple(ids))
    np.testing.assert_array_equal(from_tuple["respondent"], ids)


@pytest.mark.parametrize(
    ("include_scores", "include_flags", "suffixes"),
    [
        (True, False, ["_score"]),
        (False, True, ["_flag"]),
        (False, False, []),
    ],
)
def test_screen_table_column_selection(
    include_scores: bool, include_flags: bool, suffixes: list[str]
) -> None:
    result = _screen_result()
    table = screen_table(result, include_scores=include_scores, include_flags=include_flags)

    expected = ["flag_count", "valid_index_count", "consensus_eligible", "consensus_flag"]
    for name in result["indices_used"]:
        expected.extend(f"{name}{suffix}" for suffix in suffixes)
    assert list(table) == expected


@pytest.mark.parametrize(
    ("respondent_ids", "error", "message"),
    [
        (["a", "b"], ValueError, "expected 5, received 2"),
        (["a", "b", "c", "d", "e", "f"], ValueError, "expected 5, received 6"),
        (["a", "b", "c", "d", 5], ValueError, "respondent IDs must be strings"),
        (["a", "b", None, "d", "e"], ValueError, "respondent IDs must be strings"),
        ("abcde", TypeError, "not a single string"),
        (b"abcde", TypeError, "not a single string"),
        (5, TypeError, "sequence of strings"),
    ],
)
def test_screen_table_rejects_misaligned_or_non_string_identifiers(
    respondent_ids: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        screen_table(_screen_result(), respondent_ids=respondent_ids)


def test_tables_reject_identifiers_that_numpy_would_truncate() -> None:
    # Fixed-width NumPy Unicode drops a trailing NUL, which would merge "r1\0" into "r1".
    ids = ["r1", "r1\0", "r3", "r4", "r5"]
    with pytest.raises(ValueError, match="respondent IDs cannot end with a NUL character"):
        screen_table(_screen_result(), respondent_ids=ids)
    with pytest.raises(ValueError, match="respondent IDs cannot end with a NUL character"):
        composite_table(_composite_summary(), respondent_ids=ids[:4])

    embedded = ["r\0" + str(position) for position in range(5)]
    assert screen_table(_screen_result(), respondent_ids=embedded)["respondent"].tolist() == (
        embedded
    )


@pytest.mark.parametrize(
    ("respondent_ids", "error"),
    [("abcde", TypeError), (5, TypeError), (["a", "b"], ValueError), ([1, 2, 3, 4, 5], ValueError)],
)
def test_tables_and_archives_raise_the_same_identifier_errors(
    tmp_path: "Path", respondent_ids: Any, error: type[Exception]
) -> None:
    from ier import save_screen_archive

    result = _screen_result()
    destination = tmp_path / "screening.npz"
    with pytest.raises(error):
        screen_table(result, respondent_ids=respondent_ids)
    with pytest.raises(error):
        save_screen_archive(destination, result, respondent_ids=respondent_ids)
    assert not destination.exists()


_IDS = ["r1", "r2", "r3", "r4", "r5"]
_UNORDERED_IDS: list[tuple[Callable[[], Any], str]] = [
    (lambda: set(_IDS), "not a set"),
    (lambda: frozenset(_IDS), "not a set"),
    (lambda: dict.fromkeys(_IDS).keys(), "not a set"),
    (lambda: dict.fromkeys(_IDS), "not a mapping"),
    (lambda: OrderedDict.fromkeys(_IDS), "not a mapping"),
    (lambda: (value for value in _IDS), "not an iterator"),
    (lambda: iter(_IDS), "not an iterator"),
    (lambda: map(str, _IDS), "not an iterator"),
]


@pytest.mark.parametrize(
    ("make_ids", "message"),
    _UNORDERED_IDS,
    ids=["set", "frozenset", "dict_keys", "dict", "ordered_dict", "generator", "iter", "map"],
)
def test_tables_and_archives_reject_identifiers_without_respondent_order(
    tmp_path: "Path", make_ids: Callable[[], Any], message: str
) -> None:
    # A set's hash order (or a consumed iterator) would attach IDs to the wrong rows.
    from ier import save_response_time_archive, save_score_archive, save_screen_archive

    result = _screen_result()
    summary = _composite_summary()
    destination = tmp_path / "ids.npz"
    expected = f"^respondent_ids must be a sequence of strings in respondent order, {message}$"
    calls: list[Callable[[Any], object]] = [
        lambda ids: screen_table(result, respondent_ids=ids),
        lambda ids: composite_table(summary, respondent_ids=ids),
        lambda ids: save_screen_archive(destination, result, respondent_ids=ids),
        lambda ids: save_score_archive(destination, result["scores"], respondent_ids=ids),
        lambda ids: save_response_time_archive(
            destination, [1.0] * 5, [True] * 5, threshold=2.0, respondent_ids=ids
        ),
    ]
    for call in calls:
        respondent_ids = make_ids()
        with pytest.raises(TypeError, match=expected):
            call(respondent_ids)
        if isinstance(respondent_ids, Iterator):
            # Rejected before reading, so a caller's iterator is left unconsumed.
            assert next(respondent_ids) == "r1"
    assert not destination.exists()


def test_tables_and_archives_accept_ordered_identifier_containers(tmp_path: "Path") -> None:
    from ier import load_screen_archive, save_screen_archive

    pd = pytest.importorskip("pandas")
    result = _screen_result()
    destination = tmp_path / "ids.npz"
    containers: list[Any] = [
        tuple(_IDS),
        np.asarray(_IDS),
        np.asarray(_IDS, dtype=object),
        pd.Index(_IDS),
        pd.Series(_IDS),
        dict(enumerate(_IDS)).values(),
    ]
    for respondent_ids in containers:
        assert screen_table(result, respondent_ids=respondent_ids)["respondent"].tolist() == _IDS
        save_screen_archive(destination, result, respondent_ids=respondent_ids)
        assert load_screen_archive(destination)["respondent_ids"] == _IDS


@pytest.mark.parametrize("option", ["include_scores", "include_flags"])
@pytest.mark.parametrize("value", [1, 0, None, "yes", np.bool_(True)])
def test_screen_table_rejects_non_boolean_column_selection(option: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{option} must be a boolean"):
        screen_table(_screen_result(), **{option: value})  # type: ignore[arg-type]


def test_screen_table_rejects_columns_that_do_not_match_respondent_count() -> None:
    result = dict(_screen_result())
    result["flag_counts"] = np.zeros(4, dtype=np.int_)
    with pytest.raises(ValueError, match="flag_count must contain one value per respondent"):
        screen_table(cast("ScreenResult", result))

    nested = dict(_screen_result())
    nested["scores"] = {**nested["scores"], "irv": np.zeros((5, 1))}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="irv_score must contain one value per respondent"):
        screen_table(cast("ScreenResult", nested))


def test_screen_table_preserves_pandas_index() -> None:
    pd = pytest.importorskip("pandas")
    from ier import screen

    df = pd.DataFrame(
        np.array(
            [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1], [1, 1, 5, 5, 3]],
            dtype=float,
        ),
        index=["p01", "p02", "p03", "p04"],
    )
    result = screen(df, indices=["irv", "longstring"])
    frame = pd.DataFrame(screen_table(result), index=df.index)

    assert list(frame.index) == ["p01", "p02", "p03", "p04"]
    assert list(frame.columns) == list(screen_table(result))
    np.testing.assert_array_equal(frame["irv_score"].to_numpy(), result["scores"]["irv"])
    assert frame["irv_flag"].dtype == bool

    labelled = pd.DataFrame(screen_table(result, respondent_ids=list(df.index)))
    assert labelled["respondent"].tolist() == ["p01", "p02", "p03", "p04"]


def test_screen_table_builds_polars_frame() -> None:
    pl = pytest.importorskip("polars")
    result = _screen_result()
    frame = pl.DataFrame(screen_table(result, respondent_ids=["a", "b", "c", "d", "e"]))

    assert frame.columns == ["respondent", *screen_table(result)]
    assert frame["respondent"].to_list() == ["a", "b", "c", "d", "e"]
    assert frame["flag_count"].to_list() == [2, 1, 0, 1, 3]
    assert frame["irv_flag"].dtype == pl.Boolean


def _composite_summary() -> "CompositeSummary":
    return composite_scores_summary(
        {
            "irv": [0.2, 1.0, np.nan, 1.4],
            "longstring": [6.0, 2.0, 1.0, np.nan],
            "person_total": [0.1, 0.5, 0.6, 0.4],
        },
        min_valid_indices=2,
    )


def test_composite_table_columns_and_references() -> None:
    summary = _composite_summary()
    table = composite_table(summary)

    assert list(table) == [
        "composite_score",
        "valid_index_count",
        "irv_score",
        "longstring_score",
        "person_total_score",
    ]
    assert table["composite_score"] is summary["composite"]
    assert table["valid_index_count"] is summary["valid_index_counts"]
    for name, scores in summary["indices"].items():
        assert table[f"{name}_score"] is scores
    np.testing.assert_array_equal(table["valid_index_count"], [3, 3, 2, 2])


def test_composite_table_matches_cli_component_csv_header() -> None:
    summary = _composite_summary()
    ids = ["a", "b", "c", "d"]
    handle = StringIO()
    _write_composite_csv(
        handle,
        CompositeReport(
            summary["composite"],
            summary["method"],
            ids,
            component_scores=summary["indices"],
            valid_index_counts=summary["valid_index_counts"],
        ),
    )
    table = composite_table(summary, respondent_ids=ids)

    assert list(table) == _header(handle.getvalue())
    assert table["respondent"].tolist() == ids


def test_composite_table_can_omit_components_and_validates_inputs() -> None:
    summary = _composite_summary()
    assert list(composite_table(summary, include_components=False)) == [
        "composite_score",
        "valid_index_count",
    ]
    with pytest.raises(ValueError, match="include_components must be a boolean"):
        composite_table(summary, include_components=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected 4, received 3"):
        composite_table(summary, respondent_ids=["a", "b", "c"])


def _flag_result(flags: dict[str, Any]) -> "ScreenResult":
    n_respondents = len(next(iter(flags.values())))
    return cast(
        "ScreenResult",
        {
            "flags": {name: np.asarray(values) for name, values in flags.items()},
            "scores": {name: np.zeros(n_respondents) for name in flags},
            "indices_used": list(flags),
            "n_respondents": n_respondents,
        },
    )


def test_overlap_and_jaccard_match_hand_counts() -> None:
    result = screen_scores(
        {
            "irv": [0.1, 0.2, 0.9, 0.3, 1.0, 1.1],
            "longstring": [6.0, 1.0, 7.0, 8.0, 2.0, 1.0],
            "mahad": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        thresholds={"irv": 0.5, "longstring": 5.0, "mahad": 10.0},
    )
    # irv flags rows 0, 1, 3; longstring flags 0, 2, 3; mahad flags nobody.
    names, overlap = index_agreement(result, kind="overlap")
    assert names == ["irv", "longstring", "mahad"]
    assert overlap.dtype == np.float64
    np.testing.assert_array_equal(overlap, [[3, 2, 0], [2, 3, 0], [0, 0, 0]])

    names, jaccard = index_agreement(result)
    assert names == ["irv", "longstring", "mahad"]
    np.testing.assert_array_equal(
        jaccard,
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, np.nan]],
    )
    assert index_agreement(result, "jaccard")[1].tobytes() == jaccard.tobytes()


def test_overlap_counts_cross_row_batches_and_include_presence_indices() -> None:
    rng = np.random.default_rng(7)
    flags = {name: rng.random(300_001) < 0.3 for name in ("irv", "onset", "custom")}
    result = _flag_result(flags)

    names, overlap = index_agreement(result, kind="overlap")
    assert names == ["irv", "onset", "custom"]
    expected = [[np.count_nonzero(flags[left] & flags[right]) for right in names] for left in names]
    np.testing.assert_array_equal(overlap, expected)


def test_agreement_of_an_empty_screen_is_empty() -> None:
    result = screen_scores({}, n_respondents=4, errors={"mad": "unconfigured"})
    for kind in ("overlap", "jaccard", "spearman"):
        names, matrix = index_agreement(result, kind)  # type: ignore[arg-type]
        assert names == []
        assert matrix.shape == (0, 0)


def test_agreement_takes_screen_result_like_the_plot_helpers() -> None:
    result = _screen_result()
    names, matrix = index_agreement(screen_result=result, kind="overlap")
    assert names == result["indices_used"]
    np.testing.assert_array_equal(matrix, index_agreement(result, "overlap")[1])
    with pytest.raises(TypeError, match="result"):
        index_agreement(result=result)  # type: ignore[call-arg]


@pytest.mark.parametrize("kind", ["pearson", "", None, 3, "Jaccard"])
def test_agreement_rejects_unknown_kinds(kind: object) -> None:
    with pytest.raises(ValueError, match="kind must be 'overlap', 'jaccard', or 'spearman'"):
        index_agreement(_screen_result(), kind)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["overlap", "spearman"])
def test_agreement_rejects_misaligned_vectors(kind: str) -> None:
    result = dict(_screen_result())
    key = "flags" if kind == "overlap" else "scores"
    result[key] = {**result[key], "longstring": np.zeros(4)}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="longstring result vector must contain one value"):
        index_agreement(cast("ScreenResult", result), kind)  # type: ignore[arg-type]


# Registered indices whose low scores are suspicious; Spearman agreement negates them.
_LOW_DIRECTION = frozenset({"irv", "psychsyn", "person_total", "markov"})


def _oriented(name: str, values: np.ndarray) -> np.ndarray:
    return -values if name in _LOW_DIRECTION else values


def _oracle_ranks(values: list[float]) -> list[Fraction]:
    return [
        Fraction(sum(other < value for other in values))
        + Fraction(sum(other == value for other in values) + 1, 2)
        for value in values
    ]


def _oracle_spearman(left: np.ndarray, right: np.ndarray) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right, strict=True)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 3:
        return math.nan
    left_ranks = _oracle_ranks([a for a, _ in pairs])
    right_ranks = _oracle_ranks([b for _, b in pairs])
    left_mean = sum(left_ranks) / len(pairs)
    right_mean = sum(right_ranks) / len(pairs)
    covariance = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_variance = sum((a - left_mean) ** 2 for a in left_ranks)
    right_variance = sum((b - right_mean) ** 2 for b in right_ranks)
    if left_variance == 0 or right_variance == 0:
        return math.nan
    return float(covariance) / math.sqrt(float(left_variance) * float(right_variance))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_spearman_matches_brute_force_rank_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_respondents = 41
    irv = rng.normal(size=n_respondents)
    irv[rng.random(n_respondents) < 0.2] = np.nan
    psychsyn = rng.normal(size=n_respondents) + irv  # identical availability to irv
    longstring = rng.integers(0, 4, size=n_respondents).astype(float)  # heavy ties
    mahad = np.round(rng.normal(size=n_respondents), 1)
    mahad[rng.random(n_respondents) < 0.3] = np.nan
    person_total = np.full(n_respondents, 0.25)  # constant scores
    markov = np.full(n_respondents, np.nan)
    markov[[3, 17]] = [0.1, 0.2]  # too few observations
    guttman = np.full(n_respondents, np.nan)
    guttman[[0, 1, 2, 5, 8]] = [1.0, 1.0, 2.0, 3.0, 3.0]  # sparse, tied
    onset = np.where(rng.random(n_respondents) < 0.5, 12.0, np.nan)
    scores = {
        "irv": irv,
        "longstring": longstring,
        "psychsyn": psychsyn,
        "mahad": mahad,
        "person_total": person_total,
        "markov": markov,
        "onset": onset,
        "guttman": guttman,
    }
    result = screen_scores(scores)

    names, matrix = index_agreement(result, kind="spearman")

    assert names == [name for name in scores if name != "onset"]
    oriented = {name: _oriented(name, scores[name]) for name in names}
    expected = np.array(
        [[_oracle_spearman(oriented[left], oriented[right]) for right in names] for left in names]
    )
    np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(matrix, matrix.T)
    defined = ~np.isnan(np.diagonal(expected))
    np.testing.assert_array_equal(np.diagonal(matrix)[defined], 1.0)
    assert np.isnan(matrix[names.index("person_total")]).all()
    assert np.isnan(matrix[names.index("markov")]).all()


def test_spearman_handles_monotone_transforms_and_reversals() -> None:
    base = np.array([0.5, 3.0, 1.0, 9.0, 2.0, 4.0])
    result = screen_scores(
        {
            "irv": -base,  # low irv is suspicious, so its suspiciousness ranks follow base
            "longstring": np.exp(base),
            "mahad": -base,
            "person_total": np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        }
    )
    names, matrix = index_agreement(result, kind="spearman")
    assert names == ["irv", "longstring", "mahad", "person_total"]
    np.testing.assert_allclose(matrix[0, :3], [1.0, 1.0, -1.0], rtol=0, atol=1e-15)
    expected = _oracle_spearman(base, -result["scores"]["person_total"])
    np.testing.assert_allclose(matrix[0, 3], expected, rtol=0, atol=1e-12)


def test_spearman_orients_scores_by_flag_direction() -> None:
    # Respondent 0 is the most suspicious on both indices and respondent 5 the least:
    # irv flags low scores while longstring flags high scores.
    result = screen_scores(
        {
            "irv": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "longstring": [10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
            "psychsyn": [0.1, 0.3, 0.2, 0.9, 0.7, 0.8],
        },
        thresholds={"irv": 0.3, "longstring": 7.0, "psychsyn": 0.25},
    )
    _, jaccard = index_agreement(result)
    names, rho = index_agreement(result, kind="spearman")

    assert names == ["irv", "longstring", "psychsyn"]
    assert jaccard[0, 1] == 1.0
    np.testing.assert_allclose([rho[0, 1], rho[1, 0]], [1.0, 1.0], rtol=0, atol=1e-15)
    # Two low-direction indices keep the sign of their raw rank correlation.
    raw = _oracle_spearman(result["scores"]["irv"], result["scores"]["psychsyn"])
    np.testing.assert_allclose(rho[0, 2], raw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rho[1, 2], raw, rtol=0, atol=1e-12)
    assert raw > 0


def test_spearman_ignores_non_finite_scores_and_keeps_unregistered_indices() -> None:
    result = _flag_result({"onset": [False] * 5, "custom": [False] * 5, "irv": [False] * 5})
    result["scores"] = {
        "onset": np.array([np.nan, 3.0, np.nan, 9.0, 1.0]),
        "custom": np.array([1.0, np.inf, 3.0, 4.0, -np.inf]),
        "irv": np.array([2.0, 1.0, 6.0, 8.0, 0.5]),
    }
    names, matrix = index_agreement(result, kind="spearman")

    assert names == ["custom", "irv"]
    # Unregistered scores keep their raw orientation; low-direction irv is negated.
    expected = _oracle_spearman(result["scores"]["custom"], -result["scores"]["irv"])
    np.testing.assert_allclose(matrix[0, 1], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.diagonal(matrix), [1.0, 1.0])


def test_million_respondent_overlap_is_fast() -> None:
    rng = np.random.default_rng(11)
    names = [f"index_{position}" for position in range(11)]
    flags = {name: rng.random(1_000_000) < 0.01 for name in names}
    result = cast(
        "ScreenResult",
        {"flags": flags, "scores": {}, "indices_used": names, "n_respondents": 1_000_000},
    )

    elapsed = math.inf
    for _ in range(3):
        start = time.perf_counter()
        _, overlap = index_agreement(result, kind="overlap")
        elapsed = min(elapsed, time.perf_counter() - start)

    assert elapsed < 0.5
    assert overlap[0, 0] == np.count_nonzero(flags["index_0"])
    assert overlap[3, 7] == np.count_nonzero(flags["index_3"] & flags["index_7"])
