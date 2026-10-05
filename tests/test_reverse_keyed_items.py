"""IndexOptions.reverse_keyed_items recodes items only for indices that need keyed responses."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

import ier._registry
from ier import (
    IndexOptions,
    composite,
    composite_scores,
    composite_summary,
    gpoly,
    ht,
    index_catalog,
    reverse_score,
    screen,
    u3poly,
)
from ier._registry import INDEX_REGISTRY, IndexSpec, score_registered_indices
from ier.cli import _build_parser, _options_from_args, main

if TYPE_CHECKING:
    from collections.abc import Callable

_ROOT = Path(__file__).resolve().parents[1]
# Two reverse-worded items in each of three six-item scales.
_REVERSED = [1, 4, 7, 10, 13, 16]
_KEYED = [name for name, spec in INDEX_REGISTRY.items() if spec.keyed_input]
_OPTIONS = IndexOptions(
    scale_min=1,
    scale_max=5,
    evenodd_factors=[6, 6, 6],
    reliability_random_seed=7,
    mad_positive_items=[0, 2],
    mad_negative_items=[1, 4],
    semantic_item_pairs=[(0, 2), (3, 5)],
    infrequency_item_indices=[17],
    infrequency_expected_responses=[5],
)


def _presented_responses(seed: int = 4) -> np.ndarray:
    """Three correlated scales as presented, with reverse-worded items and careless rows."""
    rng = np.random.default_rng(seed)
    traits = rng.normal(size=(80, 3))
    latent = np.repeat(traits, 6, axis=1) + rng.normal(scale=0.8, size=(80, 18))
    presented = np.clip(np.rint(3 + 1.1 * latent), 1, 5)
    presented[:, _REVERSED] = 6 - presented[:, _REVERSED]
    presented[:6] = rng.integers(1, 6, size=(6, 18))
    presented[6] = 3.0
    presented[7, [2, 9]] = np.nan
    return presented


def _manual_scores(
    presented: np.ndarray, indices: list[str], options: IndexOptions
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Score keyed indices on a recoded copy and every other index as presented."""
    recoded = reverse_score(presented, _REVERSED, options.scale_min, options.scale_max)
    keyed = [name for name in indices if name in _KEYED]
    raw = [name for name in indices if name not in _KEYED]
    keyed_result = screen(recoded, indices=keyed, options=options, min_flags=1)
    raw_result = screen(presented, indices=raw, options=options, min_flags=1)
    scores = {**keyed_result["scores"], **raw_result["scores"]}
    errors = {**keyed_result["errors"], **raw_result["errors"]}
    # Restore selection order, which also fixes the order composites reduce in.
    ordered = {name: scores[name] for name in indices if name in scores}
    return ordered, {name: errors[name] for name in indices if name in errors}


@pytest.mark.parametrize("workers", [1, 4])
def test_screen_scores_keyed_indices_on_recoded_items_and_others_as_presented(
    workers: int,
) -> None:
    presented = _presented_responses()
    original = presented.copy()
    indices = list(INDEX_REGISTRY)
    options = replace(_OPTIONS, reverse_keyed_items=_REVERSED)

    result = screen(presented, indices=indices, options=options, min_flags=1, workers=workers)

    expected_scores, expected_errors = _manual_scores(presented, indices, _OPTIONS)
    assert result["errors"] == expected_errors
    assert result["indices_used"] == [name for name in indices if name in expected_scores]
    for name, expected in expected_scores.items():
        np.testing.assert_array_equal(result["scores"][name], expected, err_msg=name)
    np.testing.assert_array_equal(presented, original)

    unkeyed = screen(presented, indices=indices, options=_OPTIONS, min_flags=1)
    for name in (name for name in _KEYED if name in result["scores"]):
        # Recoding changes what every scored keyed index measures on this survey.
        assert not np.array_equal(
            result["scores"][name], unkeyed["scores"][name], equal_nan=True
        ), name


def test_recoding_restores_even_odd_consistency_without_hiding_straightlining() -> None:
    presented = _presented_responses()
    options = replace(_OPTIONS, reverse_keyed_items=_REVERSED, evenodd_method="halves")

    keyed = screen(presented, indices=["evenodd", "longstring"], options=options, min_flags=1)
    raw = screen(presented, indices=["evenodd", "longstring"], options=_OPTIONS, min_flags=1)

    attentive = slice(8, None)
    assert np.nanmean(keyed["scores"]["evenodd"][attentive]) > 0.5
    assert np.nanmean(keyed["scores"]["evenodd"][attentive]) > (
        np.nanmean(raw["scores"]["evenodd"][attentive]) + 0.5
    )
    np.testing.assert_array_equal(keyed["scores"]["longstring"], raw["scores"]["longstring"])
    assert keyed["scores"]["longstring"][6] == 18


@pytest.mark.parametrize("workers", [1, 3])
def test_composites_route_recoded_items_to_keyed_components(workers: int) -> None:
    presented = _presented_responses()
    indices = ["irv", "longstring", "lz", "evenodd", "individual_reliability", "guttman"]
    options = replace(_OPTIONS, reverse_keyed_items=_REVERSED)

    details = composite_summary(presented, indices=indices, options=options, workers=workers)

    expected_scores, _ = _manual_scores(presented, indices, _OPTIONS)
    assert details["indices_used"] == indices
    for name in indices:
        np.testing.assert_array_equal(details["indices"][name], expected_scores[name])
    # The configured MAD pairs add mad to the best subset; only lz reads recoded items.
    best_subset = composite(presented, method="best_subset", options=options, workers=workers)
    subset_scores, _ = _manual_scores(presented, ["mad", "irv", "longstring", "lz"], _OPTIONS)
    np.testing.assert_array_equal(best_subset, composite_scores(subset_scores))


def test_omitted_scale_bounds_are_inferred_from_the_whole_matrix() -> None:
    presented = _presented_responses()
    presented[8, 0] = 6.0
    options = IndexOptions(reverse_keyed_items=[1, 4], reliability_random_seed=3)
    indices = ["guttman", "individual_reliability", "irv"]

    result = screen(presented, indices=indices, options=options)

    recoded = reverse_score(presented, [1, 4])
    np.testing.assert_array_equal(recoded[:, [1, 4]], 7.0 - presented[:, [1, 4]])
    expected = screen(recoded, indices=indices[:2], options=IndexOptions(reliability_random_seed=3))
    for name in indices[:2]:
        np.testing.assert_array_equal(result["scores"][name], expected["scores"][name])
    np.testing.assert_array_equal(
        result["scores"]["irv"], screen(presented, indices=["irv"])["scores"]["irv"]
    )


def test_integer_responses_keep_their_exact_recoded_dtype() -> None:
    presented = np.nan_to_num(_presented_responses(), nan=3.0).astype(np.int16)
    options = IndexOptions(scale_min=1, scale_max=5, reverse_keyed_items=_REVERSED)
    received: list[np.dtype[Any]] = []

    def keyed_scorer(x: np.ndarray, options: IndexOptions) -> np.ndarray:
        del options
        received.append(x.dtype)
        return np.zeros(len(x))

    additions = {"keyed_probe": IndexSpec("keyed_probe", keyed_scorer, "high", keyed_input=True)}
    with patch.dict(INDEX_REGISTRY, additions):
        score_registered_indices(presented, ["keyed_probe"], options)

    assert received == [np.dtype(np.int16)]
    result = screen(presented, indices=["guttman", "lz"], options=options)
    recoded = reverse_score(presented, _REVERSED, 1, 5)
    expected = screen(recoded, indices=["guttman", "lz"])
    for name in ["guttman", "lz"]:
        np.testing.assert_array_equal(result["scores"][name], expected["scores"][name])


@pytest.mark.parametrize("workers", [1, 2])
def test_keyed_scorers_share_one_recoded_copy_and_others_read_the_input(workers: int) -> None:
    presented = _presented_responses()
    received: dict[str, list[np.ndarray]] = {}

    def recorder(label: str) -> Callable[[np.ndarray, IndexOptions], np.ndarray]:
        def scorer(x: np.ndarray, options: IndexOptions) -> np.ndarray:
            del options
            received.setdefault(label, []).append(x)
            return np.zeros(len(x))

        return scorer

    additions = {
        "raw_probe": IndexSpec("raw_probe", recorder("raw_probe"), "high"),
        "keyed_first": IndexSpec("keyed_first", recorder("keyed_first"), "high", keyed_input=True),
        "keyed_second": IndexSpec(
            "keyed_second", recorder("keyed_second"), "high", keyed_input=True
        ),
    }
    options = IndexOptions(scale_min=1, scale_max=5, reverse_keyed_items=[0, 3])
    with (
        patch.dict(INDEX_REGISTRY, additions),
        patch.object(ier._registry, "reverse_score", wraps=reverse_score) as recode,
    ):
        scores, errors = score_registered_indices(
            presented, list(additions), options, workers=workers
        )

    assert errors == {}
    assert list(scores) == list(additions)
    assert recode.call_count == 1
    assert received["raw_probe"][0] is presented
    keyed_matrix = received["keyed_first"][0]
    assert received["keyed_second"][0] is keyed_matrix
    np.testing.assert_array_equal(keyed_matrix, reverse_score(presented, [0, 3], 1, 5))


def _unused_top_category(seed: int = 0) -> np.ndarray:
    """Six five-category items on which nobody chose category 5."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(120, 1)) + rng.normal(scale=0.8, size=(120, 6))
    presented = np.clip(np.rint(2.5 + latent), 1, 4)
    presented[:, [1, 4]] = 5 - presented[:, [1, 4]]
    return presented


@pytest.mark.parametrize(
    ("bounds", "scale"),
    [
        ({}, (1, 5)),
        ({"scale_min": 1}, (1, 5)),
        ({"scale_max": 5}, (1, 5)),
        ({"scale_min": 0}, (0, 4)),
        ({"scale_min": 1, "scale_max": 5}, (1, 5)),
    ],
    ids=["observed-minimum", "scale-min", "scale-max", "zero-based", "both-bounds"],
)
@pytest.mark.parametrize("workers", [1, 3])
def test_person_fit_ncat_sets_the_scale_person_fit_items_are_reflected_on(
    bounds: dict[str, int], scale: tuple[int, int], workers: int
) -> None:
    presented = _unused_top_category()
    assert (presented.min(), presented.max()) == (1, 4)
    options = IndexOptions(person_fit_ncat=5, reverse_keyed_items=[1, 4], **bounds)
    indices = ["gpoly", "guttman", "u3poly_fit", "lz"]

    result = screen(presented, indices=indices, options=options, workers=workers)

    assert result["errors"] == {}
    declared = reverse_score(presented, [1, 4], *scale)
    np.testing.assert_array_equal(result["scores"]["gpoly"], gpoly(declared, ncat=5, **bounds))
    np.testing.assert_array_equal(
        result["scores"]["u3poly_fit"], u3poly(declared, ncat=5, **bounds)
    )
    # The other keyed indices keep scale_min and scale_max, inferred when omitted.
    observed = reverse_score(presented, [1, 4], bounds.get("scale_min"), bounds.get("scale_max"))
    expected = screen(observed, indices=["guttman", "lz"])["scores"]
    for name in ["guttman", "lz"]:
        np.testing.assert_array_equal(result["scores"][name], expected[name], err_msg=name)


def test_reflecting_on_the_observed_scale_would_change_person_fit_scores() -> None:
    presented = _unused_top_category()
    options = IndexOptions(person_fit_ncat=5, scale_min=1, reverse_keyed_items=[1, 4])

    result = screen(presented, indices=["gpoly", "u3poly_fit"], options=options)

    observed = reverse_score(presented, [1, 4], 1, None)
    declared = reverse_score(presented, [1, 4], 1, 5)
    np.testing.assert_array_equal(declared[:, [1, 4]], observed[:, [1, 4]] + 1)
    for name, scorer in [("gpoly", gpoly), ("u3poly_fit", u3poly)]:
        stale = scorer(observed, ncat=5, scale_min=1)
        assert not np.array_equal(result["scores"][name], stale, equal_nan=True), name


def test_ht_keeps_the_observed_scale_when_person_fit_ncat_is_set() -> None:
    binary = (np.random.default_rng(3).random((60, 6)) < 0.6).astype(float)
    options = IndexOptions(person_fit_ncat=3, scale_min=0, reverse_keyed_items=[0, 3])

    result = screen(binary, indices=["ht", "gpoly"], options=options)

    assert result["errors"] == {}
    np.testing.assert_array_equal(result["scores"]["ht"], ht(reverse_score(binary, [0, 3])))
    np.testing.assert_array_equal(
        result["scores"]["gpoly"],
        gpoly(reverse_score(binary, [0, 3], 0, 2), ncat=3, scale_min=0),
    )


@pytest.mark.parametrize(
    ("bounds", "copies"),
    [({"scale_min": 1}, 2), ({"scale_max": 5}, 2), ({"scale_min": 1, "scale_max": 5}, 1)],
)
@pytest.mark.parametrize("workers", [1, 2])
def test_each_keyed_scale_is_recoded_once(
    bounds: dict[str, int], copies: int, workers: int
) -> None:
    presented = _unused_top_category()
    options = IndexOptions(person_fit_ncat=5, reverse_keyed_items=[1, 4], **bounds)
    indices = ["gpoly", "guttman", "u3poly_fit", "lz"]

    with patch.object(ier._registry, "reverse_score", wraps=reverse_score) as recode:
        scores, errors = score_registered_indices(presented, indices, options, workers=workers)

    assert errors == {}
    assert list(scores) == indices
    assert recode.call_count == copies


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            IndexOptions(person_fit_ncat=3, scale_min=1, scale_max=5),
            r"ncat must equal scale_max - scale_min \+ 1",
        ),
        (IndexOptions(person_fit_ncat=5, scale_min=0.5), "scale_min must be a finite integer"),
        (IndexOptions(person_fit_ncat=1), "ncat must be an integer of at least 2"),
    ],
)
def test_invalid_person_fit_scales_fail_with_the_scorer_message(
    options: IndexOptions, message: str
) -> None:
    presented = _unused_top_category()
    keyed = replace(options, reverse_keyed_items=[1, 4])
    indices = ["gpoly", "guttman", "u3poly_fit"]

    result = screen(presented, indices=indices, options=keyed)

    unkeyed = screen(presented, indices=["gpoly", "u3poly_fit"], options=options)
    assert result["errors"] == unkeyed["errors"]
    assert list(result["errors"]) == ["gpoly", "u3poly_fit"]
    assert all(re.fullmatch(message, error) for error in result["errors"].values())
    assert result["indices_used"] == ["guttman"]
    with pytest.raises(ValueError, match=f"index 'gpoly' failed: {message}") as error:
        screen(presented, indices=["gpoly"], options=keyed, strict=True)
    assert isinstance(error.value.__cause__, ValueError)


def test_person_fit_ncat_without_responses_recodes_nothing() -> None:
    missing = np.full((4, 3), np.nan)
    options = IndexOptions(person_fit_ncat=4, reverse_keyed_items=[0])

    scores, errors = score_registered_indices(missing, ["gpoly", "u3poly_fit"], options)

    assert errors == {}
    for name in ["gpoly", "u3poly_fit"]:
        assert np.isnan(scores[name]).all(), name


@pytest.mark.parametrize(
    ("indices", "reverse_keyed_items"),
    [
        (None, None),
        (["guttman", "lz", "evenodd", "individual_reliability"], None),
        (["irv", "longstring", "mahad", "psychsyn", "acquiescence"], [1, 4]),
    ],
)
def test_no_recoded_copy_without_items_or_a_selected_keyed_index(
    indices: list[str] | None, reverse_keyed_items: list[int] | None
) -> None:
    presented = _presented_responses()
    options = replace(_OPTIONS, reverse_keyed_items=reverse_keyed_items)

    with patch.object(
        ier._registry, "reverse_score", side_effect=AssertionError("unexpected copy")
    ) as recode:
        screen(presented, indices=indices, options=options, workers=2)
        composite(presented, indices=["irv", "longstring"], options=options)

    recode.assert_not_called()


_INVALID_KEYING = [
    (IndexOptions(reverse_keyed_items=[18]), "item index 18 out of bounds"),
    (IndexOptions(reverse_keyed_items=[1, 1]), "items cannot contain duplicates"),
    (IndexOptions(reverse_keyed_items=[]), "items cannot be empty"),
    (IndexOptions(reverse_keyed_items=[1.5]), "must contain integer column indices"),  # type: ignore[list-item]
    (IndexOptions(reverse_keyed_items=4), "not iterable"),  # type: ignore[arg-type]
    (
        IndexOptions(scale_min=1, scale_max=4, reverse_keyed_items=[1]),
        "within scale_min and scale_max",
    ),
    (IndexOptions(scale_min=5, scale_max=1, reverse_keyed_items=[1]), "greater than or equal"),
]


@pytest.mark.parametrize(("options", "message"), _INVALID_KEYING)
@pytest.mark.parametrize("workers", [1, 2])
def test_invalid_keying_soft_fails_only_keyed_indices(
    options: IndexOptions, message: str, workers: int
) -> None:
    presented = _presented_responses()
    indices = ["irv", "guttman", "longstring", "lz"]

    result = screen(presented, indices=indices, options=options, min_flags=1, workers=workers)

    assert result["indices_used"] == ["irv", "longstring"]
    assert list(result["errors"]) == ["guttman", "lz"]
    for error in result["errors"].values():
        assert error.startswith("reverse_keyed_items could not be applied: ")
        assert message in error
    _, diagnostics = composite(presented, indices=indices, options=options, return_diagnostics=True)
    assert diagnostics == result["errors"]


@pytest.mark.parametrize(("options", "message"), _INVALID_KEYING)
def test_strict_mode_reports_invalid_keying_for_the_first_keyed_index(
    options: IndexOptions, message: str
) -> None:
    presented = _presented_responses()

    with pytest.raises(ValueError, match="index 'guttman' failed: reverse_keyed_items") as error:
        screen(presented, indices=["irv", "guttman", "lz"], options=options, strict=True)

    assert message in str(error.value)
    assert isinstance(error.value.__cause__, (ValueError, TypeError))
    with pytest.raises(ValueError, match="index 'lz' failed: reverse_keyed_items"):
        composite(presented, indices=["irv", "lz"], options=options, strict=True, workers=2)


def test_missing_required_options_are_reported_before_keying_failures() -> None:
    presented = _presented_responses()
    options = IndexOptions(reverse_keyed_items=[99])

    result = screen(presented, indices=["evenodd", "guttman"], options=options)

    assert (
        result["errors"]["evenodd"] == "evenodd_factors must be provided when using evenodd index"
    )
    assert "out of bounds" in result["errors"]["guttman"]


def test_catalog_reports_each_index_input() -> None:
    catalog = index_catalog()

    for name, spec in INDEX_REGISTRY.items():
        assert catalog[name]["uses_keyed_responses"] is spec.keyed_input
    assert {"evenodd", "individual_reliability", "guttman", "lz"} <= set(_KEYED)


def _documented_names(text: str, pattern: str, *, marker: str = "`") -> set[str]:
    """Return the code-formatted index names in the one passage the pattern captures."""
    matches = re.findall(pattern, " ".join(text.split()))
    assert len(matches) == 1, pattern
    return set(re.findall(rf"{marker}([a-z0-9_]+){marker}", matches[0]))


def test_docs_and_docstring_name_every_keyed_index() -> None:
    keyed = set(_KEYED)
    screening = (_ROOT / "docs" / "workflows" / "screening.md").read_text(encoding="utf-8")
    indices = (_ROOT / "docs" / "indices.md").read_text(encoding="utf-8")
    architecture = (_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")

    assert _documented_names(screening, r"The keyed indices (.+?) read one recoded copy") == keyed
    assert _documented_names(indices, r"scale_max\)` before (.+?), which assume") == keyed
    assert _documented_names(architecture, r"`keyed_input=True` \((.+?)\) assume") == keyed
    table_rows = [line.split("|") for line in indices.splitlines() if line.startswith("| ")]
    recoded = " ".join(row[2] for row in table_rows if row[1].strip() == "Recoded")
    assert set(re.findall(r"`([a-z0-9_]+)`", recoded)) == keyed
    docstring = reverse_score.__doc__ or ""
    assert _documented_names(docstring, r"same direction: (.+?), the indices", marker="``") == keyed


def _write_survey(root: Path) -> tuple[Path, np.ndarray]:
    presented = np.nan_to_num(_presented_responses(), nan=3.0)
    source = root / "survey.csv"
    header = ",".join(f"q{column}" for column in range(presented.shape[1]))
    rows = [",".join(str(int(value)) for value in row) for row in presented]
    source.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return source, presented


def _cli_json(arguments: list[str]) -> dict[str, Any]:
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        code = main([*arguments, "--format", "json"])
    assert code == 0
    payload: dict[str, Any] = json.loads(stdout.getvalue())
    return payload


def test_cli_option_builds_index_options() -> None:
    args = _build_parser().parse_args(["screen", "survey.csv", "--reverse-keyed-items", "1, 4,7"])
    assert _options_from_args(args).reverse_keyed_items == [1, 4, 7]
    args = _build_parser().parse_args(["composite", "survey.csv"])
    assert _options_from_args(args).reverse_keyed_items is None
    args = _build_parser().parse_args(["screen", "survey.csv", "--reverse-keyed-items", " , "])
    with pytest.raises(ValueError, match="--reverse-keyed-items must include at least one item"):
        _options_from_args(args)


def test_cli_screen_and_composite_recode_keyed_indices(tmp_path: Path) -> None:
    source, presented = _write_survey(tmp_path)
    items = ",".join(map(str, _REVERSED))
    common = ["--scale-min", "1", "--scale-max", "5", "--reverse-keyed-items", items]
    recoded = reverse_score(presented, _REVERSED, 1, 5)

    screened = _cli_json(["screen", str(source), "--indices", "guttman", "longstring", *common])
    assert (
        screened["scores"]["guttman"]
        == screen(recoded, indices=["guttman"])["scores"]["guttman"].tolist()
    )
    assert (
        screened["scores"]["longstring"]
        == screen(presented, indices=["longstring"])["scores"]["longstring"].tolist()
    )

    combined = _cli_json(
        ["composite", str(source), "--indices", "lz", "irv", "--include-components", *common]
    )
    np.testing.assert_array_equal(
        combined["component_scores"]["lz"], screen(recoded, indices=["lz"])["scores"]["lz"]
    )
    np.testing.assert_array_equal(
        combined["component_scores"]["irv"], screen(presented, indices=["irv"])["scores"]["irv"]
    )


def test_cli_config_file_supplies_reverse_keyed_items(tmp_path: Path) -> None:
    source, presented = _write_survey(tmp_path)
    config = tmp_path / "ier.toml"
    config.write_text(
        "[screen]\nscale_min = 1\nscale_max = 5\nreverse_keyed_items = [1, 4, 7]\n",
        encoding="utf-8",
    )

    payload = _cli_json(["screen", str(source), "--indices", "guttman", "--config", str(config)])

    recoded = reverse_score(presented, [1, 4, 7], 1, 5)
    expected = screen(recoded, indices=["guttman"])["scores"]["guttman"]
    assert payload["scores"]["guttman"] == expected.tolist()


def test_cli_reports_invalid_keying_as_a_skipped_index_or_strict_error(tmp_path: Path) -> None:
    source, _ = _write_survey(tmp_path)
    arguments = ["screen", str(source), "--indices", "irv", "guttman"]
    arguments += ["--reverse-keyed-items", "40"]

    stderr = StringIO()
    with patch("sys.stderr", stderr):
        payload = _cli_json(arguments)
    assert list(payload["scores"]) == ["irv"]
    assert "warning: index 'guttman' was skipped: reverse_keyed_items could not be applied" in (
        stderr.getvalue()
    )

    stderr = StringIO()
    with patch("sys.stderr", stderr):
        code = main([*arguments, "--strict"])
    assert code == 1
    assert "error: index 'guttman' failed: reverse_keyed_items" in stderr.getvalue()
    assert "item index 40 out of bounds" in stderr.getvalue()
