"""Tests for public index registry discovery."""

import re
import threading
import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from ier import (
    composite,
    composite_summary,
    index_catalog,
    irv,
    longstring_scores,
    mad,
    psychant,
    screen,
)
from ier._registry import (
    INDEX_REGISTRY,
    IndexOptions,
    IndexSpec,
    numeric_override,
    resolve_index_overrides,
    score_registered_indices,
    validate_index_names,
)
from ier.mad import run_mad_index


class TestIndexCatalog(unittest.TestCase):
    def test_catalog_describes_all_registered_indices(self) -> None:
        catalog = index_catalog()

        self.assertEqual(len(catalog), 26)
        self.assertEqual(
            catalog["irv"],
            {
                "flag_direction": "low",
                "flag_mode": "percentile",
                "default_screen": True,
                "default_composite": True,
                "composite_enabled": True,
                "required_options": (),
                "alternative_options": (),
                "uses_keyed_responses": False,
            },
        )
        self.assertEqual(catalog["onset"]["flag_mode"], "present")
        self.assertFalse(catalog["onset"]["composite_enabled"])
        self.assertEqual(
            catalog["missing_rate"],
            {
                "flag_direction": "high",
                "flag_mode": "percentile",
                "default_screen": False,
                "default_composite": False,
                "composite_enabled": True,
                "required_options": (),
                "alternative_options": (),
                "uses_keyed_responses": False,
            },
        )
        self.assertEqual(catalog["evenodd"]["required_options"], ("evenodd_factors",))
        self.assertEqual(catalog["evenodd"]["alternative_options"], ())
        # Either answer form configures infrequency, so neither is strictly required.
        self.assertEqual(catalog["infrequency"]["required_options"], ("infrequency_item_indices",))
        self.assertEqual(
            catalog["infrequency"]["alternative_options"],
            (("infrequency_expected_responses", "infrequency_acceptable_ranges"),),
        )
        keyed = {name for name, metadata in catalog.items() if metadata["uses_keyed_responses"]}
        self.assertLessEqual({"evenodd", "individual_reliability", "guttman", "lz"}, keyed)
        presented = {
            "irv",
            "longstring",
            "longstring_pattern",
            "avgstr",
            "markov",
            "autocorrelation",
            "onset",
            "mahad",
            "psychsyn",
            "psychant",
            "person_total",
            "u3_poly",
            "midpoint",
            "acquiescence",
            "mad",
            "semantic_syn",
            "semantic_ant",
            "infrequency",
            "missing_rate",
        }
        self.assertFalse(keyed & presented)
        self.assertEqual(
            catalog["avgstr"],
            {
                "flag_direction": "high",
                "flag_mode": "percentile",
                "default_screen": False,
                "default_composite": False,
                "composite_enabled": True,
                "required_options": (),
                "alternative_options": (),
                "uses_keyed_responses": False,
            },
        )

    def test_catalog_returns_independent_metadata(self) -> None:
        catalog = index_catalog()
        catalog["irv"]["default_screen"] = False

        self.assertTrue(index_catalog()["irv"]["default_screen"])


class TestParallelIndexScoring(unittest.TestCase):
    def test_workers_run_concurrently_and_results_keep_selection_order(self) -> None:
        rendezvous = threading.Barrier(2)
        second_completed = threading.Event()

        def first_score(x: np.ndarray, options: IndexOptions) -> np.ndarray:
            del options
            rendezvous.wait(timeout=2)
            if not second_completed.wait(timeout=2):
                raise RuntimeError("second scorer did not run concurrently")
            return np.full(len(x), 1.0)

        def second_score(x: np.ndarray, options: IndexOptions) -> np.ndarray:
            del options
            rendezvous.wait(timeout=2)
            second_completed.set()
            return np.full(len(x), 2.0)

        additions = {
            "parallel_first": IndexSpec("parallel_first", first_score, "high"),
            "parallel_second": IndexSpec("parallel_second", second_score, "high"),
        }
        with patch.dict(INDEX_REGISTRY, additions):
            scores, errors = score_registered_indices(
                np.zeros((3, 2)),
                list(additions),
                IndexOptions(),
                workers=2,
            )

        self.assertEqual(list(scores), list(additions))
        np.testing.assert_array_equal(scores["parallel_first"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(scores["parallel_second"], [2.0, 2.0, 2.0])
        self.assertEqual(errors, {})

    def test_parallel_failures_retain_selection_order_and_strict_context(self) -> None:
        def broken_score(x: np.ndarray, options: IndexOptions) -> np.ndarray:
            del x, options
            raise RuntimeError("calculation failed")

        additions = {
            "parallel_missing": IndexSpec(
                "parallel_missing",
                lambda x, options: np.zeros(len(x)),
                "high",
                required_error=lambda options: "configuration missing",
            ),
            "parallel_broken": IndexSpec("parallel_broken", broken_score, "high"),
            "parallel_valid": IndexSpec(
                "parallel_valid",
                lambda x, options: np.arange(len(x), dtype=float),
                "high",
            ),
        }
        with patch.dict(INDEX_REGISTRY, additions):
            scores, errors = score_registered_indices(
                np.zeros((3, 2)),
                list(additions),
                IndexOptions(),
                workers=3,
            )
            with self.assertRaisesRegex(
                ValueError,
                "index 'parallel_missing' failed: configuration missing",
            ):
                score_registered_indices(
                    np.zeros((3, 2)),
                    list(additions),
                    IndexOptions(),
                    strict=True,
                    workers=3,
                )

        self.assertEqual(list(errors), ["parallel_missing", "parallel_broken"])
        self.assertEqual(list(scores), ["parallel_valid"])

    def test_workers_must_be_a_positive_integer(self) -> None:
        for workers in [0, -1, 1.5, True]:
            with (
                self.subTest(workers=workers),
                self.assertRaisesRegex(
                    ValueError,
                    "workers must be a positive integer",
                ),
            ):
                score_registered_indices(
                    np.zeros((2, 2)),
                    ["irv"],
                    IndexOptions(),
                    workers=workers,  # type: ignore[arg-type]
                )

    def test_duplicate_indices_are_rejected_before_scoring(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate index 'irv'"):
            validate_index_names(["irv", "longstring", "irv"])

        with self.assertRaisesRegex(ValueError, "duplicate index 'irv'"):
            score_registered_indices(
                np.zeros((2, 2)),
                ["irv", "irv"],
                IndexOptions(),
                workers=2,
            )


class TestRegistryOptionWiring(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2026)
        self.data = rng.integers(1, 6, size=(40, 12)).astype(float)
        self.data[:5, :8] = 3.0
        self.data[7, ::3] = np.nan

    def test_irv_split_options_reach_the_scorer(self) -> None:
        cases = [
            (IndexOptions(), irv(self.data)),
            (IndexOptions(irv_num_split=2), irv(self.data, split=True, num_split=2)),
            (IndexOptions(irv_num_split=5), irv(self.data, split=True, num_split=5)),
            (
                IndexOptions(irv_split_points=[0, 4, 12]),
                irv(self.data, split=True, split_points=[0, 4, 12]),
            ),
        ]
        for options, expected in cases:
            with self.subTest(options=options):
                result = screen(self.data, indices=["irv"], options=options)
                np.testing.assert_array_equal(result["scores"]["irv"], expected)
        np.testing.assert_array_equal(
            composite(
                self.data, indices=["irv"], standardize=False, options=IndexOptions(irv_num_split=2)
            ),
            -irv(self.data, split=True, num_split=2),
        )

    def test_invalid_irv_split_options_are_reported(self) -> None:
        count_message = "num_split must be a positive integer"
        for options, message in [
            (IndexOptions(irv_num_split=0), count_message),
            # True and 1.0 equal 1 but are not section counts; they must not
            # silently select the unsplit computation.
            (IndexOptions(irv_num_split=True), count_message),
            (IndexOptions(irv_num_split=cast("Any", 1.0)), count_message),
            (IndexOptions(irv_num_split=cast("Any", 2.0)), count_message),
            # The count is validated even when split points select the sections.
            (
                IndexOptions(
                    irv_num_split=cast("Any", np.bool_(True)), irv_split_points=[0, 4, 12]
                ),
                count_message,
            ),
            (
                IndexOptions(irv_split_points=[0, 5]),
                "last split point must be 12 (number of columns)",
            ),
        ]:
            with self.subTest(options=options):
                result = screen(self.data, indices=["irv", "longstring"], options=options)
                self.assertEqual(result["errors"], {"irv": message})
                with self.assertRaisesRegex(
                    ValueError, re.escape(f"index 'irv' failed: {message}")
                ):
                    screen(self.data, indices=["irv"], options=options, strict=True)

    def test_avgstr_scores_average_runs_and_joins_composites(self) -> None:
        expected = longstring_scores(self.data, avg=True)
        result = screen(self.data, indices=["avgstr", "longstring"])
        np.testing.assert_array_equal(result["scores"]["avgstr"], expected)
        # Only the respondents with a long straightlined run have high averages.
        flagged = set(np.flatnonzero(result["flags"]["avgstr"]).tolist())
        self.assertTrue(flagged)
        self.assertLessEqual(flagged, set(range(5)))
        details = composite_summary(self.data, indices=["avgstr"], standardize=False)
        np.testing.assert_array_equal(details["indices"]["avgstr"], expected)
        np.testing.assert_array_equal(details["composite"], expected)
        strict = screen(self.data, indices=["avgstr"], options=IndexOptions(na_rm=False))
        self.assertIn("data contains missing values", strict["errors"]["avgstr"])

    def test_infrequency_accepts_ranges_or_expected_responses(self) -> None:
        data = np.array([[1, 5, 4], [2, 4, 7], [4, 3, 2]])
        ranges = IndexOptions(
            infrequency_item_indices=[0, 2], infrequency_acceptable_ranges=[(1, 2), (4, 7)]
        )
        result = screen(data, indices=["infrequency"], options=ranges)
        np.testing.assert_array_equal(result["scores"]["infrequency"], [0, 0, 2])
        expected = IndexOptions(
            infrequency_item_indices=[0, 2], infrequency_expected_responses=[1, 4]
        )
        result = screen(data, indices=["infrequency"], options=expected)
        np.testing.assert_array_equal(result["scores"]["infrequency"], [0, 2, 2])

        message = (
            "infrequency_item_indices and either infrequency_expected_responses or "
            "infrequency_acceptable_ranges must be provided when using infrequency index"
        )
        for options in [
            IndexOptions(),
            IndexOptions(infrequency_item_indices=[0]),
            IndexOptions(infrequency_acceptable_ranges=[(1, 2)]),
        ]:
            with self.subTest(options=options):
                result = screen(data, indices=["infrequency"], options=options)
                self.assertEqual(result["errors"], {"infrequency": message})
                with self.assertRaisesRegex(ValueError, "either infrequency_expected_responses"):
                    screen(data, indices=["infrequency"], options=options, strict=True)
        both = IndexOptions(
            infrequency_item_indices=[0],
            infrequency_expected_responses=[1],
            infrequency_acceptable_ranges=[(1, 2)],
        )
        result = screen(data, indices=["infrequency"], options=both)
        self.assertIn("exactly one of expected_responses", result["errors"]["infrequency"])

    def test_invalid_pattern_length_is_reported_without_aborting_screening(self) -> None:
        result = screen(
            self.data,
            indices=["longstring_pattern", "longstring"],
            options=IndexOptions(longstring_max_pattern_length=1),
        )
        self.assertEqual(
            result["errors"],
            {"longstring_pattern": "max_pattern_length must be an integer of at least 2"},
        )
        self.assertIn("longstring", result["scores"])


_CONFIGURED = IndexOptions(
    evenodd_factors=[12, 12],
    mad_positive_items=[0, 2, 4],
    mad_negative_items=[1, 3, 5],
    semantic_item_pairs=[(0, 1), (2, 3)],
    infrequency_item_indices=[0, 5],
    infrequency_expected_responses=[3.0, 3.0],
    reliability_n_splits=5,
    reliability_random_seed=0,
    onset_window_size=4,
    onset_min_items=8,
)


@pytest.fixture(scope="module")
def survey() -> np.ndarray:
    rng = np.random.default_rng(20261005)
    data = rng.integers(1, 6, size=(40, 24)).astype(float)
    data[:4] = 3.0
    data[4:8, 12:] = 3.0
    data[9, ::5] = np.nan
    return data


@pytest.mark.parametrize("name", list(INDEX_REGISTRY))
def test_every_registered_scorer_runs_through_screen(name: str, survey: np.ndarray) -> None:
    # Ht is defined for dichotomous items only; score agreement with the upper categories.
    data = np.where(np.isnan(survey), np.nan, survey > 3) if name == "ht" else survey
    result = screen(data, indices=[name], options=_CONFIGURED)

    assert result["errors"] == {}
    assert result["indices_used"] == [name]
    assert result["scores"][name].shape == (len(survey),)


def test_psychant_registry_scorer_reaches_the_public_function(survey: np.ndarray) -> None:
    options = IndexOptions(psychant_critval=-0.1, psychsyn_item_correlations="pairwise")
    result = screen(survey, indices=["psychant"], options=options)
    expected = psychant(survey, critval=-0.1, item_correlations="pairwise")

    np.testing.assert_array_equal(result["scores"]["psychant"], expected)


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("evenodd", IndexOptions(), "evenodd_factors must be provided when using evenodd index"),
        (
            "mad",
            IndexOptions(mad_positive_items=[0, 2]),
            "mad_positive_items and mad_negative_items must be provided when using mad index",
        ),
        (
            "semantic_syn",
            IndexOptions(),
            "semantic_item_pairs must be provided when using semantic_syn index",
        ),
        (
            "semantic_ant",
            IndexOptions(),
            "semantic_item_pairs must be provided when using semantic_ant index",
        ),
        (
            "infrequency",
            IndexOptions(infrequency_expected_responses=[3.0]),
            "infrequency_item_indices and either infrequency_expected_responses or "
            "infrequency_acceptable_ranges must be provided when using infrequency index",
        ),
    ],
)
def test_missing_required_options_soft_fail_with_metadata_messages(
    name: str, options: IndexOptions, message: str, survey: np.ndarray
) -> None:
    result = screen(survey, indices=[name, "irv"], options=options)

    assert result["errors"] == {name: message}
    assert result["indices_used"] == ["irv"]
    with pytest.raises(ValueError, match=re.escape(f"index '{name}' failed: {message}")):
        screen(survey, indices=[name], options=options, strict=True)


def test_custom_specs_derive_required_option_messages() -> None:
    addition = {
        "configured": IndexSpec(
            "configured",
            lambda x, options: np.zeros(len(x)),
            "high",
            required_options=("scale_min", "scale_max"),
        )
    }
    with patch.dict(INDEX_REGISTRY, addition):
        _, missing = score_registered_indices(np.zeros((2, 2)), ["configured"], IndexOptions())
        scores, errors = score_registered_indices(
            np.zeros((2, 2)), ["configured"], IndexOptions(scale_min=1, scale_max=5)
        )

    assert missing == {
        "configured": "scale_min and scale_max must be provided when using configured index"
    }
    assert errors == {}
    np.testing.assert_array_equal(scores["configured"], [0.0, 0.0])


def test_custom_specs_require_one_option_from_each_alternative_group() -> None:
    addition = {
        "alternatives": IndexSpec(
            "alternatives",
            lambda x, options: np.zeros(len(x)),
            "high",
            required_options=("scale_min",),
            alternative_options=(("mad_scale_min", "mad_scale_max"), ("reliability_factors",)),
        )
    }
    message = (
        "scale_min and either mad_scale_min or mad_scale_max and reliability_factors "
        "must be provided when using alternatives index"
    )
    configured = [
        IndexOptions(scale_min=1, mad_scale_min=1, reliability_factors=[2]),
        IndexOptions(scale_min=1, mad_scale_max=5, reliability_factors=[2]),
        IndexOptions(scale_min=1, mad_scale_min=1, mad_scale_max=5, reliability_factors=[2]),
    ]
    incomplete = [
        IndexOptions(),
        IndexOptions(mad_scale_min=1, reliability_factors=[2]),
        IndexOptions(scale_min=1, reliability_factors=[2]),
        IndexOptions(scale_min=1, mad_scale_max=5),
    ]
    with patch.dict(INDEX_REGISTRY, addition):
        assert index_catalog()["alternatives"]["alternative_options"] == (
            ("mad_scale_min", "mad_scale_max"),
            ("reliability_factors",),
        )
        for options in configured:
            scores, errors = score_registered_indices(np.zeros((2, 2)), ["alternatives"], options)
            assert errors == {}
            np.testing.assert_array_equal(scores["alternatives"], [0.0, 0.0])
        for options in incomplete:
            _, errors = score_registered_indices(np.zeros((2, 2)), ["alternatives"], options)
            assert errors == {"alternatives": message}


def test_infrequency_has_no_custom_required_error() -> None:
    # The catalog metadata alone defines the enforced configuration.
    spec = INDEX_REGISTRY["infrequency"]
    assert spec.required_error is None
    catalog = index_catalog()["infrequency"]
    assert catalog["required_options"] == spec.required_options
    assert catalog["alternative_options"] == spec.alternative_options


def test_run_mad_index_remains_a_validated_public_helper(survey: np.ndarray) -> None:
    np.testing.assert_array_equal(
        run_mad_index(survey, [0, 2], [1, 3], scale_min=1, scale_max=5, na_rm=True),
        mad(survey, positive_items=[0, 2], negative_items=[1, 3], scale_min=1, scale_max=5),
    )
    with pytest.raises(ValueError, match="mad_positive_items and mad_negative_items"):
        run_mad_index(survey, None, [1, 3], scale_min=None, scale_max=None, na_rm=True)


@pytest.mark.parametrize("indices", ["irv", "longstring"])
def test_string_index_selections_are_rejected(indices: str, survey: np.ndarray) -> None:
    message = "indices must be a list of index names, not a string"
    with pytest.raises(ValueError, match=message):
        validate_index_names(indices)
    with pytest.raises(ValueError, match=message):
        screen(survey, indices=cast("Any", indices))
    with pytest.raises(ValueError, match=message):
        composite(survey, indices=cast("Any", indices))


@pytest.mark.parametrize("values", [[("irv", 1.0)], (("irv", 1.0),), "irv", 1.0])
def test_non_mapping_overrides_raise_type_errors(values: object) -> None:
    with pytest.raises(
        TypeError, match="^levels must be a mapping of registered index names to numbers$"
    ):
        resolve_index_overrides(
            cast("Any", values), ["irv"], label="level", convert=numeric_override("level")
        )


def test_override_resolution_keeps_mapping_order_and_rejections() -> None:
    convert = numeric_override("level", "a small number", lambda value: abs(value) < 10)

    resolved = resolve_index_overrides(
        {"longstring": "2", "irv": -1}, ["irv", "longstring"], label="level", convert=convert
    )

    assert list(resolved.items()) == [("longstring", 2.0), ("irv", -1.0)]
    assert resolve_index_overrides(None, ["irv"], label="level", convert=convert) == {}
    for values, message in [
        ({"unknown": 1.0}, "unknown level index: unknown"),
        ({"mahad": 1.0}, "level index is not selected: mahad"),
        ({"irv": 10.0}, "level for irv must be a small number"),
        ({"irv": True}, "level for irv must be a small number"),
        ({"irv": None}, "level for irv must be a small number"),
        ({"irv": 10**400}, "level for irv must be a small number"),
        ({"irv": 1.0}, "irv is rejected"),
    ]:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            resolve_index_overrides(
                cast("Any", values),
                ["irv"],
                label="level",
                convert=convert,
                accepts=lambda name, message=message: (
                    f"{name} is rejected" if "rejected" in message else None
                ),
            )
