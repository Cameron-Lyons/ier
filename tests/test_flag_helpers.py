"""Standalone flag helpers agree with screen() decisions and registry directions."""

import inspect
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import ier
from ier import screen
from ier._flagging import threshold_flags
from ier._registry import INDEX_REGISTRY

FlagHelper = Callable[..., tuple[np.ndarray, np.ndarray]]

# Registry name, helper, and the helper's defining module.
_HELPERS: list[tuple[str, FlagHelper, str]] = [
    ("psychsyn", ier.psychsyn_flag, "ier.psychsyn"),
    ("psychant", ier.psychant_flag, "ier.psychsyn"),
    ("person_total", ier.person_total_flag, "ier.person_total"),
    ("u3_poly", ier.u3_poly_flag, "ier.u3_poly"),
    ("midpoint", ier.midpoint_responding_flag, "ier.u3_poly"),
]
_IDS = [name for name, _, _ in _HELPERS]


def _responses(seed: int = 20261005) -> np.ndarray:
    """Factor-structured Likert responses with antonym items and some careless rows."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(300, 1))
    data = np.clip(np.rint(3 + latent + rng.normal(scale=0.5, size=(300, 12))), 1, 5)
    data[:, 6:] = 6 - data[:, 6:]
    data[:20] = rng.integers(1, 6, size=(20, 12))
    data[20:25] = 3
    data[25:30] = rng.choice([1, 5], size=(5, 12))
    return data


@pytest.mark.parametrize(("name", "helper", "module"), _HELPERS, ids=_IDS)
def test_helpers_use_the_registry_flag_direction(
    name: str, helper: FlagHelper, module: str
) -> None:
    with patch(f"{module}.threshold_flags", wraps=threshold_flags) as flagged:
        helper(_responses())
    assert flagged.call_count == 1
    assert flagged.call_args.kwargs["direction"] == INDEX_REGISTRY[name].flag_direction


@pytest.mark.parametrize(("name", "helper", "module"), _HELPERS, ids=_IDS)
def test_default_percentile_flags_match_screen(name: str, helper: FlagHelper, module: str) -> None:
    data = _responses()
    scores, flags = helper(data)
    result = screen(data, indices=[name])
    np.testing.assert_array_equal(scores, result["scores"][name])
    np.testing.assert_array_equal(flags, result["flags"][name])
    assert 0 < np.count_nonzero(flags) < len(data)


@pytest.mark.parametrize(("name", "helper", "module"), _HELPERS, ids=_IDS)
def test_fixed_thresholds_include_ties_and_skip_unavailable_scores(
    name: str, helper: FlagHelper, module: str
) -> None:
    data = _responses()
    data[0] = np.nan
    scores, flags = helper(data, threshold=0.5)
    assert np.isnan(scores[0]) and not flags[0]
    expected = scores <= 0.5 if INDEX_REGISTRY[name].flag_direction == "low" else scores >= 0.5
    np.testing.assert_array_equal(flags, expected)


@pytest.mark.parametrize(
    ("helper", "keywords", "scorer"),
    [
        (
            ier.psychsyn_flag,
            {"critval": 0.4, "item_correlations": "pairwise"},
            lambda x: ier.psychsyn(x, critval=0.4, item_correlations="pairwise"),
        ),
        (
            ier.psychant_flag,
            {"critval": -0.4, "resample_na": True, "random_seed": 3},
            lambda x: ier.psychant(x, critval=-0.4, resample_na=True, random_seed=3),
        ),
        (
            ier.person_total_flag,
            {"na_rm": False},
            lambda x: ier.person_total(x, na_rm=False),
        ),
        (
            ier.u3_poly_flag,
            {"scale_min": 0, "scale_max": 5},
            lambda x: ier.u3_poly(x, scale_min=0, scale_max=5),
        ),
        (
            ier.midpoint_responding_flag,
            {"scale_min": 1, "scale_max": 6, "tolerance": 0.5},
            lambda x: ier.midpoint_responding(x, scale_min=1, scale_max=6, tolerance=0.5),
        ),
    ],
    ids=_IDS,
)
def test_scorer_options_pass_through(
    helper: FlagHelper, keywords: dict[str, Any], scorer: Callable[[np.ndarray], np.ndarray]
) -> None:
    data = _responses()
    data[np.random.default_rng(5).random(data.shape) < 0.02] = np.nan
    scores, flags = helper(data, percentile=10.0, **keywords)
    np.testing.assert_array_equal(scores, scorer(data))
    assert flags.dtype == bool and flags.shape == scores.shape


@pytest.mark.parametrize(
    ("helper", "scorer"),
    [
        (ier.u3_poly_flag, ier.u3_poly),
        (ier.midpoint_responding_flag, ier.midpoint_responding),
    ],
    ids=["u3_poly", "midpoint"],
)
def test_response_style_helpers_lead_with_scorer_parameters(
    helper: FlagHelper, scorer: Callable[..., np.ndarray]
) -> None:
    # Like acquiescence_flag(): scorer options first, in the scorer's order.
    helper_params = list(inspect.signature(helper).parameters.values())
    scorer_params = list(inspect.signature(scorer).parameters.values())
    assert [param.name for param in helper_params] == [
        *(param.name for param in scorer_params),
        "threshold",
        "percentile",
    ]
    for helper_param, scorer_param in zip(helper_params, scorer_params, strict=False):
        assert helper_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert helper_param.default == scorer_param.default


def test_positional_scale_bounds_match_the_scorers() -> None:
    # Nobody chose an endpoint of the 1-5 scale, so no extreme responders exist.
    x = np.array(
        [
            [2, 4, 2, 4, 2, 4],
            [3, 3, 3, 3, 3, 3],
            [2, 3, 4, 3, 2, 3],
            [4, 4, 4, 4, 4, 2],
            [3, 2, 3, 4, 3, 3],
        ],
        dtype=float,
    )
    scores, flags = ier.u3_poly_flag(x, 1, 5)
    np.testing.assert_array_equal(scores, ier.u3_poly(x, 1, 5))
    np.testing.assert_array_equal(scores, np.zeros(len(x)))
    assert not flags.any()

    # Nobody chose 5, so inferred bounds would put the midpoint at 2.5 instead of 3.
    y = np.array([[3, 3, 3, 3], [1, 2, 3, 4], [2, 2, 1, 4], [4, 3, 3, 1]], dtype=float)
    scores, flags = ier.midpoint_responding_flag(y, 1, 5, 0.0, 0.75)
    np.testing.assert_array_equal(scores, ier.midpoint_responding(y, 1, 5))
    np.testing.assert_array_equal(scores, [1.0, 0.25, 0.0, 0.5])
    np.testing.assert_array_equal(flags, [True, False, False, False])

    scores, flags = ier.u3_poly_flag(y, 1, 5, 0.25, 50.0)
    np.testing.assert_array_equal(scores, [0.0, 0.25, 0.25, 0.25])
    np.testing.assert_array_equal(flags, [False, True, True, True])
    scores, flags = ier.midpoint_responding_flag(y, 1, 5, 1.0, None, 50.0)
    np.testing.assert_array_equal(scores, [1.0, 0.75, 0.75, 0.75])
    np.testing.assert_array_equal(flags, [True, False, False, False])


def test_new_helpers_and_psychsyn_tools_are_exported() -> None:
    from ier.psychsyn import psychsyn_critval, psychsyn_summary  # noqa: PLC0415

    names = [
        "psychsyn_flag",
        "psychant_flag",
        "person_total_flag",
        "u3_poly_flag",
        "midpoint_responding_flag",
        "psychsyn_critval",
        "psychsyn_summary",
    ]
    assert set(names) <= set(ier.__all__)
    assert ier.psychsyn_critval is psychsyn_critval
    assert ier.psychsyn_summary is psychsyn_summary
    assert all(callable(getattr(ier, name)) for name in names)
