"""Validate orchestration requests: overrides, selections, decisions, and options."""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from ier import (
    composite,
    composite_flag,
    composite_probability,
    composite_scores,
    composite_scores_summary,
    composite_summary,
    save_score_archive,
    screen,
    screen_scores,
)
from ier._registry import (
    IndexOptions,
    numeric_override,
    resolve_index_overrides,
    score_registered_indices,
)
from ier.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_RAW_COMPOSITES: dict[str, Callable[..., object]] = {
    "composite": composite,
    "flag": composite_flag,
    "summary": composite_summary,
    "probability": composite_probability,
}
_SCREEN_OVERRIDES = {
    "thresholds": {"longstring": 3.0, "irv": 0.5},
    "percentiles": {"longstring": 80.0, "irv": 90.0},
}
_BEST_SUBSET_WARNING = (
    "warning: --indices is ignored with --method best_subset; this will become an error\n"
)


@pytest.fixture
def survey() -> np.ndarray:
    return np.random.default_rng(7).integers(1, 6, size=(30, 8)).astype(float)


class _ItemsOnly:
    """Minimal mapping-like override that exposes only ``items()``."""

    def __init__(self, pairs: list[tuple[object, object]]) -> None:
        self._pairs = pairs

    def items(self) -> Iterator[tuple[object, object]]:
        return iter(self._pairs)


@pytest.mark.parametrize("label", list(_SCREEN_OVERRIDES))
def test_series_screen_overrides_match_mappings(survey: np.ndarray, label: str) -> None:
    pd = pytest.importorskip("pandas")
    mapping = _SCREEN_OVERRIDES[label]
    selected = ["irv", "longstring"]

    expected = screen(survey, indices=selected, **{label: mapping})
    result = screen(survey, indices=selected, **{label: pd.Series(mapping)})
    replay = screen_scores(expected["scores"], **{label: pd.Series(mapping)})

    for reused in (result, replay):
        assert reused["thresholds"] == expected["thresholds"]
        assert reused["threshold_sources"] == expected["threshold_sources"]
        assert reused["percentiles"] == expected["percentiles"]
        np.testing.assert_array_equal(reused["flag_counts"], expected["flag_counts"])


def test_series_weights_match_mapping_weights(survey: np.ndarray) -> None:
    pd = pytest.importorskip("pandas")
    weights = {"longstring": 0.5, "irv": 2.0}
    selected = ["irv", "longstring"]

    expected = composite_summary(survey, indices=selected, weights=weights)
    result = composite_summary(survey, indices=selected, weights=pd.Series(weights))

    np.testing.assert_array_equal(result["composite"], expected["composite"])
    assert list(result["weights"].items()) == [("irv", 2.0), ("longstring", 0.5)]
    np.testing.assert_array_equal(
        composite(survey, indices=selected, weights=pd.Series(weights)), expected["composite"]
    )
    np.testing.assert_array_equal(
        composite_scores(expected["indices"], weights=pd.Series(weights)),
        composite_scores(expected["indices"], weights=weights),
    )


def test_series_overrides_reject_repeated_labels_and_booleans(survey: np.ndarray) -> None:
    pd = pytest.importorskip("pandas")
    repeated = pd.Series([1.0, 2.0], index=["irv", "irv"])

    with pytest.raises(ValueError, match="^duplicate weight index: irv$"):
        composite(survey, indices=["irv"], weights=repeated)
    with pytest.raises(ValueError, match="^duplicate threshold index: irv$"):
        screen(survey, indices=["irv"], thresholds=repeated)
    with pytest.raises(ValueError, match="^weight for irv must be a positive finite number$"):
        composite(survey, indices=["irv"], weights=pd.Series({"irv": True}))


def test_mapping_like_overrides_need_only_items(survey: np.ndarray) -> None:
    convert = numeric_override("level")
    resolved = resolve_index_overrides(
        cast("Any", _ItemsOnly([("longstring", "2"), ("irv", -1)])),
        ["irv", "longstring"],
        label="level",
        convert=convert,
    )

    assert list(resolved.items()) == [("longstring", 2.0), ("irv", -1.0)]
    result = screen(survey, indices=["irv"], thresholds=cast("Any", _ItemsOnly([("irv", 0.5)])))
    assert result["thresholds"] == {"irv": 0.5}
    assert result["threshold_sources"] == {"irv": "fixed"}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (_ItemsOnly([("irv", 1.0), ("irv", 2.0)]), "duplicate level index: irv"),
        (_ItemsOnly([(["irv"], 1.0)]), "unknown level index: ['irv']"),
        (_ItemsOnly([(("irv",), 1.0)]), "unknown level index: ('irv',)"),
        (_ItemsOnly([("irv", np.True_)]), "level for irv must be a finite number"),
        ({"irv": np.False_}, "level for irv must be a finite number"),
    ],
)
def test_mapping_like_overrides_reject_ambiguous_entries(values: object, message: str) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        resolve_index_overrides(
            cast("Any", values), ["irv"], label="level", convert=numeric_override("level")
        )


@pytest.mark.parametrize(
    "values",
    [SimpleNamespace(items=[("irv", 1.0)]), np.array([("irv", 1.0)]), b"irv"],
)
def test_objects_without_callable_items_remain_type_errors(values: object) -> None:
    with pytest.raises(
        TypeError, match="^levels must be a mapping of registered index names to numbers$"
    ):
        resolve_index_overrides(
            cast("Any", values), ["irv"], label="level", convert=numeric_override("level")
        )


@pytest.mark.parametrize("container", ["numpy", "pandas"])
def test_composite_helpers_accept_array_like_selections(survey: np.ndarray, container: str) -> None:
    names = ["irv", "longstring"]
    if container == "pandas":
        selection: Any = pytest.importorskip("pandas").Index(names)
    else:
        selection = np.array(names)

    np.testing.assert_array_equal(
        composite(survey, indices=selection), composite(survey, indices=names)
    )
    flag_scores, flags = composite_flag(survey, indices=selection)
    expected_scores, expected_flags = composite_flag(survey, indices=names)
    np.testing.assert_array_equal(flag_scores, expected_scores)
    np.testing.assert_array_equal(flags, expected_flags)
    np.testing.assert_array_equal(
        composite_probability(survey, indices=selection),
        composite_probability(survey, indices=names),
    )
    summary = composite_summary(survey, indices=selection)
    assert summary["indices_used"] == names
    assert all(type(name) is str for name in summary["indices_used"])
    assert list(summary["weights"]) == names
    # screen() already accepted the same selection; both orchestrators now agree.
    assert screen(survey, indices=selection)["indices_used"] == names


@patch("ier.composite.validate_matrix_input", side_effect=AssertionError("matrix converted"))
def test_empty_array_like_selection_raises_documented_error(
    _convert_mock: Any, survey: np.ndarray
) -> None:
    for call in _RAW_COMPOSITES.values():
        with pytest.raises(ValueError, match="^indices must name at least one registered index$"):
            call(survey, indices=np.array([], dtype=str))
        with pytest.raises(ValueError, match="^invalid index 'u3_poly'"):
            call(survey, indices=np.array(["irv", "u3_poly"]))


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        ({"percentile": 150}, "^percentile must be a finite number between 0 and 100$"),
        ({"percentile": True}, "^percentile must be a finite number between 0 and 100$"),
        ({"threshold": "x"}, "^threshold must be a finite number$"),
        ({"threshold": float("nan")}, "^threshold must be a finite number$"),
        ({"threshold": True}, "^threshold must be a finite number$"),
    ],
)
def test_composite_flag_validates_decisions_before_conversion(
    survey: np.ndarray, decision: dict[str, object], message: str
) -> None:
    with (
        patch("ier.composite.validate_matrix_input", side_effect=AssertionError("converted")),
        patch("ier.composite.score_registered_indices", side_effect=AssertionError("scored")),
        pytest.raises(ValueError, match=message),
    ):
        composite_flag(survey, **cast("Any", decision))


@pytest.mark.parametrize("strict", ["yes", 1, None])
@pytest.mark.parametrize("name", ["screen", *_RAW_COMPOSITES])
def test_strict_is_validated_before_matrix_conversion(
    survey: np.ndarray, name: str, strict: object
) -> None:
    module = "screen" if name == "screen" else "composite"
    call = screen if name == "screen" else _RAW_COMPOSITES[name]
    with (
        patch(f"ier.{module}.validate_matrix_input", side_effect=AssertionError("converted")),
        pytest.raises(ValueError, match="^strict must be a boolean$"),
    ):
        call(survey, strict=strict)


def test_direct_registry_scoring_still_validates_strict(survey: np.ndarray) -> None:
    with pytest.raises(ValueError, match="^strict must be a boolean$"):
        score_registered_indices(survey, ["irv"], IndexOptions(), strict=cast("Any", "yes"))


@pytest.mark.parametrize("method", [["mean"], {}, {"mean": 1}, ("mean",)])
@pytest.mark.parametrize("name", list(_RAW_COMPOSITES))
def test_container_methods_raise_documented_value_error(
    survey: np.ndarray, name: str, method: object
) -> None:
    with (
        patch("ier.composite.validate_matrix_input", side_effect=AssertionError("converted")),
        pytest.raises(ValueError, match="^method must be 'mean', 'sum', 'max', or 'best_subset'$"),
    ):
        _RAW_COMPOSITES[name](survey, method=method)


@pytest.mark.parametrize("value", [["mean"], {}, {"drop": 1}])
def test_precomputed_container_options_raise_documented_value_errors(value: object) -> None:
    scores = {"irv": [0.1, 0.2, 0.3]}
    for combine in (composite_scores, composite_scores_summary):
        with pytest.raises(
            ValueError, match="^method must be 'mean', 'sum', or 'max' for precomputed scores$"
        ):
            combine(scores, method=cast("Any", value))
        with pytest.raises(ValueError, match="^unsupported must be 'error' or 'drop'$"):
            combine(scores, unsupported=cast("Any", value))


def test_drop_mode_names_dropped_indices_when_nothing_remains() -> None:
    scores = {"u3_poly": [0.1, 0.2, 0.3], "midpoint": [0.0, 0.5, 1.0]}
    errors = {"acquiescence": "scale bounds are required", "mad": "items were not configured"}
    message = (
        "^scores must contain at least one composite-enabled index after dropping "
        "unsupported indices: u3_poly, midpoint$"
    )

    for combine in (composite_scores, composite_scores_summary):
        with pytest.raises(ValueError, match=message):
            combine(scores, unsupported="drop")
        with pytest.raises(ValueError, match=message):
            combine(scores, errors=errors, unsupported="drop")


def test_cli_skip_unsupported_explains_an_empty_composite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "screen.npz"
    save_score_archive(archive, {"u3_poly": [0.1, 0.2, 0.3], "midpoint": [0.0, 0.5, 1.0]})

    assert main(["composite-scores", str(archive), "--skip-unsupported"]) == 1

    assert capsys.readouterr().err == (
        "warning: skipped indices that are not composite-enabled: u3_poly, midpoint\n"
        "error: scores must contain at least one composite-enabled index after dropping "
        "unsupported indices: u3_poly, midpoint\n"
    )


@pytest.fixture
def response_csv(tmp_path: Path) -> Path:
    path = tmp_path / "responses.csv"
    rows = np.random.default_rng(5).integers(1, 6, size=(40, 10))
    header = ",".join(f"q{item}" for item in range(10))
    np.savetxt(path, rows, fmt="%d", delimiter=",", header=header, comments="")
    return path


@pytest.mark.parametrize("components", [[], ["--include-components"]])
def test_cli_best_subset_warns_once_and_ignores_indices(
    response_csv: Path, capsys: pytest.CaptureFixture[str], components: list[str]
) -> None:
    base = ["composite", str(response_csv), "--method", "best_subset", "--format", "json"]
    with warnings.catch_warnings():
        # The CLI must not forward ignored indices into the library DeprecationWarning.
        warnings.simplefilter("error")
        assert main([*base, *components]) == 0
        plain = capsys.readouterr()
        assert main([*base, *components, "--indices", "mahad", "psychsyn"]) == 0
        ignored = capsys.readouterr()

    assert plain.err == ""
    assert ignored.err == _BEST_SUBSET_WARNING
    assert ignored.out == plain.out


def test_cli_explicit_indices_without_best_subset_do_not_warn(
    response_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["composite", str(response_csv), "--indices", "irv", "longstring", "--format", "json"]

    assert main([*argv, "--method", "mean"]) == 0

    assert capsys.readouterr().err == ""
