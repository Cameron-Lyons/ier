"""Retained failures preserve fresh-workflow coverage and audit information."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ier import (
    composite_scores,
    composite_scores_summary,
    composite_summary,
    load_score_archive,
    save_score_archive,
    screen,
    screen_scores,
)
from ier._registry import INDEX_REGISTRY

_INDICES = ["mad", "irv", "longstring", "evenodd"]
_DATA = np.asarray([[1, 1, 1, 1], [1, 2, 3, 4], [np.nan, np.nan, np.nan, np.nan], [2, 2, 3, 4]])


def _forbid_rescoring(monkeypatch: pytest.MonkeyPatch) -> None:
    def scorer(*args: Any) -> np.ndarray:
        raise AssertionError("a retained-score workflow must never recalculate an index")

    for name in _INDICES:
        monkeypatch.setitem(INDEX_REGISTRY, name, replace(INDEX_REGISTRY[name], scorer=scorer))


@pytest.mark.parametrize("minimum", [2, 3, 4])
def test_screen_archive_replay_preserves_failed_selection_and_consensus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minimum: int,
) -> None:
    settings = {
        "min_flags": 1,
        "min_valid_indices": minimum,
        "thresholds": {"longstring": 3.0, "mad": 2.0},
        "percentiles": {"irv": 90.0, "evenodd": 85.0},
    }
    direct = screen(_DATA, indices=_INDICES, **settings)
    destination = tmp_path / "screen.npz"
    save_score_archive(destination, direct["scores"], errors=direct["errors"])
    saved = load_score_archive(destination)
    for values in saved["scores"].values():
        values.flags.writeable = False
    _forbid_rescoring(monkeypatch)

    replay = screen_scores(saved["scores"], errors=saved["errors"], **settings)

    assert list(replay["errors"]) == ["mad", "evenodd"]
    assert replay["errors"] == direct["errors"]
    assert replay["errors"] is not saved["errors"]
    assert replay["indices_used"] == ["irv", "longstring"]
    assert replay["thresholds"] == direct["thresholds"]
    assert replay["threshold_sources"] == direct["threshold_sources"]
    assert replay["percentiles"] == direct["percentiles"]
    for name in replay["indices_used"]:
        assert replay["scores"][name] is saved["scores"][name]
        np.testing.assert_array_equal(replay["flags"][name], direct["flags"][name])
        for key in direct["summary"][name]:
            np.testing.assert_equal(replay["summary"][name][key], direct["summary"][name][key])
    np.testing.assert_array_equal(replay["valid_index_counts"], [2, 2, 1, 2])
    for key in ["flag_counts", "consensus_eligible", "consensus_flags"]:
        np.testing.assert_array_equal(replay[key], direct[key])
    if minimum > 2:
        assert not replay["consensus_eligible"].any()
        assert not replay["consensus_flags"].any()


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("minimum", [2, 3])
def test_composite_archive_replay_preserves_failed_weights_and_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    standardize: bool,
    minimum: int,
) -> None:
    settings = {
        "method": method,
        "standardize": standardize,
        "weights": {"irv": 2.0, "longstring": 0.5, "mad": 1e308},
        "min_valid_indices": minimum,
    }
    direct = composite_summary(_DATA, indices=_INDICES, **settings)
    destination = tmp_path / "components.npz"
    save_score_archive(
        destination, direct["indices"], result_type="composite", errors=direct["errors"]
    )
    saved = load_score_archive(destination)
    _forbid_rescoring(monkeypatch)

    replay = composite_scores_summary(saved["scores"], errors=saved["errors"], **settings)
    scores = composite_scores(saved["scores"], errors=saved["errors"], **settings)

    np.testing.assert_array_equal(replay["composite"], direct["composite"])
    np.testing.assert_array_equal(scores, direct["composite"])
    np.testing.assert_array_equal(replay["valid_index_counts"], [2, 2, 1, 2])
    assert replay["weights"] == direct["weights"]
    assert replay["weights"]["mad"] == 1e308
    assert replay["errors"] == direct["errors"]
    assert replay["errors"] is not saved["errors"]
    assert replay["indices_used"] == direct["indices_used"]
    for key in ["mean", "std", "min", "max", "n_valid", "n_total"]:
        np.testing.assert_equal(replay[key], direct[key])
    if minimum > 2:
        assert np.isnan(scores).all()


@pytest.mark.parametrize("workflow", [screen_scores, composite_scores, composite_scores_summary])
@pytest.mark.parametrize(
    ("errors", "exception", "message"),
    [
        ([], TypeError, "errors must be a mapping"),
        ({"unknown": "failed"}, ValueError, "invalid index"),
        ({"irv": "failed"}, ValueError, "both scores and errors"),
        ({"mad": " "}, ValueError, "messages must be nonblank"),
        ({"mad": 1}, ValueError, "messages must be strings"),
        ({1: "failed"}, ValueError, "names must be nonblank strings"),
        ({" ": "failed"}, ValueError, "names must be nonblank strings"),
    ],
)
def test_reusable_workflows_reject_invalid_failure_provenance(
    workflow: Callable[..., Any],
    errors: Any,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        workflow({"irv": [0.1, 0.2]}, errors=errors)


@pytest.mark.parametrize("workflow", [composite_scores, composite_scores_summary])
def test_composite_replay_rejects_screening_only_failed_indices(
    workflow: Callable[..., Any],
) -> None:
    with pytest.raises(ValueError, match="invalid index 'onset'"):
        workflow({"irv": [0.1, 0.2]}, errors={"onset": "window too short"})


def test_replay_error_metadata_does_not_relax_completeness_validation() -> None:
    for workflow in [screen_scores, composite_scores, composite_scores_summary]:
        with pytest.raises(ValueError, match="selected indices \\(2\\)"):
            workflow({"irv": [0.1, 0.2]}, errors={"mad": "not configured"}, min_valid_indices=3)


def test_screen_replay_still_validates_cutoff_conflicts_for_failed_indices() -> None:
    with pytest.raises(ValueError, match="both a threshold and percentile"):
        screen_scores(
            {"irv": [0.1, 0.2]},
            errors={"mad": "not configured"},
            thresholds={"mad": 1.0},
            percentiles={"mad": 90.0},
        )
