"""Tests for validated reusable-score archive loading."""

from pathlib import Path

import numpy as np
import pytest

from ier import (
    PsychsynModel,
    composite_scores,
    composite_summary,
    fit_psychsyn_model,
    fit_response_time_mixture,
    flag_consensus,
    flag_consensus_archives,
    load_archive,
    load_flag_consensus_archive,
    load_psychsyn_model,
    load_response_time_archive,
    load_response_time_mixture_model,
    load_score_archive,
    merge_flag_consensus_archives,
    merge_score_archives,
    psychsyn_model_scores,
    response_time_mixture_scores,
    response_time_score_flags,
    save_flag_consensus_archive,
    save_psychsyn_model,
    save_response_time_archive,
    save_response_time_mixture_model,
    save_score_archive,
    screen,
    screen_scores,
)
from ier._cli_npz import _write_composite_npz, _write_response_time_npz, _write_screen_npz


def test_screen_archive_round_trip_supports_reuse(tmp_path: Path) -> None:
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    result = screen(
        data,
        indices=["irv", "longstring", "mad"],
        thresholds={"irv": 0.5, "longstring": 3.0},
        min_flags=1,
    )
    destination = tmp_path / "screen.npz"
    respondent_ids = ["case-1", "case-2", "case-3"]
    _write_screen_npz(destination, result, respondent_ids)

    loaded = load_score_archive(destination)
    reused = screen_scores(
        loaded["scores"],
        thresholds={"irv": 0.5, "longstring": 3.0},
        min_flags=1,
    )

    assert loaded["schema_version"] == 1
    assert loaded["result_type"] == "screen"
    assert loaded["n_respondents"] == 3
    assert list(loaded["scores"]) == ["irv", "longstring"]
    assert loaded["respondent_ids"] == respondent_ids
    assert list(loaded["errors"]) == ["mad"]
    assert "mad_positive_items" in loaded["errors"]["mad"]
    for name in loaded["scores"]:
        np.testing.assert_array_equal(loaded["scores"][name], result["scores"][name])
        np.testing.assert_array_equal(reused["flags"][name], result["flags"][name])
    np.testing.assert_array_equal(reused["consensus_flags"], result["consensus_flags"])


def test_detailed_composite_archive_round_trip_supports_reuse(tmp_path: Path) -> None:
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [2.0, 5.0, 1.0, 4.0, 3.0],
        ]
    )
    details = composite_summary(data, indices=["irv", "longstring", "person_total"])
    destination = tmp_path / "composite.npz"
    _write_composite_npz(
        destination,
        details["composite"],
        details["method"],
        errors=details["errors"],
        component_scores=details["indices"],
        valid_index_counts=details["valid_index_counts"],
    )

    loaded = load_score_archive(str(destination))
    reused = composite_scores(loaded["scores"])

    assert loaded["result_type"] == "composite"
    assert loaded["n_respondents"] == 4
    assert loaded["respondent_ids"] is None
    assert loaded["errors"] == {}
    assert list(loaded["scores"]) == details["indices_used"]
    np.testing.assert_allclose(reused, details["composite"], rtol=1e-14, atol=1e-14)


def test_score_archive_merge_aligns_ids_and_preserves_order(tmp_path: Path) -> None:
    patterns = tmp_path / "patterns.npz"
    consistency = tmp_path / "consistency.npz"
    save_score_archive(
        patterns,
        {"longstring": np.asarray([10.0, 20.0, 30.0])},
        respondent_ids=["case-a", "case-b", "case-c"],
        errors={"mad": "missing item configuration"},
    )
    save_score_archive(
        consistency,
        {"irv": np.asarray([0.3, 0.1, 0.2])},
        respondent_ids=["case-c", "case-a", "case-b"],
        errors={"psychsyn": "no qualifying pairs"},
    )

    merged = merge_score_archives([patterns, consistency])

    assert merged["schema_version"] == 1
    assert merged["result_type"] == "screen"
    assert merged["n_respondents"] == 3
    assert list(merged["scores"]) == ["longstring", "irv"]
    assert merged["respondent_ids"] == ["case-a", "case-b", "case-c"]
    assert merged["errors"] == {
        "mad": "missing item configuration",
        "psychsyn": "no qualifying pairs",
    }
    np.testing.assert_array_equal(merged["scores"]["longstring"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(merged["scores"]["irv"], [0.1, 0.2, 0.3])


def test_score_archive_merge_supports_composite_reuse_and_successful_retry(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_score_archive(
        first,
        {"irv": np.asarray([0.1, 0.2])},
        errors={"mad": "missing item configuration"},
    )
    save_score_archive(
        second,
        {"mad": np.asarray([0.4, 0.8])},
    )

    merged = merge_score_archives([first, second], result_type="composite")
    combined = composite_scores(merged["scores"], standardize=False)

    assert merged["result_type"] == "composite"
    assert merged["respondent_ids"] is None
    assert merged["errors"] == {}
    np.testing.assert_allclose(combined, [0.15, 0.3], rtol=0.0, atol=1e-15)


def test_archive_consensus_aligns_ids_and_applies_registered_cutoffs(
    tmp_path: Path,
) -> None:
    scores_path = tmp_path / "scores.npz"
    timing_path = tmp_path / "timing.npz"
    save_score_archive(
        scores_path,
        {
            "irv": np.asarray([0.1, 0.8, np.nan]),
            "longstring": np.asarray([2.0, 10.0, 7.0]),
        },
        respondent_ids=["case-a", "case-b", "case-c"],
    )
    save_response_time_archive(
        timing_path,
        np.asarray([0.4, 0.5, 2.0]),
        np.asarray([True, True, False]),
        threshold=0.5,
        respondent_ids=["case-c", "case-a", "case-b"],
    )

    combined = flag_consensus_archives(
        scores_path,
        timing_path,
        indices=["longstring", "irv"],
        thresholds={"longstring": 5.0, "irv": 0.5},
        timing_name="speed",
        min_flags=2,
        min_valid_signals=3,
    )

    assert combined["schema_version"] == 1
    assert combined["result_type"] == "flag_consensus"
    assert combined["n_respondents"] == 3
    assert combined["n_signals"] == 3
    assert combined["signal_names"] == ["longstring", "irv", "speed"]
    assert combined["respondent_ids"] == ["case-a", "case-b", "case-c"]
    assert combined["min_flags"] == 2
    assert combined["min_valid_signals"] == 3
    np.testing.assert_array_equal(combined["scores"]["speed"], [0.5, 2.0, 0.4])
    np.testing.assert_array_equal(combined["flags"]["longstring"], [False, True, True])
    np.testing.assert_array_equal(combined["flags"]["irv"], [True, False, False])
    np.testing.assert_array_equal(combined["flags"]["speed"], [True, False, True])
    np.testing.assert_array_equal(combined["flag_counts"], [2, 1, 2])
    np.testing.assert_array_equal(combined["valid_signal_counts"], [3, 3, 2])
    np.testing.assert_array_equal(combined["consensus_eligible"], [True, True, False])
    np.testing.assert_array_equal(combined["consensus_flags"], [True, False, False])


def test_archive_consensus_supports_shared_unidentified_row_order(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.npz"
    timing_path = tmp_path / "timing.npz"
    save_score_archive(scores_path, {"irv": [0.1, 0.8]})
    save_response_time_archive(
        timing_path,
        [0.4, 2.0],
        [True, False],
        threshold=0.5,
    )

    combined = flag_consensus_archives(
        scores_path,
        timing_path,
        thresholds={"irv": 0.5},
        min_flags=1,
    )

    assert combined["respondent_ids"] is None
    np.testing.assert_array_equal(combined["flags"]["irv"], [True, False])
    np.testing.assert_array_equal(combined["flags"]["response_time"], [True, False])
    np.testing.assert_array_equal(combined["consensus_flags"], [True, False])


def test_archive_consensus_rejects_unsafe_alignment_and_selection(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.npz"
    timing_path = tmp_path / "timing.npz"
    unidentified = tmp_path / "unidentified.npz"
    fewer = tmp_path / "fewer.npz"
    different_ids = tmp_path / "different-ids.npz"
    save_score_archive(
        scores_path,
        {"irv": [0.1, 0.8, 0.4], "longstring": [2.0, 10.0, 7.0]},
        respondent_ids=["case-a", "case-b", "case-c"],
    )
    save_response_time_archive(
        timing_path,
        [0.4, 2.0, 0.3],
        [True, False, True],
        threshold=0.5,
        respondent_ids=["case-a", "case-b", "case-c"],
    )
    save_response_time_archive(
        unidentified,
        [0.4, 2.0, 0.3],
        [True, False, True],
        threshold=0.5,
    )
    save_response_time_archive(fewer, [0.4, 2.0], [True, False], threshold=0.5)
    save_response_time_archive(
        different_ids,
        [0.4, 2.0, 0.3],
        [True, False, True],
        threshold=0.5,
        respondent_ids=["case-a", "case-b", "other"],
    )

    with pytest.raises(TypeError, match="indices must be a sequence"):
        flag_consensus_archives(scores_path, timing_path, indices="irv")
    with pytest.raises(ValueError, match="at least one stored index"):
        flag_consensus_archives(scores_path, timing_path, indices=[])
    with pytest.raises(ValueError, match="nonblank strings"):
        flag_consensus_archives(scores_path, timing_path, indices=[""])
    with pytest.raises(ValueError, match="indices must not contain duplicates"):
        flag_consensus_archives(scores_path, timing_path, indices=["irv", "irv"])
    with pytest.raises(ValueError, match="does not contain selected index: mad"):
        flag_consensus_archives(scores_path, timing_path, indices=["mad"])
    with pytest.raises(ValueError, match="must not duplicate"):
        flag_consensus_archives(scores_path, timing_path, timing_name="irv")
    with pytest.raises(ValueError, match="timing_name must be a nonblank string"):
        flag_consensus_archives(scores_path, timing_path, timing_name=" ")
    with pytest.raises(ValueError, match="both include respondent IDs"):
        flag_consensus_archives(scores_path, unidentified)
    with pytest.raises(ValueError, match="same number of respondents"):
        flag_consensus_archives(scores_path, fewer)
    with pytest.raises(ValueError, match="respondent ID sets must match"):
        flag_consensus_archives(scores_path, different_ids)


def test_flag_consensus_archive_merge_aligns_ids_and_recomputes_decisions(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_flag_consensus_archive(
        first,
        {
            "pattern": [True, False, True],
            "consistency": [False, True, False],
        },
        scores={"pattern": [1.0, np.nan, 1.0]},
        min_flags=1,
        respondent_ids=["case-a", "case-b", "case-c"],
    )
    save_flag_consensus_archive(
        second,
        {
            "speed": [True, True, False],
            "timing": [False, True, True],
        },
        scores={"speed": [0.4, 0.5, 2.0]},
        min_flags=2,
        respondent_ids=["case-c", "case-a", "case-b"],
    )

    merged = merge_flag_consensus_archives(
        [first, second],
        min_flags=3,
        min_valid_signals=4,
    )

    assert merged["schema_version"] == 1
    assert merged["result_type"] == "flag_consensus"
    assert merged["n_respondents"] == 3
    assert merged["n_signals"] == 4
    assert merged["signal_names"] == ["pattern", "consistency", "speed", "timing"]
    assert list(merged["scores"]) == ["pattern", "speed"]
    assert merged["respondent_ids"] == ["case-a", "case-b", "case-c"]
    assert merged["min_flags"] == 3
    assert merged["min_valid_signals"] == 4
    np.testing.assert_array_equal(merged["flags"]["speed"], [True, False, True])
    np.testing.assert_array_equal(merged["flags"]["timing"], [True, True, False])
    np.testing.assert_array_equal(merged["scores"]["speed"], [0.5, 2.0, 0.4])
    np.testing.assert_array_equal(merged["flag_counts"], [3, 2, 2])
    np.testing.assert_array_equal(merged["valid_signal_counts"], [4, 3, 4])
    np.testing.assert_array_equal(merged["consensus_eligible"], [True, False, True])
    np.testing.assert_array_equal(merged["consensus_flags"], [True, False, False])


def test_flag_consensus_archive_merge_supports_shared_unidentified_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_flag_consensus_archive(first, {"pattern": [True, False]})
    save_flag_consensus_archive(second, {"speed": [False, True]})

    merged = merge_flag_consensus_archives([first, second], min_flags=1)

    assert merged["respondent_ids"] is None
    assert merged["signal_names"] == ["pattern", "speed"]
    np.testing.assert_array_equal(merged["consensus_flags"], [True, True])


def test_flag_consensus_archive_merge_rejects_unsafe_inputs(tmp_path: Path) -> None:
    base = tmp_path / "base.npz"
    aligned = tmp_path / "aligned.npz"
    unidentified = tmp_path / "unidentified.npz"
    fewer = tmp_path / "fewer.npz"
    different_ids = tmp_path / "different-ids.npz"
    duplicate = tmp_path / "duplicate.npz"
    save_flag_consensus_archive(
        base,
        {"pattern": [True, False, True]},
        respondent_ids=["case-a", "case-b", "case-c"],
    )
    save_flag_consensus_archive(
        aligned,
        {"speed": [False, True, False]},
        respondent_ids=["case-a", "case-b", "case-c"],
    )
    save_flag_consensus_archive(unidentified, {"speed": [False, True, False]})
    save_flag_consensus_archive(fewer, {"speed": [False, True]})
    save_flag_consensus_archive(
        different_ids,
        {"speed": [False, True, False]},
        respondent_ids=["case-a", "case-b", "other"],
    )
    save_flag_consensus_archive(
        duplicate,
        {"pattern": [False, True, False]},
        respondent_ids=["case-a", "case-b", "case-c"],
    )

    with pytest.raises(TypeError, match="paths must be a sequence"):
        merge_flag_consensus_archives(str(base))
    with pytest.raises(ValueError, match="at least two flag-consensus archives"):
        merge_flag_consensus_archives([base])
    with pytest.raises(ValueError, match="same number of respondents"):
        merge_flag_consensus_archives([base, fewer])
    with pytest.raises(ValueError, match="all include respondent IDs"):
        merge_flag_consensus_archives([base, unidentified])
    with pytest.raises(ValueError, match="respondent ID sets must match"):
        merge_flag_consensus_archives([base, different_ids])
    with pytest.raises(ValueError, match="duplicate consensus signal"):
        merge_flag_consensus_archives([base, duplicate])
    with pytest.raises(ValueError, match="min_flags must be a positive integer"):
        merge_flag_consensus_archives([base, aligned], min_flags=0)
    with pytest.raises(ValueError, match="cannot exceed the number of flag signals"):
        merge_flag_consensus_archives([base, aligned], min_valid_signals=3)


def test_score_archive_merge_rejects_unsafe_alignment_and_conflicts(tmp_path: Path) -> None:
    base = tmp_path / "base.npz"
    unidentified = tmp_path / "unidentified.npz"
    fewer = tmp_path / "fewer.npz"
    different_ids = tmp_path / "different-ids.npz"
    duplicate = tmp_path / "duplicate.npz"
    conflicting_error = tmp_path / "conflicting-error.npz"
    screening_only = tmp_path / "screening-only.npz"
    save_score_archive(
        base,
        {"irv": [0.1, 0.2, 0.3]},
        respondent_ids=["case-a", "case-b", "case-c"],
        errors={"mad": "first failure"},
    )
    save_score_archive(unidentified, {"longstring": [1.0, 2.0, 3.0]})
    save_score_archive(fewer, {"longstring": [1.0, 2.0]})
    save_score_archive(
        different_ids,
        {"longstring": [1.0, 2.0, 3.0]},
        respondent_ids=["case-a", "case-b", "other"],
    )
    save_score_archive(
        duplicate,
        {"irv": [0.3, 0.2, 0.1]},
        respondent_ids=["case-a", "case-b", "case-c"],
    )
    save_score_archive(
        conflicting_error,
        {"longstring": [1.0, 2.0, 3.0]},
        respondent_ids=["case-a", "case-b", "case-c"],
        errors={"mad": "different failure"},
    )
    save_score_archive(
        screening_only,
        {"midpoint": [0.0, 1.0, 0.0]},
        respondent_ids=["case-a", "case-b", "case-c"],
    )

    with pytest.raises(TypeError, match="sequence"):
        merge_score_archives(str(base))
    with pytest.raises(ValueError, match="at least two"):
        merge_score_archives([base])
    with pytest.raises(ValueError, match="all include respondent IDs"):
        merge_score_archives([base, unidentified])
    with pytest.raises(ValueError, match="same number of respondents"):
        merge_score_archives([unidentified, fewer])
    with pytest.raises(ValueError, match="ID sets must match"):
        merge_score_archives([base, different_ids])
    with pytest.raises(ValueError, match="duplicate score index"):
        merge_score_archives([base, duplicate])
    with pytest.raises(ValueError, match="conflicting error messages"):
        merge_score_archives([base, conflicting_error])
    with pytest.raises(ValueError, match="invalid index"):
        merge_score_archives([base, screening_only], result_type="composite")


def test_aggregate_only_composite_archive_has_actionable_error(tmp_path: Path) -> None:
    destination = tmp_path / "aggregate.npz"
    _write_composite_npz(destination, np.array([0.1, 0.2]), "mean")

    with pytest.raises(ValueError, match="--include-components"):
        load_score_archive(destination)


def test_response_time_archive_is_not_a_registered_score_archive(tmp_path: Path) -> None:
    destination = tmp_path / "timing.npz"
    _write_response_time_npz(
        destination,
        np.array([1.0, 2.0]),
        np.array([True, False]),
        "median",
        "low",
        1.5,
    )

    with pytest.raises(ValueError, match="result_type must be 'screen' or 'composite'"):
        load_score_archive(destination)


def test_response_time_archive_round_trip_supports_reflagging(tmp_path: Path) -> None:
    destination = tmp_path / "timing.npz"
    scores = np.asarray([0.5, 1.0, 1.5, 2.0, np.nan])
    threshold = 1.0
    flags = scores < threshold
    respondent_ids = ["case-1", "case-2", "case-3", "case-4", "case-5"]
    _write_response_time_npz(
        destination,
        scores,
        flags,
        "median",
        "low",
        threshold,
        respondent_ids,
    )

    loaded = load_response_time_archive(destination)
    stricter = response_time_score_flags(
        loaded["scores"],
        threshold=0.75,
        direction=loaded["flag_direction"],
    )

    assert loaded["schema_version"] == 1
    assert loaded["result_type"] == "response_time"
    assert loaded["n_respondents"] == 5
    assert loaded["metric"] == "median"
    assert loaded["flag_direction"] == "low"
    assert loaded["threshold"] == threshold
    assert loaded["threshold_source"] is None
    assert loaded["percentile"] is None
    assert loaded["respondent_ids"] == respondent_ids
    np.testing.assert_array_equal(loaded["scores"], scores)
    np.testing.assert_array_equal(loaded["flags"], flags)
    np.testing.assert_array_equal(stricter, np.asarray([True, False, False, False, False]))


def test_response_time_mixture_archive_preserves_high_tail(tmp_path: Path) -> None:
    destination = tmp_path / "mixture.npz"
    scores = np.asarray([0.01, 0.4, 0.8, 0.99])
    flags = scores >= 0.8
    _write_response_time_npz(
        destination,
        scores,
        flags,
        "mixture",
        "high",
        0.8,
    )

    loaded = load_response_time_archive(destination)

    assert loaded["metric"] == "mixture"
    assert loaded["flag_direction"] == "high"
    assert loaded["respondent_ids"] is None
    np.testing.assert_array_equal(loaded["flags"], flags)


def test_response_time_mixture_model_archive_round_trip_supports_later_scoring(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260803)
    reference = np.vstack(
        [
            rng.lognormal(mean=-0.7, sigma=0.2, size=(20, 7)),
            rng.lognormal(mean=1.2, sigma=0.35, size=(40, 7)),
        ]
    )
    later = rng.lognormal(mean=0.5, sigma=0.7, size=(11, 7))
    model = fit_response_time_mixture(reference, n_components=3, random_seed=42)
    destination = tmp_path / "timing-model.npz"

    save_response_time_mixture_model(destination, model)
    loaded = load_response_time_mixture_model(destination)

    with np.load(destination, allow_pickle=False) as raw:
        assert raw.files == [
            "schema_version",
            "result_type",
            "n_components",
            "weights",
            "means",
            "variances",
            "log_transform",
        ]
        assert raw["schema_version"].item() == 1
        assert raw["result_type"].item() == "response_time_mixture_model"
        assert raw["n_components"].item() == 3
        assert raw["log_transform"].dtype == np.bool_
        assert all(not raw[name].dtype.hasobject for name in raw.files)

    assert loaded.n_components == model.n_components
    assert loaded.fast_component == model.fast_component
    assert loaded.log_transform == model.log_transform
    assert not loaded.weights.flags.writeable
    assert not loaded.means.flags.writeable
    assert not loaded.variances.flags.writeable
    np.testing.assert_array_equal(loaded.weights, model.weights)
    np.testing.assert_array_equal(loaded.means, model.means)
    np.testing.assert_array_equal(loaded.variances, model.variances)
    np.testing.assert_array_equal(
        response_time_mixture_scores(later, loaded),
        response_time_mixture_scores(later, model),
    )


def _response_time_mixture_model_payload() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "result_type": np.asarray("response_time_mixture_model", dtype=np.str_),
        "n_components": np.asarray(2, dtype=np.int64),
        "weights": np.asarray([0.25, 0.75], dtype=np.float64),
        "means": np.asarray([-1.0, 1.0], dtype=np.float64),
        "variances": np.asarray([0.5, 1.5], dtype=np.float64),
        "log_transform": np.asarray(True, dtype=np.bool_),
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": np.asarray(2)}, "unsupported.*schema version"),
        ({"result_type": np.asarray("response_time")}, "result_type"),
        ({"n_components": np.asarray(1)}, "at least 2"),
        ({"n_components": np.asarray(3)}, "lengths must match"),
        ({"weights": np.asarray([[0.25, 0.75]])}, "numeric vector"),
        ({"weights": np.asarray(["0.25", "0.75"])}, "numeric vector"),
        ({"weights": np.asarray([0.0, 1.0])}, "weights must be positive"),
        ({"weights": np.asarray([0.4, 0.5])}, "weights must sum to one"),
        ({"means": np.asarray([-1.0, np.inf])}, "finite values"),
        ({"variances": np.asarray([0.5, 0.0])}, "variances must be positive"),
        ({"log_transform": np.asarray(1)}, "Boolean scalar"),
        ({"unexpected": np.asarray(1)}, "unexpected member"),
    ],
)
def test_malformed_response_time_mixture_model_archive_is_rejected(
    tmp_path: Path,
    updates: dict[str, np.ndarray],
    message: str,
) -> None:
    payload = _response_time_mixture_model_payload()
    payload.update(updates)
    destination = tmp_path / "malformed-model.npz"
    np.savez(destination, **payload)

    with pytest.raises(ValueError, match=message):
        load_response_time_mixture_model(destination)


def test_response_time_mixture_model_requires_complete_pickle_free_npz(
    tmp_path: Path,
) -> None:
    missing_payload = _response_time_mixture_model_payload()
    missing_payload.pop("means")
    missing = tmp_path / "missing-model.npz"
    np.savez(missing, **missing_payload)

    object_payload = _response_time_mixture_model_payload()
    object_payload["weights"] = np.asarray([object(), object()], dtype=object)
    unsafe = tmp_path / "object-model.npz"
    np.savez(unsafe, **object_payload)

    plain = tmp_path / "model.npy"
    np.save(plain, np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="missing required member: means"):
        load_response_time_mixture_model(missing)
    with pytest.raises(ValueError, match="not pickle-free"):
        load_response_time_mixture_model(unsafe)
    with pytest.raises(ValueError, match="must be an NPZ archive"):
        load_response_time_mixture_model(plain)


def test_response_time_mixture_model_writer_validates_before_replacing(
    tmp_path: Path,
) -> None:
    model = fit_response_time_mixture(
        np.asarray([[0.4, 0.5], [0.6, 0.7], [4.0, 5.0], [6.0, 7.0]]),
        random_seed=42,
    )
    destination = tmp_path / "model.npz"
    save_response_time_mixture_model(destination, model)
    original = destination.read_bytes()

    model.weights.setflags(write=True)
    model.weights[0] = 0.0
    with pytest.raises(ValueError, match="weights must be positive"):
        save_response_time_mixture_model(destination, model)

    assert destination.read_bytes() == original
    with pytest.raises(ValueError, match="must end in .npz"):
        save_response_time_mixture_model(tmp_path / "model.bin", model)
    with pytest.raises(TypeError, match="model must be"):
        save_response_time_mixture_model(
            tmp_path / "other.npz",
            object(),  # type: ignore[arg-type]
        )
    assert not (tmp_path / "other.npz").exists()


def test_psychsyn_model_archive_round_trip_supports_fixed_scoring(tmp_path: Path) -> None:
    reference = np.asarray(
        [
            [1.0, 1.0, 1.0, 6.0],
            [2.0, 2.0, 2.0, 5.0],
            [3.0, 3.0, 3.0, 4.0],
            [4.0, 4.0, 4.0, 3.0],
            [5.0, 5.0, 5.0, 2.0],
            [6.0, 6.0, 6.0, 1.0],
        ]
    )
    later = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 3.0, 2.0]])
    model = fit_psychsyn_model(reference, critval=0.99)
    destination = tmp_path / "psychsyn-model.npz"

    save_psychsyn_model(destination, model)
    loaded = load_psychsyn_model(destination)

    assert loaded.n_items == 4
    assert loaded.n_pairs == 3
    assert loaded.critval == 0.99
    assert loaded.anto is False
    assert not loaded.item_pairs.flags.writeable
    np.testing.assert_array_equal(loaded.item_pairs, model.item_pairs)
    np.testing.assert_array_equal(
        psychsyn_model_scores(later, loaded),
        psychsyn_model_scores(later, model),
    )
    with np.load(destination, allow_pickle=False) as raw:
        assert set(raw.files) == {
            "schema_version",
            "result_type",
            "n_items",
            "critval",
            "anto",
            "item_pairs",
        }
        assert raw["item_pairs"].dtype.kind in "iu"

    empty_path = tmp_path / "empty-psychsyn-model.npz"
    save_psychsyn_model(
        empty_path,
        PsychsynModel(np.empty((0, 2), dtype=np.int64), n_items=4),
    )
    empty = load_psychsyn_model(empty_path)
    assert empty.n_pairs == 0
    assert empty.item_pairs.shape == (0, 2)
    assert not empty.item_pairs.flags.writeable


def _psychsyn_model_payload() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "result_type": np.asarray("psychsyn_model", dtype=np.str_),
        "n_items": np.asarray(4, dtype=np.int64),
        "critval": np.asarray(0.6, dtype=np.float64),
        "anto": np.asarray(False, dtype=np.bool_),
        "item_pairs": np.asarray([[1, 0], [2, 0], [2, 1]], dtype=np.int64),
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": np.asarray(2)}, "unsupported.*schema version"),
        ({"result_type": np.asarray("screen")}, "result_type"),
        ({"n_items": np.asarray(1)}, "at least 2"),
        ({"n_items": np.asarray(4.0)}, "integer scalar"),
        ({"n_items": np.asarray(np.iinfo(np.uint64).max)}, "platform index range"),
        ({"critval": np.asarray(np.nan)}, "must be finite"),
        ({"anto": np.asarray(0)}, "Boolean scalar"),
        ({"item_pairs": np.asarray([1, 0])}, "two-column integer array"),
        ({"item_pairs": np.asarray([[1.0, 0.0]])}, "two-column integer array"),
        ({"item_pairs": np.asarray([[4, 0]])}, "outside the fitted item range"),
        ({"item_pairs": np.asarray([[1, 1]])}, "cannot pair an item with itself"),
        ({"item_pairs": np.asarray([[1, 0], [0, 1]])}, "duplicate pairs"),
        ({"unexpected": np.asarray(1)}, "unexpected member"),
    ],
)
def test_malformed_psychsyn_model_archive_is_rejected(
    tmp_path: Path,
    updates: dict[str, np.ndarray],
    message: str,
) -> None:
    payload = _psychsyn_model_payload()
    payload.update(updates)
    destination = tmp_path / "malformed-psychsyn-model.npz"
    np.savez(destination, **payload)

    with pytest.raises(ValueError, match=message):
        load_psychsyn_model(destination)


def test_psychsyn_model_requires_complete_pickle_free_npz(tmp_path: Path) -> None:
    missing_payload = _psychsyn_model_payload()
    missing_payload.pop("item_pairs")
    missing = tmp_path / "missing-psychsyn-model.npz"
    np.savez(missing, **missing_payload)

    object_payload = _psychsyn_model_payload()
    object_payload["item_pairs"] = np.asarray([[object(), object()]], dtype=object)
    unsafe = tmp_path / "unsafe-psychsyn-model.npz"
    np.savez(unsafe, **object_payload)

    plain = tmp_path / "psychsyn-model.npy"
    np.save(plain, np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="missing required member: item_pairs"):
        load_psychsyn_model(missing)
    with pytest.raises(ValueError, match="not pickle-free"):
        load_psychsyn_model(unsafe)
    with pytest.raises(ValueError, match="must be an NPZ archive"):
        load_psychsyn_model(plain)


def test_psychsyn_model_writer_validates_before_replacing(tmp_path: Path) -> None:
    model = PsychsynModel(
        np.asarray([[1, 0], [2, 0], [2, 1]]),
        n_items=3,
        critval=0.6,
    )
    destination = tmp_path / "psychsyn-model.npz"
    save_psychsyn_model(destination, model)
    original = destination.read_bytes()

    model.item_pairs.setflags(write=True)
    model.item_pairs[0] = [1, 1]
    with pytest.raises(ValueError, match="cannot pair an item with itself"):
        save_psychsyn_model(destination, model)

    assert destination.read_bytes() == original
    with pytest.raises(ValueError, match="must end in .npz"):
        save_psychsyn_model(tmp_path / "model.bin", model)
    with pytest.raises(TypeError, match="model must be"):
        save_psychsyn_model(tmp_path / "other.npz", object())  # type: ignore[arg-type]
    assert not (tmp_path / "other.npz").exists()


def test_generic_archive_loader_inspects_psychsyn_model(tmp_path: Path) -> None:
    model = PsychsynModel(np.asarray([[1, 0], [2, 0]]), n_items=3, critval=-0.7, anto=True)
    destination = tmp_path / "psychant-model.npz"
    save_psychsyn_model(destination, model)

    loaded = load_archive(destination)

    assert loaded["result_type"] == "psychsyn_model"
    assert loaded["schema_version"] == 1
    assert loaded["n_items"] == 3
    assert loaded["n_pairs"] == 2
    assert loaded["critval"] == -0.7
    assert loaded["anto"] is True
    np.testing.assert_array_equal(loaded["item_pairs"], model.item_pairs)
    assert not loaded["item_pairs"].flags.writeable


def test_generic_archive_loader_auto_detects_supported_result_types(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.npz"
    timing_path = tmp_path / "timing.npz"
    consensus_path = tmp_path / "consensus.npz"
    model_path = tmp_path / "timing-model.npz"
    save_score_archive(
        score_path,
        {"irv": [0.1, 0.2], "longstring": [2.0, 5.0]},
        respondent_ids=["case-a", "case-b"],
    )
    save_response_time_archive(
        timing_path,
        [0.5, 1.5],
        [True, False],
        threshold=1.0,
        threshold_source="fixed",
    )
    save_flag_consensus_archive(
        consensus_path,
        {
            "irv": np.asarray([False, True]),
            "response_time": np.asarray([True, False]),
        },
        scores={"irv": [0.1, 0.2]},
        min_flags=1,
    )
    model = fit_response_time_mixture(
        np.asarray([[0.4], [0.5], [4.0], [5.0]]),
        random_seed=42,
    )
    save_response_time_mixture_model(model_path, model)

    score_archive = load_archive(score_path)
    timing_archive = load_archive(timing_path)
    consensus_archive = load_archive(consensus_path)
    model_archive = load_archive(model_path)

    assert score_archive["result_type"] == "screen"
    assert list(score_archive["scores"]) == ["irv", "longstring"]
    assert score_archive["respondent_ids"] == ["case-a", "case-b"]
    assert timing_archive["result_type"] == "response_time"
    assert timing_archive["threshold_source"] == "fixed"
    np.testing.assert_array_equal(timing_archive["flags"], [True, False])
    assert consensus_archive["result_type"] == "flag_consensus"
    assert consensus_archive["signal_names"] == ["irv", "response_time"]
    np.testing.assert_array_equal(consensus_archive["consensus_flags"], [True, True])
    assert model_archive["result_type"] == "response_time_mixture_model"
    assert model_archive["schema_version"] == 1
    assert model_archive["n_components"] == model.n_components
    assert model_archive["fast_component"] == model.fast_component
    assert model_archive["log_transform"] is True
    np.testing.assert_array_equal(model_archive["weights"], model.weights)
    np.testing.assert_array_equal(model_archive["means"], model.means)
    np.testing.assert_array_equal(model_archive["variances"], model.variances)
    assert not model_archive["weights"].flags.writeable


def test_generic_archive_loader_rejects_plain_and_unknown_results(tmp_path: Path) -> None:
    plain = tmp_path / "plain.npy"
    unknown = tmp_path / "unknown.npz"
    np.save(plain, np.asarray([1.0, 2.0]))
    np.savez(
        unknown,
        schema_version=np.asarray(1, dtype=np.int64),
        result_type=np.asarray("unknown", dtype=np.str_),
    )

    with pytest.raises(ValueError, match="must be an NPZ archive"):
        load_archive(plain)
    with pytest.raises(ValueError, match="screen.*composite.*response_time.*mixture_model"):
        load_archive(unknown)


def test_flag_consensus_archive_round_trip_supports_reuse(tmp_path: Path) -> None:
    destination = tmp_path / "consensus.npz"
    flags = {
        "pattern": np.asarray([True, False, False]),
        "response_time": np.asarray([False, True, True]),
    }
    scores = {"pattern": np.asarray([1.0, np.nan, 3.0])}
    respondent_ids = ["case-a", "case-b", "case-c"]

    save_flag_consensus_archive(
        destination,
        flags,
        scores=scores,
        min_flags=1,
        min_valid_signals=2,
        respondent_ids=respondent_ids,
    )

    with np.load(destination, allow_pickle=False) as raw:
        assert raw.files == [
            "schema_version",
            "result_type",
            "n_respondents",
            "n_signals",
            "signal_names",
            "score_names",
            "min_flags",
            "min_valid_signals",
            "flag_counts",
            "valid_signal_counts",
            "consensus_eligible",
            "consensus_flags",
            "flag__0",
            "flag__1",
            "score__0",
            "respondent_ids",
        ]
        assert all(not raw[name].dtype.hasobject for name in raw.files)
        assert raw["flag_counts"].dtype == np.uint8
        assert raw["valid_signal_counts"].dtype == np.uint8

    loaded = load_flag_consensus_archive(destination)
    reused = flag_consensus(
        loaded["flags"],
        scores=loaded["scores"],
        min_flags=2,
    )

    assert loaded["schema_version"] == 1
    assert loaded["result_type"] == "flag_consensus"
    assert loaded["signal_names"] == ["pattern", "response_time"]
    assert list(loaded["scores"]) == ["pattern"]
    assert loaded["min_flags"] == 1
    assert loaded["min_valid_signals"] == 2
    assert loaded["respondent_ids"] == respondent_ids
    np.testing.assert_array_equal(loaded["flag_counts"], [1, 1, 1])
    np.testing.assert_array_equal(loaded["valid_signal_counts"], [2, 1, 2])
    np.testing.assert_array_equal(loaded["consensus_eligible"], [True, False, True])
    np.testing.assert_array_equal(loaded["consensus_flags"], [True, False, True])
    np.testing.assert_array_equal(reused["consensus_flags"], [False, False, False])


def _flag_consensus_payload() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "result_type": np.asarray("flag_consensus", dtype=np.str_),
        "n_respondents": np.asarray(2, dtype=np.int64),
        "n_signals": np.asarray(2, dtype=np.int64),
        "signal_names": np.asarray(["pattern", "timing"], dtype=np.str_),
        "score_names": np.asarray(["pattern"], dtype=np.str_),
        "min_flags": np.asarray(1, dtype=np.int64),
        "min_valid_signals": np.asarray(2, dtype=np.int64),
        "flag_counts": np.asarray([1, 1], dtype=np.int64),
        "valid_signal_counts": np.asarray([2, 1], dtype=np.int64),
        "consensus_eligible": np.asarray([True, False], dtype=np.bool_),
        "consensus_flags": np.asarray([True, False], dtype=np.bool_),
        "flag__0": np.asarray([True, False], dtype=np.bool_),
        "flag__1": np.asarray([False, True], dtype=np.bool_),
        "score__0": np.asarray([1.0, np.nan], dtype=np.float64),
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": np.asarray(2)}, "unsupported.*schema version"),
        ({"result_type": np.asarray("screen")}, "result_type"),
        ({"n_respondents": np.asarray(0)}, "must be positive"),
        ({"n_signals": np.asarray(3)}, "signal_names must match"),
        (
            {"signal_names": np.asarray(["pattern", "pattern"])},
            "signal names must be unique",
        ),
        ({"score_names": np.asarray(["other"])}, "must be selected signals"),
        ({"flag__0": np.asarray([1, 0])}, "boolean vector"),
        ({"score__0": np.asarray([np.inf, np.nan])}, "finite values or NaN"),
        ({"score__0": np.asarray([np.nan, np.nan])}, "false where scores"),
        ({"min_valid_signals": np.asarray(3)}, "cannot exceed"),
        ({"flag_counts": np.asarray([0, 1])}, "flag_counts is inconsistent"),
        ({"unexpected": np.asarray(1)}, "unexpected member"),
    ],
)
def test_malformed_flag_consensus_archive_is_rejected(
    tmp_path: Path,
    updates: dict[str, np.ndarray],
    message: str,
) -> None:
    payload = _flag_consensus_payload()
    payload.update(updates)
    destination = tmp_path / "malformed-consensus.npz"
    np.savez(destination, **payload)

    with pytest.raises(ValueError, match=message):
        load_flag_consensus_archive(destination)


def test_flag_consensus_archive_requires_complete_pickle_free_npz(tmp_path: Path) -> None:
    missing_payload = _flag_consensus_payload()
    missing_payload.pop("flag__1")
    missing = tmp_path / "missing-consensus.npz"
    np.savez(missing, **missing_payload)

    unsafe_payload = _flag_consensus_payload()
    unsafe_payload["score__0"] = np.asarray([object(), object()], dtype=object)
    unsafe = tmp_path / "unsafe-consensus.npz"
    np.savez(unsafe, **unsafe_payload)

    with pytest.raises(ValueError, match="missing required member: flag__1"):
        load_flag_consensus_archive(missing)
    with pytest.raises(ValueError, match="not pickle-free"):
        load_flag_consensus_archive(unsafe)


def test_flag_consensus_archive_writer_validates_before_replacing(tmp_path: Path) -> None:
    destination = tmp_path / "consensus.npz"
    destination.write_bytes(b"keep")

    with pytest.raises(ValueError, match="false where scores"):
        save_flag_consensus_archive(
            destination,
            {"pattern": [True]},
            scores={"pattern": [np.nan]},
        )
    assert destination.read_bytes() == b"keep"

    with pytest.raises(ValueError, match="end in .npz"):
        save_flag_consensus_archive(tmp_path / "consensus.json", {"pattern": [True]})


def _response_time_payload() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "result_type": np.asarray("response_time", dtype=np.str_),
        "n_respondents": np.asarray(2, dtype=np.int64),
        "metric": np.asarray("median", dtype=np.str_),
        "flag_direction": np.asarray("low", dtype=np.str_),
        "threshold": np.asarray(1.5, dtype=np.float64),
        "scores": np.asarray([1.0, 2.0], dtype=np.float64),
        "flags": np.asarray([True, False], dtype=np.bool_),
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": np.asarray(3)}, "unsupported.*schema version"),
        ({"result_type": np.asarray("screen")}, "result_type"),
        ({"n_respondents": np.asarray(0)}, "must be positive"),
        ({"metric": np.asarray("unknown")}, "unsupported metric"),
        ({"flag_direction": np.asarray("sideways")}, "flag_direction"),
        ({"flag_direction": np.asarray("high")}, "requires 'low'"),
        (
            {
                "metric": np.asarray("mixture"),
                "flag_direction": np.asarray("low"),
            },
            "requires 'high'",
        ),
        ({"threshold": np.asarray(np.inf)}, "threshold must be finite"),
        ({"threshold": np.asarray([1.5])}, "numeric scalar"),
        ({"scores": np.asarray([1.0])}, "scores must match n_respondents"),
        ({"scores": np.asarray([1.0, np.inf])}, "finite values or NaN"),
        ({"flags": np.asarray([1, 0])}, "boolean vector"),
        ({"flags": np.asarray([True])}, "flags must match n_respondents"),
        ({"flags": np.asarray([False, True])}, "flags are inconsistent"),
        ({"unexpected": np.asarray(1)}, "unexpected member"),
    ],
)
def test_malformed_response_time_archive_is_rejected(
    tmp_path: Path,
    updates: dict[str, np.ndarray],
    message: str,
) -> None:
    payload = _response_time_payload()
    payload.update(updates)
    destination = tmp_path / "malformed-timing.npz"
    np.savez(destination, **payload)

    with pytest.raises(ValueError, match=message):
        load_response_time_archive(destination)


def test_response_time_archive_requires_complete_pickle_free_npz(tmp_path: Path) -> None:
    missing_payload = _response_time_payload()
    missing_payload.pop("scores")
    missing = tmp_path / "missing-timing.npz"
    np.savez(missing, **missing_payload)

    object_payload = _response_time_payload()
    object_payload["scores"] = np.asarray([object(), object()], dtype=object)
    unsafe = tmp_path / "object-timing.npz"
    np.savez(unsafe, **object_payload)

    plain = tmp_path / "timing.npy"
    np.save(plain, np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="missing required member: scores"):
        load_response_time_archive(missing)
    with pytest.raises(ValueError, match="not pickle-free"):
        load_response_time_archive(unsafe)
    with pytest.raises(ValueError, match="must be an NPZ archive"):
        load_response_time_archive(plain)


def test_public_response_time_writer_round_trip_preserves_result(tmp_path: Path) -> None:
    destination = tmp_path / "timing.npz"
    scores = np.asarray([0.5, 1.0, 2.0, np.nan])
    flags = np.asarray([True, True, False, False])
    respondent_ids = ["case-1", "case-2", "case-3", "case-4"]

    save_response_time_archive(
        str(destination),
        scores,
        flags,
        threshold=1.0,
        metric="median",
        respondent_ids=respondent_ids,
    )

    with np.load(destination, allow_pickle=False) as raw:
        assert raw.files == [
            "schema_version",
            "result_type",
            "n_respondents",
            "metric",
            "flag_direction",
            "threshold",
            "scores",
            "flags",
            "respondent_ids",
        ]
        assert raw["scores"].dtype == np.float64
        assert raw["flags"].dtype == np.bool_
        assert all(not raw[name].dtype.hasobject for name in raw.files)

    loaded = load_response_time_archive(destination)
    assert loaded["metric"] == "median"
    assert loaded["flag_direction"] == "low"
    assert loaded["threshold"] == 1.0
    assert loaded["threshold_source"] is None
    assert loaded["percentile"] is None
    assert loaded["respondent_ids"] == respondent_ids
    np.testing.assert_array_equal(loaded["scores"], scores)
    np.testing.assert_array_equal(loaded["flags"], flags)


def test_public_response_time_writer_records_fixed_cutoff_provenance(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "fixed-timing.npz"
    scores = np.asarray([0.5, 1.0, 2.0, np.nan])
    flags = np.asarray([True, True, False, False])

    save_response_time_archive(
        destination,
        scores,
        flags,
        threshold=1.0,
        threshold_source="fixed",
    )

    with np.load(destination, allow_pickle=False) as raw:
        assert raw["schema_version"].item() == 2
        assert raw["threshold_source"].item() == "fixed"
        assert np.isnan(raw["percentile"].item())

    loaded = load_response_time_archive(destination)
    assert loaded["schema_version"] == 2
    assert loaded["threshold_source"] == "fixed"
    assert loaded["percentile"] is None


def test_public_response_time_writer_records_percentile_cutoff_provenance(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "percentile-timing.npz"
    scores = np.asarray([0.5, 1.0, 1.0, 2.0, np.nan])
    flags = np.asarray([True, False, False, False, False])

    save_response_time_archive(
        destination,
        scores,
        flags,
        threshold=1.0,
        percentile=50.0,
    )

    loaded = load_response_time_archive(destination)
    assert loaded["schema_version"] == 2
    assert loaded["threshold_source"] == "percentile"
    assert loaded["percentile"] == 50.0
    np.testing.assert_array_equal(loaded["flags"], flags)


def _response_time_v2_payload() -> dict[str, np.ndarray]:
    payload = _response_time_payload()
    payload.update(
        {
            "schema_version": np.asarray(2, dtype=np.int64),
            "threshold_source": np.asarray("percentile", dtype=np.str_),
            "percentile": np.asarray(50.0, dtype=np.float64),
        }
    )
    return payload


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"threshold_source": np.asarray("other")}, "threshold_source"),
        (
            {
                "threshold_source": np.asarray("fixed"),
                "percentile": np.asarray(50.0),
            },
            "must be absent",
        ),
        ({"percentile": np.asarray(np.nan)}, "percentile is required"),
        ({"percentile": np.asarray(101.0)}, "between 0 and 100"),
        ({"threshold": np.asarray(1.75)}, "inconsistent with its percentile"),
        (
            {
                "n_respondents": np.asarray(3),
                "scores": np.asarray([1.0, 1.5, 2.0]),
                "flags": np.asarray([True, True, False]),
            },
            "flags are inconsistent",
        ),
        (
            {
                "threshold_source": np.asarray("fixed"),
                "percentile": np.asarray(np.nan),
                "n_respondents": np.asarray(3),
                "scores": np.asarray([1.0, 1.5, 2.0]),
                "flags": np.asarray([True, False, False]),
            },
            "flags are inconsistent",
        ),
    ],
)
def test_malformed_response_time_v2_provenance_is_rejected(
    tmp_path: Path,
    updates: dict[str, np.ndarray],
    message: str,
) -> None:
    payload = _response_time_v2_payload()
    payload.update(updates)
    destination = tmp_path / "malformed-timing-v2.npz"
    np.savez(destination, **payload)

    with pytest.raises(ValueError, match=message):
        load_response_time_archive(destination)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"threshold_source": "unknown"}, "threshold_source"),
        (
            {"threshold_source": "fixed", "percentile": 50.0},
            "must be absent",
        ),
        ({"threshold_source": "percentile"}, "percentile is required"),
        (
            {"threshold_source": "percentile", "percentile": 25.0},
            "inconsistent with its percentile",
        ),
    ],
)
def test_public_response_time_writer_rejects_invalid_provenance(
    tmp_path: Path,
    metadata: dict[str, object],
    message: str,
) -> None:
    destination = tmp_path / "timing.npz"

    with pytest.raises(ValueError, match=message):
        save_response_time_archive(
            destination,
            [1.0, 2.0],
            [True, False],
            threshold=1.5,
            **metadata,  # type: ignore[arg-type]
        )
    assert not destination.exists()


def test_public_response_time_writer_enforces_fixed_cutoff_ties(tmp_path: Path) -> None:
    destination = tmp_path / "timing.npz"

    with pytest.raises(ValueError, match="flags are inconsistent"):
        save_response_time_archive(
            destination,
            [1.0, 1.5, 2.0],
            [True, False, False],
            threshold=1.5,
            threshold_source="fixed",
        )
    assert not destination.exists()


def test_public_response_time_writer_accepts_exclusive_mixture_flags(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "mixture.npz"
    scores = np.asarray([0.2, 0.8, 0.95])
    flags = np.asarray([False, False, True])

    save_response_time_archive(
        destination,
        scores,
        flags,
        threshold=0.8,
        metric="mixture",
        flag_direction="high",
    )

    loaded = load_response_time_archive(destination)
    assert loaded["metric"] == "mixture"
    assert loaded["flag_direction"] == "high"
    np.testing.assert_array_equal(loaded["flags"], flags)


@pytest.mark.parametrize(
    ("scores", "flags", "metadata", "message"),
    [
        ([1.0, 2.0], [True, False], {"threshold": 1.5, "metric": "unknown"}, "metric"),
        (
            [1.0, 2.0],
            [True, False],
            {"threshold": 1.5, "flag_direction": "sideways"},
            "flag_direction",
        ),
        (
            [1.0, 2.0],
            [False, True],
            {"threshold": 1.5, "flag_direction": "high"},
            "requires 'low'",
        ),
        (
            [0.1, 0.9],
            [False, True],
            {"threshold": 0.5, "metric": "mixture"},
            "requires 'high'",
        ),
        ([1.0, 2.0], [True, False], {"threshold": np.inf}, "threshold must be finite"),
        ([1.0, 2.0], [True, False], {"threshold": True}, "numeric scalar"),
        ([1.0, 2.0], [True, False], {"threshold": [1.5]}, "numeric scalar"),
        ([], [], {"threshold": 1.5}, "scores cannot be empty"),
        ([[1.0, 2.0]], [True, False], {"threshold": 1.5}, "scores must be one-dimensional"),
        ([1.0, np.inf], [True, False], {"threshold": 1.5}, "finite values or NaN"),
        ([1.0, 2.0], [1, 0], {"threshold": 1.5}, "boolean vector"),
        ([1.0, 2.0], [[True, False]], {"threshold": 1.5}, "boolean vector"),
        ([1.0, 2.0], [True], {"threshold": 1.5}, "match n_respondents"),
        ([1.0, 2.0], [False, True], {"threshold": 1.5}, "flags are inconsistent"),
    ],
)
def test_public_response_time_writer_rejects_invalid_core_inputs(
    tmp_path: Path,
    scores: object,
    flags: object,
    metadata: dict[str, object],
    message: str,
) -> None:
    destination = tmp_path / "timing.npz"

    with pytest.raises(ValueError, match=message):
        save_response_time_archive(  # type: ignore[arg-type]
            destination,
            scores,
            flags,
            **metadata,  # type: ignore[arg-type]
        )
    assert not destination.exists()


def test_public_response_time_writer_rejects_path_and_identifiers(tmp_path: Path) -> None:
    scores = [1.0, 2.0]
    flags = [True, False]

    with pytest.raises(ValueError, match="must end in .npz"):
        save_response_time_archive(tmp_path / "timing.bin", scores, flags, threshold=1.5)
    with pytest.raises(TypeError, match="sequence of strings"):
        save_response_time_archive(
            tmp_path / "timing.npz",
            scores,
            flags,
            threshold=1.5,
            respondent_ids="ab",
        )
    for respondent_ids, message in [
        (["only-one"], "ID count"),
        (["case-1", " "], "IDs must be nonblank"),
        (["same", "same"], "IDs must be unique"),
        (["case-1", 2], "IDs must be strings"),
    ]:
        with pytest.raises(ValueError, match=message):
            save_response_time_archive(
                tmp_path / "timing.npz",
                scores,
                flags,
                threshold=1.5,
                respondent_ids=respondent_ids,  # type: ignore[arg-type]
            )


def test_public_response_time_writer_validates_before_touching_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "timing.npz"
    destination.write_bytes(b"existing-content")

    with pytest.raises(ValueError, match="flags are inconsistent"):
        save_response_time_archive(
            destination,
            [1.0, 2.0],
            [False, True],
            threshold=1.5,
        )

    assert destination.read_bytes() == b"existing-content"


def _base_payload() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "result_type": np.asarray("screen", dtype=np.str_),
        "n_respondents": np.asarray(2, dtype=np.int64),
        "index_names": np.asarray(["irv"], dtype=np.str_),
        "score__irv": np.asarray([0.1, 0.2], dtype=np.float64),
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": np.asarray(2)}, "unsupported.*schema version"),
        ({"schema_version": np.asarray([1])}, "integer scalar"),
        ({"result_type": np.asarray("unknown")}, "result_type"),
        ({"n_respondents": np.asarray(0)}, "must be positive"),
        ({"index_names": np.asarray([], dtype=np.str_)}, "does not contain reusable"),
        ({"index_names": np.asarray(["irv", "irv"])}, "must be unique"),
        ({"index_names": np.asarray(["unknown"])}, "invalid index"),
        ({"score__irv": np.asarray([0.1])}, "must match n_respondents"),
        ({"score__irv": np.asarray([0.1, np.inf])}, "finite values or NaN"),
        ({"respondent_ids": np.asarray(["same", "same"])}, "IDs must be unique"),
        ({"respondent_ids": np.asarray(["case-1"])}, "ID count"),
        ({"respondent_ids": np.asarray(["case-1", " "])}, "IDs must be nonblank"),
        ({"error_names": np.asarray(["mad"])}, "stored together"),
        (
            {
                "error_names": np.asarray(["mad"]),
                "error_messages": np.asarray([" "]),
            },
            "error messages must be nonblank",
        ),
    ],
)
def test_malformed_archive_metadata_is_rejected(
    tmp_path: Path,
    updates: dict[str, np.ndarray],
    message: str,
) -> None:
    payload = _base_payload()
    payload.update(updates)
    destination = tmp_path / "malformed.npz"
    np.savez(destination, **payload)

    with pytest.raises(ValueError, match=message):
        load_score_archive(destination)


def test_missing_and_undeclared_score_members_are_rejected(tmp_path: Path) -> None:
    missing_payload = _base_payload()
    missing_payload.pop("score__irv")
    missing = tmp_path / "missing-score.npz"
    np.savez(missing, **missing_payload)

    extra_payload = _base_payload()
    extra_payload["score__longstring"] = np.asarray([1.0, 2.0])
    extra = tmp_path / "extra-score.npz"
    np.savez(extra, **extra_payload)

    with pytest.raises(ValueError, match="missing declared score member"):
        load_score_archive(missing)
    with pytest.raises(ValueError, match="undeclared score member"):
        load_score_archive(extra)


def test_composite_archive_rejects_screening_only_indices(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["result_type"] = np.asarray("composite")
    payload["index_names"] = np.asarray(["midpoint"])
    payload.pop("score__irv")
    payload["score__midpoint"] = np.asarray([0.1, 0.2])
    destination = tmp_path / "invalid-composite.npz"
    np.savez(destination, **payload)

    with pytest.raises(ValueError, match="invalid index"):
        load_score_archive(destination)


def test_object_score_member_cannot_enable_pickling(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["score__irv"] = np.asarray([object(), object()], dtype=object)
    destination = tmp_path / "object-score.npz"
    np.savez(destination, **payload)

    with pytest.raises(ValueError, match="not pickle-free"):
        load_score_archive(destination)


def test_plain_npy_file_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "scores.npy"
    np.save(destination, np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="must be an NPZ archive"):
        load_score_archive(destination)


def test_public_writer_round_trip_preserves_order_and_metadata(tmp_path: Path) -> None:
    destination = tmp_path / "scores.npz"
    scores = {
        "irv": np.asarray([0.1, np.nan, 0.8]),
        "longstring": np.asarray([2.0, 5.0, 9.0]),
    }
    respondent_ids = ["case-1", "case-2", "case-3"]
    errors = {"mad": "positive-item configuration is required"}

    save_score_archive(
        str(destination),
        scores,
        respondent_ids=respondent_ids,
        errors=errors,
    )

    with np.load(destination, allow_pickle=False) as raw:
        assert raw.files == [
            "schema_version",
            "result_type",
            "n_respondents",
            "index_names",
            "error_names",
            "error_messages",
            "score__irv",
            "score__longstring",
            "respondent_ids",
        ]
        assert all(not raw[name].dtype.hasobject for name in raw.files)

    loaded = load_score_archive(destination)
    assert loaded["result_type"] == "screen"
    assert loaded["n_respondents"] == 3
    assert list(loaded["scores"]) == ["irv", "longstring"]
    assert loaded["respondent_ids"] == respondent_ids
    assert loaded["errors"] == errors
    for name, values in scores.items():
        np.testing.assert_array_equal(loaded["scores"][name], values)


def test_public_writer_composite_round_trip_supports_reuse(tmp_path: Path) -> None:
    destination = tmp_path / "components.npz"
    scores = {
        "irv": np.asarray([0.1, 0.2, 0.9]),
        "person_total": np.asarray([5.0, 8.0, 2.0]),
    }

    save_score_archive(destination, scores, result_type="composite")
    loaded = load_score_archive(destination)

    assert loaded["result_type"] == "composite"
    np.testing.assert_allclose(
        composite_scores(loaded["scores"]),
        composite_scores(scores),
        rtol=1e-14,
        atol=1e-14,
    )


def test_public_writer_rejects_invalid_core_inputs(tmp_path: Path) -> None:
    destination = tmp_path / "scores.npz"
    valid = {"irv": np.asarray([0.1, 0.2])}

    with pytest.raises(ValueError, match="must end in .npz"):
        save_score_archive(tmp_path / "scores.bin", valid)
    with pytest.raises(ValueError, match="result_type"):
        save_score_archive(destination, valid, result_type="unknown")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scores must be a mapping"):
        save_score_archive(destination, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not contain reusable"):
        save_score_archive(destination, {})
    with pytest.raises(ValueError, match="invalid index"):
        save_score_archive(destination, {"unknown": [0.1, 0.2]})
    with pytest.raises(ValueError, match="invalid index"):
        save_score_archive(
            destination,
            {"midpoint": [0.1, 0.2]},
            result_type="composite",
        )
    with pytest.raises(ValueError, match="same respondent count"):
        save_score_archive(destination, {"irv": [0.1], "longstring": [1.0, 2.0]})
    with pytest.raises(ValueError, match="finite values or NaN"):
        save_score_archive(destination, {"irv": [0.1, np.inf]})


def test_public_writer_rejects_invalid_error_metadata(tmp_path: Path) -> None:
    destination = tmp_path / "scores.npz"
    scores = {"irv": [0.1, 0.2]}

    with pytest.raises(TypeError, match="errors must be a mapping"):
        save_score_archive(destination, scores, errors=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both scores and errors"):
        save_score_archive(destination, scores, errors={"irv": "failed"})
    with pytest.raises(ValueError, match="invalid index"):
        save_score_archive(destination, scores, errors={"unknown": "failed"})
    with pytest.raises(ValueError, match="messages must be nonblank"):
        save_score_archive(destination, scores, errors={"mad": "  "})
    with pytest.raises(ValueError, match="messages must be strings"):
        save_score_archive(destination, scores, errors={"mad": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="invalid index"):
        save_score_archive(
            destination,
            scores,
            result_type="composite",
            errors={"midpoint": "failed"},
        )


def test_public_writer_rejects_invalid_respondent_ids(tmp_path: Path) -> None:
    destination = tmp_path / "scores.npz"
    scores = {"irv": [0.1, 0.2]}

    with pytest.raises(TypeError, match="sequence of strings"):
        save_score_archive(destination, scores, respondent_ids="ab")
    with pytest.raises(ValueError, match="ID count"):
        save_score_archive(destination, scores, respondent_ids=["only-one"])
    with pytest.raises(ValueError, match="IDs must be nonblank"):
        save_score_archive(destination, scores, respondent_ids=["case-1", " "])
    with pytest.raises(ValueError, match="IDs must be unique"):
        save_score_archive(destination, scores, respondent_ids=["same", "same"])
    with pytest.raises(ValueError, match="IDs must be strings"):
        save_score_archive(
            destination,
            scores,
            respondent_ids=["case-1", 2],  # type: ignore[list-item]
        )


def test_public_writer_validates_before_touching_destination(tmp_path: Path) -> None:
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"existing-content")

    with pytest.raises(ValueError, match="finite values or NaN"):
        save_score_archive(destination, {"irv": [0.1, np.inf]})

    assert destination.read_bytes() == b"existing-content"
