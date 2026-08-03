"""Tests for validated reusable-score archive loading."""

from pathlib import Path

import numpy as np
import pytest

from ier import (
    composite_scores,
    composite_summary,
    load_response_time_archive,
    load_score_archive,
    response_time_score_flags,
    save_response_time_archive,
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
