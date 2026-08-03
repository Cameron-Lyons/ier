"""Tests for validated reusable-score archive loading."""

from pathlib import Path

import numpy as np
import pytest

from ier import (
    composite_scores,
    composite_summary,
    load_score_archive,
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
