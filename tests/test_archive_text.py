"""Archive text retains identifiers and fails cleanly on corrupt Unicode."""

from pathlib import Path

import numpy as np
import pytest

from ier import (
    load_response_time_archive,
    load_score_archive,
    save_response_time_archive,
    save_score_archive,
    screen_scores,
)
from ier._cli_composite import CompositeReport, ResponseTimeReport
from ier._cli_npz import _write_composite_npz, _write_response_time_npz, _write_screen_npz


@pytest.mark.parametrize("writer", ["score", "timing", "screen-cli", "composite-cli", "timing-cli"])
def test_trailing_nul_ids_cannot_produce_unreloadable_archive(tmp_path: Path, writer: str) -> None:
    path = tmp_path / "result.npz"
    path.write_bytes(b"previous complete result")
    scores = np.asarray([0.1, 0.2])
    # These are distinct Python identifiers but become duplicate strings if
    # coerced to NumPy Unicode before validation.
    ids = ["case\0", "case"]
    with pytest.raises(ValueError, match="respondent IDs cannot end with a NUL"):
        if writer == "score":
            save_score_archive(path, {"irv": scores}, respondent_ids=ids)
        elif writer == "timing":
            save_response_time_archive(
                path, scores, [True, False], threshold=0.15, respondent_ids=ids
            )
        elif writer == "screen-cli":
            _write_screen_npz(path, screen_scores({"irv": scores}), ids)
        elif writer == "composite-cli":
            _write_composite_npz(path, CompositeReport(scores, "mean", ids))
        else:
            _write_response_time_npz(
                path, ResponseTimeReport(scores, scores <= 0.15, "median", "low", 0.15, ids)
            )
    assert path.read_bytes() == b"previous complete result"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("writer", ["score", "screen-cli", "composite-cli"])
def test_trailing_nul_error_messages_are_rejected_before_encoding(
    tmp_path: Path, writer: str
) -> None:
    path = tmp_path / "result.npz"
    errors = {"mad": "item pairs unavailable\0"}
    scores = np.asarray([0.1, 0.2])
    with pytest.raises(ValueError, match="error messages cannot end with a NUL"):
        if writer == "score":
            save_score_archive(path, {"irv": scores}, errors=errors)
        elif writer == "screen-cli":
            _write_screen_npz(path, screen_scores({"irv": scores}, errors=errors))
        else:
            _write_composite_npz(path, CompositeReport(scores, "mean", errors=errors))
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("compressed", [False, True])
def test_embedded_nul_and_unicode_metadata_remain_lossless(
    tmp_path: Path, compressed: bool
) -> None:
    ids = ["α\0one", "é-two"]
    errors = {"mad": "bad\0item pairs"}
    path = tmp_path / "score.npz"
    save_score_archive(
        path, {"irv": [0.1, 0.2]}, respondent_ids=ids, errors=errors, compressed=compressed
    )
    saved = load_score_archive(path)
    assert saved["respondent_ids"] == ids
    assert saved["errors"] == errors
    timing_path = tmp_path / "timing.npz"
    save_response_time_archive(
        timing_path,
        [0.1, 0.2],
        [True, False],
        threshold=0.15,
        respondent_ids=ids,
        compressed=compressed,
    )
    assert load_response_time_archive(timing_path)["respondent_ids"] == ids


@pytest.mark.parametrize(
    ("timing", "member"),
    [
        (False, "result_type"),
        (False, "index_names"),
        (False, "error_names"),
        (False, "error_messages"),
        (False, "respondent_ids"),
        (True, "result_type"),
        (True, "metric"),
        (True, "flag_direction"),
        (True, "respondent_ids"),
    ],
)
def test_invalid_unicode_codepoints_have_contextual_archive_error(
    tmp_path: Path, timing: bool, member: str
) -> None:
    path = tmp_path / "damaged.npz"
    payload = {
        "schema_version": np.asarray(1),
        "result_type": np.asarray("response_time" if timing else "screen"),
        "n_respondents": np.asarray(2),
        "respondent_ids": np.asarray(["case-1", "case-2"]),
    }
    if timing:
        payload.update(
            {
                "scores": np.asarray([0.1, 0.2]),
                "flags": np.asarray([True, False]),
                "threshold": np.asarray(0.15),
                "metric": np.asarray("median"),
                "flag_direction": np.asarray("low"),
            }
        )
    else:
        payload.update(
            {
                "index_names": np.asarray(["irv"]),
                "score__irv": np.asarray([0.1, 0.2]),
                "error_names": np.asarray(["mad"]),
                "error_messages": np.asarray(["unavailable item pairs"]),
            }
        )
    # NumPy's Unicode storage is UTF-32. This codepoint is outside the Unicode
    # range and normally raises an uncaught SystemError during .item/.tolist.
    payload[member].reshape(-1).view(np.uint32)[0] = 0x110000
    np.savez(path, **payload)
    loader = load_response_time_archive if timing else load_score_archive
    with pytest.raises(ValueError, match=f"member {member} contains invalid Unicode"):
        loader(path)


@pytest.mark.parametrize("dtype", ["<U6", ">U6"])
def test_unicode_validation_preserves_byte_order_and_supplementary_characters(
    tmp_path: Path, dtype: str
) -> None:
    path = tmp_path / "unicode.npz"
    ids = ["😀-one", "𐐀-two"]
    np.savez(
        path,
        schema_version=np.asarray(1),
        result_type=np.asarray("screen", dtype=dtype),
        n_respondents=np.asarray(2),
        index_names=np.asarray(["irv"], dtype=dtype),
        score__irv=np.asarray([0.1, 0.2]),
        respondent_ids=np.asarray(ids, dtype=dtype),
    )
    assert load_score_archive(path)["respondent_ids"] == ids


def test_unicode_validation_checks_codepoints_after_first_bounded_slice(tmp_path: Path) -> None:
    path = tmp_path / "long-error.npz"
    message = np.asarray(["a" * 70_000])
    message.view(np.uint32)[-1] = 0x110000
    np.savez(
        path,
        schema_version=np.asarray(1),
        result_type=np.asarray("screen"),
        n_respondents=np.asarray(2),
        index_names=np.asarray(["irv"]),
        score__irv=np.asarray([0.1, 0.2]),
        error_names=np.asarray(["mad"]),
        error_messages=message,
    )
    with pytest.raises(ValueError, match="member error_messages contains invalid Unicode"):
        load_score_archive(path)
