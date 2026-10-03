"""Compressed archives retain complete reusable results and atomic writes."""

from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pytest

from ier import (
    load_response_time_archive,
    load_score_archive,
    response_time_score_flags,
    save_response_time_archive,
    save_score_archive,
    screen_scores,
)
from ier._cli_npz import _write_composite_npz, _write_response_time_npz, _write_screen_npz


@pytest.mark.parametrize("result_type", ["screen", "composite"])
def test_compressed_score_archives_reduce_storage_and_preserve_reuse(
    tmp_path: Path, result_type: str
) -> None:
    scores = {
        "irv": np.tile([0.0, 0.25, 1.0, np.nan], 2048),
        "longstring": np.tile([8.0, 2.0, 1.0, 3.0], 2048),
    }
    ids = [f"répondant-{index:05}" for index in range(8192)]
    errors = {"mad": "item pairs were unavailable"}
    paths = [tmp_path / "stored.npz", tmp_path / "compressed.npz"]
    for path, compressed in zip(paths, [False, True], strict=True):
        save_score_archive(
            path,
            scores,
            result_type=result_type,  # type: ignore[arg-type]
            respondent_ids=ids,
            errors=errors,
            compressed=compressed,
        )
        with ZipFile(path) as container:
            assert {member.compress_type for member in container.infolist()} == {
                ZIP_DEFLATED if compressed else ZIP_STORED
            }
        loaded = load_score_archive(path)
        assert loaded["result_type"] == result_type
        assert loaded["respondent_ids"] == ids
        assert loaded["errors"] == errors
        assert list(loaded["scores"]) == list(scores)
        for name, values in scores.items():
            np.testing.assert_array_equal(loaded["scores"][name], values)
        reused = screen_scores(
            loaded["scores"], thresholds={"irv": 0.1, "longstring": 7.0}, min_flags=2
        )
        np.testing.assert_array_equal(
            reused["consensus_flags"], np.tile([True, False, False, False], 2048)
        )
        np.testing.assert_array_equal(reused["valid_index_counts"], np.tile([2, 2, 2, 1], 2048))

    assert paths[1].stat().st_size < paths[0].stat().st_size / 4


@pytest.mark.parametrize("metric", ["median", "mixture"])
def test_compressed_timing_round_trip_reflags_without_original_data(
    tmp_path: Path, metric: str
) -> None:
    scores = np.tile([0.1, 0.5, 0.9, np.nan], 1024)
    direction = "high" if metric == "mixture" else "low"
    flags = scores >= 0.5 if direction == "high" else scores <= 0.5
    ids = [f"case-{index}" for index in range(len(scores))]
    path = tmp_path / "timing.npz"
    save_response_time_archive(
        path,
        scores,
        flags,
        threshold=0.5,
        metric=metric,  # type: ignore[arg-type]
        flag_direction=direction,  # type: ignore[arg-type]
        respondent_ids=ids,
        compressed=True,
    )
    loaded = load_response_time_archive(path)
    assert loaded["respondent_ids"] == ids
    assert loaded["metric"] == metric
    assert loaded["flag_direction"] == direction
    np.testing.assert_array_equal(loaded["scores"], scores)
    np.testing.assert_array_equal(loaded["flags"], flags)
    reflagged = response_time_score_flags(
        loaded["scores"], cutoff_percentile=50, direction=loaded["flag_direction"]
    )
    expected = scores > 0.5 if direction == "high" else scores < 0.5
    np.testing.assert_array_equal(reflagged, expected)


@pytest.mark.parametrize("output", ["screen", "composite", "response_time"])
def test_cli_npz_serializers_compress_every_member_without_changing_payload(
    tmp_path: Path, output: str
) -> None:
    scores = np.asarray([0.1, 0.25, 0.75, np.nan])
    ids = ["α", "β", "γ", "δ"]
    original: dict[str, np.ndarray] | None = None
    for compressed in [False, True]:
        path = tmp_path / f"{output}-{compressed}.npz"
        if output == "screen":
            result = screen_scores({"irv": scores}, thresholds={"irv": 0.2})
            _write_screen_npz(path, result, ids, compressed=compressed)
        elif output == "composite":
            _write_composite_npz(
                path,
                scores,
                "mean",
                ids,
                component_scores={"irv": scores},
                valid_index_counts=np.isfinite(scores).astype(np.int64),
                compressed=compressed,
            )
        else:
            _write_response_time_npz(
                path, scores, scores <= 0.25, "median", "low", 0.25, ids, compressed=compressed
            )
        with ZipFile(path) as container:
            assert {member.compress_type for member in container.infolist()} == {
                ZIP_DEFLATED if compressed else ZIP_STORED
            }
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        if original is None:
            original = payload
        else:
            assert list(payload) == list(original)
            for name, values in original.items():
                np.testing.assert_array_equal(payload[name], values)
                assert payload[name].dtype == values.dtype


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_compressed_write_interruption_preserves_destination(
    tmp_path: Path, existing: bool, error: type[BaseException]
) -> None:
    path = tmp_path / "scores.npz"
    if existing:
        save_score_archive(path, {"irv": [1.0, 2.0]})
    previous = path.read_bytes() if existing else None
    original_save = np.save
    calls = 0

    def interrupt(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise error("stream interrupted after several compressed members")
        original_save(*args, **kwargs)  # type: ignore[arg-type]

    with (
        patch("ier.archive.np.save", side_effect=interrupt),
        pytest.raises(error, match="interrupted"),
    ):
        save_score_archive(path, {"irv": [0.1, 0.2]}, compressed=True)
    if existing:
        assert path.read_bytes() == previous
        np.testing.assert_array_equal(load_score_archive(path)["scores"]["irv"], [1.0, 2.0])
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


@pytest.mark.parametrize("value", [0, 1, "true", None])
@pytest.mark.parametrize("timing", [False, True])
def test_invalid_compression_option_is_rejected_before_destination_changes(
    tmp_path: Path, value: object, timing: bool
) -> None:
    path = tmp_path / "result.npz"
    path.write_bytes(b"previous complete result")
    with pytest.raises(ValueError, match="compressed must be a boolean"):
        if timing:
            save_response_time_archive(
                path,
                [0.1, 0.2],
                [True, False],
                threshold=0.15,
                compressed=value,  # type: ignore[arg-type]
            )
        else:
            save_score_archive(path, {"irv": [0.1, 0.2]}, compressed=value)  # type: ignore[arg-type]
    assert path.read_bytes() == b"previous complete result"
    assert list(tmp_path.iterdir()) == [path]
