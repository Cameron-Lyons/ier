"""Compressed archives retain complete reusable results and atomic writes."""

from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pytest

from ier import (
    load_response_time_archive,
    load_score_archive,
    load_screen_archive,
    response_time_score_flags,
    save_response_time_archive,
    save_score_archive,
    save_screen_archive,
    screen_scores,
)
from ier._cli_composite import CompositeReport, ResponseTimeReport
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
                CompositeReport(
                    scores,
                    "mean",
                    ids,
                    component_scores={"irv": scores},
                    valid_index_counts=np.isfinite(scores).astype(np.int64),
                ),
                compressed=compressed,
            )
        else:
            _write_response_time_npz(
                path,
                ResponseTimeReport(scores, scores <= 0.25, "median", "low", 0.25, ids),
                compressed=compressed,
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


def _save(writer: str, path: Path, **options: object) -> None:
    """Write a small archive with one public writer and the given compression options."""
    scores = np.tile([0.1, 0.25, 0.75, np.nan], 256)
    if writer == "score":
        save_score_archive(path, {"irv": scores}, **options)  # type: ignore[arg-type]
    elif writer == "screen":
        result = screen_scores({"irv": scores}, thresholds={"irv": 0.2})
        save_screen_archive(path, result, **options)  # type: ignore[arg-type]
    else:
        save_response_time_archive(
            path,
            scores,
            scores <= 0.25,
            threshold=0.25,
            **options,  # type: ignore[arg-type]
        )


def _load_scores(writer: str, path: Path) -> np.ndarray:
    if writer == "score":
        return load_score_archive(path)["scores"]["irv"]
    if writer == "screen":
        return load_screen_archive(path)["result"]["scores"]["irv"]
    return load_response_time_archive(path)["scores"]


@pytest.mark.parametrize("level", [*range(1, 10), np.int64(9)])
@pytest.mark.parametrize("writer", ["score", "screen", "timing"])
def test_every_deflate_level_round_trips(tmp_path: Path, writer: str, level: int) -> None:
    path = tmp_path / "result.npz"
    _save(writer, path, compressed=True, compression_level=level)

    with ZipFile(path) as container:
        assert {member.compress_type for member in container.infolist()} == {ZIP_DEFLATED}
    np.testing.assert_array_equal(
        _load_scores(writer, path), np.tile([0.1, 0.25, 0.75, np.nan], 256)
    )


@pytest.mark.parametrize("level", [0, 10, -1, True, 1.5, "6", np.float64(3.0)])
@pytest.mark.parametrize("writer", ["score", "screen", "timing"])
def test_invalid_compression_level_is_rejected_before_destination_changes(
    tmp_path: Path, writer: str, level: object
) -> None:
    path = tmp_path / "result.npz"
    path.write_bytes(b"previous complete result")
    with pytest.raises(ValueError, match="compression_level must be an integer from 1 to 9"):
        _save(writer, path, compressed=True, compression_level=level)
    assert path.read_bytes() == b"previous complete result"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("writer", ["score", "screen", "timing"])
def test_compression_level_requires_compressed_output(tmp_path: Path, writer: str) -> None:
    path = tmp_path / "result.npz"
    path.write_bytes(b"previous complete result")
    with pytest.raises(ValueError, match="compression_level requires compressed=True"):
        _save(writer, path, compression_level=6)
    assert path.read_bytes() == b"previous complete result"
    _save(writer, path, compression_level=None)
    with ZipFile(path) as container:
        assert {member.compress_type for member in container.infolist()} == {ZIP_STORED}


def test_compression_level_is_applied_and_defaults_to_fastest(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    scores = {
        "longstring": np.tile(rng.integers(1, 10, size=997).astype(float), 300),
        "irv": np.tile(np.round(rng.random(1009), 2), 300)[:299_100],
    }
    sizes: dict[str, int] = {}
    for label, level in (("default", None), ("1", 1), ("6", 6), ("9", 9)):
        path = tmp_path / f"level-{label}.npz"
        save_score_archive(path, scores, compressed=True, compression_level=level)
        sizes[label] = path.stat().st_size
        loaded = load_score_archive(path)
        for name, values in scores.items():
            np.testing.assert_array_equal(loaded["scores"][name], values)

    assert sizes["default"] == sizes["1"]
    assert sizes["9"] < sizes["6"] < sizes["1"]
