"""Benchmark validated score and response-time NPZ persistence.

Usage:
    uv run python benchmarks/bench_archive.py
    uv run python benchmarks/bench_archive.py --respondents 500000 --indices 15
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from _measurement import measure_many

from ier import (
    index_catalog,
    load_response_time_archive,
    load_score_archive,
    save_response_time_archive,
    save_score_archive,
)
from ier.archive import _stream_npz_archive


def _raw_load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["index_names"].tolist()
        return {name: archive[f"score__{name}"] for name in names}


def _raw_response_time_load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {"scores": archive["scores"], "flags": archive["flags"]}


def _validated_response_time_load(path: Path) -> dict[str, np.ndarray]:
    loaded = load_response_time_archive(path)
    return {"scores": loaded["scores"], "flags": loaded["flags"]}


def _raw_response_time_save(
    path: Path,
    scores: np.ndarray,
    flags: np.ndarray,
    threshold: float,
) -> None:
    _stream_npz_archive(
        path,
        {
            "schema_version": np.asarray(1, dtype=np.int64),
            "result_type": np.asarray("response_time", dtype=np.str_),
            "n_respondents": np.asarray(len(scores), dtype=np.int64),
            "metric": np.asarray("median", dtype=np.str_),
            "flag_direction": np.asarray("low", dtype=np.str_),
            "threshold": np.asarray(threshold, dtype=np.float64),
            "scores": scores,
            "flags": flags,
        },
    )


def _raw_save(path: Path, scores: dict[str, np.ndarray]) -> None:
    payload = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "result_type": np.asarray("screen", dtype=np.str_),
        "n_respondents": np.asarray(len(next(iter(scores.values()))), dtype=np.int64),
        "index_names": np.asarray(list(scores), dtype=np.str_),
        "error_names": np.asarray([], dtype=np.str_),
        "error_messages": np.asarray([], dtype=np.str_),
    }
    for name, values in scores.items():
        payload[f"score__{name}"] = values
    _stream_npz_archive(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--indices", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--write-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    available_names = list(index_catalog())
    if args.respondents < 1 or args.repeats < 1 or args.write_repeats < 1:
        parser.error("respondents, repeats, and write-repeats must be positive")
    if not 1 <= args.indices <= len(available_names):
        parser.error(f"indices must be between 1 and {len(available_names)}")

    names = available_names[: args.indices]
    rng = np.random.default_rng(args.seed)
    scores = {name: rng.normal(size=args.respondents) for name in names}
    timing_scores = rng.lognormal(size=args.respondents)
    timing_threshold = float(np.percentile(timing_scores, 5))
    timing_flags = timing_scores < timing_threshold

    with tempfile.TemporaryDirectory() as directory:
        raw_path = Path(directory) / "raw-scores.npz"
        validated_path = Path(directory) / "validated-scores.npz"
        raw_timing_path = Path(directory) / "raw-timing.npz"
        timing_path = Path(directory) / "validated-timing.npz"
        _raw_save(raw_path, scores)
        save_score_archive(validated_path, scores)
        _raw_response_time_save(
            raw_timing_path,
            timing_scores,
            timing_flags,
            timing_threshold,
        )
        save_response_time_archive(
            timing_path,
            timing_scores,
            timing_flags,
            threshold=timing_threshold,
        )

        loads = measure_many(
            {
                "raw": lambda: _raw_load(validated_path),
                "validated": lambda: load_score_archive(validated_path)["scores"],
                "raw_timing": lambda: _raw_response_time_load(timing_path),
                "validated_timing": lambda: _validated_response_time_load(timing_path),
            },
            args.repeats,
        )
        writes = measure_many(
            {
                "raw": lambda: _raw_save(raw_path, scores),
                "validated": lambda: save_score_archive(validated_path, scores),
                "raw_timing": lambda: _raw_response_time_save(
                    raw_timing_path, timing_scores, timing_flags, timing_threshold
                ),
                "validated_timing": lambda: save_response_time_archive(
                    timing_path, timing_scores, timing_flags, threshold=timing_threshold
                ),
            },
            args.write_repeats,
        )

    for name in names:
        np.testing.assert_array_equal(loads["validated"].result[name], loads["raw"].result[name])
    for name in ("scores", "flags"):
        np.testing.assert_array_equal(
            loads["validated_timing"].result[name], loads["raw_timing"].result[name]
        )

    print(
        f"respondents={args.respondents} indices={args.indices} "
        f"load_repeats={args.repeats} write_repeats={args.write_repeats}"
    )
    for label, measurements in (("load", loads), ("save", writes)):
        for prefix, raw_name, validated_name in (
            ("", "raw", "validated"),
            ("response-time ", "raw_timing", "validated_timing"),
        ):
            raw = measurements[raw_name]
            validated = measurements[validated_name]
            print(
                f"raw {prefix}{label}: median={raw.median_seconds:.4f}s peak={raw.peak_mib:.1f} MiB"
            )
            print(
                f"validated {prefix}{label}: median={validated.median_seconds:.4f}s "
                f"peak={validated.peak_mib:.1f} MiB "
                f"overhead={validated.median_seconds / raw.median_seconds:.2f}x"
            )


if __name__ == "__main__":
    main()
