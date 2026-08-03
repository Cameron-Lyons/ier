"""Benchmark validated score and response-time NPZ persistence.

Usage:
    uv run python benchmarks/bench_archive.py
    uv run python benchmarks/bench_archive.py --respondents 500000 --indices 15
"""

from __future__ import annotations

import argparse
import gc
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ier import (
    index_catalog,
    load_response_time_archive,
    load_score_archive,
    save_score_archive,
)
from ier.archive import _write_npz_archive

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(
    operation: Callable[[], dict[str, np.ndarray]],
    repeats: int,
) -> tuple[float, float, dict[str, np.ndarray]]:
    timings: list[float] = []
    peaks: list[int] = []
    result: dict[str, np.ndarray] | None = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    assert result is not None
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024, result


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
    _write_npz_archive(path, payload)


def _measure_write_pair(
    raw_operation: Callable[[], None],
    validated_operation: Callable[[], None],
    repeats: int,
) -> tuple[float, float, float, float]:
    timings: dict[str, list[float]] = {"raw": [], "validated": []}
    peaks: dict[str, list[int]] = {"raw": [], "validated": []}
    operations = {"raw": raw_operation, "validated": validated_operation}
    for repeat in range(repeats):
        order = ("raw", "validated") if repeat % 2 == 0 else ("validated", "raw")
        for label in order:
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            operations[label]()
            timings[label].append(time.perf_counter() - started)
            peaks[label].append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
    return (
        statistics.median(timings["raw"]),
        statistics.median(peaks["raw"]) / 1024 / 1024,
        statistics.median(timings["validated"]),
        statistics.median(peaks["validated"]) / 1024 / 1024,
    )


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
        timing_path = Path(directory) / "timing.npz"
        _raw_save(raw_path, scores)
        save_score_archive(validated_path, scores)
        _write_npz_archive(
            timing_path,
            {
                "schema_version": np.asarray(1, dtype=np.int64),
                "result_type": np.asarray("response_time", dtype=np.str_),
                "n_respondents": np.asarray(args.respondents, dtype=np.int64),
                "metric": np.asarray("median", dtype=np.str_),
                "flag_direction": np.asarray("low", dtype=np.str_),
                "threshold": np.asarray(timing_threshold, dtype=np.float64),
                "scores": timing_scores,
                "flags": timing_flags,
            },
        )

        raw_seconds, raw_peak, raw = _measure(lambda: _raw_load(validated_path), args.repeats)
        validated_seconds, validated_peak, validated = _measure(
            lambda: load_score_archive(validated_path)["scores"],
            args.repeats,
        )
        raw_timing_seconds, raw_timing_peak, raw_timing = _measure(
            lambda: _raw_response_time_load(timing_path),
            args.repeats,
        )
        validated_timing_seconds, validated_timing_peak, validated_timing = _measure(
            lambda: _validated_response_time_load(timing_path),
            args.repeats,
        )
        raw_save_seconds, raw_save_peak, validated_save_seconds, validated_save_peak = (
            _measure_write_pair(
                lambda: _raw_save(raw_path, scores),
                lambda: save_score_archive(validated_path, scores),
                args.write_repeats,
            )
        )

    for name in names:
        np.testing.assert_array_equal(validated[name], raw[name])
    for name in ("scores", "flags"):
        np.testing.assert_array_equal(validated_timing[name], raw_timing[name])

    print(
        f"respondents={args.respondents} indices={args.indices} "
        f"load_repeats={args.repeats} write_repeats={args.write_repeats}"
    )
    print(f"raw load: median={raw_seconds:.4f}s peak={raw_peak:.1f} MiB")
    print(
        f"validated load: median={validated_seconds:.4f}s peak={validated_peak:.1f} MiB "
        f"overhead={validated_seconds / raw_seconds:.2f}x"
    )
    print(
        f"raw response-time load: median={raw_timing_seconds:.4f}s peak={raw_timing_peak:.1f} MiB"
    )
    print(
        f"validated response-time load: median={validated_timing_seconds:.4f}s "
        f"peak={validated_timing_peak:.1f} MiB "
        f"overhead={validated_timing_seconds / raw_timing_seconds:.2f}x"
    )
    print(f"raw save: median={raw_save_seconds:.4f}s peak={raw_save_peak:.1f} MiB")
    print(
        f"validated save: median={validated_save_seconds:.4f}s "
        f"peak={validated_save_peak:.1f} MiB "
        f"overhead={validated_save_seconds / raw_save_seconds:.2f}x"
    )


if __name__ == "__main__":
    main()
