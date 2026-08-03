"""Benchmark validated reusable-score NPZ loading.

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

from ier import index_catalog, load_score_archive
from ier._cli_npz import _write_npz_archive

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--indices", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    available_names = list(index_catalog())
    if args.respondents < 1 or args.repeats < 1:
        parser.error("respondents and repeats must be positive")
    if not 1 <= args.indices <= len(available_names):
        parser.error(f"indices must be between 1 and {len(available_names)}")

    names = available_names[: args.indices]
    rng = np.random.default_rng(args.seed)
    payload = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "result_type": np.asarray("screen", dtype=np.str_),
        "n_respondents": np.asarray(args.respondents, dtype=np.int64),
        "index_names": np.asarray(names, dtype=np.str_),
        "error_names": np.asarray([], dtype=np.str_),
        "error_messages": np.asarray([], dtype=np.str_),
    }
    for name in names:
        payload[f"score__{name}"] = rng.normal(size=args.respondents)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scores.npz"
        _write_npz_archive(path, payload)

        raw_seconds, raw_peak, raw = _measure(lambda: _raw_load(path), args.repeats)
        validated_seconds, validated_peak, validated = _measure(
            lambda: load_score_archive(path)["scores"],
            args.repeats,
        )

    for name in names:
        np.testing.assert_array_equal(validated[name], raw[name])

    print(f"respondents={args.respondents} indices={args.indices} repeats={args.repeats}")
    print(f"raw load: median={raw_seconds:.4f}s peak={raw_peak:.1f} MiB")
    print(
        f"validated load: median={validated_seconds:.4f}s peak={validated_peak:.1f} MiB "
        f"overhead={validated_seconds / raw_seconds:.2f}x"
    )


if __name__ == "__main__":
    main()
