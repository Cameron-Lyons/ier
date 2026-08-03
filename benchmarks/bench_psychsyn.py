"""Benchmark direct and fixed-model psychometric synonym scoring.

Missing responses are distributed across the matrix so the benchmark exercises
pairwise-complete item discovery and respondent scoring. Independent data with
a high cutoff isolates the bounded no-pair discovery path. Each scenario also
measures scoring with one retained item-pair calibration.
The retained path is round-tripped through the public archive boundary, and
model load latency plus archive size are reported separately.

Usage:
    uv run python benchmarks/bench_psychsyn.py
    uv run python benchmarks/bench_psychsyn.py --respondents 16000 --items 50
    uv run python benchmarks/bench_psychsyn.py --respondents 100000 --items 80 \
        --critval 0.99 --missing-rate 0 --independent
    uv run python benchmarks/bench_psychsyn.py --respondents 200 --items 3000 \
        --critval 0.99 --missing-rate 0 --independent
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from ier import (
    IndexOptions,
    fit_psychsyn_model,
    load_psychsyn_model,
    psychsyn,
    psychsyn_model_scores,
    save_psychsyn_model,
    screen,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(
    operation: Callable[[], tuple[np.ndarray, np.ndarray]],
    repeats: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Measure one psychometric scoring operation."""
    timings: list[float] = []
    peaks: list[int] = []
    result: np.ndarray | None = None
    diagnostic: np.ndarray | None = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        result, diagnostic = operation()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert result is not None and diagnostic is not None
    if not (np.isfinite(result).all() or np.isnan(result).all()):
        raise RuntimeError("benchmark produced non-finite psychometric synonym scores")
    return (
        statistics.median(timings),
        statistics.median(peaks) / 1024 / 1024,
        result,
        diagnostic,
    )


def _measure_action(operation: Callable[[], object], repeats: int) -> tuple[float, float]:
    """Measure one shared workflow without retaining its structured result."""
    timings: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    return statistics.median(timings), statistics.median(peaks) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=8_000)
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--critval", type=float, default=0.6)
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.respondents < 2 or args.items < 3 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents must be at least 2, items at least 3, repeats positive, "
            "and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if not 0.0 <= args.critval <= 1.0:
        parser.error("critval must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    if args.independent:
        data = rng.normal(size=(args.respondents, args.items))
    else:
        latent = rng.normal(size=(args.respondents, 1))
        data = latent + rng.normal(scale=0.1, size=(args.respondents, args.items))
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan

    fitted_model = fit_psychsyn_model(data, critval=args.critval)
    load_timings: list[float] = []
    with TemporaryDirectory() as directory:
        model_path = Path(directory) / "psychsyn-model.npz"
        save_psychsyn_model(model_path, fitted_model)
        archive_bytes = model_path.stat().st_size
        model = fitted_model
        for _ in range(args.repeats):
            started = time.perf_counter()
            model = load_psychsyn_model(model_path)
            load_timings.append(time.perf_counter() - started)
    for _ in range(args.warmup):
        psychsyn(data, critval=args.critval)
        psychsyn_model_scores(data, model)

    direct_seconds, direct_peak, result, diagnostic = _measure(
        lambda: psychsyn(data, critval=args.critval, diag=True),
        args.repeats,
    )
    fixed_seconds, fixed_peak, fixed_result, fixed_diagnostic = _measure(
        lambda: psychsyn_model_scores(data, model, diag=True),
        args.repeats,
    )
    np.testing.assert_array_equal(fixed_result, result)
    np.testing.assert_array_equal(fixed_diagnostic, diagnostic)
    shared_options = IndexOptions(psychsyn_model=model)
    shared_result = screen(
        data,
        indices=["psychsyn"],
        options=shared_options,
        thresholds={"psychsyn": 0.0},
        min_flags=1,
    )
    np.testing.assert_array_equal(shared_result["scores"]["psychsyn"], fixed_result)
    shared_seconds, shared_peak = _measure_action(
        lambda: screen(
            data,
            indices=["psychsyn"],
            options=shared_options,
            thresholds={"psychsyn": 0.0},
            min_flags=1,
        ),
        args.repeats,
    )

    print(
        f"shape={data.shape} selected_pairs={model.n_pairs} "
        f"missing_rate={args.missing_rate} independent={args.independent} "
        f"repeats={args.repeats} warmup={args.warmup}"
    )
    print(f"psychsyn: median={direct_seconds:.4f}s peak={direct_peak:.1f} MiB")
    print(
        f"fixed model: median={fixed_seconds:.4f}s peak={fixed_peak:.1f} MiB "
        f"speedup={direct_seconds / fixed_seconds:.1f}x"
    )
    print(
        f"model archive: size={archive_bytes / 1024:.1f} KiB "
        f"load_median={statistics.median(load_timings):.4f}s"
    )
    print(f"shared fixed-model screen: median={shared_seconds:.4f}s peak={shared_peak:.1f} MiB")


if __name__ == "__main__":
    main()
