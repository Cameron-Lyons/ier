"""Benchmark bounded carelessness-onset scoring on complete and missing responses.

Usage:
    uv run python benchmarks/bench_onset.py
    uv run python benchmarks/bench_onset.py --respondents 200000 --items 100
    uv run python benchmarks/bench_onset.py --respondents 10000 --missing-rate 0.1
    uv run python benchmarks/bench_onset.py --order F --missing-rate 1
    uv run python benchmarks/bench_onset.py --items 200 --window-size 50 --min-items 50
    uv run python benchmarks/bench_onset.py --dtype int64 --integer-offset 1152921504606846976
"""

from __future__ import annotations

import argparse

import numpy as np
from _measurement import measure

from ier import onset


def _measure(
    data: np.ndarray,
    *,
    window_size: int,
    min_items: int,
    repeats: int,
    expected_sample: np.ndarray | None,
) -> tuple[float, float]:
    measurement = measure(
        lambda: onset(data, window_size=window_size, min_items=min_items), repeats
    )
    result = measurement.result
    detected = np.isfinite(result)
    observed = np.count_nonzero(~np.isnan(data), axis=1)
    if (
        np.isinf(result).any()
        or np.any(observed[detected] < max(min_items, window_size + 2))
        or np.any(result[detected] < window_size)
        or np.any(result[detected] >= observed[detected])
        or np.any(result[detected] != np.floor(result[detected]))
    ):
        raise RuntimeError("benchmark produced invalid onset scores")
    if expected_sample is not None:
        np.testing.assert_array_equal(result[: len(expected_sample)], expected_sample)
    return measurement.median_seconds, measurement.peak_mib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--categories", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--min-items", type=int, default=20)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument(
        "--dtype", choices=("float32", "float64", "int64", "uint64"), default="float64"
    )
    parser.add_argument("--integer-offset", type=int, default=0)
    parser.add_argument("--structure", choices=("categorical", "continuous"), default="categorical")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if (
        args.respondents < 1
        or args.items < 1
        or args.categories < 2
        or args.window_size < 2
        or args.min_items < args.window_size
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error(
            "respondents, items, and repeats must be positive, categories and "
            "window-size at least 2, min-items at least window-size, "
            "and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    integer_input = np.dtype(args.dtype).kind in "iu"
    if integer_input:
        if args.missing_rate or args.structure != "categorical":
            parser.error("integer inputs require --missing-rate 0 and --structure categorical")
        limits = np.iinfo(args.dtype)
        if (
            args.integer_offset + 1 < limits.min
            or args.integer_offset + args.categories > limits.max
        ):
            parser.error("integer-offset places responses outside the selected dtype")
    elif args.integer_offset:
        parser.error("integer-offset requires an integer dtype")

    rng = np.random.default_rng(args.seed)
    shape = (args.respondents, args.items)
    data = (
        rng.integers(0, args.categories, size=shape)
        if args.structure == "categorical"
        else rng.uniform(0, args.categories - 1, size=shape)
    ).astype(args.dtype)
    expected_sample = (
        onset(data[:8].astype(float), window_size=args.window_size, min_items=args.min_items)
        if integer_input
        else None
    )
    data += args.integer_offset + 1
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    if args.order == "F":
        data = np.asfortranarray(data)

    for _ in range(args.warmup):
        onset(
            data,
            window_size=args.window_size,
            min_items=args.min_items,
        )

    seconds, peak = _measure(
        data,
        window_size=args.window_size,
        min_items=args.min_items,
        repeats=args.repeats,
        expected_sample=expected_sample,
    )
    print(
        f"shape={data.shape} categories={args.categories} dtype={data.dtype} "
        f"structure={args.structure} integer_offset={args.integer_offset} "
        f"window_size={args.window_size} min_items={args.min_items} "
        f"missing_rate={args.missing_rate} order={args.order} repeats={args.repeats} "
        f"warmup={args.warmup}"
    )
    print(f"onset: median={seconds:.4f}s peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()
