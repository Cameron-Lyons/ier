"""Benchmark predefined semantic, MAD, and acquiescence item-pair scoring.

Usage:
    uv run python benchmarks/bench_pair_differences.py
    uv run python benchmarks/bench_pair_differences.py --respondents 200000 --items 100
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
from _measurement import measure

from ier import acquiescence, mad, semantic_ant, semantic_syn

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(
    operation: Callable[[], np.ndarray], repeats: int, available: np.ndarray
) -> tuple[float, float]:
    measurement = measure(operation, repeats)
    result = measurement.result
    if not np.array_equal(np.isfinite(result), available) or not np.isnan(result[~available]).all():
        raise RuntimeError("benchmark produced incorrect score availability")
    return measurement.median_seconds, measurement.peak_mib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--missing-rate", type=float, default=0.1)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--dtype", choices=("float64", "int64", "uint64"), default="float64")
    parser.add_argument("--integer-offset", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if args.respondents < 1 or args.items < 2 or args.repeats < 1 or args.warmup < 0:
        parser.error(
            "respondents and repeats must be positive, items at least 2, and warmup nonnegative"
        )
    if not 0.0 <= args.missing_rate <= 1.0:
        parser.error("missing-rate must be between 0 and 1")
    if not np.isfinite(args.scale) or args.scale <= 0 or not np.isfinite(args.offset):
        parser.error("scale must be positive and finite, and offset must be finite")
    scale_min = args.offset + args.scale
    scale_max = args.offset + 5 * args.scale
    if not np.isfinite([scale_min, scale_max]).all() or scale_min >= scale_max:
        parser.error("scale and offset must produce distinct finite response bounds")
    integer_input = np.dtype(args.dtype).kind in "iu"
    if integer_input:
        if args.missing_rate or args.scale != 1 or args.offset:
            parser.error("integer inputs require --missing-rate 0 and use --integer-offset")
        scale_min, scale_max = args.integer_offset + 1, args.integer_offset + 5
        limits = np.iinfo(args.dtype)
        if scale_min < limits.min or scale_max > limits.max:
            parser.error("integer-offset places responses outside the selected dtype")
    elif args.integer_offset:
        parser.error("integer-offset requires an integer dtype")

    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, 5, size=(args.respondents, args.items)).astype(args.dtype)
    if integer_input:
        data += scale_min
    else:
        data += 1
        with np.errstate(under="ignore"):
            data *= args.scale
            data += args.offset
    if args.missing_rate:
        data[rng.random(data.shape) < args.missing_rate] = np.nan
    data = np.asarray(data, order=args.order)

    n_pairs = args.items // 2
    positive_items = list(range(n_pairs))
    negative_items = list(range(n_pairs, n_pairs * 2))
    pairs = list(zip(positive_items, negative_items, strict=True))
    available = np.any(
        np.isfinite(data[:, positive_items]) & np.isfinite(data[:, negative_items]), axis=1
    )
    operations: dict[str, Callable[[], np.ndarray]] = {
        "acquiescence": lambda: acquiescence(
            data,
            scale_min=scale_min,
            scale_max=scale_max,
            positive_items=positive_items,
            negative_items=negative_items,
        ),
        "semantic_syn": lambda: semantic_syn(data, pairs),
        "semantic_ant": lambda: semantic_ant(data, pairs, scale_min=scale_min, scale_max=scale_max),
        "mad": lambda: mad(data, item_pairs=pairs, scale_min=scale_min, scale_max=scale_max),
    }

    for _ in range(args.warmup):
        for operation in operations.values():
            operation()

    print(
        f"shape={data.shape} pairs={n_pairs} missing_rate={args.missing_rate} "
        f"order={args.order} scale={args.scale} offset={args.offset} "
        f"dtype={args.dtype} integer_offset={args.integer_offset} "
        f"available={np.count_nonzero(available)} repeats={args.repeats} warmup={args.warmup}"
    )
    for name, operation in operations.items():
        seconds, peak = _measure(operation, args.repeats, available)
        print(f"{name}: median={seconds:.4f}s peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()
