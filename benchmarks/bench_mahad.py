"""Benchmark bounded Mahalanobis distance scoring.

Usage:
    uv run python benchmarks/bench_mahad.py
    uv run python benchmarks/bench_mahad.py --respondents 200000 --items 100
    uv run python benchmarks/bench_mahad.py --na-rm --missing-row-rate 0.1
    uv run python benchmarks/bench_mahad.py --operation qq --items 2 --respondents 10000

Timing excludes allocation tracing; peak allocation is measured separately.
"""

from __future__ import annotations

import argparse
import platform

import numpy as np
from _measurement import measure

from ier import mahad, mahad_qqplot


def _measure(data: np.ndarray, repeats: int, *, na_rm: bool, operation: str) -> tuple[float, float]:
    measurement = measure(
        lambda: mahad_qqplot(data, na_rm=na_rm) if operation == "qq" else mahad(data, na_rm=na_rm),
        repeats,
    )
    result = measurement.result
    valid = ~np.isnan(data).any(axis=1)
    if operation == "qq":
        if not isinstance(result, tuple):
            raise RuntimeError("benchmark expected theoretical and observed quantiles")
        for quantiles in result:
            if (
                quantiles.shape != (int(np.count_nonzero(valid)),)
                or not np.isfinite(quantiles).all()
                or np.any(quantiles < 0)
                or np.any(np.diff(quantiles) < 0)
            ):
                raise RuntimeError("benchmark produced invalid Q-Q quantiles")
        return measurement.median_seconds, measurement.peak_mib
    if not isinstance(result, np.ndarray):
        raise RuntimeError("benchmark expected distance-only output")
    if not np.isfinite(result[valid]).all() or not np.isnan(result[~valid]).all():
        raise RuntimeError("benchmark produced invalid distances")
    return measurement.median_seconds, measurement.peak_mib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--operation", choices=("distance", "qq"), default="distance")
    parser.add_argument("--na-rm", action="store_true", help="Enable complete-case handling")
    parser.add_argument("--missing-row-rate", type=float, default=0.0)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    if (
        args.respondents < max(2, args.items)
        or args.items < 1
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error(
            "respondents must be at least max(2, items), items and repeats positive, "
            "and warmup nonnegative"
        )
    if not 0.0 <= args.missing_row_rate < 1.0:
        parser.error("missing-row-rate must be in [0, 1)")

    rng = np.random.default_rng(args.seed)
    data = np.array(rng.normal(size=(args.respondents, args.items)), order=args.order)
    missing_rows = rng.random(len(data)) < args.missing_row_rate
    n_complete = int(np.count_nonzero(~missing_rows))
    if n_complete < max(2, args.items):
        parser.error("missing-row-rate leaves insufficient complete observations")
    data[missing_rows, 0] = np.nan
    na_rm = args.na_rm or args.missing_row_rate > 0.0

    for _ in range(args.warmup):
        if args.operation == "qq":
            mahad_qqplot(data, na_rm=na_rm)
        else:
            mahad(data, na_rm=na_rm)

    seconds, peak = _measure(data, args.repeats, na_rm=na_rm, operation=args.operation)
    print(f"Python {platform.python_version()} / NumPy {np.__version__}")
    print(
        f"shape={data.shape} complete_rows={n_complete} na_rm={na_rm} order={args.order} "
        f"missing_row_rate={args.missing_row_rate} repeats={args.repeats} "
        f"warmup={args.warmup} seed={args.seed}"
    )
    print(f"mahad_{args.operation}: median={seconds:.4f}s peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()
