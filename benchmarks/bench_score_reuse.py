"""Compare complete CLI runs with saved-score sensitivity analysis.

Usage:
    uv run python benchmarks/bench_score_reuse.py
    uv run python benchmarks/bench_score_reuse.py --workflow composite

Both paths include input loading and NPZ serialization. Inputs and the initial
archive are prepared outside measurement. Every saved output member is checked
against a fresh matrix-scoring run after timing.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from _measurement import measure_many

from ier.cli import main as run_cli


def _run(arguments: list[str]) -> None:
    if run_cli(arguments) != 0:
        raise RuntimeError("CLI benchmark scoring failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workflow", choices=["screen", "composite"], default="screen")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.respondents < 2 or args.items < 2 or args.repeats < 1:
        parser.error("respondents and items must be at least 2; repeats must be positive")
    data = np.random.default_rng(args.seed).integers(1, 6, (args.respondents, args.items))
    data[0].fill(3)
    indices = ["irv", "longstring", "person_total", "markov", "mahad"]
    decision_options = ["--percentile", "90"]
    if args.workflow == "composite":
        decision_options += ["--include-components", "--weight", "irv=2"]
    shared = ["--indices", *indices, *decision_options, "--format", "npz"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        matrix_path = root / "survey.npy"
        archive_path = root / "initial.npz"
        full_path = root / "full.npz"
        reused_path = root / "reused.npz"
        np.save(matrix_path, data, allow_pickle=False)
        _run([args.workflow, str(matrix_path), *shared, "--output", str(archive_path)])
        measurements = measure_many(
            {
                "full": lambda: _run(
                    [args.workflow, str(matrix_path), *shared, "--output", str(full_path)]
                ),
                "reused": lambda: _run(
                    [
                        f"{args.workflow}-scores",
                        str(archive_path),
                        *shared,
                        "--output",
                        str(reused_path),
                    ]
                ),
            },
            args.repeats,
        )
        with (
            np.load(full_path, allow_pickle=False) as full,
            np.load(reused_path, allow_pickle=False) as reused,
        ):
            assert set(full.files) == set(reused.files)
            for name in full.files:
                np.testing.assert_array_equal(full[name], reused[name], err_msg=name)

    print(f"workflow={args.workflow} respondents={args.respondents} items={args.items}")
    for name, measured in measurements.items():
        print(f"{name}: median={measured.median_seconds:.4f}s peak={measured.peak_mib:.1f} MiB")
    speedup = measurements["full"].median_seconds / measurements["reused"].median_seconds
    print(f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
