"""Compare complete CLI runs with saved-score sensitivity analysis.

Usage:
    uv run python benchmarks/bench_score_reuse.py
    uv run python benchmarks/bench_score_reuse.py --workflow composite
    uv run python benchmarks/bench_score_reuse.py --workflow response-time

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
    parser.add_argument(
        "--workflow", choices=["screen", "composite", "response-time"], default="screen"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.respondents < 2 or args.items < 2 or args.repeats < 1:
        parser.error("respondents and items must be at least 2; repeats must be positive")
    rng = np.random.default_rng(args.seed)
    indices = ["irv", "longstring", "person_total", "markov", "mahad"]
    data: np.ndarray
    if args.workflow == "response-time":
        respondent_times = rng.lognormal(1.5, 0.4, (args.respondents, 1))
        respondent_times[: max(1, args.respondents // 5)] *= 0.15
        data = respondent_times * rng.lognormal(0.0, 0.25, (args.respondents, args.items))
        if args.respondents > 2:
            data[-1].fill(np.nan)
        scoring_options = [
            "--metric",
            "mixture",
            "--components",
            "2",
            "--random-seed",
            str(args.seed),
        ]
        selection_options: list[str] = []
    else:
        data = rng.integers(1, 6, (args.respondents, args.items))
        data[0].fill(3)
        scoring_options = ["--indices", *indices]
        selection_options = scoring_options
    output_options = ["--format", "npz"]
    if args.workflow == "composite":
        output_options += ["--include-components", "--weight", "irv=2"]
    decision_options = ["--percentile", "90"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        matrix_path = root / "survey.npy"
        archive_path = root / "initial.npz"
        full_path = root / "full.npz"
        reused_path = root / "reused.npz"
        np.save(matrix_path, data, allow_pickle=False)
        _run(
            [
                args.workflow,
                str(matrix_path),
                *scoring_options,
                "--percentile",
                "95",
                *output_options,
                "--output",
                str(archive_path),
            ]
        )
        measurements = measure_many(
            {
                "full": lambda: _run(
                    [
                        args.workflow,
                        str(matrix_path),
                        *scoring_options,
                        *decision_options,
                        *output_options,
                        "--output",
                        str(full_path),
                    ]
                ),
                "reused": lambda: _run(
                    [
                        f"{args.workflow}-scores",
                        str(archive_path),
                        *selection_options,
                        *decision_options,
                        *output_options,
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
            if args.workflow == "response-time":
                probabilities = reused["scores"]
                observed = np.isfinite(probabilities)
                np.testing.assert_array_equal(observed, np.any(np.isfinite(data), axis=1))
                assert np.all((probabilities[observed] >= 0) & (probabilities[observed] <= 1))
                expected_cutoff = np.percentile(probabilities[observed], 90)
                np.testing.assert_allclose(reused["threshold"], expected_cutoff, rtol=1e-14)
                np.testing.assert_array_equal(reused["flags"], probabilities > expected_cutoff)
                assert reused["metric"].item() == "mixture"
                assert reused["flag_direction"].item() == "high"

    print(f"workflow={args.workflow} respondents={args.respondents} items={args.items}")
    for name, measured in measurements.items():
        print(f"{name}: median={measured.median_seconds:.4f}s peak={measured.peak_mib:.1f} MiB")
    speedup = measurements["full"].median_seconds / measurements["reused"].median_seconds
    print(f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
