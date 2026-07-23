"""Command-line interface for IER screening."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from ier import IndexOptions, __version__, composite, screen


def _load_matrix(path: Path, delimiter: str | None) -> np.ndarray:
    """Load a respondent × item matrix from CSV/TSV/whitespace text."""
    if delimiter is None:
        with path.open(newline="", encoding="utf-8") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                delim = dialect.delimiter
            except csv.Error:
                delim = ","
            reader = csv.reader(handle, delimiter=delim)
            rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"no data rows found in {path}")

    # Drop a header row if the first cell is non-numeric.
    start = 0
    try:
        float(rows[0][0])
    except ValueError:
        start = 1

    data_rows = rows[start:]
    if not data_rows:
        raise ValueError(f"no numeric data rows found in {path}")

    try:
        matrix = np.array([[float(cell) for cell in row] for row in data_rows], dtype=float)
    except ValueError as err:
        raise ValueError(f"failed to parse numeric matrix from {path}: {err}") from err

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"expected a 2D respondent×item matrix in {path}")
    return matrix


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ier",
        description="Detect insufficient effort / careless responding in survey matrices.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    screen_parser = sub.add_parser("screen", help="Run multi-index screening on a CSV matrix.")
    screen_parser.add_argument(
        "data",
        type=Path,
        help="Path to CSV/TSV of respondent × item scores",
    )
    screen_parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index names to compute (default: package screen defaults)",
    )
    screen_parser.add_argument(
        "--delimiter",
        default=None,
        help="CSV delimiter (auto-detect if omitted)",
    )
    screen_parser.add_argument("--scale-min", type=float, default=None)
    screen_parser.add_argument("--scale-max", type=float, default=None)
    screen_parser.add_argument("--percentile", type=float, default=95.0)
    screen_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show the N respondents with the most flags (default: 10)",
    )

    composite_parser = sub.add_parser(
        "composite", help="Compute a composite IER score for each respondent."
    )
    composite_parser.add_argument(
        "data", type=Path, help="Path to CSV/TSV of respondent × item scores"
    )
    composite_parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index names to include (default: package composite defaults)",
    )
    composite_parser.add_argument("--delimiter", default=None, help="CSV delimiter")
    composite_parser.add_argument(
        "--method",
        choices=["mean", "sum", "max", "best_subset"],
        default="mean",
    )
    composite_parser.add_argument("--scale-min", type=float, default=None)
    composite_parser.add_argument("--scale-max", type=float, default=None)
    composite_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show the N highest composite scores (default: 10)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ier`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        matrix = _load_matrix(args.data, args.delimiter)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    options = IndexOptions(scale_min=args.scale_min, scale_max=args.scale_max)

    if args.command == "screen":
        result = screen(
            matrix,
            indices=args.indices,
            options=options,
            percentile=args.percentile,
        )
        print(f"respondents: {result['n_respondents']}")
        print(f"indices: {', '.join(result['indices_used'])}")
        if result["errors"]:
            print("errors:")
            for name, message in sorted(result["errors"].items()):
                print(f"  {name}: {message}")
        counts = result["flag_counts"]
        order = np.argsort(counts)[::-1][: max(args.top, 0)]
        print("top flagged respondents (index, flag_count):")
        for idx in order:
            print(f"  {int(idx)}\t{int(counts[idx])}")
        return 0

    scores = composite(matrix, indices=args.indices, method=args.method, options=options)
    assert isinstance(scores, np.ndarray)
    order = np.argsort(scores)[::-1][: max(args.top, 0)]
    print(f"respondents: {len(scores)}")
    print(f"method: {args.method}")
    print("top composite scores (index, score):")
    for idx in order:
        print(f"  {int(idx)}\t{float(scores[idx]):.6f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
