"""Command-line interface for IER screening."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ier import IndexOptions, __version__, composite, screen

if TYPE_CHECKING:
    from ier.types import ScreenResult


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

    widths = {len(row) for row in data_rows}
    if len(widths) != 1:
        raise ValueError(
            f"jagged CSV in {path}: rows have unequal lengths {sorted(widths)}; "
            "expected a rectangular respondent×item matrix"
        )

    try:
        matrix = np.array([[float(cell) for cell in row] for row in data_rows], dtype=float)
    except ValueError as err:
        raise ValueError(f"failed to parse numeric matrix from {path}: {err}") from err

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"expected a 2D respondent×item matrix in {path}")
    return matrix


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    return [int(part) for part in parts]


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    return [float(part) for part in parts]


def _parse_pair_list(raw: str | None) -> list[tuple[int, int]] | None:
    if raw is None:
        return None
    pairs: list[tuple[int, int]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        left, sep, right = chunk.partition(",")
        if not sep:
            raise ValueError(f"invalid pair '{chunk}'; expected 'i,j' pairs separated by ';'")
        pairs.append((int(left.strip()), int(right.strip())))
    return pairs or None


def _options_from_args(args: argparse.Namespace) -> IndexOptions:
    return IndexOptions(
        na_rm=args.na_rm,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        psychsyn_critval=args.psychsyn_critval,
        psychant_critval=args.psychant_critval,
        evenodd_factors=_parse_int_list(args.evenodd_factors),
        mad_positive_items=_parse_int_list(args.mad_positive_items),
        mad_negative_items=_parse_int_list(args.mad_negative_items),
        mad_scale_max=args.mad_scale_max,
        longstring_max_pattern_length=args.longstring_max_pattern_length,
        midpoint_tolerance=args.midpoint_tolerance,
        guttman_normalize=args.guttman_normalize,
        onset_window_size=args.onset_window_size,
        onset_min_items=args.onset_min_items,
        reliability_n_splits=args.reliability_n_splits,
        reliability_random_seed=args.reliability_random_seed,
        semantic_item_pairs=_parse_pair_list(args.semantic_item_pairs),
        infrequency_item_indices=_parse_int_list(args.infrequency_item_indices),
        infrequency_expected_responses=_parse_float_list(args.infrequency_expected_responses),
        infrequency_proportion=args.infrequency_proportion,
    )


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--delimiter",
        default=None,
        help="CSV delimiter (auto-detect if omitted)",
    )
    parser.add_argument("--scale-min", type=float, default=None)
    parser.add_argument("--scale-max", type=float, default=None)
    parser.add_argument(
        "--na-rm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop incomplete rows / pairwise NaNs where supported (default: true)",
    )
    parser.add_argument("--psychsyn-critval", type=float, default=0.6)
    parser.add_argument("--psychant-critval", type=float, default=-0.6)
    parser.add_argument(
        "--evenodd-factors",
        default=None,
        help="Comma-separated factor lengths, e.g. '5,5'",
    )
    parser.add_argument("--mad-positive-items", default=None, help="Comma-separated item indices")
    parser.add_argument("--mad-negative-items", default=None, help="Comma-separated item indices")
    parser.add_argument("--mad-scale-max", type=int, default=None)
    parser.add_argument("--longstring-max-pattern-length", type=int, default=5)
    parser.add_argument("--midpoint-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--guttman-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--onset-window-size", type=int, default=10)
    parser.add_argument("--onset-min-items", type=int, default=20)
    parser.add_argument("--reliability-n-splits", type=int, default=100)
    parser.add_argument("--reliability-random-seed", type=int, default=None)
    parser.add_argument(
        "--semantic-item-pairs",
        default=None,
        help="Pairs as 'i,j;i,j' (0-based item indices)",
    )
    parser.add_argument("--infrequency-item-indices", default=None)
    parser.add_argument("--infrequency-expected-responses", default=None)
    parser.add_argument(
        "--infrequency-proportion",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text summary)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON/CSV output to this path (default: stdout)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="For text format: show the top N respondents (default: 10)",
    )


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
    screen_parser.add_argument("--percentile", type=float, default=95.0)
    _add_shared_options(screen_parser)

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
    composite_parser.add_argument(
        "--method",
        choices=["mean", "sum", "max", "best_subset"],
        default="mean",
    )
    _add_shared_options(composite_parser)

    return parser


def _write_output(text: str, path: Path | None) -> None:
    if path is None:
        print(text)
        return
    path.write_text(text, encoding="utf-8")


def _emit_screen_text(result: ScreenResult, top: int) -> str:
    lines = [
        f"respondents: {result['n_respondents']}",
        f"indices: {', '.join(result['indices_used'])}",
    ]
    if result["errors"]:
        lines.append("errors:")
        for name, message in sorted(result["errors"].items()):
            lines.append(f"  {name}: {message}")
    counts = result["flag_counts"]
    order = np.argsort(counts)[::-1][: max(top, 0)]
    lines.append("top flagged respondents (index, flag_count):")
    for idx in order:
        lines.append(f"  {int(idx)}\t{int(counts[idx])}")
    return "\n".join(lines)


def _emit_screen_json(result: ScreenResult) -> str:
    payload = {
        "n_respondents": result["n_respondents"],
        "n_indices": result["n_indices"],
        "indices_used": result["indices_used"],
        "errors": result["errors"],
        "flag_counts": np.asarray(result["flag_counts"]).tolist(),
        "scores": {name: np.asarray(arr).tolist() for name, arr in result["scores"].items()},
        "flags": {
            name: np.asarray(arr).astype(bool).tolist() for name, arr in result["flags"].items()
        },
        "summary": result["summary"],
    }
    return json.dumps(payload, indent=2)


def _emit_screen_csv(result: ScreenResult) -> str:
    n = result["n_respondents"]
    scores = result["scores"]
    flags = result["flags"]
    fieldnames = ["respondent", "flag_count"]
    for name in result["indices_used"]:
        fieldnames.extend([f"{name}_score", f"{name}_flag"])

    rows: list[dict[str, object]] = []
    counts = np.asarray(result["flag_counts"])
    for i in range(n):
        row: dict[str, object] = {"respondent": i, "flag_count": int(counts[i])}
        for name in result["indices_used"]:
            row[f"{name}_score"] = float(scores[name][i])
            row[f"{name}_flag"] = int(bool(flags[name][i]))
        rows.append(row)

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _emit_composite_text(scores: np.ndarray, method: str, top: int) -> str:
    order = np.argsort(scores)[::-1][: max(top, 0)]
    lines = [
        f"respondents: {len(scores)}",
        f"method: {method}",
        "top composite scores (index, score):",
    ]
    for idx in order:
        lines.append(f"  {int(idx)}\t{float(scores[idx]):.6f}")
    return "\n".join(lines)


def _emit_composite_json(scores: np.ndarray, method: str) -> str:
    return json.dumps(
        {"method": method, "scores": scores.tolist(), "n_respondents": len(scores)},
        indent=2,
    )


def _emit_composite_csv(scores: np.ndarray) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["respondent", "composite_score"])
    for i, score in enumerate(scores):
        writer.writerow([i, float(score)])
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ier`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        matrix = _load_matrix(args.data, args.delimiter)
        options = _options_from_args(args)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    if args.command == "screen":
        result = screen(
            matrix,
            indices=args.indices,
            options=options,
            percentile=args.percentile,
        )
        if args.format == "json":
            text = _emit_screen_json(result)
        elif args.format == "csv":
            text = _emit_screen_csv(result)
        else:
            text = _emit_screen_text(result, args.top)
        _write_output(text, args.output)
        return 0

    scores_result = composite(matrix, indices=args.indices, method=args.method, options=options)
    if not isinstance(scores_result, np.ndarray):
        print("error: unexpected composite return type", file=sys.stderr)
        return 1
    scores = scores_result

    if args.format == "json":
        text = _emit_composite_json(scores, args.method)
    elif args.format == "csv":
        text = _emit_composite_csv(scores)
    else:
        text = _emit_composite_text(scores, args.method, args.top)
    _write_output(text, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
