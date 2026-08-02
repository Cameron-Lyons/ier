"""Command-line interface for IER screening."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np

from ier import (
    IndexOptions,
    __version__,
    composite,
    index_catalog,
    response_time,
    response_time_consistency,
    response_time_mixture,
    screen,
)
from ier._flagging import resolve_threshold, threshold_flags

if TYPE_CHECKING:
    from ier.types import IndexCatalog, ScreenResult


def _row_starts_with_non_numeric_value(row: list[str]) -> bool:
    """Return whether the first non-empty cell cannot be parsed as a number."""
    for cell in row:
        if not cell.strip():
            continue
        try:
            float(cell)
        except ValueError:
            return True
        return False
    return False


def _parse_numeric_cell(cell: str) -> float:
    """Parse a matrix cell, treating blank delimited fields as missing values."""
    return float(cell) if cell.strip() else np.nan


def _read_rows_from_stream(handle: TextIO, delimiter: str | None) -> list[list[str]]:
    """Read non-empty rows from a forward-only text stream."""
    sample_lines: list[str] = []
    sample_size = 0
    while sample_size < 4096:
        line = handle.readline()
        if not line:
            break
        sample_lines.append(line)
        sample_size += len(line)

    lines = chain(sample_lines, handle)
    if delimiter is None:
        sample = "".join(sample_lines)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            rows = []
            for line in lines:
                row = line.split()
                if row:
                    rows.append(row)
            return rows

    reader = csv.reader(lines, delimiter=delimiter)
    return [row for row in reader if row and any(cell.strip() for cell in row)]


def _input_label(path: Path) -> str:
    """Return a readable source label for errors."""
    return "standard input" if path == Path("-") else str(path)


def _read_rows(path: Path, delimiter: str | None) -> list[list[str]]:
    """Read plain, gzip-compressed, or standard-input delimited rows."""
    if delimiter is not None and (len(delimiter) != 1 or delimiter in "\r\n"):
        raise ValueError("delimiter must be exactly one non-newline character")

    if path == Path("-"):
        rows = _read_rows_from_stream(sys.stdin, delimiter)
    elif path.suffix.casefold() == ".gz":
        with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
            rows = _read_rows_from_stream(handle, delimiter)
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = _read_rows_from_stream(handle, delimiter)

    if not rows:
        raise ValueError(f"no data rows found in {_input_label(path)}")
    return rows


def _load_input(
    path: Path,
    delimiter: str | None,
    id_column: str | None = None,
    item_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Load selected numeric items and optionally preserve a named identifier."""
    rows = _read_rows(path, delimiter)
    source = _input_label(path)

    selected_names: list[str] | None = None
    if item_columns is not None:
        selected_names = [name.strip() for name in item_columns]
        if not selected_names or any(not name for name in selected_names):
            raise ValueError("item columns must include at least one nonblank name")
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("item columns cannot contain duplicate names")

    identifiers: list[str] | None = None
    id_index: int | None = None
    item_indices: list[int] | None = None
    header: list[str] | None = None
    if id_column is not None or selected_names is not None:
        header = [cell.strip() for cell in rows[0]]
        start = 1

    if id_column is not None:
        assert header is not None
        matches = [index for index, name in enumerate(header) if name == id_column]
        if not matches:
            raise ValueError(f"ID column '{id_column}' was not found in the header")
        if len(matches) > 1:
            raise ValueError(f"ID column '{id_column}' appears more than once in the header")
        id_index = matches[0]

    if selected_names is not None:
        assert header is not None
        item_indices = []
        for name in selected_names:
            matches = [index for index, header_name in enumerate(header) if header_name == name]
            if not matches:
                raise ValueError(f"item column '{name}' was not found in the header")
            if len(matches) > 1:
                raise ValueError(f"item column '{name}' appears more than once in the header")
            if matches[0] == id_index:
                raise ValueError(
                    f"ID column '{id_column}' cannot also be selected as an item column"
                )
            item_indices.append(matches[0])

    if header is None:
        start = int(_row_starts_with_non_numeric_value(rows[0]))

    data_rows = rows[start:]
    if not data_rows:
        raise ValueError(f"no numeric data rows found in {source}")

    widths = {len(row) for row in data_rows}
    if header is not None:
        widths.add(len(header))
    if len(widths) != 1:
        raise ValueError(
            f"jagged delimited input in {source}: rows have unequal lengths {sorted(widths)}; "
            "expected a rectangular respondent×item matrix"
        )

    if id_index is not None:
        identifiers = [row[id_index].strip() for row in data_rows]
        if any(not identifier for identifier in identifiers):
            raise ValueError(f"ID column '{id_column}' contains blank values")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"ID column '{id_column}' contains duplicate values")

    if item_indices is not None:
        data_rows = [[row[index] for index in item_indices] for row in data_rows]
    elif id_index is not None:
        data_rows = [row[:id_index] + row[id_index + 1 :] for row in data_rows]
        if not data_rows[0]:
            raise ValueError("input must contain at least one item column besides the ID column")

    try:
        matrix = np.array(
            [[_parse_numeric_cell(cell) for cell in row] for row in data_rows],
            dtype=float,
        )
    except ValueError as err:
        raise ValueError(f"failed to parse numeric matrix from {source}: {err}") from err

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"expected a 2D respondent×item matrix in {source}")
    return matrix, identifiers


def _load_matrix(path: Path, delimiter: str | None) -> np.ndarray:
    """Load a respondent × item matrix from CSV/TSV/whitespace text."""
    matrix, _ = _load_input(path, delimiter)
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


def _parse_name_list(raw: list[str] | None) -> list[str] | None:
    """Parse repeated comma-separated column-name arguments."""
    if raw is None:
        return None
    names = [name.strip() for entry in raw for name in entry.split(",") if name.strip()]
    if not names:
        raise ValueError("--item-columns must include at least one column name")
    return names


def _parse_pair_list(raw: str | None) -> list[tuple[int, int]] | None:
    if raw is None:
        return None
    pairs: list[tuple[int, int]] = []
    for raw_chunk in raw.split(";"):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        left, sep, right = chunk.partition(",")
        if not sep:
            raise ValueError(f"invalid pair '{chunk}'; expected 'i,j' pairs separated by ';'")
        pairs.append((int(left.strip()), int(right.strip())))
    return pairs or None


def _parse_thresholds(raw: list[str] | None) -> dict[str, float] | None:
    if raw is None:
        return None

    thresholds: dict[str, float] = {}
    for entry in raw:
        name, separator, value = entry.partition("=")
        name = name.strip()
        if not separator or not name or not value.strip():
            raise ValueError(f"invalid threshold '{entry}'; expected INDEX=VALUE")
        if name in thresholds:
            raise ValueError(f"duplicate threshold for index: {name}")
        try:
            cutoff = float(value)
        except ValueError as err:
            raise ValueError(f"invalid threshold value for {name}: {value.strip()}") from err
        if not np.isfinite(cutoff):
            raise ValueError(f"threshold for {name} must be a finite number")
        thresholds[name] = cutoff
    return thresholds


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


def _add_matrix_input_options(parser: argparse.ArgumentParser) -> None:
    """Add delimited-matrix selection options shared by scoring commands."""
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Input delimiter (auto-detect comma, tab, semicolon, or whitespace if omitted)",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        metavar="NAME",
        help="Named header column to preserve as respondent identifiers",
    )
    parser.add_argument(
        "--item-columns",
        action="append",
        default=None,
        metavar="NAME[,NAME...]",
        help="Named header columns to score, in order; comma-separate or repeat",
    )


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output controls shared by scoring commands."""
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
        help="Write output to a path, optionally .gz; use '-' for stdout",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="For text format: show the top N respondents (default: 10)",
    )


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    _add_matrix_input_options(parser)
    parser.add_argument("--scale-min", type=float, default=None)
    parser.add_argument("--scale-max", type=float, default=None)
    parser.add_argument(
        "--na-rm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop incomplete rows / pairwise NaNs where supported (default: true)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested index cannot be computed",
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
    _add_output_options(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ier",
        description="Detect insufficient effort / careless responding in survey matrices.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    screen_parser = sub.add_parser(
        "screen", help="Run multi-index screening on a delimited matrix."
    )
    screen_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace item scores; use '-' for standard input",
    )
    screen_parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help="Index names to compute (default: package screen defaults)",
    )
    screen_parser.add_argument("--percentile", type=float, default=95.0)
    screen_parser.add_argument(
        "--threshold",
        action="append",
        default=None,
        metavar="INDEX=VALUE",
        help="Fixed per-index cutoff; repeat for multiple indices",
    )
    screen_parser.add_argument(
        "--min-flags",
        type=int,
        default=2,
        help="Minimum per-index flags for a consensus flag (default: 2)",
    )
    _add_shared_options(screen_parser)

    composite_parser = sub.add_parser(
        "composite", help="Compute a composite IER score for each respondent."
    )
    composite_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace item scores; use '-' for standard input",
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

    response_time_parser = sub.add_parser(
        "response-time",
        help="Score and flag a delimited response-time matrix.",
    )
    response_time_parser.add_argument(
        "data",
        type=Path,
        help="CSV/TSV/whitespace timing values; use '-' for standard input",
    )
    response_time_parser.add_argument(
        "--metric",
        choices=["mean", "median", "sd", "min", "consistency", "mixture"],
        default="median",
        help="Timing score to compute (default: median)",
    )
    response_time_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed inclusive flag cutoff in score units",
    )
    response_time_parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Percentile cutoff (default: 5 for low scores, 95 for mixture)",
    )
    response_time_parser.add_argument(
        "--components",
        type=int,
        default=2,
        help="Gaussian components for the mixture metric (default: 2)",
    )
    response_time_parser.add_argument(
        "--log-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log-transform median times for the mixture metric (default: true)",
    )
    response_time_parser.add_argument("--random-seed", type=int, default=None)
    _add_matrix_input_options(response_time_parser)
    _add_output_options(response_time_parser)

    indices_parser = sub.add_parser(
        "indices", help="List registered indices and orchestration metadata."
    )
    indices_parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    indices_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write output to a path, optionally .gz; use '-' for stdout",
    )

    return parser


def _write_output(text: str, path: Path | None) -> None:
    if path is None or path == Path("-"):
        print(text)
        return
    if path.suffix.casefold() == ".gz":
        with gzip.open(path, mode="wt", newline="", encoding="utf-8") as handle:
            handle.write(text)
        return
    path.write_text(text, encoding="utf-8")


def _json_numbers(values: np.ndarray) -> list[float | None]:
    """Convert numeric arrays to strict-JSON values, representing non-finite values as null."""
    return [float(value) if np.isfinite(value) else None for value in values]


def _csv_number(value: float) -> float | None:
    """Return a CSV-safe number, using an empty cell for non-finite values."""
    return float(value) if np.isfinite(value) else None


def _respondent_labels(
    n_respondents: int,
    respondent_ids: list[str] | None,
) -> list[int | str]:
    """Return validated output labels for respondent-aligned results."""
    if respondent_ids is None:
        return list(range(n_respondents))
    if len(respondent_ids) != n_respondents:
        raise ValueError("respondent ID count must match result length")
    labels: list[int | str] = list(respondent_ids)
    return labels


def _emit_index_catalog_text(catalog: IndexCatalog) -> str:
    lines = [
        "index\tdirection\tflag_mode\tscreen_default\tcomposite\tcomposite_default"
        "\trequired_options"
    ]
    for name, metadata in catalog.items():
        required = ",".join(metadata["required_options"]) or "-"
        lines.append(
            "\t".join(
                (
                    name,
                    metadata["flag_direction"],
                    metadata["flag_mode"],
                    "yes" if metadata["default_screen"] else "no",
                    "yes" if metadata["composite_enabled"] else "no",
                    "yes" if metadata["default_composite"] else "no",
                    required,
                )
            )
        )
    return "\n".join(lines)


def _emit_index_catalog_json(catalog: IndexCatalog) -> str:
    return json.dumps(
        {"n_indices": len(catalog), "indices": catalog},
        indent=2,
    )


def _emit_index_catalog_csv(catalog: IndexCatalog) -> str:
    fieldnames = [
        "index",
        "flag_direction",
        "flag_mode",
        "default_screen",
        "default_composite",
        "composite_enabled",
        "required_options",
    ]
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for name, metadata in catalog.items():
        writer.writerow(
            {
                "index": name,
                "flag_direction": metadata["flag_direction"],
                "flag_mode": metadata["flag_mode"],
                "default_screen": metadata["default_screen"],
                "default_composite": metadata["default_composite"],
                "composite_enabled": metadata["composite_enabled"],
                "required_options": ",".join(metadata["required_options"]),
            }
        )
    return buf.getvalue()


def _emit_screen_text(
    result: ScreenResult,
    top: int,
    respondent_ids: list[str] | None = None,
) -> str:
    lines = [
        f"respondents: {result['n_respondents']}",
        f"indices: {', '.join(result['indices_used'])}",
        (
            f"consensus flagged: {int(np.sum(result['consensus_flags']))} "
            f"(min_flags={result['min_flags']})"
        ),
        "flag thresholds: "
        + ", ".join(
            f"{name}={'presence' if cutoff is None else f'{cutoff:g}'}"
            for name, cutoff in result["thresholds"].items()
        ),
    ]
    if result["errors"]:
        lines.append("errors:")
        for name, message in sorted(result["errors"].items()):
            lines.append(f"  {name}: {message}")
    counts = result["flag_counts"]
    labels = _respondent_labels(result["n_respondents"], respondent_ids)
    order = np.argsort(counts)[::-1][: max(top, 0)]
    label_name = "identifier" if respondent_ids is not None else "index"
    lines.append(f"top flagged respondents ({label_name}, flag_count):")
    for idx in order:
        lines.append(f"  {labels[int(idx)]}\t{int(counts[idx])}")
    return "\n".join(lines)


def _emit_screen_json(
    result: ScreenResult,
    respondent_ids: list[str] | None = None,
) -> str:
    summary = {
        name: {
            "mean": stats["mean"] if np.isfinite(stats["mean"]) else None,
            "std": stats["std"] if np.isfinite(stats["std"]) else None,
            "min": stats["min"] if np.isfinite(stats["min"]) else None,
            "max": stats["max"] if np.isfinite(stats["max"]) else None,
            "n_flagged": stats["n_flagged"],
        }
        for name, stats in result["summary"].items()
    }
    payload = {
        "n_respondents": result["n_respondents"],
        "n_indices": result["n_indices"],
        "indices_used": result["indices_used"],
        "errors": result["errors"],
        "thresholds": result["thresholds"],
        "flag_counts": np.asarray(result["flag_counts"]).tolist(),
        "consensus_flags": np.asarray(result["consensus_flags"]).astype(bool).tolist(),
        "min_flags": result["min_flags"],
        "scores": {name: _json_numbers(np.asarray(arr)) for name, arr in result["scores"].items()},
        "flags": {
            name: np.asarray(arr).astype(bool).tolist() for name, arr in result["flags"].items()
        },
        "summary": summary,
    }
    if respondent_ids is not None:
        payload["respondent_ids"] = _respondent_labels(result["n_respondents"], respondent_ids)
    return json.dumps(payload, indent=2, allow_nan=False)


def _emit_screen_csv(
    result: ScreenResult,
    respondent_ids: list[str] | None = None,
) -> str:
    n = result["n_respondents"]
    scores = result["scores"]
    flags = result["flags"]
    fieldnames = ["respondent", "flag_count", "consensus_flag"]
    for name in result["indices_used"]:
        fieldnames.extend([f"{name}_score", f"{name}_flag"])

    rows: list[dict[str, object]] = []
    counts = np.asarray(result["flag_counts"])
    consensus = np.asarray(result["consensus_flags"])
    labels = _respondent_labels(n, respondent_ids)
    for i in range(n):
        row: dict[str, object] = {
            "respondent": labels[i],
            "flag_count": int(counts[i]),
            "consensus_flag": int(bool(consensus[i])),
        }
        for name in result["indices_used"]:
            row[f"{name}_score"] = _csv_number(scores[name][i])
            row[f"{name}_flag"] = int(bool(flags[name][i]))
        rows.append(row)

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _emit_composite_text(
    scores: np.ndarray,
    method: str,
    top: int,
    respondent_ids: list[str] | None = None,
) -> str:
    order = np.argsort(scores)[::-1][: max(top, 0)]
    labels = _respondent_labels(len(scores), respondent_ids)
    label_name = "identifier" if respondent_ids is not None else "index"
    lines = [
        f"respondents: {len(scores)}",
        f"method: {method}",
        f"top composite scores ({label_name}, score):",
    ]
    for idx in order:
        lines.append(f"  {labels[int(idx)]}\t{float(scores[idx]):.6f}")
    return "\n".join(lines)


def _emit_composite_json(
    scores: np.ndarray,
    method: str,
    respondent_ids: list[str] | None = None,
) -> str:
    payload: dict[str, object] = {
        "method": method,
        "scores": _json_numbers(scores),
        "n_respondents": len(scores),
    }
    if respondent_ids is not None:
        payload["respondent_ids"] = _respondent_labels(len(scores), respondent_ids)
    return json.dumps(payload, indent=2, allow_nan=False)


def _emit_composite_csv(
    scores: np.ndarray,
    respondent_ids: list[str] | None = None,
) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["respondent", "composite_score"])
    labels = _respondent_labels(len(scores), respondent_ids)
    for label, score in zip(labels, scores, strict=True):
        writer.writerow([label, _csv_number(score)])
    return buf.getvalue()


def _score_response_times(
    matrix: np.ndarray,
    metric: str,
    components: int,
    log_transform: bool,
    random_seed: int | None,
) -> tuple[np.ndarray, Literal["high", "low"]]:
    """Compute one timing metric and its suspicious-score direction."""
    if metric == "consistency":
        return response_time_consistency(matrix), "low"
    if metric == "mixture":
        return (
            response_time_mixture(
                matrix,
                n_components=components,
                log_transform=log_transform,
                random_seed=random_seed,
            ),
            "high",
        )
    return response_time(matrix, metric=metric), "low"


def _emit_response_time_text(
    scores: np.ndarray,
    flags: np.ndarray,
    metric: str,
    direction: Literal["high", "low"],
    cutoff: float,
    top: int,
    respondent_ids: list[str] | None = None,
) -> str:
    """Render timing results as a compact human-readable summary."""
    labels = _respondent_labels(len(scores), respondent_ids)
    valid_indices = np.flatnonzero(np.isfinite(scores))
    order = valid_indices[np.argsort(scores[valid_indices])]
    if direction == "high":
        order = order[::-1]
    order = order[: max(top, 0)]
    label_name = "identifier" if respondent_ids is not None else "index"
    lines = [
        f"respondents: {len(scores)}",
        f"metric: {metric}",
        f"flag direction: {direction}",
        f"threshold: {cutoff:g}",
        f"flagged: {int(np.sum(flags))}",
        f"top suspicious respondents ({label_name}, score):",
    ]
    for index in order:
        lines.append(f"  {labels[int(index)]}\t{float(scores[index]):.6f}")
    return "\n".join(lines)


def _emit_response_time_json(
    scores: np.ndarray,
    flags: np.ndarray,
    metric: str,
    direction: Literal["high", "low"],
    cutoff: float,
    respondent_ids: list[str] | None = None,
) -> str:
    """Render timing results as strict JSON."""
    payload: dict[str, object] = {
        "n_respondents": len(scores),
        "metric": metric,
        "flag_direction": direction,
        "threshold": cutoff,
        "scores": _json_numbers(scores),
        "flags": flags.astype(bool).tolist(),
    }
    if respondent_ids is not None:
        payload["respondent_ids"] = _respondent_labels(len(scores), respondent_ids)
    return json.dumps(payload, indent=2, allow_nan=False)


def _emit_response_time_csv(
    scores: np.ndarray,
    flags: np.ndarray,
    respondent_ids: list[str] | None = None,
) -> str:
    """Render respondent-aligned timing scores and flags as CSV."""
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["respondent", "response_time_score", "response_time_flag"])
    labels = _respondent_labels(len(scores), respondent_ids)
    for label, score, flag in zip(labels, scores, flags, strict=True):
        writer.writerow([label, _csv_number(score), int(bool(flag))])
    return buf.getvalue()


def _run_command(args: argparse.Namespace) -> int:
    """Execute one parsed CLI command, allowing user-facing failures to bubble to main()."""
    if args.command == "indices":
        catalog = index_catalog()
        if args.format == "json":
            text = _emit_index_catalog_json(catalog)
        elif args.format == "csv":
            text = _emit_index_catalog_csv(catalog)
        else:
            text = _emit_index_catalog_text(catalog)
        _write_output(text, args.output)
        return 0

    matrix, respondent_ids = _load_input(
        args.data,
        args.delimiter,
        args.id_column,
        _parse_name_list(args.item_columns),
    )
    if args.command == "response-time":
        scores, direction = _score_response_times(
            matrix,
            args.metric,
            args.components,
            args.log_transform,
            args.random_seed,
        )
        percentile = args.percentile
        if percentile is None:
            percentile = 95.0 if direction == "high" else 5.0
        cutoff = resolve_threshold(scores, args.threshold, percentile)
        flags = threshold_flags(
            scores,
            threshold=cutoff,
            percentile=percentile,
            direction=direction,
            inclusive=args.threshold is not None,
        )

        if args.format == "json":
            text = _emit_response_time_json(
                scores,
                flags,
                args.metric,
                direction,
                cutoff,
                respondent_ids,
            )
        elif args.format == "csv":
            text = _emit_response_time_csv(scores, flags, respondent_ids)
        else:
            text = _emit_response_time_text(
                scores,
                flags,
                args.metric,
                direction,
                cutoff,
                args.top,
                respondent_ids,
            )
        _write_output(text, args.output)
        return 0

    options = _options_from_args(args)
    if args.command == "screen":
        result = screen(
            matrix,
            indices=args.indices,
            options=options,
            percentile=args.percentile,
            min_flags=args.min_flags,
            thresholds=_parse_thresholds(args.threshold),
            strict=args.strict,
        )
        if args.format == "json":
            text = _emit_screen_json(result, respondent_ids)
        elif args.format == "csv":
            text = _emit_screen_csv(result, respondent_ids)
        else:
            text = _emit_screen_text(result, args.top, respondent_ids)
        _write_output(text, args.output)
        return 0

    scores_result = composite(
        matrix,
        indices=args.indices,
        method=args.method,
        options=options,
        strict=args.strict,
    )
    if not isinstance(scores_result, np.ndarray):
        print("error: unexpected composite return type", file=sys.stderr)
        return 1
    scores = scores_result

    if args.format == "json":
        text = _emit_composite_json(scores, args.method, respondent_ids)
    elif args.format == "csv":
        text = _emit_composite_csv(scores, respondent_ids)
    else:
        text = _emit_composite_text(scores, args.method, args.top, respondent_ids)
    _write_output(text, args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ier`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _run_command(args)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
