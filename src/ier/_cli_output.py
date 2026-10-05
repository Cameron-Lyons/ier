"""Text, JSON, and CSV serializers for command-line results."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np

from ier._atomic_output import atomic_output_path
from ier._cli_streams import _open_text_path

if TYPE_CHECKING:
    from ier._cli_composite import CompositeReport, ResponseTimeReport
    from ier.types import IndexCatalog, ScreenResult


_JSON_ARRAY_CHUNK_SIZE = 4096
_CSV_CHUNK_SIZE = 1024
_CSV_BITS = np.array(["0", "1"], dtype=object)
_TEXT_RANK_BATCH_SIZE = 16_384


@dataclass(frozen=True)
class _JsonArray:
    """A large one-dimensional value sequence serialized in bounded chunks."""

    values: Sequence[object] | np.ndarray
    kind: Literal["number", "integer", "boolean", "string"]


def _write_output(text: str, path: Path | None) -> None:
    _write_stream_output(path, lambda handle: handle.write(text))


@contextmanager
def _output_stream(path: Path | None) -> Iterator[TextIO]:
    """Stage result files for atomic replacement, or stream to standard output."""
    if path is None or path == Path("-"):
        yield sys.stdout
        return
    with atomic_output_path(path) as staged_path, _open_text_path(staged_path, "w") as handle:
        yield handle


def _json_chunk_values(array: _JsonArray, start: int, stop: int) -> Sequence[object]:
    """Materialize one bounded JSON-ready chunk from a large value sequence."""
    if array.kind == "number":
        return _csv_float_column(array.values, start, stop)
    values = array.values[start:stop]
    if array.kind == "integer":
        integers: list[int] = np.asarray(values, dtype=np.int_).tolist()
        return integers
    if array.kind == "boolean":
        booleans: list[bool] = np.asarray(values, dtype=np.bool_).tolist()
        return booleans
    return [str(value) for value in values]


def _write_json_array(handle: TextIO, array: _JsonArray) -> None:
    """Write a large JSON array without retaining its complete Python representation."""
    handle.write("[")
    for start in range(0, len(array.values), _JSON_ARRAY_CHUNK_SIZE):
        if start:
            handle.write(",")
        chunk = _json_chunk_values(
            array,
            start,
            min(start + _JSON_ARRAY_CHUNK_SIZE, len(array.values)),
        )
        encoded = json.dumps(chunk, allow_nan=False, separators=(",", ":"))
        handle.write(encoded[1:-1])
    handle.write("]")


def _write_json_value(handle: TextIO, value: object, indent: int = 0) -> None:
    """Write strict JSON recursively, streaming marked large arrays."""
    if isinstance(value, _JsonArray):
        _write_json_array(handle, value)
        return
    if isinstance(value, Mapping):
        handle.write("{")
        if value:
            handle.write("\n")
            last_index = len(value) - 1
            for index, (key, item) in enumerate(value.items()):
                handle.write(" " * (indent + 2))
                json.dump(key, handle, allow_nan=False)
                handle.write(": ")
                _write_json_value(handle, item, indent + 2)
                handle.write("," if index < last_index else "")
                handle.write("\n")
            handle.write(" " * indent)
        handle.write("}")
        return
    json.dump(value, handle, allow_nan=False)


def _write_stream_output(
    path: Path | None,
    writer: Callable[[TextIO], object],
) -> None:
    """Stream output to a plain, compressed, or standard-output destination."""
    with _output_stream(path) as handle:
        writer(handle)
        if path is None or path == Path("-"):
            handle.write("\n")


def _csv_float_column(
    values: Sequence[object] | np.ndarray, start: int, stop: int
) -> list[float | None]:
    """Return one chunk of numbers, using None (an empty CSV cell) for non-finite values."""
    numeric = np.asarray(values[start:stop], dtype=np.float64)
    column: list[float | None] = numeric.tolist()
    for index in np.flatnonzero(~np.isfinite(numeric)):
        column[int(index)] = None
    return column


def _csv_bit_column(values: Sequence[object] | np.ndarray, start: int, stop: int) -> list[str]:
    """Return one chunk of truth values as preformatted 0/1 cells."""
    bits = np.asarray(values[start:stop], dtype=np.bool_).view(np.uint8)
    # NumPy stores True as any nonzero byte (bool views and foreign NPZ members
    # need not use 1), so clip the raw bytes onto the two-cell lookup table.
    cells: list[str] = _CSV_BITS.take(bits, mode="clip").tolist()
    return cells


def _csv_int_column(values: Sequence[object] | np.ndarray, start: int, stop: int) -> list[int]:
    """Return one chunk of counts as Python integers."""
    integers: list[int] = np.asarray(values[start:stop], dtype=np.int64).tolist()
    return integers


def _respondent_label_values(
    n_respondents: int,
    respondent_ids: list[str] | None,
) -> Sequence[int | str]:
    """Return validated, potentially lazy labels for respondent-aligned results."""
    if respondent_ids is None:
        return range(n_respondents)
    if len(respondent_ids) != n_respondents:
        raise ValueError("respondent ID count must match result length")
    return respondent_ids


def _ranked_rows(
    scores: np.ndarray,
    top: int,
    direction: Literal["high", "low"] = "high",
) -> np.ndarray:
    """Select finite scores by rank, resolving ties in original respondent order.

    Small previews retain only the requested rows plus one bounded input batch.
    Large previews sort directly, since their output already scales with input.
    """
    limit = min(max(top, 0), len(scores))
    if not limit:
        return np.empty(0, dtype=np.intp)
    if limit * 8 >= len(scores):
        candidates = np.flatnonzero(np.isfinite(scores))
    else:
        candidates = np.empty(0, dtype=np.intp)
        cutoff: np.ndarray | None = None
        for start in range(0, len(scores), _TEXT_RANK_BATCH_SIZE):
            block = scores[start : start + _TEXT_RANK_BATCH_SIZE]
            available = np.isfinite(block)
            if cutoff is not None:
                # Later rows tied with a full preview cannot outrank its earlier
                # respondents. Retain only rows that improve the current cutoff.
                available &= block > cutoff if direction == "high" else block < cutoff
            rows = np.flatnonzero(available) + start
            if not len(rows):
                continue
            candidates = np.concatenate((candidates, rows))
            if len(candidates) <= limit:
                continue
            values = scores[candidates]
            position = len(values) - limit if direction == "high" else limit - 1
            # Keep the dtype in a small array for exact large-integer comparisons
            # on both NumPy 1.x and 2.x, without retaining the partition buffer.
            cutoff = np.partition(values, position)[position : position + 1].copy()
            keep = values > cutoff if direction == "high" else values < cutoff
            ties = np.flatnonzero(values == cutoff)
            # Candidates stay in input order, including ties at the cutoff.
            keep[ties[: limit - np.count_nonzero(keep)]] = True
            candidates = candidates[keep]

    values = scores[candidates]
    order = np.argsort(values)
    if direction == "high":
        order = order[::-1]
    # Only ties within the preview or at its boundary affect returned row order.
    preview = values[order[: limit + 1]]
    if np.any(preview[1:] == preview[:-1]):
        if direction == "high":
            # Reversing a stable sort's input and result preserves input-order
            # ties in descending ranks without negating unsigned or extreme scores.
            order = np.argsort(values[::-1], kind="stable")[::-1][:limit]
            result: np.ndarray = candidates[::-1][order]
            return result
        order = np.argsort(values, kind="stable")
    result = candidates[order[:limit]]
    return result


def _alternative_options_text(groups: Sequence[Sequence[str]]) -> str:
    """Join option groups with ';' and the interchangeable options in each with '|'."""
    return ";".join("|".join(group) for group in groups)


def _emit_index_catalog_text(catalog: IndexCatalog) -> str:
    lines = [
        "index\tdirection\tflag_mode\tscreen_default\tcomposite\tcomposite_default"
        "\trequired_options\talternative_options\tkeyed_responses"
    ]
    for name, metadata in catalog.items():
        required = ",".join(metadata["required_options"]) or "-"
        alternatives = _alternative_options_text(metadata["alternative_options"]) or "-"
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
                    alternatives,
                    "yes" if metadata["uses_keyed_responses"] else "no",
                )
            )
        )
    return "\n".join(lines)


def _emit_index_catalog_json(catalog: IndexCatalog) -> str:
    return json.dumps(
        {"n_indices": len(catalog), "indices": catalog},
        indent=2,
    )


def _write_index_catalog_csv(handle: TextIO, catalog: IndexCatalog) -> None:
    """Write the index catalog directly to a CSV stream."""
    fieldnames = [
        "index",
        "flag_direction",
        "flag_mode",
        "default_screen",
        "default_composite",
        "composite_enabled",
        "required_options",
        "alternative_options",
        "uses_keyed_responses",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
                "alternative_options": _alternative_options_text(metadata["alternative_options"]),
                "uses_keyed_responses": metadata["uses_keyed_responses"],
            }
        )


def _screen_threshold_text(result: ScreenResult, name: str) -> str:
    """Format one screening cutoff with its decision provenance."""
    source = result["threshold_sources"][name]
    if source == "presence":
        return "presence"

    cutoff = result["thresholds"][name]
    assert cutoff is not None
    if source == "fixed":
        return f"{cutoff:g} (fixed)"

    percentile = result["percentiles"][name]
    assert percentile is not None
    return f"{cutoff:g} (tail percentile={percentile:g})"


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
            f"{name}={_screen_threshold_text(result, name)}" for name in result["thresholds"]
        ),
        "index coverage:",
    ]
    for name in result["indices_used"]:
        stats = result["summary"][name]
        rate = stats["flag_rate"]
        rate_text = "n/a" if not np.isfinite(rate) else f"{rate:.1%}"
        lines.append(
            f"  {name}: valid={stats['n_valid']}/{result['n_respondents']}, "
            f"unavailable={stats['n_unavailable']}, "
            f"flagged={stats['n_flagged']}/{stats['n_valid']} ({rate_text})"
        )
    if result["min_valid_indices"] is not None:
        lines.append(
            f"consensus eligible: {int(np.sum(result['consensus_eligible']))} "
            f"(min_valid_indices={result['min_valid_indices']})"
        )
    if result["errors"]:
        lines.append("errors:")
        for name, message in sorted(result["errors"].items()):
            lines.append(f"  {name}: {message}")
    counts = result["flag_counts"]
    valid_counts = result["valid_index_counts"]
    eligible = result["consensus_eligible"]
    labels = _respondent_label_values(result["n_respondents"], respondent_ids)
    order = _ranked_rows(counts, top)
    label_name = "identifier" if respondent_ids is not None else "index"
    if result["min_valid_indices"] is None:
        lines.append(f"top flagged respondents ({label_name}, flag_count):")
        for idx in order:
            lines.append(f"  {labels[int(idx)]}\t{int(counts[idx])}")
    else:
        lines.append(
            f"top flagged respondents ({label_name}, flag_count, valid_index_count, eligible):"
        )
        for idx in order:
            lines.append(
                f"  {labels[int(idx)]}\t{int(counts[idx])}\t{int(valid_counts[idx])}"
                f"\t{int(bool(eligible[idx]))}"
            )
    return "\n".join(lines)


def _write_screen_json(
    handle: TextIO,
    result: ScreenResult,
    respondent_ids: list[str] | None = None,
) -> None:
    """Write screening JSON while bounding respondent-array allocation."""
    summary = {
        name: {
            "mean": stats["mean"] if np.isfinite(stats["mean"]) else None,
            "std": stats["std"] if np.isfinite(stats["std"]) else None,
            "min": stats["min"] if np.isfinite(stats["min"]) else None,
            "max": stats["max"] if np.isfinite(stats["max"]) else None,
            "n_valid": stats["n_valid"],
            "n_unavailable": stats["n_unavailable"],
            "n_flagged": stats["n_flagged"],
            "flag_rate": stats["flag_rate"] if np.isfinite(stats["flag_rate"]) else None,
        }
        for name, stats in result["summary"].items()
    }
    payload = {
        "n_respondents": result["n_respondents"],
        "n_indices": result["n_indices"],
        "indices_used": result["indices_used"],
        "errors": result["errors"],
        "thresholds": result["thresholds"],
        "threshold_sources": result["threshold_sources"],
        "percentiles": result["percentiles"],
        "flag_counts": _JsonArray(np.asarray(result["flag_counts"]), "integer"),
        "valid_index_counts": _JsonArray(
            np.asarray(result["valid_index_counts"]),
            "integer",
        ),
        "consensus_eligible": _JsonArray(
            np.asarray(result["consensus_eligible"]),
            "boolean",
        ),
        "consensus_flags": _JsonArray(np.asarray(result["consensus_flags"]), "boolean"),
        "min_flags": result["min_flags"],
        "min_valid_indices": result["min_valid_indices"],
        "scores": {
            name: _JsonArray(np.asarray(arr), "number") for name, arr in result["scores"].items()
        },
        "flags": {
            name: _JsonArray(np.asarray(arr), "boolean") for name, arr in result["flags"].items()
        },
        "summary": summary,
    }
    if respondent_ids is not None:
        payload["respondent_ids"] = _JsonArray(
            _respondent_label_values(result["n_respondents"], respondent_ids),
            "string",
        )
    _write_json_value(handle, payload)


def _write_screen_csv(
    handle: TextIO,
    result: ScreenResult,
    respondent_ids: list[str] | None = None,
) -> None:
    """Write respondent-aligned screening results directly to a CSV stream."""
    n = result["n_respondents"]
    scores = result["scores"]
    flags = result["flags"]
    fieldnames = [
        "respondent",
        "flag_count",
        "valid_index_count",
        "consensus_eligible",
        "consensus_flag",
    ]
    for name in result["indices_used"]:
        fieldnames.extend([f"{name}_score", f"{name}_flag"])

    counts = np.asarray(result["flag_counts"])
    valid_counts = np.asarray(result["valid_index_counts"])
    eligible = np.asarray(result["consensus_eligible"])
    consensus = np.asarray(result["consensus_flags"])
    labels = _respondent_label_values(n, respondent_ids)
    writer = csv.writer(handle)
    writer.writerow(fieldnames)
    # Convert bounded column chunks at once rather than formatting cell by cell.
    for start in range(0, n, _CSV_CHUNK_SIZE):
        stop = min(start + _CSV_CHUNK_SIZE, n)
        columns: list[Sequence[object]] = [
            labels[start:stop],
            _csv_int_column(counts, start, stop),
            _csv_int_column(valid_counts, start, stop),
            _csv_bit_column(eligible, start, stop),
            _csv_bit_column(consensus, start, stop),
        ]
        for name in result["indices_used"]:
            columns.append(_csv_float_column(scores[name], start, stop))
            columns.append(_csv_bit_column(flags[name], start, stop))
        writer.writerows(zip(*columns, strict=True))


def _emit_composite_text(report: CompositeReport, top: int) -> str:
    scores = report.scores
    flags = report.flags
    probabilities = report.probabilities
    component_scores = report.component_scores
    valid_index_counts = report.valid_index_counts
    order = _ranked_rows(scores, top)
    labels = _respondent_label_values(len(scores), report.respondent_ids)
    label_name = "identifier" if report.respondent_ids is not None else "index"
    lines = [
        f"respondents: {len(scores)}",
        f"method: {report.method}",
        f"standardized: {str(report.standardized).lower()}",
    ]
    if report.weights:
        lines.append(
            "weights: " + ", ".join(f"{name}={weight:g}" for name, weight in report.weights.items())
        )
    if report.min_valid_indices is not None:
        lines.append(f"minimum valid indices: {report.min_valid_indices}")
    if probabilities is not None:
        lines.append("probability: logistic (uncalibrated)")
    if flags is not None:
        assert report.flag_threshold is not None
        threshold_source = "percentile" if report.flag_percentile is not None else "fixed"
        lines.append(f"threshold: {report.flag_threshold:g} ({threshold_source})")
        if report.flag_percentile is not None:
            lines.append(f"percentile: {report.flag_percentile:g}")
        lines.append(f"flagged: {int(np.sum(flags))}")
    if report.errors:
        lines.append("errors:")
        for name, message in sorted(report.errors.items()):
            lines.append(f"  {name}: {message}")
    detail_names = list(component_scores) if component_scores is not None else []
    if detail_names:
        lines.append("indices: " + ", ".join(detail_names))
    columns = [label_name, "score"]
    if probabilities is not None:
        columns.append("probability")
    if flags is not None:
        columns.append("flag")
    if component_scores is not None:
        columns.extend(["valid_indices", *detail_names])
    lines.append(f"top composite scores ({', '.join(columns)}):")
    for idx in order:
        fields = [str(labels[int(idx)]), f"{float(scores[idx]):.6f}"]
        if probabilities is not None:
            fields.append(f"{float(probabilities[idx]):.6f}")
        if flags is not None:
            fields.append(str(int(bool(flags[idx]))))
        if component_scores is not None:
            assert valid_index_counts is not None
            fields.append(str(int(valid_index_counts[idx])))
            fields.extend(f"{float(component_scores[name][idx]):.6f}" for name in detail_names)
        lines.append("  " + "\t".join(fields))
    return "\n".join(lines)


def _write_composite_json(handle: TextIO, report: CompositeReport) -> None:
    """Write composite JSON while bounding respondent-array allocation."""
    scores = report.scores
    payload: dict[str, object] = {
        "method": report.method,
        "standardized": report.standardized,
        "scores": _JsonArray(scores, "number"),
        "n_respondents": len(scores),
        "errors": dict(report.errors),
    }
    if report.weights:
        payload["weights"] = dict(report.weights)
    if report.min_valid_indices is not None:
        payload["min_valid_indices"] = report.min_valid_indices
    if report.probabilities is not None:
        payload["probability_scale"] = "uncalibrated_logistic"
        payload["probabilities"] = _JsonArray(np.asarray(report.probabilities), "number")
    if report.flags is not None:
        assert report.flag_threshold is not None
        payload["threshold"] = report.flag_threshold
        payload["threshold_source"] = (
            "percentile" if report.flag_percentile is not None else "fixed"
        )
        if report.flag_percentile is not None:
            payload["percentile"] = report.flag_percentile
        payload["flags"] = _JsonArray(np.asarray(report.flags), "boolean")
    if report.component_scores is not None:
        assert report.valid_index_counts is not None
        payload["indices_used"] = list(report.component_scores)
        payload["component_scores"] = {
            name: _JsonArray(np.asarray(values), "number")
            for name, values in report.component_scores.items()
        }
        payload["valid_index_counts"] = _JsonArray(
            np.asarray(report.valid_index_counts),
            "integer",
        )
    if report.respondent_ids is not None:
        payload["respondent_ids"] = _JsonArray(
            _respondent_label_values(len(scores), report.respondent_ids),
            "string",
        )
    _write_json_value(handle, payload)


def _write_composite_csv(handle: TextIO, report: CompositeReport) -> None:
    """Write respondent-aligned composite scores directly to a CSV stream."""
    scores = report.scores
    flags = report.flags
    probabilities = report.probabilities
    component_scores = report.component_scores
    valid_index_counts = report.valid_index_counts
    detail_names = list(component_scores) if component_scores is not None else []
    writer = csv.writer(handle)
    header = ["respondent", "composite_score"]
    if probabilities is not None:
        header.append("composite_probability")
    if flags is not None:
        header.append("composite_flag")
    if component_scores is not None:
        header.extend(["valid_index_count", *(f"{name}_score" for name in detail_names)])
    writer.writerow(header)
    n_respondents = len(scores)
    labels = _respondent_label_values(n_respondents, report.respondent_ids)
    for start in range(0, n_respondents, _CSV_CHUNK_SIZE):
        stop = min(start + _CSV_CHUNK_SIZE, n_respondents)
        columns: list[Sequence[object]] = [
            labels[start:stop],
            _csv_float_column(scores, start, stop),
        ]
        if probabilities is not None:
            columns.append(_csv_float_column(probabilities, start, stop))
        if flags is not None:
            columns.append(_csv_bit_column(flags, start, stop))
        if component_scores is not None:
            assert valid_index_counts is not None
            columns.append(_csv_int_column(valid_index_counts, start, stop))
            columns.extend(
                _csv_float_column(component_scores[name], start, stop) for name in detail_names
            )
        writer.writerows(zip(*columns, strict=True))


def _emit_response_time_text(report: ResponseTimeReport, top: int) -> str:
    """Render timing results as a compact human-readable summary."""
    scores = report.scores
    labels = _respondent_label_values(len(scores), report.respondent_ids)
    order = _ranked_rows(scores, top, report.direction)
    label_name = "identifier" if report.respondent_ids is not None else "index"
    lines = [
        f"respondents: {len(scores)}",
        f"metric: {report.metric}",
        f"flag direction: {report.direction}",
        f"threshold: {report.cutoff:g}",
        f"flagged: {int(np.sum(report.flags))}",
        f"top suspicious respondents ({label_name}, score):",
    ]
    for index in order:
        lines.append(f"  {labels[int(index)]}\t{float(scores[index]):.6f}")
    return "\n".join(lines)


def _write_response_time_json(handle: TextIO, report: ResponseTimeReport) -> None:
    """Write timing JSON while bounding respondent-array allocation."""
    scores = report.scores
    payload: dict[str, object] = {
        "n_respondents": len(scores),
        "metric": report.metric,
        "flag_direction": report.direction,
        "threshold": report.cutoff,
        "scores": _JsonArray(scores, "number"),
        "flags": _JsonArray(report.flags, "boolean"),
    }
    if report.respondent_ids is not None:
        payload["respondent_ids"] = _JsonArray(
            _respondent_label_values(len(scores), report.respondent_ids),
            "string",
        )
    _write_json_value(handle, payload)


def _write_response_time_csv(handle: TextIO, report: ResponseTimeReport) -> None:
    """Write respondent-aligned timing scores and flags directly to a CSV stream."""
    scores = report.scores
    n_respondents = len(scores)
    labels = _respondent_label_values(n_respondents, report.respondent_ids)
    writer = csv.writer(handle)
    writer.writerow(["respondent", "response_time_score", "response_time_flag"])
    for start in range(0, n_respondents, _CSV_CHUNK_SIZE):
        stop = min(start + _CSV_CHUNK_SIZE, n_respondents)
        writer.writerows(
            zip(
                labels[start:stop],
                _csv_float_column(scores, start, stop),
                _csv_bit_column(report.flags, start, stop),
                strict=True,
            )
        )
