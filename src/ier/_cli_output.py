"""Text, JSON, and CSV serializers for command-line results."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np

from ier._cli_composite import (
    validate_composite_components,
    validate_composite_flags,
    validate_composite_probabilities,
)

if TYPE_CHECKING:
    from ier.types import (
        FlagConsensusArchive,
        IndexCatalog,
        InspectableArchive,
        ResponseTimeThresholdSource,
        ScreenResult,
    )


_JSON_ARRAY_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class _JsonArray:
    """A large one-dimensional value sequence serialized in bounded chunks."""

    values: Sequence[object] | np.ndarray
    kind: Literal["number", "integer", "boolean", "string"]


def _write_output(text: str, path: Path | None) -> None:
    with _output_stream(path) as handle:
        handle.write(text)
        if path is None or path == Path("-"):
            handle.write("\n")


@contextmanager
def _output_stream(path: Path | None) -> Iterator[TextIO]:
    """Yield a text output stream without closing standard output."""
    if path is None or path == Path("-"):
        yield sys.stdout
        return
    if path.suffix.casefold() == ".gz":
        with gzip.open(path, mode="wt", newline="", encoding="utf-8") as handle:
            yield handle
        return
    with path.open(mode="w", newline="", encoding="utf-8") as handle:
        yield handle


def _json_chunk_values(array: _JsonArray, start: int, stop: int) -> list[object]:
    """Materialize one bounded JSON-ready chunk from a large value sequence."""
    values = array.values[start:stop]
    if array.kind == "number":
        numeric = np.asarray(values, dtype=float)
        chunk: list[object] = numeric.tolist()
        for index in np.flatnonzero(~np.isfinite(numeric)):
            chunk[int(index)] = None
        return chunk
    if array.kind == "integer":
        return [int(value) for value in np.asarray(values, dtype=np.int_)]
    if array.kind == "boolean":
        return [bool(value) for value in np.asarray(values, dtype=np.bool_)]
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


def _write_json_output(
    path: Path | None,
    writer: Callable[[TextIO], None],
) -> None:
    """Write JSON to a plain, gzip, or standard-output destination."""
    with _output_stream(path) as handle:
        writer(handle)
        if path is None or path == Path("-"):
            handle.write("\n")


def _csv_number(value: float) -> float | None:
    """Return a CSV-safe number, using an empty cell for non-finite values."""
    return float(value) if np.isfinite(value) else None


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
            }
        )


def _archive_info_payload(archive: InspectableArchive) -> dict[str, object]:
    """Build one compact JSON-ready summary from a validated archive."""
    payload: dict[str, object] = {
        "schema_version": archive["schema_version"],
        "result_type": archive["result_type"],
    }
    if archive["result_type"] == "response_time_mixture_model":
        payload.update(
            {
                "n_components": archive["n_components"],
                "fast_component": archive["fast_component"],
                "log_transform": archive["log_transform"],
                "weights": _JsonArray(archive["weights"], "number"),
                "means": _JsonArray(archive["means"], "number"),
                "variances": _JsonArray(archive["variances"], "number"),
            }
        )
        return payload
    if archive["result_type"] == "psychsyn_model":
        payload.update(
            {
                "n_items": archive["n_items"],
                "n_pairs": archive["n_pairs"],
                "critval": archive["critval"],
                "mode": "antonym" if archive["anto"] else "synonym",
            }
        )
        return payload

    payload.update(
        {
            "n_respondents": archive["n_respondents"],
            "has_respondent_ids": archive["respondent_ids"] is not None,
        }
    )
    if archive["result_type"] == "flag_consensus":
        n_flagged = int(np.sum(archive["consensus_flags"]))
        n_eligible = int(np.sum(archive["consensus_eligible"]))
        payload.update(
            {
                "n_signals": archive["n_signals"],
                "signals": archive["signal_names"],
                "availability_scores": list(archive["scores"]),
                "min_flags": archive["min_flags"],
                "min_valid_signals": archive["min_valid_signals"],
                "n_eligible": n_eligible,
                "n_consensus_flagged": n_flagged,
                "consensus_rate": n_flagged / archive["n_respondents"],
            }
        )
    elif archive["result_type"] == "response_time":
        n_flagged = int(np.sum(archive["flags"]))
        payload.update(
            {
                "metric": archive["metric"],
                "flag_direction": archive["flag_direction"],
                "threshold": archive["threshold"],
                "threshold_source": archive["threshold_source"],
                "percentile": archive["percentile"],
                "n_flagged": n_flagged,
                "flag_rate": n_flagged / archive["n_respondents"],
            }
        )
    else:
        payload.update(
            {
                "n_indices": len(archive["scores"]),
                "indices": list(archive["scores"]),
                "errors": archive["errors"],
            }
        )
    return payload


def _emit_archive_info_text(archive: InspectableArchive) -> str:
    """Render a compact human-readable summary of a validated archive."""
    lines = [
        f"result type: {archive['result_type']}",
        f"schema version: {archive['schema_version']}",
    ]
    if archive["result_type"] == "response_time_mixture_model":
        lines.extend(
            (
                f"components: {archive['n_components']}",
                f"log transform: {'yes' if archive['log_transform'] else 'no'}",
                f"fast component: {archive['fast_component']}",
                "parameters:",
            )
        )
        for component, (weight, mean, variance) in enumerate(
            zip(
                archive["weights"],
                archive["means"],
                archive["variances"],
                strict=True,
            )
        ):
            lines.append(f"  {component}: weight={weight:g} mean={mean:g} variance={variance:g}")
        return "\n".join(lines)
    if archive["result_type"] == "psychsyn_model":
        lines.extend(
            (
                f"mode: {'antonym' if archive['anto'] else 'synonym'}",
                f"items: {archive['n_items']}",
                f"pairs: {archive['n_pairs']}",
                f"correlation threshold: {archive['critval']:g}",
            )
        )
        return "\n".join(lines)

    lines.extend(
        (
            f"respondents: {archive['n_respondents']}",
            f"respondent identifiers: {'yes' if archive['respondent_ids'] is not None else 'no'}",
        )
    )
    if archive["result_type"] == "flag_consensus":
        n_flagged = int(np.sum(archive["consensus_flags"]))
        n_eligible = int(np.sum(archive["consensus_eligible"]))
        min_valid = archive["min_valid_signals"]
        coverage_rule = "all rows" if min_valid is None else f"min_valid_signals={min_valid}"
        lines.extend(
            (
                f"signals ({archive['n_signals']}): {', '.join(archive['signal_names'])}",
                "availability scores: " + (", ".join(archive["scores"]) or "none"),
                f"consensus eligible: {n_eligible}/{archive['n_respondents']} ({coverage_rule})",
                f"consensus flagged: {n_flagged}/{archive['n_respondents']} "
                f"({n_flagged / archive['n_respondents']:.1%}, "
                f"min_flags={archive['min_flags']})",
            )
        )
    elif archive["result_type"] == "response_time":
        source = archive["threshold_source"]
        if source == "percentile":
            percentile = archive["percentile"]
            assert percentile is not None
            threshold_label = f"percentile={percentile:g}"
        else:
            threshold_label = "legacy" if source is None else source
        n_flagged = int(np.sum(archive["flags"]))
        lines.extend(
            (
                f"metric: {archive['metric']}",
                f"flag direction: {archive['flag_direction']}",
                f"threshold: {archive['threshold']:g} ({threshold_label})",
                f"flagged: {n_flagged}/{archive['n_respondents']} "
                f"({n_flagged / archive['n_respondents']:.1%})",
            )
        )
    else:
        lines.append(f"indices ({len(archive['scores'])}): {', '.join(archive['scores'])}")
        lines.append(f"soft failures: {len(archive['errors'])}")
        for name, message in archive["errors"].items():
            lines.append(f"  {name}: {message}")
    return "\n".join(lines)


def _emit_flag_consensus_text(result: FlagConsensusArchive, top: int) -> str:
    """Render a compact cross-domain consensus summary."""
    n_respondents = result["n_respondents"]
    n_flagged = int(np.sum(result["consensus_flags"]))
    lines = [
        f"respondents: {n_respondents}",
        f"signals: {', '.join(result['signal_names'])}",
        f"consensus flagged: {n_flagged}/{n_respondents} "
        f"({n_flagged / n_respondents:.1%}, min_flags={result['min_flags']})",
        "signal coverage:",
    ]
    for name in result["signal_names"]:
        score = result["scores"].get(name)
        n_valid = n_respondents if score is None else int(np.count_nonzero(~np.isnan(score)))
        n_signal_flagged = int(np.count_nonzero(result["flags"][name]))
        rate = n_signal_flagged / n_valid if n_valid else float("nan")
        rate_text = "n/a" if not np.isfinite(rate) else f"{rate:.1%}"
        lines.append(
            f"  {name}: valid={n_valid}/{n_respondents}, "
            f"flagged={n_signal_flagged}/{n_valid} ({rate_text})"
        )
    if result["min_valid_signals"] is not None:
        lines.append(
            f"consensus eligible: {int(np.sum(result['consensus_eligible']))}/{n_respondents} "
            f"(min_valid_signals={result['min_valid_signals']})"
        )

    counts = result["flag_counts"]
    valid_counts = result["valid_signal_counts"]
    eligible = result["consensus_eligible"]
    labels = _respondent_label_values(n_respondents, result["respondent_ids"])
    order = np.argsort(counts)[::-1][: max(top, 0)]
    label_name = "identifier" if result["respondent_ids"] is not None else "index"
    lines.append(
        f"top flagged respondents ({label_name}, flag_count, valid_signal_count, eligible):"
    )
    for index in order:
        lines.append(
            f"  {labels[int(index)]}\t{int(counts[index])}\t{int(valid_counts[index])}"
            f"\t{int(bool(eligible[index]))}"
        )
    return "\n".join(lines)


def _write_flag_consensus_json(handle: TextIO, result: FlagConsensusArchive) -> None:
    """Write reusable consensus JSON with bounded respondent-array allocation."""
    payload = {
        "n_respondents": result["n_respondents"],
        "n_signals": result["n_signals"],
        "signal_names": result["signal_names"],
        "flag_counts": _JsonArray(result["flag_counts"], "integer"),
        "valid_signal_counts": _JsonArray(result["valid_signal_counts"], "integer"),
        "consensus_eligible": _JsonArray(result["consensus_eligible"], "boolean"),
        "consensus_flags": _JsonArray(result["consensus_flags"], "boolean"),
        "min_flags": result["min_flags"],
        "min_valid_signals": result["min_valid_signals"],
        "scores": {name: _JsonArray(values, "number") for name, values in result["scores"].items()},
        "flags": {name: _JsonArray(values, "boolean") for name, values in result["flags"].items()},
    }
    if result["respondent_ids"] is not None:
        payload["respondent_ids"] = _JsonArray(result["respondent_ids"], "string")
    _write_json_value(handle, payload)


def _write_flag_consensus_csv(handle: TextIO, result: FlagConsensusArchive) -> None:
    """Write respondent-aligned consensus inputs and decisions to CSV."""
    fieldnames = [
        "respondent",
        "flag_count",
        "valid_signal_count",
        "consensus_eligible",
        "consensus_flag",
    ]
    for name in result["signal_names"]:
        if name in result["scores"]:
            fieldnames.append(f"{name}_score")
        fieldnames.append(f"{name}_flag")

    labels = _respondent_label_values(result["n_respondents"], result["respondent_ids"])
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for index in range(result["n_respondents"]):
        row: dict[str, object] = {
            "respondent": labels[index],
            "flag_count": int(result["flag_counts"][index]),
            "valid_signal_count": int(result["valid_signal_counts"][index]),
            "consensus_eligible": int(bool(result["consensus_eligible"][index])),
            "consensus_flag": int(bool(result["consensus_flags"][index])),
        }
        for name in result["signal_names"]:
            if name in result["scores"]:
                row[f"{name}_score"] = _csv_number(result["scores"][name][index])
            row[f"{name}_flag"] = int(bool(result["flags"][name][index]))
        writer.writerow(row)


def _write_archive_info_json(handle: TextIO, archive: InspectableArchive) -> None:
    """Write strict JSON metadata for a validated archive."""
    _write_json_value(handle, _archive_info_payload(archive))


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
    order = np.argsort(counts)[::-1][: max(top, 0)]
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


def _emit_screen_json(
    result: ScreenResult,
    respondent_ids: list[str] | None = None,
) -> str:
    output = StringIO()
    _write_screen_json(output, result, respondent_ids)
    return output.getvalue()


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
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(n):
        row: dict[str, object] = {
            "respondent": labels[i],
            "flag_count": int(counts[i]),
            "valid_index_count": int(valid_counts[i]),
            "consensus_eligible": int(bool(eligible[i])),
            "consensus_flag": int(bool(consensus[i])),
        }
        for name in result["indices_used"]:
            row[f"{name}_score"] = _csv_number(scores[name][i])
            row[f"{name}_flag"] = int(bool(flags[name][i]))
        writer.writerow(row)


def _emit_composite_text(
    scores: np.ndarray,
    method: str,
    top: int,
    respondent_ids: list[str] | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    errors: Mapping[str, str] | None = None,
    component_scores: Mapping[str, np.ndarray] | None = None,
    valid_index_counts: np.ndarray | None = None,
    standardized: bool = True,
    flags: np.ndarray | None = None,
    flag_threshold: float | None = None,
    flag_percentile: float | None = None,
    probabilities: np.ndarray | None = None,
) -> str:
    validate_composite_components(len(scores), component_scores, valid_index_counts)
    validate_composite_flags(len(scores), flags, flag_threshold, flag_percentile)
    validate_composite_probabilities(len(scores), probabilities)
    finite_rows = np.flatnonzero(np.isfinite(scores))
    order = finite_rows[np.argsort(scores[finite_rows])[::-1]][: max(top, 0)]
    labels = _respondent_label_values(len(scores), respondent_ids)
    label_name = "identifier" if respondent_ids is not None else "index"
    lines = [
        f"respondents: {len(scores)}",
        f"method: {method}",
        f"standardized: {str(standardized).lower()}",
    ]
    if weights:
        lines.append(
            "weights: " + ", ".join(f"{name}={weight:g}" for name, weight in weights.items())
        )
    if min_valid_indices is not None:
        lines.append(f"minimum valid indices: {min_valid_indices}")
    if probabilities is not None:
        lines.append("probability: logistic (uncalibrated)")
    if flags is not None:
        assert flag_threshold is not None
        threshold_source = "percentile" if flag_percentile is not None else "fixed"
        lines.append(f"threshold: {flag_threshold:g} ({threshold_source})")
        if flag_percentile is not None:
            lines.append(f"percentile: {flag_percentile:g}")
        lines.append(f"flagged: {int(np.sum(flags))}")
    if errors:
        lines.append("errors:")
        for name, message in sorted(errors.items()):
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


def _emit_composite_json(
    scores: np.ndarray,
    method: str,
    respondent_ids: list[str] | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    errors: Mapping[str, str] | None = None,
    component_scores: Mapping[str, np.ndarray] | None = None,
    valid_index_counts: np.ndarray | None = None,
    standardized: bool = True,
    flags: np.ndarray | None = None,
    flag_threshold: float | None = None,
    flag_percentile: float | None = None,
    probabilities: np.ndarray | None = None,
) -> str:
    output = StringIO()
    _write_composite_json(
        output,
        scores,
        method,
        respondent_ids,
        weights,
        min_valid_indices,
        errors,
        component_scores,
        valid_index_counts,
        standardized,
        flags,
        flag_threshold,
        flag_percentile,
        probabilities,
    )
    return output.getvalue()


def _write_composite_json(
    handle: TextIO,
    scores: np.ndarray,
    method: str,
    respondent_ids: list[str] | None = None,
    weights: Mapping[str, float] | None = None,
    min_valid_indices: int | None = None,
    errors: Mapping[str, str] | None = None,
    component_scores: Mapping[str, np.ndarray] | None = None,
    valid_index_counts: np.ndarray | None = None,
    standardized: bool = True,
    flags: np.ndarray | None = None,
    flag_threshold: float | None = None,
    flag_percentile: float | None = None,
    probabilities: np.ndarray | None = None,
) -> None:
    """Write composite JSON while bounding respondent-array allocation."""
    validate_composite_components(len(scores), component_scores, valid_index_counts)
    validate_composite_flags(len(scores), flags, flag_threshold, flag_percentile)
    validate_composite_probabilities(len(scores), probabilities)
    payload: dict[str, object] = {
        "method": method,
        "standardized": standardized,
        "scores": _JsonArray(scores, "number"),
        "n_respondents": len(scores),
        "errors": dict(errors or {}),
    }
    if weights:
        payload["weights"] = dict(weights)
    if min_valid_indices is not None:
        payload["min_valid_indices"] = min_valid_indices
    if probabilities is not None:
        payload["probability_scale"] = "uncalibrated_logistic"
        payload["probabilities"] = _JsonArray(np.asarray(probabilities), "number")
    if flags is not None:
        assert flag_threshold is not None
        payload["threshold"] = flag_threshold
        payload["threshold_source"] = "percentile" if flag_percentile is not None else "fixed"
        if flag_percentile is not None:
            payload["percentile"] = flag_percentile
        payload["flags"] = _JsonArray(np.asarray(flags), "boolean")
    if component_scores is not None:
        assert valid_index_counts is not None
        payload["indices_used"] = list(component_scores)
        payload["component_scores"] = {
            name: _JsonArray(np.asarray(values), "number")
            for name, values in component_scores.items()
        }
        payload["valid_index_counts"] = _JsonArray(
            np.asarray(valid_index_counts),
            "integer",
        )
    if respondent_ids is not None:
        payload["respondent_ids"] = _JsonArray(
            _respondent_label_values(len(scores), respondent_ids),
            "string",
        )
    _write_json_value(handle, payload)


def _write_composite_csv(
    handle: TextIO,
    scores: np.ndarray,
    respondent_ids: list[str] | None = None,
    component_scores: Mapping[str, np.ndarray] | None = None,
    valid_index_counts: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> None:
    """Write respondent-aligned composite scores directly to a CSV stream."""
    validate_composite_components(len(scores), component_scores, valid_index_counts)
    validate_composite_probabilities(len(scores), probabilities)
    if flags is not None and len(flags) != len(scores):
        raise ValueError("composite flag length must match composite score length")
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
    labels = _respondent_label_values(len(scores), respondent_ids)
    for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        row: list[object] = [label, _csv_number(score)]
        if probabilities is not None:
            row.append(_csv_number(probabilities[index]))
        if flags is not None:
            row.append(int(bool(flags[index])))
        if component_scores is not None:
            assert valid_index_counts is not None
            row.append(int(valid_index_counts[index]))
            row.extend(_csv_number(component_scores[name][index]) for name in detail_names)
        writer.writerow(row)


def _emit_response_time_text(
    scores: np.ndarray,
    flags: np.ndarray,
    metric: str,
    direction: Literal["high", "low"],
    cutoff: float,
    top: int,
    respondent_ids: list[str] | None = None,
    *,
    threshold_source: ResponseTimeThresholdSource | None = None,
    percentile: float | None = None,
) -> str:
    """Render timing results as a compact human-readable summary."""
    labels = _respondent_label_values(len(scores), respondent_ids)
    valid_indices = np.flatnonzero(np.isfinite(scores))
    order = valid_indices[np.argsort(scores[valid_indices])]
    if direction == "high":
        order = order[::-1]
    order = order[: max(top, 0)]
    label_name = "identifier" if respondent_ids is not None else "index"
    threshold_line = f"threshold: {cutoff:g}"
    if threshold_source is not None:
        threshold_line += f" ({threshold_source})"
    lines = [
        f"respondents: {len(scores)}",
        f"metric: {metric}",
        f"flag direction: {direction}",
        threshold_line,
    ]
    if percentile is not None:
        lines.append(f"percentile: {percentile:g}")
    lines.extend(
        [
            f"flagged: {int(np.sum(flags))}",
            f"top suspicious respondents ({label_name}, score):",
        ]
    )
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
    *,
    threshold_source: ResponseTimeThresholdSource | None = None,
    percentile: float | None = None,
) -> str:
    """Render timing results as strict JSON."""
    output = StringIO()
    _write_response_time_json(
        output,
        scores,
        flags,
        metric,
        direction,
        cutoff,
        respondent_ids,
        threshold_source=threshold_source,
        percentile=percentile,
    )
    return output.getvalue()


def _write_response_time_json(
    handle: TextIO,
    scores: np.ndarray,
    flags: np.ndarray,
    metric: str,
    direction: Literal["high", "low"],
    cutoff: float,
    respondent_ids: list[str] | None = None,
    *,
    threshold_source: ResponseTimeThresholdSource | None = None,
    percentile: float | None = None,
) -> None:
    """Write timing JSON while bounding respondent-array allocation."""
    payload: dict[str, object] = {
        "n_respondents": len(scores),
        "metric": metric,
        "flag_direction": direction,
        "threshold": cutoff,
        "scores": _JsonArray(scores, "number"),
        "flags": _JsonArray(flags, "boolean"),
    }
    if threshold_source is not None:
        payload["threshold_source"] = threshold_source
    if percentile is not None:
        payload["percentile"] = percentile
    if respondent_ids is not None:
        payload["respondent_ids"] = _JsonArray(
            _respondent_label_values(len(scores), respondent_ids),
            "string",
        )
    _write_json_value(handle, payload)


def _write_response_time_csv(
    handle: TextIO,
    scores: np.ndarray,
    flags: np.ndarray,
    respondent_ids: list[str] | None = None,
) -> None:
    """Write respondent-aligned timing scores and flags directly to a CSV stream."""
    writer = csv.writer(handle)
    writer.writerow(["respondent", "response_time_score", "response_time_flag"])
    labels = _respondent_label_values(len(scores), respondent_ids)
    for label, score, flag in zip(labels, scores, flags, strict=True):
        writer.writerow([label, _csv_number(score), int(bool(flag))])
