"""Text, JSON, and CSV serializers for command-line results."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ier.types import IndexCatalog, ScreenResult


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


def _json_numbers(values: np.ndarray) -> list[float | None]:
    """Convert numeric arrays to strict-JSON values, representing non-finite values as null."""
    return [float(value) if np.isfinite(value) else None for value in values]


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
    labels = _respondent_label_values(result["n_respondents"], respondent_ids)
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
        payload["respondent_ids"] = _respondent_label_values(
            result["n_respondents"], respondent_ids
        )
    return json.dumps(payload, indent=2, allow_nan=False)


def _write_screen_csv(
    handle: TextIO,
    result: ScreenResult,
    respondent_ids: list[str] | None = None,
) -> None:
    """Write respondent-aligned screening results directly to a CSV stream."""
    n = result["n_respondents"]
    scores = result["scores"]
    flags = result["flags"]
    fieldnames = ["respondent", "flag_count", "consensus_flag"]
    for name in result["indices_used"]:
        fieldnames.extend([f"{name}_score", f"{name}_flag"])

    counts = np.asarray(result["flag_counts"])
    consensus = np.asarray(result["consensus_flags"])
    labels = _respondent_label_values(n, respondent_ids)
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(n):
        row: dict[str, object] = {
            "respondent": labels[i],
            "flag_count": int(counts[i]),
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
) -> str:
    order = np.argsort(scores)[::-1][: max(top, 0)]
    labels = _respondent_label_values(len(scores), respondent_ids)
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
        payload["respondent_ids"] = _respondent_label_values(len(scores), respondent_ids)
    return json.dumps(payload, indent=2, allow_nan=False)


def _write_composite_csv(
    handle: TextIO,
    scores: np.ndarray,
    respondent_ids: list[str] | None = None,
) -> None:
    """Write respondent-aligned composite scores directly to a CSV stream."""
    writer = csv.writer(handle)
    writer.writerow(["respondent", "composite_score"])
    labels = _respondent_label_values(len(scores), respondent_ids)
    for label, score in zip(labels, scores, strict=True):
        writer.writerow([label, _csv_number(score)])


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
    labels = _respondent_label_values(len(scores), respondent_ids)
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
        payload["respondent_ids"] = _respondent_label_values(len(scores), respondent_ids)
    return json.dumps(payload, indent=2, allow_nan=False)


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
