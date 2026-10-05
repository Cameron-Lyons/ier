"""Validated persistence for reusable results in versioned NumPy archives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile
from zlib import error as ZlibError

import numpy as np
from numpy.lib.format import MAGIC_PREFIX

from ier._archive_input import read_npz_member
from ier._atomic_output import atomic_output_path
from ier._flagging import threshold_flags
from ier._registry import (
    INDEX_REGISTRY,
    composite_index_names,
    validate_index_errors,
    validate_index_names,
)
from ier._validation import validate_score_array, validate_score_vectors
from ier.screen import _reduce_screen_results
from ier.tables import _respondent_id_values
from ier.types import (
    IndexThresholdSource,
    ResponseTimeFlagDirection,
    ResponseTimeMetric,
    ScoreArchiveResultType,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.lib.npyio import NpzFile
    from numpy.typing import ArrayLike

    from ier.types import (
        BoolArray,
        IndexPercentileMap,
        IndexThresholdMap,
        IndexThresholdSourceMap,
        IntArray,
        ResponseTimeArchive,
        ScoreArchive,
        ScreenArchive,
        ScreenIndexSummary,
        ScreenResult,
    )

_ARCHIVE_SCHEMA_VERSION = 1
_DEFAULT_COMPRESSION_LEVEL = 1
_SCREEN_SUMMARY_COLUMNS = ("mean", "std", "min", "max")
_SCREEN_DECISION_MEMBERS = (
    "n_indices",
    "min_flags",
    "thresholds",
    "threshold_sources",
    "percentiles",
    "flag_counts",
    "valid_index_counts",
    "consensus_eligible",
    "consensus_flags",
    "summary_columns",
    "summary_statistics",
    "summary_n_flagged",
    "summary_n_valid",
    "summary_n_unavailable",
    "summary_flag_rate",
)
_SCREEN_OPTIONAL_MEMBERS = ("min_valid_indices", "error_names", "error_messages", "respondent_ids")
_SCREEN_VECTOR_KINDS = {"b": "a boolean", "f": "a floating-point", "iu": "an integer"}
# Respondent-aligned decisions, their dtype kinds, and their mismatch messages.
_SCREEN_RESPONDENT_VECTORS = (
    ("flag_counts", "iu", "are inconsistent with its index flags"),
    ("valid_index_counts", "iu", "are inconsistent with its available scores"),
    ("consensus_eligible", "b", "is inconsistent with min_valid_indices"),
    ("consensus_flags", "b", "are inconsistent with min_flags and consensus eligibility"),
)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _literal_choice(value: object, literal: object, message: str) -> str:
    """Return one string option permitted by a ``Literal`` alias or raise ``ValueError``."""
    if not isinstance(value, str) or value not in get_args(literal):
        raise ValueError(message)
    return value


def _validate_archive_index_names(
    names: list[str],
    result_type: ScoreArchiveResultType,
    *,
    allow_empty: bool = False,
    label: str = "index",
) -> None:
    """Validate ordered score or error names against the relevant registry."""
    if not names and not allow_empty:
        raise ValueError("score archive does not contain reusable index scores")
    if any(not isinstance(name, str) for name in names):
        raise ValueError(f"score archive {label} names must be strings")
    if any(not name.strip() for name in names):
        raise ValueError(f"score archive {label} names must be nonblank")
    if len(names) != len(set(names)):
        raise ValueError(f"score archive {label} names must be unique")
    validate_index_names(
        names,
        composite_index_names() if result_type == "composite" else None,
    )


def _validate_error_metadata(
    names: list[str],
    messages: list[str],
    score_names: set[str],
    result_type: ScoreArchiveResultType,
) -> dict[str, str]:
    """Validate aligned soft-failure metadata and preserve its order."""
    if len(names) != len(messages):
        raise ValueError("score archive error names and messages must have equal lengths")
    _validate_archive_index_names(names, result_type, allow_empty=True, label="error")
    return validate_index_errors(
        dict(zip(names, messages, strict=True)),
        list(score_names),
        composite_index_names() if result_type == "composite" else None,
    )


def _validate_error_input(
    errors: Mapping[str, str] | None,
    score_names: set[str],
    result_type: ScoreArchiveResultType,
) -> dict[str, str]:
    """Validate caller-supplied soft failures before they are encoded."""
    if errors is None:
        return {}
    if not isinstance(errors, Mapping):
        raise TypeError("errors must be a mapping of registered index names to messages")
    error_items = list(errors.items())
    validated = _validate_error_metadata(
        [name for name, _ in error_items],
        [message for _, message in error_items],
        score_names,
        result_type,
    )
    _validate_archive_strings(list(validated.values()), name="archive error messages")
    return validated


def _validate_archive_strings(values: Sequence[str], *, name: str) -> None:
    """Reject text that NumPy's fixed-width Unicode format silently truncates."""
    if any(map(str.endswith, values, repeat("\0"))):
        raise ValueError(f"{name} cannot end with a NUL character")


def _validate_respondent_id_values(respondent_ids: Sequence[str], n_respondents: int) -> list[str]:
    """Validate aligned, unique, nonblank respondent identifiers before encoding."""
    # Shared with screen_table() so both raise the same exception types.
    values = _respondent_id_values(respondent_ids, n_respondents, prefix="archive ")
    if not all(map(str.strip, values)):
        raise ValueError("archive respondent IDs must be nonblank")
    if len(values) != len(set(values)):
        raise ValueError("archive respondent IDs must be unique")
    return values


def _validate_archived_respondent_ids(values: list[str], n_respondents: int) -> list[str]:
    """Validate the alignment, blank, and duplicate rules for decoded identifiers."""
    # tolist() on a NumPy Unicode array always yields str values with trailing
    # NULs already stripped, so the input type and NUL checks cannot fail here.
    if len(values) != n_respondents:
        raise ValueError("archive respondent ID count must match n_respondents")
    if not all(map(str.strip, values)):
        raise ValueError("archive respondent IDs must be nonblank")
    if len(values) != len(set(values)):
        raise ValueError("archive respondent IDs must be unique")
    return values


def _respondent_ids_member(
    respondent_ids: Sequence[str] | None, n_respondents: int
) -> dict[str, np.ndarray]:
    """Validate optional identifiers and return their archive member, if any."""
    if respondent_ids is None:
        return {}
    values = _validate_respondent_id_values(respondent_ids, n_respondents)
    return {"respondent_ids": np.asarray(values, dtype=np.str_)}


def _archive_header(result_type: str, n_respondents: int) -> dict[str, np.ndarray]:
    """Return the versioned metadata members that begin every result archive."""
    return {
        "schema_version": np.asarray(_ARCHIVE_SCHEMA_VERSION, dtype=np.int64),
        "result_type": np.asarray(result_type, dtype=np.str_),
        "n_respondents": np.asarray(n_respondents, dtype=np.int64),
    }


def _int64_setting(value: int, name: str) -> np.ndarray:
    """Encode one integer screen setting, rejecting values an int64 member cannot hold."""
    # screen() bounds min_flags only from below, so huge values are valid results.
    if isinstance(value, (int, np.integer)) and not _INT64_MIN <= int(value) <= _INT64_MAX:
        raise ValueError(f"screen result {name} must fit in a 64-bit signed integer")
    return np.asarray(value, dtype=np.int64)


def _summary_statistics(summary: Mapping[str, ScreenIndexSummary], names: list[str]) -> np.ndarray:
    """Encode each index's mean, std, min, and max as one row of a float matrix."""
    return np.asarray(
        [
            [
                summary[name]["mean"],
                summary[name]["std"],
                summary[name]["min"],
                summary[name]["max"],
            ]
            for name in names
        ],
        dtype=np.float64,
    ).reshape(len(names), len(_SCREEN_SUMMARY_COLUMNS))


def _require_npz_suffix(path: str | Path, *, label: str) -> Path:
    """Return an explicit archive destination after checking its suffix."""
    destination = Path(path)
    if destination.suffix.casefold() != ".npz":
        raise ValueError(f"{label} archive output path must end in .npz")
    return destination


def _resolve_compression_level(compression_level: int | None, *, compressed: bool) -> int:
    """Return a validated DEFLATE level, defaulting to the fastest setting."""
    if compression_level is None:
        return _DEFAULT_COMPRESSION_LEVEL
    if (
        isinstance(compression_level, bool)
        or not isinstance(compression_level, (int, np.integer))
        or not 1 <= compression_level <= 9
    ):
        raise ValueError("compression_level must be an integer from 1 to 9 or None")
    if not compressed:
        raise ValueError("compression_level requires compressed=True")
    return int(compression_level)


def _stream_npz_members(
    path: Path,
    payload: dict[str, np.ndarray],
    *,
    compressed: bool = False,
    compression_level: int = _DEFAULT_COMPRESSION_LEVEL,
) -> None:
    """Stream typed arrays directly into one stored or DEFLATE-compressed archive."""
    with ZipFile(
        path,
        mode="w",
        compression=ZIP_DEFLATED if compressed else ZIP_STORED,
        compresslevel=compression_level if compressed else None,
        allowZip64=True,
    ) as archive:
        for name, value in payload.items():
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                np.save(member, value, allow_pickle=False)


def _write_npz_atomically(
    path: Path,
    payload: dict[str, np.ndarray],
    *,
    compressed: bool = False,
    compression_level: int | None = None,
) -> None:
    """Atomically stream one typed, pickle-free NPZ archive into place."""
    if not isinstance(compressed, bool):
        raise ValueError("compressed must be a boolean")
    level = _resolve_compression_level(compression_level, compressed=compressed)
    if any(value.dtype.hasobject for value in payload.values()):
        raise ValueError("NPZ archive cannot contain object arrays")

    with atomic_output_path(path) as staged_path:
        _stream_npz_members(staged_path, payload, compressed=compressed, compression_level=level)


def _require_member(archive: NpzFile, name: str) -> np.ndarray:
    """Load one required pickle-free member with a contextual error."""
    if name not in archive.files:
        raise ValueError(f"NPZ archive is missing required member: {name}")
    try:
        value = read_npz_member(archive, name)
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(f"NPZ archive member {name} is not pickle-free") from error
        raise ValueError(f"NPZ archive member {name} is malformed: {error}") from error
    except (EOFError, BadZipFile, ZlibError, NotImplementedError, RuntimeError) as error:
        raise ValueError(f"NPZ archive member {name} cannot be read: {error}") from error
    if not isinstance(value, np.ndarray):
        raise ValueError(f"NPZ archive member {name} must be a NumPy array")
    return value


@contextmanager
def _open_npz_archive(path: str | Path, *, label: str) -> Iterator[NpzFile]:
    """Own the input stream, even when NumPy fails while opening the ZIP container."""
    with Path(path).open("rb") as handle:
        # A wrong-format NPY matrix can be much larger than an archive's metadata.
        # Reject it before NumPy reads its header or allocates its entire payload.
        if handle.read(len(MAGIC_PREFIX)) == MAGIC_PREFIX:
            raise ValueError(f"{label} archive must be an NPZ archive")
        handle.seek(0)
        try:
            loaded = np.load(handle, allow_pickle=False)
        except (EOFError, ValueError, BadZipFile) as error:
            raise ValueError(
                f"{label} archive could not be read as an NPZ archive: {error}"
            ) from error
        if isinstance(loaded, np.ndarray):
            raise ValueError(f"{label} archive must be an NPZ archive")
        with cast("NpzFile", loaded) as archive:
            yield archive


def _integer_value(value: np.ndarray, name: str) -> int:
    """Return one validated integer scalar member."""
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"NPZ archive member {name} must be an integer scalar")
    return int(value.item())


def _integer_scalar(archive: NpzFile, name: str) -> int:
    return _integer_value(_require_member(archive, name), name)


def _string_scalar(archive: NpzFile, name: str) -> str:
    value = _require_member(archive, name)
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"NPZ archive member {name} must be a Unicode string scalar")
    _validate_unicode_codepoints(value, name=name)
    return str(value.item())


def _string_values(value: np.ndarray, name: str) -> list[str]:
    """Return the decoded text of one validated Unicode vector member."""
    if value.ndim != 1 or value.dtype.kind != "U":
        raise ValueError(f"NPZ archive member {name} must be a Unicode string vector")
    _validate_unicode_codepoints(value, name=name)
    return cast("list[str]", value.tolist())


def _string_vector(archive: NpzFile, name: str) -> list[str]:
    return _string_values(_require_member(archive, name), name)


def _validate_unicode_codepoints(value: np.ndarray, *, name: str) -> None:
    """Check UTF-32 storage before NumPy creates Python strings from archive text."""
    # Depending on the NumPy conversion path, invalid codepoints can raise a
    # SystemError or even create invalid Python strings. Check in bounded slices
    # so a large ID vector does not need a full-size codepoint mask.
    codepoints = value.reshape(-1).view(np.dtype(f"{value.dtype.byteorder}u4"))
    for start in range(0, codepoints.size, 65_536):
        if np.any(codepoints[start : start + 65_536] > 0x10FFFF):
            raise ValueError(f"NPZ archive member {name} contains invalid Unicode")


def _numeric_scalar(archive: NpzFile, name: str) -> float:
    """Load one finite real numeric scalar from an archive."""
    return _finite_numeric_scalar(
        _require_member(archive, name),
        name=f"NPZ archive member {name}",
    )


def _finite_numeric_scalar(value: object, *, name: str) -> float:
    """Validate and return one finite real numeric scalar."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric scalar") from error
    if array.shape != () or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric scalar")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_boolean_vector(
    values: ArrayLike,
    n_respondents: int,
    *,
    name: str,
) -> BoolArray:
    """Validate one aligned one-dimensional boolean vector."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a boolean vector") from error
    if array.ndim != 1 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean vector")
    if len(array) != n_respondents:
        raise ValueError(f"{name} must match n_respondents")
    return cast("BoolArray", array)


def _read_archive_header(
    archive: NpzFile,
    *,
    label: str,
    result_types: object,
    allowed_members: frozenset[str] | None = None,
) -> tuple[int, str, int]:
    """Validate member names and the versioned metadata shared by every archive."""
    if len(archive.files) != len(set(archive.files)):
        raise ValueError(f"{label} archive member names must be unique")
    if allowed_members is not None:
        unexpected = set(archive.files) - allowed_members
        if unexpected:
            raise ValueError(f"{label} archive contains unexpected member: {min(unexpected)}")

    schema_version = _integer_scalar(archive, "schema_version")
    if schema_version != _ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {label} archive schema version: {schema_version}; "
            f"expected {_ARCHIVE_SCHEMA_VERSION}"
        )

    choices = " or ".join(map(repr, get_args(result_types)))
    result_type = _literal_choice(
        _string_scalar(archive, "result_type"),
        result_types,
        f"{label} archive result_type must be {choices}",
    )

    n_respondents = _integer_scalar(archive, "n_respondents")
    if n_respondents < 1:
        raise ValueError(f"{label} archive n_respondents must be positive")
    return schema_version, result_type, n_respondents


def _load_errors(
    archive: NpzFile,
    score_names: set[str],
    result_type: ScoreArchiveResultType,
) -> dict[str, str]:
    """Load aligned soft-failure metadata when present."""
    has_names = "error_names" in archive.files
    has_messages = "error_messages" in archive.files
    if has_names != has_messages:
        raise ValueError("score archive error names and messages must be stored together")
    if not has_names:
        return {}

    names = _string_vector(archive, "error_names")
    messages = _string_vector(archive, "error_messages")
    # Decoded Unicode members cannot end with NUL, so only registry rules apply.
    return _validate_error_metadata(names, messages, score_names, result_type)


def _load_respondent_ids(archive: NpzFile, n_respondents: int) -> list[str] | None:
    """Load optional aligned respondent identifiers."""
    if "respondent_ids" not in archive.files:
        return None
    respondent_ids = _string_vector(archive, "respondent_ids")
    return _validate_archived_respondent_ids(respondent_ids, n_respondents)


def _load_score_members(
    archive: NpzFile,
    result_type: ScoreArchiveResultType,
    n_respondents: int,
) -> dict[str, np.ndarray]:
    """Load declared raw registered-index score members in archive order."""
    if "index_names" not in archive.files:
        if result_type == "composite":
            raise ValueError(
                "composite archive does not include component scores; "
                "write it with --include-components"
            )
        raise ValueError("screen archive is missing required member: index_names")

    names = _string_vector(archive, "index_names")
    _validate_archive_index_names(names, result_type, allow_empty=result_type == "screen")

    expected_members = {f"score__{name}" for name in names}
    actual_members = {name for name in archive.files if name.startswith("score__")}
    missing = expected_members - actual_members
    extra = actual_members - expected_members
    if missing:
        raise ValueError(f"score archive is missing declared score member: {min(missing)}")
    if extra:
        raise ValueError(f"score archive contains undeclared score member: {min(extra)}")

    raw_scores = {name: _require_member(archive, f"score__{name}") for name in names}
    scores, _ = validate_score_vectors(raw_scores, n_respondents=n_respondents)
    return scores


def _read_score_archive(archive: NpzFile) -> ScoreArchive:
    """Validate one open NPZ archive and extract its reusable score payload."""
    schema_version, archived_type, n_respondents = _read_archive_header(
        archive, label="score", result_types=ScoreArchiveResultType
    )
    result_type = cast("ScoreArchiveResultType", archived_type)

    scores = _load_score_members(archive, result_type, n_respondents)
    errors = _load_errors(archive, set(scores), result_type)
    if not scores and not errors:
        raise ValueError("score archive does not contain reusable index scores or failures")
    respondent_ids = _load_respondent_ids(archive, n_respondents)
    return {
        "schema_version": schema_version,
        "result_type": result_type,
        "n_respondents": n_respondents,
        "scores": scores,
        "respondent_ids": respondent_ids,
        "errors": errors,
    }


def _screen_vector(
    members: Mapping[str, np.ndarray],
    name: str,
    *,
    kinds: str,
    length: int,
    aligned_with: str,
) -> np.ndarray:
    """Return one screen decision vector after checking its type and alignment."""
    value = members[name]
    if value.ndim != 1 or value.dtype.kind not in kinds:
        raise ValueError(f"screen archive {name} must be {_SCREEN_VECTOR_KINDS[kinds]} vector")
    if len(value) != length:
        raise ValueError(f"screen archive {name} must match {aligned_with}")
    return value


def _same_float(stored: float, expected: float) -> bool:
    """Compare one recorded statistic with its recomputed value, treating NaNs as equal."""
    return stored == expected or (math.isnan(stored) and math.isnan(expected))


def _screen_settings(
    members: Mapping[str, np.ndarray],
    n_indices: int,
    n_selected: int,
) -> tuple[int, int | None]:
    """Validate the recorded index count and respondent-level consensus settings."""
    if _integer_value(members["n_indices"], "n_indices") != n_indices:
        raise ValueError("screen archive n_indices must match index_names")
    min_flags = _integer_value(members["min_flags"], "min_flags")
    if min_flags < 1:
        raise ValueError("screen archive min_flags must be positive")
    if "min_valid_indices" not in members:
        return min_flags, None
    min_valid_indices = _integer_value(members["min_valid_indices"], "min_valid_indices")
    if not 1 <= min_valid_indices <= n_selected:
        raise ValueError(
            "screen archive min_valid_indices must be between 1 and the number of "
            f"selected indices ({n_selected})"
        )
    return min_flags, min_valid_indices


def _screen_cutoff(
    name: str,
    source: IndexThresholdSource,
    threshold: float,
    percentile: float,
) -> tuple[float | None, float | None]:
    """Validate one index's recorded cutoff provenance and return its public values."""
    if source == "presence":
        if not math.isnan(threshold):
            raise ValueError(f"screen archive threshold for presence-flagged {name} must be NaN")
        if not math.isnan(percentile):
            raise ValueError(f"screen archive percentile for presence-flagged {name} must be NaN")
        return None, None
    if not math.isfinite(threshold):
        raise ValueError(f"screen archive threshold for {name} must be finite")
    if source == "fixed":
        if not math.isnan(percentile):
            raise ValueError(f"screen archive percentile for fixed-threshold {name} must be NaN")
        return threshold, None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"screen archive percentile for {name} must be between 0 and 100")
    return threshold, percentile


def _screen_index_flags(
    members: Mapping[str, np.ndarray],
    scores: dict[str, np.ndarray],
    n_respondents: int,
) -> tuple[
    dict[str, BoolArray],
    IndexThresholdMap,
    IndexThresholdSourceMap,
    IndexPercentileMap,
]:
    """Recompute each index's flags from its score and recorded cutoff."""
    names = list(scores)
    thresholds = _screen_vector(
        members, "thresholds", kinds="f", length=len(names), aligned_with="index_names"
    )
    sources = _string_values(members["threshold_sources"], "threshold_sources")
    if len(sources) != len(names):
        raise ValueError("screen archive threshold_sources must match index_names")
    percentiles = _screen_vector(
        members, "percentiles", kinds="f", length=len(names), aligned_with="index_names"
    )

    flags: dict[str, BoolArray] = {}
    applied_thresholds: IndexThresholdMap = {}
    threshold_sources: IndexThresholdSourceMap = {}
    applied_percentiles: IndexPercentileMap = {}
    for position, name in enumerate(names):
        spec = INDEX_REGISTRY[name]
        source = cast(
            "IndexThresholdSource",
            _literal_choice(
                sources[position],
                IndexThresholdSource,
                f"screen archive threshold source for {name} must be "
                "'fixed', 'percentile', or 'presence'",
            ),
        )
        if (source == "presence") != (spec.flag_mode == "present"):
            expected_source = (
                "'presence'" if spec.flag_mode == "present" else "'fixed' or 'percentile'"
            )
            raise ValueError(
                f"screen archive threshold source for {name} must be {expected_source}"
            )
        cutoff, tail_percentile = _screen_cutoff(
            name, source, float(thresholds[position]), float(percentiles[position])
        )

        # Mirror _build_screen_result: fixed cutoffs include ties, percentile
        # cutoffs exclude them, and presence flags mark every available score.
        expected = (
            ~np.isnan(scores[name])
            if cutoff is None
            else threshold_flags(
                scores[name],
                threshold=cutoff,
                percentile=50.0,
                direction=spec.flag_direction,
                inclusive=source == "fixed",
            )
        )
        stored = _screen_vector(
            members,
            f"flag__{name}",
            kinds="b",
            length=n_respondents,
            aligned_with="n_respondents",
        )
        if not np.array_equal(stored, expected):
            raise ValueError(
                f"screen archive flags for {name} are inconsistent with its scores "
                f"and {source} rule"
            )
        flags[name] = cast("BoolArray", expected)
        applied_thresholds[name] = cutoff
        threshold_sources[name] = source
        applied_percentiles[name] = tail_percentile
    return flags, applied_thresholds, threshold_sources, applied_percentiles


def _validate_screen_summary(
    members: Mapping[str, np.ndarray],
    summary: dict[str, ScreenIndexSummary],
) -> None:
    """Require recorded summaries to agree with those rebuilt from the restored scores.

    Counts, flag rates, and extrema must match exactly, NaN included. Means and
    standard deviations depend on summation order, so the rebuilt values replace
    the recorded ones instead of being compared with them.
    """
    names = list(summary)
    columns = _string_values(members["summary_columns"], "summary_columns")
    if tuple(columns) != _SCREEN_SUMMARY_COLUMNS:
        raise ValueError("screen archive summary_columns must be mean, std, min, and max")
    statistics = members["summary_statistics"]
    if statistics.shape != (len(names), len(_SCREEN_SUMMARY_COLUMNS)) or (
        statistics.dtype.kind != "f"
    ):
        raise ValueError(
            "screen archive summary_statistics must be a floating-point matrix "
            "with one row per index and one column per summary statistic"
        )
    stored_flagged, stored_valid, stored_unavailable = (
        _screen_vector(members, member, kinds="iu", length=len(names), aligned_with="index_names")
        for member in ("summary_n_flagged", "summary_n_valid", "summary_n_unavailable")
    )
    flag_rates = _screen_vector(
        members, "summary_flag_rate", kinds="f", length=len(names), aligned_with="index_names"
    )

    for position, name in enumerate(names):
        expected = summary[name]
        if (
            stored_flagged[position] != expected["n_flagged"]
            or stored_valid[position] != expected["n_valid"]
            or stored_unavailable[position] != expected["n_unavailable"]
        ):
            raise ValueError(
                f"screen archive summary counts for {name} are inconsistent with its "
                "scores and flags"
            )
        if not _same_float(float(flag_rates[position]), expected["flag_rate"]):
            raise ValueError(
                f"screen archive summary flag rate for {name} is inconsistent with its counts"
            )
        _, _, minimum, maximum = (float(value) for value in statistics[position])
        if not (_same_float(minimum, expected["min"]) and _same_float(maximum, expected["max"])):
            raise ValueError(
                f"screen archive summary min and max for {name} are inconsistent with its scores"
            )


def _screen_result_from_members(
    members: Mapping[str, np.ndarray],
    scores: dict[str, np.ndarray],
    errors: dict[str, str],
    n_respondents: int,
) -> ScreenResult:
    """Rebuild a screen result after recomputing every recorded decision."""
    # An all-failed archive has no score vector to confirm n_respondents, so check
    # the stored lengths before allocating anything sized by that header value.
    stored = [
        _screen_vector(
            members, name, kinds=kinds, length=n_respondents, aligned_with="n_respondents"
        )
        for name, kinds, _ in _SCREEN_RESPONDENT_VECTORS
    ]
    names = list(scores)
    min_flags, min_valid_indices = _screen_settings(members, len(names), len(names) + len(errors))
    flags, thresholds, threshold_sources, percentiles = _screen_index_flags(
        members, scores, n_respondents
    )

    # Rebuild counts and summaries exactly as screen() does, then apply the
    # consensus rule from _build_screen_result.
    flag_counts, valid_index_counts, summary = _reduce_screen_results(scores, flags, n_respondents)
    consensus_eligible = (
        np.ones(n_respondents, dtype=bool)
        if min_valid_indices is None
        else valid_index_counts >= min_valid_indices
    )
    consensus_flags = (flag_counts >= min_flags) & consensus_eligible

    recomputed = (flag_counts, valid_index_counts, consensus_eligible, consensus_flags)
    for (name, _, message), value, expected in zip(
        _SCREEN_RESPONDENT_VECTORS, stored, recomputed, strict=True
    ):
        if not np.array_equal(value, expected):
            raise ValueError(f"screen archive {name} {message}")
    _validate_screen_summary(members, summary)

    return {
        "scores": scores,
        "flags": flags,
        "thresholds": thresholds,
        "threshold_sources": threshold_sources,
        "percentiles": percentiles,
        "flag_counts": cast("IntArray", flag_counts),
        "valid_index_counts": cast("IntArray", valid_index_counts),
        "consensus_eligible": cast("BoolArray", consensus_eligible),
        "consensus_flags": consensus_flags,
        "min_flags": min_flags,
        "min_valid_indices": min_valid_indices,
        "n_indices": len(names),
        "indices_used": names,
        "errors": errors,
        "n_respondents": n_respondents,
        "summary": summary,
    }


def _read_screen_archive(archive: NpzFile) -> ScreenArchive:
    """Validate one open NPZ archive and rebuild its complete screen result."""
    schema_version, _, n_respondents = _read_archive_header(
        archive, label="screen", result_types=Literal["screen"]
    )
    if "threshold_sources" not in archive.files:
        raise ValueError(
            "screen archive does not contain screening decisions; "
            "use load_score_archive() for score-only archives"
        )

    scores = _load_score_members(archive, "screen", n_respondents)
    errors = _load_errors(archive, set(scores), "screen")
    if not scores and not errors:
        raise ValueError("screen archive does not contain index scores or failures")
    flag_members = [f"flag__{name}" for name in scores]
    allowed_members = {
        "schema_version",
        "result_type",
        "n_respondents",
        "index_names",
        *_SCREEN_DECISION_MEMBERS,
        *_SCREEN_OPTIONAL_MEMBERS,
        *(f"score__{name}" for name in scores),
        *flag_members,
    }
    unexpected = set(archive.files) - allowed_members
    if unexpected:
        raise ValueError(f"screen archive contains unexpected member: {min(unexpected)}")

    members = {
        name: _require_member(archive, name) for name in (*_SCREEN_DECISION_MEMBERS, *flag_members)
    }
    if "min_valid_indices" in archive.files:
        members["min_valid_indices"] = _require_member(archive, "min_valid_indices")
    result = _screen_result_from_members(members, scores, errors, n_respondents)
    return {
        "schema_version": schema_version,
        "result": result,
        "respondent_ids": _load_respondent_ids(archive, n_respondents),
    }


def _validate_response_time_values(
    scores: ArrayLike,
    flags: ArrayLike,
    metric: object,
    direction: object,
    threshold: object,
    *,
    expected_respondents: int | None = None,
) -> tuple[
    np.ndarray,
    BoolArray,
    ResponseTimeMetric,
    ResponseTimeFlagDirection,
    float,
]:
    """Validate one complete reusable response-time result."""
    validated_metric = cast(
        "ResponseTimeMetric",
        _literal_choice(
            metric, ResponseTimeMetric, "response-time archive contains an unsupported metric"
        ),
    )
    validated_direction = cast(
        "ResponseTimeFlagDirection",
        _literal_choice(
            direction,
            ResponseTimeFlagDirection,
            "response-time archive flag_direction must be 'high' or 'low'",
        ),
    )
    # Fast-component probabilities are suspicious when high; direct summaries,
    # consistency, and response time effort are suspicious when low.
    expected_direction = "high" if validated_metric == "mixture" else "low"
    if validated_direction != expected_direction:
        raise ValueError(
            f"response-time metric {validated_metric!r} requires "
            f"{expected_direction!r} flag_direction"
        )

    validated_threshold = _finite_numeric_scalar(
        threshold,
        name="response-time archive threshold",
    )
    validated_scores = validate_score_array(scores, name="response-time archive scores")
    if expected_respondents is not None and len(validated_scores) != expected_respondents:
        raise ValueError("response-time archive scores must match n_respondents")
    if validated_metric == "effort":
        # Response time effort is a proportion of answered items, like the
        # cutoffs that response_time_effort_flag() and the CLI accept, so a value
        # outside [0, 1] is an item time or a corrupted score. NaN stays missing.
        if not 0.0 <= validated_threshold <= 1.0:
            raise ValueError(
                "response-time archive effort threshold must be an RTE proportion between 0 and 1"
            )
        if ((validated_scores < 0.0) | (validated_scores > 1.0)).any():
            raise ValueError(
                "response-time archive effort scores must be RTE proportions between 0 and 1"
            )
    validated_flags = _validate_boolean_vector(
        flags,
        len(validated_scores),
        name="response-time archive flags",
    )

    expected_flags = np.empty_like(validated_flags)
    inclusive_compare = np.greater_equal if validated_direction == "high" else np.less_equal
    exclusive_compare = np.greater if validated_direction == "high" else np.less
    inclusive_compare(validated_scores, validated_threshold, out=expected_flags)
    if not np.array_equal(validated_flags, expected_flags):
        exclusive_compare(validated_scores, validated_threshold, out=expected_flags)
        if not np.array_equal(validated_flags, expected_flags):
            raise ValueError(
                "response-time archive flags are inconsistent with its threshold and direction"
            )

    # Return the recomputed flags, like the screen loader: a stored Boolean byte
    # other than 0 or 1 compares as True but would otherwise survive in the array.
    return (
        validated_scores,
        expected_flags,
        validated_metric,
        validated_direction,
        validated_threshold,
    )


def _read_response_time_archive(archive: NpzFile) -> ResponseTimeArchive:
    """Validate one open NPZ archive and extract its response-time payload."""
    schema_version, _, n_respondents = _read_archive_header(
        archive,
        label="response-time",
        result_types=Literal["response_time"],
        allowed_members=frozenset(
            {
                "schema_version",
                "result_type",
                "n_respondents",
                "metric",
                "flag_direction",
                "threshold",
                "scores",
                "flags",
                "respondent_ids",
            }
        ),
    )

    scores, flags, metric, direction, threshold = _validate_response_time_values(
        _require_member(archive, "scores"),
        _require_member(archive, "flags"),
        _string_scalar(archive, "metric"),
        _string_scalar(archive, "flag_direction"),
        _numeric_scalar(archive, "threshold"),
        expected_respondents=n_respondents,
    )

    respondent_ids = _load_respondent_ids(archive, n_respondents)
    return {
        "schema_version": schema_version,
        "result_type": "response_time",
        "n_respondents": n_respondents,
        "metric": metric,
        "flag_direction": direction,
        "threshold": threshold,
        "scores": scores,
        "flags": flags,
        "respondent_ids": respondent_ids,
    }


def save_score_archive(
    path: str | Path,
    scores: Mapping[str, ArrayLike],
    *,
    result_type: ScoreArchiveResultType = "screen",
    respondent_ids: Sequence[str] | None = None,
    errors: Mapping[str, str] | None = None,
    n_respondents: int | None = None,
    compressed: bool = False,
    compression_level: int | None = None,
) -> None:
    """
    Save reusable registered-index scores as a versioned, pickle-free NPZ archive.

    Score and metadata validation completes before the destination is opened.
    Compatible float64 arrays are streamed without an intermediate score matrix,
    and mapping insertion order is preserved. Screen archives may retain only
    failed indices when ``n_respondents`` is supplied explicitly. Composite
    archives require component scores supported by ``composite_scores()``.

    Parameters:
    - path: Explicit destination ending in ``.npz``.
    - scores: Ordered mapping of registered index names to aligned score vectors.
    - result_type: ``"screen"`` or ``"composite"``.
    - respondent_ids: Optional aligned, unique, nonblank string identifiers,
      with no trailing NUL character.
    - errors: Optional ordered mapping of failed index names to nonblank messages,
      with no trailing NUL character.
    - n_respondents: Optional positive respondent count, checked against score
      vectors. Required for a screen archive containing only failed indices.
    - compressed: Use streaming DEFLATE compression to reduce storage, at the
      cost of extra CPU when writing and loading. Defaults to ``False``.
    - compression_level: Optional DEFLATE level from 1 (fastest) to 9 (smallest)
      for ``compressed=True``. ``None`` uses level 1; any level loads the same way.

    Example:
        >>> from ier import load_score_archive, save_score_archive, screen_scores
        >>> scores = {"irv": [0.1, 0.7, 0.4], "longstring": [3.0, 8.0, 5.0]}
        >>> save_score_archive("scores.npz", scores, respondent_ids=["a", "b", "c"])
        >>> saved = load_score_archive("scores.npz")
        >>> print(saved["respondent_ids"])
        ['a', 'b', 'c']
        >>> updated = screen_scores(saved["scores"], percentile=95, min_flags=1)
        >>> updated["consensus_flags"].tolist()
        [True, True, False]
    """
    destination = _require_npz_suffix(path, label="score")

    validated_result_type = cast(
        "ScoreArchiveResultType",
        _literal_choice(
            result_type,
            ScoreArchiveResultType,
            "score archive result_type must be 'screen' or 'composite'",
        ),
    )
    if not isinstance(scores, Mapping):
        raise TypeError("scores must be a mapping of registered index names to score arrays")
    score_items = list(scores.items())
    score_names = [name for name, _ in score_items]
    _validate_archive_index_names(
        score_names,
        validated_result_type,
        allow_empty=validated_result_type == "screen",
    )
    if not score_items and n_respondents is None:
        raise ValueError(
            "score archive does not contain reusable index scores; n_respondents is required"
        )
    validated_scores, respondent_count = validate_score_vectors(
        dict(score_items), n_respondents=n_respondents
    )

    validated_errors = _validate_error_input(errors, set(score_names), validated_result_type)
    if not validated_scores and not validated_errors:
        raise ValueError("score archive must contain reusable index scores or failures")

    id_member = _respondent_ids_member(respondent_ids, respondent_count)

    payload = _archive_header(validated_result_type, respondent_count)
    payload.update(
        {
            "index_names": np.asarray(score_names, dtype=np.str_),
            "error_names": np.asarray(list(validated_errors), dtype=np.str_),
            "error_messages": np.asarray(list(validated_errors.values()), dtype=np.str_),
        }
    )
    for name, values in validated_scores.items():
        payload[f"score__{name}"] = values
    payload.update(id_member)
    _write_npz_atomically(
        destination, payload, compressed=compressed, compression_level=compression_level
    )


def save_screen_archive(
    path: str | Path,
    result: ScreenResult,
    *,
    respondent_ids: Sequence[str] | None = None,
    compressed: bool = False,
    compression_level: int | None = None,
) -> None:
    """
    Save a complete ``screen()`` result as a versioned, pickle-free NPZ archive.

    The archive uses the same schema as ``ier screen --format npz``: raw scores,
    per-index flags, recorded thresholds and their sources, percentile settings,
    respondent counts, consensus decisions, summaries, and soft failures. Before
    the destination is opened, every flag, count, consensus decision, summary
    flag rate, and score minimum and maximum is recomputed from the scores and
    recorded cutoffs, so the archive always reloads through
    ``load_screen_archive()``. Summary means and standard deviations are
    written as recomputed from the scores. Integer settings must fit in 64 bits.
    The write is atomic.

    Reload the archive to keep the original decisions exactly. Rebuilding them
    with ``screen_scores(thresholds=...)`` instead treats every recorded cutoff
    as an inclusive fixed threshold, which flips ties at percentile cutoffs.

    Parameters:
    - path: Explicit destination ending in ``.npz``.
    - result: A ``ScreenResult`` from ``screen()`` or ``screen_scores()``.
    - respondent_ids: Optional aligned, unique, nonblank string identifiers,
      with no trailing NUL character.
    - compressed: Use streaming DEFLATE compression to reduce storage, at the
      cost of extra CPU when writing and loading. Defaults to ``False``.
    - compression_level: Optional DEFLATE level from 1 (fastest) to 9 (smallest)
      for ``compressed=True``. ``None`` uses level 1; any level loads the same way.

    Example:
        >>> from ier import load_screen_archive, save_screen_archive, screen_scores
        >>> result = screen_scores(
        ...     {"irv": [0.1, 0.7, 0.4, 0.9], "longstring": [3.0, 8.0, 5.0, 8.0]},
        ...     percentile=75,
        ...     min_flags=1,
        ... )
        >>> save_screen_archive("screening.npz", result, respondent_ids=["a", "b", "c", "d"])
        >>> restored = load_screen_archive("screening.npz")["result"]
        >>> flags = restored["flags"]
        >>> flags["longstring"].tolist()
        [False, False, False, False]
        >>> restored["consensus_flags"].tolist()
        [True, False, False, False]
    """
    destination = _require_npz_suffix(path, label="screen")
    if not isinstance(result, Mapping):
        raise TypeError("result must be a ScreenResult mapping")

    names = list(result["indices_used"])
    _validate_archive_index_names(names, "screen", allow_empty=True)
    if not isinstance(result["scores"], Mapping) or list(result["scores"]) != names:
        raise ValueError("screen result scores must follow indices_used")
    scores, n_respondents = validate_score_vectors(
        result["scores"], n_respondents=result["n_respondents"]
    )
    errors = _validate_error_input(result["errors"], set(names), "screen")
    if not scores and not errors:
        raise ValueError("screen archive must contain index scores or failures")

    payload = _archive_header("screen", n_respondents)
    payload.update(
        {
            "n_indices": _int64_setting(result["n_indices"], "n_indices"),
            "min_flags": _int64_setting(result["min_flags"], "min_flags"),
            "index_names": np.asarray(names, dtype=np.str_),
            "thresholds": np.asarray(
                [
                    np.nan if result["thresholds"][name] is None else result["thresholds"][name]
                    for name in names
                ],
                dtype=np.float64,
            ),
            "threshold_sources": np.asarray(
                [result["threshold_sources"][name] for name in names],
                dtype=np.str_,
            ),
            "percentiles": np.asarray(
                [
                    np.nan if result["percentiles"][name] is None else result["percentiles"][name]
                    for name in names
                ],
                dtype=np.float64,
            ),
            "flag_counts": np.asarray(result["flag_counts"], dtype=np.int64),
            "valid_index_counts": np.asarray(result["valid_index_counts"], dtype=np.int64),
            "consensus_eligible": np.asarray(result["consensus_eligible"], dtype=np.bool_),
            "consensus_flags": np.asarray(result["consensus_flags"], dtype=np.bool_),
            "summary_columns": np.asarray(_SCREEN_SUMMARY_COLUMNS, dtype=np.str_),
            "summary_statistics": _summary_statistics(result["summary"], names),
            "summary_n_flagged": np.asarray(
                [result["summary"][name]["n_flagged"] for name in names],
                dtype=np.int64,
            ),
            "summary_n_valid": np.asarray(
                [result["summary"][name]["n_valid"] for name in names],
                dtype=np.int64,
            ),
            "summary_n_unavailable": np.asarray(
                [result["summary"][name]["n_unavailable"] for name in names],
                dtype=np.int64,
            ),
            "summary_flag_rate": np.asarray(
                [result["summary"][name]["flag_rate"] for name in names],
                dtype=np.float64,
            ),
        }
    )
    if result["min_valid_indices"] is not None:
        payload["min_valid_indices"] = _int64_setting(
            result["min_valid_indices"], "min_valid_indices"
        )
    payload["error_names"] = np.asarray(list(errors), dtype=np.str_)
    payload["error_messages"] = np.asarray(list(errors.values()), dtype=np.str_)
    for name in names:
        payload[f"score__{name}"] = scores[name]
        payload[f"flag__{name}"] = np.asarray(result["flags"][name], dtype=np.bool_)
    # Validate the encoded members exactly as load_screen_archive() will read them,
    # then store the means and standard deviations that the loader rebuilds.
    restored = _screen_result_from_members(payload, scores, errors, n_respondents)
    payload["summary_statistics"] = _summary_statistics(restored["summary"], names)
    payload.update(_respondent_ids_member(respondent_ids, n_respondents))
    _write_npz_atomically(
        destination, payload, compressed=compressed, compression_level=compression_level
    )


def save_response_time_archive(
    path: str | Path,
    scores: ArrayLike,
    flags: ArrayLike,
    *,
    threshold: float,
    metric: ResponseTimeMetric = "median",
    flag_direction: ResponseTimeFlagDirection = "low",
    respondent_ids: Sequence[str] | None = None,
    compressed: bool = False,
    compression_level: int | None = None,
) -> None:
    """
    Save reusable response-time results as a versioned, pickle-free NPZ archive.

    Scores, Boolean flags, metric/direction compatibility, the finite threshold,
    and optional respondent identifiers are validated before the destination is
    opened. Flags may follow either the inclusive fixed-cutoff rule or the
    tie-exclusive rule of percentile cutoffs and strict response time effort
    cutoffs. An ``"effort"`` threshold and every available effort score must lie
    between 0 and 1, the range of response time effort.

    Parameters:
    - path: Explicit destination ending in ``.npz``.
    - scores: Per-respondent direct timing scores, response time effort
      proportions, or mixture probabilities.
    - flags: Aligned Boolean decisions produced from the recorded threshold.
    - threshold: Resolved finite cutoff in the score's units.
    - metric: Timing metric represented by the score vector.
    - flag_direction: Suspicious tail, ``"low"`` or ``"high"``.
    - respondent_ids: Optional aligned, unique, nonblank string identifiers,
      with no trailing NUL character.
    - compressed: Use streaming DEFLATE compression to reduce storage, at the
      cost of extra CPU when writing and loading. Defaults to ``False``.
    - compression_level: Optional DEFLATE level from 1 (fastest) to 9 (smallest)
      for ``compressed=True``. ``None`` uses level 1; any level loads the same way.

    Example:
        >>> from ier import response_time_score_flags, save_response_time_archive
        >>> scores = [0.5, 1.2, 2.0]
        >>> flags = response_time_score_flags(scores, threshold=1.0)
        >>> save_response_time_archive(
        ...     "timing.npz", scores, flags, threshold=1.0, metric="median"
        ... )
        >>> flags.tolist()
        [True, False, False]
    """
    destination = _require_npz_suffix(path, label="response-time")

    validated_scores, validated_flags, validated_metric, validated_direction, cutoff = (
        _validate_response_time_values(
            scores,
            flags,
            metric,
            flag_direction,
            threshold,
        )
    )
    id_member = _respondent_ids_member(respondent_ids, len(validated_scores))

    payload = _archive_header("response_time", len(validated_scores))
    payload.update(
        {
            "metric": np.asarray(validated_metric, dtype=np.str_),
            "flag_direction": np.asarray(validated_direction, dtype=np.str_),
            "threshold": np.asarray(cutoff, dtype=np.float64),
            "scores": validated_scores,
            "flags": validated_flags,
        }
    )
    payload.update(id_member)
    _write_npz_atomically(
        destination, payload, compressed=compressed, compression_level=compression_level
    )


def load_score_archive(path: str | Path) -> ScoreArchive:
    """
    Load reusable registered-index scores from a versioned NPZ archive.

    The loader always disables pickling and validates schema version, result type,
    member names, registry membership, vector shape, respondent alignment,
    optional identifiers, and soft-failure metadata. Screen archives retaining
    only failed indices preserve their explicit respondent count; pass it to
    ``screen_scores()`` as ``n_respondents`` when reusing an empty score mapping.
    Full composite CLI archives must have been written with
    ``--include-components`` so their raw public index scores are present;
    compact archives from ``save_score_archive()`` are directly compatible.
    Complete screen archives load here as scores only; use
    ``load_screen_archive()`` to restore their recorded decisions.

    Parameters:
    - path: Path to a screen or detailed composite NPZ archive.

    Returns:
    - A ``ScoreArchive`` containing ordered raw score vectors, result metadata,
      optional respondent IDs, and any recorded per-index soft failures.

    Example:
        >>> from ier import composite_scores, load_score_archive, save_score_archive
        >>> from ier import screen_scores
        >>> save_score_archive(
        ...     "scores.npz",
        ...     {"irv": [0.1, 0.7, 0.4], "longstring": [3.0, 8.0, 5.0]},
        ...     errors={"mad": "item pairs were not configured"},
        ... )
        >>> saved = load_score_archive("scores.npz")
        >>> list(saved["scores"]), saved["errors"]
        (['irv', 'longstring'], {'mad': 'item pairs were not configured'})
        >>> updated_screen = screen_scores(
        ...     saved["scores"], percentile=99, errors=saved["errors"]
        ... )
        >>> updated_screen["errors"]
        {'mad': 'item pairs were not configured'}
        >>> save_score_archive(
        ...     "components.npz", saved["scores"], result_type="composite"
        ... )
        >>> saved_components = load_score_archive("components.npz")
        >>> updated_composite = composite_scores(
        ...     saved_components["scores"], weights={"irv": 2.0}
        ... )
        >>> updated_composite.shape
        (3,)
    """
    with _open_npz_archive(path, label="score") as archive:
        return _read_score_archive(archive)


def load_screen_archive(path: str | Path) -> ScreenArchive:
    """
    Load a complete ``screen()`` result from a versioned, pickle-free NPZ archive.

    Accepts archives from ``save_screen_archive()`` and ``ier screen --format npz``.
    Pickling is always disabled. Beyond the schema, member shapes, registry
    names, identifiers, and soft failures, the loader recomputes every per-index
    flag from its score and recorded cutoff: fixed thresholds include ties,
    percentile cutoffs exclude them, and presence rules flag available scores.
    Flag counts, valid-index counts, consensus decisions, and summary counts,
    flag rates, minima, and maxima must also agree exactly. Summary means and
    standard deviations are rebuilt from the restored scores, so the restored
    summary always describes the restored scores. Flags are returned as
    canonical Booleans. The restored result can be passed directly to the
    ``plot_*`` helpers without recomputing any index.

    Parameters:
    - path: Path to a complete screen NPZ archive.

    Returns:
    - A ``ScreenArchive`` with the schema version, the restored ``ScreenResult``,
      and optional respondent identifiers.

    Example:
        >>> from ier import load_screen_archive, save_screen_archive, screen_scores
        >>> original = screen_scores(
        ...     {"irv": [0.1, 0.7, 0.4], "onset": [float("nan"), 2.0, float("nan")]},
        ...     thresholds={"irv": 0.4},
        ...     min_flags=1,
        ... )
        >>> save_screen_archive("screening.npz", original, respondent_ids=["a", "b", "c"])
        >>> saved = load_screen_archive("screening.npz")
        >>> print(saved["respondent_ids"])
        ['a', 'b', 'c']
        >>> restored = saved["result"]
        >>> restored["threshold_sources"]
        {'irv': 'fixed', 'onset': 'presence'}
        >>> restored["consensus_flags"].tolist()
        [True, True, True]
    """
    with _open_npz_archive(path, label="screen") as archive:
        return _read_screen_archive(archive)


def load_response_time_archive(path: str | Path) -> ResponseTimeArchive:
    """
    Load response-time results from a versioned, pickle-free NPZ archive.

    The loader validates every schema field, the metric and suspicious-tail
    pairing, the 0 to 1 range of ``"effort"`` thresholds and scores, vector
    shape and respondent alignment, optional identifiers, and agreement
    between the stored flags and threshold. Flags are returned as
    canonical Booleans recomputed from that rule. The returned score vector
    can be passed directly to ``response_time_score_flags()`` to apply a new
    fixed or percentile cutoff without recomputing the timing metric. Fixed
    ``"effort"`` cutoffs from the CLI flag response time effort strictly below
    the threshold, like ``response_time_effort_flag()``; compare saved effort
    scores with ``<`` to apply that rule to a new cutoff.

    Parameters:
    - path: Path to a response-time NPZ archive written by the CLI.

    Returns:
    - A ``ResponseTimeArchive`` containing scores, flags, cutoff metadata, and
      optional respondent identifiers.

    Example:
        >>> from ier import load_response_time_archive, response_time_score_flags
        >>> from ier import save_response_time_archive
        >>> scores = [0.5, 1.2, 2.0, 3.1]
        >>> flags = response_time_score_flags(scores, threshold=1.0)
        >>> save_response_time_archive("timing.npz", scores, flags, threshold=1.0)
        >>> saved = load_response_time_archive("timing.npz")
        >>> saved["flags"].tolist()
        [True, False, False, False]
        >>> strict = response_time_score_flags(
        ...     saved["scores"], threshold=1.5,
        ...     direction=saved["flag_direction"],
        ... )
        >>> strict.tolist()
        [True, True, False, False]
    """
    with _open_npz_archive(path, label="response-time") as archive:
        return _read_response_time_archive(archive)
