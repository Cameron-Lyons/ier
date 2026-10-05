# CLI Output Formats

The scoring and saved-score commands support human-readable summaries, interoperable text
formats, and lossless NumPy archives:

| Format | Destination | Best for |
|--------|-------------|----------|
| `text` | file, compressed file, or standard output | Interactive review |
| `json` | file, compressed file, or standard output | Structured metadata and web tooling |
| `csv` | file, compressed file, or standard output | Row-oriented statistics workflows |
| `npz` | explicit `.npz` file | Fast, typed Python and NumPy workflows |

CSV rows and JSON respondent arrays are written forward-only with bounded output
allocation. JSON converts non-finite numbers to `null`, and CSV emits empty
cells. NPZ preserves NumPy dtypes and non-finite values exactly without object
arrays or pickling. For text formats, `.gz`, `.bz2`, and `.xz` suffixes select
gzip, bzip2, and XZ compression from the Python standard library. XZ output uses
its fast low-memory preset to limit compression workspace.

Text previews honor `--top N` (default 10). Small previews select respondents
in bounded batches instead of sorting every result. Screening ranks higher
flag counts first, composite output ranks higher scores first, and response-time
output follows its suspicious tail. Equal ranking values retain original input
row order, including ties at the preview boundary. Composite and timing previews
omit non-finite scores. Use `--top 0` for summary metadata without respondent rows.

Regular output files in every format are completed in a temporary directory
beside their destination, then replaced atomically. Serialization failures,
compression-finalization failures, and handled interruptions leave previous
results intact and clean up staged output. Successful writes retain existing
permission bits and update symbolic-link targets without replacing the links.
For compressed output through a symbolic link, the requested path's suffix
selects the compressor. Standard output, named pipes, and device paths stream
directly and can contain partial output after a failure.

Screen and composite commands report soft per-index failures on standard error
without mixing warnings into standard output. Their JSON results retain an
`errors` object, while NPZ archives use aligned `error_names` and
`error_messages` vectors. Text repeats the failures inline. CSV stays
row-oriented and carries no global metadata, so retain standard error when its
diagnostics matter.

## NumPy archives

Write archives from any scoring command:

```bash
ier screen responses.npy --indices irv longstring --format npz --output screening.npz
ier screen responses.npy --indices irv longstring --format npz --compress --output compact.npz
ier composite responses.csv --format npz --output composite.npz
ier response-time timings.csv --format npz --output timing.npz
```

NPZ output requires `--output` with a `.npz` suffix. It cannot target standard
output or an additional compression layer; the NPZ container is already a ZIP
archive.
Members are uncompressed by default for speed. Add `--compress` to any fresh or
saved-score command to use streaming DEFLATE compression inside the NPZ
container; this option requires `--format npz`. Compression preserves every
array and metadata field and requires no schema change or special loading
option. Repeated scores, flags, and IDs can become much smaller, while
continuous scores often compress modestly and take more CPU to write and load.
`--compress` uses DEFLATE level 1 by default; add `--compress-level 1-9` to
trade write time for size (9 is smallest). Any level loads the same way.
Archives use the same atomic file replacement as the text formats.
Load every archive with pickling disabled:

```python
import numpy as np

with np.load("screening.npz", allow_pickle=False) as result:
    print(result["schema_version"].item())
    print(result["result_type"].item())
    print(result["index_names"])
    irv_scores = result["score__irv"]
    irv_flags = result["flag__irv"]
```

For validated score reuse, prefer the public save/load pair. The writer creates
a compact score-only archive; full CLI archives retain the additional flags and
decision metadata documented below.

```python
from ier import load_score_archive, save_score_archive, screen_scores

save_score_archive(
    "raw-scores.npz",
    {"irv": [0.1, 0.7], "longstring": [3.0, 8.0]},
    respondent_ids=["case-1", "case-2"],
    compressed=True,
)
saved = load_score_archive("raw-scores.npz")
updated = screen_scores(
    saved["scores"], errors=saved["errors"], n_respondents=saved["n_respondents"], percentile=99
)
print(saved["respondent_ids"])
print(saved["errors"])
```

`save_score_archive()` validates the destination, result type, registered index
names, aligned vectors, optional IDs, and soft failures before opening the file.
It streams compatible arrays without constructing a respondent-by-index matrix;
the shared atomic boundary protects the destination from later I/O failures.
Every public archive writer accepts the Boolean `compressed` option, which
defaults to `False`, and an optional `compression_level` from 1 to 9 for
compressed output. Archive respondent IDs and error
messages cannot end with a NUL character because fixed-width NumPy Unicode
would silently truncate that character; embedded NULs are preserved. Invalid
Unicode codepoints in external archive metadata produce a contextual error.
`load_score_archive()` always disables pickling and validates the complete
schema. It accepts compact public archives, screen CLI archives, and composite
CLI archives written with `--include-components`. Aggregate-only composite and
response-time archives do not contain reusable registered-index vectors and are
rejected with a contextual error.

Screen archives can also retain a run where every selected index failed. The
stored respondent count, identifiers, and nonblank failure messages remain
available even though `scores` is empty. Supply `n_respondents` explicitly to
`save_score_archive()` and `screen_scores()` for this case. `screen-scores` uses
the archived count automatically and emits aligned zero-coverage results in every
format; `--strict` still rejects retained failures. Composite reuse requires
at least one available component vector.

Reuse those scores directly from the CLI for sensitivity analysis:

```bash
ier screen-scores screening.npz --index-percentile irv=99 --min-flags 2 --format json
ier composite-scores screening.npz --indices irv longstring --weight irv=2 \
  --include-components --percentile 95 --format npz --output reweighted.npz
```

These commands never read the original responses or run an index. They apply new
decisions to raw scores, preserve respondent IDs, and support every output
format above. Defaults use all saved scores; `--indices` selects and orders a
subset of available scores and excludes unselected failures. Selecting a failed
or absent index produces an error explaining why no scores are available.
Without a subset, historical index failures are retained in output and reported
on standard error; `--strict` rejects them before writing. Composite output
retains only failures for composite-enabled indices, although strict validation
still audits every failure before conversion. Saved screening-only scores must
be excluded from composites with `--indices` or skipped with `--skip-unsupported`,
which names the skipped indices on standard error.

Input matrix options, worker counts, and `best_subset` are unavailable for saved
scores. Select the desired component names explicitly. Composite archives need
`--include-components` when they are first written to remain reusable. Input and
output may use the same archive path; replacement occurs only after the new
result is complete.

### Common schema

All archives use schema version `1` and include:

| Key | Value |
|-----|-------|
| `schema_version` | Integer scalar |
| `result_type` | `screen`, `composite`, or `response_time` |
| `n_respondents` | Integer scalar |
| `respondent_ids` | Optional Unicode vector when `--id-column` is used |

Row positions are respondent identifiers when `respondent_ids` is absent.

### Screen schema

`screen` archives include `n_indices`, `min_flags`, `index_names`, `thresholds`,
`threshold_sources`, `percentiles`, `flag_counts`, `valid_index_counts`,
`consensus_eligible`, and `consensus_flags`.
When `--min-valid-indices` is supplied, the scalar `min_valid_indices` records
the completeness requirement. Numeric thresholds and percentile settings align
with `index_names`; `NaN` represents a fixed/presence rule without a percentile,
or a presence-based index without a numeric threshold. `threshold_sources`
distinguishes those cases. JSON exposes equivalent mappings and uses `null` for
unset values. CSV includes `valid_index_count` and `consensus_eligible` columns
for every respondent but intentionally omits global cutoff metadata.

Each successful index has a `score__NAME` float vector and `flag__NAME` boolean
vector. Summary values use `summary_columns`, `summary_statistics`,
`summary_n_flagged`, `summary_n_valid`, `summary_n_unavailable`, and
`summary_flag_rate`. The rate uses valid scores as its denominator and is `NaN`
when an index has no available scores. JSON exposes the same values inside each
index summary and uses `null` for an unavailable rate. Soft failures are stored
in aligned `error_names` and `error_messages` Unicode vectors.

`save_screen_archive()` writes this schema from Python, and
`load_screen_archive()` restores the complete `ScreenResult` from either writer.
The loader recomputes each `flag__NAME` vector from its score, threshold, and
source, then checks the respondent counts, consensus decisions, and summary
counts, so the restored result can be plotted or reported without rescoring.

### Composite schema

`composite` archives include the `method` string scalar, `standardized` boolean
scalar, and respondent-aligned `scores` float vector. When `--weight` is
supplied, aligned `weight_names` and `weights` vectors record the explicit
overrides; selected indices not listed there use weight 1. When
`--min-valid-indices` is supplied, `min_valid_indices` records the integer
completeness requirement. JSON composite output uses equivalent `method`,
`standardized`, and optional `weights` and `min_valid_indices` fields. Text also
records the effective standardization setting. Aligned `error_names` and
`error_messages` vectors preserve soft failures; JSON uses the `errors` object.

When `--threshold` or `--percentile` is supplied, NPZ and JSON add `threshold`,
`threshold_source`, and respondent-aligned boolean `flags`. Percentile output
also records the requested `percentile`. CSV adds `composite_flag`, while text
reports the cutoff, source, flagged count, and row-level flag. These fields are
absent when flagging is not requested.

With `--include-probability`, NPZ and JSON add respondent-aligned
`probabilities` plus `probability_scale="uncalibrated_logistic"`. CSV adds
`composite_probability`, and text adds the value to each ranked row. The vector
is transformed from the already-computed aggregate scores without rescoring
indices. It is absent by default. Flag thresholds remain in the original
composite-score units even when probability output is included.

With `--include-components`, composite NPZ adds `index_names`,
`valid_index_counts`, and one `score__NAME` float vector per successful index.
JSON adds `indices_used`, `valid_index_counts`, and a `component_scores` object;
CSV adds `valid_index_count` and `NAME_score` columns; text shows the same fields
for ranked respondents. These are raw public index scores before direction
correction, standardization, and weighting. The aggregate `scores` vector is
unchanged.

### Response-time schema

`response-time` archives include `metric`, `flag_direction`, `threshold`,
respondent-aligned `scores`, and boolean `flags`.

| `metric` | `flag_direction` | Scores |
|----------|------------------|--------|
| `mean`, `median`, `sd`, `min` | `low` | Per-respondent timing summaries |
| `consistency` | `low` | Coefficients of variation |
| `mixture` | `high` | Fast-component probabilities |
| `effort` | `low` | Response time effort (RTE) proportions |

`effort` extends the allowed metric values without a schema change, so archives
from earlier releases load unchanged; earlier readers reject only the new
metric value. Fixed effort cutoffs, including the default 0.90, flag RTE
strictly below the threshold and verify under the tie-exclusive rule. An effort
archive's threshold and every available score must lie between 0 and 1, the
range of RTE, so an item time saved as an effort cutoff is rejected.

Load and reflag the retained scores through the public validated boundary:

```python
from ier import (
    load_response_time_archive,
    response_time_score_flags,
    save_response_time_archive,
)

saved = load_response_time_archive("timing.npz")
revised = response_time_score_flags(
    saved["scores"],
    threshold=1.0,
    direction=saved["flag_direction"],
)
save_response_time_archive(
    "revised-timing.npz",
    saved["scores"],
    revised,
    threshold=1.0,
    metric=saved["metric"],
    flag_direction=saved["flag_direction"],
    respondent_ids=saved["respondent_ids"],
)
```

`load_response_time_archive()` disables pickling and verifies schema version,
metric and direction compatibility, finite cutoff metadata, the RTE range of
effort cutoffs and scores, aligned score and flag vectors, optional respondent
identifiers, and agreement between stored flags and their cutoff rule. It accepts both inclusive fixed-threshold flags and
tie-exclusive percentile flags. `response_time_score_flags()` includes ties at a
fixed cutoff; to apply the strict effort rule to a new cutoff in Python, compare
the saved effort scores directly, as in `saved["scores"] < 0.8`, which never
flags unavailable scores.

`save_response_time_archive()` writes the same CLI-compatible schema and checks
all scores, Boolean flags, cutoff metadata, direction rules, and optional
identifiers before opening the destination. It streams the validated vectors
without stacking or adding a runtime dependency.

Reflag timing archives or convert them to another output format from the CLI:

```bash
ier response-time-scores timing.npz --threshold 1.0 --format csv --output revised.csv
ier response-time-scores timing.npz --percentile 1 --format npz --output stricter.npz
ier response-time-scores timing.npz --format json --output timing.json.gz
ier response-time-scores effort.npz --threshold 0.8 --format json
```

`response-time-scores` validates the archive before using its retained scores,
metric, suspicious-tail direction, and respondent IDs. It never reads the
original timing matrix or refits a mixture. Direct metrics, consistency, and
effort use the low tail; mixture probabilities use the high tail. Choose either
`--threshold` for an inclusive fixed cutoff or `--percentile` for a strict
sample-percentile cutoff that excludes ties. Effort is the exception to the
inclusive fixed rule: a fixed RTE cutoff must lie between 0 and 1 and flags
scores strictly below it, on both timing commands. These options are mutually
exclusive on both timing commands.

With neither cutoff option, the saved threshold and exact flags are preserved,
including decisions for scores tied at the cutoff. This allows conversion to
JSON, CSV, or text without changing the original decision. All output formats,
compressed text destinations, and `--top` previews are supported. The input and
NPZ output may share a path; atomic replacement protects the original archive
until the revised result is complete. Matrix selection and metric-fitting
options are unavailable for saved timing scores.

Consumers should reject unsupported future `schema_version` values rather than
assuming their layout is unchanged.

## Input inspection

`ier inspect DATA` loads a response matrix with the same input options as the
scoring commands and reports how it was parsed, without scoring any index:

```bash
ier inspect export.csv --id-column ResponseId --item-pattern 'Q*' --missing-value NA
ier inspect responses.csv --format json --output inspection.json
```

Text output is a short summary that lists at most ten item names and the ten
most-missing items. `--format json` writes one object with every item, and
`--output` writes either format with the same atomic replacement and `.gz`,
`.bz2`, or `.xz` compression as result files.

| Key | Value |
|-----|-------|
| `source` | Input path, or `standard input` |
| `input_format` | `delimited` or `npy` |
| `delimiter` | Field delimiter; `null` for whitespace-separated or `.npy` input |
| `delimiter_detection` | `explicit` (`--delimiter`), `sniffer` (comma, tab, or semicolon found by `csv.Sniffer`), `fallback` (chosen from the first records' fields), or `whitespace`; `null` for `.npy` input |
| `header` | `present` (declared with `--header present` or implied by named columns), `auto-detected` (non-numeric first cell), or `absent` |
| `id_column` | `--id-column` name, or `null` |
| `n_columns` | Input columns, including identifier and unselected columns |
| `n_respondents`, `n_items` | Shape of the matrix that scoring commands would use |
| `item_names` | Selected header names in scoring order, or `null` without a header |
| `item_positions` | 1-based input column of each selected item |
| `missing_cells`, `missing_by_item` | Missing cells overall and for each selected item |
| `infinite_cells` | Infinite cells, which are left out of the value summary |
| `observed_min`, `observed_max` | Smallest and largest finite responses, or `null` when there are none |
| `distinct_values` | Number of distinct finite responses |
| `non_integer_values` | Whether any finite response has a fractional part |
| `respondents_at_min`, `respondents_at_max` | Respondents with at least one response at each observed extreme |
| `suggested_options` | `--scale-min=VALUE` and `--scale-max=VALUE` arguments for the observed extremes, or an empty list |
| `warnings` | Human-readable cautions about the observed scale |

When `--scale-min` and `--scale-max` are omitted, scale-aware indices infer the
response scale from the observed extremes, so `suggested_options` reproduces
the bounds a scoring command would use. Pass the bounds explicitly when the
questionnaire's scale is wider. Each entry is one complete argument that
scoring commands accept as written; the `=` form keeps a negative value in
exponent notation, such as `-1e-05`, from being read as another option.
`warnings` notes an observed extreme used by
fewer than 1% of respondents, a single distinct value, no finite responses, or
infinite cells.

## Index catalog

`ier indices` lists the registry metadata that `index_catalog()` returns, one
entry per registered index in registry order:

```bash
ier indices
ier indices --format json --output indices.json
ier indices --format csv --output indices.csv
```

| Key | Value |
|-----|-------|
| `flag_direction` | `high` or `low`: the suspicious tail of the index |
| `flag_mode` | `percentile` for tail cutoffs, or `present` when any available score flags |
| `default_screen` | Whether `screen` scores the index when `--indices` is omitted |
| `default_composite` | Whether `composite` includes the index when `--indices` is omitted |
| `composite_enabled` | Whether composites accept the index |
| `required_options` | `IndexOptions` fields that must all be set before the index can run |
| `alternative_options` | Groups of interchangeable `IndexOptions` fields; at least one field in each group must be set |
| `uses_keyed_responses` | Whether the index reads responses with `--reverse-keyed-items` reverse-scored |

JSON writes `{"n_indices": N, "indices": {NAME: METADATA}}`, with
`required_options` as an array of field names and `alternative_options` as an
array of arrays, for example
`[["infrequency_expected_responses", "infrequency_acceptable_ranges"]]`. CSV
writes an `index` column followed by these keys, with `True`/`False` booleans,
comma-separated `required_options`, and `alternative_options` groups separated
by `;` with the fields inside a group separated by `|`; an empty cell means no
options. Text writes the same rows tab-separated under the shorter headings
`index`, `direction`, `flag_mode`, `screen_default`, `composite`,
`composite_default`, `required_options`, `alternative_options`, and
`keyed_responses`, with
`yes`/`no` booleans and `-` for no options. `--output` accepts the same `.gz`,
`.bz2`, and `.xz` suffixes and atomic replacement as result files.
