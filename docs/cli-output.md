# CLI Output Formats

The three scoring commands support human-readable summaries, interoperable text
formats, and lossless NumPy archives:

| Format | Destination | Best for |
|--------|-------------|----------|
| `text` | file, `.gz`, or standard output | Interactive review |
| `json` | file, `.gz`, or standard output | Structured metadata and web tooling |
| `csv` | file, `.gz`, or standard output | Row-oriented statistics workflows |
| `npz` | explicit `.npz` file | Fast, typed Python and NumPy workflows |

CSV rows and JSON respondent arrays are written forward-only with bounded output
allocation. JSON converts non-finite numbers to `null`, and CSV emits empty
cells. NPZ preserves NumPy dtypes and non-finite values exactly without object
arrays or pickling.

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
ier composite responses.csv --format npz --output composite.npz
ier response-time timings.csv --format npz --output timing.npz
```

NPZ output requires `--output` with a `.npz` suffix. It cannot target standard
output or an additional `.gz` layer; the NPZ container is already a ZIP archive.
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
)
saved = load_score_archive("screening.npz")
updated = screen_scores(saved["scores"], percentile=99)
print(saved["respondent_ids"])
print(saved["errors"])
```

`save_score_archive()` validates the destination, result type, registered index
names, aligned vectors, optional IDs, and soft failures before opening the file.
It streams compatible arrays without constructing a respondent-by-index matrix.
`load_score_archive()` always disables pickling and validates the complete
schema. It accepts compact public archives, screen CLI archives, and composite
CLI archives written with `--include-components`. Aggregate-only composite and
response-time archives do not contain reusable registered-index vectors and are
rejected with a contextual error.

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

Load and reflag the retained scores through the public validated boundary:

```python
from ier import load_response_time_archive, response_time_score_flags

saved = load_response_time_archive("timing.npz")
revised = response_time_score_flags(
    saved["scores"],
    cutoff_percentile=1,
    direction=saved["flag_direction"],
)
```

`load_response_time_archive()` disables pickling and verifies schema version,
metric and direction compatibility, finite cutoff metadata, aligned score and
flag vectors, optional respondent identifiers, and agreement between stored
flags and their cutoff rule. It accepts both inclusive fixed-threshold flags and
tie-exclusive percentile flags.

Consumers should reject unsupported future `schema_version` values rather than
assuming their layout is unchanged.
