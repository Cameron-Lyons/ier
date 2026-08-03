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
`flag_counts`, and `consensus_flags`. Numeric thresholds align with
`index_names`; `NaN` represents a presence-based index without a numeric cutoff.

Each successful index has a `score__NAME` float vector and `flag__NAME` boolean
vector. Summary values use `summary_columns`, `summary_statistics`, and
`summary_n_flagged`. Soft failures are stored in aligned `error_names` and
`error_messages` Unicode vectors.

### Composite schema

`composite` archives include the `method` scalar and respondent-aligned `scores`
float vector. When `--weight` is supplied, aligned `weight_names` and `weights`
vectors record the explicit overrides; selected indices not listed there use
weight 1. When `--min-valid-indices` is supplied, `min_valid_indices` records the
integer completeness requirement. JSON composite output uses equivalent
optional `weights` and `min_valid_indices` fields. Aligned `error_names` and
`error_messages` vectors preserve soft failures; JSON uses the `errors` object.

### Response-time schema

`response-time` archives include `metric`, `flag_direction`, `threshold`,
respondent-aligned `scores`, and boolean `flags`.

Consumers should reject unsupported future `schema_version` values rather than
assuming their layout is unchanged.
