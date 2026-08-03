# CLI Output Formats

Scoring and archive-reflagging commands support human-readable summaries,
interoperable text formats, and lossless NumPy archives:

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
ier screen-reflag screening.npz --percentile 99 --format npz --output stricter.npz
ier composite responses.csv --format npz --output composite.npz
ier composite-recombine composite.npz --weight irv=2 \
  --include-components --format npz --output reweighted.npz
ier response-time timings.csv --format npz --output timing.npz
```

NPZ output requires `--output` with a `.npz` suffix. It cannot target standard
output or an additional `.gz` layer; the NPZ container is already a ZIP archive.
Every writer completes the archive in a temporary directory beside the target
before an atomic replacement. Handled write failures leave existing content
intact and clean up partial output; successful replacement retains existing
permission bits.
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

For validated type auto-detection and compact metadata inspection, use either
the generic Python loader or the matching command:

```python
from ier import load_archive

saved = load_archive("results.npz")
print(saved["result_type"])
```

```bash
ier archive-info results.npz
ier archive-info results.npz --format json --output archive-metadata.json
ier archive-info timing-model.npz --format json
```

The text and strict-JSON summaries omit respondent vectors. Score archives list
stored indices and soft failures; response-time archives report the metric,
tail direction, cutoff provenance, and aggregate flag rate. Timing-model archives
report component count, transform, fastest-component position, and validated
weights, means, and variances.

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
It streams compatible arrays without constructing a respondent-by-index matrix;
the shared atomic boundary protects the destination from later I/O failures.
`load_score_archive()` always disables pickling and validates the complete
schema. It accepts compact public archives, screen CLI archives, and composite
CLI archives written with `--include-components`. Aggregate-only composite and
response-time archives do not contain reusable registered-index vectors and are
rejected with a contextual error.

### Psychometric pair model archives

Reusable synonym and antonym calibrations have a dedicated model schema:

| Member | Type |
| --- | --- |
| `schema_version` | Integer scalar (`1`) |
| `result_type` | Unicode scalar (`psychsyn_model`) |
| `n_items` | Integer scalar |
| `critval` | Finite numeric scalar |
| `anto` | Boolean scalar |
| `item_pairs` | Two-column integer array |

`save_psychsyn_model()` snapshots and validates the immutable model before
atomically replacing its destination. `load_psychsyn_model()` rejects missing,
extra, incorrectly typed, duplicate, self-referencing, or out-of-range pairs and
returns an independently owned read-only array. Empty two-column pair arrays are
valid. Generic `load_archive()` retains the validated pairs, while
`archive-info` reports only the mode, threshold, item count, and pair count so
metadata output remains bounded for wide calibrations.

```bash
ier psychsyn-fit reference.csv psychsyn-model.npz --critval 0.6
ier psychsyn-score later.csv psychsyn-model.npz \
  --format npz --output psychsyn-scores.npz
```

The scoring result is an ordinary one-index screen archive and can therefore be
loaded with `load_score_archive()` or passed to `screen-reflag` without the
response matrix.

### Response-time mixture model archives

Persist a fitted reference-cohort calibration separately from respondent result
archives:

```python
from ier import (
    fit_response_time_mixture,
    load_response_time_mixture_model,
    response_time_mixture_scores,
    save_response_time_mixture_model,
)

model = fit_response_time_mixture(reference_times, n_components=3, random_seed=42)
save_response_time_mixture_model("timing-model.npz", model)
loaded = load_response_time_mixture_model("timing-model.npz")
later_probabilities = response_time_mixture_scores(later_times, loaded)
```

The schema stores only a version, model result type, component count, weights,
means, variances, and the log-transform choice. The writer validates a fresh
snapshot before atomically replacing the destination; the loader disables
pickling, rejects missing or extra members, checks strict scalar/vector dtypes,
and reconstructs independent read-only parameter arrays. Model archives are not
respondent results, so use the dedicated loader when scoring. The generic loader
and `archive-info` expose the same validated parameters for discovery and
inspection without constructing a respondent result.

The CLI can write and apply this model schema directly:

```bash
ier response-time-fit reference-times.csv timing-model.npz --components 3 --random-seed 42
ier response-time later-times.csv --mixture-model timing-model.npz \
  --format npz --output scored.npz
```

`response-time-fit` writes only the reusable calibration. The later scoring
command writes an ordinary response-time result archive, including scores,
high-tail flags, identifiers, and cutoff provenance; that result can be passed
to `response-time-reflag` without the original timing matrix.

Reuse validated registered-index scores directly from the CLI:

```bash
ier screen-reflag screening.npz --percentile 99 --min-flags 3 --format json
ier screen-reflag screening.npz --indices irv longstring \
  --threshold longstring=8 --index-percentile irv=90 \
  --format npz --output screening.npz
```

`screen-reflag` accepts compact public score archives, full screen archives,
and detailed composite archives. It optionally selects stored index names in
the requested order, reapplies the same fixed or percentile per-index rules,
coverage requirement, and consensus count as `screen`, and preserves respondent
identifiers and archived soft failures. The shared serializers produce the same
screen text, JSON, CSV, and NPZ contracts. An NPZ destination may be the input
path because validation and loading finish before atomic replacement starts.

Recombine retained composite components through the same validated boundary:

```bash
ier composite-recombine composite.npz --weight irv=2 --method mean --format json
ier composite-recombine composite.npz --indices irv longstring --method max \
  --no-standardize --min-valid-indices 2 --include-components \
  --format npz --output composite.npz
```

`composite-recombine` accepts compact composite archives, compatible screen
archives, and detailed composite output. It optionally selects unique stored
components in order, then reuses the same direction correction, mean/sum/max
reductions, weights, standardization, completeness, fixed or percentile flags,
uncalibrated probability transform, and serializers as `composite`. Identifiers
and archived soft failures are preserved. In-place NPZ replacement requires
`--include-components`, preventing a reusable source from being silently
replaced by aggregate-only output.

### Common fields

Screen and composite archives use schema version `1`. Response-time archives use
version `2` when cutoff provenance is recorded and remain readable at legacy
version `1`. Every archive includes:

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
respondent-aligned `scores`, and boolean `flags`. Current CLI output writes schema
version `2`, adding `threshold_source` (`fixed` or `percentile`) and a scalar
`percentile`. Fixed cutoffs store `NaN` for `percentile`; derived cutoffs retain
the requested value, including the default low-tail `5` or high-tail `95`.

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
    threshold_source="fixed",
)
```

`load_response_time_archive()` disables pickling and verifies schema version,
metric and direction compatibility, finite cutoff metadata, aligned score and
flag vectors, optional respondent identifiers, and agreement between stored
flags and their cutoff rule. Legacy v1 archives accept either inclusive fixed
flags or tie-exclusive percentile flags because their source was not recorded.
For v2, the loader enforces the rule named by `threshold_source` and recomputes a
percentile-derived cutoff from the stored scores and requested percentile.

Reuse those validated scores directly from the CLI:

```bash
ier response-time-reflag timing.npz --percentile 1 --format json --output strict.json
ier response-time-reflag timing.npz --threshold 1.0 --format npz --output timing.npz
```

`response-time-reflag` requires exactly one new fixed or percentile cutoff. It
preserves the archived metric, suspicious-tail direction, score vector, and
optional respondent identifiers, then writes through the same text, JSON, CSV,
or NPZ serializers as `response-time`. An NPZ destination may be the input path;
the validated arrays are loaded before the atomic replacement begins.

`save_response_time_archive()` writes legacy v1 when both provenance arguments
are omitted, preserving existing calls. Pass `threshold_source="fixed"` or pass
`threshold_source="percentile", percentile=VALUE` to write v2; supplying only
`percentile` infers the latter. The writer checks
all scores, Boolean flags, cutoff metadata, direction rules, and optional
identifiers before opening the destination. It streams the validated vectors
without stacking or adding a runtime dependency.

Consumers should reject unsupported future `schema_version` values rather than
assuming their layout is unchanged.
