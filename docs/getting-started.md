# Getting Started

## Installation

### From PyPI

```bash
pip install insufficient-effort
```

### With optional dependencies

```bash
pip install "insufficient-effort[plot]"
```

| Extra | Provides |
|-------|----------|
| *(none)* | All statistical indices, including chi-square and response-time mixture helpers |
| `full` | Empty compatibility alias retained for existing installation commands |
| `plot` | matplotlib helpers (`plot_distributions`, etc.) |

### From source (development)

```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
uv sync --all-groups
```

## Input shapes

Functions expect a matrix where **rows are respondents** and **columns are items**.
Accepted inputs:

- nested lists / tuples
- NumPy arrays
- objects with `__array__` (e.g. pandas DataFrames)

Polars DataFrames usually work via `__array__`, but converting with
`.to_numpy()` is the most explicit path:

```python
import polars as pl
from ier import irv

df = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
scores = irv(df.to_numpy())
```

```python
import pandas as pd
from ier import irv

df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
scores = irv(df)
```

## Missing data

Most scorers accept `na_rm=True` (often the default) to skip incomplete rows or
pairwise comparisons rather than failing on NaNs.

```python
import numpy as np
from ier import irv, mahad

data = np.array([
    [1, 2],
    [2, 3],
    [np.nan, 4],
    [3, 4],
], dtype=float)

irv(data, na_rm=True)
mahad(data, na_rm=True, method="iqr")
```

## Quick screening

```python
from ier import IndexOptions, screen

result = screen(responses, options=IndexOptions(scale_min=1, scale_max=5))
print(result["flag_counts"])
print(result["consensus_flags"])
```

Or from the CLI:

```bash
ier screen responses.csv --scale-min 1 --scale-max 5 --min-flags 2
ier screen responses.csv --indices irv longstring missing_rate --min-valid-indices 2
ier screen responses.csv --index-percentile irv=90 --index-percentile longstring=99
```

For files with a respondent identifier column, preserve it in every output format
by naming its header:

```bash
ier screen responses.csv --id-column participant_id --format csv --output screening.csv
```

Identifier values must be unique and nonblank. The selected column is excluded
from the numeric item matrix before scoring.

Survey exports may also contain demographics, conditions, or other non-item
metadata. Select only named numeric item columns, in scoring order, with a
comma-separated or repeated option:

```bash
ier screen responses.csv \
  --id-column participant_id \
  --item-columns q1,q2,q3 \
  --item-columns q4,q5,q6
```

Named item selection requires a header. Item-index options such as
`--mad-positive-items` use zero-based positions in the selected order, not the
original file's column positions.

For large headerless numeric matrices, save an uncompressed NumPy array and pass
it directly. The CLI memory-maps `.npy` input read-only instead of copying it:

```bash
ier screen responses.npy --indices irv longstring --format json
```

Binary input must contain one non-empty, two-dimensional, real numeric array.
Header options, `--delimiter`, and `.npy.gz` input are not supported.

Timing matrices have a dedicated command so their units cannot be mixed with
item-response indices:

```bash
ier response-time timings.csv --metric median --threshold 1.0
ier response-time timings.csv --metric mixture --random-seed 42 --format json
ier response-time-fit reference-times.csv timing-model.npz --components 3 --random-seed 42
ier response-time later-times.csv --mixture-model timing-model.npz --threshold 0.5
```

Retain a timing score vector when comparing decision cutoffs:

```python
from ier import response_time, response_time_score_flags

median_times = response_time(timings, metric="median")
strict_flags = response_time_score_flags(median_times, cutoff_percentile=1)
```

Use `direction="high"` when reflagging fast-component mixture probabilities.

Scoring commands also accept forward-only standard input and gzip-compressed
files without extra packages:

```bash
cat responses.csv | ier screen - --indices irv longstring --format json
ier screen responses.csv.gz --format json --output screening.json.gz
```

CSV rows and JSON respondent arrays are forward-only for plain files, gzip
files, and standard output, so large respondent-level exports do not retain the
complete document in memory.

Independent indices can be scored concurrently for larger matrices:

```python
result = screen(responses, workers=4)
scores = composite(responses, workers=4)
```

```bash
ier screen responses.npy --workers 4 --format json --output screening.json
```

The default `workers=1` path remains sequential. Parallel scoring retains the
requested index and failure order but may use more temporary memory, so benchmark
representative data before choosing a worker count.

IRV can also be averaged across questionnaire sections to expose localized
changes in response variability:

```python
section_result = screen(
    responses,
    indices=["irv"],
    options=IndexOptions(irv_num_split=4),
)
custom_result = screen(
    responses,
    indices=["irv"],
    options=IndexOptions(irv_split_points=[0, 10, 25, 40]),
)
```

```bash
ier screen responses.csv --indices irv --irv-num-split 4
ier screen responses.csv --indices irv --irv-split-points 0,10,25,40
```

Either option enables section scoring. Boundaries must start at zero and end at
the item count; if both forms are supplied, custom boundaries take precedence.
The section mean is accumulated with respondent-sized temporary vectors, so its
memory use does not grow with the number of sections.

Missing-response psychometric retries can be reproduced across runs with scoped
seeds:

```python
seeded = screen(
    responses,
    indices=["psychsyn", "psychant"],
    options=IndexOptions(
        psychsyn_random_seed=17,
        psychant_random_seed=29,
    ),
)
```

```bash
ier screen responses.csv --indices psychsyn --psychsyn-random-seed 17
ier screen responses.csv --indices psychant --psychant-random-seed 29
```

Seeds affect only retry draws for undefined within-person correlations. Each
scorer uses an isolated random stream and leaves NumPy's global state untouched.

Balanced acquiescence scoring is available without preprocessing item polarity:

```bash
ier screen responses.csv --indices acquiescence --scale-min 1 --scale-max 5 \
  --acquiescence-positive-items 0,2,4 \
  --acquiescence-negative-items 1,3,5
```

Both lists are required together. Their zero-based positions refer to the
scored item matrix after any `--item-columns` selection, and entries are paired
in list order.

Command-line missing-rate scoring supports respondent-specific skip logic with
a separate applicability matrix:

```bash
ier screen responses.csv --indices missing_rate \
  --missing-applicable-mask applicable.npy
```

The mask must match the scored response matrix after item-column selection.
True or `1` cells identify expected responses; false or `0` cells are excluded
from both the missing count and denominator. Files may be delimited text,
gzip-compressed text, or safe pickle-free `.npy`; Boolean NumPy masks remain
read-only memory maps.

Final screening flag counts and composite reductions use respondent-sized
workspaces rather than another respondent-by-index matrix, keeping post-scoring
memory bounded as the number of selected indices grows.

Reuse the returned score vectors when comparing alternative decision rules:

```python
from ier import screen_scores

strict = screen_scores(
    result["scores"],
    percentile=99,
    min_flags=3,
    min_valid_indices=3,
)
```

This returns a fresh screening result without recalculating any index.

Detailed composite results support the same reuse pattern for alternative
weights, reductions, or completeness rules:

```python
from ier import composite_scores, composite_summary

details = composite_summary(responses, indices=["irv", "longstring", "person_total"])
weighted = composite_scores(
    details["indices"],
    weights={"irv": 2.0, "person_total": 0.5},
    min_valid_indices=2,
)
```

For fast, lossless NumPy workflows, write a versioned, pickle-free archive:

```bash
ier screen responses.npy --indices irv longstring --format npz --output screening.npz
```

NPZ output requires a `.npz` file path and preserves typed flags, metadata, and
non-finite values. Writes use same-directory atomic replacement, so existing
results survive an interrupted serialization. See
[CLI output formats](cli-output.md) for the schema.

Save a compact reusable-score archive directly from Python, or reload compatible
CLI output for later sensitivity work:

```python
from ier import load_score_archive, save_score_archive, screen_scores

save_score_archive("raw-scores.npz", result["scores"], errors=result["errors"])
saved = load_score_archive("screening.npz")
revised = screen_scores(saved["scores"], percentile=99)
```

If the result type is not known beforehand, use the generic validated loader or
inspect its metadata from the command line:

```python
from ier import load_archive

saved = load_archive("results.npz")
print(saved["result_type"], saved["n_respondents"])
```

```bash
ier archive-info results.npz
ier archive-info results.npz --format json --output archive-metadata.json
```

Both paths auto-detect screen, composite, and response-time results and run the
same complete pickle-disabled validators as the specialized loaders.

`save_score_archive()` validates every vector and metadata field before opening
the staged archive and atomically replaces the destination only after every
member is complete. Detailed composite NPZ output produced with
`--include-components` works with the same loader and `composite_scores()`.

Reapply screening rules from the command line without loading the original
response matrix or rerunning its indices:

```bash
ier screen-reflag screening.npz --percentile 99 --min-flags 3 \
  --format json --output stricter.json
ier screen-reflag screening.npz --indices irv longstring \
  --threshold longstring=8 --index-percentile irv=90 \
  --format npz --output revised.npz
```

The optional `--indices` list selects and orders stored vectors; omitting it
uses all of them. The command validates the archive, preserves respondent IDs
and stored soft failures, and supports text, JSON, CSV, and NPZ. An NPZ output
may safely replace the input archive through atomic replacement.

Recombine retained composite components with new weights, reductions, or
decision rules without rescoring the response matrix:

```bash
ier composite-recombine composite.npz --weight irv=2 --weight longstring=0.5 \
  --min-valid-indices 2 --percentile 95 --format json
ier composite-recombine composite.npz --indices irv longstring --method max \
  --no-standardize --include-components --include-probability \
  --format npz --output revised.npz
```

The source may be a compact composite score archive, compatible screen output,
or composite NPZ written with `--include-components`. Stored vectors retain
their public directions and requested order; identifiers and diagnostics are
carried forward. In-place NPZ output requires `--include-components` to avoid
discarding the reusable source vectors.

Response-time results have matching `save_response_time_archive()` and
`load_response_time_archive()` boundaries; retained `scores` feed directly into
`response_time_score_flags()` and can be written back with revised flags. Current
CLI text, JSON, and NPZ output also records whether the cutoff was fixed or
percentile-derived and retains the requested percentile. The archive loader
supports both legacy v1 files and provenance-aware v2 files.

Reference-cohort mixture calibration has a separate compact model archive:

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
probabilities = response_time_mixture_scores(later_times, loaded)
```

This archive contains no respondent data. It is versioned, pickle-free, strictly
validated on load, and atomically replaced on save. Loaded parameter vectors are
independent and read-only.

Create and reuse the same calibration from the command line:

```bash
ier response-time-fit reference-times.csv timing-model.npz --components 3 --random-seed 42
ier response-time later-times.csv --mixture-model timing-model.npz --format npz \
  --output later-timing.npz
```

The saved model implies the mixture metric, retains the calibrated transform and
high-tail flag direction, and works with the normal text, JSON, CSV, and NPZ
response-time outputs. Fitting-only options are rejected when `--mixture-model`
is present.

Reapply a cutoff from the command line without loading the original timing
matrix or recalculating its metric:

```bash
ier response-time-reflag timing.npz --percentile 1 --format json --output strict.json
```

The command requires exactly one of `--threshold` or `--percentile`, validates
the archive before reuse, and preserves its scores, metric direction, and
respondent identifiers in every output format.

## Next steps

- Run multi-index screening with [`screen()`](workflows/screening.md)
- Combine signals with [`composite()`](workflows/composite.md)
- Browse the [index catalog](indices.md)
- Choose a machine-readable [CLI output format](cli-output.md)
