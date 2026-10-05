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

### pandas nullable dtypes

pandas extension dtypes (`Int64`, `Float64`, `boolean`, and the result of
`DataFrame.convert_dtypes()`) and `pd.NA` are accepted and converted to float64,
with missing values becoming `NaN`. The same conversion applies to object
columns and to nested lists that use `None` for missing responses. Boolean,
integer, and floating NumPy arrays are used without a copy, so pass an integer
array when integers above 2**53 must stay exact.

```python
import pandas as pd
from ier import screen

df = pd.DataFrame([[1, 2, 3, 4], [4, None, 2, 1], [3, 3, 3, 3]]).convert_dtypes()
result = screen(df, indices=["irv", "longstring"])
assert result["errors"] == {}
```

Complex, datetime, timedelta, and non-numeric text matrices raise `ValueError`
with one message; `pd.NaT` counts as datetime data and is rejected rather than
treated as missing. Scalars, strings, mappings, sets, and generators raise
`TypeError` (which also derives from `ValueError`).

### NumPy masked arrays

Masked cells of a `numpy.ma.MaskedArray`, such as sentinel codes masked with
`np.ma.masked_equal`, are missing responses. An array with at least one masked
cell is copied with those cells set to `NaN` (Boolean and integer data become
float64); a masked array without masked cells is used without a copy.

```python
import numpy as np
from ier import missing_rate

coded = np.array([[1, 2, 3, 4], [3, -99, 3, -99], [5, 4, 3, 2]])
masked = np.ma.masked_equal(coded, -99)
assert missing_rate(masked).tolist() == [0.0, 0.5, 0.0]
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
ier screen responses.csv --indices acquiescence --scale-min 1 --scale-max 5 \
  --acquiescence-positive-items 0,2 --acquiescence-negative-items 1,3
ier screen responses.csv --indices infrequency \
  --infrequency-item-indices 3,7 --infrequency-acceptable-ranges '1:2,5:'
```

`ier screen --help` lists every option with its default, grouped into input,
index, screening decision, and output settings. Index option defaults match
`IndexOptions()`. Save a combination of options in a
[configuration file](#configuration-files) to reuse it.

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

Exports from survey platforms such as Qualtrics or LimeSurvey often place
metadata such as `StartDate`, `Duration`, and `IPAddress` beside dozens of items.
Select the items by a shell-style header pattern, or name only the metadata to
leave out:

```bash
ier screen export.csv --id-column ResponseId --item-pattern 'Q*'
ier screen export.csv --id-column ResponseId \
  --exclude-column StartDate --exclude-column Duration
```

`--item-pattern` is case-sensitive and repeatable; matching columns keep their
header order and are selected once even when several patterns match. Every
pattern must match at least one column, and patterns cannot be combined with
`--item-columns`. `--exclude-column` removes exact header names after any item
selection; used alone, it scores every column except the ID column and the
excluded names. Quote patterns so the shell does not expand them.

Named item selection requires a header. Item-index options such as
`--mad-positive-items` use zero-based positions in the selected order, not the
original file's column positions.

Some survey exports prepend report titles or metadata. Use `--skip-rows N` to
discard exactly `N` physical input lines before delimiter and header detection:

```bash
ier screen survey-export.csv --skip-rows 2 --id-column participant_id
```

Blank preamble lines count toward `N`. The option works for plain, compressed,
and forward-only standard-input streams; it does not apply to `.npy` matrices.

Delimited input detects a header automatically. For ambiguous files, make the
contract explicit: `--header present` always treats the first non-empty row as a
header, including when every column name looks numeric, while `--header absent`
requires the first row to contain data. Named ID or item columns require `auto` or
`present` mode. When a header is detected or declared, every non-empty data row
must contain the same number of columns; mismatches fail before scoring rather
than silently redefining the matrix width.

Check what was detected before scoring with `ier inspect`. It accepts the same
input options as the scoring commands and reports the delimiter and how it was
chosen, the header decision, the identifier and selected item columns, the
matrix shape, missing cells overall and per item, and the observed values:

```bash
ier inspect export.csv --id-column ResponseId --item-pattern 'Q*' --missing-value NA
```

Without `--scale-min` and `--scale-max`, scale-aware indices such as midpoint
responding and acquiescence infer the response scale from the observed minimum
and maximum. `inspect` prints those bounds as suggested `--scale-min=VALUE` and
`--scale-max=VALUE` options that can be pasted into a scoring command, and warns
when an observed extreme is used by fewer than 1% of respondents, which often
signals an unused scale category or a data-entry error. See
[CLI output formats](cli-output.md#input-inspection) for the JSON fields.

Blank cells are always loaded as missing values. Survey exports that use explicit
markers can map each exact, whitespace-trimmed token to `NaN` without preprocessing:

```bash
ier screen responses.csv --missing-value NA --missing-value -99
```

The option may be repeated and also works with compressed input and standard
input. It applies only to scored numeric cells, so identifier and unselected
metadata values remain unchanged.

A single leading UTF-8 byte-order mark is removed automatically before delimiter
and header detection. This applies equally to plain files, compressed input, and
standard input; a marker elsewhere in the matrix remains invalid data.

For large headerless numeric matrices, save an uncompressed NumPy array and pass
it directly. The CLI memory-maps `.npy` input read-only instead of copying it:

```bash
ier screen responses.npy --indices irv longstring --format json
```

Binary input must contain one non-empty, two-dimensional, real numeric array.
Header, missing-value, and delimiter options and compressed `.npy` input are not
supported.

Timing matrices have a dedicated command so their units cannot be mixed with
item-response indices:

```bash
ier response-time timings.csv --metric median --threshold 1.0
ier response-time timings.csv --metric mixture --random-seed 42 --format json
```

Retain a timing score vector when comparing decision cutoffs:

```python
from ier import response_time, response_time_score_flags

median_times = response_time(timings, metric="median")
strict_flags = response_time_score_flags(median_times, cutoff_percentile=1)
```

Use `direction="high"` when reflagging fast-component mixture probabilities.

Scoring commands also accept forward-only standard input and gzip-, bzip2-, or
XZ-compressed files without extra packages:

```bash
cat responses.csv | ier screen - --indices irv longstring --format json
ier screen responses.csv.gz --format json --output screening.json.gz
ier screen responses.csv.xz --format csv --output screening.csv.xz
```

CSV rows and JSON respondent arrays are forward-only for plain files, compressed
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

`save_score_archive()` validates every vector and metadata field before opening
the staged archive and atomically replaces the destination only after every
member is complete. Detailed composite NPZ output produced with
`--include-components` works with the same loader and `composite_scores()`.
Response-time results have matching `save_response_time_archive()` and
`load_response_time_archive()` boundaries; retained direct, consistency, and
mixture `scores` feed directly into `response_time_score_flags()` and can be
written back with revised flags. Effort cutoffs are strict instead, with a 0.90
default, while `response_time_score_flags()` includes ties at a fixed cutoff and
defaults to a percentile; reflag saved effort scores with `scores < cutoff`, or
recompute them with `response_time_effort_flag(times, threshold=...)`.

## Configuration files

A realistic screen combines many options. Save them in a TOML file to version and
share the analysis, then pass it with `--config` to any scoring or saved-score
command (`screen`, `composite`, `screen-scores`, `composite-scores`,
`response-time`, and `response-time-scores`):

```toml
# ier.toml
[screen]
indices = ["irv", "longstring", "acquiescence", "infrequency"]
id_column = "participant_id"
item_pattern = ["Q*"]
missing_value = ["NA", "-99"]
scale_min = 1
scale_max = 5
acquiescence_positive_items = [0, 2]
acquiescence_negative_items = [1, 3]
infrequency_item_indices = [3, 7]
infrequency_expected_responses = [5, 1]
threshold = { irv = 0.25, longstring = 8 }
min_flags = 2

[composite]
indices = ["irv", "longstring", "person_total"]
id_column = "participant_id"
item_pattern = ["Q*"]
missing_value = ["NA", "-99"]
weight = { irv = 2 }
percentile = 95
```

```bash
ier screen responses.csv --config ier.toml --format csv --output screening.csv
ier composite responses.csv --config ier.toml --format json
```

Keys are long option names without the leading dashes, written with dashes or
underscores (`scale-min` or `scale_min`); `ier COMMAND --help` lists them. Each
option may be set once, so a file that spells one option two ways, such as
`scale-min` and `scale_min` or `missing_value` and `missing_values`, is rejected
with both keys named. A file with `[screen]`, `[composite]`, or other command
tables applies only the running command's table, which must exist, and keeps
every option inside a table. A file without command tables applies all of its
keys to whichever command reads it.

Write single-value options as strings or numbers, and switches as `true` or
`false`; `na_rm = false` is the same as `--no-na-rm`. Use arrays for `indices`
and repeatable options such as `missing_value`, `exclude_column`, `item_column`,
and `item_pattern`, and tables for the `INDEX=VALUE` options `threshold`,
`index_percentile`, and `weight`. Arrays for comma-separated options such as
`acquiescence_positive_items` are joined with commas. Every value passes through
the same validation as the command line. Type and choice errors, mutually
exclusive options, and malformed item lists, pairs, ranges, and `INDEX=VALUE`
tables are reported before any data is read, with errors naming the option,
section, and file; unknown options are rejected. Checks that depend on other
options, the indices, or the data, such as `compress` without `format = "npz"`
or unknown index names, run with the command itself.

Options given on the command line take precedence. A repeatable option there
replaces the configured list instead of extending it, and an option replaces any
configured option it cannot be combined with, such as `--percentile` for a
configured composite `threshold` or `--item-columns` for a configured
`item_pattern`. A switch enabled in the file, such as `strict = true`, cannot be
turned off from the command line. The data path is always given on the command
line, and relative paths such as `output` resolve against the working directory.

## Result tables and plots

Screening results are NumPy arrays, so a DataFrame index does not survive
`screen()`. `screen_table()` arranges a result as respondent-aligned columns that
pandas and polars accept directly; reattach the index when building the frame:

```python
import pandas as pd
from ier import screen, screen_table

df = pd.read_csv("responses.csv", index_col="participant_id")
result = screen(df)
table = pd.DataFrame(screen_table(result), index=df.index)
print(table[table["consensus_flag"]])
```

Columns follow the CLI CSV order: `flag_count`, `valid_index_count`,
`consensus_eligible`, `consensus_flag`, then `{name}_score` and `{name}_flag` per
index. Arrays are shared with the result rather than copied. Polars frames have
no row index, so pass identifiers as a leading string column instead:

```python
import polars as pl

frame = pl.DataFrame(screen_table(result, respondent_ids=df.index.astype(str).tolist()))
```

`composite_table(details)` arranges a `composite_summary()` result the same way.

Before relying on a consensus rule, check whether the indices agree.
`index_agreement()` returns index names and a co-flag (`"overlap"` or
`"jaccard"`) or pairwise-complete Spearman score-rank matrix, and
`plot_index_agreement()` draws it. Spearman scores are oriented by each index's
flag direction, so a positive correlation means both indices rank the same
respondents as more suspicious:

```python
from ier import index_agreement, plot_index_agreement

names, jaccard = index_agreement(result)
names, rho = index_agreement(result, kind="spearman")
fig = plot_index_agreement(result, kind="spearman")
```

## Next steps

- Run multi-index screening with [`screen()`](workflows/screening.md)
- Combine signals with [`composite()`](workflows/composite.md)
- Browse the [index catalog](indices.md)
- Choose a machine-readable [CLI output format](cli-output.md)
