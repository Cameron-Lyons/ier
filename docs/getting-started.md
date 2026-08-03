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
```

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
non-finite values. See [CLI output formats](cli-output.md) for its schema.

Reload registered-index vectors safely for later sensitivity work:

```python
from ier import load_score_archive, screen_scores

saved = load_score_archive("screening.npz")
revised = screen_scores(saved["scores"], percentile=99)
```

Detailed composite NPZ output produced with `--include-components` works with
the same loader and `composite_scores()`.

## Next steps

- Run multi-index screening with [`screen()`](workflows/screening.md)
- Combine signals with [`composite()`](workflows/composite.md)
- Browse the [index catalog](indices.md)
- Choose a machine-readable [CLI output format](cli-output.md)
