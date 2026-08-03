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

CSV output is forward-only for plain files, gzip files, and standard output, so
large respondent-level exports do not retain the complete document in memory.

For fast, lossless NumPy workflows, write a versioned, pickle-free archive:

```bash
ier screen responses.npy --indices irv longstring --format npz --output screening.npz
```

NPZ output requires a `.npz` file path and preserves typed flags, metadata, and
non-finite values. See [CLI output formats](cli-output.md) for its schema.

## Next steps

- Run multi-index screening with [`screen()`](workflows/screening.md)
- Combine signals with [`composite()`](workflows/composite.md)
- Browse the [index catalog](indices.md)
- Choose a machine-readable [CLI output format](cli-output.md)
