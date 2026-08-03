# IER

Python package for detecting **Insufficient Effort Responding (IER)** / careless
responding in survey data.

For a comprehensive methods review, see
[Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub).

## Features

- Multiple detection families: consistency, response patterns, response styles, outliers, omissions, response times, attention checks
- Workflow APIs: `screen()` and `composite()` configured via `IndexOptions`
- Configurable multi-index consensus decisions for respondent-level screening
- Fixed or sample-relative per-index screening thresholds
- Programmatic and CLI index catalog with defaults and configuration requirements
- CLI preservation of named respondent identifier columns
- CLI selection of named item columns from files containing metadata
- CLI workflows for item screening, composite scores, and response-time analysis
- NumPy-first inputs (lists, arrays, array-compatible DataFrames)
- Configurable soft or strict per-index failures during screening and composite scoring
- Full type annotations (`py.typed`)

## Installation

```bash
pip install insufficient-effort
```

Optional extras:

```bash
pip install "insufficient-effort[plot]"
```

The base install is NumPy-only. Chi-square flagging, Q-Q quantiles, IRT theta
estimation, and response-time mixture scoring are implemented locally and do
not require SciPy. The legacy `full` extra remains accepted as an empty
compatibility alias.

## Quick Start

```python
import numpy as np
from ier import IndexOptions, composite, irv, screen

data = np.array([
    [1, 2, 3, 4, 5, 4],
    [2, 3, 4, 3, 2, 1],
    [3, 3, 3, 3, 3, 3],
    [1, 5, 1, 5, 1, 5],
], dtype=float)

print("IRV:", irv(data))

result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
print("Indices:", result["indices_used"])
print("Flag counts:", result["flag_counts"])
print("Consensus flags:", result["consensus_flags"])

scores = composite(data, indices=["irv", "longstring", "person_total", "markov"])
print("Composite:", scores)
```

## CLI

```bash
ier screen data.csv --scale-min 1 --scale-max 5
ier screen data.csv --format json --output screen.json
ier screen data.csv --threshold irv=0.25 --threshold longstring=8
ier screen data.csv --indices irv mad --strict
ier screen data.csv --id-column participant_id --format csv --output screening.csv
ier screen data.csv --id-column participant_id --item-columns q1,q2,q3,q4
ier composite data.csv --indices irv longstring
ier composite data.csv --format csv --output scores.csv
ier response-time timings.csv --metric median --threshold 1.0
ier response-time timings.csv --metric mixture --random-seed 42 --format json
ier screen responses.csv.gz --format json --output screening.json.gz
ier screen responses.npy --indices irv longstring
cat responses.csv | ier screen - --indices irv longstring --format json
ier indices --format json
ier --version
```

Input matrices may be comma-, tab-, semicolon-, or whitespace-delimited. Common
delimiters are auto-detected unless `--delimiter` is supplied. Blank fields in
delimited files are loaded as missing values (`NaN`). Use `--id-column NAME` to
remove a named header column from scoring and preserve its unique, nonblank values
in text, JSON, and CSV output. Use `--item-columns q1,q2,...` to select and order
the numeric item matrix while ignoring unselected metadata columns; repeat the
option to build the selection in groups.

Uncompressed `.npy` files are memory-mapped read-only for fast, low-overhead
loading of large headerless real numeric matrices. Because binary arrays have no
column headers, `--id-column`, `--item-columns`, and `--delimiter` do not apply.
Use an uncompressed `.npy` file rather than `.npy.gz` to preserve memory mapping.

Use `-` as the input path for a forward-only standard-input pipeline or as the
output path for standard output. Files ending in `.gz` are read and written
transparently using the Python standard library. CSV results are written one row
at a time, so output allocation stays bounded for plain, compressed, and
standard-output destinations.

`ier response-time` accepts a separate respondent-by-timing matrix and supports
mean, median, standard-deviation, minimum, consistency, and Gaussian-mixture
scores. Fixed thresholds are inclusive; sample-relative defaults flag the low
5% for direct timing metrics and the high 5% for mixture probabilities.

## Documentation

Full docs live in [`docs/`](docs/) (MkDocs):

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Index catalog](docs/indices.md)
- [Screening workflow](docs/workflows/screening.md)
- [Composite guidance](docs/workflows/composite.md)
- [Threshold guidance](docs/thresholds.md)
- [R package notes](docs/r-comparison.md)
- [Changelog](CHANGELOG.md)

Build locally:

```bash
uv sync --group docs
uv run --no-sync mkdocs serve
```

Examples:

```bash
uv run python examples/basic_screening.py
uv run python examples/composite_scoring.py
uv run python examples/careless_responding_walkthrough.py
```

Benchmarks:

```bash
uv run python benchmarks/bench_screen.py
uv run python benchmarks/bench_cli_csv.py
uv run python benchmarks/bench_detection.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).

## Citation

Citation metadata for reference managers and GitHub's **Cite this repository**
feature is available in [`CITATION.cff`](CITATION.cff). The equivalent BibTeX
entry is:

```bibtex
@software{ier2026,
  title={IER: Python package for detecting Insufficient Effort Responding},
  author={Lyons, Cameron},
  year={2026},
  url={https://github.com/Cameron-Lyons/ier}
}
```

## References

- Curran, P. G. (2016). Methods for the detection of carelessly invalid responses in survey data. *Journal of Experimental Social Psychology*, 66, 4-19.
- Dunn, A. M., Heggestad, E. D., Shanock, L. R., & Theilgard, N. (2018). Intra-individual response variability as an indicator of insufficient effort responding. *Journal of Business and Psychology*, 33(1), 105-121.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data. *Psychological Methods*, 17(3), 437-455.
