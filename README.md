# IER

Python package for detecting **Insufficient Effort Responding (IER)** / careless
responding in survey data.

For a comprehensive methods review, see
[Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub).

## Features

- Multiple detection families: consistency, response patterns, response styles, outliers, omissions, response times, attention checks
- Workflow APIs: `screen()` and `composite()` configured via `IndexOptions`
- Validated per-index weights across all composite scoring helpers
- Standardized or raw-score composite combination from Python and the CLI
- Opt-in fixed or sample-percentile composite flags in every CLI output format
- Opt-in uncalibrated logistic composite values in every CLI output format
- Opt-in minimum valid-index requirements for defensible composite coverage
- Configurable multi-index consensus decisions for respondent-level screening
- Fixed or sample-relative per-index screening thresholds
- Programmatic and CLI index catalog with defaults and configuration requirements
- CLI preservation of named respondent identifier columns
- CLI selection of named item columns from files containing metadata
- CLI workflows for item screening, composite scores, and response-time analysis
- NumPy-first inputs (lists, arrays, array-compatible DataFrames)
- Configurable soft or strict per-index failures during screening and composite scoring
- Opt-in soft-failure diagnostics from every composite Python helper
- Composite CLI diagnostics preserved across human-readable and structured outputs
- Opt-in raw component scores and availability counts in every composite CLI format
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
from ier import IndexOptions, composite, composite_probability, irv, screen

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

weighted = composite(
    data,
    indices=["irv", "longstring"],
    weights={"irv": 2.0, "longstring": 0.5},
)
print("Weighted composite:", weighted)

complete_enough = composite(
    data,
    indices=["irv", "longstring", "person_total"],
    min_valid_indices=2,
)
print("Coverage-filtered composite:", complete_enough)

probabilities, failures = composite_probability(
    data,
    indices=["irv", "mad"],
    return_diagnostics=True,
)
print("Logistic composite:", probabilities)
print("Unavailable indices:", failures)

# Opt in to concurrent index scoring for larger matrices.
large_result = screen(data, workers=4)
```

## CLI

```bash
ier screen data.csv --scale-min 1 --scale-max 5
ier screen data.csv --format json --output screen.json
ier screen data.csv --threshold irv=0.25 --threshold longstring=8
ier screen data.csv --indices irv mad --strict
ier screen data.csv --workers 4
ier screen data.csv --id-column participant_id --format csv --output screening.csv
ier screen data.csv --id-column participant_id --item-columns q1,q2,q3,q4
ier screen data.csv --format npz --output screening.npz
ier composite data.csv --indices irv longstring
ier composite data.csv --indices irv longstring --no-standardize
ier composite data.csv --indices irv longstring --percentile 95 --format csv
ier composite data.csv --indices irv longstring --threshold 1.5 --format json
ier composite data.csv --indices irv longstring --include-probability --format csv
ier composite data.csv --indices irv longstring --weight irv=2 --weight longstring=0.5
ier composite data.csv --indices irv longstring markov --min-valid-indices 2
ier composite data.csv --indices irv longstring --include-components --format json
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
in text, JSON, CSV, and NPZ output. Use `--item-columns q1,q2,...` to select and order
the numeric item matrix while ignoring unselected metadata columns; repeat the
option to build the selection in groups.

`screen()` and all composite helpers accept `workers=N`; the corresponding CLI
commands use `--workers N`. The default is sequential (`1`) for predictable
resource use. Higher values preserve index and failure ordering and can improve
large multi-index workloads, but they may increase peak memory. The standard
library provides the worker pool, so this adds no dependency.

After index scoring, screening flag counts and composite scores are reduced one
index at a time. Large multi-index workflows therefore avoid a second
respondent-by-index matrix while retaining every per-index score and flag in the
result.

All composite helpers accept optional positive finite `weights`. Weighting is
applied after low-is-suspicious indices are direction-corrected and after
optional standardization; unspecified selected indices retain weight 1.

Composite scores are standardized per index by default. Pass
`ier composite --no-standardize` to combine directed scores in their original
units. Text, JSON, and NPZ record the effective setting; CSV remains a compact
respondent table.

Add either `--threshold VALUE` or `--percentile VALUE` to emit respondent-level
composite flags without running the indices again. Fixed cutoffs flag scores at
or above the value; percentile cutoffs flag only scores strictly above the
sample cutoff. Omitting both options preserves score-only output.

Pass `--include-probability` to add the overflow-safe logistic transform beside
each composite score without scoring the indices again. JSON and NPZ label the
scale as `uncalibrated_logistic`, CSV adds `composite_probability`, and text
shows the value in ranked rows. These values remain sample-relative ranking
aids, not calibrated probabilities. Fixed and percentile flag cutoffs continue
to use the original composite-score units.

Set `min_valid_indices=N` to return `NaN` when fewer than `N` selected index
scores are available for a respondent. This opt-in rule applies after scoring
failures and missing-value handling, and `composite_summary()` reports the
per-respondent valid-index counts used by the rule.

Uncompressed `.npy` files are memory-mapped read-only for fast, low-overhead
loading of large headerless real numeric matrices. Because binary arrays have no
column headers, `--id-column`, `--item-columns`, and `--delimiter` do not apply.
Use an uncompressed `.npy` file rather than `.npy.gz` to preserve memory mapping.

Use `-` as the input path for a forward-only standard-input pipeline or as the
output path for standard output. Files ending in `.gz` are read and written
transparently using the Python standard library. CSV rows and JSON respondent
arrays are written in bounded chunks, so output allocation stays bounded for
plain, compressed, and standard-output destinations.

When a requested screen or composite index soft-fails, every CLI format emits a
concise warning on standard error. Text output also lists the failure, JSON
includes an `errors` object, and NPZ includes aligned `error_names` and
`error_messages` arrays. CSV remains a clean respondent-level table; use its
standard-error stream to retain the diagnostic or pass `--strict` to fail
immediately.

Pass `ier composite --include-components` to audit how the aggregate was built.
Text, JSON, CSV, and NPZ then include successful raw per-index scores and each
respondent's valid-index count. The option is explicit because component arrays
increase output size; the default aggregate-only path and schemas remain lean.

All scoring commands accept `--format npz --output FILE.npz` for fast, typed,
pickle-free result archives. NPZ preserves boolean flags, non-finite scores, and
structured metadata without adding a dependency. See
[CLI output formats](docs/cli-output.md) for the versioned schema and loading examples.

`ier response-time` accepts a separate respondent-by-timing matrix and supports
mean, median, standard-deviation, minimum, consistency, and Gaussian-mixture
scores. Fixed thresholds are inclusive; sample-relative defaults flag the low
5% for direct timing metrics and the high 5% for mixture probabilities.

## Documentation

Full docs live in [`docs/`](docs/) (MkDocs):

- [Getting started](docs/getting-started.md)
- [CLI output formats](docs/cli-output.md)
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
uv run python benchmarks/bench_evenodd.py
uv run python benchmarks/bench_psychsyn.py
uv run python benchmarks/bench_person_total.py
uv run python benchmarks/bench_row_reductions.py
uv run python benchmarks/bench_lz.py
uv run python benchmarks/bench_markov.py
uv run python benchmarks/bench_response_time.py
uv run python benchmarks/bench_orchestration.py
uv run python benchmarks/bench_cli_output.py
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
