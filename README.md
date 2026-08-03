# IER

Python package for detecting **Insufficient Effort Responding (IER)** / careless
responding in survey data.

For a comprehensive methods review, see
[Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub).

## Features

- Multiple detection families: consistency, response patterns, response styles, outliers, omissions, response times, attention checks
- Workflow APIs: `screen()` and `composite()` configured via `IndexOptions`
- Reusable `screen_scores()`, `composite_scores()`, and `response_time_score_flags()` layers
- Validated, pickle-free score and response-time archive persistence
- Auto-detecting archive loading and text/JSON metadata inspection
- Archive-backed screening, composite, and timing sensitivity commands without rescoring
- Bounded pairwise-complete item discovery and scoring for psychometric synonym analyses
- Contracted, active-row-compacted LZ estimation for complete and missing responses
- Allocation-bounded split-half reliability with stable raw-moment correlations
- Exact low-allocation two-point correlations for common even-odd factor designs
- Batched dense, sparse, and grouped missing-response Markov entropy scoring
- Batched missing-response longest-run and repeating-pattern scoring
- Batched dependency-free chi-square quantiles for large Mahalanobis Q-Q plots
- Grouped missing-aware medians for response-time summaries and mixtures
- Adaptive bounded Guttman counters for narrow and wide response scales
- Validated per-index weights across all composite scoring helpers
- Standardized or raw-score composite combination from Python and the CLI
- Opt-in fixed or sample-percentile composite flags in every CLI output format
- Opt-in uncalibrated logistic composite values in every CLI output format
- Opt-in minimum valid-index requirements for defensible composite coverage
- Configurable multi-index consensus decisions for respondent-level screening
- Opt-in minimum valid-index requirements and eligibility reporting for screening consensus
- Fixed or per-index sample-relative screening thresholds with cutoff provenance
- Skip-logic-aware missing-response rates with required-item subsets and applicability masks
- Configurable attention-check missing policies with count or proportion scoring
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
print("Valid index counts:", result["valid_index_counts"])
print("IRV coverage:", result["summary"]["irv"])
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
ier screen data.csv --index-percentile irv=90 --index-percentile longstring=99
ier screen data.csv --indices irv longstring missing_rate --min-valid-indices 2
ier screen data.csv --indices irv mad --strict
ier screen data.csv --workers 4
ier screen data.csv --indices missing_rate --missing-item-indices 0,1,4
ier screen data.csv --indices infrequency \
  --infrequency-item-indices 3,7 \
  --infrequency-expected-responses 5,1 --infrequency-missing fail
ier screen data.csv --id-column participant_id --format csv --output screening.csv
ier screen data.csv --id-column participant_id --item-columns q1,q2,q3,q4
ier screen data.csv --format npz --output screening.npz
ier screen-reflag screening.npz --percentile 99 --format json --output stricter.json
ier screen-reflag screening.npz --indices irv longstring --threshold longstring=8
ier composite data.csv --indices irv longstring
ier composite data.csv --indices irv longstring --no-standardize
ier composite data.csv --indices irv longstring --percentile 95 --format csv
ier composite data.csv --indices irv longstring --threshold 1.5 --format json
ier composite data.csv --indices irv longstring --include-probability --format csv
ier composite data.csv --indices irv longstring --weight irv=2 --weight longstring=0.5
ier composite data.csv --indices irv longstring markov --min-valid-indices 2
ier composite data.csv --indices irv longstring --include-components --format json
ier composite data.csv --format csv --output scores.csv
ier composite-recombine composite.npz --weight irv=2 --method mean --format json
ier composite-recombine composite.npz --indices irv longstring --no-standardize
ier archive-info composite.npz --format json
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

Missing-response scoring is opt-in because planned omissions are often valid.
Use `IndexOptions(missing_item_indices=[...])` or CLI
`--missing-item-indices 0,1,...` for a fixed required-item subset. Python
workflows can additionally provide a respondent-by-item Boolean
`missing_applicable_mask`; false cells are excluded from the missing-rate
denominator.

`screen()` and all composite helpers accept `workers=N`; the corresponding CLI
commands use `--workers N`. The default is sequential (`1`) for predictable
resource use. Higher values preserve index and failure ordering and can improve
large multi-index workloads, but they may increase peak memory. The standard
library provides the worker pool, so this adds no dependency.

After index scoring, screening flag counts and composite scores are reduced one
index at a time. Large multi-index workflows therefore avoid a second
respondent-by-index matrix while retaining every per-index score and flag in the
result.

Each screening index summary reports valid and unavailable score counts plus the
flagged count and valid-score flag rate. This makes coverage differences visible
without recomputing them from the retained arrays.

Use `screen_scores(result["scores"], ...)` to compare new fixed cutoffs,
tail percentiles, consensus thresholds, or completeness rules without running
the indices again. The reusable path validates registered, equally sized score
vectors and returns the same screening result structure while retaining compatible
NumPy arrays by reference.

Set `screen(..., min_valid_indices=N)` or CLI `--min-valid-indices N` to require
at least `N` available index scores before a respondent is eligible for a
consensus decision. Results always include per-respondent `valid_index_counts`
and `consensus_eligible`; omitting the requirement preserves existing decisions.

Use `screen(..., percentiles={"irv": 90, "longstring": 99})` or repeat CLI
`--index-percentile INDEX=VALUE` to tune sample-relative sensitivity by signal.
Values follow the global tail convention: high-direction indices resolve at
`p`, while low-direction indices resolve at `100-p`. Results report each actual
numeric cutoff, its fixed/percentile/presence source, and the requested tail
percentile so exported decisions remain reproducible.

All composite helpers accept optional positive finite `weights`. Weighting is
applied after low-is-suspicious indices are direction-corrected and after
optional standardization; unspecified selected indices retain weight 1.

Use `composite_scores(details["indices"], ...)` with the raw component mapping
from `composite_summary()` to compare weights, mean/sum/max reductions,
standardization, or completeness rules without recalculating any index. Direction
correction remains automatic and inputs are not mutated.

Use `save_score_archive("scores.npz", scores)` to persist any ordered mapping of
raw registered-index vectors directly from Python, then
`load_score_archive("scores.npz")` to restore the vectors, optional respondent
IDs, and soft failures. Full CLI screen output and detailed composite archives
written with `--include-components` are compatible as well; schema, registry,
alignment, and pickle-free safety checks run before reuse.

CLI users can apply the same decision layer to those validated archives without
the original response matrix:

```bash
ier screen-reflag screening.npz --percentile 99 --min-flags 3 --format json
ier screen-reflag screening.npz --indices irv longstring \
  --threshold longstring=8 --index-percentile irv=90 --output revised.npz --format npz
```

Omitting `--indices` reuses every stored vector; an explicit list selects and
orders the vectors used for consensus. Respondent identifiers and archived soft
failures are carried forward. An NPZ output may replace the source archive
because the validated input is loaded before atomic output begins.

Detailed composite archives and compatible score archives can also be
recombined without the original matrix:

```bash
ier composite-recombine composite.npz --weight irv=2 --weight longstring=0.5 \
  --min-valid-indices 2 --percentile 95 --format json
ier composite-recombine composite.npz --indices irv longstring --method max \
  --no-standardize --include-components --format npz --output revised.npz
```

The command reuses the existing direction-aware `composite_scores()` layer and
the ordinary composite serializers. It preserves identifiers and archived soft
failures, supports optional component and uncalibrated probability output, and
never reruns an index. Replacing the source NPZ requires `--include-components`
so the output remains reusable.

Response-time results have matching `save_response_time_archive()` and
`load_response_time_archive()` boundaries. The writer preserves prepared scores,
Boolean flags, cutoff provenance, requested percentile, and optional identifiers
after verifying the same fixed-inclusive or percentile-exclusive decision
contract enforced by the loader. Existing writer calls without provenance retain
the legacy schema; pass `threshold_source` and, for derived cutoffs, `percentile`
to produce the richer schema used by current CLI output.

When the archive type is not known in advance, `load_archive()` reads the
declared result type and applies the complete score or response-time validator
in one pass. `ier archive-info results.npz` exposes the same auto-detection as a
compact text summary; add `--format json` for structured metadata. The command
reports dimensions and identifier presence plus stored index/failure metadata
or timing cutoff provenance and flag rates without printing respondent vectors.

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
structured metadata without adding a dependency. Writers stage a complete
archive beside the destination and replace it atomically, so an interrupted
write cannot truncate an existing result. See
[CLI output formats](docs/cli-output.md) for the versioned schema and loading examples.

`ier response-time` accepts a separate respondent-by-timing matrix and supports
mean, median, standard-deviation, minimum, consistency, and Gaussian-mixture
scores. Missing-aware median summaries and mixture preprocessing reuse bounded
NaN-free row groups, keeping temporary matrix workspaces independent of respondent count.
Fixed thresholds are inclusive; sample-relative defaults flag the low
5% for direct timing metrics and the high 5% for mixture probabilities. Text,
JSON, and NPZ preserve whether the resolved cutoff was fixed or percentile-based
and record the requested percentile when applicable. Retain
any returned score vector and pass it to `response_time_score_flags()` to compare
cutoffs without recalculating row summaries or refitting a mixture. NPZ output
loads through `load_response_time_archive()`, which validates its schema and
returns the stored scores and provenance ready for the same sensitivity workflow.
Use `save_response_time_archive()` to create the identical interoperable schema
from Python. CLI users can reapply a cutoff without rescoring the original matrix:

```bash
ier response-time-reflag timing.npz --percentile 1 --format npz --output strict.npz
ier response-time-reflag timing.npz --threshold 1.0 --format csv --output flags.csv
```

The command validates legacy v1 and provenance-aware v2 input, preserves timing
scores, metric direction, and respondent identifiers, and requires exactly one
new fixed or percentile cutoff.

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
uv run python benchmarks/bench_pair_differences.py
uv run python benchmarks/bench_mahad.py
uv run python benchmarks/bench_guttman.py
uv run python benchmarks/bench_reliability.py
uv run python benchmarks/bench_onset.py
uv run python benchmarks/bench_person_total.py
uv run python benchmarks/bench_row_reductions.py
uv run python benchmarks/bench_lz.py
uv run python benchmarks/bench_markov.py
uv run python benchmarks/bench_response_time.py
uv run python benchmarks/bench_orchestration.py
uv run python benchmarks/bench_flagging.py
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
