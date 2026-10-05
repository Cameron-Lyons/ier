# IER

Python package for detecting **Insufficient Effort Responding (IER)** / careless
responding in survey data.

For a comprehensive methods review, see
[Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub).

## Features

**Detection indices**

- Consistency: IRV (optionally split), even-odd (item pairs or careless-style
  `method="halves"`), psychometric synonyms/antonyms (complete or
  careless-compatible `item_correlations="pairwise"`), person-total correlation,
  scale-aware resampled individual reliability, semantic pairs, and MAD
- Response patterns: longstring, average run length (`avgstr`), repeating
  patterns, Markov transition entropy, lagged autocorrelation, and onset
- Response styles: extreme, midpoint, and balanced acquiescence responding
- Outliers and person fit: Mahalanobis distance with Q-Q quantiles, Guttman
  errors, lz, and PerFit-style polytomous `gpoly` (Gp / Gnormed.poly) and
  `u3poly` (U3poly) plus dichotomous `ht` (Ht)
- Omissions and attention checks: skip-logic-aware missing rates, and bogus or
  diligence items with one expected answer or inclusive acceptable ranges
- Response times: row summaries, consistency, Gaussian mixtures, and item-level
  response time effort with normative NT10 thresholds
- `*_flag()` helpers for every index, plus `reverse_score()` for reverse-keyed items

**Screening and composites**

- `screen()` and `composite()` configured through one `IndexOptions` object,
  with an index catalog of defaults, flag directions, and required options
- Fixed or per-index percentile cutoffs with recorded provenance, multi-index
  consensus, and minimum valid-index requirements
- Standardized or raw, weighted mean/sum/max composites with optional cutoff
  flags and uncalibrated logistic values
- Reusable `screen_scores()`, `composite_scores()`, and
  `response_time_score_flags()` layers for sensitivity analysis without
  rescoring
- Reverse-keyed items recoded once (`IndexOptions.reverse_keyed_items`,
  `--reverse-keyed-items`) for consistency and person-fit indices only, so one
  run serves keyed and as-presented indices
- Soft or strict per-index failures with retained diagnostics, and opt-in
  thread-based parallel scoring

**Results, persistence, and plots**

- `screen_table()` and `composite_table()` for index-preserving pandas or
  Polars DataFrames, and `index_agreement()` for co-flag and rank agreement
- Validated, pickle-free NPZ archives for scores, complete screening results,
  and response times, with optional streamed compression and atomic writes
- Distribution plots with cutoffs, composite histograms, agreement matrices,
  flag counts, and block-aggregated heatmaps for large surveys

**Command line**

- `ier screen`, `composite`, `response-time`, and saved-score replay commands
  with text, JSON, CSV, and NPZ output, including compressed and standard-stream
  input and output
- Survey exports with metadata: respondent ID preservation, named, glob-pattern,
  or excluded item columns, preamble rows, and missing-value tokens
- Shareable TOML `--config` files, `ier inspect` to check how a file is parsed,
  and `ier indices` to browse the catalog

**Engineering**

- NumPy-only base install; inputs may be lists, arrays, memory-mapped `.npy`
  files, pandas (including nullable dtypes), or Polars frames
- Numerically careful kernels with exact repairs for overflow, underflow, and
  cancellation, tested against brute-force and exact-rational oracles
- Full type annotations (`py.typed`) and docstring examples checked as doctests

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

Keep pandas row labels, persist complete decisions, and compare indices:

```python
import pandas as pd
from ier import (
    IndexOptions,
    index_agreement,
    load_screen_archive,
    save_screen_archive,
    screen,
    screen_table,
)

frame = pd.read_csv("survey.csv", index_col="participant_id")
result = screen(frame, options=IndexOptions(scale_min=1, scale_max=5))

report = pd.DataFrame(screen_table(result), index=frame.index)
print(report.sort_values("flag_count", ascending=False).head())

save_screen_archive("screening.npz", result, respondent_ids=frame.index.astype(str).tolist())
restored = load_screen_archive("screening.npz")["result"]

names, jaccard = index_agreement(restored, "jaccard")
print(pd.DataFrame(jaccard, index=names, columns=names).round(2))
```

**pandas nullable dtypes.** DataFrames with extension dtypes (`Int64`, `Float64`,
`boolean`, or the result of `convert_dtypes()`), object columns, and `pd.NA` or
`None` missing values are accepted by every index and by `screen()` and
`composite()`. They are converted to float64 with missing values as `NaN`.
Integer NumPy arrays are used without conversion. Masked cells of a
`numpy.ma.MaskedArray` (for example `np.ma.masked_equal(data, -99)`) are missing
responses: an array with masked cells is copied with them set to `NaN`. Complex,
datetime, timedelta, and non-numeric text matrices, including `pd.NaT` values,
raise `ValueError`.

## CLI

```bash
ier screen data.csv --scale-min 1 --scale-max 5
ier screen data.csv --format json --output screen.json
ier screen data.csv --threshold irv=0.25 --threshold longstring=8
ier screen data.csv --index-percentile irv=90 --index-percentile longstring=99
ier screen data.csv --indices irv longstring missing_rate --min-valid-indices 2
ier screen data.csv --indices irv mad --strict
ier screen data.csv --workers 4
ier screen data.csv --indices acquiescence --scale-min 1 --scale-max 5 \
  --acquiescence-positive-items 0,2 --acquiescence-negative-items 1,3
ier screen data.csv --indices missing_rate --missing-item-indices 0,1,4
ier screen data.csv --indices missing_rate --missing-applicable-mask applicable.csv
ier screen data.csv --indices infrequency \
  --infrequency-item-indices 3,7 \
  --infrequency-expected-responses 5,1 --infrequency-missing fail
ier screen data.csv --indices infrequency \
  --infrequency-item-indices 3,7 --infrequency-acceptable-ranges '1:2,4:'
ier screen data.csv --indices irv avgstr longstring --irv-num-split 2
ier screen data.csv --indices psychsyn psychant --psychsyn-item-correlations pairwise
ier screen data.csv --indices evenodd individual_reliability \
  --evenodd-factors 8,8,8 --evenodd-method halves --reliability-factors 8,8,8
ier screen data.csv --indices autocorrelation longstring_pattern --autocorrelation-max-lag 8
ier screen data.csv --id-column participant_id --format csv --output screening.csv
ier screen data.csv --id-column participant_id --item-columns q1,q2,q3,q4
ier screen data.csv --id-column participant_id --item-column 'Q1, agreement' --item-column Q2
ier screen export.csv --id-column ResponseId --item-pattern 'Q*'
ier screen export.csv --id-column ResponseId --exclude-column StartDate --exclude-column Duration
ier screen data.csv --format npz --output screening.npz
ier screen data.csv --format npz --compress --output compact-screening.npz
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
ier response-time timings.csv --metric effort --effort-max-threshold 10
ier screen responses.csv.gz --format json --output screening.json.gz
ier screen responses.csv.xz --format csv --output screening.csv.xz
ier screen survey-export.csv --skip-rows 2 --id-column participant_id
ier screen responses.npy --indices irv longstring
ier screen-scores screening.npz --index-percentile irv=99 --min-flags 2
ier composite-scores screening.npz --indices irv longstring --weight irv=2
ier composite-scores screening.npz --skip-unsupported --format csv
ier response-time-scores timing.npz --threshold 1.0 --format json
ier response-time-scores timing.npz --format csv --output timing.csv
cat responses.csv | ier screen - --indices irv longstring --format json
ier screen data.csv --config ier.toml --percentile 99
ier inspect export.csv --id-column ResponseId --item-pattern 'Q*' --missing-value NA
ier indices --format json
ier --version
```

Input matrices may be comma-, tab-, semicolon-, or whitespace-delimited. Common
delimiters are auto-detected unless `--delimiter` is supplied. Blank fields in
delimited files are loaded as missing values (`NaN`). Repeat `--missing-value TOKEN`
to map explicit survey-export markers such as `NA` or `-99` to missing values.
Use `--skip-rows N` to discard exactly `N` physical preamble lines before
delimiter and header detection.
One leading UTF-8 byte-order mark is removed automatically from delimited input.
Header detection is automatic by default; use `--header present` for numeric column
names or `--header absent` to require every row to be numeric. Use `--id-column NAME` to
remove a named header column from scoring and preserve its unique, nonblank values
in text, JSON, CSV, and NPZ output. Use `--item-columns q1,q2,...` to select and order
the numeric item matrix while ignoring unselected metadata columns; repeat the
option to build the selection in groups. Use repeatable `--item-column NAME` for
an exact header name containing a comma. Both forms can be mixed, retaining their
command-line order. Survey exports with metadata columns such as `StartDate`,
`Duration`, or `IPAddress` can instead select items with repeatable
`--item-pattern GLOB` (case-sensitive shell-style patterns such as `'Q*'`; matches
keep header order, once each) or drop named columns with repeatable
`--exclude-column NAME`. Exclusions apply after any item selection; on their own
they score every column except the ID column and the excluded names. Patterns
cannot be combined with exact `--item-columns` names, every pattern and excluded
name must match the header, and a pattern that selects the ID column is an error
unless that column is also excluded.
Run `ier screen --help` for every option, grouped into input, index, decision,
and output settings with their defaults. Scoring and saved-score commands also
read option values from a TOML file passed with `--config`, flat or in
per-command sections; command-line options take precedence. `ier inspect`
reports the detected delimiter, header, selected columns, missing cells, and
observed response range, with suggested `--scale-min`/`--scale-max` values.
Quoted delimiters, escaped quotes, and multiline identifiers are preserved.
Entirely missing delimited records retain their respondent position; physically
blank lines are skipped.
Malformed quoted records fail with their physical line number before an existing
result file is replaced; jagged records retain their detected delimiter and
report the actual and expected column counts.
Cells that cannot be parsed as numbers report their data row, column name, and
position.

Missing-response scoring is opt-in because planned omissions are often valid.
Use `IndexOptions(missing_item_indices=[...])` or CLI
`--missing-item-indices 0,1,...` for a fixed required-item subset. Supply a
respondent-by-item Boolean `missing_applicable_mask` in Python, or
`--missing-applicable-mask PATH` on fresh CLI screening and composite commands;
false cells are excluded from the missing-rate denominator. CLI masks accept
headerless 0/1 text (plain, gzip, bzip2, xz, or `-` for standard input) or an
uncompressed Boolean `.npy` file. Their shape and order must match the response
matrix after item selection, excluding identifiers and metadata. Response data
and a mask cannot both use standard input.

`screen()` and all composite helpers accept `workers=N`; the corresponding CLI
commands use `--workers N`. The default is sequential (`1`) for predictable
resource use. Higher values preserve index and failure ordering and can improve
large multi-index workloads, but they may increase peak memory. The standard
library provides the worker pool, so this adds no dependency.

Balanced acquiescence mode pairs positively and negatively worded items by their
0-based matrix positions. Supply both equal-length lists through
`IndexOptions` or the two `--acquiescence-*-items` options; unequal lists fail
instead of silently dropping configured items. Supply raw agreement responses
for both item polarities, without reverse-scoring negative items. Pair means are
normalized using the configured or inferred response-scale bounds.

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

Pass `errors=result["errors"]` alongside retained scores to preserve soft-failure
provenance and the original selected-index count when reapplying completeness
rules. Failed indices remain unavailable; they never contribute to respondent
coverage or decisions.

If every selected index failed, also pass `n_respondents=result["n_respondents"]`
to `screen_scores()` and `save_score_archive()`. Saved-screen CLI commands retain
that count automatically so respondent rows and identifiers survive replay.

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
`composite_scores_summary()` additionally reports component availability and
aggregate statistics using the same reduction pass.
Both composite reuse APIs accept `errors=details["errors"]` to retain failed
selections when validating weights and minimum coverage. Saved-score CLI commands
carry this provenance automatically.

Use `save_score_archive("scores.npz", scores)` to persist any ordered mapping of
raw registered-index vectors directly from Python, then
`load_score_archive("scores.npz")` to restore the vectors, optional respondent
IDs, and soft failures. Full CLI screen output and detailed composite archives
written with `--include-components` are compatible as well; schema, registry,
alignment, and pickle-free safety checks run before reuse.
To keep a complete `screen()` result, including its flags, cutoffs, consensus
decisions, and summaries, use `save_screen_archive("screening.npz", result)` and
`load_screen_archive("screening.npz")["result"]`. The loader recomputes every
recorded decision and also reads CLI `screen --format npz` archives.

The `screen-scores` and `composite-scores` CLI commands reuse these archives
without the original item matrix. They support the corresponding cutoffs,
percentiles, weights, coverage controls, and all four output formats. Saved
respondent IDs remain aligned. Optional `--indices` selects saved components in
the requested order; by default all saved scores are used. Composite inputs must
contain composite-enabled indices, so select a subset or pass
`--skip-unsupported` when the screening archive also contains response-style or
onset scores. Archive failures remain visible
unless an explicit subset excludes them; `--strict` rejects retained failures.
Composite output omits failures belonging only to screening indices.

Response-time results have matching `save_response_time_archive()` and
`load_response_time_archive()` boundaries. The writer preserves prepared scores,
Boolean flags, cutoff metadata, and optional identifiers after verifying the
same fixed or percentile decision contract enforced by the loader.
Use `ier response-time-scores timing.npz` to export the saved scores and exact
decisions without the original timings or another mixture fit. Supplying
`--threshold` or `--percentile` applies a new cutoff using the saved metric and
flag direction; omitting both preserves the stored flags, including cutoff ties.

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
column headers, `--id-column`, `--item-column`, `--item-columns`, `--item-pattern`,
`--exclude-column`, and `--delimiter` do not apply.
Use an uncompressed `.npy` file rather than a compressed `.npy` file to preserve
memory mapping.

Use `-` as the input path for a forward-only standard-input pipeline or as the
output path for standard output. Files ending in `.gz`, `.bz2`, or `.xz` are read
and written transparently using the Python standard library. CSV rows and JSON
respondent arrays are written in bounded chunks, so output allocation stays
bounded for plain, compressed, and standard-output destinations.

Regular output files are replaced atomically only after serialization and
compression finish. Handled write failures preserve previous results and remove
staged output. Existing file permission bits and symbolic links are retained;
standard output, pipes, and device destinations continue streaming directly.

Text output selects a small `--top N` preview in bounded batches and preserves
input row order when ranking values tie. Use `--top 0` to show summary metadata
without selecting respondent rows.

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

Add `--compress` to any fresh or saved-score command using `--format npz` to
compress the archive's members. Python writers accept `compressed=True` in
`save_score_archive()`, `save_screen_archive()`, and `save_response_time_archive()`.
Compression can save
substantial storage for repeated scores, flags, and identifiers, but costs extra
CPU when writing and reading; uncompressed output remains the default. Both
forms load and replay through the same APIs, with identical scores and decisions.
Compressed output uses the fastest DEFLATE level by default; choose 1–9 with
`--compress-level` or `compression_level=` when smaller files matter more.

`ier response-time` accepts a separate respondent-by-timing matrix and supports
mean, median, standard-deviation, minimum, consistency, Gaussian-mixture, and
response time effort (`--metric effort`) scores. Fixed thresholds are inclusive;
sample-relative defaults flag the low 5% for direct timing metrics and the high
5% for mixture probabilities. Effort instead flags RTE strictly below a fixed
0.90 by default, like `response_time_effort_flag()`, and a fixed effort
`--threshold` is strict as well; `--effort-fraction`, `--effort-max-threshold`,
or a shared `--effort-threshold` set the item thresholds. Retain a direct,
consistency, or mixture score vector and pass it to
`response_time_score_flags()` to compare cutoffs without recalculating row
summaries or refitting a mixture. That function includes ties at a fixed cutoff
and defaults to a percentile, so reflag effort with
`response_time_effort_flag(times, threshold=...)` or compare retained RTE scores
with `scores < cutoff`. NPZ output
loads through `load_response_time_archive()`, which validates its schema and
returns the stored scores ready for the same sensitivity workflow. Use
`save_response_time_archive()` to create the identical interoperable schema from
Python.

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
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_person_fit.py
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
uv run python benchmarks/bench_score_reuse.py
uv run python benchmarks/bench_score_reuse.py --workflow composite
uv run python benchmarks/bench_score_reuse.py --workflow response-time
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
- Emons, W. H. M. (2008). Nonparametric person-fit analysis of polytomous item scores. *Applied Psychological Measurement*, 32(3), 224-247.
- Gottfried, J., Ježek, S., Králová, M., & Řiháček, T. (2022). Autocorrelation screening: A potentially efficient method for detecting repetitive response patterns in questionnaire data. *Practical Assessment, Research, and Evaluation*, 27, Article 2.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data. *Psychological Methods*, 17(3), 437-455.
- Niessen, A. S. M., Meijer, R. R., & Tendeiro, J. N. (2016). Detecting careless respondents in web-based questionnaires: Which method to use? *Journal of Research in Personality*, 63, 1-11.
- Sijtsma, K., & Meijer, R. R. (1992). A method for investigating the intersection of item response functions in Mokken's nonparametric IRT model. *Applied Psychological Measurement*, 16(2), 149-157.
- Tendeiro, J. N., Meijer, R. R., & Niessen, A. S. M. (2016). PerFit: An R package for person-fit analysis in IRT. *Journal of Statistical Software*, 74(5), 1-27.
- Wise, S. L., & Kong, X. (2005). Response time effort: A new measure of examinee motivation in computer-based tests. *Applied Measurement in Education*, 18(2), 163-183.
- Wise, S. L., & Ma, L. (2012). Setting response time thresholds for a CAT item pool: The normative threshold method. Paper presented at the annual meeting of the National Council on Measurement in Education, Vancouver, Canada.
