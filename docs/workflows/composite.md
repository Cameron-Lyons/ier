# Composite Scores

`composite()` combines multiple IER indices into one sample-relative score.
Higher values indicate stronger careless-responding signal **within the sample**.

```python
from ier import IndexOptions, composite, composite_probability

opts = IndexOptions()
scores = composite(
    data,
    indices=["irv", "longstring", "person_total", "markov"],
    options=opts,
)
weighted_scores = composite(
    data,
    indices=["irv", "longstring", "person_total"],
    weights={"irv": 2.0, "person_total": 0.5},
)
coverage_filtered = composite(
    data,
    indices=["irv", "longstring", "person_total"],
    min_valid_indices=2,
)
ranks = composite_probability(data, indices=["irv", "longstring"], options=opts)
auditable_ranks, probability_errors = composite_probability(
    data,
    indices=["irv", "mad"],
    options=opts,
    return_diagnostics=True,
)

# Available on composite(), composite_flag(), composite_summary(), and
# composite_probability().
parallel_scores = composite(data, indices=["irv", "longstring"], workers=4)
```

## Important caveats

!!! warning "Not a calibrated probability"
    `composite_probability()` applies a logistic transform to standardized
    composite scores. Values lie in `[0, 1]` but are **not** validated
    probabilities of IER unless you calibrate against labeled data from a
    comparable survey. The transform is numerically stable for extreme scores,
    including infinite inputs, without emitting overflow warnings.

Practical guidance:

1. Prefer multi-index agreement over any single cutoff.
2. Review flagged cases substantively (open text, completion time, attention checks).
3. Report which indices, combination method, weights, and completeness rule you used.
4. Use `method="best_subset"` when you want the Curran/Meade-Craig style mix of
   consistency, pattern, and (optionally) MAD signals.

## Allowed indices

Composite-enabled indices include:

`irv`, `longstring`, `longstring_pattern`, `mahad`, `psychsyn`, `psychant`,
`person_total`, `markov`, `guttman`, `individual_reliability`, `evenodd`, `mad`,
`lz`, `semantic_syn`, `semantic_ant`, `infrequency`, `missing_rate`

Configure `missing_rate` with `IndexOptions(missing_item_indices=[...])` for a
fixed required-item subset or `missing_applicable_mask=...` for respondent-specific
skip logic. Respondents without applicable items contribute an unavailable
component score, so `min_valid_indices` can enforce the desired composite coverage.

The `infrequency` component also accepts `infrequency_missing`. Policies `omit`
and `propagate` can produce unavailable scores; combine them with
`min_valid_indices` when attention-check availability is required for a composite.

The `lz` component accepts `lz_difficulty`, `lz_discrimination`, `lz_theta`,
and `lz_model` through `IndexOptions`, matching the direct `lz()` API. Supplying
independently calibrated values avoids fallback estimation from the response
matrix and preserves the same low-score direction correction during
combination. The command-line equivalents are `--lz-difficulty`,
`--lz-discrimination`, `--lz-theta`, and `--lz-model`.

The `irv` component accepts `irv_num_split` for equal questionnaire sections or
`irv_split_points` for explicit boundaries through `IndexOptions`. Supplying
either option enables section scoring before direction correction and
combination. The command-line equivalents are `--irv-num-split` and
`--irv-split-points`.

Screening-only response-style indices (`u3_poly`, `midpoint`, `acquiescence`,
`onset`) are excluded from composite combination because they measure different
constructs and can dilute pattern/consistency signals.

## Combination methods

| Method | Behavior |
|--------|----------|
| `mean` | Mean of (optionally standardized) directed scores |
| `sum` | Sum of directed scores |
| `max` | Max of directed scores |
| `best_subset` | Forces `["mad", "irv", "longstring", "lz"]` when MAD items are provided, else `["irv", "longstring", "lz"]`, combined with `mean` |

Direction is handled automatically: low-is-bad indices are sign-flipped before
combination so that higher composite always means more IER signal.

Each directed index is standardized to a z-score by default so components with
different units contribute on a comparable scale. Set `standardize=False` in
Python or pass `ier composite --no-standardize` when original score units are
required. Raw-score combinations can be dominated by wider-ranging components,
so report the setting and justify any weights.

## Index weights

All composite helpers accept a partial `weights` mapping. Values must be
positive finite numbers, and every named index must be selected by the resolved
method. Selected indices omitted from the mapping retain weight 1.

Weights are applied after direction correction and optional standardization:

- `mean` and `best_subset` compute a weighted mean. When an index score is
  missing for one respondent, its weight is omitted from that respondent's
  denominator.
- `sum` computes a weighted sum; a respondent missing every index retains the
  established score of zero.
- `max` takes the maximum weighted directed score.

Multiplying every weight by the same constant leaves weighted means unchanged.
`composite_summary()` includes the full resolved weight mapping, including
default weight 1 values.

## Reusing component scores

Weight, method, standardization, and completeness sensitivity checks do not need
to recalculate the component indices. Retain the raw component mapping from one
detailed run and pass it to `composite_scores()`:

```python
from ier import composite_scores, composite_summary

initial = composite_summary(
    data,
    indices=["irv", "longstring", "person_total"],
)

weighted = composite_scores(
    initial["indices"],
    weights={"irv": 2.0, "longstring": 0.75},
)
raw_maximum = composite_scores(
    initial["indices"],
    method="max",
    standardize=False,
    min_valid_indices=2,
)
```

Inputs use the original public index directions. Low-is-suspicious signals are
reversed automatically, matching `composite()`. Only composite-enabled registered
indices are accepted, and every vector must be one-dimensional, non-empty,
equally sized, and contain only finite values or `NaN`. The input arrays are never
mutated.

`best_subset` is intentionally unavailable on this path because its purpose is
selecting and calculating a predefined component set. Supply that set explicitly
and use `method="mean"` once its raw scores have been retained.

On the bundled 10,000-respondent, 80-item benchmark, evaluating five weight
scenarios from retained scores takes 1.4 ms and 0.7 MiB peak temporary allocation
instead of 39.6 ms and 12.9 MiB for five full runs, a 28.4x speedup without
another dependency.

Component scores exported with `ier composite --include-components --format npz`
or saved directly from `composite_summary()` can be reloaded for a later session:

```python
from ier import composite_scores, load_score_archive, save_score_archive

save_score_archive(
    "raw-components.npz",
    initial["indices"],
    result_type="composite",
    errors=initial["errors"],
)
saved = load_score_archive("composite.npz")
reweighted = composite_scores(saved["scores"], weights={"irv": 2.0})
```

Aggregate-only composite archives intentionally fail this load because they do
not contain the raw registered-index vectors needed for a new combination.

Command-line workflows can reuse the same archives directly:

```bash
ier composite-recombine composite.npz --weight irv=2 --weight longstring=0.5 \
  --min-valid-indices 2 --percentile 95 --format json
ier composite-recombine composite.npz --indices irv longstring --method max \
  --no-standardize --include-components --include-probability \
  --format npz --output revised.npz
```

The optional `--indices` list selects and orders unique retained components;
omitting it uses every stored vector. Mean, sum, and maximum reductions are
available, while `best_subset` remains a raw-matrix selection workflow. Stored
identifiers and soft failures are preserved across text, JSON, CSV, and NPZ.
Replacing the source NPZ requires `--include-components` so later recombination
remains possible. Five scenarios that each reload the archive take 3.2 ms and
1.2 MiB peak traced allocation on the same benchmark, versus 39.6 ms and
12.9 MiB for full recomputation, a 12.4x speedup.

## Composite completeness

By default, composite methods retain their established missing-value behavior:
means renormalize over available components, maxima ignore missing components,
and sums return zero when every component is missing. Set
`min_valid_indices=N` when a composite should only be reported from at least
`N` available component scores. Respondents below the minimum receive `NaN`
before flagging or logistic transformation.

The minimum counts components, not their weights. A single heavily weighted
index therefore cannot satisfy a two-index requirement. A selected index that
soft-fails for the whole matrix is unavailable for every respondent and does
not lower the requested minimum. `composite_summary()` returns both the resolved
`min_valid_indices` value and a respondent-aligned `valid_index_counts` array so
coverage decisions can be audited.

By default, one invalid configured index is returned in diagnostics while other
indices continue. Pass `strict=True` to `composite()`, `composite_flag()`,
`composite_summary()`, or `composite_probability()` when every selected index
must succeed. For probability output, pass `return_diagnostics=True` to receive
`(probabilities, diagnostics)`; omitting it retains the established array-only
return value.

The CLI does not discard these soft failures. Every format writes a warning to
standard error, text output includes an `errors` section, JSON includes an
`errors` object, and NPZ stores aligned `error_names` and `error_messages`
vectors. CSV remains a respondent-only table, so retain the separate
standard-error stream when provenance is required.

All composite helpers accept `workers`. The default of `1` evaluates indices
sequentially; larger values retain index and diagnostic order while trading
higher temporary memory for potential throughput improvements.

## CLI

Blank fields in comma-, tab-, or semicolon-delimited input are loaded as missing
values (`NaN`) and follow each index's documented missing-data behavior.

```bash
ier composite data.csv --indices irv longstring --method mean
ier composite data.csv --indices irv longstring --no-standardize
ier composite data.csv --indices irv longstring --percentile 95 --format csv
ier composite data.csv --indices irv longstring --threshold 1.5 --format json
ier composite data.csv --indices irv longstring --include-probability --format csv
ier composite data.csv --indices irv longstring --weight irv=2 --weight longstring=0.5
ier composite data.csv --indices irv longstring markov --min-valid-indices 2
ier composite data.csv --indices irv longstring --include-components --format json
ier composite data.csv --indices irv mad --strict
ier composite data.csv --indices irv longstring --workers 4
ier composite data.csv --indices lz --no-standardize \
  --lz-difficulty difficulty.npy --lz-discrimination discrimination.csv \
  --lz-theta theta.npy --lz-model 2pl
ier composite data.csv --format json --output composite.json
ier composite data.csv --format csv --evenodd-factors 5,5 --indices irv evenodd
ier composite data.npy --indices irv longstring --format json
ier composite data.npy --indices irv longstring --format npz --output composite.npz
ier composite-recombine composite.npz --weight irv=2 --format json
ier composite-recombine composite.npz --method max --include-components \
  --format npz --output revised.npz
```

Uncompressed `.npy` matrices are memory-mapped read-only. They must be non-empty,
two-dimensional, and real numeric; header selection and delimiter options do not
apply to this binary format.

LZ parameter files may instead contain one numeric row or column in plain or
gzip-compressed delimited text, or a one-dimensional `.npy` array. Binary
vectors are also pickle-free, read-only memory maps. Parameter paths cannot use
standard input because the response matrix already owns that forward-only
stream.

CSV rows and JSON respondent arrays are written forward-only to plain files,
gzip files, or standard output without retaining the complete export in memory.

JSON output is standards-compliant: unavailable or non-finite scores are encoded
as `null`. CSV output represents those scores as empty cells so numeric columns
remain compatible with spreadsheet and statistics tools.
NPZ output preserves the numeric score vector, combination method,
standardization setting, and optional respondent IDs. Text, JSON, and NPZ
outputs also record explicitly supplied weight overrides; unlisted selected
indices use weight 1. When supplied, the minimum valid-index rule and any soft
failures are recorded alongside those fields. See the
[versioned archive schema](../cli-output.md).

Add `--include-components` when command-line results need respondent-level
provenance. Every format then includes the successfully computed raw index
scores and respondent-level availability counts. JSON and NPZ call the vector
`valid_index_counts`, CSV uses `valid_index_count`, and text labels it
`valid_indices`. Failed indices remain in the existing error metadata instead
of appearing as score columns. Component values are the raw public index
outputs, before composite direction correction, standardization, or weighting.
JSON arrays and CSV rows remain streamed, while NPZ stores one typed array per
successful index.

Add `--include-probability` when an export needs the same overflow-safe logistic
transform exposed by `composite_probability()`. The command transforms the
already-computed aggregate vector, so index scoring still runs once. JSON and
NPZ include `probability_scale="uncalibrated_logistic"`, CSV adds
`composite_probability`, and text adds a probability column. Omitting the
option preserves every existing schema. These values are not calibrated, and
`--threshold` / `--percentile` continue to operate in composite-score units.

Add one of `--threshold VALUE` or `--percentile VALUE` when the export should
include a composite decision for each respondent. A fixed threshold is inclusive
(`score >= threshold`); a sample-percentile threshold is strict
(`score > resolved cutoff`) so ties at the percentile are not flagged. The two
options are mutually exclusive. Text, JSON, and NPZ record the resolved cutoff
and whether it was fixed or sample-derived, while CSV adds a `composite_flag`
column. Without either option, output remains score-only.
