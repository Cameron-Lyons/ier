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
ranks = composite_probability(data, indices=["irv", "longstring"], options=opts)

# Available on composite(), composite_flag(), composite_summary(), and
# composite_probability().
parallel_scores = composite(data, indices=["irv", "longstring"], workers=4)
```

## Important caveats

!!! warning "Not a calibrated probability"
    `composite_probability()` applies a logistic transform to standardized
    composite scores. Values lie in `[0, 1]` but are **not** validated
    probabilities of IER unless you calibrate against labeled data from a
    comparable survey.

Practical guidance:

1. Prefer multi-index agreement over any single cutoff.
2. Review flagged cases substantively (open text, completion time, attention checks).
3. Report which indices, combination method, and weights you used.
4. Use `method="best_subset"` when you want the Curran/Meade-Craig style mix of
   consistency, pattern, and (optionally) MAD signals.

## Allowed indices

Composite-enabled indices include:

`irv`, `longstring`, `longstring_pattern`, `mahad`, `psychsyn`, `psychant`,
`person_total`, `markov`, `guttman`, `individual_reliability`, `evenodd`, `mad`,
`lz`, `semantic_syn`, `semantic_ant`, `infrequency`, `missing_rate`

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

By default, one invalid configured index is returned in diagnostics while other
indices continue. Pass `strict=True` to `composite()`, `composite_flag()`,
`composite_summary()`, or `composite_probability()` when every selected index
must succeed.

All composite helpers accept `workers`. The default of `1` evaluates indices
sequentially; larger values retain index and diagnostic order while trading
higher temporary memory for potential throughput improvements.

## CLI

Blank fields in comma-, tab-, or semicolon-delimited input are loaded as missing
values (`NaN`) and follow each index's documented missing-data behavior.

```bash
ier composite data.csv --indices irv longstring --method mean
ier composite data.csv --indices irv longstring --weight irv=2 --weight longstring=0.5
ier composite data.csv --indices irv mad --strict
ier composite data.csv --indices irv longstring --workers 4
ier composite data.csv --format json --output composite.json
ier composite data.csv --format csv --evenodd-factors 5,5 --indices irv evenodd
ier composite data.npy --indices irv longstring --format json
ier composite data.npy --indices irv longstring --format npz --output composite.npz
```

Uncompressed `.npy` matrices are memory-mapped read-only. They must be non-empty,
two-dimensional, and real numeric; header selection and delimiter options do not
apply to this binary format.

CSV rows and JSON respondent arrays are written forward-only to plain files,
gzip files, or standard output without retaining the complete export in memory.

JSON output is standards-compliant: unavailable or non-finite scores are encoded
as `null`. CSV output represents those scores as empty cells so numeric columns
remain compatible with spreadsheet and statistics tools.
NPZ output preserves the numeric score vector, combination method, and optional
respondent IDs. Text, JSON, and NPZ outputs also record explicitly supplied
weight overrides; unlisted selected indices use weight 1. See the
[versioned archive schema](../cli-output.md).
