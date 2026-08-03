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
ranks = composite_probability(data, indices=["irv", "longstring"], options=opts)
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
3. Report which indices and combination method you used.
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

By default, one invalid configured index is returned in diagnostics while other
indices continue. Pass `strict=True` to `composite()`, `composite_flag()`,
`composite_summary()`, or `composite_probability()` when every selected index
must succeed.

## CLI

Blank fields in comma-, tab-, or semicolon-delimited input are loaded as missing
values (`NaN`) and follow each index's documented missing-data behavior.

```bash
ier composite data.csv --indices irv longstring --method mean
ier composite data.csv --indices irv mad --strict
ier composite data.csv --format json --output composite.json
ier composite data.csv --format csv --evenodd-factors 5,5 --indices irv evenodd
ier composite data.npy --indices irv longstring --format json
ier composite data.npy --indices irv longstring --format npz --output composite.npz
```

Uncompressed `.npy` matrices are memory-mapped read-only. They must be non-empty,
two-dimensional, and real numeric; header selection and delimiter options do not
apply to this binary format.

CSV results are written row by row to plain files, gzip files, or standard output
without retaining the complete export in memory.

JSON output is standards-compliant: unavailable or non-finite scores are encoded
as `null`. CSV output represents those scores as empty cells so numeric columns
remain compatible with spreadsheet and statistics tools.
NPZ output preserves the numeric score vector, combination method, and optional
respondent IDs using the [versioned archive schema](../cli-output.md).
