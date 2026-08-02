# Notes Relative to R Packages

Several R packages implement overlapping careless-responding indices. IER aims
to provide a NumPy-first Python API with typed orchestration via `screen()` and
`composite()`.

## Related R packages

| Concept | Common R reference | IER function |
|--------|--------------------|--------------|
| Intra-individual response variability | `careless::irv` | `irv()` |
| Longest identical string | `careless::longstring` | `longstring_scores()` / registry `"longstring"` |
| Psychometric synonyms / antonyms | `careless::psychsyn` / `psychant` | `psychsyn()` / `psychant()` |
| Mahalanobis distance | `careless::mahad` | `mahad()` |
| Even–odd consistency | `careless::evenodd` | `evenodd()` |
| Person-fit / Guttman errors | PerFit / custom | `guttman()`, `lz()` |
| Transition entropy | custom / Meade & Craig style | `markov()` |
| Carelessness onset | changepoint literature | `onset()` |

Exact function names, defaults, and NA handling differ across implementations.
Do **not** expect bit-identical scores without aligning:

- missing-data policy (`na_rm`)
- correlation critical values (`psychsyn_critval`)
- Mahalanobis flagging method (`chi2` vs `iqr` vs `zscore`)
- whether scores are normalized (e.g., Guttman proportions)
- random seeds for resampling methods (`individual_reliability`)
- IRV divisor (`ddof`); IER matches NumPy / typical R `sd` on a vector with
  population vs sample conventions checked explicitly in tests

## Golden fixtures in this repo

`tests/test_golden_parity.py` locks hand-verified / regression values for:

`irv`, `longstring`, `longstring_pattern`, `mahad` (iqr), `psychsyn`,
`evenodd`, `guttman`, `markov`, `person_total`, `midpoint`, `lz`, and `onset`.

JSON copies under `tests/fixtures/parity/` power a harness that loads the same
matrices and expected vectors. Treat JSON as the portable contract if you want
to regenerate expectations from R and drop in a replacement file.

## Regenerating fixtures from R

1. Export the fixture `matrix` from JSON to CSV.
2. Score the matching R function with aligned options (NA policy, critval, …).
3. Replace the `expected` vectors (use `null` for NaN).
4. Run `pytest tests/test_golden_parity.py -q`.

Example sketch for IRV / longstring:

```r
library(jsonlite)
library(careless)
fix <- fromJSON("tests/fixtures/parity/irv_longstring.json")
x <- as.matrix(fix$matrix)
```

## Suggested validation workflow

If you need parity with an existing R pipeline:

1. Export the same respondent × item matrix from both environments.
2. Compare one index at a time on complete cases.
3. Match options explicitly (critical values, normalization, seeds).
4. Treat residual differences as implementation notes in your methods section.

## What IER adds for Python users

- Unified `screen()` / `composite()` registry with soft per-index errors
- Shared `IndexOptions` config object (sole config surface for orchestration APIs)
- Strict typing (`py.typed`) and CI across Python 3.11–3.14
- Dependency-free statistical routines with an optional matplotlib plotting extra
- CLI: `ier screen data.csv` / `ier composite data.csv` with JSON/CSV export
- Explicit documentation that composite logistic scores are uncalibrated
- Response-time helpers kept out of band (timing matrices ≠ item responses)
- Architecture note covering registry, flagging, and NA policy
- Synthetic detection-rate benchmark (`benchmarks/bench_detection.py`)
