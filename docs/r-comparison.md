# Notes Relative to R Packages

Several R packages implement overlapping careless-responding indices. IER aims
to provide a NumPy-first Python API with typed orchestration via `screen()` and
`composite()`.

## Related R ecosystems

Common reference points in the literature and tooling include packages in the
careless-responding / person-fit space (names vary by CRAN status and forks),
often covering:

- longstring / IRV-style pattern indices
- psychometric synonyms / antonyms
- Mahalanobis distance outliers
- response-time summaries
- attention-check scoring

Exact function names, defaults, and NA handling differ across implementations.
Do **not** expect bit-identical scores without aligning:

- missing-data policy (`na_rm`)
- correlation critical values (`psychsyn_critval`)
- Mahalanobis flagging method (`chi2` vs `iqr` vs `zscore`)
- whether scores are normalized (e.g., Guttman proportions)
- random seeds for resampling methods (`individual_reliability`)

## Suggested validation workflow

If you need parity with an existing R pipeline:

1. Export the same respondent × item matrix from both environments.
2. Compare one index at a time on complete cases.
3. Match options explicitly (critical values, normalization, seeds).
4. Treat residual differences as implementation notes in your methods section.

## What IER adds for Python users

- Unified `screen()` / `composite()` registry with soft per-index errors
- Strict typing (`py.typed`) and CI across Python 3.11–3.14
- Optional SciPy / matplotlib extras without forcing them on base installs
- Explicit documentation that composite logistic scores are uncalibrated
