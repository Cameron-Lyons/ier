# Notes Relative to R Packages

Several R packages implement overlapping careless-responding indices. IER aims
to provide a NumPy-first Python API with typed orchestration via `screen()` and
`composite()`.

## Related R packages

| Concept | Common R reference | IER function |
|---------|--------------------|--------------|
| Intra-individual response variability | `careless::irv` | `irv()` |
| Longest identical string | `careless::longstring` | `longstring_scores()` / registry `"longstring"` |
| Psychometric synonyms / antonyms | `careless::psychsyn` / `psychant` | `psychsyn()` / `psychant()` |
| Mahalanobis distance | `careless::mahad` | `mahad()` |
| Even–odd consistency | `careless::evenodd` | `evenodd()` |
| Person-fit / Guttman errors | PerFit / custom | `guttman()`, `lz()` |

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

`tests/test_golden_parity.py` locks hand-verified `irv` and `longstring` values
on a small matrix so regressions are caught in CI. Use that file as a template
when adding parity checks for additional indices.

## Suggested validation workflow

If you need parity with an existing R pipeline:

1. Export the same respondent × item matrix from both environments.
2. Compare one index at a time on complete cases.
3. Match options explicitly (critical values, normalization, seeds).
4. Treat residual differences as implementation notes in your methods section.

## What IER adds for Python users

- Unified `screen()` / `composite()` registry with soft per-index errors
- Shared `IndexOptions` config object (preferred over long kwargs lists)
- Strict typing (`py.typed`) and CI across Python 3.11–3.14
- Optional SciPy / matplotlib extras without forcing them on base installs
- CLI: `ier screen data.csv` / `ier composite data.csv`
- Explicit documentation that composite logistic scores are uncalibrated
