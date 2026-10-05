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
| Even–odd consistency | `careless::evenodd` | `evenodd(method="halves")` (equals `-careless::evenodd`) |
| Person-fit / Guttman errors | PerFit / custom | `guttman()`, `lz()` |
| Polytomous Guttman errors | `PerFit::Gpoly` / `Gnormed.poly` | `gpoly(normalize=False)` / `gpoly()` |
| Polytomous U3 | `PerFit::U3poly` | `u3poly()` (not `u3_poly()`) |
| Transposed scalability | `PerFit::Ht` | `ht()` |
| Transition entropy | custom / Meade & Craig style | `markov()` |
| Carelessness onset | changepoint literature | `onset()` |
| Lagged autocorrelation | `responsePatterns::rp.acors` | `autocorrelation()` |

Exact function names, defaults, and NA handling differ across implementations.
Do **not** expect bit-identical scores without aligning:

- missing-data policy (`na_rm`)
- correlation critical values (`psychsyn_critval`)
- psychometric synonym/antonym item correlations: `careless::psychsyn` uses
  pairwise-complete correlations, matched by `item_correlations="pairwise"`
  (`IndexOptions.psychsyn_item_correlations`). IER's default `"complete"` mode
  treats an item with any missing response as undefined. Pairwise mode needs
  three shared respondents per item pair where R accepts two, and scores
  respondents with at least two answered pairs where careless requires three.
  A respondent whose answered pairs have zero within-pair variance scores 0.0
  in IER, while careless returns `NA` or, with `resample_na`, reshuffles the
  pair order. IER accepts `resample_na` and `random_seed` for compatibility, but
  they have no effect on scores
- Mahalanobis flagging method (`chi2` vs `iqr` vs `zscore`)
- whether scores are normalized (e.g., Guttman proportions)
- Guttman item order: IER orders items easiest first (descending item mean,
  ties by column) and counts pairs where the harder item scores higher, the
  error direction of PerFit's `G`. IER 1.12 and earlier counted the reverse
  (Guttman-consistent) pairs, so guttman scores from those releases differ
- random seeds for resampling methods (`individual_reliability`)
- even-odd orientation and algorithm: `evenodd(method="halves")` reproduces
  `careless::evenodd` with the sign flipped, so higher IER scores mean more
  consistent responding, and straight-lined profiles are `NaN` in both. The
  default `method="item_pairs"` is IER-specific and correlates items within each
  factor rather than half-scale means across factors
- IRV divisor (`ddof`); IER matches NumPy / typical R `sd` on a vector with
  population vs sample conventions checked explicitly in tests
- item-step person fit: `gpoly()`, `u3poly()`, and `ht()` follow PerFit's
  definitions of `Gpoly`, `Gnormed.poly`, `U3poly`, and `Ht`, including step
  popularity ties broken by item and then step (`rank(ties.method = "first")`).
  IER infers the category range from the observed responses unless `scale_min`
  or `ncat` is given, while PerFit's `Ncat` assumes categories coded from 0.
  Normalized `gpoly` and `u3poly` return `NaN` for perfect response vectors,
  where every step is failed or every step is passed, while PerFit's
  `Gnormed.poly` and `U3poly`, and `Gnormed` for dichotomous items, return 0.
  Incomplete respondents are `NaN` for normalized `gpoly`, `u3poly`, and `ht`
  (raw `gpoly` counts observed step pairs). PerFit's default
  `NA.method = "Pairwise"` does not impute: it scores incomplete respondents from
  their observed items, and only its `Hotdeck`, `NPModel`, and `PModel` methods
  impute missing responses. U3 weights clip steps passed by every or no
  respondent to half a response from the boundary. The `u3_poly()` index is the
  proportion of extreme responses, not PerFit's U3
- autocorrelation lag range: `autocorrelation()` defaults to `max_lag=10`, where
  `rp.acors` uses up to `J - 3` lags; pass `max_lag=None` for that range. IER
  lags start at `min_lag` (default 1) and each lag needs three response pairs.
  As in `rp.acors`, a lag scores 1 when either lagged window has zero variance,
  so identical responses score 1 at every usable lag and a straight-liner whose
  first or last response differs does too. Both return the smallest lag among
  tied maxima; IER counts correlations within floating-point rounding as tied

## Golden fixtures in this repo

`tests/test_golden_parity.py` locks hand-verified / regression values for:

`irv`, `longstring`, `longstring_pattern`, `mahad` (iqr), `psychsyn`,
`evenodd` (`method="item_pairs"`), `guttman`, `markov`, `person_total`,
`midpoint`, `lz`, and `onset`. `tests/test_evenodd_halves.py` checks
`evenodd(method="halves")` against a respondent-by-respondent port of careless's
`R/evenodd.R`, including missing halves, odd-sized and single-item factors.

`gpoly()`, `u3poly()`, and `ht()` have no R-generated fixtures. Their values
follow PerFit's published definitions and are validated in
`tests/test_person_fit.py` against independent brute-force oracles written from
the original papers: explicit enumeration of step pairs, maxima and U3 weight
extremes over every nested response pattern for small scales, and the naive
`O(N^2 J)` covariance form of Ht in exact integers. Dichotomous `gpoly()` also
reproduces `guttman()` counts divided by `r * (J - r)`.

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
