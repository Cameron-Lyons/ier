# Architecture

This note explains how IER orchestrates indices, handles missing data, and
flags respondents. It is aimed at contributors and methods-curious users.

## Layers

```text
CLI / examples
      │
screen() / composite()     ← public orchestration
screen_scores() / composite_scores() / response_time_score_flags()
                                    ← reusable decision layer
      │
IndexOptions + registry    ← shared config + index catalog
      │
per-index modules          ← irv, longstring, mahad, …
      │
_validation / _flagging    ← shared input checks and threshold helpers
```

- **NumPy-first core.** Base installs depend only on NumPy. Statistical
  routines, including chi-square flagging, response-time mixtures, and IRT
  theta estimation, are implemented locally. Matplotlib is only required for
  plotting helpers.
- **Registry.** `src/ier/_registry.py` maps string names to scorers, default
  screen/composite membership, flag direction, and required `IndexOptions`
  fields (e.g. `evenodd_factors`).
- **Soft per-index errors.** `score_registered_indices()` catches validation /
  runtime failures per index and returns them in an `errors` dict instead of
  aborting the whole screen. Public orchestration APIs also accept `strict=True`
  to raise a contextual error as soon as a selected index fails. Composite
  callers still raise if *no* index succeeds under the default soft policy.
- **Opt-in parallelism.** Registry orchestration evaluates indices sequentially
  by default. `workers>1` uses a lazily imported standard-library thread pool,
  then records scores and failures in selection order. This keeps default import
  cost and resource use stable while allowing NumPy-heavy scorers to overlap.
- **Bounded reductions.** Screening flag counts, valid-score counts, and summary
  coverage share one index-at-a-time pass; composite mean, sum, and maximum values
  use the same bounded approach. Orchestration retains its documented per-index
  result vectors but does not construct another complete respondent-by-index
  matrix for final reductions.
- **Reusable decisions.** `screen_scores()` reapplies direction-aware flagging,
  completeness, consensus, and summaries to retained registered score vectors.
  `composite_scores()` reuses raw composite-enabled vectors for alternative
  weights, standardization, completeness, and reductions.
  `response_time_score_flags()` reapplies low- or high-tail cutoffs to retained
  direct timing scores or mixture probabilities. All three support sensitivity
  analysis without rerunning scorers.
- **Archive-backed decisions.** Screening reflagging, composite recombination,
  and response-time reflagging load validated retained vectors before applying
  new decisions. They share cutoff, combination, probability, and output paths
  with their original scoring commands. NPZ output can therefore atomically
  replace a reusable source without retaining the raw input matrix or adding a
  runtime dependency; composite replacement explicitly requires component
  retention.

## Command-line boundaries

The command-line path is split by responsibility:

- `cli.py` defines arguments, converts index options, coordinates commands, and
  performs destination checks that must precede potentially expensive input.
- `_cli_results.py` derives optional cutoffs and probabilities, then routes
  catalog, archive metadata, screen, composite, and response-time results to the
  selected format.
- `_cli_input.py` owns forward-only delimited input, gzip handling, named-column
  selection, memory-mapped NumPy input, and the shared safe numeric-vector
  loader for external model parameters.
- `_cli_composite.py` validates shared respondent alignment and flag metadata
  contracts for every composite serializer.
- `_cli_output.py` renders text plus bounded strict JSON and CSV results.
- `_cli_npz.py` assembles complete command result payloads and delegates the
  low-level typed, pickle-free NumPy writer.
- `archive.py` owns atomic same-directory staging, the shared stream writer, and
  public validated save/load boundaries for reusable registered score vectors
  and response-time results, plus a dedicated mixture-calibration model schema.
  Its generic result loader reads the declared result type once and dispatches
  to the same complete specialized validators; model archives use their dedicated
  loader because they contain parameters rather than respondent results.

Screen and composite commands carry the registry's ordered soft-failure map
through text, JSON, and NPZ serializers and mirror failures to standard error
for every format. Score computation still runs once, and CSV remains a compact
respondent table.

Detailed composite output is explicit through `--include-components`. The CLI
reuses `composite_summary()` so scoring still runs once, JSON wraps each
component in the bounded array writer, CSV emits one row at a time, and NPZ
writes separate typed members without stacking another respondent-by-index
matrix. The default aggregate-only path does not allocate or serialize these
details.

Composite standardization is resolved before either aggregate-only or detailed
scoring begins. The same boolean is passed to the public scoring API and written
to text, JSON, and NPZ metadata, avoiding a second calculation or inference from
the resulting values. CSV intentionally remains a respondent-only table.

Optional composite flagging runs after aggregate scoring and reuses that score
vector. Cutoff resolution and boolean comparison use the same shared helpers as
the public flagging APIs. JSON writes the flag vector in bounded chunks, CSV
streams it row by row, and NPZ stores it as a boolean member; no component is
rescored and the score-only path allocates no flag vector.

Keeping parsing, matrix construction, serialization, and orchestration separate
makes format-specific changes independently testable without adding runtime
packages or coupling them to statistical index implementations.

## IndexOptions

Orchestration APIs accept configuration **only** via
`options=IndexOptions(...)`. That keeps `screen()` / `composite()` signatures
stable as indices grow. Per-function kwargs remain available when calling an
index module directly (e.g. `mahad(x, method="iqr")`).

## Flagging policy

`screen()` turns scores into boolean flags using registry metadata:

| Mode | Typical indices | Rule |
|------|-----------------|------|
| Low / high percentile | IRV, longstring, mahad, … | Extreme tail relative to the sample |
| Fixed low / high threshold | Any percentile-mode index | At or beyond a validated cutoff |
| Presence | `onset` | Any detected changepoint is flagged |

Percentile defaults are **sample-relative heuristics**, not calibrated
diagnostic cutoffs. `screen(thresholds=...)` accepts fixed cutoffs when a survey
or validation study provides them, while `screen(percentiles=...)` tunes
sample-relative tail sensitivity by index. Results retain the actual cutoff,
its fixed/percentile/presence source, and the requested tail percentile. See
[Threshold Guidance](thresholds.md).

## Missing data

Most indices honor `IndexOptions.na_rm` (default `True`) and document
row-wise vs listwise behavior in their docstrings. Policies are intentionally
not identical across indices: IRV can use `nanstd`, Mahalanobis may drop
incomplete rows, Markov may require complete sequences depending on `na_rm`.
When comparing to R packages, align NA policy first.

The opt-in `missing_rate` index separates observed missingness from survey
applicability. A fixed `missing_item_indices` subset works through registry and
command-line scoring. Respondent-by-item applicability masks work through both
Python and command-line workflows; the latter validates file values in bounded
row batches and preserves read-only memory mapping for Boolean NumPy inputs.
False cells do not contribute to either the missing count or denominator. A
zero denominator yields `NaN`, which the shared flagging policy leaves unflagged.

Attention-check scoring keeps its legacy missing-as-pass behavior by default but
also supports missing-as-failure, available-case omission, and strict propagation.
The policy is carried through `IndexOptions`; unavailable scores reuse the same
flagging and composite-coverage rules as every other registry index.

Screening counts available scores alongside flags without stacking either set of
vectors. An optional `min_valid_indices` rule marks rows with insufficient score
coverage as ineligible for consensus while retaining the individual scores,
flags, counts, and eligibility decision for audit.

## Composite scores

`composite()` z-combines selected indices with direction multipliers so that
higher composite values mean more evidence of careless responding.
Optional positive weights are applied after direction correction and
standardization. Weighted means renormalize over available scores per
respondent, so an unavailable index does not silently dilute the remaining
evidence.
Direction multipliers and weights are applied one vector at a time during the
reduction. Raw component vectors remain available for detailed summaries, while
ordinary and reusable composite paths avoid retaining a second mapping of
direction-corrected arrays.
An optional minimum valid-index rule masks under-supported respondent scores
after reduction. Equal-weight means reuse their existing denominator counts, so
the rule needs no additional respondent-sized workspace on that path; other
methods allocate one integer count vector only when the rule is enabled.
`composite_probability()` applies a logistic transform for convenience — it is
**not** a calibrated probability of carelessness. Do not treat it as a
posterior or diagnostic probability without your own validation study. Its
shared piecewise NumPy kernel evaluates positive and negative values separately,
avoiding overflow while preserving finite-tail precision and exact infinite
endpoints. The lz theta and likelihood paths reuse the same kernel for both
complete batches and missing-data rows.
Complete-data LZ fallback calibration estimates every point-biserial item
discrimination through one contraction with the shared centered total-score
vector. Missing-response fallback calibration instead uses bounded column
blocks, while masked ability and likelihood calculations run across bounded
respondent batches. Complete and masked theta solvers compact their workspaces
as respondents converge, so later safeguarded Newton iterations evaluate only
the rows still solving. Both missing-removal and strict policies retain their
established unavailable-value, constant-response, and extreme-parameter
semantics without scalar per-item or per-respondent dispatch. Shared index
configuration forwards optional calibrated difficulty, discrimination, and
ability arrays to this same kernel. Omitted arrays retain fallback estimation,
while supplied arrays avoid recomputing their corresponding calibration stage.
The CLI computes this transform from the final aggregate vector only when
`--include-probability` is requested. JSON and CSV then serialize it
forward-only, while NPZ stores one additional typed vector; index scoring is not
repeated and default output schemas remain unchanged.

## Response-time helpers

Timing matrices are a different data modality than item responses. Helpers in
`response_time.py` are public but **intentionally outside** the screen /
composite registry so they are not mixed into item-response pipelines by
accident. Gaussian-mixture fitting reuses responsibility and scratch buffers
throughout EM. Its expectation step follows a fast probability-space path for
ordinary observations and normalizes only underflowed rows in log space.
`fit_response_time_mixture()` exposes the fitted weights, means, and variances as
an immutable calibration, while `response_time_mixture_scores()` applies those
parameters to later cohorts with one expectation step. Returned posterior vectors
own their one-dimensional storage rather than retaining the responsibility matrix.
The model save/load pair streams those small parameter vectors through the shared
atomic NPZ writer and reconstructs a fully validated read-only calibration, so
cross-process reuse retains the same numerical and safety contracts.
Retained summary vectors and mixture probabilities pass through the same shared
single-vector validation and threshold boundary, so cutoff sensitivity analysis
does not recalculate row summaries or refit EM. The archive reflag CLI loads that
validated vector directly, then shares cutoff resolution and all four output
serializers with the original response-time command.

Markov transition entropy counts categorical pairs in bounded row batches for
up to 64 observed states. Higher-cardinality inputs lexicographically sort origin states
and transition pairs within 16,384-transition-cell batches, deriving observed
run counts without a dense global state-square allocation or respondent-wise
dispatch. Both paths evaluate the equivalent count form of conditional entropy.
Missing-response rows are grouped by retained length and compressed in bounded
batches, preserving the post-removal response order while reusing the same dense
or sparse kernel.

Numeric longest-run and repeating-pattern indices use the shared retained-length
compression iterator when responses are missing. Each bounded group preserves
post-removal item order and reuses the complete-response vectorized kernel,
avoiding respondent-wise array construction. Markov and carelessness-onset
scoring share the same grouping primitive with index-specific minimum lengths
and workspace limits.

Even–odd consistency reduces factor correlations directly into respondent-level
sums and valid-factor counts. Factors with exactly two paired observations use
the closed-form product of response-difference signs, avoiding centered matrices;
larger factors retain centered contraction reductions. Peak allocation therefore
does not grow with the factor count.
Psychometric synonym and antonym scoring reuse the same row-correlation kernel
in bounded respondent batches and report each respondent's actual available-pair
count. Missing-response discovery accumulates pairwise-complete counts, sums,
squares, and products in bounded respondent and strict-lower-triangle item
tiles. Per-column finite origins prevent raw-moment cancellation without a
matrix-sized centered copy. Complete matrices retain the faster normalized-
column kernel. Both paths avoid a complete item-by-item correlation matrix, and
the critical-value catalog shares the same dispatch. Seeded retries only process
undefined rows that retain at least three usable pairs.
Predefined semantic pairs and MAD item pairs share a bounded absolute-difference
reducer. Pair selection, optional reverse scoring, and missing-aware means stay
within the common element budget instead of materializing complete pair matrices.
Mahalanobis scoring uses the same row budget for both centered covariance
accumulation and quadratic-form evaluation. Only the item-by-item covariance and
pseudo-inverse remain resident outside each block, so temporary allocation does
not grow with the respondent count. Theoretical Q-Q coordinates solve
chi-square quantiles through bounded vectorized regularized-gamma and safeguarded
Newton batches, with the independent scalar solver retained as a fallback.
Guttman scoring likewise batches item means, difficulty-ordered selection,
valid-response counts, and error accumulation by respondent. It compares the
number of full cumulative-category passes with the triangular item-pair count,
using the cheaper exact counter for each scale width. Both paths retain the same
row bound, so narrow categorical scales keep their cumulative-count advantage
while wider scales avoid redundant passes and larger workspaces.
Split-half individual reliability generates the established seeded item splits
once, then reuses each bounded respondent block across them. Row correlations
use raw sums, squares, and cross-products without allocating complete centered
half matrices. A scale-aware cancellation check sends only numerically risky
row pairs through stable centering; pairwise-complete missing semantics and
per-respondent valid-split counts remain unchanged.
Complete-response onset detection derives stable sliding-window variability
from rolling means and bounded deviation buffers. Its changepoint test retains
only prefix and candidate-position workspaces instead of complete centered and
test-statistic matrices. Missing-response blocks compress rows into equal
retained-length groups and reuse the same bounded complete-response kernel.
Person–total correlation calculates item-profile means and respondent
correlations in bounded batches as well. The shared kernel accepts the index's
undefined-correlation policy, so constant person or item profiles remain
unavailable rather than being assigned a synthetic score.
IRV, acquiescence, and response-style summaries share bounded row mean and
population-standard-deviation reductions. Missing-aware blocks track valid
counts directly, so an entirely unavailable row returns an unavailable score
without constructing a complete boolean or centered workspace for the input.
Response-time summaries use the same reductions. Missing-aware medians group
respondents by retained timing count and partition bounded NaN-free matrices,
avoiding the general missing-value reducer while preserving row order and
all-missing results. Median-based mixture preprocessing therefore does not
duplicate the complete timing matrix before fitting its respondent-level model.

## Optional dependencies

All statistical functionality is available in the NumPy-only base install.
Plotting remains optional and reports a centralized install hint from
`_optional_imports.py`: `pip install 'insufficient-effort[plot]'`.

## Parity and simulation

- Hand-locked regression fixtures live in `tests/test_golden_parity.py` and
  JSON under `tests/fixtures/parity/`.
- Detection-rate simulation: `benchmarks/bench_detection.py`.
- Screening throughput plus in-memory and archive-backed sensitivity reuse:
  `benchmarks/bench_screen.py`.
- Multi-factor even–odd throughput and memory: `benchmarks/bench_evenodd.py`.
- Psychometric synonym missing-data throughput and memory: `benchmarks/bench_psychsyn.py`.
- Predefined semantic/MAD pair throughput and memory: `benchmarks/bench_pair_differences.py`.
- Mahalanobis covariance, distance, and Q-Q throughput and memory:
  `benchmarks/bench_mahad.py`.
- Guttman error-scoring throughput and memory: `benchmarks/bench_guttman.py`.
- Split-half reliability throughput and memory: `benchmarks/bench_reliability.py`.
- Carelessness-onset throughput and memory: `benchmarks/bench_onset.py`.
- Person–total correlation throughput and memory: `benchmarks/bench_person_total.py`.
- Row-wise response reduction throughput and memory: `benchmarks/bench_row_reductions.py`.
- Longest-run and repeating-pattern throughput and memory: `benchmarks/bench_longstring.py`.
- Lz person-fit throughput and memory: `benchmarks/bench_lz.py`.
- Markov transition-entropy throughput and memory: `benchmarks/bench_markov.py`.
- Response-time mixture EM, reusable calibrated scoring, in-memory cutoff reuse,
  and archive-backed CLI cutoff reuse:
  `benchmarks/bench_response_time.py`.
- Screen/composite reduction memory: `benchmarks/bench_orchestration.py`.
- In-memory and archive-backed composite sensitivity analysis:
  `benchmarks/bench_composite.py`.
- Validated and atomic score and response-time archive loading, saving, and
  cutoff-provenance verification:
  `benchmarks/bench_archive.py`.
- Shared fixed and percentile flagging throughput and memory: `benchmarks/bench_flagging.py`.
- CLI JSON, CSV, and NPZ serialization: `benchmarks/bench_cli_output.py`.
