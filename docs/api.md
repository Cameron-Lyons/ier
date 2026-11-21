# API Reference

Every name below is importable from the package root, for example
`from ier import screen, irv_flag`.

## Screening and composites

::: ier.screen.screen

::: ier.screen_scores

::: ier.composite.composite

::: ier.composite_flag

::: ier.composite_probability

::: ier.composite_scores

::: ier.composite_scores_summary

::: ier.composite_summary

::: ier.IndexOptions

::: ier.index_catalog

## Result tables and agreement

::: ier.screen_table

::: ier.composite_table

::: ier.index_agreement

## Consistency indices

::: ier.irv.irv

::: ier.irv_flag

::: ier.evenodd.evenodd

::: ier.evenodd_flag

::: ier.psychsyn.psychsyn

::: ier.psychsyn_flag

::: ier.psychant

::: ier.psychant_flag

::: ier.psychsyn_critval

::: ier.psychsyn_summary

::: ier.person_total.person_total

::: ier.person_total_flag

::: ier.individual_reliability

::: ier.individual_reliability_flag

::: ier.semantic_syn

::: ier.semantic_syn_flag

::: ier.semantic_ant

::: ier.semantic_ant_flag

::: ier.mad.mad

::: ier.mad_flag

::: ier.reverse_score

## Response patterns

::: ier.longstring_scores

::: ier.longstring_flag

::: ier.longstring.longstring

::: ier.longstring_pattern

::: ier.longstring_pattern_flag

::: ier.markov.markov

::: ier.markov_flag

::: ier.markov_summary

::: ier.autocorrelation.autocorrelation

::: ier.autocorrelation_flag

::: ier.onset.onset

::: ier.onset_flag

::: ier.response_pattern

## Response styles

::: ier.u3_poly.u3_poly

::: ier.u3_poly_flag

::: ier.midpoint_responding

::: ier.midpoint_responding_flag

::: ier.acquiescence.acquiescence

::: ier.acquiescence_flag

## Outliers and person fit

::: ier.mahad.mahad

::: ier.mahad_summary

::: ier.mahad_qqplot

::: ier.guttman.guttman

::: ier.guttman_flag

::: ier.lz.lz

::: ier.lz_flag

::: ier.gpoly

::: ier.gpoly_flag

::: ier.u3poly

::: ier.u3poly_flag

::: ier.ht

::: ier.ht_flag

## Omissions and attention checks

::: ier.missing_rate

::: ier.missing_rate_flag

::: ier.infrequency.infrequency

::: ier.infrequency_flag

## Response times

::: ier.response_time.response_time

::: ier.response_time_flag

::: ier.response_time_consistency

::: ier.response_time_mixture

::: ier.response_time_effort

::: ier.response_time_effort_flag

::: ier.response_time_score_flags

## Persisting results

`save_screen_archive()` and `load_screen_archive()` persist a complete
`ScreenResult`: scores, per-index flags, recorded thresholds and their sources,
percentile settings, counts, consensus decisions, summaries, and soft failures.
They share the schema written by `ier screen --format npz`, so CLI archives load
the same way.

```python
from ier import load_screen_archive, plot_flagged_heatmap, save_screen_archive, screen

result = screen(data, percentile=95, min_flags=2)
save_screen_archive("screening.npz", result, respondent_ids=ids)

saved = load_screen_archive("screening.npz")
assert saved["result"]["consensus_flags"].tolist() == result["consensus_flags"].tolist()
figure = plot_flagged_heatmap(saved["result"])
```

The writer validates every decision before opening the destination and replaces
it atomically; the loader recomputes each flag from its score and recorded
cutoff, and checks counts, consensus decisions, and summary counts, flag rates,
minima, and maxima. Summary means and standard deviations are rebuilt from the
restored scores. Rebuilding
decisions with `screen_scores(thresholds=saved["result"]["thresholds"])` is not
equivalent, because explicit thresholds include ties while percentile cutoffs
exclude them.

All archive writers accept `compressed=True` and an optional
`compression_level` from 1 (fastest, the default) to 9 (smallest). Any level
loads through the same functions.

::: ier.save_screen_archive

::: ier.load_screen_archive

::: ier.save_score_archive

::: ier.load_score_archive

::: ier.save_response_time_archive

::: ier.load_response_time_archive

## Plotting

::: ier.plot_distributions

::: ier.plot_flag_counts

::: ier.plot_flagged_heatmap

::: ier.plot_composite

::: ier.plot_index_agreement

## Result types

::: ier.ScreenResult

::: ier.ScreenIndexSummary

::: ier.CompositeSummary

::: ier.ScoreArchive

::: ier.ScreenArchive

::: ier.ResponseTimeArchive

::: ier.IndexMetadata

::: ier.IndexCatalog

## Type aliases

::: ier.MatrixLike

::: ier.FloatArray

::: ier.BoolArray

::: ier.IntArray

::: ier.IndexScoreMap

::: ier.IndexFlagMap

::: ier.IndexErrorMap

::: ier.IndexThresholdMap

::: ier.IndexPercentileMap

::: ier.IndexThresholdSource

::: ier.IndexThresholdSourceMap

::: ier.ScoreArchiveResultType

::: ier.AgreementKind

::: ier.CombineMethod

::: ier.CompositeMethod

::: ier.EvenOddMethod

::: ier.FlagDirection

::: ier.FlagMode

::: ier.InfrequencyMissingPolicy

::: ier.ItemCorrelationMode

::: ier.ResponseTimeFlagDirection

::: ier.ResponseTimeMetric

## Package version

::: ier.__version__

## Return-shape controls

Boolean options that select a documented return type are strict Python
Booleans. This includes `diag`, `flag`, `avg`, `return_lags`,
`return_item_flags`, and `return_diagnostics`. Passing `0` or `1`, a string,
an array, or a NumPy Boolean scalar raises `ValueError` before the input matrix
is converted or scored. This keeps runtime return types aligned with the
published overloads.
