# API Reference

::: ier
    options:
      members:
        - AgreementKind
        - EvenOddMethod
        - IndexOptions
        - ItemCorrelationMode
        - ResponseTimeArchive
        - ResponseTimeFlagDirection
        - ResponseTimeMetric
        - ScoreArchive
        - ScreenArchive
        - __version__
        - index_catalog
        - load_score_archive
        - load_response_time_archive
        - load_screen_archive
        - save_response_time_archive
        - save_score_archive
        - save_screen_archive
        - screen
        - screen_scores
        - composite
        - composite_flag
        - composite_probability
        - composite_scores
        - composite_scores_summary
        - composite_summary
        - screen_table
        - composite_table
        - index_agreement
        - irv
        - longstring_scores
        - longstring
        - longstring_pattern
        - mahad
        - mahad_summary
        - psychsyn
        - psychant
        - psychsyn_flag
        - psychant_flag
        - psychsyn_critval
        - psychsyn_summary
        - person_total
        - person_total_flag
        - markov
        - autocorrelation
        - autocorrelation_flag
        - missing_rate
        - missing_rate_flag
        - u3_poly
        - u3_poly_flag
        - midpoint_responding
        - midpoint_responding_flag
        - acquiescence
        - guttman
        - gpoly
        - gpoly_flag
        - u3poly
        - u3poly_flag
        - ht
        - ht_flag
        - individual_reliability
        - onset
        - evenodd
        - reverse_score
        - mad
        - mad_flag
        - lz
        - semantic_syn
        - semantic_syn_flag
        - semantic_ant
        - semantic_ant_flag
        - infrequency
        - infrequency_flag
        - response_time
        - response_time_consistency
        - response_time_effort
        - response_time_effort_flag
        - response_time_flag
        - response_time_mixture
        - response_time_score_flags
        - plot_distributions
        - plot_flag_counts
        - plot_flagged_heatmap
        - plot_composite
        - plot_index_agreement

## Persisting screening results

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
