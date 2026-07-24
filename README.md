# IER

Python package for detecting **Insufficient Effort Responding (IER)** / careless
responding in survey data.

For a comprehensive methods review, see
[Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub).

## Features

- Multiple detection families: consistency, response patterns, response styles, outliers, response times, attention checks
- Workflow APIs: `screen()` and `composite()` configured via `IndexOptions`
- CLI: `ier screen data.csv` / `ier composite data.csv` (text / JSON / CSV)
- NumPy-first inputs (lists, arrays, array-compatible DataFrames)
- Soft per-index errors during screening and composite scoring
- Full type annotations (`py.typed`)

## Installation

```bash
pip install insufficient-effort
```

Optional extras:

```bash
pip install "insufficient-effort[full]"   # SciPy helpers
pip install "insufficient-effort[plot]"   # matplotlib visualizations
```

Base install is NumPy-only. Default `screen()` / `composite()` work without SciPy.
SciPy is required for `mahad(..., flag=True, method="chi2")` and
`response_time_mixture()`.

## Quick Start

```python
import numpy as np
from ier import IndexOptions, composite, irv, screen

data = np.array([
    [1, 2, 3, 4, 5, 4],
    [2, 3, 4, 3, 2, 1],
    [3, 3, 3, 3, 3, 3],  # straightlining
    [1, 5, 1, 5, 1, 5],  # alternating
], dtype=float)

print("IRV:", irv(data))

result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
print("Indices:", result["indices_used"])
print("Flag counts:", result["flag_counts"])

scores = composite(data, indices=["irv", "longstring", "person_total", "markov"])
print("Composite:", scores)
```

## CLI

```bash
ier screen data.csv --scale-min 1 --scale-max 5
ier screen data.csv --format json --output screen.json
ier composite data.csv --indices irv longstring
ier composite data.csv --format csv --output scores.csv
ier --version
```

## Documentation

Full docs live in [`docs/`](docs/) (MkDocs):

- [Getting started](docs/getting-started.md)
- [Index catalog](docs/indices.md)
- [Screening workflow](docs/workflows/screening.md)
- [Composite guidance](docs/workflows/composite.md)
- [Threshold guidance](docs/thresholds.md)
- [R package notes](docs/r-comparison.md)
- [Changelog](CHANGELOG.md)

Build locally:

```bash
uv sync --extra docs
uv run mkdocs serve
```

Examples:

```bash
uv run python examples/basic_screening.py
uv run python examples/composite_scoring.py
uv run python examples/careless_responding_walkthrough.py
```

Benchmark:

```bash
uv run python benchmarks/bench_screen.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{ier2026,
  title={IER: Python package for detecting Insufficient Effort Responding},
  author={Lyons, Cameron},
  year={2026},
  url={https://github.com/Cameron-Lyons/ier}
}
```

## References

- Curran, P. G. (2016). Methods for the detection of carelessly invalid responses in survey data. *Journal of Experimental Social Psychology*, 66, 4-19.
- Dunn, A. M., Heggestad, E. D., Shanock, L. R., & Theilgard, N. (2018). Intra-individual response variability as an indicator of insufficient effort responding. *Journal of Business and Psychology*, 33(1), 105-121.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data. *Psychological Methods*, 17(3), 437-455.
