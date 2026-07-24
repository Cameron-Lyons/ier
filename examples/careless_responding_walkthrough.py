"""
# Careless responding walkthrough

This notebook-style script mirrors a typical analysis path:

1. Simulate attentive vs careless respondents
2. Screen with multiple indices
3. Inspect multi-index agreement
4. Form a composite ranking

Run with:

```bash
uv run python examples/careless_responding_walkthrough.py
```
"""

from __future__ import annotations

import numpy as np

from ier import IndexOptions, composite, screen


def simulate(n_attentive: int = 80, n_careless: int = 20, n_items: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    attentive = rng.integers(1, 6, size=(n_attentive, n_items)).astype(float)
    careless = np.full((n_careless, n_items), 3.0)
    # Mix in a few alternating careless rows
    for i in range(0, n_careless, 4):
        careless[i, :] = np.tile([1.0, 5.0], n_items // 2)
    labels = np.array([0] * n_attentive + [1] * n_careless)
    data = np.vstack([attentive, careless])
    order = rng.permutation(data.shape[0])
    return data[order], labels[order]


def main() -> None:
    data, labels = simulate()
    result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
    agreement = result["flag_counts"] >= 2
    hit_rate = float(np.mean(agreement[labels == 1]))
    false_alarm = float(np.mean(agreement[labels == 0]))

    ranking = composite(data, indices=["irv", "longstring", "person_total", "markov"])
    top = np.argsort(ranking)[::-1][:10]
    precision_at_10 = float(np.mean(labels[top]))

    print(f"respondents={data.shape[0]} items={data.shape[1]}")
    print(f"indices_used={result['indices_used']}")
    print(f"multi-index (>=2 flags) hit_rate={hit_rate:.2f} false_alarm={false_alarm:.2f}")
    print(f"composite precision@10={precision_at_10:.2f}")


if __name__ == "__main__":
    main()
