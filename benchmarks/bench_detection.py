"""Simulate attentive vs careless responders and report screen() detection rates.

This is a labeled synthetic study for applied credibility checks — not a claim
of real-world sensitivity/specificity. Use it to sanity-check that common
careless patterns are enriched in the flagged set.

Usage:
    uv run python benchmarks/bench_detection.py
    uv run python benchmarks/bench_detection.py --attentive 400 --careless 100 --items 40
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from ier import IndexOptions, screen

CarelessFactory = Callable[[np.random.Generator, int, int], np.ndarray]


def _attentive(rng: np.random.Generator, n: int, items: int) -> np.ndarray:
    """Mildly correlated Likert-like responses (not pure uniform noise)."""
    latent = rng.normal(size=(n, 1))
    noise = rng.normal(scale=0.85, size=(n, items))
    continuous = 3.0 + 0.7 * latent + noise
    return np.clip(np.rint(continuous), 1, 5).astype(float)


def _straightline(rng: np.random.Generator, n: int, items: int) -> np.ndarray:
    values = rng.integers(1, 6, size=n)
    return np.repeat(values.astype(float)[:, None], items, axis=1)


def _alternating(rng: np.random.Generator, n: int, items: int) -> np.ndarray:
    low = rng.integers(1, 3, size=n)
    high = rng.integers(4, 6, size=n)
    pattern = np.empty((n, items), dtype=float)
    pattern[:, 0::2] = low[:, None]
    pattern[:, 1::2] = high[:, None]
    return pattern


def _random_uniform(rng: np.random.Generator, n: int, items: int) -> np.ndarray:
    return rng.integers(1, 6, size=(n, items)).astype(float)


CARELESS_PATTERNS: dict[str, CarelessFactory] = {
    "straightline": _straightline,
    "alternating": _alternating,
    "random": _random_uniform,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attentive", type=int, default=400)
    parser.add_argument("--careless", type=int, default=100)
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pattern",
        choices=[*CARELESS_PATTERNS, "mixed"],
        default="mixed",
        help="Careless response pattern to inject",
    )
    parser.add_argument(
        "--flag-threshold",
        type=int,
        default=2,
        help="Respondents flagged by at least this many indices count as detected",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    attentive = _attentive(rng, args.attentive, args.items)

    if args.pattern == "mixed":
        per = args.careless // 3
        rem = args.careless - 2 * per
        careless = np.vstack(
            [
                _straightline(rng, per, args.items),
                _alternating(rng, per, args.items),
                _random_uniform(rng, rem, args.items),
            ]
        )
    else:
        careless = CARELESS_PATTERNS[args.pattern](rng, args.careless, args.items)

    data = np.vstack([attentive, careless])
    labels = np.array([0] * args.attentive + [1] * args.careless)

    result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
    n_flags = result["flag_counts"]
    detected = n_flags >= args.flag_threshold

    tp = int(np.sum(detected & (labels == 1)))
    fp = int(np.sum(detected & (labels == 0)))
    fn = int(np.sum(~detected & (labels == 1)))
    tn = int(np.sum(~detected & (labels == 0)))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"shape={data.shape} pattern={args.pattern} flag_threshold={args.flag_threshold}")
    print(f"indices={result['indices_used']}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"sensitivity={sens:.3f} specificity={spec:.3f}")
    print("per-index careless flag rate:")
    for name in result["indices_used"]:
        flags = result["flags"][name]
        rate = float(np.mean(flags[labels == 1]))
        print(f"  {name}: {rate:.3f}")


if __name__ == "__main__":
    main()
