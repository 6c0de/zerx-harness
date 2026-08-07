"""How much of an ARC-AGI-3 leaderboard score is skill, and how much is luck?

Answered from Tufa Labs' own published run: 25 public games x 20 passes, 500
game-runs, in `example-run/score.json` of github.com/Tufalabs/duck-harness.

Why this exists
---------------
Kaggle scores one pass per game (`bm.n_passes = 1`, and competition mode
refuses a second environment per game), while that reference run has twenty. So
a single Kaggle run is one random draw per game from the observed distribution
— which is exactly what this script simulates, and exactly why the same
submission scored 1.30, 1.21 and 0.77 on different days.

It answers two decision questions with numbers instead of intuition:

1. What is the chance a single run clears a target score?
2. How much does submitting from several accounts change that?

    python eval/duck_variance.py                    # public-set distribution
    python eval/duck_variance.py --calibrate 1.21   # projected to the private set
    python eval/duck_variance.py --games 110 --calibrate 1.21

The private projection is a first-order model, not a measurement: it rescales
the observed public distribution so its mean matches a known leaderboard result,
and draws `--games` samples because averaging over more games reduces spread.
It cannot know whether the hidden games are hard in the same *shape*.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

SCORE_URL = (
    "https://raw.githubusercontent.com/Tufalabs/duck-harness/main/"
    "example-run/score.json"
)
CACHE = Path(__file__).resolve().parents[1] / "reference" / "duck" / "score.json"


def load_scores() -> dict:
    """Per-game lists of per-pass scores, cached so a rerun needs no network."""
    if CACHE.exists():
        payload = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(SCORE_URL, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload), encoding="utf-8")
    return {
        game: [float(v) for v in entry["trial_scores"].values()]
        for game, entry in payload["games"].items()
    }


def describe_pool(pool: List[float]) -> None:
    pool = sorted(pool)
    def q(p: float) -> float:
        return pool[int(p * (len(pool) - 1))]
    zeros = sum(1 for v in pool if v == 0.0)
    print(f"{len(pool)} observed game-runs")
    print(f"  scoring exactly 0 : {zeros} ({100 * zeros / len(pool):.0f}%)")
    print(f"  median            : {q(.50):.2f}")
    print(f"  p75 / p90 / max   : {q(.75):.2f} / {q(.90):.2f} / {q(1.0):.2f}")
    print("  -> half of all runs produce nothing; the score lives in the tail")


def simulate(pool: List[float], games: int, scale: float, trials: int,
             rng: random.Random) -> List[float]:
    """`trials` independent runs, each averaging `games` draws from the pool."""
    return [
        sum(rng.choice(pool) for _ in range(games)) * scale / games
        for _ in range(trials)
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=25,
                        help="games in the evaluation set (25 public, 55/110 private)")
    parser.add_argument("--calibrate", type=float, default=0.0,
                        help="rescale so the mean matches this leaderboard score")
    parser.add_argument("--trials", type=int, default=60000)
    parser.add_argument("--accounts", type=int, default=3)
    parser.add_argument("--targets", default="1.00,1.20,1.50")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    per_game = load_scores()
    pool = [v for scores in per_game.values() for v in scores]
    observed_mean = statistics.mean(
        statistics.mean(scores) for scores in per_game.values()
    )
    scale = (args.calibrate / observed_mean) if args.calibrate > 0 else 1.0

    print(f"reference run: {len(per_game)} games x "
          f"{len(next(iter(per_game.values())))} passes, mean {observed_mean:.2f}")
    describe_pool(pool)
    if scale != 1.0:
        print(f"\ncalibrated to a {args.calibrate:.2f} leaderboard result "
              f"(x{scale:.3f})")

    rng = random.Random(args.seed)
    runs = sorted(simulate(pool, args.games, scale, args.trials, rng))
    mean = statistics.mean(runs)
    print(f"\n=== one Kaggle run, {args.games} games, 1 pass each ===")
    print(f"mean {mean:.2f}   sd {statistics.pstdev(runs):.2f}   "
          f"p05 {runs[int(.05 * len(runs))]:.2f}   "
          f"median {runs[len(runs) // 2]:.2f}   "
          f"p95 {runs[int(.95 * len(runs))]:.2f}")

    targets = [float(t) for t in args.targets.split(",")]
    print(f"\n{'target':>8} {'1 run':>10} {f'best of {args.accounts}':>14}")
    for target in targets:
        single = 100 * sum(1 for v in runs if v >= target) / len(runs)
        hits = 0
        batch = max(2000, args.trials // 10)
        for _ in range(batch):
            best = max(
                simulate(pool, args.games, scale, 1, rng)[0]
                for _ in range(args.accounts)
            )
            if best >= target:
                hits += 1
        print(f"{target:>8.2f} {single:>9.0f}% {100 * hits / batch:>13.0f}%")

    print("\nThe gap between those two columns is the whole argument for "
          "spending submission slots on independent runs rather than on "
          "tuning a solution whose variance already dominates its mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
