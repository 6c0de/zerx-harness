"""Measure which trivial strategies actually complete levels, per game.

Ground truth for the policy design. arXiv:2605.25931 Table 9 claims every
public game falls to a single action repeated 50-200 times; this script checks
that claim against the installed engine instead of trusting it, and reports the
action count each strategy needed — which is what RHAE actually scores.

    python eval/strategy_probe.py --budget 250
    python eval/strategy_probe.py --games ft09,ls20 --budget 400
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arc_agi
from arcengine import GameAction, GameState

from zerx.single_play import as_grid

logging.getLogger("arc_agi").setLevel(logging.ERROR)
logging.getLogger("arc_agi.scorecard").setLevel(logging.ERROR)


def run_strategy(arc, game_id: str, name: str, budget: int) -> Tuple[str, int, int]:
    """Play `budget` actions of one fixed strategy. Returns (name, levels, actions_to_first_level)."""
    env = arc.make(game_id)
    frame = env.reset()
    if frame is None:
        return (name, 0, -1)

    first_level_at = -1
    sweep = _sweep_points()
    for step in range(budget):
        if frame.state is GameState.WIN:
            break
        if frame.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            frame = env.reset()
            continue

        if name.startswith("ACTION6"):
            if name == "ACTION6@center":
                x, y = 32, 32
            elif name == "ACTION6@sweep":
                x, y = sweep[step % len(sweep)]
            else:  # ACTION6@null - the library vulnerability, measured not used
                x = y = None
            frame = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        else:
            frame = env.step(GameAction[name])

        if frame is None:
            break
        if first_level_at < 0 and frame.levels_completed >= 1:
            first_level_at = step + 1

    levels = frame.levels_completed if frame else 0
    return (name, levels, first_level_at)


def _sweep_points() -> List[Tuple[int, int]]:
    """A stride-4 lattice over the 64x64 grid, centre-out.

    Centre-out because interactive elements sit near the middle of the board
    far more often than in the corners, and RHAE charges for every click that
    lands on nothing.
    """
    pts = [(x, y) for y in range(2, 64, 4) for x in range(2, 64, 4)]
    pts.sort(key=lambda p: (p[0] - 32) ** 2 + (p[1] - 32) ** 2)
    return pts


STRATEGIES = (
    "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7",
    "ACTION6@center", "ACTION6@sweep",
)


def probe_game(arc, info, budget: int) -> dict:
    started = time.time()
    out = {"game": info.game_id.split("-")[0], "levels": len(info.baseline_actions or []),
           "baseline": info.baseline_actions or [], "results": {}}
    for name in STRATEGIES:
        try:
            _, levels, first_at = run_strategy(arc, info.game_id, name, budget)
        except Exception as exc:  # a strategy that crashes the engine is data too
            out["results"][name] = ("ERR", str(exc)[:60])
            continue
        out["results"][name] = (levels, first_at)
    out["seconds"] = round(time.time() - started, 1)

    best = max(
        (v for v in out["results"].values() if isinstance(v[0], int)),
        key=lambda v: (v[0], -v[1] if v[1] > 0 else -10 ** 6),
        default=(0, -1),
    )
    winners = [
        f"{k}:lv{v[0]}@{v[1]}"
        for k, v in out["results"].items()
        if isinstance(v[0], int) and v[0] > 0
    ]
    print(
        f"{out['game']:<6} levels={out['levels']:<3} baseline={out['baseline']} "
        f"best={best[0]}  {' '.join(winners) or '-- none --'}  ({out['seconds']}s)",
        flush=True,
    )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="")
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    arc = arc_agi.Arcade()
    infos = arc.get_environments()
    if args.games:
        wanted = {g.strip().lower() for g in args.games.split(",") if g.strip()}
        infos = [i for i in infos if i.game_id.split("-")[0].lower() in wanted]

    print(f"probing {len(infos)} games x {len(STRATEGIES)} strategies x {args.budget} actions")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda i: probe_game(arc, i, args.budget), infos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
