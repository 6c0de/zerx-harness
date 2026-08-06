"""Score a policy locally with the competition's own RHAE implementation,
through the competition's own gateway.

`arc_agi` ships the real scorer
(`arc_agi.scorecard.EnvironmentScorecard.from_scorecard`) *and* the real human
baselines (`EnvironmentInfo.baseline_actions`) for all 25 public games, so the
number this prints is computed by the same code Kaggle runs.

**It must run through the gateway.** An earlier version of this harness drove
`arc_agi` in process (`LocalEnvironmentWrapper`) and produced numbers that did
not transfer at all: in-process, two consecutive RESETs open a fresh
zero-action play, so exploration is free and a policy that explores then
replays a minimal solution scored 1.68 here. Kaggle does not run that path. Its
agent talks HTTP to a gateway sidecar with `competition_mode=True`, and
`arc_agi/api.py`'s RESET handler explicitly refuses to execute a reset when
`_action_count == 0` — precisely the condition that would start a new play. One
play per game, actions cumulative, exploration charged to the next level.
Measured, not inferred: `eval/gateway_smoke.py`.

So this harness starts a real competition-mode gateway and plays against it
over HTTP, exactly as the submission does.

    python eval/local_rhae.py                     # all 25 public games
    python eval/local_rhae.py --games ls20,vc33   # a subset
    python eval/local_rhae.py --seconds 60 --workers 8

The one thing it still cannot tell you is how the *private* games behave; treat
a local number as a strong signal about the agent, not a leaderboard
prediction.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arc_agi
from arcengine import GameAction, GameState

from zerx.single_play import SinglePlayAgent, as_grid

logging.getLogger("arc_agi").setLevel(logging.ERROR)
logging.getLogger("arc_agi.scorecard").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def build_policy(name: str, *, seconds: float, budget: int, rollout: int,
                 sticky: float, seed: int):
    """Return a policy object exposing `.step(...) -> ActionKey | None`.

    Kept as a factory so competing policies can be A/B'd through the same
    scorer in one command, which is the only way to compare them honestly.
    """
    if name == "single":
        return SinglePlayAgent(max_seconds=seconds, max_actions=budget,
                               sticky=sticky, seed=seed)
    raise SystemExit(f"unknown --policy {name!r}")


def play_one(
    arc, info, card_id: str, budget: int, verbose: bool,
    seconds: float = 200.0, rollout: int = 120, sticky: float = 0.7,
    policy_name: str = "single",
) -> dict:
    """Play a single game to exhaustion and return a small summary."""
    started = time.time()
    env = arc.make(info.game_id, scorecard_id=card_id)
    if env is None:
        return {"game": info.game_id, "error": "make() returned None"}
    if "Remote" not in type(env).__name__:
        # Guard rail, not politeness. Driving the engine in process measures a
        # different game than the one Kaggle scores — see this module's
        # docstring — and it silently produces much better numbers.
        return {"game": info.game_id,
                "error": f"got {type(env).__name__}, expected the HTTP wrapper"}

    policy = build_policy(
        policy_name, seconds=seconds, budget=budget, rollout=rollout,
        sticky=sticky, seed=abs(hash(info.game_id)) % 997,
    )
    frame = env.reset()
    if frame is None:
        return {"game": info.game_id, "error": "reset() returned None"}

    actions = 0
    while actions < budget:
        grid = as_grid(frame.frame[-1]) if len(frame.frame) else ()
        game_over = frame.state in (GameState.GAME_OVER, GameState.NOT_PLAYED)
        legal = [GameAction.from_id(i).name for i in (frame.available_actions or [])]
        chosen = policy.step(
            grid, legal, frame.levels_completed, game_over,
            frame.state is GameState.WIN,
        )
        if chosen is None:
            break
        name, x, y = chosen

        if name == "RESET":
            frame = env.reset()
        else:
            data = {"x": x, "y": y} if name == "ACTION6" else None
            frame = env.step(GameAction[name], data=data)
        if frame is None:
            break
        actions += 1

    elapsed = time.time() - started
    result = {
        "game": info.game_id,
        "actions": actions,
        "levels": frame.levels_completed if frame else 0,
        "state": frame.state.name if frame else "?",
        "seconds": round(elapsed, 1),
    }
    if verbose:
        print(
            f"  {result['game']:<20} levels={result['levels']:<3} "
            f"actions={result['actions']:<5} {result['state']:<12} {result['seconds']}s",
            flush=True,
        )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="", help="comma-separated game id prefixes")
    parser.add_argument("--budget", type=int, default=40000, help="max actions per game")
    parser.add_argument("--seconds", type=float, default=200.0, help="wall clock per game")
    parser.add_argument("--rollout", type=int, default=120, help="actions per rollout")
    parser.add_argument("--sticky", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--policy", default="single", choices=("single",))
    parser.add_argument("--port", type=int, default=8972)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    # Stand up the competition's own gateway and talk to it over HTTP, with
    # competition_mode on. Anything less measures a different game.
    from eval.gateway_smoke import start_gateway

    server_arc = start_gateway(args.port)
    base = f"http://127.0.0.1:{args.port}/"
    os.environ.update({
        "ARC_BASE_URL": base, "ARC_API_KEY": "test-key-123",
        "OPERATION_MODE": "online", "SCHEME": "http",
        "HOST": "127.0.0.1", "PORT": str(args.port),
    })

    arc = arc_agi.Arcade()
    infos = arc.get_environments()
    if args.games:
        wanted = {g.strip().lower() for g in args.games.split(",") if g.strip()}
        infos = [i for i in infos if i.game_id.split("-")[0].lower() in wanted]
    if not infos:
        print("no matching games", file=sys.stderr)
        return 1

    card_id = arc.open_scorecard(tags=["local-rhae"])
    print(f"playing {len(infos)} games via gateway (competition_mode=True), "
          f"policy={args.policy}, {args.seconds}s/game", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                lambda info: play_one(
                    arc, info, card_id, args.budget, not args.quiet,
                    args.seconds, args.rollout, args.sticky, args.policy,
                ),
                infos,
            )
        )

    # Read results off the server's own scorecard manager rather than the HTTP
    # endpoint: the client is not the scorecard's owner, so /api/scorecard
    # answers 403 for it.
    from arc_agi.scorecard import EnvironmentScorecard

    raw = server_arc.scorecard_manager.get_scorecard(card_id, arc.arc_api_key)
    scorecard = EnvironmentScorecard.from_scorecard(raw, infos)

    print("\n=== per game ===")
    rows = []
    for env in sorted(scorecard.environments, key=lambda e: -e.score):
        # The scored run is the best one, not the first — mirror the scorer.
        run = max(env.runs, key=lambda r: r.score)
        completed = [
            (i + 1, int(s), a, b)
            for i, (s, a, b) in enumerate(
                zip(run.level_scores or [], run.level_actions or [], run.level_baseline_actions or [])
            )
            if s > 0
        ]
        rows.append((env.id, env.score, run.levels_completed, run.actions, completed))
        detail = " ".join(f"L{i}:{s}%({a}/{b})" for i, s, a, b in completed[:6])
        print(f"{env.id:<20} score={env.score:6.2f}  levels={run.levels_completed:<3} "
              f"actions={run.actions:<5} {detail}")

    print(f"\nTOTAL RHAE = {scorecard.score:.4f}   "
          f"(levels {scorecard.total_levels_completed}/{scorecard.total_levels}, "
          f"{scorecard.total_actions} actions)")
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n{len(errors)} game(s) failed: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
