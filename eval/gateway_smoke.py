"""Reproduce the Kaggle deployment topology locally and assert its real rules.

Kaggle does not run the engine in process. It runs it in a gateway sidecar and
the agent talks HTTP to it (`arc_agi.remote_wrapper`) with
`competition_mode=True`. Nothing in the submission notebook exercises that path
before a submission: the gameplay cell only runs when
`KAGGLE_IS_COMPETITION_RERUN` is set, which the Save & Run All phase does not
set. So without this script, the first execution of the real topology is the
scored one.

**What it found, 2026-08-07.** The project had built a strategy on the
in-process behaviour that two consecutive RESETs open a fresh, zero-action play
— making exploration free and the shortest replay the only thing scored. Local
runs measured 1.68 that way. Through a real competition-mode gateway the same
policy returned `full_reset` frames **0** and scorecard plays **1**:
`arc_agi/api.py`'s RESET handler deliberately refuses to execute a reset when
`_action_count == 0`, which is exactly the condition that would start a new
play. The guard exists only in competition mode, which is why it looked open in
process.

So this script now asserts the rules that actually hold, and fails if the
in-process assumptions ever creep back in:

* the wrapper really is the HTTP one (otherwise nothing here proves anything),
* the gateway answers every action,
* the run stays a single play, with actions accumulating,

and it reports the HTTP action rate, which is the real budget number for
planning a 9-hour run.

    python eval/gateway_smoke.py                 # default game, ~30s
    python eval/gateway_smoke.py --game vc33

Exits non-zero if the topology does not behave, so it can gate a submission.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.getLogger("arc_agi").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

PORT = 8971  # deliberately not 8001: don't collide with a real gateway


def start_gateway(port: int) -> "object":
    """Run a competition-mode gateway in a daemon thread and wait for it."""
    import arc_agi

    server_arc = arc_agi.Arcade()

    def serve() -> None:
        server_arc.listen_and_serve(
            host="127.0.0.1", port=port, competition_mode=True,
            use_reloader=False,
        )

    threading.Thread(target=serve, daemon=True).start()

    import requests

    for _ in range(60):
        try:
            requests.get(f"http://127.0.0.1:{port}/api/games", timeout=2)
            return server_arc
        except Exception:  # noqa: BLE001 - the server simply is not up yet
            time.sleep(0.5)
    raise SystemExit("gateway did not come up")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="vc33")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    server_arc = start_gateway(args.port)
    base = f"http://127.0.0.1:{args.port}/"
    print(f"gateway up on {base} (competition_mode=True)")

    # A second, independent client configured exactly as the Kaggle run cell
    # configures the framework: talk to the gateway over HTTP, not in process.
    os.environ["ARC_BASE_URL"] = base
    os.environ["ARC_API_KEY"] = "test-key-123"
    os.environ["OPERATION_MODE"] = "online"
    os.environ["SCHEME"], os.environ["HOST"], os.environ["PORT"] = (
        "http", "127.0.0.1", str(args.port),
    )

    import arc_agi
    from arcengine import GameAction, GameState

    from zerx.single_play import SinglePlayAgent, as_grid

    client = arc_agi.Arcade()
    infos = [i for i in client.get_environments()
             if i.game_id.split("-")[0].lower() == args.game.lower()]
    if not infos:
        return _fail(f"game {args.game!r} not offered by the gateway")
    info = infos[0]

    card_id = client.open_scorecard(tags=["gateway-smoke"])
    env = client.make(info.game_id, scorecard_id=card_id)
    if env is None:
        return _fail("make() returned None against the gateway")
    print(f"wrapper: {type(env).__name__}")
    if "Remote" not in type(env).__name__:
        return _fail(
            f"expected a remote wrapper, got {type(env).__name__} — this test "
            "is not exercising the HTTP path and proves nothing"
        )

    # ---- the actual measurement ----
    policy = SinglePlayAgent(max_seconds=args.seconds, seed=7)
    frame = env.reset()
    full_reset_seen = 1 if getattr(frame, "full_reset", False) else 0
    actions = 0
    while True:
        grid = as_grid(frame.frame[-1]) if len(frame.frame) else ()
        legal = [GameAction.from_id(i).name for i in (frame.available_actions or [])]
        legal.append("RESET")
        chosen = policy.step(
            grid, legal, frame.levels_completed,
            frame.state in (GameState.GAME_OVER, GameState.NOT_PLAYED),
            frame.state is GameState.WIN,
        )
        if chosen is None:
            break
        name, x, y = chosen
        if name == "RESET":
            frame = env.reset()
        else:
            frame = env.step(GameAction[name],
                             data={"x": x, "y": y} if name == "ACTION6" else None)
        if frame is None:
            return _fail(f"gateway returned None after {actions} actions")
        if getattr(frame, "full_reset", False):
            full_reset_seen += 1
        actions += 1

    elapsed = max(1e-9, time.time() - policy._started)
    scorecard = server_arc.scorecard_manager.get_scorecard(card_id, client.arc_api_key)
    card = scorecard.cards.get(info.game_id) if scorecard else None

    print()
    print(f"actions          : {actions}  ({actions / elapsed:.0f}/s over HTTP)")
    print(f"full_reset frames: {full_reset_seen}")
    print(f"scorecard plays  : {card.total_plays if card else 0}")
    print(f"per-play actions : {card.actions[:12] if card else []}")
    print(f"levels per play  : {card.levels_completed[:12] if card else []}")

    # ---- verdicts ----
    if actions < 10:
        return _fail(
            f"only {actions} action(s) got through. The gateway is not serving "
            "the policy; nothing below this can be trusted."
        )
    if not card or card.total_plays < 1:
        return _fail("the gateway recorded no play at all for this game")

    if card.total_plays > 1 or full_reset_seen:
        # Not a failure — a finding. If ARC Prize ever relaxes the competition
        # -mode reset guard, exploration becomes free again and the retracted
        # record-and-replay strategy becomes worth far more than careful play.
        # That would be the single biggest scoring change available, so it must
        # never pass unnoticed.
        print(
            f"\nNOTE — the reset guard appears to be OPEN: plays="
            f"{card.total_plays}, full_reset frames={full_reset_seen}. "
            "Competition mode used to collapse this to a single play. Re-read "
            "arc_agi/api.py's RESET handler before planning around it, then see "
            "the retracted experiment in docs/superpowers/experiments/."
        )
    else:
        print("\nOK — single play, actions cumulative, reset guard closed: "
              "the rules zerx/single_play.py is written against.")

    if actions / elapsed < 1.0:
        print(f"WARNING: only {actions / elapsed:.1f} actions/s over HTTP. A "
              "9-hour run cannot issue many actions at this rate.")
    return 0


def _fail(message: str) -> int:
    print(f"\nFAIL: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
