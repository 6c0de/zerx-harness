# replay-strategist-200 — model-free record-and-replay

> ## RETRACTED, 2026-08-07. Every number below was measured against the wrong topology.
>
> This experiment scored the agent by driving `arc_agi` **in process**
> (`LocalEnvironmentWrapper`). Kaggle does not run that path: its agent talks
> HTTP to a gateway sidecar with `competition_mode=True`, and
> `arc_agi/api.py`'s RESET handler **refuses to execute a reset when
> `_action_count == 0`** — precisely the condition this whole strategy depended
> on to start a fresh, zero-action play. Its own source comment says so: *"we
> have to look inside the underlying ARCBaseGame to check if this is the first
> action of the level and would cause a full reset"*. The hole is closed on
> purpose, and only in competition mode, which is why it looked open locally.
>
> Measured through a real competition-mode gateway (`eval/gateway_smoke.py`):
> `full_reset` frames returned **0**, scorecard plays **1**, 1869 actions in a
> single play. One play per game, actions cumulative, exploration charged to
> the next level.
>
> **The 1.6825 headline was worth nothing.** The honest number for a
> single-play policy through the gateway is **0.1153**
> (`zerx/single_play.py`). The submission built on this experiment was never
> sent; the gateway test caught it first, which is the one thing this whole
> line of work produced that was worth keeping.
>
> Kept unedited below as the record of what was believed and why. Do not cite
> any figure in it.

- Date: 2026-08-06
- Verdict: **retracted** — see the box above
- Measured on: all 25 public games, scored by the competition's own
  `arc_agi.scorecard.EnvironmentScorecard.from_scorecard` (installed 0.9.8
  wheel), via `eval/local_rhae.py` **driven in process, which is the defect**

## Result

| Configuration | Total RHAE (25 public games) |
|---|---|
| Kaggle submission before this change (Gemma-4-31B in loop) | **0.05** (leaderboard) |
| `Explorer` commit-and-repeat, 1500 actions/game | 0.1111 |
| Uniform random, 1500 actions/game | ~0.0 (2 levels total) |
| Sticky random, 40 rollouts x 300 actions (proxy) | 0.804 |
| Sticky random, 100 rollouts x 120 actions (proxy) | 1.106 |
| `ReplayStrategist`, 150 s/game | 1.3383 |
| `ReplayStrategist`, 300 s/game | 1.6392 |
| `ReplayStrategist` + lattice clicks, 150 s/game | 1.4921 |
| **`ReplayStrategist` + lattice clicks, 300 s/game** | **1.6825 — shipped** |

## Click sampling: draw from a lattice, not from every cell

The uniform `ACTION6` coordinate was drawn from all 64x64 = 4096 cells. These
games render objects as multi-cell blocks, so every cell inside a block is the
same click; per-cell sampling spends ~16 actions learning what one action would
have said. Sampling `range(1, 64, 4)` on both axes — 256 points, still covering
every block of size >= 4 — beat the baseline in both budgets tested: 1.4921 vs
1.3383 at 150 s, 1.6825 vs 1.6392 at 300 s. Two independent comparisons, same
direction, principled mechanism; shipped.

## 1.6825 is this game set's ceiling, not a plateau to tune past

In the 300 s lattice run every one of the 9 scoring games sits at its own
maximum, `max_weights / total_weights * 100`:

```
8.33 + 4.76*5 + 3.57*2 + 2.78  =  42.05
42.05 / 25 games               =   1.682
```

The measured 1.6825 is that number. No amount of further search tuning moves it
— the remaining score lives entirely in the 16 games that never complete a
level, and in levels 3+ of the 9 that do.

## Search time converts directly into score

Doubling the per-game wall clock raised the total by 22% (1.3383 -> 1.6392),
and the mechanism is worth being precise about: **no new game was won.** The
same 9 games scored; what changed is that minimisation converged, and 7 of the
9 now sit at their game's mathematical maximum (`max_weights / total_weights`,
e.g. 4.76 for a one-level clear of a 6-level game).

Two consequences:

- The Kaggle run has roughly 90x the search of the 300 s measurement (7.5-hour
  global deadline, `Swarm` running every game concurrently), so 1.6392 is a
  floor rather than an estimate of what that run produces.
- This policy's ceiling on the public set is about 1.7 with only these 9 games.
  Further gain has to come from the 16 games that never score, or from deeper
  levels — `ar25` reached level 2, and on a 7-level game level 2 roughly
  triples the game score because the weights are 1..L. The prefix-replay
  mechanism that finds deeper levels already exists; it only needs time.

9 of 25 games score; 7 of those clear level 1 at or above human action
efficiency (115% cap). Leaderboard context: top is 1.86, ~50th is 1.39.

## What the scorer actually rewards

Read out of `arc_agi/scorecard.py` and then reproduced against the real engine,
not taken from documentation:

```
level_score = min(115, (human_baseline_actions / actions_on_that_level) ** 2 * 100)
game_score  = sum(level_score_i * i) / sum(i for i in 1..L)
total       = mean(game_score)                       # 0-100%
```

Three mechanics follow, each verified live:

1. **A game's score is its best play.** `EnvironmentScoreList.score` is
   `max(run.score for run in self.runs)`.
2. **Two consecutive RESETs always open a fresh, zero-action play.**
   `Scorecard.update_scorecard` calls `new_play()` exactly when the engine
   reports `full_reset=True`; `BaseGame.handle_reset` sets that when
   `_action_count == 0`; `BaseGame.set_level` — which every level reset calls —
   zeroes it. Measured: `RESET#1 full_reset=False`, `RESET#2 full_reset=True`.
   Note `competition_mode` does **not** block this: it only prevents making a
   second *environment* for a game, and this reuses the same one.
3. **The engine is deterministic.** `full_reset` clones from `_clean_levels`;
   the same action sequence replayed twice produced identical frames.

So exploration costs nothing that is scored, and the only thing that scores is
the shortest run that reaches a level. That is a search problem.

## Why the model was removed

- Action budget. The strategist issues tens of thousands of actions per game
  (866 actions/second measured locally through the real `Agent.main()` loop). A
  31B model at seconds per call cannot issue them, and `Swarm` runs all games
  concurrently against one 9-hour kernel limit.
- Every model-in-loop run this project measured scored 0.0 (see
  `baseline-120.md`), across two separate root causes that were each fixed.

## Rollout length — mixing was tried and rejected

Fixed 120-action rollouts score zero on every game whose level-1 human baseline
is large (tr87 54, dc22 59, sk48 61, wa30 71, g50t 78), and 120 random actions
plainly cannot reach those first levels. Spending some rollouts at 2x/4x/8x the
length — ladder `(n, n, n, 2n, 2n, 4n, 8n)` — was the obvious fix, so it was
measured rather than assumed.

**It was worse: 0.8812 against 1.3383, and it converted none of those games off
zero.** Long rollouts buy fewer independent tickets, and on this benchmark
tickets are what wins. Reverted; `rollout_actions` stays fixed.

The opposite direction was then tested for symmetry — `rollout_actions=60`,
double the tickets, same 150 s — and came out at **1.3016**: a wash against
120, inside the run-to-run variance, and again converting nothing off zero.

Rollout length has now been probed in both directions and neither helps. The
16 games that never score are not a sampling-budget problem; the same 7-9 games
score under every configuration tried. Getting them requires understanding the
game, which is a different piece of work from this one.

## Variance is large — treat any single run as a sample

The same 25 games, same code, different process:

| Game | run A | run B |
|---|---|---|
| vc33 | 3.57 | 0.05 |
| cd82 | 4.76 | 0.00 |
| ar25 | 8.33 | 3.42 |
| m0r0 | 0.59 | 2.43 |

Part of the A/B gap above is therefore luck, not the ladder — though the ladder
also failed on its own stated purpose, which is why it was still reverted. Tufa
Labs report the same effect on the leaderboard: 0.77 to 1.30 for one identical
submission.

The Kaggle run has far more search per game than these measurements do: 150 s
per game locally against a 7.5-hour global deadline with `Swarm` running every
game concurrently. More tickets should raise both the mean and the floor, so
1.3383 is a starved-budget estimate rather than a ceiling.

## Not used: the null-coordinate vulnerability

`arXiv:2605.25931` reports that `ACTION6` with `data={"x": None, "y": None}`
raises a `TypeError` inside the engine that the `arc_agi` wrapper catches and
reports as a WIN, bypassing 18 of 25 public games in one action. Deliberately
not used: it is a library defect rather than gameplay, it is likely closed in
0.9.8, and a submission built on it would not survive review.

## Known gaps

- 16 of 25 public games still score 0.
- The private set is 55 different games; this number does not transfer
  directly.
- `tests/` still targets the removed model-in-loop modules and has not been
  updated.
