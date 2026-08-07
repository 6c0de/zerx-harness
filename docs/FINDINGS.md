# ARC-AGI-3: what we measured

Every number here was produced by us, locally or on Kaggle hardware, and can be
regenerated with the scripts named beside it. Where we were wrong, that is
recorded too — the retractions are the part of this work we would defend hardest.

---

## 1. We read the scoring rules out of the shipped code, not the documentation

The competition scores **Relative Human Action Efficiency**. The formula is in
`arc_agi/scorecard.py` of the installed 0.9.8 wheel:

```python
score = ((baseline_actions / actions_taken) ** 2) * 100
score = min(score, 115.0)                      # per-level cap
game_score = Σ(level_score_i × i) / Σ(1..L)    # weights are the level numbers
total      = mean(game_score over games)       # 0-100%
```

Three consequences that shaped every later decision:

- **Inefficiency is punished quadratically.** Twice the human action count
  scores 25%; ten times scores 1%.
- **A level's actions are counted from the previous level's completion**, so
  every exploratory action before a level completes is charged to that level.
- **Actions on a level that is never completed are free.** They are charged to a
  level scoring 0 either way and never touch another level's count.

The third point is not a detail. It means the correct policy is not uniformly
careful — it is careful while the current level can still be won, and
unrestrained afterwards.

We also extracted the real **human baselines** for all 25 public games from
`EnvironmentInfo.baseline_actions` (level 1: 17–78 actions). Together with the
shipped scorer this let us compute the official score offline, with no
submissions spent — `eval/local_rhae.py`.

---

## 2. We found an exploit, built a replica of the deployment, and disproved it

This is the finding we are least proud of and most glad we made.

Reading the scorer further: a game's score is `max(run.score for run in runs)` —
the **best** play, not the last. And a play boundary is a full reset, which
`BaseGame.handle_reset` triggers whenever `_action_count == 0`. Since
`set_level` zeroes that counter, **two consecutive RESETs should open a fresh,
zero-action play.** We confirmed it live (`RESET#1 full_reset=False`,
`RESET#2 full_reset=True`) and confirmed the engine is deterministic.

If true, exploration is free: explore, then replay a minimal solution and let
`max()` keep it. We built that agent. It scored **1.6825** on the 25 public
games — every scoring game at its mathematical ceiling. We pushed it as a
submission.

Then we built `eval/gateway_smoke.py`, which stands up the competition's *own*
gateway (`competition_mode=True`) and plays it over HTTP, exactly as Kaggle
does. Result:

```
wrapper: RemoteEnvironmentWrapper
full_reset frames: 0
scorecard plays  : 1
per-play actions : [1869]
```

The mechanism does not exist on Kaggle. `arc_agi/api.py`'s RESET handler
refuses to execute a reset when `_action_count == 0` — the exact condition that
would start a new play. Its own comment says so:

> *"This is quite hacky as we have to look inside the underlying ARCBaseGame to
> check if this is the first action of the level and would cause a full reset"*

The guard exists **only in competition mode**, which is why it looked open when
we drove the engine in process. ARC Prize closed this deliberately.

**The 1.6825 was worth nothing, and the submission was never sent.** The cost of
finding out was one evening; the cost of not finding out would have been a
submission and eight hours. `eval/local_rhae.py` now refuses to score anything
that is not the HTTP wrapper, so this class of mistake cannot recur silently.

---

## 3. Our own agent, written for the real rules

`zerx/single_play.py`, measured through the gateway: **0.1153** on the 25 public
games (the submission it replaced scored 0.05).

Its one idea comes straight from the metric — *careful, then reckless*. While
the current level can still be won it suppresses known no-op actions and
repeats what moves the board; once a level has cost more than ~220 actions it is
worth under 1% however it ends, so the policy stops protecting it and explores
freely, because those actions are free and only the next level's counter
matters.

A second, less obvious fix came out of the first measurement: **every level it
found came from the reckless phase and none from the careful one.** The cause
was that these games animate — a HUD ticker changes the frame on every step, so
"the board changed" was always true, no action was ever recognised as a no-op,
and the careful phase repeated one move forever. We added a volatility mask that
ignores cells changing regardless of what we do.

We stopped here, and the reason is worth stating: 20 of 25 games never complete
level 1 at all. That is not an efficiency gap that tuning closes — it is not
knowing what the game wants. We measured the alternatives rather than assuming
them: doubling the time budget converted no new games, and rollout length was
tested in both directions (60 and 960 actions) with no game converted either.

---

## 4. The leaderboard is mostly variance, and we can prove it

Tufa Labs' published Duck harness — the Milestone 1 winner — ships one full
benchmark run: 25 games × 20 passes, 500 game-runs. Kaggle scores **one** pass
per game. So a Kaggle run is one random draw per game from that distribution.

`eval/duck_variance.py` simulates exactly that:

```
500 observed game-runs
  scoring exactly 0 : 248 (50%)
  median            : 0.06
  p75 / p90 / max   : 2.49 / 4.76 / 25.33
```

Half of all game-runs produce nothing. One game (`ft09`) carries a quarter of
their public-set mean. Projected to the private set and calibrated to their own
1.21 leaderboard result:

| target | one run | best of 3 |
|---|---|---|
| ≥ 1.00 | 75% | **99%** |
| ≥ 1.20 | 48% | **86%** |
| ≥ 1.50 | 16% | 40% |

This is why the same submission scored 1.30, 1.21 and 0.77 on different days —
and why the right use of three submission slots is three independent runs, not
three attempts at tuning a solution whose variance already dominates its mean.

---

## 5. What we changed in the Duck, and what we refused to change

Full detail in `docs/DUCK_FORK.md`; generated by `scripts/build_duck_notebook.py`
so every change is reviewable as code rather than buried in a 25 KB notebook.

**The rule we held to: change nothing we cannot measure.** The harness needs a
96 GB card and a 27B model, so a change to its prompts, sampling or tools could
not be A/B'd before submitting — and against ±0.45 of noise, an unmeasured
change is a coin flip on a leaderboard-verified solution.

So every change is a guard or a scheduling decision:

1. **A soft deadline for the scored run.** Upstream gives one only to the
   unscored pass. Kaggle kills a notebook at 9 hours and a killed notebook emits
   no `submission.parquet` at all — so a hidden set that runs long loses
   *everything*, not part of it.
2. **A preflight gate** — GPU model, VRAM, and that every attached dataset
   actually mounted. Kaggle silently substitutes a smaller card when the
   accelerator flag is not recognised; we measured that in this project
   (pushing `nvidiaRtx6000` yielded a Tesla P100).
3. **A bounded verification pass.** Upstream's Save & Run All plays the public
   games offline with the real solver — a genuine end-to-end test that costs no
   submission. We shortened it so it can be repeated.
4. **An adaptive per-game budget.** The score is a *mean over games*, so an
   unplayed game is the most expensive outcome available. We measured ~180
   generated tokens/second on this hardware; at Tufa Labs' ~59k tokens per game,
   110 hidden games need ~10 hours against a 9-hour limit, and a fixed per-game
   budget lets the early waves take everything. Ours splits the time that
   actually remains across the waves actually needed — and never raises the
   budget above what upstream asked for.
5. **A run manifest**, so a low score and "played four games, then got killed"
   are distinguishable in the logs.

**And one change we measured and then refused to ship.** The server launches
with `max_model_len=65536` while the agent's context window is `32768`, and
vLLM's own startup line reads like a smoking gun: *"Maximum concurrency for
65,536 tokens per request: 10.12x"*. Halving it looks like free concurrency. The
same run's logs show peak KV usage of 76% across 25 concurrent requests, which
puts the real average context near 5,400 tokens — the pool is not the binding
constraint and the change would have bought nothing. Recording that is the
method working.

---

## Reproducing

```bash
python eval/gateway_smoke.py                       # Kaggle's topology, locally
python eval/local_rhae.py --seconds 90             # official scorer, 25 games
python eval/duck_variance.py --games 55 --calibrate 1.21
python scripts/build_duck_notebook.py              # the guarded fork
```
