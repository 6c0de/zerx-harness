# baseline-120 Local Regression & Fallback-Loop Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove (or disprove) the "no regressions" promotion criterion across all 25 public games, and root-cause the exact stuck-action-loop behavior `docs/superpowers/plans/parallel-baseline-120/README.md` reports for `ls20`+`vc33`, encoding the confirmed finding as regression tests.

**Architecture:** One new test file, `tests/test_real_game_regression.py`, driving the real local `arc_agi` engine and the real `MyAgent`/`decide()` exactly as they exist on this branch's base commit (including the known, not-yet-fixed hardcoded-`GemmaModelBackend` construction bug) — no code under test is modified, because the investigation below found no bug in any file this track is permitted to own.

**Tech Stack:** Python 3, pytest, the real `arc_agi` PyPI package (local `OperationMode.NORMAL` engine, network-reachable, read-only, same precedent already used by `scripts/play_local.py` and `tests/test_my_agent.py`), the vendored `ARC-AGI-3-Agents` framework already present at `vendor/ARC-AGI-3-Agents`.

## Global Constraints

- Base commit: `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb` (branch `feat/baseline-120-local-regression`, forked from it).
- Local suite baseline on this commit: 261 passed, 0 failed (verified this session, matches `docs/HANDOFF.md`).
- Files this track owns: `tests/test_real_game_regression.py` (new), `docs/HANDOFF.md` (one-line status only). Conditionally `zerx/heuristics.py` — **not used**, see "Investigation findings" below: no bug was found in that file.
- Do not touch: `zerx/model_backend.py`, `zerx/config.py`, `eval/run_ablation.py`, `scripts/build_colab_notebook.py`, `agent/my_agent.py` (Track 1 owns backend-selection wiring this round).
- No behavior change for any existing test with no env vars set — this track adds new, independently-tested surface only.
- Push only to `feat/baseline-120-local-regression` — never merge to `master`.
- All 25 public games must be covered by the crash-safety sweep (not a subset); measured (not estimated) wall-clock cost must be documented.

---

## Investigation findings (completed before this plan was written, per `superpowers:systematic-debugging`'s discipline — reproduce first, form a hypothesis, verify against the actual code path, not guesses)

**Reproduction, verified firsthand (not trusted secondhand):** ran `.venv/Scripts/python.exe scripts/play_local.py --game ls20,vc33 --max-steps 50` on this exact base commit. Result differs from `docs/superpowers/plans/parallel-baseline-120/README.md`'s stated finding in one material way, documented precisely below rather than silently accepted.

### Confirmed: `vc33` (ACTION6-legal game) — matches the reported finding exactly

50/50 actions were `ACTION6`, `levels_completed=0` throughout — reproduced byte-for-byte. Root cause, traced through the actual code, not assumed:

1. `agent/my_agent.py:156`'s `MyAgent.__init__` unconditionally constructs `GemmaModelBackend(self._config.model_revision)`, ignoring `Config.backend` entirely (already flagged in `docs/HANDOFF.md`'s "Known failures" #1 and owned by Track 1 this round — not re-litigated here). With no vLLM server on `localhost:8000`, every `backend.generate()` call raises; `zerx/policy.py`'s `decide()` (`config.candidate_count == 1` default path, line ~227-231) catches it and `parsed` stays `None` on every single call.
2. With `parsed is None`, `decide()` falls to `if candidates and ActionName.ACTION6 in legal_actions:` (`zerx/policy.py:239`) — `vc33`'s frames always have ≥6 non-background objects, so this is always true, and `candidates[0]` (the top-ranked object from `zerx/heuristics.py`'s `rank_click_candidates`) is always returned as `source="fallback_heuristic"`.
3. **New finding, empirically confirmed via direct instrumentation (not in the original report):** `Decision(source="fallback_heuristic", target_object_label=top.object_label, ...)` *does* correctly set `target_object_label` (verified by reading `zerx/policy.py:239-249` and confirmed live: `agent/my_agent.py`'s outcome-feedback gate at line ~179 does fire every step from step 2 onward). So `zerx/heuristics.py`'s `DeadSignatureTracker.record_outcome` *is* being called every single step — this rules out person-3's "outcome feedback might not be firing" hypothesis.
4. **The actual mechanism:** instrumenting `record_outcome`'s arguments directly shows it is called every step with `effective=True` — never `False` — for whichever object is currently top-ranked (observed: `vc33`'s frame has a shrinking/growing bar-like region in row 0 that changes every single step regardless of what was clicked, most likely a passive timer/counter). Since `TransitionRecord.effective` (`zerx/transitions.py:58-62`) is a **whole-grid** pixel diff (`changed_pixels > 0`), that unrelated animating region alone is enough to mark the transition "effective" even though the *actually-clicked* object never itself changes. `DeadSignatureTracker.record_outcome`'s `effective=True` branch always *decreases* the penalty (`zerx/heuristics.py:39-40`), so the penalty for the repeatedly-clicked signature never accumulates (`penalties={}` stays empty for the entire run) and `rank_click_candidates` never reorders — the same top candidate wins every single step, producing the observed zero-coordinate-variation `ACTION6` loop.

This is not a new, undiscovered bug in any file this track owns. It is the **first empirical confirmation** of `STRATEGY.md` §5.4's already-documented, already-scoped limitation: *"`zerx/transitions.py`'s `TransitionRecord.effective` ... cannot currently tell a real gameplay change from a HUD-only animation ... This is a known, accepted simplification for the baseline, not an oversight — fixing it properly needs object-level correspondence ... explicitly `exp-150-duck-tools` Variant A scope, not baseline scope."* `zerx/transitions.py` is not owned by any of `baseline-120`'s 4 tracks this round (absent from the README's ownership matrix), and `zerx/heuristics.py` itself has no defect — it faithfully honors the `effective` value it's given; the *input* is what's misleading, and that input's source is explicitly out of this track's scope. Per `person-3-local-regression.md`'s own branching rule ("if the root cause turns out to live in a file another track owns... document the finding precisely... let the integration owner sequence the fix"), this is documented here, not patched.

### Corrected: `ls20` (no ACTION6) — the original report's claim does not hold for this game, on this platform

Directly checked `ls20`'s real legal actions (`ProbeAgent` instrumentation, live engine): `{'ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'RESET'}` — **`ACTION6` is never legal for `ls20`.** Fifteen consecutive `choose_action` calls against the real `ls20` engine, on this exact base commit, always returned `ACTION1`, `source="fallback_deterministic"` — never `ACTION6`.

This is a **completely different, simpler mechanism** than `vc33`'s, and it is fully and only explained by the missing backend (no second bug): since `ActionName.ACTION6 in legal_actions` is always `False` for `ls20`, `decide()` never reaches the candidate/heuristic system at all (`zerx/policy.py:179`, `:239` both gate on it) — it falls straight to `_deterministic_fallback` (`zerx/policy.py:98-108`), whose `_FALLBACK_PREFERENCE` is a fixed, static tuple `(ACTION5, ACTION1, ACTION2, ACTION3, ACTION4, ACTION6)`. `ACTION5` isn't legal, `ACTION1` is → `ACTION1` wins, unconditionally, every call, completely independent of game state, `zerx/heuristics.py`, or `DeadSignatureTracker`. This is the fallback chain (`AGENTS.md`'s `validated model action -> validated heuristic action -> validated deterministic legal fallback -> RESET if terminal`) operating exactly as designed when nothing above it succeeds — a correct, documented behavior, not a bug.

**Why the original report's wording ("every single action across both 50-step runs was ACTION6") doesn't match this platform:** `docs/superpowers/plans/parallel-baseline-120/README.md`'s reproduction command, run verbatim on this exact base commit on this (Windows) machine, **crashes with an unrelated `UnicodeEncodeError` immediately after `vc33` finishes, before `ls20` is ever played** — `scripts/play_local.py:114`'s final per-game summary line hardcodes a `→` (→) character, which the Windows `cp1254` console codepage cannot encode, so the script's `for` loop over games terminates via an uncaught exception right after game 1/2. `ls20` genuinely never ran in this reproduction. The original report was almost certainly produced on a non-Windows machine (`docs/HANDOFF.md`'s own precedent notes the Day 3 integration session ran on macOS) where the arrow prints fine and both games complete — meaning its `ls20` claim was likely either imprecise, or observed under different circumstances this session cannot reproduce as stated. This `scripts/play_local.py` bug is real, previously undocumented, and **out of this track's scope to fix** (not in the ownership matrix, not `tests/test_real_game_regression.py`, not `docs/HANDOFF.md`) — recorded here for the record; my own sweep below drives `MyAgent` directly rather than going through this script, so it is not exposed to this crash.

### Conclusion (the two-part deliverable `person-3-local-regression.md` asks for)

**Is the stuck-loop behavior fully explained by the missing backend wiring alone, or is there a second bug?** Both, depending on the game's legal-action set — and in neither case is the second factor a *new, unowned-file* bug requiring a fix in this track:

- Games without `ACTION6` (e.g. `ls20`): **fully and only** explained by the missing backend (Track 1's fix is the complete remedy — once real model reasoning runs, it can propose `ACTION2`/`ACTION3`/`ACTION4` instead of always `ACTION1`).
- Games with `ACTION6` (e.g. `vc33`): explained by the missing backend **plus** a first empirical confirmation of `STRATEGY.md` §5.4's already-documented, already-scoped (`exp-150-duck-tools` Variant A) HUD-vs-gameplay-change limitation in `zerx/transitions.py` (owned by nobody this round). Track 1's fix alone will very likely still break this specific loop in practice (real Gemma reasoning doesn't depend on `DeadSignatureTracker`'s penalty to pick a different coordinate), but the *heuristic-only* fallback path will keep exhibiting this exact degenerate behavior on HUD-animated games even after Track 1 lands, until `exp-150` is built.

Both mechanisms are encoded as regression tests below (Task 2), matching the "confirmed non-bug is an acceptable, valuable outcome — document it, don't force a fix" instruction in `person-3-local-regression.md`.

---

## Task 1: Crash-safety sweep across all 25 public games

**Files:**
- Create: `tests/test_real_game_regression.py`

**Interfaces:**
- Consumes: `agent.my_agent.MyAgent` (real class, unmodified), `arc_agi.Arcade`/`OperationMode` (real engine, same pattern as `tests/test_my_agent.py` and `scripts/play_local.py`).
- Produces: nothing consumed by later tasks in this plan — this is a standalone test.

Game ids are **hardcoded**, not fetched from `arc.get_environments()` at module/collection time — fetching at collection time would make a network call during `pytest`'s collection phase for the *entire* suite (including fast, no-network unit tests), which must never depend on network reachability. The 25 ids below were confirmed live via `.venv/Scripts/python.exe scripts/play_local.py --list` this session:

```python
ALL_PUBLIC_GAME_IDS = [
    "su15", "sb26", "ft09", "cd82", "sk48", "tr87", "sc25", "ls20", "g50t",
    "bp35", "lf52", "m0r0", "vc33", "tn36", "r11l", "dc22", "sp80", "ka59",
    "cn04", "s5i5", "re86", "ar25", "tu93", "lp85", "wa30",
]
```

Measured per-action wall-clock cost this session (`vc33`, 5 real `choose_action` calls through the real, currently-hardcoded `GemmaModelBackend` against an unreachable `localhost:8000`): **8.15s/action** (dominated by the missing-backend's connection/retry path — the exact same Track-1-owned condition this sweep deliberately tests *as-is*). At 25 games × 5 steps/game, that's **≈ 1019s (≈ 17 minutes)** — too slow for the default fast suite, so this test is marked `@pytest.mark.slow_local_engine` (unregistered custom marker — matches this repo's existing, already-unregistered `cerebras_live` precedent from `AGENTS.md`'s testing-gates section; produces a benign `PytestUnknownMarkWarning`, not a failure, and is excluded from the default run via `-m "not slow_local_engine"` or simply not selected unless `-m slow_local_engine` is passed).

Step cap is **5** (not the 20-30 `person-3-local-regression.md` suggested as a starting estimate) — chosen and documented here specifically because the *measured* per-action cost on this platform (8.15s) is far higher than whatever assumption produced that suggested range; 5 steps is enough to prove "reaches a terminal state or the step cap without raising" (the actual assertion) without an unreasonable total run time.

- [ ] **Step 1: Write the test**

```python
"""Real-game regression tests against the live local arc_agi engine — no
GPU, no model backend required. Drives agent/my_agent.py's real MyAgent
exactly as it exists on this branch's base commit (including the known,
not-yet-fixed hardcoded-GemmaModelBackend construction — see
docs/superpowers/plans/2026-08-05-baseline-120-local-regression.md's
"Investigation findings" for the full root-cause writeup this file's
tests encode).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

from agent.my_agent import MyAgent, _to_game_frame  # noqa: E402
from zerx.config import Config  # noqa: E402
from zerx.heuristics import DeadSignatureTracker  # noqa: E402
from zerx.memory import MemoryState  # noqa: E402
from zerx.model_backend import FakeModelBackend  # noqa: E402
from zerx.policy import decide  # noqa: E402
from zerx.types import ActionName  # noqa: E402

ALL_PUBLIC_GAME_IDS = [
    "su15", "sb26", "ft09", "cd82", "sk48", "tr87", "sc25", "ls20", "g50t",
    "bp35", "lf52", "m0r0", "vc33", "tn36", "r11l", "dc22", "sp80", "ka59",
    "cn04", "s5i5", "re86", "ar25", "tu93", "lp85", "wa30",
]

SWEEP_STEP_CAP = 5  # see plan doc: measured 8.15s/action on this platform


@pytest.fixture(scope="module")
def arcade():
    return arc_agi.Arcade(operation_mode=OperationMode.NORMAL)


@pytest.mark.slow_local_engine
@pytest.mark.parametrize("game_id", ALL_PUBLIC_GAME_IDS)
def test_crash_safety_sweep(arcade, game_id):
    """No unhandled exception escapes MyAgent.choose_action for any public
    game, and the run always reaches a terminal GameState or the step cap
    -- baseline-100/baseline-110's "no regressions" promotion criterion,
    verified against the FULL 25-game public set for the first time (prior
    sessions only ever exercised ls20+vc33).
    """
    env = arcade.make(game_id)
    if env is None:
        pytest.skip(f"arcade.make({game_id!r}) returned None -- game unavailable")

    agent = MyAgent(
        card_id="regression-sweep",
        game_id=game_id,
        agent_name=f"regression-sweep.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
    )
    agent.MAX_ACTIONS = SWEEP_STEP_CAP

    try:
        agent.main()
    except Exception as exc:  # noqa: BLE001 - the exact thing this test checks for
        pytest.fail(
            f"{game_id}: unhandled exception escaped MyAgent.main() after "
            f"{agent.action_counter} action(s): {type(exc).__name__}: {exc}"
        )

    assert agent.action_counter <= SWEEP_STEP_CAP
    assert agent.frames[-1].state is not None
```

- [ ] **Step 2: Run it (opt-in, slow — run explicitly, don't wait for the default suite)**

Run: `.venv/Scripts/pytest.exe tests/test_real_game_regression.py -v -m slow_local_engine`
Expected: 25 passed (or documented skips for any `arcade.make()` returning `None`), no unhandled-exception failures. This run is slow (~17 minutes measured) — run it once, record the actual wall-clock time and pass/fail breakdown in this plan file's "Sweep result" section below (added as part of this step, not deferred).

- [ ] **Step 3: Confirm the fast default suite is unaffected**

Run: `.venv/Scripts/pytest.exe tests/ -q`
Expected: 261 passed still passes with no `-m` filter needed for the count to stay the same — `slow_local_engine`-marked tests still *run* by default (pytest runs all tests regardless of marks unless `-m` excludes them), so this step's real purpose is confirming the sweep doesn't break collection or leak state into other test files, not confirming speed. If the full unfiltered run is too slow to include in the routine full-suite command, note that explicitly in this plan's "Sweep result" section as a documented trade-off (matching `person-3-local-regression.md`'s own anticipation of this decision) rather than silently leaving it ambiguous.

- [ ] **Step 4: Commit**

```bash
git add tests/test_real_game_regression.py
git commit -m "test(regression): add 25-game crash-safety sweep against the real local engine"
```

---

## Task 2: Fallback-loop characterization tests (lock in both confirmed mechanisms)

**Files:**
- Modify: `tests/test_real_game_regression.py` (append)

**Interfaces:**
- Consumes: `zerx.policy.decide` (real function, unmodified), a real `GameFrame` pulled once from the live engine per game (fast — no repeated network round-trips; `decide()` itself is pure/local).

- [ ] **Step 1: Write the tests**

Append to `tests/test_real_game_regression.py`:

```python
def _live_frame(arcade, game_id: str):
    """One real initial GameFrame from the live engine, translated the
    same way agent/my_agent.py does. decide() is pure/local, so a single
    fetched frame is enough to characterize its behavior deterministically
    -- no need to step the live engine repeatedly for these tests.
    """
    env = arcade.make(game_id)
    assert env is not None, f"arcade.make({game_id!r}) returned None"
    agent = MyAgent(
        card_id="characterization",
        game_id=game_id,
        agent_name=f"characterization.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
    )
    return _to_game_frame(agent.frames[-1])


def test_ls20_fallback_loop_is_fully_explained_by_missing_backend(arcade):
    """Mechanism A (plan doc): ls20 has no ACTION6 in its legal-action set,
    so decide() never reaches the candidate/heuristic system at all and
    always falls to the same static _deterministic_fallback choice. This
    locks in that today's (Track-1-fix-pending) behavior is exactly
    ACTION1, every call, regardless of zerx/heuristics.py.
    """
    frame = _live_frame(arcade, "ls20")
    assert ActionName.ACTION6 not in frame.legal_actions
    backend = FakeModelBackend(responses=[])  # every .generate() raises
    memory = MemoryState()
    dead_signatures = DeadSignatureTracker()
    actions = set()
    sources = set()
    for _ in range(20):
        decision, memory = decide(
            frame=frame, history=(), memory=memory,
            dead_signatures=dead_signatures, config=Config(),
            backend=backend, actions_taken=0,
        )
        actions.add(decision.action.name)
        sources.add(decision.source)
    assert actions == {ActionName.ACTION1}
    assert sources == {"fallback_deterministic"}


def test_vc33_fallback_loop_never_diversifies_when_transitions_report_effective(arcade):
    """Mechanism B (plan doc): vc33 has ACTION6 legal and multiple ranked
    click candidates, so the candidate/heuristic path IS reachable -- but
    zerx/heuristics.py's DeadSignatureTracker never down-ranks the
    repeatedly-chosen candidate here, because this test simulates exactly
    what the real agent/my_agent.py wiring does when zerx/transitions.py's
    whole-grid diff reports effective=True every step (STRATEGY.md 5.4's
    documented HUD-vs-gameplay-change limitation, confirmed live this
    session on vc33's animated top-row bar). This is not a bug in
    zerx/heuristics.py -- it faithfully honors the effective value it's
    given; this test locks in that faithful (but, on this game, misled)
    behavior so a future exp-150 fix has a measurable "before".
    """
    frame = _live_frame(arcade, "vc33")
    assert ActionName.ACTION6 in frame.legal_actions
    backend = FakeModelBackend(responses=[])
    memory = MemoryState()
    dead_signatures = DeadSignatureTracker()
    coordinates = set()
    sources = set()
    for _ in range(20):
        decision, memory = decide(
            frame=frame, history=(), memory=memory,
            dead_signatures=dead_signatures, config=Config(),
            backend=backend, actions_taken=0,
        )
        assert decision.action.name == ActionName.ACTION6
        coordinates.add((decision.action.x, decision.action.y))
        sources.add(decision.source)
        # Matches agent/my_agent.py's real outcome-feedback wiring, driven
        # by the confirmed-live finding: zerx/transitions.py reports
        # effective=True every step on vc33 regardless of the click target.
        assert decision.target_object_label is not None
        target = next(
            obj for obj in __import__("zerx.perception", fromlist=["perceive"])
            .perceive(frame).objects
            if obj.label == decision.target_object_label
        )
        dead_signatures.record_outcome(target, effective=True)
    assert len(coordinates) == 1, (
        "expected zero coordinate variation -- if this now fails, "
        "DeadSignatureTracker's penalty mechanism started diversifying "
        "the choice, which would mean exp-150's fix (or an equivalent) "
        "landed and this characterization test should be revisited"
    )
    assert sources <= {"heuristic", "fallback_heuristic"}
```

- [ ] **Step 2: Run them**

Run: `.venv/Scripts/pytest.exe tests/test_real_game_regression.py -v -k "fallback_loop"`
Expected: both PASS, confirming the written findings above are encoded correctly (if either fails, the investigation write-up above is wrong somewhere and must be corrected before proceeding — do not adjust the test to match unexpected output without first re-reading the relevant code path).

- [ ] **Step 3: Run the full suite one more time**

Run: `.venv/Scripts/pytest.exe tests/ -q`
Expected: 261 + all new tests in this file pass (28 total: 25 sweep + 2 characterization + none removed — record the actual final count here once run, matching this plan's "Sweep result" section).

- [ ] **Step 4: Commit**

```bash
git add tests/test_real_game_regression.py
git commit -m "test(regression): lock in both confirmed fallback-loop mechanisms (ls20, vc33)"
```

---

## Task 3: `docs/HANDOFF.md` status update, final verification, push

**Files:**
- Modify: `docs/HANDOFF.md` (one-line status addition only, under the existing `baseline-120` tracking area — do not rewrite the file)

- [ ] **Step 1: Add the status line**

Append a new subsection (do not edit any existing content) documenting: branch, base commit, crash-safety sweep pass/fail summary across all 25 games, the measured wall-clock cost from Task 1 Step 2, and the one-sentence root-cause conclusion from "Investigation findings" above (both mechanisms, and the `scripts/play_local.py` Unicode-crash finding, flagged as out-of-scope-for-this-track).

- [ ] **Step 2: Commit**

```bash
git add docs/HANDOFF.md
git commit -m "docs(handoff): record baseline-120-local-regression track status"
```

- [ ] **Step 3: Push to the track branch**

```bash
git push origin feat/baseline-120-local-regression
```

---

## Sweep result (measured, not estimated)

`.venv/Scripts/pytest.exe tests/test_real_game_regression.py -v -m slow_local_engine`:

**23 passed, 2 skipped, 0 failed — 1227.51s (20m 27s) wall-clock, 5-step cap × 25 games.**

No unhandled exception escaped `MyAgent.choose_action`/`.main()` for any of the 25 public games — the `baseline-100`/`baseline-110` "no regressions" promotion criterion holds against the *full* public game set (previously only `ls20`+`vc33` had ever been checked).

Two games (`g50t`, `m0r0`) were skipped in this run because `arcade.make()` returned `None` — but re-running those exact two individually immediately afterward, both **passed cleanly** (`2 passed in 108.18s`). This points to a transient condition against the live `three.arcprize.org` API (rate limiting or a momentary fetch failure), not a permanent per-game gap — worth re-confirming in a future run rather than treating as a fixed characteristic of those two games.

Both fallback-loop characterization tests (`test_ls20_fallback_loop_is_fully_explained_by_missing_backend`, `test_vc33_fallback_loop_never_diversifies_when_transitions_report_effective`) pass, confirming the "Investigation findings" write-up above is encoded correctly — both initially caught a real bug in this plan's own test helper (`_live_frame` was reading the un-reset placeholder frame before one real step had been taken), not a wrong assumption about `zerx/policy.py`/`zerx/heuristics.py` — fixed the helper, re-verified, both passed for the right reason.

**Full-suite trade-off, as `person-3-local-regression.md` anticipated:** `pytest tests/ -q` with no `-m` filter now takes on the order of 20+ minutes (the 261 pre-existing tests run in ~110s; this file's 27 tests add the ~20-minute live-engine cost) — too slow for routine iteration. Recommend `pytest tests/ -q -m "not slow_local_engine"` (or `--ignore=tests/test_real_game_regression.py` to also skip the two fast characterization tests) as the fast, every-commit command, and reserve the unfiltered run for pre-push/pre-merge verification, matching this repo's existing `cerebras_live` precedent.

---

## Self-review notes

- **Spec coverage:** both `person-3-local-regression.md` deliverables covered — Task 1 (25-game crash-safety sweep, hardcoded step cap with measured justification) and Task 2 (root-cause characterization encoded as tests, with the plan's own prose write-up satisfying "a written finding... stating plainly"). The file's "conditionally: zerx/heuristics.py" branch is explicitly *not* used, with the reasoning (no bug in that file; the real gap is in `zerx/transitions.py`, unowned this round, already scoped to `exp-150`) written out rather than silently skipped. The `scripts/play_local.py` Unicode crash is a genuinely new finding outside this track's ownership — recorded, not fixed.
- **Placeholder scan:** no TBD/TODO; the one deliberate "fill in" (Sweep result) is explicit about being filled during execution, matching `person-3-local-regression.md`'s own "run it, don't estimate" requirement — not a plan-writing shortcut.
- **Type consistency:** `decide()`'s call signature (`frame`, `history`, `memory`, `dead_signatures`, `config`, `backend`, `actions_taken`) matches its real definition in `zerx/policy.py` exactly, used identically in both Task 2 tests; `_to_game_frame`/`MyAgent` imports match `agent/my_agent.py`'s real, unmodified names.
