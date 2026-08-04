# Zerx Strategy for ARC-AGI-3

Updated: 2026-08-04

## 1. Purpose and sources

This file is the strategic direction for the Zerx ARC-AGI-3 agent. It complements, rather than replaces:

- [`AGENTS.md`](AGENTS.md), the authoritative operating contract for coding agents;
- [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md), ownership, handoff, promotion, and the 5-day delivery schedule;
- [`docs/superpowers/specs/2026-08-03-arc-agi3-baseline-design.md`](docs/superpowers/specs/2026-08-03-arc-agi3-baseline-design.md), the initial architecture;
- [`docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md`](docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md), the current implementation sequence (Tasks 1–15, `baseline-100-minimal` plus the transition-ledger portion of `baseline-110-evidence`);
- experiment records, which determine whether an idea is retained, reverted, or investigated.

Five prior-art systems were inspected, as evidence, not as code to copy — their ideas are reimplemented in the Zerx architecture, isolated behind configuration, tested, measured, and retained only when repeated comparisons support them:

- **ReKi** (`milestone1-2nd-solution.ipynb`) — Gemma vision policy with memory, short plans, click-path heuristics;
- **Murad/Forge VLM** (`arc-agi-3-lb-0-86-3rd-place-candidate-milestone.ipynb`) — same core structure, wrapped in a configurable candidate-generation/arbitration surface;
- **ProjectForty2 FORGE** (`forge-arc-agi-3-agent.ipynb`) — source-assisted state cloning, BFS/A*/IDDFS search, CNN fallback (an unrelated system to Murad/Forge VLM — see the naming note below);
- **Tycho** ([NIMI-research/Tycho](https://github.com/NIMI-research/Tycho), commit [`f68912a`](https://github.com/NIMI-research/Tycho/tree/f68912a764372ead0a610db2e1c011d41ce5197e), verified against the live repo before adoption) — persistent tool-using actor, durable evidence workspace, optional executable world model with exact verification, bounded planning, optional builder;
- **Duck** (Tufa Labs, `tufa-labs-duck-harness-june-30-milestone-winner.ipynb`, the June 30, 2026 milestone's #1 solution) — a persistent tool-calling conversation where the model writes ephemeral Python that inspects deterministic object/transition evidence and calls `action(...)` from inside a sandboxed script.

Note: "Murad/Forge VLM" and "ProjectForty2 FORGE" are two unrelated systems that happen to share the word "Forge" — earlier Zerx documents used the bare name "Forge" for the former. This file's naming is canonical.

The strategic objective is not architectural novelty. It is the highest reproducible Kaggle-valid RHAE score obtainable under the project's real constraints (5-day window, Gemma-4-31B only, no hidden-state access). Since RHAE rewards completion *and* action efficiency, the agent must learn enough about a game to act correctly while avoiding unnecessary interaction. Every component leaves room for later refinement, replacement, or removal.

---

## 2. Strategic principles

### 2.1 Optimize measured performance, not apparent sophistication

A larger agent stack is not automatically a stronger agent. Murad/Forge VLM's notebook contains multi-candidate generation, confidence fields, frame descriptors, click-failure radii, and an optional LLM arbiter — yet its own *selected, submitted* profile explicitly disables all of them (one candidate, arbiter off, confidence prompt off, frame descriptor off, click-failure radius zero). This is strong evidence that additional reasoning machinery can increase latency and failure surface without improving action quality. Zerx begins with the simplest complete VLM policy and adds complexity only through controlled experiments.

### 2.2 Separate uncertain reasoning from deterministic enforcement

Gemma handles tasks where semantic visual reasoning is valuable: inferring the controllable object, inferring action effects from transitions, forming and revising goal hypotheses, choosing between exploration and execution, proposing the next action. Deterministic code handles tasks where reliability matters more than open-ended reasoning: frame normalization, connected components, click-candidate generation, unchanged-transition detection, repeated-state/failed-action tracking, JSON parsing/repair, legality validation, coordinate clamping, terminal/reset handling, time/config boundaries, experiment metadata. This split is common to every successful notebook inspected and is a core Zerx rule (see `zerx/policy.py` vs. `zerx/heuristics.py`/`zerx/perception.py`/`zerx/transitions.py`).

### 2.3 Treat each interaction as both control and experiment

An action can produce direct progress or reveal a causal rule (or both). The agent should favor actions with high expected progress or high information value, while penalizing repeated, irreversible, or already-disproved actions:

```
U(a) = E[progress | a] + λ·I(a) − μ·R(a) − ν·D(a)
```

where `I(a)` is expected information gain, `R(a)` is irreversible/terminal risk, `D(a)` is redundancy with previously-failed tests. The baseline does not compute this numerically — it's encoded in prompts, memory, candidate scoring, and (later) phase control.

### 2.4 Maintain an explicit distinction between evidence and hypothesis

Memory must not store every model statement as fact. Distinguish: **observed transition** (what changed after an action), **confirmed rule** (supported by repeated or discriminating evidence), **working hypothesis** (plausible, unverified), **rejected hypothesis** (contradicted), **open question**, **current plan**. This reduces self-reinforcing hallucination in reflection memory. (Deferred to `baseline-130-hypothesis` — see §7 — `zerx/memory.py` ships as simple free-text first.)

### 2.5 Prefer purposeful state novelty over blind novelty

A new frame is not necessarily progress — animation, timers, cursor blinking, and irrelevant motion can change pixels without improving the state; conversely an action may change hidden state while the visible frame barely changes. The baseline uses visible transition evidence (`zerx/transitions.py`'s `changed_pixels`/`score_delta`) while acknowledging partial observability. Never equate raw pixel difference with success — richer state signatures (Duck's HUD-vs-gameplay classification, see §5.5) are a documented later refinement, not a baseline assumption.

### 2.6 Keep all strategic features ablatable

Every material behavior has a typed configuration field and an experiment hypothesis: memory on/off and refresh interval, context-frame count, perception format, click-heuristic on/off, dead-signature filtering on/off, plan-queue length, thinking mode, candidate count, arbiter on/off, confidence-prompt on/off, descriptor on/off, fallback strategy, exploration/execution thresholds, deterministic-vs-model reflection, temperature/token limits. A feature that can't be isolated can't be credited for an improvement and is hard to safely remove — see `zerx/config.py`'s `Config` and `Config.from_env`.

---

## 3. What the inspected notebooks actually implement

### 3.1 ReKi — Gemma vision policy with memory, plans, and click-path improvements

`milestone1-2nd-solution.ipynb`'s `MyAgent` starts a local vLLM OpenAI-compatible server for Gemma-4-31B. Its loop: observe latest frame → update prior-action transition data → periodically reflect → attempt to dequeue a planned action → otherwise build a multimodal prompt from recent labeled images, transition history, legal actions, ineffective actions, and reflection memory → request one JSON object with a plan summary and 1–4 actions → bounded JSON repair → normalize/validate → queue the plan → execute the first action → log the transition/trace → fall back to a legal action on failure.

Notebook constants (informative starting points, not defaults to copy blindly): `MAX_HISTORY=12`, `MAX_FRAME_MEMORY=11`, `ACTION_CONTEXT_FRAMES=4`, `REFLECTION_INTERVAL=10`, `MAX_REFLECTION_CHARS=1800`, `MAX_PLAN_ACTIONS=4`, `MAX_NEW_TOKENS=1024`, `REPAIR_MAX_NEW_TOKENS=256`, `FRAME_BORDER_IGNORE=3`. It labels chronological images and tells the model to ignore the outer frame border and trust numeric transition summaries over unsupported visual guesses.

**Reflection memory.** Accumulates transition records; after `REFLECTION_INTERVAL` steps, asks the model to rewrite a compact strategic memory, treated as authoritative-but-revisable. *Advantages:* reduces rediscovery, compresses history, gives the policy a place to revise its world model, works without training. *Disadvantages:* the reflection model can convert a weak hypothesis into confident prose; an incorrect summary persists and biases later actions; consumes wall-clock/inference budget; a fixed interval may fire too early or too late; free text is hard to verify.

*Zerx application:* `zerx/memory.py`, **structured, not one paragraph** — target schema (this is `baseline-130-hypothesis`'s deliverable, not the current baseline's `MemoryState`):

```json
{
  "confirmed_rules": [],
  "working_hypotheses": [],
  "rejected_hypotheses": [],
  "open_questions": [],
  "current_goal": "",
  "current_plan": [],
  "notable_failures": []
}
```

The rendered prompt may include a compact textual form; the stored source of truth stays machine-readable. Reflection resets or correctly partitions between games. Both deterministic summarization and model-generated reflection stay viable experiments — the design must not couple the rest of the pipeline to one summarizer implementation (our `zerx/memory.py`'s `Summarizer` callable already achieves this).

**Short multi-action plans.** Reki returns 1–4 actions, executes the first, queues the rest, revalidating legality/ineffectiveness before each queued execution. *Advantages:* fewer model calls, allows simple multi-step paths, may help under strict runtime limits. *Disadvantages:* the world can change after the first action, invalidating the rest; a mistaken plan wastes several actions before replanning; uncertain semantics compound error; queued clicks can go stale; fewer model calls ≠ better RHAE if the queue is poor.

*Zerx application:* `AGENTS.md` already excludes multi-action queues from baseline — correct, unchanged. A safer intermediate design for a later experiment: ask for one executable action plus an optional non-binding plan summary; replan after every transition; test queue length 2 before 4; invalidate the queue whenever the state hash, legal-action set, level, or hypothesis materially changes (see `exp-140-vlm-refinement`, §7).

**State-specific ineffective-action memory.** Records the previous frame hash and action; when the next observation has zero relevant changed pixels and no completion delta, marks that exact action ineffective *for that exact state*. Future prompts list ineffective actions and normalization skips them. *Advantages:* directly prevents repeated no-ops, easy to implement/inspect, reduces action waste (matters strongly under RHAE). *Disadvantages:* a visually-unchanged action may still update hidden state; time-sensitive games may make the same action useful later; exact hashes can be too strict; approximate hashes can merge distinct states.

*Zerx application — this is a concrete, well-specified near-term refinement, not yet built:* Zerx's `zerx/transitions.py` (Task 13 of the current plan) already computes `repeated_state` and `.effective` per transition, and `zerx/heuristics.py`'s `DeadSignatureTracker` already grades *structural* signatures. What's missing is the *narrower, more confident* exact-state layer Reki has in addition to the broad one: a small `(before_hash, action_signature) → ineffective` set, populated from `TransitionRecord.effective is False`, checked by `decide()` to suppress (not just down-rank) proposing the *literal same action from the literal same state* — since if the state is truly identical, the outcome is already known with high confidence. Recommended record shape once implemented:

```text
state_signature
action_signature
attempt_count
visible_change
level_delta
later_disconfirmed
```

Suppression should still not be a *permanent universal ban* on the underlying action/object type — only on that exact (state, action) pair — consistent with §2.5/§5's "graded, not hard" preference elsewhere. This is `baseline-115-exact-state-memory` in the ladder (§7) — small, cheap, well-specified, deliberately sequenced *after* the current plan's Tasks 1–15 land and are verified, not folded in retroactively.

**Salient click fallback.** A pure-NumPy path segments connected components and prioritizes small/rare/button-like objects for the click fallback. *Advantages:* collapses a 4096-coordinate space into a small object set, avoids clicking background, cheap, rescues malformed/empty VLM output. *Disadvantages:* small/rare doesn't always mean interactive; large regions can be valid targets; components can split/merge incorrectly; center pixel may be non-interactive; a heuristic score isn't automatically calibrated.

*Zerx application — already built* (`zerx/heuristics.py`'s `rank_click_candidates`), with one recommended prompt-side enhancement: **expose ranked candidates to the model, not just raw objects**, so Gemma can select by label instead of guessing raw coordinates. See §8's `build_prompt()` note — this is folded into the current plan (Task 12), not deferred, because it's cheap and directly reduces the coordinate-hallucination failure mode. Target candidate feature set for future scoring refinement:

```json
{"x": 31, "y": 20, "color": 8, "area": 6, "bbox": [29, 19, 33, 21], "rarity": 0.96, "compactness": 0.73, "edge_distance": 14, "signature": "..."}
```

Keep `heuristic_first` off until a calibrated threshold beats VLM selection on measured comparisons.

**Structural dead signatures.** A click target's structural signature accumulates a dead count on repeated no-change clicks; success can mark it effective again. *Zerx application — already built* as `zerx/heuristics.py`'s graded `DeadSignatureTracker` (down-rank via `record_outcome`/`penalty`, never a hard veto) — this is exactly what the strategy called for, done during the Tycho merge. Keep separate scopes in mind for a later refinement: exact state, current level, current game, transferable pattern (currently Zerx scopes by `(color, size-bucket)` signature only, game-global — level/game scoping is a documented future refinement, not baseline).

**Prompt strategy.** Useful instructions confirmed across notebooks: images are chronological; labels are metadata, not gameplay; ignore a configured border; infer the controllable object and causal effects; prefer purposeful new states over repeated ones; don't invent counters/goals; obey the exact legal-action set; avoid actions already known ineffective; use-but-revise reflection memory; return only structured JSON; use one exploratory action when uncertain.

*Zerx application:* start from a concise version, ablate prompt sections by hash — never let prompt text become undocumented configuration. Store the template in code (`zerx/policy.py`'s `build_prompt`), hash it, record the hash with every run (`Config.config_hash()` already covers config; prompt-hash tracking is part of `eval/run_ablation.py`'s `ExperimentRecord`, §8).

### 3.2 Murad/Forge VLM — configurable candidate generation and arbitration

Shares ReKi's underlying structure (Gemma-4-31B/vLLM, labeled images, reflection, transitions, plans, JSON parse/repair, legality, ineffective-state tracking) but adds a much broader configuration surface. Its **selected, submitted** profile:

```text
LLM_ACTION_CANDIDATES=1
LLM_CANDIDATE_ARBITER=0
LLM_CLICK_FAILURE_RADIUS=0
LLM_CONFIDENCE_PROMPT=0
LLM_INCLUDE_FRAME_DESCRIPTOR=0
LLM_MAX_PLAN_ACTIONS=4
LLM_REFLECTION_INTERVAL=10
```

The notebook contains complex options; the submitted profile prefers the simple path — direct evidence for §2.1.

**Multiple candidates + static scoring.** Can request several responses, score each (parse validity, confidence, plan length, reset penalty, click plausibility, usable-action presence), pick the best. *Disadvantages:* multiplies latency/tokens; diversity may be low from the same model+evidence; static scoring can pick confident-but-wrong plans; more calls raise timeout risk; can overfit public games; **the winning profile uses one candidate**. *Zerx application:* not in baseline. Later hypothesis-driven experiment (`exp-140-vlm-refinement`): "At ambiguous states with ≥2 plausible action types, 3 low-cost candidates plus deterministic legality/risk scoring improves solve rate enough to justify latency" — test candidate count 1 vs 2/3 under identical seed/temperature/tokens, measuring quality, latency, timeout rate, parse-failure rate, diversity, per-game RHAE, and single-game overfitting.

**LLM arbiter.** Another multimodal call chooses between candidates. *Disadvantages:* another expensive call, same-model correlated biases, may prefer eloquent reasoning over effective action, another JSON/failure path, less time for actual play, **the winning profile disables it**. *Zerx application:* `arbiter_on=False` by default (already Zerx's design). Before testing it: implement a deterministic candidate scorer, collect candidate traces, only add an LLM arbiter if traces show recurring cases where valid candidates exist but deterministic selection picks wrong.

**Confidence prompting.** Optional self-reported 0–1 confidence; below-threshold triggers a reversible diagnostic action instead of a long plan. *Disadvantages:* LM confidence is usually uncalibrated; may reward overconfidence during selection; **disabled in the winning profile**; a single scalar hides different uncertainty types. *Zerx application:* don't use self-reported confidence as a control threshold initially. If collected, split by type (`control_object_confidence`, `action_semantics_confidence`, `goal_confidence`, `next_action_confidence`) and prefer deterministic proxies (repeated-transition agreement, hypothesis consistency, unexplained-change count, prior failure of the same action, candidate strength, repeated-state). Calibrate against observed outcomes before any confidence-driven behavior.

**Frame descriptor.** A compact deterministic visual-statistics blob appended to the prompt. *Disadvantages:* extra tokens; poor descriptors distract; hand-engineered stats may not match game semantics; **disabled in the winning profile**. *Zerx application:* `zerx/perception.py`'s ASCII grid + labeled-object table is already a more interpretable descriptor than a generic stats blob. Test perception formats independently (labeled images only / ASCII only / images+object-table / images+diff / full hybrid) rather than assuming "more modalities = better."

**Click-failure radius.** Avoid clicks within a configurable radius of prior failures; **the winning profile sets it to zero**. *Zerx application:* prefer component-aware failure memory (already Zerx's `DeadSignatureTracker`, scoped by signature not raw coordinate) over a fixed global radius — a small radius may remain a future experiment, not the primary mechanism.

**Strategic conclusion:** candidate generation, arbitration, confidence, and descriptors are all plausible but none are free, and the highest-scoring submitted profile disabled all of them. Zerx preserves the ability to test these (a `ModelBackend` that can return one-or-several responses without forcing every backend to pay multi-candidate cost) while resisting pressure to enable them without evidence.

### 3.3 ProjectForty2 FORGE — source-assisted search with neural fallback

Fundamentally different from the VLM notebooks: locates a game's Python source, imports the game class, deep-copies states, hashes them, and BFS/A*/IDDFS-searches for a level-completing sequence, with hidden-field probing and a CNN fallback. Self-described as "v18," building on a v10 score and a v15 regression.

**Game-source loading + state cloning + BFS/hidden-field probing.** *Disadvantages, decisively:* depends on implementation access unavailable in private evaluation; couples the agent to internal engine structure that can change without notice; measures program introspection more than interface-limited interactive reasoning; may raise competition-validity concerns; offers no transfer to observation-only environments.

*Zerx application: **excluded, permanently**, per `AGENTS.md`'s hard "never, under any circumstance" list* — no game source, no engine cloning, no hidden-field access, no unbounded search. This is a competition-integrity and generalization decision, not a sequencing one: it never becomes a later experiment.

**What *is* transferable (the lesson, not the mechanism):**

- *Action scanning/deduplication* (probe legal actions, keep those that produce visible change, dedupe equivalent results) → Zerx already does the observation-only version of this via `zerx/heuristics.py`'s candidate generation plus `zerx/transitions.py`'s outcome feedback; no cloned-state probing needed.
- *Exact verification when a model exists* → matches Tycho's world-model verifier (§4, deferred experiment).
- *History as latent-state evidence* — identical observations can correspond to different latent states (`b_t = f(o_{t-k:t}, a_{t-k:t-1})`); useful signals: action count/order since last visible change, suspected mode/phase, inferred counters, timed delays, repeated sequences, level/reset boundaries. Zerx's bounded `history` parameter (already threaded through `decide()`) is the current, minimal version of this; richer belief-state summarization is `baseline-130-hypothesis` territory.
- *Warm-up unlock testing* — when nothing seems to change, test one action then rescan, rather than declaring the game inert. *Zerx application:* an escalation ladder for the RECOVER phase (§4, deferred): verify normalization/terminal state → retry one low-risk action once → try a different action type → try a short two-action sequence → consider click candidates → reset only with evidence the run is unrecoverable. Record the result as "possible mode unlock," never a confirmed rule from one observation.
- *Cross-level transfer* — transfer confirmed action semantics, object-role correspondences, goal patterns, and successful subgoal sequences *before* exact coordinates/plans; verify a transferred rule with one discriminating action before replaying a long sequence; disable transfer after contradictory evidence.
- *CNN fallback* — **not part of the 5-day baseline**, full stop (no training/evaluation case exists, and it's far outside the Gemma-only contract). Preserved only as long-term research ideas: action-effect embeddings, temporal-difference perception, object-level policy heads instead of a 4096-way click head, imitation from verified trajectories. Any future learned component must beat the deterministic-heuristics baseline *after* accounting for training/packaging/GPU/reproducibility cost — a very high bar given the project's Gemma-only, offline-Kaggle constraint.

---

## 4. What Tycho does differently

Tycho does not require the policy model to emit a compact JSON analysis-and-action object immediately. The actor has a persistent per-game conversation and filesystem workspace — it can inspect exact grids, diffs, frames, animations, prior attempts, and durable notes, then commits exactly one scored environment action. This is a richer scientific workspace than ReKi's or Murad/Forge VLM's prompt-local reflection.

Its optional `world_model.py` is a deterministic model with `State` (visible + latent variables), `init_state(grid0, level)`, `transition(state, action)`, `render(state)`, `outcome(state)`, and optional action/subgoal/heuristic/planner-state hooks. Every semantic edit is automatically checked against recorded transitions; verification reports exactness, known-cell accuracy, prediction coverage, first divergence, and outcome errors. A planner can search the learned model, but candidate plans are replay-validated and remain advisory. See Tycho's [architecture](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/docs/ARCHITECTURE.md) and [world-model tools](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/tycho/workspace/wmlib_template.py).

Tycho compares four policies on the same harness, on the **25 public games** (verified against the live repo README before citing):

| Policy | Model | RHAE (public games) |
|---|---|---:|
| No world model | Claude Opus 4.8 | 79.07 |
| Single actor model | Claude Opus 4.8 | 85.36 |
| Actor-controlled builder | Claude Opus 4.8 | 88.49 |
| Falsification-triggered builder | Claude Opus 4.8 | 83.07 |
| Actor-controlled builder | GPT-5.6 Sol | 100.00 |
| Actor-controlled builder | Claude Opus 5 | 100.00 |

**Do not confuse these numbers with the hidden competition leaderboard.** These are scores on the 25 games Tycho (and everyone else) can iterate against directly — a much easier setting than the private evaluation set the actual leaderboard scores against, where frontier closed models scored well under 1% as of early August 2026. This supports optional, actor-controlled formalization as a real capability signal, but it is not evidence of private-game generalization, and the runs are stochastic, expensive, and model-specific. Sources: [README/results](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/README.md), [reproduction caveats](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/docs/REPRODUCING.md), [metrics artifact](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/artifacts/appendix_metrics.json).

**Adopted now** (already built, during the Tycho merge — see the local-skeleton plan's Tasks 5/13/14): the evidence-first transition ledger (`zerx/transitions.py`) as baseline infrastructure, and graded/soft negative affordances (`zerx/heuristics.py`'s `DeadSignatureTracker`) — both cheap, both self-contained, both shipped correctly from the start rather than retrofitted.

**Deferred as isolated experiments** (`exp-200`/`exp-210`/`exp-220`, §7): durable belief separation and structured verification (both depend on `baseline-130-hypothesis`'s structured memory landing first); the executable world model itself (behind `world_model_on=False`, deterministic/sandboxed/network-disabled/resource-bounded, unable to access the ARC engine or repo internals, every claimed transition verified against observed history with coverage reporting so abstention can't masquerade as accuracy); bounded planning (only once transition/outcome predictions are well-supported, the plan names its starting observation hash, branching/node budgets are bounded, clicks come from focused candidates, the plan is replay-validated, and the actor may still ignore it); the builder specialist (test actor-requested invocation *before* automatic falsification-triggered invocation — Tycho's own Opus 4.8 evidence shows automatic triggering scores *lower*, 83.07 vs 88.49).

---

## 5. Duck (Tufa Labs) — the milestone winner's architecture, and what's safely reusable

### 5.1 Why it's a separate case

Duck won the June 30, 2026 milestone (an earlier internal version reportedly scored 1.21 RHAE; this more reproducible released version did not repeat that exact result — **treat the score as evidence of potential, not a stable controlled measurement**). Its mechanism is a persistent tool-calling conversation where the model does not return a structured action directly — it writes and runs ephemeral Python (fresh isolated process per call, small stdlib allowlist, time/output-limited, error-sanitized) that inspects deterministic segmentation and transition evidence, then calls `action(...)` from inside that script.

Released configuration (informative, not a Zerx default): `vrfai/Qwen3.6-27B-FP8` on local vLLM, 65,536-token model context, 32,768-token analyzer context, reasoning enabled, temperature 0.6/top-p 0.95/top-k 20, 4× upscaled grid images, 30s Python-tool timeout, ~1,024 tool-output tokens, one RTX Pro 6000, up to 28 concurrent games.

**Zerx decision, unchanged from before this merge:** model-written Python that can call the real environment is **excluded from baseline scope** (`AGENTS.md`) — not because it's a bad idea (it plausibly explains a chunk of Duck's win), but because it's high-risk, high-complexity, and unnecessary to reach a strong first score. What *is* reusable without going anywhere near arbitrary code execution against the live environment: Duck's object representation, its transition interface's explicit result metadata, and its "model chooses computation, deterministic code executes it" division of labor. See `exp-150-duck-tools` (§5.6/§7) for the fully specified follow-on experiment.

### 5.2 Where Duck sits relative to the other three

Direct VLM policy (ReKi) → deterministic object/transition evidence → ephemeral programmatic analysis and bounded search (Duck) → optional persistent verified world model (Tycho). Compared to Murad/Forge VLM, Duck spends extra inference on tool-mediated computation instead of multiple model proposals voting over the same evidence. Compared to ProjectForty2 FORGE, Duck searches only abstractions built from *permitted* observations — no cloned/hidden state. Compared to Tycho, Duck's code is ephemeral and its working world model is conversational, not a persistent verified artifact — lighter and more flexible, but without Tycho's formal prediction-coverage/falsification contract.

### 5.3 Deterministic object representation

Four-connected same-color segmentation, each object exposing: ID, color, pixel count, a **position-independent color-and-shape hash** (tracks an object across frames when it moves, without needing identical coordinates), clockwise outer boundary, enclosed child-object IDs, edge-sharing adjacency. The raw numeric grid is deliberately hidden from model-written Python — the model gets a compact symbolic grid for small local checks, an image for visual context, and structured objects for analysis.

*Zerx application (future — `exp-150-duck-tools` Variant A, §5.6):* extend `zerx/perception.py`'s `LabeledObject` toward a richer `SceneObject`:

```python
@dataclass(frozen=True)
class SceneObject:
    object_id: int
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    boundary: tuple[tuple[int, int], ...]
    shape_hash: str
    child_ids: tuple[int, ...]
    adjacent_ids: tuple[int, ...]
```

Keep both a shape-only hash (movement tracking) and a contextual signature (affordance evidence) — don't conflate them.

### 5.4 Transition interface and gameplay-change classification

Duck exposes `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `last_action`, `last_action_result`, `valid_actions` as explicit runtime state, with each transition linking `before frame → executed action → after frame → result metadata` (board-changed flag, reward, valid actions, level completion, game over, run completion, terminal state) — this avoids a real, common bug class where an agent compares the current frame against an already-current history entry and wrongly concludes nothing changed. It also instructs the model to distinguish meaningful gameplay changes from HUD-only changes (timers, progress bars, step counters).

*Zerx's current limitation, documented honestly:* `zerx/transitions.py`'s `TransitionRecord.effective` (`changed_pixels > 0 or score_delta != 0`) cannot currently tell a real gameplay change from a HUD-only animation — a shrinking timer bar would incorrectly count as "effective." This is a known, accepted simplification for the baseline, not an oversight — fixing it properly needs object-level correspondence (§5.3), which is explicitly `exp-150-duck-tools` scope, not baseline scope. Track it as the first Variant-A follow-on:

```text
NO_CHANGE
HUD_ONLY
OBJECT_MOVE
OBJECT_APPEAR_DISAPPEAR
RECOLOR_OR_TRANSFORM
LEVEL_BOUNDARY
TERMINAL
UNKNOWN_CHANGE
```

Deterministic where possible, explicitly uncertain otherwise — a shrinking edge bar is never, by itself, proof a puzzle action succeeded.

### 5.5 What's genuinely useful, sequenced

1. **Relational connected components** (§5.3) — representation-only, no code execution.
2. **Object correspondence across frames** — match by shape hash, color, overlap, centroid displacement, bbox proximity, area change, containment/adjacency context, edge contact; never equate identical hashes with identical semantic roles when duplicates exist.
3. **Gameplay-change classification** (§5.4).
4. **Compact, fixed analysis API** — not unrestricted repository access: list salient objects, compare two frames, find correspondences, inspect a small local crop, query transitions by action/object signature, score click candidates, bounded pathfinding over a caller-supplied abstract graph. Return compact summaries; never dump a full 64×64 grid into tool output unless a narrow diagnostic explicitly needs it.
5. **Programmatic probes** — a low-risk discriminating action derived from a small computation, with competing hypotheses/expected results/selected probe/observed outcome/which-hypotheses-survived all recorded. This is where Duck's tool use meets Zerx's evidence-first strategy directly.
6. **Bounded search, only after rule discovery** — deterministic node/time limits, a compact state representation, legal-action filtering, object-level candidates, explicit plan-start hashes, and never a claim that failure-to-find proves impossibility.

**What not to adopt initially:** unrestricted model-authored Python; action execution directly from arbitrary code; multi-action batches without state checks; unlimited tool calls per decision; a fixed 30-second search on every step; a huge context window as a substitute for structured memory; public-game-specific prompt rules presented as universal truths; position-independent hashes as the *sole* identity mechanism; silent removal of old conversational evidence; dependence on model-specific tool-call markup recovery; promoting anything based on the reported 1.21 run alone.

### 5.6 `exp-150-duck-tools` (fully specified, not yet started)

**Objective:** determine whether deterministic object evidence and a restricted programmatic analysis tool improve Gemma's action efficiency/reliability beyond `baseline-120-reki-core`.

**Baseline:** `baseline-120-reki-core` — same model, same prompt/image format, one action/decision, reflection memory, object-level click proposals, legality validation, soft ineffective-action evidence.

**Variant A — segmentation only.** Add relational connected-component summaries (§5.3), no code execution. Purpose: measure how much improvement comes from representation alone.

**Variant B — fixed analysis functions.** A fixed API for object matching, transition diffs, local crops, candidate scoring, bounded pathfinding. Purpose: measure deterministic computation without arbitrary code generation.

**Variant C — restricted ephemeral programs.** Isolated model-written Python: no network, no filesystem outside a temp dir, no ARC-engine imports, no process creation, no reflection/dynamic imports, a narrow module allowlist, strict CPU/memory/output/wall-clock limits, observation copies only, **no direct environment-action callback** — the program returns a recommendation; Zerx validates and executes at most one action outside the sandbox. Purpose: measure the added value and reliability cost of flexible code generation, without ever letting generated code touch the live environment directly.

**Variant D — state-checked short plan.** Only after Variant C succeeds: a plan of at most three actions, each requiring the current observation signature to match the plan's expected state before executing, or control returns to the policy. Purpose: test Duck-like batching without blind continuation.

**Suggested configuration:**

```text
duck_objects_on = false
duck_fixed_tools_on = false
duck_python_on = false
duck_short_plan_on = false

duck_python_timeout_s = 2
duck_python_memory_mb = 128
duck_python_output_chars = 4000
duck_max_tool_calls_per_decision = 2
duck_search_max_nodes = 5000
duck_search_timeout_s = 1
duck_plan_max_actions = 1
duck_require_state_match = true
```

Enable only one primary factor per experiment run.

**Required tests (when this experiment starts):** segmentation (border components, nested objects, adjacent same/different-color objects, duplicate shapes, holes/thin boundaries, hash stability under translation, distinct contextual identities for duplicate hashes, timer/HUD-like edge strips); transition evidence (true no-op, HUD-only change, movement, recoloring, animation frames, reset, level completion, game over, legal-action changes); sandbox (blocked network/filesystem/engine-import access, timeout/memory enforcement, output truncation, exception sanitization, deterministic serialization, invalid-action rejection, inability to execute a real action directly); plans (stale start-state rejection, early stop after terminal evidence, mismatch after the first action, invalid/newly-unavailable action, out-of-board click coordinates, branch explosion/timeout).

**Metrics:** game/level completion, RHAE/action efficiency, resets/terminal failures, model calls/tokens, tool calls/failures/timeouts/latency, generated-code syntax/runtime failure rate, scored no-ops, HUD-only actions mistaken for progress, repeated-state count, click accuracy, search plans produced/followed, state-mismatch cancellations, total compute/wall-clock cost. Promote the Duck layer only when it improves held-out performance at reasonable inference/latency cost without reducing action safety — prefer the representation-only or fixed-tool variants if they match arbitrary Python's gains.

**Recommended order:** (1) relational object segmentation, (2) object correspondence + change classification, (3) compact fixed analysis functions, (4) run Variants A and B, (5) add the isolated recommendation-only Python sandbox if evidence supports it, (6) run Variant C, (7) test state-checked short plans only after single-action tool use is reliable, (8) compare the winning Duck variant against `exp-200-world-model`.

The preferred Zerx form keeps Duck's main advantage while retaining the one-action boundary and keeping arbitrary model-written code away from the live environment:

```
Gemma selects a computation → restricted tool analyzes copied evidence →
tool returns a recommendation → Zerx validates one legal action →
environment executes it → transition evidence is recorded
```

---

## 6. Zerx baseline control loop

1. **Normalize state** — `GameFrame`, exact legal-action set, immediate terminal detection, bounded history.
2. **Perceive** — labeled image, compact grid/ASCII, connected components, object table, ranked click candidates, transition diff from the prior frame.
3. **Update evidence** — attach the previous action's result once the next frame exists (never before); record changed pixels, level/score delta, repeated-state status, legality; update exact-state and structural ineffective-action evidence; revise hypothesis confidence *only* from recorded evidence (once hypotheses exist — `baseline-130`).
4. **Determine strategic phase** — `EXPLORE` (semantics/goal unknown) / `VERIFY` (a high-value hypothesis needs a discriminating test) / `EXECUTE` (mechanics+goal well-supported) / `RECOVER` (repeated failures, malformed output, or loops). **Deferred** — `baseline-125-phase-control`, §7; needs the hypothesis structure from `baseline-130` to be meaningful, so it's sequenced after, not before it despite the number.
5. **Build one policy request** — recent labeled images plus useful compact data, legal actions, structured memory (once it exists), recent transitions, ranked click candidates when ACTION6 is legal, known ineffective actions; ask for one executable action plus a concise evidence-linked rationale; require one JSON object.
6. **Parse and validate** — deterministic extraction, at most one bounded repair, reject unavailable actions, validate ACTION6 coordinates, never assume fixed ACTION1–5 semantics, never return ACTION7 unless explicitly legal.
7. **Apply strategy constraints** — reject an exact known no-op in the unchanged state (once `baseline-115` lands), down-rank (never hard-veto) weak structural signatures, prefer a low-risk diagnostic action under high uncertainty, prefer execution when a supported plan exists, treat budget pressure as a prompt/strategy signal — never permission to invent an unvalidated move.
8. **Fallback safely** — validated model action → validated heuristic candidate → deterministic legal fallback → reset only when terminal or strategically justified.
9. **Record** — decision path, action, state/prompt hashes, memory version, candidate list, model latency, parse/repair/fallback status, and the transition result on the *next* step.

This is the same loop the current implementation plan (Tasks 1–15) already builds for steps 1, 2 (minus candidate exposure in-prompt — see §8), 3, 6, 8, 9, and most of 7 (soft down-ranking). Step 4 (phase control) and the richer parts of steps 3/5/7 that depend on structured memory are explicitly sequenced later (§7) — this loop description is the target shape across the whole ladder, not a claim that Tasks 1–15 already implement all nine steps in full.

---

## 7. Experiment ladder and acceptance gates

| ID | Change from previous | Promote when |
|---|---|---|
| `baseline-100-minimal` | One Gemma call, one legal action, parse/repair/fallback, trace | End-to-end stable and reproducible — **this + the ledger below is what Tasks 1–15 build** |
| `baseline-110-evidence` | Transition records, exact diffs, repeated-state/no-op detection, wired against real games | No regressions; evidence is complete |
| `baseline-115-exact-state-memory` | Exact `(state, action)` ineffective suppression (§3.1) alongside the existing graded structural signatures | Fewer repeated no-ops without new false suppressions |
| `baseline-120-reki-core` | Reflection (still simple) + click proposals + soft failure memory validated against real games | Better completion/action efficiency on held-out seeds/games |
| `baseline-125-phase-control` | EXPLORE/VERIFY/EXECUTE/RECOVER (§6 step 4) | Phase labels measurably change behavior, not just prompt text, without regressing the games `baseline-120` already handled |
| `baseline-130-hypothesis` | Structured claims (§2.4/§3.1's schema) and contradiction/probe checks | Fewer repeated probes and belief reversals |
| `exp-140-vlm-refinement` | Murad/Forge VLM-informed: multi-candidate, arbiter, confidence, click-failure-radius, frame-descriptor ablations (§3.2) | A stated hypothesis is confirmed with matched seed/temperature/token budgets — not "more machinery" by default |
| `exp-150-duck-tools` | Duck-informed: object segmentation → fixed tools → sandboxed recommendation-only Python → state-checked short plans (§5.6, fully specified above) | Improves held-out performance at reasonable cost without reducing action safety |
| `exp-200-world-model` | Optional executable model + verifier, no planner (§4) | Useful prediction coverage/accuracy without action regression |
| `exp-210-planner` | Bounded planner over the verified model | Plans validate and improve efficiency after full inference cost |
| `exp-220-builder` | Actor-requested specialist builder (test before any automatic trigger — see §4's 83.07-vs-88.49 evidence) | Beats single actor at matched budget and reliability |

Compare per game and per seed. Record completion, RHAE/action efficiency, resets, scored actions, model calls, latency, malformed outputs, no-ops, repeated states, inference cost, verification exactness/coverage (once relevant), planner recommendations/validation/follow-rate (once relevant), and tool-call/sandbox metrics (once relevant). **Never promote on one aggregate public-game score alone** — that's exactly the mistake the Tycho-numbers caveat (§4) and Duck's 1.21-run caveat (§5.1) both warn against.

The current local-skeleton plan ([`docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md`](docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md), Tasks 1–15) implements `baseline-100-minimal` plus the transition-ledger portion of `baseline-110-evidence`. Everything from `baseline-115` onward is a separate, not-yet-written follow-on plan — write each one only once its predecessor is built, tested, and green.

### 7.1 Decision policy for accepting a strategy change

**Retain when:** the hypothesis was stated before the run; it's isolated by configuration or branch; tests pass; the run is reproducible; it improves aggregate or strategically-important per-game results; regressions are understood and acceptable; the gain isn't solely from extra illegal information; latency/action-count/failure rates stay acceptable; Colab results reproduce anything selected via Cerebras; the experiment record concludes `keep`.

**Revert when:** it adds complexity without measurable benefit; the benefit appeared only once; it causes new legality/timeout failures; it overfits a public game; it depends on unavailable implementation details; it can't be reproduced.

**Leave `investigate` when:** aggregate results hide conflicting per-game effects; the sample is too small; the backend differs from deployment; a measurement/logging defect exists; a narrower variant might work instead.

---

## 8. Concrete mapping to the Zerx package

- **`zerx/perception.py`** — normalized grid conversion, labeled-image rendering, connected components, object descriptors, transition-diff helpers, click-candidate features. Must not own policy decisions.
- **`zerx/heuristics.py`** — candidate ranking, graded dead-signature evidence, deterministic fallback proposals, optional heuristic-first score. Must expose reasons/scores for traceability. *(Already reflects §3.1's "graded, never permanent" requirement.)*
- **`zerx/transitions.py`** — evidence-first ledger (§4/§6 step 3); not in the original notebook-derived package list, added during the Tycho merge — the concrete implementation of "record" (§6 step 9) and the input to `baseline-115`'s exact-state memory and `exp-150`'s change classification.
- **`zerx/memory.py`** — immediate transition records, level/game memory, hypothesis status (once `baseline-130` lands), reflection scheduling, versioning, reset/level-transition behavior. Must preserve evidence links and support swapping the summarizer.
- **`zerx/policy.py`** — prompt construction (including ranked click candidates — see the note below), one-action schema, model call, deterministic extraction, bounded repair, normalization, legality validation. Must not read environment variables directly.
- **`zerx/budget.py`** — observable action/wall-time telemetry, soft-cap phase signals, loop/no-progress indicators. Must not assume access to hidden human-median action counts.
- **`zerx/model_backend.py`** + **`zerx/backends/cerebras_dev.py`** — a narrow backend protocol; every backend returns the same response shape and passes through the same parser/legal guard (already built, including the Cerebras dev lane — see `AGENTS.md`'s "Cerebras development boundary").
- **`zerx/config.py`** — every strategic toggle and its serialization. Defaults reflect the simplest stable profile: memory on only once its stage is reached, heuristic-first off, arbiter off, one candidate, one executable action, bounded repair, exact legal guard always on.
- **`agent/my_agent.py`** — harness adaptation and orchestration only; must never become a second strategy-implementation layer.
- **`eval/run_ablation.py`** — configuration sweeps, repeated runs, structured result records, baseline comparison, keep/revert/investigate decision support.

**Immediate, in-scope enhancement to `zerx/policy.py`'s `build_prompt()`:** the current implementation (Task 12 of the local-skeleton plan) lists perceived objects but not the *ranked click candidates* `zerx/heuristics.py` already computes. Per §3.1's "salient click fallback" application, the prompt should show the model its top-ranked candidates (label, coordinates, score) so it can select by label instead of guessing raw coordinates — directly reduces the coordinate-hallucination failure mode `AGENTS.md` already worries about. This is small, cheap, and folded into the current plan (not deferred) — see the local-skeleton plan's updated Task 12.

---

## 9. Risks and failure modes

| Risk | Mitigation |
|---|---|
| Hallucinated mechanics | Evidence-linked hypotheses (once built); explicit contradiction tracking; a verification phase; numeric transition summaries; prompt instruction not to invent goals/counters |
| Repeated no-op actions | Exact-state ineffective memory (`baseline-115`); structural signature evidence (built); repeated-state detection (built); fallback candidate diversity |
| Overconfident action queues | Single-action baseline (current); queue invalidation; EXECUTE-only queue experiments (`exp-140`); immediate replan after unexpected change |
| Invalid model output | JSON mode; deterministic parser; one repair; strict normalization; legal fallback — all built (Task 11) |
| Visual-state aliasing | Bounded history (built); latent-mode hypotheses (`baseline-130`); action-sequence context; never assume identical frames are always identical states |
| Excessive latency | One call by default (built); bounded context; compact memory; conditional reflection; separate wall-time metrics; no arbiter by default |
| Heuristic overreach | Candidates before automatic action; heuristic-first off until calibrated (built); soft down-ranking before hard veto (built); per-game regression checks |
| Public-environment overfitting | No game-ID logic (hard rule, `AGENTS.md`); no fixed action semantics (hard rule); repeated tests; generic object/transition abstractions; per-game regression inspection |
| Backend mismatch | Cerebras results labeled as proxy (built — see `AGENTS.md`); Colab exact-model gate; Kaggle deployment source of truth; model revision/precision/package/prompt-hash recorded per run |
| Improvement stagnation | Modular interfaces; stored traces; explicit open questions; negative-result records; every component replaceable; nothing treated as permanent merely because it exists |
| HUD/animation mistaken for progress | Documented current limitation (§5.4); fix scoped to `exp-150-duck-tools` Variant A, not silently left unaddressed |
| Duck-style code execution risk | Never against the live environment — sandboxed, recommendation-only, validated by Zerx before execution, per `exp-150`'s Variant C contract |

---

## 10. Current strategic recommendation

The strongest direction is a **ReKi-inspired, single-call Gemma vision policy with deterministic perception, legality, memory, and click support** — exactly what the local-skeleton plan (Tasks 1–15) builds, plus Tycho's evidence-ledger and soft-affordance ideas folded in during the merge, plus one small Duck/Reki-informed prompt enhancement (ranked candidates in-prompt).

- Gemma-4-31B as the submitted policy.
- One executable action per call.
- Chronological visual context plus concise numeric transitions.
- Structured, revisable memory — *later* (`baseline-130`); simple free-text memory *now*.
- Exact-state ineffective-action tracking — *next, well-specified* (`baseline-115`), not yet built.
- Object-level click candidates, shown to the model, not just used for fallback — *in current scope*.
- Dead signatures as graded evidence, never a hard veto — *built*.
- `heuristic_first` off until calibrated; candidate generation and arbiter off by default — *built/planned*.
- Explore/Verify/Execute/Recover as an explicit later phase, not a baseline requirement.
- Every decision and transition recorded for ablation — *built* (`zerx/transitions.py`).
- Planning only after observed mechanics support it (`exp-200`/`exp-210`).
- Direct game-source introspection rejected, permanently, not as a sequencing choice.
- Interfaces left open for later world models, planners, learned rankers, Duck-style tool use, and better reflection systems — none of it scaffolded prematurely.

This is not a frozen design. Every module, prompt, threshold, memory rule, and planning mechanism stays open to evidence-based improvement. The project should keep asking not only whether a feature works, but why, where it fails, whether something simpler matches it, and what the next experiment should be.

---

## 11. Notebook reference map

Use these names in experiment records so it's precise which prior-art idea is being tested.

**`milestone1-2nd-solution.ipynb` (ReKi):** `MyAgent.choose_action()` (plan dequeue, model call, repair, fallback, tracing); `_build_prompt()` (legal/ineffective actions, reflection, transitions); `_run_reflection()`/`_build_reflection_prompt()` (periodic memory); `_normalize_action_specs()`/`_dequeue_action()` (queue validation); `_observe_frame()`/`_changed_pixels()` (transition evidence); `_ineffective_actions_for_current_state()` (state-specific no-op memory); `_grid_components()`/`_grid_saliency()`/`_salient_click_coordinate()` (click ranking); `_seg_sig_at()`/`_record_deadsig()` (structural dead-click tracking); `_build_context_images()`/`_label_image()` (chronological visual context).

**`arc-agi-3-lb-0-86-3rd-place-candidate-milestone.ipynb` (Murad/Forge VLM):** top-level `PROFILE_ENV` (the disabled-everything winning profile); `_generate_action_response()` (candidate generation); `_action_candidate_count()`; `_candidate_static_score()`; `_select_candidate_with_arbiter()`; `_confidence_prompt_enabled()`; `_include_frame_descriptor()`/`_frame_descriptor()`; `_click_failure_radius()`/`_click_near_failed()`; shared reflection/plan/parsing/legality/history/tracing methods.

**`forge-arc-agi-3-agent.ipynb` (ProjectForty2 FORGE):** `find_game_source_and_class()`; `BFSSolver.load()`; `_scan_actions()`; `_state_hash()`; `_probe_hidden_fields()`; `solve_level()`; `_try_transfer()`; `ForgeNet`/`CBAM`/`ActionEffectAttention` (CNN fallback); `_tensor()`/`_frame_to_tensor()`; `_train()`; the CLTI demonstration-injection block; `choose_action()` (BFS-first, CNN fallback).

**Tycho (NIMI-research/Tycho, commit `f68912a`):** `docs/ARCHITECTURE.md`; `tycho/workspace/wmlib_template.py` (`State`/`init_state`/`transition`/`render`/`outcome` contract); README results table.

**Duck (Tufa Labs, `tufa-labs-duck-harness-june-30-milestone-winner.ipynb`, plus the `jeroencottaar/taaf-kaggle-source-share` v4 dataset, `ARC3-Inference@aa69123`, `tufa-arc-agi-framework@fe9f7c4`):** the tool-calling loop (segmentation inspection → transition inspection → world-model revision → scorer/probe/pathfinder authoring → `action(...)` call → evidence refresh); the object schema (§5.3); the transition-result metadata (§5.4).

---

## 12. Do not adopt (hard rules, codified in `AGENTS.md`)

- hidden game source or runtime fields;
- cloning the actual engine state;
- unscored counterfactual actions against the real implementation;
- game-ID branches, public-game lookup tables, or memorized solutions;
- unbounded BFS/A*/IDDFS;
- a CNN fallback without a stated training/evaluation case;
- mandatory code generation before acting;
- multi-action execution queues before each action is revalidated;
- automatic builder invocation on every mismatch (test actor-requested first — §4);
- unrestricted model-authored Python with a direct environment-action callback (Duck's *released* form — §5.6's Variant C is the safe version);
- claims based only on public-game aggregate score (§4, §5.1 — the exact mistake both the Tycho and Duck numbers could invite if quoted carelessly).

These are hard, permanent rules, not sequencing choices — unlike everything in §7's ladder, they don't become "later experiments."

---

## Final architecture boundary

The Zerx action path:

`GameFrame → legal actions → perception/evidence → optional memory/model advice → one model response → deterministic extraction/repair → legality validation → one action → transition record`

`agent/my_agent.py` stays thin. New capability belongs in `zerx/` modules and must be independently disableable.
