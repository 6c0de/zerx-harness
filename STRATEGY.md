# Zerx Strategy for ARC-AGI-3

Updated: 2026-08-04
Tycho revision inspected: [`f68912a`](https://github.com/NIMI-research/Tycho/tree/f68912a764372ead0a610db2e1c011d41ce5197e) — verified against the live repo (README, reported scorecards) before adoption.

This file records prior-art analysis, adoption decisions, trade-offs, and
experiment sequencing. It does not override the operating, safety,
ownership, evaluation-integrity, or deployment rules in
[`AGENTS.md`](AGENTS.md). Where this file proposes module names, defer to
the actual names already in use — see
[`docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md`](docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md)
for what's real in this repository.

## Decision

Keep the existing Zerx organization and its safe single-action baseline. Adopt Tycho's evidence discipline now; add executable world models only as an isolated, optional experiment after the Gemma baseline is stable.

The recommended progression is:

1. deterministic one-action Gemma policy;
2. transition ledger and exact replay evidence;
3. reflection memory and object-level click candidates;
4. graded ineffective-action evidence;
5. optional textual hypothesis verifier;
6. optional bounded executable world model and planner;
7. actor-requested specialist builder only if ablation justifies its cost.

Do not copy source-assisted state cloning, hidden engine inspection, or free search against the real implementation.

## Prior-art comparison

| System | Core method | Distinctive strength | Main limitation | Zerx decision |
|---|---|---|---|---|
| ReKi | Gemma vision policy, recent frames, reflection, short plans, click heuristics, dead signatures | Simple, transferable interface-only policy | Natural-language beliefs can drift; coordinates and long plans remain fragile | Adopt core after minimal baseline |
| Murad/Forge VLM | Configurable Gemma policy with candidates, confidence, descriptors, repair, optional arbiter | Demonstrates value of configuration and ablation | Best profile reportedly disabled much extra machinery; added calls raise latency and failure surface | Keep arbiter and multi-candidate generation off by default (already our design — see baseline spec) |
| ProjectForty2 FORGE | Imports game implementation, clones state, searches with BFS/A*/IDDFS, inspects hidden fields; CNN fallback | Converts interaction into deterministic graph search | Depends on privileged simulator/internal state and weakly represents general hidden-game reasoning | Reject implementation access; retain only generic lessons about state identity, replay, bounded search, and transfer |
| Tycho | One growing tool-using conversation, durable evidence workspace, optional executable model, exact transition/outcome verification, bounded planning, optional builder | Connects hypotheses to falsifiable predictions; model is advisory; clean policy ablations | Very high inference/tool budget, complex harness, model-dependent results, public-game evaluation only | Adopt evidence contracts first; gate executable modeling and builder behind experiments |

Note: "Murad/Forge VLM" and "ProjectForty2 FORGE" are two unrelated systems
that happen to share the word "Forge" — earlier Zerx documents used the bare
name "Forge" for the former. This file's naming is canonical going forward;
see the baseline spec's prior-art section.

## What Tycho does differently

Tycho does not require the policy model to emit a compact JSON analysis-and-action object immediately. The actor has a persistent per-game conversation and filesystem workspace. It can inspect exact grids, diffs, frames, animations, prior attempts, and durable notes, then commits exactly one scored environment action. This is a richer scientific workspace than ReKi or Murad/Forge VLM's prompt-local reflection.

Its optional `world_model.py` is a deterministic model with:

- `State` for visible and latent variables;
- `init_state(grid0, level)`;
- `transition(state, action)`;
- `render(state)`;
- `outcome(state)`;
- optional action, subgoal, heuristic, and planner-state hooks.

Each semantic edit is automatically checked against recorded transitions. Verification reports exactness, known-cell accuracy, prediction coverage, first divergence, and outcome errors. A planner can search the learned model, but candidate plans are replay-validated and remain advisory. See Tycho's [architecture](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/docs/ARCHITECTURE.md) and [world-model tools](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/tycho/workspace/wmlib_template.py).

Tycho compares four policies on the same harness, on the **25 public games**
(not the hidden competition leaderboard — this is the critical caveat, see
below). Its README reports:

| Policy | Model | RHAE (public games) |
|---|---|---:|
| No world model | Claude Opus 4.8 | 79.07 |
| Single actor model | Claude Opus 4.8 | 85.36 |
| Actor-controlled builder | Claude Opus 4.8 | 88.49 |
| Falsification-triggered builder | Claude Opus 4.8 | 83.07 |
| Actor-controlled builder | GPT-5.6 Sol | 100.00 |
| Actor-controlled builder | Claude Opus 5 | 100.00 |

**Do not confuse these numbers with the hidden competition leaderboard.**
These are scores on the 25 games Tycho (and everyone else) can iterate
against directly — that's a much easier setting than the private evaluation
set the actual leaderboard scores against (where, as of early August 2026,
frontier closed models scored well under 1%). This supports optional,
actor-controlled formalization as a real capability signal, but it is not
evidence of private-game generalization, and the runs are stochastic,
expensive, and model-specific. Sources: [Tycho README/results](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/README.md), [reproduction caveats](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/docs/REPRODUCING.md), and [metrics artifact](https://github.com/NIMI-research/Tycho/blob/f68912a764372ead0a610db2e1c011d41ce5197e/artifacts/appendix_metrics.json).

## Adopt now

### 1. Evidence-first transition ledger

Every action must produce a durable record linking its cause and consequence. In Zerx terms (see the local-skeleton plan's `zerx/transitions.py` task) this is built from types already in `zerx/types.py` — `GameFrame`, `Action` — plus a hash/diff of the grid:

```python
@dataclass(frozen=True)
class TransitionRecord:
    step: int
    before_hash: str
    action: Action
    after_hash: str
    changed_pixels: int
    change_bbox: tuple[int, int, int, int] | None
    legal_before: frozenset[ActionName]
    legal_after: frozenset[ActionName]
    score_delta: int
    terminal: bool
    repeated_state: bool
```

(Zerx's version omits Tycho's `level`/`level_delta` fields until Task 1's
inspection of the real upstream frame API confirms a level number is
actually exposed — don't fabricate a field we can't populate honestly.)

Store exact frames separately; the ledger is an index and summary, not a replacement. Use it for no-op detection, loop detection, reflection evidence, and hypothesis testing. This is baseline infrastructure, not an advanced policy feature — it must exist even when memory, world modeling, and planning are off, because it enables fair post-run diagnosis regardless of which features are ablated.

### 2. Durable belief separation — deferred, not skipped

Tycho keeps compact fields for confirmed rules, working hypotheses, rejected hypotheses, open questions, failures, and the current plan. This is real and valuable, but it's Phase-3/4 material here (see the experiment ladder below) — it replaces `zerx/memory.py`'s simple free-text `MemoryState`, which should ship as-is for the minimal baseline first. Don't rewrite `memory.py` before `baseline-100-minimal` and `baseline-110-evidence` are stable.

### 3. Verification as a general contract — deferred, not skipped

Before executable modeling, a textual/structured verifier should ask: which transitions support this rule, which contradict it, what next low-risk action best distinguishes competing rules, is a proposed plan valid only from a named state hash. This captures much of Tycho's value without requiring Gemma to write Python reliably — but it depends on the belief-separation structure above, so it's also Phase 4, not baseline.

### 4. Focused click actions

Convert the 64×64 click space into salient components: center, bounding box, color, area, rarity, compactness, signature, prior failures/successes, and a score explanation. Never plan over all 4,096 cells. **Zerx already does this** — `zerx/heuristics.py`'s `rank_click_candidates` reduces to labeled-object centers, not raw grid cells. No change needed here beyond the soft-affordance upgrade below.

### 5. Soft negative affordances

Do not permanently ban an object signature after one failure. Track evidence by level/state context and down-rank gradually. Clear or reduce the penalty after a success. **This changes `zerx/heuristics.py`'s `DeadSignatureTracker`** from a hard exclusion set to a graded, decaying penalty — see the local-skeleton plan's updated Task 5. Unlike belief separation/verification, this is a small, self-contained change to a module that hasn't been built yet, so there's no reason to ship the inferior hard-exclusion version first.

## Adopt later as isolated experiments

### Executable world model

Add behind `world_model_on=False`. The minimum contract should be deterministic, sandboxed, network-disabled, resource-bounded, and unable to access the ARC engine or repository internals. Verify every claimed transition against observed history and report coverage so abstention cannot masquerade as accuracy.

Planning is permitted only when:

- transition and outcome predictions are sufficiently supported;
- the plan names its starting observation hash;
- branching and node budgets are bounded;
- clicks come from focused candidates;
- the candidate plan is replayed through the learned model;
- the actor may ignore it.

### Builder specialist

If Gemma can use tools reliably, test an actor-requested world-model builder after the single-actor model experiment. Do not use an automatic falsification trigger first: Tycho's own Opus 4.8 evidence shows more builder calls and a *lower* overall score than actor-controlled invocation (83.07 vs 88.49). Builder output must be compact: confidence, evidence scope, first contradiction, assumptions, plan start hash, validation status, and suggested probe/action.

## Do not adopt

- hidden game source or runtime fields;
- cloning the actual engine state;
- unscored counterfactual actions in the real implementation;
- game-ID branches, public-game lookup tables, or memorized solutions;
- unbounded BFS/A*/IDDFS;
- a CNN fallback without a training/evaluation case;
- mandatory code generation before acting;
- multi-action execution queues before each action is revalidated;
- automatic builder invocation on every mismatch;
- claims based only on public-game aggregate score (see the Tycho-numbers caveat above — this is exactly the mistake to avoid).

These are also codified as hard rules in `AGENTS.md`'s scope section.

## Experiment ladder and acceptance gates

| ID | Change from previous | Promote when |
|---|---|---|
| `baseline-100-minimal` | One Gemma call, one legal action, parse/repair/fallback, trace | End-to-end stable and reproducible |
| `baseline-110-evidence` | Transition records, exact diffs, repeated-state/no-op detection | No regressions; evidence is complete |
| `baseline-120-reki-core` | Reflection + click proposals + soft failure memory | Better completion/action efficiency on held-out seeds/games |
| `baseline-130-hypothesis` | Structured claims and contradiction/probe checks | Fewer repeated probes and belief reversals |
| `exp-200-world-model` | Optional executable model + verifier, no planner | Useful prediction coverage and accuracy without action regression |
| `exp-210-planner` | Bounded planner over verified model | Plans validate and improve efficiency after full inference cost |
| `exp-220-builder` | Actor-requested specialist builder | Beats single actor at matched budget and reliability |

Compare per game and per seed. Record completion, RHAE/action efficiency, resets, scored actions, model calls, latency, malformed outputs, no-ops, repeated states, inference cost, verification exactness, coverage, planner recommendations, validated plans, and plan-follow rate. Never promote on one aggregate public score alone.

The local-skeleton plan (Tasks 1–15) implements `baseline-100-minimal` plus
the transition-ledger portion of `baseline-110-evidence`. `baseline-120`
onward are follow-on plans, written once this one is built and green.

## Final architecture boundary

The Zerx action path remains:

`GameFrame → legal actions → perception/evidence → optional memory/model advice → one model response → deterministic extraction/repair → legality validation → one action → transition record`

`agent/my_agent.py` remains thin. New capability belongs in `zerx/` modules and must be independently disableable.
