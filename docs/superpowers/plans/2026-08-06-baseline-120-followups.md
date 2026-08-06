# `baseline-120` Follow-ons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a per-step trace-capture/export system, a pygame live+replay
visualizer built on it, and a project README (including `ARC_API_KEY`
documentation), per
`docs/superpowers/specs/2026-08-06-baseline-120-followups-design.md`.

**Architecture:** `zerx/policy.py`'s `Decision` gains an optional
`raw_response` field, populated by `decide()`. A new `zerx/trace.py`
module defines the trace data model (`TraceMeta`, `TraceStep`) and two
pure recorders (`JsonlTraceWriter`, `CompositeTraceRecorder`).
`agent/my_agent.py` gets a public `trace_recorder` attribute, wired from
a new `Config.trace_export_path` field and called once per step. A new
`scripts/visualize_play.py` adds a `pygame`-based `LivePygameRecorder`
(pygame code stays out of `zerx/`) with `--live` and `--replay` modes
sharing one render/navigate path. `README.md` documents all of it,
including the no-code-change `ARC_API_KEY` finding.

**Tech Stack:** Python 3, pytest, dataclasses, `pygame` (new, dev-only
dependency, never imported under `zerx/`).

## Global Constraints

- Base commit: `master` at `b405d3b`; branch `feat/baseline-120-followups`.
- Only `zerx/config.py` reads environment variables directly — every other
  module receives a resolved `Config` (`AGENTS.md`).
- Every new behavior defaults to off/unchanged when no new env var is set
  — no existing test's behavior may change (`AGENTS.md`, matches every
  prior `baseline-120` track's own constraint).
- `zerx/trace.py` and the `Decision`/`decide()`/`Config` changes must stay
  pygame-free and fully unit-testable; pygame code lives only in
  `scripts/visualize_play.py`, never under `zerx/`, so `pygame` never
  enters the Kaggle bundle (`scripts/build_notebook.py` bundles
  `zerx/*.py` only) or `zerx/secret_scan.py`'s scope.
- Commit after every task. Full suite (`.venv/Scripts/pytest.exe tests/ -q`
  on Windows) must stay green after each task — use
  `-m "not slow_local_engine"` for fast iteration, per Track 3's own
  documented convention, and confirm the unfiltered run at the end.
- No placeholders, no speculative generality beyond what this plan and the
  approved spec describe.

---

### Task 1: `Decision.raw_response` + `decide()` threading

**Files:**
- Modify: `zerx/policy.py:79-85` (`Decision`), `zerx/policy.py:214-268`
  (`decide()`'s single-candidate branch and its return statements)
- Test: `tests/test_policy_decide.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Decision.raw_response: Optional[str] = None` — every later
  task (`zerx/trace.py`'s `describe_reasoning`, `agent/my_agent.py`) reads
  this field.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policy_decide.py`:

```python
def test_decide_populates_raw_response_on_successful_model_action():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert decision.raw_response == '{"action": "ACTION1"}'


def test_decide_populates_raw_response_even_when_parse_fails():
    frame = _frame([[0, 0], [0, 5]])  # one clickable object -> fallback_heuristic
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION0"}']),  # invalid name
        actions_taken=0,
    )
    assert decision.source == "fallback_heuristic"
    assert decision.raw_response == '{"action": "ACTION0"}'


def test_decide_raw_response_is_none_when_no_model_call_happens():
    decision, _ = decide(
        frame=_blank_frame(is_game_over=True),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),
        actions_taken=0,
    )
    assert decision.raw_response is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_policy_decide.py -v -k raw_response`
Expected: FAIL — `Decision.__init__() got an unexpected keyword argument
'raw_response'` does not occur yet (the field doesn't exist), so the
`assert decision.raw_response == ...` lines fail with `AttributeError`.

- [ ] **Step 3: Widen `Decision`**

In `zerx/policy.py`, change:

```python
@dataclass(frozen=True)
class Decision:
    action: Action
    source: str  # "model" | "heuristic" | "fallback_heuristic" | "fallback_deterministic" | "fallback_random" | "reset"
    repaired: bool = False
    budget: Optional[BudgetSignal] = None
    target_object_label: Optional[str] = None
```

to:

```python
@dataclass(frozen=True)
class Decision:
    action: Action
    source: str  # "model" | "heuristic" | "fallback_heuristic" | "fallback_deterministic" | "fallback_random" | "reset"
    repaired: bool = False
    budget: Optional[BudgetSignal] = None
    target_object_label: Optional[str] = None
    raw_response: Optional[str] = None  # the model's raw text, when a
    # model call happened this step -- populated even on a failed parse,
    # so tooling (zerx/trace.py) can show what the model actually said.
```

- [ ] **Step 4: Thread `raw_response` through `decide()`**

In `zerx/policy.py`, replace the whole block from `if
config.candidate_count > 1:` through the end of the function with:

```python
    raw_response: Optional[str] = None
    if config.candidate_count > 1:
        try:
            from zerx.candidates import generate_candidates, select_candidate

            prompt = build_prompt(perception, new_memory, candidates)
            model_candidates = generate_candidates(
                backend, prompt, legal_actions, config.candidate_count
            )
            best = select_candidate(model_candidates, config)
            parsed = best.parsed if best is not None else None
        except Exception:
            parsed = None
    else:
        try:
            raw_response = backend.generate(build_prompt(perception, new_memory, candidates))
            parsed = parse_action(raw_response, legal_actions)
        except Exception:
            parsed = None

    if parsed is not None:
        return (
            Decision(
                action=parsed.action,
                source="model",
                repaired=parsed.repaired,
                budget=budget,
                raw_response=raw_response,
            ),
            new_memory,
        )

    if candidates and ActionName.ACTION6 in legal_actions:
        top = candidates[0]
        return (
            Decision(
                action=Action(name=ActionName.ACTION6, x=top.x, y=top.y),
                source="fallback_heuristic",
                budget=budget,
                target_object_label=top.object_label,
                raw_response=raw_response,
            ),
            new_memory,
        )

    deterministic = _deterministic_fallback(legal_actions)
    if deterministic is not None:
        return (
            Decision(
                action=deterministic,
                source="fallback_deterministic",
                budget=budget,
                raw_response=raw_response,
            ),
            new_memory,
        )

    try:
        random_action = _random_fallback(legal_actions)
        return (
            Decision(
                action=random_action,
                source="fallback_random",
                budget=budget,
                raw_response=raw_response,
            ),
            new_memory,
        )
    except IndexError:
        return (
            Decision(
                action=Action(name=ActionName.RESET),
                source="fallback_random",
                budget=budget,
                raw_response=raw_response,
            ),
            new_memory,
        )
```

Note: the `config.candidate_count > 1` branch (off by default,
`candidate_count: int = 1`) is intentionally left producing
`raw_response=None` this round — it calls a separate, not-yet-audited
`zerx/candidates.py` module and is out of this plan's scope. This is a
documented, deliberate narrowing, not an oversight.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_policy_decide.py -v -k raw_response`
Expected: 3 passed.

- [ ] **Step 6: Run the fast full suite**

Run: `.venv/Scripts/pytest.exe tests/ -q -m "not slow_local_engine"`
Expected: previous count + 3, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add zerx/policy.py tests/test_policy_decide.py
git commit -m "feat(policy): capture raw model response on Decision, even on failed parse"
```

---

### Task 2: `zerx/trace.py` — trace data model and recorders

**Files:**
- Create: `zerx/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `zerx.policy.Decision` (with `raw_response`, from Task 1),
  `zerx.types.GameFrame`.
- Produces: `TraceMeta`, `TraceStep`, `TraceRecorder` (protocol),
  `JsonlTraceWriter`, `CompositeTraceRecorder`, `describe_reasoning(decision) -> str`,
  `build_trace_step(*, step_index, game_id, frame, decision,
  levels_completed, game_state) -> TraceStep` — all consumed by Task 3
  (`agent/my_agent.py`) and Task 5 (`scripts/visualize_play.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trace.py`:

```python
from __future__ import annotations

import json

from zerx.policy import Decision
from zerx.trace import (
    CompositeTraceRecorder,
    JsonlTraceWriter,
    TraceMeta,
    TraceStep,
    build_trace_step,
    describe_reasoning,
)
from zerx.types import Action, ActionName, GameFrame


def _frame():
    return GameFrame(
        grid=((0, 0), (0, 5)),
        legal_actions=frozenset({ActionName.RESET, ActionName.ACTION6}),
        is_game_over=False,
    )


def test_describe_reasoning_returns_raw_response_when_present():
    decision = Decision(
        action=Action(name=ActionName.ACTION1),
        source="model",
        raw_response='{"action": "ACTION1"}',
    )
    assert describe_reasoning(decision) == '{"action": "ACTION1"}'


def test_describe_reasoning_synthesizes_text_for_known_fallback_sources():
    decision = Decision(action=Action(name=ActionName.ACTION1), source="fallback_deterministic")
    assert describe_reasoning(decision) == (
        "no model or heuristic action available; used the static fallback preference order"
    )


def test_describe_reasoning_falls_back_to_source_name_for_unknown_source():
    decision = Decision(action=Action(name=ActionName.ACTION1), source="some_new_source")
    assert describe_reasoning(decision) == "some_new_source"


def test_build_trace_step_captures_frame_and_decision():
    decision = Decision(
        action=Action(name=ActionName.ACTION6, x=3, y=4),
        source="heuristic",
        target_object_label="obj-0",
    )
    step = build_trace_step(
        step_index=2,
        game_id="ls20",
        frame=_frame(),
        decision=decision,
        levels_completed=1,
        game_state="NOT_FINISHED",
    )
    assert step.step_index == 2
    assert step.game_id == "ls20"
    assert step.grid == ((0, 0), (0, 5))
    assert step.action_name == "ACTION6"
    assert step.action_x == 3
    assert step.action_y == 4
    assert step.source == "heuristic"
    assert step.target_object_label == "obj-0"
    assert step.levels_completed == 1
    assert step.game_state == "NOT_FINISHED"


def test_jsonl_trace_writer_appends_meta_then_steps(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(str(path))
    writer.write_meta(TraceMeta(game_id="ls20", seed=0, backend="fake", config_hash="abc123", started_at="2026-08-06T00:00:00"))
    decision = Decision(action=Action(name=ActionName.ACTION1), source="fallback_deterministic")
    step = build_trace_step(
        step_index=0, game_id="ls20", frame=_frame(), decision=decision,
        levels_completed=0, game_state="NOT_FINISHED",
    )
    writer.record(step)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    meta_line = json.loads(lines[0])
    assert meta_line["type"] == "meta"
    assert meta_line["game_id"] == "ls20"
    step_line = json.loads(lines[1])
    assert step_line["type"] == "step"
    assert step_line["action_name"] == "ACTION1"


def test_composite_trace_recorder_fans_out_to_every_child():
    class _Spy:
        def __init__(self):
            self.steps = []

        def record(self, step):
            self.steps.append(step)

    spy_a, spy_b = _Spy(), _Spy()
    composite = CompositeTraceRecorder([spy_a, spy_b])
    decision = Decision(action=Action(name=ActionName.ACTION1), source="fallback_deterministic")
    step = build_trace_step(
        step_index=0, game_id="ls20", frame=_frame(), decision=decision,
        levels_completed=0, game_state="NOT_FINISHED",
    )
    composite.record(step)
    assert len(spy_a.steps) == 1
    assert len(spy_b.steps) == 1
    assert spy_a.steps[0] is step
    assert spy_b.steps[0] is step
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_trace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zerx.trace'`.

- [ ] **Step 3: Implement `zerx/trace.py`**

```python
"""Per-step trace capture: a pure, pygame-free data model and recorders,
consumed either live (scripts/visualize_play.py's --live mode) or from a
saved JSONL file (--replay mode) -- see
docs/superpowers/specs/2026-08-06-baseline-120-followups-design.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple, Union

from zerx.policy import Decision
from zerx.types import GameFrame

_SOURCE_DESCRIPTIONS = {
    "reset": "terminal frame detected; RESET is the only legal action",
    "heuristic": "high-confidence heuristic candidate used; no model call needed",
    "fallback_heuristic": "model call failed or produced no valid action; used the top-ranked click candidate",
    "fallback_deterministic": "no model or heuristic action available; used the static fallback preference order",
    "fallback_random": "no legal action matched any fallback rule; chose randomly among legal actions",
    "fallback_exact_state_suppressed": "the model/heuristic action was a known no-op for this exact state; substituted a legal alternative",
}


def describe_reasoning(decision: Decision) -> str:
    """Human-readable reasoning text for the visualizer's panel: the raw
    model response when one exists, else a synthesized description of why
    the fallback/heuristic path fired.
    """
    if decision.raw_response:
        return decision.raw_response
    return _SOURCE_DESCRIPTIONS.get(decision.source, decision.source)


@dataclass(frozen=True)
class TraceMeta:
    game_id: str
    seed: int
    backend: str
    config_hash: str
    started_at: str  # ISO 8601


@dataclass(frozen=True)
class TraceStep:
    step_index: int
    game_id: str
    grid: Tuple[Tuple[int, ...], ...]
    action_name: str
    action_x: Optional[int]
    action_y: Optional[int]
    source: str
    repaired: bool
    target_object_label: Optional[str]
    reasoning: str
    levels_completed: int
    game_state: str


def build_trace_step(
    *,
    step_index: int,
    game_id: str,
    frame: GameFrame,
    decision: Decision,
    levels_completed: int,
    game_state: str,
) -> TraceStep:
    return TraceStep(
        step_index=step_index,
        game_id=game_id,
        grid=frame.grid,
        action_name=decision.action.name.value,
        action_x=decision.action.x,
        action_y=decision.action.y,
        source=decision.source,
        repaired=decision.repaired,
        target_object_label=decision.target_object_label,
        reasoning=describe_reasoning(decision),
        levels_completed=levels_completed,
        game_state=game_state,
    )


class TraceRecorder(Protocol):
    def record(self, step: TraceStep) -> None:
        ...


class JsonlTraceWriter:
    """Appends one JSON line per record to `path`. `write_meta` must be
    called at most once, before any `record` call, to write the file's
    header line -- callers that don't need a header (e.g. tests exercising
    `record` alone) may skip it.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write_meta(self, meta: TraceMeta) -> None:
        self._append({"type": "meta", **asdict(meta)})

    def record(self, step: TraceStep) -> None:
        self._append({"type": "step", **asdict(step)})

    def _append(self, payload: dict) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")


class CompositeTraceRecorder:
    """Fans one `record` call out to every child recorder, in order."""

    def __init__(self, recorders: Sequence[TraceRecorder]) -> None:
        self._recorders: List[TraceRecorder] = list(recorders)

    def record(self, step: TraceStep) -> None:
        for recorder in self._recorders:
            recorder.record(step)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_trace.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the fast full suite**

Run: `.venv/Scripts/pytest.exe tests/ -q -m "not slow_local_engine"`
Expected: previous count + 6, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add zerx/trace.py tests/test_trace.py
git commit -m "feat(trace): add pure trace-capture data model and JSONL/composite recorders"
```

---

### Task 3: `Config.trace_export_path`

**Files:**
- Modify: `zerx/config.py`
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Config.trace_export_path: Optional[str] = None`, env var
  `ZERX_TRACE_EXPORT_PATH` — consumed by Task 4 (`agent/my_agent.py`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py` (match the existing file's style — open
it first to confirm the exact `Config`/`from_env` import already in use,
then append in the same pattern):

```python
def test_trace_export_path_defaults_to_none():
    assert Config().trace_export_path is None


def test_trace_export_path_read_from_env():
    config = Config.from_env({"ZERX_TRACE_EXPORT_PATH": "traces/foo.jsonl"})
    assert config.trace_export_path == "traces/foo.jsonl"


def test_trace_export_path_absent_from_env_stays_none():
    config = Config.from_env({})
    assert config.trace_export_path is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_config.py -v -k trace_export_path`
Expected: FAIL — `TypeError: Config.__init__() got an unexpected keyword
argument 'trace_export_path'` is not yet the case; instead
`AttributeError: 'Config' object has no attribute 'trace_export_path'`.

- [ ] **Step 3: Add the field and env wiring**

In `zerx/config.py`, add a new helper next to the other `_env_*` helpers:

```python
def _env_optional_str(env: Mapping[str, str], key: str, default: Optional[str]) -> Optional[str]:
    return env.get(key, default)
```

Add the field at the end of `Config`'s field list (append-only, matches
this project's established convention):

```python
    gemma_base_url: str = "http://localhost:8000/v1/chat/completions"
    trace_export_path: Optional[str] = None  # dev-only: when set, MyAgent
    # writes one JSONL trace file per game via zerx/trace.py -- off by
    # default, never read outside agent/my_agent.py's construction.
```

Add the matching line at the end of `from_env`'s `cls(...)` call, right
after `gemma_base_url=...`:

```python
            trace_export_path=_env_optional_str(
                env, "ZERX_TRACE_EXPORT_PATH", cls.trace_export_path
            ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_config.py -v -k trace_export_path`
Expected: 3 passed.

- [ ] **Step 5: Run the fast full suite**

Run: `.venv/Scripts/pytest.exe tests/ -q -m "not slow_local_engine"`
Expected: previous count + 3, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add zerx/config.py tests/test_config.py
git commit -m "feat(config): add trace_export_path field for optional JSONL trace export"
```

---

### Task 4: Wire `trace_recorder` into `agent/my_agent.py`

**Files:**
- Modify: `agent/my_agent.py`
- Test: `tests/test_my_agent.py` (append)

**Interfaces:**
- Consumes: `zerx.trace.TraceRecorder`, `JsonlTraceWriter`, `build_trace_step`
  (Task 2); `Config.trace_export_path` (Task 3).
- Produces: `MyAgent.trace_recorder: Optional[TraceRecorder]` (public
  attribute) — consumed by Task 5 (`scripts/visualize_play.py`, which
  reassigns it before calling `agent.main()`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_my_agent.py`:

```python
def test_trace_recorder_is_none_by_default():
    agent = _make_agent()
    assert agent.trace_recorder is None


def test_trace_recorder_records_once_per_choose_action_call_when_attached():
    agent = _make_agent()

    class _Spy:
        def __init__(self):
            self.steps = []

        def record(self, step):
            self.steps.append(step)

    spy = _Spy()
    agent.trace_recorder = spy

    frame = FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    agent.choose_action([frame], frame)
    assert len(spy.steps) == 1
    assert spy.steps[0].step_index == 0
    assert spy.steps[0].game_id == agent.game_id

    agent.choose_action([frame, frame], frame)
    assert len(spy.steps) == 2
    assert spy.steps[1].step_index == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_my_agent.py -v -k trace_recorder`
Expected: FAIL — `AttributeError: 'MyAgent' object has no attribute
'trace_recorder'`.

- [ ] **Step 3: Wire it in `agent/my_agent.py`**

Add the import (alongside the existing `zerx.policy` import line):

```python
from zerx.trace import TraceRecorder, JsonlTraceWriter, build_trace_step
```

In `MyAgent.__init__`, after `self._structured_memory = StructuredMemoryState()`:

```python
        self.trace_recorder: Optional[TraceRecorder] = (
            JsonlTraceWriter(self._config.trace_export_path)
            if self._config.trace_export_path
            else None
        )
```

In `_choose_action_inner`, right after the `# --- end baseline-115-exact-state-memory ---`
comment that closes the exact-state-suppression block (i.e., after
`decision` has its final, possibly-replaced value, and before
`self._actions_taken += 1`), add:

```python
        if self.trace_recorder is not None:
            self.trace_recorder.record(
                build_trace_step(
                    step_index=self._actions_taken,
                    game_id=self.game_id,
                    frame=frame,
                    decision=decision,
                    levels_completed=latest_frame.levels_completed,
                    game_state=latest_frame.state.name,
                )
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_my_agent.py -v -k trace_recorder`
Expected: 2 passed.

- [ ] **Step 5: Run the fast full suite**

Run: `.venv/Scripts/pytest.exe tests/ -q -m "not slow_local_engine"`
Expected: previous count + 2, 0 failed. Confirm no existing
`test_my_agent.py` test regresses (they don't touch `trace_recorder`, so
none should).

- [ ] **Step 6: Commit**

```bash
git add agent/my_agent.py tests/test_my_agent.py
git commit -m "feat(agent): wire optional trace_recorder into choose_action, off by default"
```

---

### Task 5: `scripts/visualize_play.py` — live + replay visualizer

**Files:**
- Create: `scripts/visualize_play.py`
- Test: `tests/test_visualize_play.py`
- Modify: `requirements-zerx.txt`, `.gitignore`

**Interfaces:**
- Consumes: `zerx.trace.TraceStep`, `TraceMeta`, `JsonlTraceWriter`,
  `CompositeTraceRecorder` (Task 2); `agent.my_agent.MyAgent`,
  `MyAgent.trace_recorder` (Task 4); same `Arcade`/`MyAgentCls`
  construction pattern as `scripts/play_local.py`.
- Produces: `_load_trace(path) -> tuple[TraceMeta, list[TraceStep]]`,
  `_color_for(cell_value: int) -> tuple[int, int, int]`,
  `_clamp_index(index: int, length: int) -> int` — pure helpers, tested
  directly. `LivePygameRecorder`, `main()` — exercised manually.

- [ ] **Step 1: Add `pygame` to dependencies**

Edit `requirements-zerx.txt`:

```text
numpy
pytest
pygame
```

Install it: `.venv/Scripts/pip.exe install pygame`

- [ ] **Step 2: Add `traces/` to `.gitignore`**

Edit `.gitignore`, in the "Generated artefacts" section:

```text
# Generated artefacts
notebooks/submission.ipynb
notebooks/colab_gemma_smoke.ipynb
reference/
traces/
```

- [ ] **Step 3: Write the failing tests for the pure helpers**

Create `tests/test_visualize_play.py`. `scripts/` has no `__init__.py`
(confirmed: it's not a package), so follow the exact same import pattern
`tests/test_build_colab_notebook.py` already uses for the same reason —
insert `scripts/` onto `sys.path` and import the bare module name, not
`from scripts.visualize_play import ...`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import visualize_play  # noqa: E402
from visualize_play import _clamp_index, _color_for, _load_trace  # noqa: E402
from zerx.trace import TraceMeta, TraceStep


def test_color_for_returns_a_distinct_rgb_tuple_per_known_color_index():
    colors = {_color_for(i) for i in range(10)}
    assert len(colors) == 10
    for color in colors:
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


def test_color_for_falls_back_to_a_default_for_out_of_range_index():
    assert _color_for(99) == _color_for(99)  # deterministic, doesn't raise


def test_clamp_index_stays_within_bounds():
    assert _clamp_index(-1, length=5) == 0
    assert _clamp_index(0, length=5) == 0
    assert _clamp_index(4, length=5) == 4
    assert _clamp_index(5, length=5) == 4
    assert _clamp_index(2, length=5) == 2


def test_clamp_index_handles_empty_buffer():
    assert _clamp_index(3, length=0) == 0


def test_load_trace_parses_meta_and_steps(tmp_path):
    path = tmp_path / "trace.jsonl"
    meta = TraceMeta(game_id="ls20", seed=0, backend="fake", config_hash="abc", started_at="2026-08-06T00:00:00")
    step = TraceStep(
        step_index=0, game_id="ls20", grid=((0, 0), (0, 1)), action_name="ACTION1",
        action_x=None, action_y=None, source="fallback_deterministic", repaired=False,
        target_object_label=None, reasoning="no model call needed",
        levels_completed=0, game_state="NOT_FINISHED",
    )
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "meta", **meta.__dict__}) + "\n")
        fh.write(json.dumps({"type": "step", **step.__dict__}) + "\n")

    loaded_meta, steps = _load_trace(str(path))
    assert loaded_meta.game_id == "ls20"
    assert len(steps) == 1
    assert steps[0].action_name == "ACTION1"
    assert steps[0].grid == ((0, 0), (0, 1))
    assert isinstance(steps[0].grid, tuple) and isinstance(steps[0].grid[0], tuple)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_visualize_play.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.visualize_play'`.

- [ ] **Step 5: Implement `scripts/visualize_play.py`**

```python
"""Live + replay visualizer for zerx runs -- pure dev tooling, never
bundled into the Kaggle submission (scripts/build_notebook.py only
bundles zerx/*.py). See
docs/superpowers/specs/2026-08-06-baseline-120-followups-design.md.

Usage:
    scripts/visualize_play.py --live --game ls20 [--max-steps 80] [--save traces/ls20.jsonl] [--history-cap 500]
    scripts/visualize_play.py --replay traces/ls20-20260806T000000.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
# scripts/ has no __init__.py and this file is normally invoked directly
# (`python scripts/visualize_play.py`), where only scripts/ itself lands
# on sys.path -- ROOT must be added explicitly before any zerx/agent
# import, exactly matching scripts/play_local.py's own established
# pattern (namespace packages resolve fine once the parent dir is present,
# no __init__.py needed).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from zerx.trace import CompositeTraceRecorder, JsonlTraceWriter, TraceMeta, TraceStep  # noqa: E402

_PALETTE = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64),
    4: (255, 220, 0), 5: (170, 170, 170), 6: (240, 18, 190), 7: (255, 133, 27),
    8: (127, 219, 255), 9: (135, 12, 37),
}
_DEFAULT_COLOR = (85, 85, 85)
_CELL_PX = 10


def _color_for(cell_value: int) -> Tuple[int, int, int]:
    return _PALETTE.get(cell_value, _DEFAULT_COLOR)


def _clamp_index(index: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(index, length - 1))


def _load_trace(path: str) -> Tuple[TraceMeta, List[TraceStep]]:
    steps: List[TraceStep] = []
    meta: Optional[TraceMeta] = None
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record_type = payload.pop("type")
            if record_type == "meta":
                meta = TraceMeta(**payload)
            elif record_type == "step":
                # JSON has no tuple type -- grid round-trips as list[list[int]];
                # restore TraceStep's declared tuple[tuple[int, ...], ...] shape.
                payload["grid"] = tuple(tuple(row) for row in payload["grid"])
                steps.append(TraceStep(**payload))
    if meta is None:
        raise ValueError(f"{path}: no meta record found")
    return meta, steps


class LivePygameRecorder:
    """Renders each recorded step to a pygame window as it arrives, keeps
    a capped in-memory history, and blocks inside `record()` while paused
    -- since `record()` runs on the real game loop's own thread (see
    agent/my_agent.py's choose_action -> Agent.main()), this genuinely
    halts play, not just the display.
    """

    def __init__(self, history_cap: int = 500) -> None:
        import pygame  # imported here, not at module scope, so pure
        # helpers above stay importable/testable without a display driver

        self._pygame = pygame
        pygame.init()
        pygame.display.set_caption("zerx visualizer -- live")
        self._screen = pygame.display.set_mode((900, 700))
        self._font = pygame.font.SysFont("consolas", 16)
        self._history: "deque[TraceStep]" = deque(maxlen=history_cap)
        self._cursor = -1  # -1 == following live
        self._paused = False
        self._replay_mode = False  # set True by _run_replay; disables
        # SPACE's pause-toggle since replay has no running loop to pause

    def record(self, step: TraceStep) -> None:
        self._history.append(step)
        self._cursor = -1
        self._render(step)
        while self._pump_events():
            pass

    def _pump_events(self) -> bool:
        """Handle one batch of pygame events; returns True if the caller
        should keep blocking (still paused), False to let the game loop
        continue.
        """
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_SPACE and not self._replay_mode:
                self._paused = not self._paused
            elif self._paused and event.key == pygame.K_LEFT:
                self._cursor = _clamp_index(
                    (self._cursor if self._cursor >= 0 else len(self._history) - 1) - 1,
                    len(self._history),
                )
                self._render(self._history[self._cursor])
            elif self._paused and event.key == pygame.K_RIGHT:
                base = self._cursor if self._cursor >= 0 else len(self._history) - 1
                self._cursor = _clamp_index(base + 1, len(self._history))
                self._render(self._history[self._cursor])
        self._pygame.time.wait(16)  # ~60fps ceiling; avoids a CPU-pinning
        # busy-spin in both this pause loop and _run_replay's event loop
        return self._paused

    def _render(self, step: TraceStep) -> None:
        screen, font = self._screen, self._font
        screen.fill((20, 20, 20))
        for y, row in enumerate(step.grid):
            for x, value in enumerate(row):
                rect = (x * _CELL_PX, y * _CELL_PX, _CELL_PX, _CELL_PX)
                self._pygame.draw.rect(screen, _color_for(value), rect)
        panel_x = len(step.grid[0]) * _CELL_PX + 20 if step.grid else 20
        lines = [
            f"step {step.step_index}  game {step.game_id}",
            f"action {step.action_name} ({step.action_x}, {step.action_y})",
            f"source {step.source}  repaired {step.repaired}",
            f"state {step.game_state}  levels {step.levels_completed}",
            "",
            "reasoning:",
        ] + [step.reasoning[i : i + 48] for i in range(0, len(step.reasoning), 48)]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (230, 230, 230)), (panel_x, 10 + i * 20))
        self._pygame.display.flip()


def _run_live(args: argparse.Namespace) -> None:
    import arc_agi
    from arc_agi import OperationMode

    from agent.my_agent import MyAgent as MyAgentCls

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    if env is None:
        raise SystemExit(f"arcade.make({args.game!r}) returned None -- game unavailable")

    agent = MyAgentCls(
        card_id="visualize-play",
        game_id=args.game,
        agent_name=f"visualize-play.{args.game}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
    )
    if args.max_steps:
        agent.MAX_ACTIONS = min(agent.MAX_ACTIONS, args.max_steps)

    live_recorder = LivePygameRecorder(history_cap=args.history_cap)
    if args.save:
        writer = JsonlTraceWriter(args.save)
        writer.write_meta(
            TraceMeta(
                game_id=args.game,
                seed=0,
                backend=agent._config.backend,
                config_hash=agent._config.config_hash(),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        agent.trace_recorder = CompositeTraceRecorder([live_recorder, writer])
    else:
        agent.trace_recorder = live_recorder

    agent.main()


def _run_replay(args: argparse.Namespace) -> None:
    meta, steps = _load_trace(args.replay)
    recorder = LivePygameRecorder(history_cap=max(len(steps), 1))
    recorder._paused = True  # replay is always "paused": step-only, no running loop
    recorder._replay_mode = True  # SPACE is a no-op; there's no live loop to pause
    for step in steps:
        recorder._history.append(step)
    if steps:
        recorder._cursor = 0
        recorder._render(steps[0])
        while True:
            recorder._pump_events()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--game")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save")
    parser.add_argument("--history-cap", type=int, default=500)
    parser.add_argument("--replay")
    args = parser.parse_args()

    if args.live:
        if not args.game:
            parser.error("--live requires --game")
        _run_live(args)
    elif args.replay:
        _run_replay(args)
    else:
        parser.error("pass --live --game <id> or --replay <trace.jsonl>")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_visualize_play.py -v`
Expected: 5 passed. (These exercise only the pure helpers —
`_color_for`, `_clamp_index`, `_load_trace` — none of which import
`pygame` at module scope, so they run without a display driver.)

- [ ] **Step 7: Run the fast full suite**

Run: `.venv/Scripts/pytest.exe tests/ -q -m "not slow_local_engine"`
Expected: previous count + 5, 0 failed.

- [ ] **Step 8: Manual verification (not automated — see spec's testing strategy)**

Run live mode against a real local game and confirm a window opens,
renders the grid and reasoning panel, SPACE pauses/resumes, and ←/→
navigate history while paused:

```bash
.venv/Scripts/python.exe scripts/visualize_play.py --live --game ls20 --max-steps 20 --save traces/ls20-manual-check.jsonl
```

Then confirm replay mode loads that exact file and renders the same way:

```bash
.venv/Scripts/python.exe scripts/visualize_play.py --replay traces/ls20-manual-check.jsonl
```

Record what you observed (window opened: yes/no; grid rendered: yes/no;
pause/resume worked: yes/no; history navigation worked: yes/no) in this
plan's final task (Task 7) before claiming the feature complete — per
`AGENTS.md`'s UI-testing guidance, test suites verify code correctness,
not feature correctness.

- [ ] **Step 9: Commit**

```bash
git add scripts/visualize_play.py tests/test_visualize_play.py requirements-zerx.txt .gitignore
git commit -m "feat(visualize): add live + replay pygame visualizer sharing one render path"
```

---

### Task 6: `README.md`

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by other tasks — this is the last
  code-adjacent task, written once everything above exists so its usage
  examples are accurate.

- [ ] **Step 1: Write `README.md`**

```markdown
# zerx-harness

An ARC-AGI-3 Kaggle Prize 2026 agent built on Gemma-4-31B as a local
vision-language policy. The full mission, scope, and operating contract
live in [`AGENTS.md`](AGENTS.md) and [`STRATEGY.md`](STRATEGY.md) — read
those first if you're changing agent behavior. This file is usage docs
only.

## Setup

```bash
pip install -r requirements-zerx.txt
```

Python 3.11+ (this repo's own `.venv` targets 3.11/3.13; either works).

## Running a game locally

```bash
.venv/Scripts/python.exe scripts/play_local.py --game ls20 --max-steps 50
```

`--list` prints every public game id. Behavior is controlled entirely by
`ZERX_*` environment variables, resolved once per run by
`zerx/config.py`'s `Config.from_env()` — that module is the source of
truth for every flag and its default; don't rely on this README to stay
in sync with it as flags are added.

## Attributing runs to your own account

`arc_agi`'s client already reads the `ARC_API_KEY` environment variable
on its own (see `.venv/Lib/site-packages/arc_agi/base.py`) — no code in
this repo needs to set or forward it. If `ARC_API_KEY` is unset, every
local/Colab run is attributed to an anonymous key on
`three.arcprize.org`'s dashboard. To attribute your own runs, set it in
your own shell before running anything that touches the engine:

```bash
export ARC_API_KEY=your-key-here      # bash
$env:ARC_API_KEY = "your-key-here"    # PowerShell
```

Never commit this value or put it in a notebook cell — same rule as
`CEREBRAS_API_KEY` (see `AGENTS.md`'s Cerebras development boundary).

## Tests

```bash
.venv/Scripts/pytest.exe tests/ -q
```

One test file (`tests/test_real_game_regression.py`) drives the real
local game engine across all 25 public games and is slow (~20s once a
real backend is wired, historically up to ~20 minutes against an
unreachable model server — see `docs/HANDOFF.md`). For fast iteration:

```bash
.venv/Scripts/pytest.exe tests/ -q -m "not slow_local_engine"
```

## Visualizer

Watch a game live, or replay a saved trace:

```bash
# live, saving a trace file for later replay
.venv/Scripts/python.exe scripts/visualize_play.py --live --game ls20 --max-steps 80 --save traces/ls20.jsonl

# replay a saved trace, no live game involved
.venv/Scripts/python.exe scripts/visualize_play.py --replay traces/ls20.jsonl
```

SPACE pauses/resumes live mode (this genuinely halts play, not just the
display); ←/→ step through history while paused. `traces/` is gitignored
— generated output, not source of truth.

Setting `ZERX_TRACE_EXPORT_PATH=traces/some-file.jsonl` makes any run
(not just the visualizer) write a trace file, including headless Colab
runs — download the file afterward and replay it locally.

## Project layout, contract, and status

- [`AGENTS.md`](AGENTS.md) — the authoritative operating contract.
- [`STRATEGY.md`](STRATEGY.md) — prior-art review and adoption decisions.
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — current status and next action.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project README (setup, running locally, ARC_API_KEY, visualizer)"
```

---

### Task 7: `docs/HANDOFF.md` status update and final verification

**Files:**
- Modify: `docs/HANDOFF.md`

- [ ] **Step 1: Run the full unfiltered suite**

Run: `.venv/Scripts/pytest.exe tests/ -q`
Expected: all prior tests + this plan's ~19 new tests, 0 failed.

- [ ] **Step 2: Append a status entry to `docs/HANDOFF.md`**

Add a new subsection (do not rewrite existing content) recording: branch
`feat/baseline-120-followups`, what was built (trace export, live+replay
visualizer, README, `ARC_API_KEY` doc), the manual-verification result
from Task 5 Step 8, and the final test count. Also add a new numbered
item to "Exact next action" for the documented-but-not-built resume/fork
mechanism (see `docs/superpowers/specs/2026-08-06-baseline-120-followups-design.md`'s
"Future work" section) — a recommendation, not started.

- [ ] **Step 3: Commit**

```bash
git add docs/HANDOFF.md
git commit -m "docs(handoff): record baseline-120-followups status and next steps"
```

- [ ] **Step 4: Push the branch**

```bash
git push -u origin feat/baseline-120-followups
```

Do not merge to `master` without the human owner's explicit go-ahead —
same standing rule this whole project has followed for every branch so
far.

---

## Self-review notes

- **Spec coverage:** all 4 original items (README, visualizer, trace
  export, `ARC_API_KEY`) covered — Tasks 1-2 (trace model), 3-4 (config +
  wiring), 5 (visualizer), 6 (README incl. `ARC_API_KEY`). The "Future
  work" resume/fork section is deliberately documentation-only in Task 7,
  matching the approved spec's design-now/build-later decision.
- **Placeholder scan:** no TBD/TODO; every code block is complete,
  runnable code, not a description of code.
- **Type consistency:** `TraceStep`/`TraceMeta` field names and types are
  identical across Task 2 (definition), Task 4 (`build_trace_step` call
  site), and Task 5 (`_load_trace`, `LivePygameRecorder`) — verified by
  re-reading each usage against Task 2's dataclass definitions.
  `Decision.raw_response` (Task 1) is read only via
  `describe_reasoning()` (Task 2), never duplicated elsewhere.
