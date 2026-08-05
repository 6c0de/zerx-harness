# baseline-120 Backend Selection Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Config.backend` and `Config.platform` actually control which
`ModelBackend` class `MyAgent` constructs, closing the gap where
`agent/my_agent.py` currently hardcodes `GemmaModelBackend` regardless of
configuration (measured effect on the base commit: `Aggregate scorecard
score: 0.0`, `levels_completed=0`, every action `ACTION6`, because every
`generate()` call fails against a nonexistent local vLLM server and
`decide()` silently falls back every step).

**Architecture:** Add one pure factory function, `select_backend(config:
Config) -> ModelBackend`, to `zerx/model_backend.py`. It switches on
`config.backend` and constructs the matching backend class
(`FakeModelBackend`, `GemmaModelBackend`, or `CerebrasDevBackend`),
forwarding `config.platform` to `CerebrasDevBackend` so its existing
`platform=="kaggle"` lockout is actually reachable end-to-end (today
nothing constructs it outside its own tests, so that parameter has never
been exercised). `agent/my_agent.py`'s `MyAgent.__init__` calls this
factory instead of hardcoding `GemmaModelBackend`. A new `Config.gemma_base_url`
field lets a future run point `GemmaModelBackend` at a non-default vLLM
endpoint without editing code.

**Tech Stack:** Python 3, pytest, stdlib only (no new dependencies). No
network calls, no vLLM/torch/transformers import anywhere in this plan's
files.

## Global Constraints

- Base commit: `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb`. Branch:
  `feat/baseline-120-backend-wiring`, already checked out from that exact
  SHA.
- Full local suite is 261 passed / 0 failed on the base commit (confirmed
  this session via `.venv/Scripts/pytest.exe tests/ -q`, Windows venv
  layout). Must stay green after every task, growing only by this plan's
  new tests.
- Files this plan may touch, per `docs/superpowers/plans/parallel-baseline-120/person-1-backend-wiring.md`'s
  "Files you own this round" — **nothing else**:
  - `zerx/model_backend.py` (add `select_backend` only; do not modify
    `ModelBackend`, `FakeModelBackend`, `GemmaModelBackend`, or the
    module's existing docstring's factual claims).
  - `agent/my_agent.py` (exactly one changed line in `MyAgent.__init__`
    plus its import — nothing else in this file).
  - `zerx/config.py` (exactly one new field, appended after
    `structured_memory_on: bool = False`, plus its matching `from_env`
    line — no other field touched).
  - New file `tests/test_backend_selection.py`.
  - `tests/test_config.py` (one additive test, appended — no
    restructuring).
  - `docs/HANDOFF.md` (one-line status update at the very end).
- Do not touch: `eval/run_ablation.py`, `scripts/build_colab_notebook.py`,
  `scripts/build_notebook.py`, `zerx/backends/cerebras_dev.py` (read-only —
  its constructor signature is already exactly what's needed), any other
  `zerx/*` or `tests/*` file.
- `select_backend`'s signature is frozen (Track 2/3/4 in the parallel plan
  code against it): `def select_backend(config: Config) -> ModelBackend`.
  Do not change it after this plan lands.
- No new try/except added to `agent/my_agent.py` for backend-selection
  failures — an invalid `Config.backend` string must fail loudly at
  `MyAgent.__init__` time (construction-time validation), not be
  swallowed.
- Never commit `CEREBRAS_API_KEY` or any secret value; none of this plan's
  tests require one (all `CerebrasDevBackend` tests use the default
  "no key in environment" path or an explicit fake).
- Do not merge to `master`. Push only to `feat/baseline-120-backend-wiring`.

---

## File Structure

- `zerx/model_backend.py` — gains one new function, `select_backend`,
  appended after the existing `GemmaModelBackend` class. Needs two new
  imports at module top: `from zerx.backends.cerebras_dev import
  CerebrasDevBackend` and `from zerx.config import Config` (both stdlib-only
  modules themselves — neither imports vllm/torch/transformers, so the
  module's existing "never imports vllm/torch" test stays true unchanged).
- `zerx/config.py` — one new frozen-dataclass field (`gemma_base_url`) and
  its `from_env` line, in the existing `Config` class.
- `agent/my_agent.py` — one import line changed
  (`GemmaModelBackend` → `select_backend`) and one call-site line changed
  in `MyAgent.__init__`.
- `tests/test_backend_selection.py` — new file, one test per backend
  string plus the unknown-backend and platform-forwarding/lockout cases.
- `tests/test_config.py` — two additive tests for the new field (default
  value, env override), appended at the end of the file.
- `docs/HANDOFF.md` — "Known failures" item #1 updated to say it's fixed,
  referencing this branch/commit.

---

## Task 1: `Config.gemma_base_url` field

**Files:**
- Modify: `zerx/config.py:50` (append field after `structured_memory_on`),
  `zerx/config.py:90` (append `from_env` line after the
  `structured_memory_on=...` argument)
- Test: `tests/test_config.py` (append at end of file)

**Interfaces:**
- Produces: `Config.gemma_base_url: str` (default
  `"http://localhost:8000/v1/chat/completions"` — the exact current
  `GemmaModelBackend._DEFAULT_BASE_URL` value, so every existing test and
  the existing Colab notebook cell keep working unchanged). Consumed by
  Task 2's `select_backend`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_gemma_base_url_defaults_to_local_vllm_endpoint():
    assert Config().gemma_base_url == "http://localhost:8000/v1/chat/completions"


def test_from_env_overrides_gemma_base_url():
    cfg = Config.from_env({"ZERX_GEMMA_BASE_URL": "http://localhost:9000/v1/chat/completions"})
    assert cfg.gemma_base_url == "http://localhost:9000/v1/chat/completions"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_config.py -v -k gemma_base_url`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'gemma_base_url'`

- [ ] **Step 3: Add the field**

In `zerx/config.py`, change:

```python
    structured_memory_on: bool = False
```

to:

```python
    structured_memory_on: bool = False
    gemma_base_url: str = "http://localhost:8000/v1/chat/completions"
```

And in `from_env`'s return call, change:

```python
            structured_memory_on=_env_bool(
                env, "ZERX_STRUCTURED_MEMORY_ON", cls.structured_memory_on
            ),
        )
```

to:

```python
            structured_memory_on=_env_bool(
                env, "ZERX_STRUCTURED_MEMORY_ON", cls.structured_memory_on
            ),
            gemma_base_url=_env_str(env, "ZERX_GEMMA_BASE_URL", cls.gemma_base_url),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_config.py -v`
Expected: all pass, including the two new ones.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/Scripts/pytest.exe tests/ -q`
Expected: 263 passed, 0 failed (261 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add zerx/config.py tests/test_config.py
git commit -m "feat(config): add gemma_base_url field for vLLM endpoint override"
```

---

## Task 2: `select_backend` factory in `zerx/model_backend.py`

**Files:**
- Modify: `zerx/model_backend.py` (add imports at top, add function at
  end of file)
- Test: Create `tests/test_backend_selection.py`

**Interfaces:**
- Consumes: `Config` (from Task 1 — specifically `config.backend`,
  `config.platform`, `config.model_revision`, `config.gemma_base_url`),
  `CerebrasDevBackend(model_id: str, platform: str = "local", ...)` (from
  `zerx/backends/cerebras_dev.py`, unmodified, read-only), the existing
  `FakeModelBackend`, `GemmaModelBackend` classes in this same module.
- Produces: `select_backend(config: Config) -> ModelBackend` — frozen
  signature, consumed later by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backend_selection.py`:

```python
import pytest

from zerx.config import Config
from zerx.model_backend import FakeModelBackend, GemmaModelBackend, select_backend


def test_select_backend_fake_returns_fake_backend_with_no_scripted_responses():
    backend = select_backend(Config(backend="fake"))
    assert isinstance(backend, FakeModelBackend)
    with pytest.raises(RuntimeError):
        backend.generate("prompt")


def test_select_backend_gemma_local_returns_configured_gemma_backend():
    config = Config(
        backend="gemma_local",
        model_revision="gemma-4-31b-it",
        gemma_base_url="http://localhost:9001/v1/chat/completions",
    )
    backend = select_backend(config)
    assert isinstance(backend, GemmaModelBackend)
    assert backend.model_revision == "gemma-4-31b-it"
    assert backend.base_url == "http://localhost:9001/v1/chat/completions"


def test_select_backend_gemma_kaggle_returns_configured_gemma_backend():
    config = Config(
        backend="gemma_kaggle",
        model_revision="gemma-4-31b-it",
        gemma_base_url="http://localhost:9002/v1/chat/completions",
    )
    backend = select_backend(config)
    assert isinstance(backend, GemmaModelBackend)
    assert backend.model_revision == "gemma-4-31b-it"
    assert backend.base_url == "http://localhost:9002/v1/chat/completions"


def test_select_backend_cerebras_dev_returns_cerebras_backend_on_local_platform():
    from zerx.backends.cerebras_dev import CerebrasDevBackend

    config = Config(backend="cerebras_dev", platform="local", model_revision="gemma-4-31b")
    backend = select_backend(config)
    assert isinstance(backend, CerebrasDevBackend)
    assert backend.model_id == "gemma-4-31b"


def test_select_backend_cerebras_dev_forwards_config_platform_argument(monkeypatch):
    """Regression test for the exact bug this track fixes:
    `CerebrasDevBackend` was never reachable end-to-end, so its `platform`
    kwarg was never exercised with a real Config value. Use a platform
    value ("colab") that differs from CerebrasDevBackend's own default
    ("local") so a hardcoded/default-value bug in select_backend cannot
    accidentally pass this test.
    """
    import zerx.model_backend as model_backend_module

    captured = {}

    class _RecordingCerebrasDevBackend:
        def __init__(self, *, model_id, platform):
            captured["model_id"] = model_id
            captured["platform"] = platform

    monkeypatch.setattr(model_backend_module, "CerebrasDevBackend", _RecordingCerebrasDevBackend)

    config = Config(backend="cerebras_dev", platform="colab", model_revision="gemma-4-31b")
    backend = select_backend(config)

    assert isinstance(backend, _RecordingCerebrasDevBackend)
    assert captured["platform"] == "colab"
    assert captured["model_id"] == "gemma-4-31b"


def test_cerebras_dev_on_kaggle_platform_is_unreachable_via_config_guard():
    """Config.__post_init__ already rejects backend='cerebras_dev' with
    platform='kaggle' at Config-construction time -- select_backend never
    even runs in that case. This proves the two independent layers
    (Config's guard, CerebrasDevBackend's own guard) still compose
    correctly after this track's change, rather than assuming it.
    """
    with pytest.raises(ValueError):
        Config(backend="cerebras_dev", platform="kaggle")


def test_select_backend_raises_value_error_for_unknown_backend_string():
    with pytest.raises(ValueError):
        select_backend(Config(backend="not-a-real-backend"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_backend_selection.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_backend' from 'zerx.model_backend'`

- [ ] **Step 3: Implement `select_backend`**

In `zerx/model_backend.py`, add imports after the existing
`from typing import ...` line:

```python
from zerx.backends.cerebras_dev import CerebrasDevBackend
from zerx.config import Config
```

Append at the end of the file, after the `GemmaModelBackend` class:

```python
def select_backend(config: Config) -> ModelBackend:
    """Construct the ModelBackend named by config.backend
    ('fake' | 'cerebras_dev' | 'gemma_local' | 'gemma_kaggle'),
    forwarding config.platform to CerebrasDevBackend so its existing
    platform=='kaggle' lockout applies. Raises ValueError for any other
    backend string. 'fake' returns FakeModelBackend() with an empty
    responses list (deliberate: every call raises, exercising the
    fallback chain) -- not a general-purpose scripted-response
    constructor; callers who need scripted responses still construct
    FakeModelBackend(responses=[...]) directly.
    """
    if config.backend == "fake":
        return FakeModelBackend()
    if config.backend in ("gemma_local", "gemma_kaggle"):
        return GemmaModelBackend(config.model_revision, base_url=config.gemma_base_url)
    if config.backend == "cerebras_dev":
        return CerebrasDevBackend(model_id=config.model_revision, platform=config.platform)
    raise ValueError(f"Unknown backend: {config.backend!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_backend_selection.py -v`
Expected: all 7 pass.

- [ ] **Step 5: Confirm the module still never imports vllm/torch/transformers**

Run: `.venv/Scripts/pytest.exe tests/test_model_backend.py -v -k never_imports`
Expected: PASS (the new imports are `zerx.backends.cerebras_dev` and
`zerx.config`, neither of which imports a model library).

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `.venv/Scripts/pytest.exe tests/ -q`
Expected: 270 passed, 0 failed (263 + 7 new).

- [ ] **Step 7: Commit**

```bash
git add zerx/model_backend.py tests/test_backend_selection.py
git commit -m "feat(model_backend): add select_backend factory driven by Config.backend"
```

---

## Task 3: Wire `select_backend` into `agent/my_agent.py`

**Files:**
- Modify: `agent/my_agent.py:32` (import line), `agent/my_agent.py:156`
  (construction line)

**Interfaces:**
- Consumes: `select_backend(config: Config) -> ModelBackend` (Task 2).

- [ ] **Step 1: Confirm no other reference to `GemmaModelBackend` exists in this file**

Run: `grep -n "GemmaModelBackend" agent/my_agent.py`
Expected output (before this task's edit): two lines — the import at line
32 and the construction at line 156. If a third reference exists, stop and
re-read the file before editing (this plan assumes exactly two).

- [ ] **Step 2: Update the import**

In `agent/my_agent.py`, change:

```python
from zerx.model_backend import GemmaModelBackend
```

to:

```python
from zerx.model_backend import select_backend
```

- [ ] **Step 3: Update the construction line**

In `MyAgent.__init__`, change:

```python
        self._backend = GemmaModelBackend(self._config.model_revision)
```

to:

```python
        self._backend = select_backend(self._config)
```

- [ ] **Step 4: Verify the old hardcoded pattern is gone**

Run: `grep -n "GemmaModelBackend(self._config.model_revision)" agent/my_agent.py`
Expected: no output (empty match).

- [ ] **Step 5: Run the existing `agent/my_agent.py` test suite**

Run: `.venv/Scripts/pytest.exe tests/test_my_agent.py tests/test_my_agent_exact_state.py -v`
Expected: all pass unchanged. (These tests never reach a real network call
either before or after this change — the default `Config().backend` is
`"fake"`, so `self._backend` becomes a `FakeModelBackend` with no scripted
responses; `decide()`'s existing `except Exception` around the model call
catches its `RuntimeError` exactly as it previously caught the connection
failure from the hardcoded `GemmaModelBackend` pointed at a nonexistent
`localhost:8000` server. Same fallback path, faster and network-free.)

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `.venv/Scripts/pytest.exe tests/ -q`
Expected: 270 passed, 0 failed (same count as Task 2 — this task adds no
new tests, only fixes existing wiring).

- [ ] **Step 7: Commit**

```bash
git add agent/my_agent.py
git commit -m "fix(agent): route backend construction through select_backend

Config.backend was a real, tested field that nothing ever read to choose
which ModelBackend class to build -- MyAgent.__init__ unconditionally
constructed GemmaModelBackend regardless of config, so every
backend.generate() call failed against a nonexistent local vLLM server
and decide() silently fell back every step (measured: 0.0 aggregate
score, 0 levels completed, all-ACTION6 on ls20+vc33). This also makes
CerebrasDevBackend reachable end-to-end for the first time, with
config.platform correctly forwarded so its platform==kaggle lockout is
exercised rather than dead code."
```

---

## Task 4: Docs update and final verification

**Files:**
- Modify: `docs/HANDOFF.md` ("Known failures or risks" item #1)

- [ ] **Step 1: Update `docs/HANDOFF.md`**

Find this existing bullet under "Known failures or risks (carried over,
still real)":

```markdown
1. `zerx/backends/cerebras_dev.py`'s `platform` kwarg defaults to `"local"`
   and is never wired to the real `Config.platform` — inert today (nothing
   constructs `CerebrasDevBackend` outside its own tests). **Whichever
   track adds a backend-selection factory must forward
   `platform=config.platform` explicitly.**
```

Replace with:

```markdown
1. ~~`zerx/backends/cerebras_dev.py`'s `platform` kwarg defaults to
   `"local"` and is never wired to the real `Config.platform`~~ **Fixed**
   on `feat/baseline-120-backend-wiring` — `zerx/model_backend.py`'s new
   `select_backend(config)` factory constructs the backend named by
   `config.backend` and forwards `config.platform` to `CerebrasDevBackend`
   explicitly; `agent/my_agent.py`'s `MyAgent.__init__` now calls it
   instead of hardcoding `GemmaModelBackend`. See
   `docs/superpowers/plans/2026-08-05-baseline-120-backend-wiring.md`.
```

- [ ] **Step 2: Add a one-line status entry under "Parallel work split"**

In the Day 3 table's spirit but for this new stage — add a short note near
the existing `baseline-120` "Exact next action" section recording Track 1
as done, branch name, and final test count (fill in the actual commit SHA
after Step 4's commit):

```markdown
**Track 1 (backend selection wiring) — done.** Branch
`feat/baseline-120-backend-wiring`, commit `<fill in after final commit>`.
`select_backend(config: Config) -> ModelBackend` added to
`zerx/model_backend.py`, matching the frozen interface in
`docs/superpowers/plans/parallel-baseline-120/README.md` exactly. Full
suite: 270 passed, 0 failed (261 base + 9 new: 2 config, 7 backend
selection).
```

- [ ] **Step 3: Run the full suite one final time**

Run: `.venv/Scripts/pytest.exe tests/ -q`
Expected: 270 passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
git add docs/HANDOFF.md
git commit -m "docs(handoff): record baseline-120 backend-wiring track as done"
```

- [ ] **Step 5: Verify the branch's diff matches this plan's declared scope exactly**

Run: `git diff 8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb --stat`
Expected: only these files changed: `zerx/config.py`, `zerx/model_backend.py`,
`agent/my_agent.py`, `tests/test_config.py`, `tests/test_backend_selection.py`,
`docs/HANDOFF.md`, plus this plan file itself under
`docs/superpowers/plans/`. No other file touched.

- [ ] **Step 6: Push to the remote branch**

```bash
git push -u origin feat/baseline-120-backend-wiring
```

(Do not merge to `master` — per this track's rules, one person merges all
4 tracks in sequence per `docs/superpowers/plans/parallel-baseline-120/INTEGRATION.md`.)

---

## Self-review notes

- **Spec coverage:** every item in `person-1-backend-wiring.md`'s "Files
  you own this round," "Interface you're producing," "New `Config`
  field," "Wiring into `agent/my_agent.py`," "Tests," and "Definition of
  done" sections maps to a task above. The frozen `select_backend`
  signature (Task 2) is reproduced verbatim in its docstring.
- **Platform-forwarding test design deviation, documented:** the spec
  text says to "assert the `platform` attribute directly" on the
  `CerebrasDevBackend` returned for `platform="local"`. Reading the real
  `zerx/backends/cerebras_dev.py` (required before editing, and this
  track may not modify that file) shows `__init__` never stores
  `self.platform` — only `model_id`, `api_version`,
  `request_timeout_seconds`, `max_retries`, `_api_key`, `_http_post`,
  `last_latency_seconds` are stored. There is no `.platform` attribute to
  assert. Task 2's forwarding test instead monkeypatches
  `zerx.model_backend.CerebrasDevBackend` with a recording stub and uses
  `platform="colab"` (differs from `CerebrasDevBackend`'s own
  `platform="local"` default) so the test cannot pass by coincidence if
  `select_backend` ever hardcoded a default instead of forwarding
  `config.platform`. This is a stronger regression proof than the literal
  spec text and does not touch the read-only file.
  A second, separate test (`test_cerebras_dev_on_kaggle_platform_is_unreachable_via_config_guard`)
  covers the spec's explicit "prove `Config.__post_init__`'s guard still
  fires" requirement.
- **Placeholder scan:** no TBD/TODO markers; every step has literal code.
- **Type/signature consistency:** `select_backend(config: Config) ->
  ModelBackend` is identical across the plan header, Task 2's docstring,
  and Task 3's usage. `Config.gemma_base_url: str` is identical across
  Task 1 and its use in Task 2.
