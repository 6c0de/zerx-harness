# Developer 1 — Backend selection wiring

**Read `README.md` in this directory first** — shared context, the
empirical "before" measurement this whole plan is built on, the frozen
interface contracts, and the ownership matrix this file assumes.

- **Track:** Backend selection wiring (foundation for `baseline-120-reki-core`)
- **Base master SHA:** `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb`
- **Branch:** `feat/baseline-120-backend-wiring` (create from base SHA above; does not exist yet — you create it)

## Purpose and expected outcome

`Config.backend` (`"fake"|"cerebras_dev"|"gemma_local"|"gemma_kaggle"`) is
a real, tested field in `zerx/config.py` — but nothing in the codebase
ever reads it to decide which `ModelBackend` class to construct.
`agent/my_agent.py:156` hardcodes:

```python
self._backend = GemmaModelBackend(self._config.model_revision)
```

This session verified the practical impact directly: running
`scripts/play_local.py --game ls20,vc33 --max-steps 50` on the base
commit produces `Aggregate scorecard score: 0.0`, `levels_completed=0` on
both games, and every single action is `ACTION6` — because every
`backend.generate()` call fails (no vLLM server running) and `decide()`
silently falls back every step. Separately, `zerx/backends/cerebras_dev.py`'s
`CerebrasDevBackend` is fully built and unit-tested in isolation, but is
**completely unreachable** from `agent/my_agent.py` — the entire
Cerebras dev-lane `AGENTS.md`/`docs/TEAM_WORKFLOW.md` describe is
currently dead code end-to-end, not because Cerebras integration is
unfinished, but because nothing ever selects it.

Your job: make `Config.backend` and `Config.platform` actually control
which backend class `MyAgent` constructs, forwarding `platform` to
`CerebrasDevBackend` exactly as `docs/HANDOFF.md`'s "Known failures" #1
already flagged as the required fix. This unblocks Track 2 (real
integration testing) and Track 4 (the actual Colab run) — see this
directory's `README.md` "Track dependency graph."

## Commands to run before starting

```bash
git fetch origin
git checkout -b feat/baseline-120-backend-wiring 8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb
.venv/bin/pytest tests/ -q   # confirm 261 passed, 0 failed before you touch anything
```

## Files you own this round

- `zerx/model_backend.py` — add the new factory function (below). Do not
  modify `ModelBackend`, `FakeModelBackend`, `GemmaModelBackend`, or the
  module's existing docstring's factual claims (its "never imports
  vllm/torch" boundary stays true — your new import is `zerx.backends.cerebras_dev`,
  not a model library).
- `agent/my_agent.py` — exactly one line in `MyAgent.__init__`, nothing
  else in this file. No comment-banner block needed (Day 3's etiquette
  banners were for config-gated *additive* behavior inside
  `_choose_action_inner`; this is a straight fix to existing, already-live
  code, not a new off-by-default feature).
- `zerx/config.py` — exactly one new field, appended after the existing
  `structured_memory_on: bool = False` line (the current last field), plus
  its matching `from_env(...)` line. Do not touch any existing field.
- New file: `tests/test_backend_selection.py`.
- `tests/test_config.py` — one additive test for the new field (append,
  don't restructure the file).

## Do not touch

`eval/run_ablation.py`, `scripts/build_colab_notebook.py`,
`scripts/build_notebook.py`, `zerx/backends/cerebras_dev.py` (read it,
don't edit it — its constructor signature is already exactly what you
need), any `zerx/*` file not listed above, any `tests/*` file not listed
above.

## Interface you're producing (frozen — Track 2/3/4 code against this from day one)

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
```

Design notes, not mandates — use your judgment and document real
decisions in your own plan file:

- `"gemma_local"` and `"gemma_kaggle"` both currently map to
  `GemmaModelBackend(config.model_revision, base_url=config.gemma_base_url)`
  (see the new config field below) — there is no evidence in this
  repository today that Kaggle needs a different construction path (the
  Kaggle submission notebook builder, `scripts/build_notebook.py`, is out
  of scope for this track). If you find evidence it does need to differ,
  stop and note it in your plan file rather than guessing at Kaggle-gate
  behavior that belongs to a later, separate stage.
- `"cerebras_dev"` maps to
  `CerebrasDevBackend(model_id=config.model_revision, platform=config.platform)`
  — `CerebrasDevBackend.__init__` already raises `ValueError` when
  `platform == "kaggle"`; your job is only to make sure `config.platform`
  actually reaches it (currently nothing constructs `CerebrasDevBackend`
  outside its own tests, so this parameter has never been exercised
  end-to-end). Note `zerx/config.py`'s existing `__post_init__` already
  rejects `backend == "cerebras_dev" and platform == "kaggle"` at the
  `Config` level too — this is intentional defense-in-depth already built
  by an earlier session; your factory adds a second, independent layer at
  construction time, it does not replace the first.
- `"fake"` maps to `FakeModelBackend()` (no scripted responses). Reread
  `zerx/model_backend.py`'s `FakeModelBackend.generate()` — with an empty
  `responses` list it raises `RuntimeError` on the first call, which
  `zerx/policy.py`'s `decide()` already catches. This is what Track 3's
  local regression sweep will use to exercise the harness without a
  server or credentials.
- Any other string: raise `ValueError` with a message naming the invalid
  value — this is a real input-validation boundary, not a case to leave
  unhandled.

## New `Config` field

Append to `zerx/config.py`'s field list, after `structured_memory_on`:

```python
gemma_base_url: str = "http://localhost:8000/v1/chat/completions"
```

And the matching line in `from_env`'s return call, at the end of its
argument list:

```python
gemma_base_url=_env_str(env, "ZERX_GEMMA_BASE_URL", cls.gemma_base_url),
```

This preserves the exact current default (`GemmaModelBackend`'s own
`_DEFAULT_BASE_URL`) so every existing test and the existing Colab
notebook cell keep working unchanged. It exists so a future run can point
at a non-default vLLM endpoint without editing code — out of scope to
actually use this for anything beyond passing it through in this track;
Track 4 may use it if their Colab setup needs it.

## Wiring into `agent/my_agent.py`

Replace this one line in `MyAgent.__init__`:

```python
self._backend = GemmaModelBackend(self._config.model_revision)
```

with:

```python
self._backend = select_backend(self._config)
```

and add the import (`from zerx.model_backend import select_backend`,
alongside the existing `GemmaModelBackend` import — you may need to keep
or drop the `GemmaModelBackend` import depending on whether anything else
in the file still references it directly; check before removing).

This is the only change to `agent/my_agent.py` in this track. Do not
touch `_choose_action_inner`, the exact-state-memory block, or the
structured-memory block — those belong to Day 3's already-merged tracks
and are out of scope here.

## Tests

`tests/test_backend_selection.py` (new file) must cover:

- `select_backend(Config(backend="fake"))` returns a `FakeModelBackend`
  whose `generate()` raises on first call (no responses scripted).
- `select_backend(Config(backend="gemma_local"))` returns a
  `GemmaModelBackend` with `model_revision` and `base_url` matching the
  config (including a non-default `gemma_base_url` override).
- `select_backend(Config(backend="gemma_kaggle"))` — same shape check as
  above (document in your plan file whether you chose to make this
  identical to `gemma_local` or added any distinction, and why).
- `select_backend(Config(backend="cerebras_dev", platform="local"))`
  returns a `CerebrasDevBackend` with `platform="local"` (not the
  parameter's own default) — this is the specific regression this track
  exists to prevent; assert the `platform` attribute directly, don't just
  assert construction succeeded.
- `select_backend(Config(backend="cerebras_dev", platform="kaggle"))` —
  this should already be unreachable via `Config.__post_init__`'s
  existing guard (raises at `Config` construction, before your factory
  ever runs) — write a test proving that guard still fires, so the
  interaction between the two layers is explicit and tested, not assumed.
- `select_backend(Config(backend="not-a-real-backend"))` raises
  `ValueError`.

`tests/test_config.py`: one additive test asserting `Config().gemma_base_url`
defaults to the exact current `GemmaModelBackend` default, and that
`ZERX_GEMMA_BASE_URL` overrides it via `Config.from_env`.

No test should require a running server, network access, or
`CEREBRAS_API_KEY` — same discipline as every existing backend test in
this repository.

## Verification commands

```bash
.venv/bin/pytest tests/ -q                       # must show 261 + your new tests, 0 failed
.venv/bin/pytest tests/test_backend_selection.py tests/test_config.py -v
grep -n "GemmaModelBackend(self._config.model_revision)" agent/my_agent.py  # must return nothing
```

## Expected outputs

- `zerx/model_backend.py` gains `select_backend`, nothing else changes.
- `agent/my_agent.py` has exactly one changed line plus its import.
- `zerx/config.py` has exactly one new field plus its `from_env` line.
- New `tests/test_backend_selection.py`, ~6 focused tests.
- Full suite green, count increased by your new tests only.

## Performance / runtime bounds

All new tests are pure unit tests (no network, no subprocess) — expect
each to run in well under 100ms, consistent with the existing suite's
~1.2s total runtime for 261 tests.

## Edge cases

- `config.model_revision` empty string — not your concern to validate
  here; `GemmaModelBackend`/`CerebrasDevBackend` already accept any string
  and this track doesn't add new validation on that field.
- A future fifth backend name — out of scope; your `ValueError` branch is
  the correct, minimal handling, not a TODO to anticipate names that don't
  exist yet.

## Failure-mode behavior

If `select_backend` itself raises (unknown backend string), that should
surface as a normal Python exception at `MyAgent.__init__` time — this is
construction-time configuration validation, not a runtime path
`choose_action`'s exception boundary needs to catch differently than it
already does. Do not add new try/except handling in `my_agent.py` for
this — a mis-set `Config.backend` should fail loudly at agent construction,
not be silently swallowed into a fallback backend.

## Definition of done

- All items in "Files you own this round" complete, tests passing.
- `docs/HANDOFF.md`'s "Known failures" #1 entry is either removed or
  updated to say it's fixed, with your branch/commit referenced — a
  one-line edit, not a rewrite.
- Your own `docs/superpowers/plans/2026-08-05-baseline-120-backend-wiring.md`
  plan file exists, written before you start coding (per
  `superpowers:writing-plans` / `superpowers:test-driven-development`).

## PR checklist

- [ ] `select_backend` matches the frozen interface signature exactly (Track 2/3/4 depend on it).
- [ ] All 4 backend strings tested, including the `platform` forwarding regression test.
- [ ] `grep` for the old hardcoded line returns nothing.
- [ ] Full suite green, count reported in PR description.
- [ ] `docs/HANDOFF.md` one-line status update included.
- [ ] No edits outside "Files you own this round."

## Handoff format

Update `docs/HANDOFF.md`'s baseline-120 status area with: branch name,
final commit SHA, test count added, and one sentence confirming
`select_backend`'s signature matches this file's frozen interface exactly
(so Track 2/4 know it's safe to integrate against).

## Merge preconditions

Full local suite green on your branch, `select_backend`'s signature
unchanged from this file's spec (a signature change after Track 2/4 have
started coding against it would break their in-flight work — if you must
change it, flag it immediately, don't merge silently).

## Rollback approach

This track's change is a single-line behavioral fix plus new, isolated
code — if a problem surfaces post-merge, `git revert` the merge commit
cleanly reverts to the old (buggy but previously-shipped) hardcoded
`GemmaModelBackend` construction with no other side effects, since no
other track's code depends on `select_backend` existing at import time
(Track 2/4 call it, but their own merges happen after yours per
`INTEGRATION.md`'s order).
