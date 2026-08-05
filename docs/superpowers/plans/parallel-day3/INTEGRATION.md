# Integration — merging all 4 tracks back into `master`

**One person does this** (whoever's coordinating — could be a 5th
"integrator" session, or whichever of the 4 finishes first and has time).
Don't have multiple people merge simultaneously — that reintroduces
exactly the conflict risk the whole parallel split was designed to avoid.

## Preconditions

All 4 branches (`feat/baseline-115-exact-state-memory`,
`feat/baseline-130-hypothesis-memory`, `feat/exp-140-vlm-refinement`,
`feat/exp-150-duck-tools-ab`) are pushed to the remote, each person has
confirmed their own full local test suite is green with their feature off
by default, and each has updated `docs/HANDOFF.md`'s parallel-work-split
table with a one-line status.

## Procedure

Merge **one branch at a time**, in this order (smallest/least-invasive
first, so later merges have fewer moving parts to reconcile against):

1. `feat/baseline-115-exact-state-memory` (new file
   `zerx/exact_state_memory.py`, smallest shared-file touch)
2. `feat/exp-150-duck-tools-ab` (mostly new/extended
   `zerx/perception.py` content, second-smallest shared-file touch)
3. `feat/exp-140-vlm-refinement` (new `zerx/candidates.py`, touches
   `zerx/policy.py`'s `decide()` with a new branch)
4. `feat/baseline-130-hypothesis-memory` (biggest — `zerx/memory.py`
   restructuring, merge last so it's reconciled against everyone else's
   `Config`/`decide()` additions, not the other way around)

For each branch:

```bash
git checkout master
git pull --ff-only
git checkout -b integration/day3   # only on the FIRST branch; reuse it after
git merge feat/<branch-name> --no-edit
```

**Expect conflicts in exactly two files**, and only mechanical ones if
everyone followed `README.md`'s etiquette:

- **`zerx/config.py`** — each branch added 1–2 new fields at the end of
  the dataclass and the matching `from_env(...)` call. A conflict here is
  "both sides added lines in the same place" — keep both sides' lines,
  order doesn't matter as long as every field name stays unique. Don't
  let git's auto-merge silently drop one side's field — always inspect
  the resolved file's full field list against what each branch's diff
  added.
- **`zerx/policy.py`** (only for `exp-140`, and only if `baseline-115`
  also ended up threading something through `decide()` per its own
  file's option (b)) — each addition should be a clearly separate `if
  config.<flag>:` branch. Resolve by keeping every branch, not by trying
  to merge their logic together.

If a conflict shows up anywhere else (a genuine logic conflict, not a
field-list stacking issue), that means someone didn't follow the
additive-only rule — stop, read both sides' actual changes, and resolve
by hand with the full context of what each track was trying to do (their
`person-N-*.md` file explains the intent). Don't blindly take "ours" or
"theirs."

After each merge:

```bash
.venv\Scripts\pytest.exe tests/ -q      # Windows
# or: .venv/bin/pytest tests/ -q         # Mac/Linux
```

Must show **more tests than before** (each track adds new tests) and
**zero failures**. If anything fails, fix it before merging the next
branch — don't stack a broken merge under another one.

## After all 4 are merged into `integration/day3`

1. Run the full suite one more time, confirm the total test count is
   sensible (136 baseline + all 4 tracks' new tests).
2. Spot-check that every new `Config` field still defaults to inert
   (grep for each new flag name, confirm its default and that no other
   branch's merge accidentally flipped it).
3. Confirm `agent/my_agent.py` still constructs cleanly and
   `tests/test_my_agent.py` still passes — this file is the one every
   track was most likely to touch, worth an explicit look even if tests
   pass.
4. Merge `integration/day3` into `master`:
   ```bash
   git checkout master
   git merge integration/day3 --no-edit
   .venv\Scripts\pytest.exe tests/ -q
   git push origin master
   ```
5. Update `docs/HANDOFF.md`: new commit, new test count, mark all 4
   parallel tracks complete, state the actual next action (per
   `STRATEGY.md` §7's table, `baseline-120-reki-core` — validating what's
   now built against real games — is the natural next step, but that's a
   separate decision for the human owner, not automatic).
6. The 4 `feat/...` branches can stay (don't delete — they're the
   individual record of each track) or be deleted once `integration/day3`
   is confirmed merged and pushed — human owner's call.

## If a track didn't finish or has real failures

Don't block the other 3 on it. Merge the ones that are actually done and
green; leave the incomplete branch for a later round. Note this in
`docs/HANDOFF.md` rather than silently dropping it.
