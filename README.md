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
