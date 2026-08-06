"""Experiment record schema, JSONL writer, and Config-variant sweep
generator used by ablation runs. The reproducibility fields here match
AGENTS.md's "Configuration and reproducibility" section. The actual
"play N local games with this config" loop is wired in once
agent/my_agent.py's harness adapter (Task 14) is exercised against real
games — this module owns the record format independent of that wiring.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from zerx.config import Config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Config field name -> ZERX_* env var name, matching Config.from_env exactly.
_CONFIG_ENV_MAP = {
    "experiment_id": "ZERX_EXPERIMENT_ID",
    "heuristic_first": "ZERX_HEURISTIC_FIRST",
    "heuristic_confidence_threshold": "ZERX_HEURISTIC_CONFIDENCE_THRESHOLD",
    "memory_on": "ZERX_MEMORY_ON",
    "memory_refresh_interval": "ZERX_MEMORY_REFRESH_INTERVAL",
    "arbiter_on": "ZERX_ARBITER_ON",
    "max_actions": "ZERX_MAX_ACTIONS",
    "max_wall_seconds": "ZERX_MAX_WALL_SECONDS",
    "budget_soft_cap": "ZERX_BUDGET_SOFT_CAP",
    "model_revision": "ZERX_MODEL_REVISION",
    "backend": "ZERX_BACKEND",
    "platform": "ZERX_PLATFORM",
    "exact_state_suppression_on": "ZERX_EXACT_STATE_SUPPRESSION_ON",
    "duck_objects_on": "ZERX_DUCK_OBJECTS_ON",
    "candidate_count": "ZERX_CANDIDATE_COUNT",
    "structured_memory_on": "ZERX_STRUCTURED_MEMORY_ON",
}


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    config_hash: str
    game_id: str
    actions_taken: int
    levels_completed: int
    rhae: Optional[float]
    wall_time_seconds: float
    invalid_outputs: int
    repairs: int
    fallbacks: int
    resets: int
    exceptions: int

    def to_json_line(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def write_records(records: Iterable[ExperimentRecord], path: Path) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json_line() + "\n")


def _load_my_agent_class():
    """Import MyAgent from agent/my_agent.py, same pattern as
    scripts/play_local.py's load_my_agent_class (not imported directly —
    that script is a stable, unowned reference this round)."""
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def run_games(
    config: Config,
    game_ids: Sequence[str],
    max_steps: int = 200,
) -> List[ExperimentRecord]:
    """Play each game_id locally via the real arc_agi Arcade + MyAgent
    (same NORMAL-mode pattern as scripts/play_local.py), driving MyAgent's
    backend/platform choice by setting the matching ZERX_* environment
    variables from `config` around construction (MyAgent itself still
    calls Config.from_env() internally; this function does not change
    that). Restores prior env state afterward. Returns one ExperimentRecord
    per game_id, with `rhae` populated from arc.get_scorecard()'s
    EnvironmentScorecard for that game when available, else None.

    Known gap: invalid_outputs/repairs/fallbacks/exceptions are per-decision
    counters internal to MyAgent's choose_action() and are not currently
    aggregated anywhere observable from outside; they are recorded as 0
    here, which is a placeholder, not a measured zero.
    """
    if not game_ids:
        return []

    vendor = ROOT / "vendor" / "ARC-AGI-3-Agents"
    for path in (str(ROOT), str(vendor)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import arc_agi
    from arc_agi import OperationMode

    prior_env = {name: os.environ.get(name) for name in _CONFIG_ENV_MAP.values()}
    try:
        for field_name, env_name in _CONFIG_ENV_MAP.items():
            os.environ[env_name] = str(getattr(config, field_name))

        # `max_steps` overrides whatever `config.max_actions` says, via the
        # same env channel MyAgent.__init__ reads. The previous
        # `min(MyAgentCls.MAX_ACTIONS, max_steps)` form could only lower the
        # cap below the inherited 80, so `max_steps=200` silently ran 80.
        os.environ["ZERX_MAX_ACTIONS"] = str(max_steps)

        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        MyAgentCls = _load_my_agent_class()

        played: List[tuple] = []
        for game_id in game_ids:
            env = arc.make(game_id)
            if env is None:
                logger.warning("run_games: could not create env for %r, skipping", game_id)
                continue

            agent = MyAgentCls(
                card_id="eval-run-games",
                game_id=game_id,
                agent_name=f"MyAgent.run_games.{game_id}",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=env,
                tags=["eval-run-games"],
            )
            start = time.monotonic()
            agent.main()
            elapsed = time.monotonic() - start

            final = agent.frames[-1]
            played.append((game_id, final.levels_completed, agent.action_counter, elapsed))

        sc = arc.get_scorecard()
        score_by_game = {}
        for env_score_list in sc.environments:
            short_id = env_score_list.id.split("-")[0]
            score_by_game[short_id] = env_score_list

        records: List[ExperimentRecord] = []
        for game_id, levels_completed, actions_taken, elapsed in played:
            env_score_list = score_by_game.get(game_id)
            rhae: Optional[float] = None
            resets = 0
            if env_score_list is not None:
                resets = env_score_list.resets
                last_run = env_score_list.runs[-1] if env_score_list.runs else None
                if last_run is not None and last_run.message:
                    logger.info(
                        "run_games: %s has no rhae (%s)", game_id, last_run.message
                    )
                else:
                    rhae = env_score_list.score

            records.append(
                ExperimentRecord(
                    experiment_id=config.experiment_id,
                    config_hash=config.config_hash(),
                    game_id=game_id,
                    actions_taken=actions_taken,
                    levels_completed=levels_completed,
                    rhae=rhae,
                    wall_time_seconds=elapsed,
                    invalid_outputs=0,
                    repairs=0,
                    fallbacks=0,
                    resets=resets,
                    exceptions=0,
                )
            )
        return records
    finally:
        for env_name, value in prior_env.items():
            if value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = value


def sweep_configs(base: Config, **variants: List[bool]) -> List[Config]:
    """Single-flag-at-a-time sweep: for each keyword arg (a Config field
    name) and its list of candidate values, yield one Config per value with
    everything else held at `base`. Matches AGENTS.md's "one behavioral
    change per experiment where possible" rule rather than a combinatorial
    explosion.
    """
    configs = [base]
    for field_name, values in variants.items():
        for value in values:
            if value == getattr(base, field_name):
                continue
            configs.append(replace(base, **{field_name: value}))
    return configs
