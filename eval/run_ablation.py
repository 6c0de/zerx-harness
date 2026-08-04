"""Experiment record schema, JSONL writer, and Config-variant sweep
generator used by ablation runs. The reproducibility fields here match
AGENTS.md's "Configuration and reproducibility" section. The actual
"play N local games with this config" loop is wired in once
agent/my_agent.py's harness adapter (Task 14) is exercised against real
games — this module owns the record format independent of that wiring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional

from zerx.config import Config


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
