"""Typed, serializable configuration. Only this module reads environment
variables — feature modules receive a resolved Config via dependency
injection and must never read os.environ themselves (see AGENTS.md).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    return default if raw is None else int(raw)


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    return default if raw is None else float(raw)


def _env_str(env: Mapping[str, str], key: str, default: str) -> str:
    return env.get(key, default)


@dataclass(frozen=True)
class Config:
    experiment_id: str = "dev"
    heuristic_first: bool = False
    heuristic_confidence_threshold: float = 0.8
    memory_on: bool = True
    memory_refresh_interval: int = 10
    arbiter_on: bool = False
    budget_soft_cap: int = 50
    model_revision: str = "gemma-4-31b-it"
    backend: str = "fake"  # "fake" | "cerebras_dev" | "gemma_local" | "gemma_kaggle"
    platform: str = "local"  # "local" | "colab" | "kaggle"
    candidate_count: int = 1

    def __post_init__(self) -> None:
        if self.backend == "cerebras_dev" and self.platform == "kaggle":
            raise ValueError(
                "cerebras_dev is a development-only backend and must never be "
                "selected on platform=kaggle"
            )
        if self.budget_soft_cap <= 0:
            raise ValueError("budget_soft_cap must be positive")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Config":
        env = os.environ if env is None else env
        return cls(
            experiment_id=_env_str(env, "ZERX_EXPERIMENT_ID", cls.experiment_id),
            heuristic_first=_env_bool(env, "ZERX_HEURISTIC_FIRST", cls.heuristic_first),
            heuristic_confidence_threshold=_env_float(
                env,
                "ZERX_HEURISTIC_CONFIDENCE_THRESHOLD",
                cls.heuristic_confidence_threshold,
            ),
            memory_on=_env_bool(env, "ZERX_MEMORY_ON", cls.memory_on),
            memory_refresh_interval=_env_int(
                env, "ZERX_MEMORY_REFRESH_INTERVAL", cls.memory_refresh_interval
            ),
            arbiter_on=_env_bool(env, "ZERX_ARBITER_ON", cls.arbiter_on),
            budget_soft_cap=_env_int(env, "ZERX_BUDGET_SOFT_CAP", cls.budget_soft_cap),
            model_revision=_env_str(env, "ZERX_MODEL_REVISION", cls.model_revision),
            backend=_env_str(env, "ZERX_BACKEND", cls.backend),
            platform=_env_str(env, "ZERX_PLATFORM", cls.platform),
            candidate_count=_env_int(env, "ZERX_CANDIDATE_COUNT", cls.candidate_count),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:12]
