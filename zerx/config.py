"""Typed, serializable configuration. Only this module reads environment
variables — feature modules receive a resolved Config via dependency
injection and must never read os.environ themselves (see AGENTS.md).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        # A bare `ValueError: invalid literal for int()` names neither the
        # variable nor the value, and it escapes from MyAgent.__init__ --
        # outside choose_action's catch-all -- so one typo aborted the whole
        # game with an unattributable message. Raising loudly (rather than
        # silently defaulting) is deliberate: a silent fallback would hide a
        # misconfigured experiment and make its results quietly wrong.
        raise ValueError(
            f"{key}={raw!r} is not a valid integer (expected e.g. {key}=100)"
        ) from None


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"{key}={raw!r} is not a valid number (expected e.g. {key}=0.8)"
        ) from None


def _env_str(env: Mapping[str, str], key: str, default: str) -> str:
    return env.get(key, default)


def _env_optional_str(env: Mapping[str, str], key: str, default: Optional[str]) -> Optional[str]:
    return env.get(key, default)


@dataclass(frozen=True)
class Config:
    experiment_id: str = "dev"
    heuristic_first: bool = False
    heuristic_confidence_threshold: float = 0.8
    memory_on: bool = True
    memory_refresh_interval: int = 10
    arbiter_on: bool = False
    opening_probe_on: bool = True  # spend the opening actions trying each
    # legal action once, with no model call, so the evidence table the model
    # reads is already filled in before its first real decision. Unusually
    # for this project this defaults ON: discovering the control scheme is
    # the central task, the probe costs at most one action per action-name,
    # and without it the model spends those same actions guessing. Still
    # fully ablatable via ZERX_OPENING_PROBE_ON=false.
    opening_probe_actions: int = 12  # bound on the probe phase, counted in
    # actions taken. Large enough for all six action names plus the opening
    # RESET and some slack; small enough that a mid-game change in the legal
    # set cannot restart probing late in a run.
    max_actions: int = 400  # per-game action cap. The vendored
    # `agents.agent.Agent` base class hardcodes MAX_ACTIONS = 80 purely as a
    # "don't loop forever" guard, and MyAgent inherited it, so every Kaggle
    # game stopped at 81 actions — far below what completing a level takes.
    # agent/my_agent.py sets self.MAX_ACTIONS from this field. 400 matches
    # the upstream framework's own reasoning_agent.py choice.
    max_wall_seconds: int = 7200  # per-game wall-clock guard (0 disables).
    # Raising max_actions raises wall-clock exposure against Kaggle's ~9h
    # notebook limit; this bounds a single game's runtime so a slow/hung
    # model degrades one game instead of losing the whole run to a kill.
    budget_soft_cap: int = 400  # keep in step with max_actions: this is the
    # denominator behind BudgetSignal.should_favor_execution, which flips at
    # 80% and (in policy.decide) then prefers a heuristic click over a model
    # call. A soft cap far below the real action cap silently turns most of
    # every game into heuristic-only play with the model never consulted.
    model_revision: str = "gemma-4-31b-it"
    backend: str = "fake"  # "fake" | "cerebras_dev" | "gemma_local" | "gemma_kaggle"
    platform: str = "local"  # "local" | "colab" | "kaggle"
    competition_mode: bool = False  # scored competition run. Together
    # with internet_enabled these complete AGENTS.md's three required
    # cerebras_dev lockout conditions (ARC-HANDOFF-006); only the
    # platform=="kaggle" one existed before.
    internet_enabled: bool = True
    exact_state_suppression_on: bool = False
    duck_objects_on: bool = False  # exp-150-duck-tools Variants A+B (zerx/scene.py) — off by default
    candidate_count: int = 1
    structured_memory_on: bool = False
    gemma_base_url: str = "http://localhost:8000/v1/chat/completions"
    model_path: Optional[str] = None  # filesystem directory the weights load
    # from, used by backend="gemma_kaggle" (in-process transformers). Unlike
    # gemma_base_url — which points at a vLLM server we start ourselves on
    # Colab — Kaggle has no server to talk to: vllm is absent from the image,
    # internet is disabled, and the competition's offline wheels do not ship
    # it. See docs/superpowers/experiments/kaggle-env-probe.md.
    model_dtype: str = "bfloat16"  # the RTX PRO 6000 Blackwell measured on
    # Kaggle has ~96 GB against 62.58 GB of bf16 weights, so no quantization
    # is needed. This contradicts the 2026-08-06 fp8 decision, which assumed
    # a 48 GB card; that decision is the human owner's to revisit.
    max_new_tokens: int = 96  # decide() allows one model call per action and
    # asks for a single JSON object, so generation length is pure latency.
    trace_export_path: Optional[str] = None  # dev-only: when set, MyAgent
    # writes one JSONL trace file per game via zerx/trace.py -- off by
    # default, never read outside agent/my_agent.py's construction.

    def __post_init__(self) -> None:
        if self.backend == "cerebras_dev":
            # AGENTS.md requires rejection "whenever platform=kaggle,
            # competition mode is active, or internet is disabled". Only the
            # first of the three was ever implemented (ARC-HANDOFF-006).
            for condition, reason in (
                (self.platform == "kaggle", "platform=kaggle"),
                (self.competition_mode, "competition mode is active"),
                (not self.internet_enabled, "internet is disabled"),
            ):
                if condition:
                    raise ValueError(
                        "cerebras_dev is a development-only backend and must "
                        f"never be selected when {reason}"
                    )
        if self.budget_soft_cap <= 0:
            raise ValueError("budget_soft_cap must be positive")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.opening_probe_actions < 0:
            raise ValueError("opening_probe_actions must be >= 0")
        if self.max_actions < 1:
            raise ValueError("max_actions must be >= 1")
        if self.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds must be >= 0")
        if self.budget_soft_cap < self.max_actions:
            # Not an error — a deliberately low soft cap is a valid ablation.
            # But it is silent and expensive when unintended (the model stops
            # being called at 80% of the soft cap, not of the real horizon),
            # so it must never happen without appearing in the log.
            logger.warning(
                "budget_soft_cap=%d is below max_actions=%d: the budget signal "
                "will favor heuristic execution from action %d onward, so the "
                "model is not consulted for the remaining ~%d actions of a "
                "full-length game.",
                self.budget_soft_cap,
                self.max_actions,
                int(self.budget_soft_cap * 0.8),
                max(0, self.max_actions - int(self.budget_soft_cap * 0.8)),
            )

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
            opening_probe_on=_env_bool(
                env, "ZERX_OPENING_PROBE_ON", cls.opening_probe_on
            ),
            opening_probe_actions=_env_int(
                env, "ZERX_OPENING_PROBE_ACTIONS", cls.opening_probe_actions
            ),
            max_actions=_env_int(env, "ZERX_MAX_ACTIONS", cls.max_actions),
            max_wall_seconds=_env_int(env, "ZERX_MAX_WALL_SECONDS", cls.max_wall_seconds),
            budget_soft_cap=_env_int(env, "ZERX_BUDGET_SOFT_CAP", cls.budget_soft_cap),
            model_revision=_env_str(env, "ZERX_MODEL_REVISION", cls.model_revision),
            backend=_env_str(env, "ZERX_BACKEND", cls.backend),
            platform=_env_str(env, "ZERX_PLATFORM", cls.platform),
            competition_mode=_env_bool(
                env, "ZERX_COMPETITION_MODE", cls.competition_mode
            ),
            internet_enabled=_env_bool(
                env, "ZERX_INTERNET_ENABLED", cls.internet_enabled
            ),
            exact_state_suppression_on=_env_bool(
                env, "ZERX_EXACT_STATE_SUPPRESSION_ON", cls.exact_state_suppression_on
            ),
            duck_objects_on=_env_bool(env, "ZERX_DUCK_OBJECTS_ON", cls.duck_objects_on),
            candidate_count=_env_int(env, "ZERX_CANDIDATE_COUNT", cls.candidate_count),
            structured_memory_on=_env_bool(
                env, "ZERX_STRUCTURED_MEMORY_ON", cls.structured_memory_on
            ),
            gemma_base_url=_env_str(env, "ZERX_GEMMA_BASE_URL", cls.gemma_base_url),
            model_path=_env_optional_str(env, "ZERX_MODEL_PATH", cls.model_path),
            model_dtype=_env_str(env, "ZERX_MODEL_DTYPE", cls.model_dtype),
            max_new_tokens=_env_int(env, "ZERX_MAX_NEW_TOKENS", cls.max_new_tokens),
            trace_export_path=_env_optional_str(
                env, "ZERX_TRACE_EXPORT_PATH", cls.trace_export_path
            ),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:12]
