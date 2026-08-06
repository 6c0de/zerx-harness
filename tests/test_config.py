import json

import pytest

from zerx.config import Config


def test_default_config_values():
    cfg = Config()
    assert cfg.experiment_id == "dev"
    assert cfg.heuristic_first is False
    assert cfg.arbiter_on is False
    assert cfg.memory_on is True
    assert cfg.max_actions == 400
    assert cfg.max_wall_seconds == 7200
    # Must stay >= max_actions: the budget signal flips at 80% of this
    # value, and below the real action horizon it silences the model for
    # the remainder of every game.
    assert cfg.budget_soft_cap == 400


def test_default_budget_soft_cap_does_not_silence_the_model_early():
    """Regression: budget_soft_cap defaulted to 50 while games ran far
    longer, so BudgetSignal.should_favor_execution flipped at action 40 and
    stayed True — every later step with any click candidate skipped the
    model call entirely.
    """
    from zerx.budget import evaluate_budget

    cfg = Config()
    last_action = cfg.max_actions - 1
    assert evaluate_budget(0, cfg.budget_soft_cap).should_favor_execution is False
    midgame = evaluate_budget(cfg.max_actions // 2, cfg.budget_soft_cap)
    assert midgame.should_favor_execution is False
    assert evaluate_budget(last_action, cfg.budget_soft_cap).should_favor_execution is True


def test_max_actions_and_wall_seconds_from_env():
    cfg = Config.from_env({"ZERX_MAX_ACTIONS": "150", "ZERX_MAX_WALL_SECONDS": "60"})
    assert cfg.max_actions == 150
    assert cfg.max_wall_seconds == 60


def test_rejects_non_positive_max_actions():
    with pytest.raises(ValueError, match="max_actions"):
        Config(max_actions=0)


def test_rejects_negative_max_wall_seconds():
    with pytest.raises(ValueError, match="max_wall_seconds"):
        Config(max_wall_seconds=-1)


def test_low_budget_soft_cap_is_allowed_but_warns(caplog):
    """A low soft cap stays a legal ablation — it just must never be silent."""
    with caplog.at_level("WARNING", logger="zerx.config"):
        cfg = Config(budget_soft_cap=50, max_actions=400)
    assert cfg.budget_soft_cap == 50
    assert "budget_soft_cap=50 is below max_actions=400" in caplog.text


def test_from_env_missing_uses_defaults():
    assert Config.from_env({}) == Config()


def test_from_env_overrides_selected_fields():
    cfg = Config.from_env({
        "ZERX_HEURISTIC_FIRST": "true",
        "ZERX_BUDGET_SOFT_CAP": "25",
        "ZERX_EXPERIMENT_ID": "exp-1",
    })
    assert cfg.heuristic_first is True
    assert cfg.budget_soft_cap == 25
    assert cfg.experiment_id == "exp-1"
    assert cfg.memory_on is True  # untouched, stays default


def test_config_hash_is_deterministic():
    assert Config().config_hash() == Config().config_hash()


def test_config_hash_changes_with_field_value():
    assert Config().config_hash() != Config(heuristic_first=True).config_hash()


def test_to_json_round_trips_all_fields():
    cfg = Config(experiment_id="exp-2", budget_soft_cap=10)
    payload = json.loads(cfg.to_json())
    assert payload["experiment_id"] == "exp-2"
    assert payload["budget_soft_cap"] == 10


def test_default_backend_and_platform_are_safe():
    cfg = Config()
    assert cfg.backend == "fake"
    assert cfg.platform == "local"


def test_from_env_rejects_cerebras_dev_on_kaggle_platform():
    with pytest.raises(ValueError):
        Config.from_env({"ZERX_BACKEND": "cerebras_dev", "ZERX_PLATFORM": "kaggle"})


def test_from_env_allows_cerebras_dev_on_local_platform():
    cfg = Config.from_env({"ZERX_BACKEND": "cerebras_dev", "ZERX_PLATFORM": "local"})
    assert cfg.backend == "cerebras_dev"


def test_from_env_allows_gemma_kaggle_on_kaggle_platform():
    cfg = Config.from_env({"ZERX_BACKEND": "gemma_kaggle", "ZERX_PLATFORM": "kaggle"})
    assert cfg.platform == "kaggle"


def test_rejects_non_positive_budget_soft_cap():
    with pytest.raises(ValueError):
        Config(budget_soft_cap=0)
    with pytest.raises(ValueError):
        Config(budget_soft_cap=-1)


def test_from_env_rejects_non_positive_budget_soft_cap():
    with pytest.raises(ValueError):
        Config.from_env({"ZERX_BUDGET_SOFT_CAP": "0"})


def test_default_exact_state_suppression_is_off():
    cfg = Config()
    assert cfg.exact_state_suppression_on is False


def test_from_env_overrides_exact_state_suppression_on():
    cfg = Config.from_env({"ZERX_EXACT_STATE_SUPPRESSION_ON": "true"})
    assert cfg.exact_state_suppression_on is True


def test_from_env_missing_exact_state_suppression_keeps_default():
    cfg = Config.from_env({})
    assert cfg.exact_state_suppression_on is False


def test_duck_objects_on_defaults_false():
    config = Config()
    assert config.duck_objects_on is False


def test_duck_objects_on_from_env():
    config = Config.from_env({"ZERX_DUCK_OBJECTS_ON": "true"})
    assert config.duck_objects_on is True


def test_default_candidate_count_is_one():
    assert Config().candidate_count == 1


def test_from_env_overrides_candidate_count():
    cfg = Config.from_env({"ZERX_CANDIDATE_COUNT": "3"})
    assert cfg.candidate_count == 3


def test_rejects_non_positive_candidate_count():
    with pytest.raises(ValueError):
        Config(candidate_count=0)
    with pytest.raises(ValueError):
        Config(candidate_count=-1)


def test_from_env_rejects_non_positive_candidate_count():
    with pytest.raises(ValueError):
        Config.from_env({"ZERX_CANDIDATE_COUNT": "0"})


def test_structured_memory_on_defaults_false():
    assert Config().structured_memory_on is False


def test_from_env_missing_structured_memory_on_uses_default():
    cfg = Config.from_env({})
    assert cfg.structured_memory_on is False


def test_from_env_enables_structured_memory_on():
    cfg = Config.from_env({"ZERX_STRUCTURED_MEMORY_ON": "true"})
    assert cfg.structured_memory_on is True


def test_gemma_base_url_defaults_to_local_vllm_endpoint():
    assert Config().gemma_base_url == "http://localhost:8000/v1/chat/completions"


def test_from_env_overrides_gemma_base_url():
    cfg = Config.from_env({"ZERX_GEMMA_BASE_URL": "http://localhost:9000/v1/chat/completions"})
    assert cfg.gemma_base_url == "http://localhost:9000/v1/chat/completions"


def test_trace_export_path_defaults_to_none():
    assert Config().trace_export_path is None


def test_trace_export_path_read_from_env():
    config = Config.from_env({"ZERX_TRACE_EXPORT_PATH": "traces/foo.jsonl"})
    assert config.trace_export_path == "traces/foo.jsonl"


def test_trace_export_path_absent_from_env_stays_none():
    config = Config.from_env({})
    assert config.trace_export_path is None


def test_malformed_int_env_var_names_the_variable_and_the_value():
    """ARC-HANDOFF-006: `int()` raised a bare "invalid literal" from inside
    MyAgent.__init__, outside choose_action's catch-all, so a typo aborted
    the whole game with an unattributable message.
    """
    with pytest.raises(ValueError, match="ZERX_BUDGET_SOFT_CAP='abc'"):
        Config.from_env({"ZERX_BUDGET_SOFT_CAP": "abc"})


def test_malformed_float_env_var_names_the_variable_and_the_value():
    with pytest.raises(ValueError, match="ZERX_HEURISTIC_CONFIDENCE_THRESHOLD='high'"):
        Config.from_env({"ZERX_HEURISTIC_CONFIDENCE_THRESHOLD": "high"})


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"platform": "kaggle"}, "platform=kaggle"),
        ({"competition_mode": True}, "competition mode is active"),
        ({"internet_enabled": False}, "internet is disabled"),
    ],
)
def test_cerebras_lockout_covers_all_three_documented_conditions(kwargs, reason):
    """AGENTS.md requires rejection under all three; only the first existed."""
    with pytest.raises(ValueError, match=reason):
        Config(backend="cerebras_dev", **kwargs)


def test_cerebras_dev_is_still_allowed_in_ordinary_local_development():
    cfg = Config(backend="cerebras_dev")
    assert cfg.backend == "cerebras_dev"
