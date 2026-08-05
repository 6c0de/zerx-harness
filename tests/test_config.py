import json

import pytest

from zerx.config import Config


def test_default_config_values():
    cfg = Config()
    assert cfg.experiment_id == "dev"
    assert cfg.heuristic_first is False
    assert cfg.arbiter_on is False
    assert cfg.memory_on is True
    assert cfg.budget_soft_cap == 50


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
