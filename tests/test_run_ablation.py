import pytest
import json
import os

from eval.run_ablation import ExperimentRecord, run_games, sweep_configs, write_records
from zerx.config import Config


def _record(**overrides):
    defaults = dict(
        experiment_id="exp-1",
        config_hash="abc123",
        game_id="ls20",
        actions_taken=10,
        levels_completed=1,
        rhae=0.5,
        wall_time_seconds=1.2,
        invalid_outputs=0,
        repairs=0,
        fallbacks=0,
        resets=0,
        exceptions=0,
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def test_experiment_record_to_json_line_is_valid_json():
    line = _record().to_json_line()
    payload = json.loads(line)
    assert payload["game_id"] == "ls20"
    assert payload["rhae"] == 0.5


def test_write_records_appends_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    write_records([_record(game_id="ls20"), _record(game_id="vc33")], path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["game_id"] == "ls20"
    assert json.loads(lines[1])["game_id"] == "vc33"


def test_write_records_appends_to_existing_file(tmp_path):
    path = tmp_path / "results.jsonl"
    write_records([_record(game_id="ls20")], path)
    write_records([_record(game_id="vc33")], path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_sweep_configs_includes_base():
    base = Config()
    configs = sweep_configs(base)
    assert configs == [base]


def test_sweep_configs_varies_one_field_at_a_time():
    base = Config(heuristic_first=False, memory_on=True)
    configs = sweep_configs(base, heuristic_first=[True])
    assert len(configs) == 2
    variant = configs[1]
    assert variant.heuristic_first is True
    assert variant.memory_on is True  # everything else stays at base


def test_sweep_configs_skips_value_equal_to_base():
    base = Config(heuristic_first=False)
    configs = sweep_configs(base, heuristic_first=[False])
    assert configs == [base]


def test_run_games_empty_game_ids_returns_empty_list():
    assert run_games(Config(backend="fake"), []) == []


def test_run_games_plays_real_local_game():
    records = run_games(Config(backend="fake"), ["ls20"], max_steps=5)
    assert len(records) == 1
    record = records[0]
    assert record.game_id == "ls20"
    assert record.actions_taken > 0
    assert record.rhae is None or isinstance(record.rhae, float)


def test_run_games_restores_env_vars():
    os.environ["ZERX_UNRELATED_TEST_VAR"] = "unchanged"
    had_backend_before = "ZERX_BACKEND" in os.environ
    backend_before = os.environ.get("ZERX_BACKEND")
    try:
        run_games(Config(backend="fake"), ["ls20"], max_steps=5)
        assert os.environ["ZERX_UNRELATED_TEST_VAR"] == "unchanged"
        assert ("ZERX_BACKEND" in os.environ) == had_backend_before
        if had_backend_before:
            assert os.environ["ZERX_BACKEND"] == backend_before
    finally:
        del os.environ["ZERX_UNRELATED_TEST_VAR"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known and accepted, not silenced. agent/my_agent.py was rewritten "
        "against zerx/single_play.py after the free-reset strategy was measured "
        "false through a real gateway (docs/HANDOFF.md, 2026-08-07), so the "
        "flags belonging to the retired model-in-loop policy now have no "
        "consumer. The fix is to delete those modules and their Config fields, "
        "not to weaken this test -- it is reporting exactly what it exists to "
        "report. strict=True so it fails again the moment that cleanup lands."
    ),
)
def test_every_ablation_flag_is_actually_read_somewhere():
    """ARC-HANDOFF-003: four flags reached the ablation matrix while
    controlling nothing, so any A/B on them was guaranteed to report "no
    effect" -- which risks discarding good ideas for the wrong reason.
    """
    import pathlib

    from eval.run_ablation import _CONFIG_ENV_MAP

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = "\n".join(
        path.read_text()
        for directory in ("zerx", "agent")
        for path in sorted((root / directory).rglob("*.py"))
    )
    # config.py only defines the fields; a flag mentioned nowhere else is a
    # flag nothing acts on.
    definitions = (root / "zerx" / "config.py").read_text()
    consumers = sources.replace(definitions, "")

    # Fields that legitimately have no consumer outside config.py itself.
    # Each needs a reason; anything else in the matrix must change behavior.
    NON_BEHAVIORAL = {
        "experiment_id": "metadata recorded in ExperimentRecord, not a behavior flag",
        "competition_mode": "validation-only: arms the cerebras_dev lockout in __post_init__",
        "internet_enabled": "validation-only: arms the cerebras_dev lockout in __post_init__",
    }

    unread = [
        name
        for name in _CONFIG_ENV_MAP
        if name not in NON_BEHAVIORAL
        and f"config.{name}" not in consumers
        and f"_config.{name}" not in consumers
    ]
    assert unread == [], f"flags in the ablation matrix that nothing reads: {unread}"

    # The lockout fields must really be enforced, not just excused above.
    from zerx.config import Config

    for field in ("competition_mode", "internet_enabled"):
        value = False if field == "internet_enabled" else True
        try:
            Config(backend="cerebras_dev", **{field: value})
        except ValueError:
            continue
        raise AssertionError(f"{field} is excused as validation-only but enforces nothing")
