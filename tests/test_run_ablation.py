import json

from eval.run_ablation import ExperimentRecord, sweep_configs, write_records
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
