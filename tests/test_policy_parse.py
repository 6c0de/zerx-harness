from zerx.policy import parse_action
from zerx.types import ActionName

LEGAL = frozenset({ActionName.ACTION1, ActionName.ACTION5, ActionName.ACTION6})


def test_parse_action_valid_json_first_try():
    result = parse_action('{"action": "ACTION1"}', LEGAL)
    assert result is not None
    assert result.action.name == ActionName.ACTION1
    assert result.repaired is False


def test_parse_action_repairs_markdown_fenced_json():
    raw = '```json\n{"action": "ACTION5"}\n```'
    result = parse_action(raw, LEGAL)
    assert result is not None
    assert result.action.name == ActionName.ACTION5
    assert result.repaired is True


def test_parse_action_rejects_action_not_in_legal_actions():
    result = parse_action('{"action": "ACTION2"}', LEGAL)
    assert result is None


def test_parse_action_rejects_illegal_action6_coordinates():
    raw = '{"action": "ACTION6", "data": {"x": 100, "y": 0}}'
    assert parse_action(raw, LEGAL) is None


def test_parse_action_accepts_valid_action6_coordinates():
    raw = '{"action": "ACTION6", "data": {"x": 10, "y": 20}}'
    result = parse_action(raw, LEGAL)
    assert result is not None
    assert result.action.x == 10 and result.action.y == 20


def test_parse_action_rejects_malformed_json_after_repair_attempt():
    assert parse_action("not json at all, no braces", LEGAL) is None


def test_parse_action_rejects_missing_action_key():
    assert parse_action('{"foo": "bar"}', LEGAL) is None


def test_parse_action_rejects_unknown_action_name():
    assert parse_action('{"action": "FLY"}', LEGAL) is None


def test_parse_action_action6_requires_data():
    assert parse_action('{"action": "ACTION6"}', LEGAL) is None
