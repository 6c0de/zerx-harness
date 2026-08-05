from zerx.candidates import Candidate, static_candidate_score
from zerx.policy import ParsedAction
from zerx.types import Action, ActionName


def test_static_candidate_score_zero_for_unparsed_candidate():
    assert static_candidate_score("garbage", None) == 0.0


def test_static_candidate_score_full_for_clean_non_reset_parse():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    assert static_candidate_score('{"action": "ACTION1"}', parsed) == 1.0


def test_static_candidate_score_penalizes_repaired_output():
    clean = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    repaired = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=True)
    assert static_candidate_score("x", repaired) < static_candidate_score("x", clean)


def test_static_candidate_score_penalizes_reset_action():
    reset = ParsedAction(action=Action(name=ActionName.RESET), repaired=False)
    non_reset = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    assert static_candidate_score("x", reset) < static_candidate_score("x", non_reset)


def test_static_candidate_score_never_negative():
    reset_repaired = ParsedAction(action=Action(name=ActionName.RESET), repaired=True)
    assert static_candidate_score("x", reset_repaired) >= 0.0


def test_candidate_is_a_frozen_dataclass_with_expected_fields():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    c = Candidate(raw_response="raw", parsed=parsed, static_score=1.0)
    assert c.raw_response == "raw"
    assert c.parsed is parsed
    assert c.static_score == 1.0
