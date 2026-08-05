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


from zerx.candidates import generate_candidates
from zerx.model_backend import FakeModelBackend

LEGAL = frozenset({ActionName.RESET, ActionName.ACTION1, ActionName.ACTION5, ActionName.ACTION6})


def test_generate_candidates_calls_backend_exactly_count_times():
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'] * 3)
    candidates = generate_candidates(backend, "prompt", LEGAL, count=3)
    assert backend.call_count == 3
    assert len(candidates) == 3


def test_generate_candidates_records_parse_failure_without_crashing():
    backend = FakeModelBackend(
        responses=['{"action": "ACTION1"}', "garbage", '{"action": "ACTION5"}']
    )
    candidates = generate_candidates(backend, "prompt", LEGAL, count=3)
    assert candidates[0].parsed is not None
    assert candidates[1].parsed is None
    assert candidates[1].static_score == 0.0
    assert candidates[2].parsed is not None


def test_generate_candidates_stores_raw_response_and_score():
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'])
    candidates = generate_candidates(backend, "prompt", LEGAL, count=1)
    assert candidates[0].raw_response == '{"action": "ACTION1"}'
    assert candidates[0].static_score == 1.0
