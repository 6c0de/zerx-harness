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


def test_generate_candidates_isolates_backend_failure_to_one_candidate():
    """A backend that raises on one call (not just returns unparseable
    text) must not abort the whole batch -- the surrounding candidates
    must still be collected."""

    class FlakyBackend:
        def __init__(self, outcomes):
            self._outcomes = list(outcomes)
            self.call_count = 0

        def generate(self, prompt: str) -> str:
            self.call_count += 1
            outcome = self._outcomes.pop(0)
            if outcome is None:
                raise RuntimeError("simulated transient backend failure")
            return outcome

    backend = FlakyBackend(['{"action": "ACTION1"}', None, '{"action": "ACTION5"}'])
    candidates = generate_candidates(backend, "prompt", LEGAL, count=3)
    assert backend.call_count == 3
    assert len(candidates) == 3
    assert candidates[0].parsed is not None
    assert candidates[1].parsed is None
    assert candidates[1].static_score == 0.0
    assert candidates[2].parsed is not None


from zerx.candidates import select_best_candidate


def test_select_best_candidate_picks_highest_score():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    low = Candidate(raw_response="a", parsed=None, static_score=0.0)
    high = Candidate(raw_response="b", parsed=parsed, static_score=1.0)
    assert select_best_candidate([low, high]) is high


def test_select_best_candidate_breaks_ties_by_earliest_candidate():
    first = Candidate(raw_response="a", parsed=None, static_score=0.5)
    second = Candidate(raw_response="b", parsed=None, static_score=0.5)
    assert select_best_candidate([first, second]) is first


def test_select_best_candidate_empty_list_returns_none():
    assert select_best_candidate([]) is None


from zerx.candidates import select_candidate
from zerx.config import Config


def test_select_candidate_never_calls_arbiter_when_arbiter_on_false():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    candidates = [Candidate(raw_response="a", parsed=parsed, static_score=1.0)]
    arbiter = FakeModelBackend(responses=["0"])
    picked = select_candidate(candidates, Config(arbiter_on=False), arbiter=arbiter)
    assert picked is candidates[0]
    assert arbiter.call_count == 0


def test_select_candidate_never_calls_arbiter_when_none_provided():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    candidates = [Candidate(raw_response="a", parsed=parsed, static_score=1.0)]
    picked = select_candidate(candidates, Config(arbiter_on=True), arbiter=None)
    assert picked is candidates[0]


def test_select_candidate_consults_arbiter_when_arbiter_on_true():
    low = Candidate(
        raw_response="a",
        parsed=ParsedAction(action=Action(name=ActionName.ACTION5), repaired=False),
        static_score=1.0,
    )
    high = Candidate(
        raw_response="b",
        parsed=ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False),
        static_score=1.0,
    )
    arbiter = FakeModelBackend(responses=["1"])
    picked = select_candidate([low, high], Config(arbiter_on=True), arbiter=arbiter)
    assert picked is high
    assert arbiter.call_count == 1


def test_select_candidate_falls_back_to_deterministic_when_arbiter_output_invalid():
    only = Candidate(
        raw_response="a",
        parsed=ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False),
        static_score=1.0,
    )
    arbiter = FakeModelBackend(responses=["not an int"])
    picked = select_candidate([only], Config(arbiter_on=True), arbiter=arbiter)
    assert picked is only


def test_select_candidate_empty_list_returns_none():
    assert select_candidate([], Config(), arbiter=None) is None
