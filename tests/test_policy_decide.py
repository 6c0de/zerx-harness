from zerx.budget import BudgetSignal
from zerx.config import Config
from zerx.heuristics import ClickCandidate, DeadSignatureTracker
from zerx.memory import MemoryState
from zerx.model_backend import FakeModelBackend
from zerx.perception import LabeledObject, PerceptionResult
from zerx.policy import build_prompt, decide
from zerx.types import Action, ActionName, GameFrame

LEGAL = frozenset(
    {
        ActionName.RESET,
        ActionName.ACTION1,
        ActionName.ACTION5,
        ActionName.ACTION6,
    }
)


def _frame(grid, is_game_over=False, legal=LEGAL):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=legal,
        is_game_over=is_game_over,
    )


def _blank_frame(**kwargs):
    return _frame([[0, 0], [0, 0]], **kwargs)


def test_decide_returns_reset_when_game_over():
    decision, _ = decide(
        frame=_blank_frame(is_game_over=True),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.RESET
    assert decision.source == "reset"


def test_decide_uses_model_action_when_valid():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.ACTION1
    assert decision.source == "model"
    assert decision.repaired is False


def test_decide_repairs_markdown_fenced_model_output():
    raw = '```json\n{"action": "ACTION5"}\n```'
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[raw]),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.ACTION5
    assert decision.repaired is True


def test_decide_falls_back_to_heuristic_when_model_output_invalid():
    frame = _frame([[0, 0], [0, 5]])  # one clickable object
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=["garbage, not json"]),
        actions_taken=0,
    )
    assert decision.source == "fallback_heuristic"
    assert decision.action.name == ActionName.ACTION6


def test_decide_falls_back_to_deterministic_when_no_candidates_and_model_invalid():
    decision, _ = decide(
        frame=_blank_frame(),  # no objects -> no click candidates
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=["garbage, not json"]),
        actions_taken=0,
    )
    assert decision.source == "fallback_deterministic"
    assert decision.action.name in LEGAL


def test_decide_heuristic_first_skips_model_call_when_confident():
    frame = _frame([[0, 0], [0, 5]])
    backend = FakeModelBackend(responses=[])  # would raise if called
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(heuristic_first=True, heuristic_confidence_threshold=0.0),
        backend=backend,
        actions_taken=0,
    )
    assert decision.source == "heuristic"
    assert backend.call_count == 0


def test_decide_never_raises_when_backend_raises():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),  # raises RuntimeError internally
        actions_taken=0,
    )
    assert decision.action.name in LEGAL


def test_decide_records_budget_signal():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(budget_soft_cap=50),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=12,
    )
    assert decision.budget == BudgetSignal(actions_taken=12, soft_cap=50, should_favor_execution=False)


def test_decide_memory_refreshes_when_on_and_due():
    memory = MemoryState(summary="s", step_count=8, last_refreshed_step=0)
    _, new_memory = decide(
        frame=_blank_frame(),
        history=(),
        memory=memory,
        dead_signatures=DeadSignatureTracker(),
        config=Config(memory_on=True, memory_refresh_interval=9),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert new_memory.step_count == 9
    assert new_memory.last_refreshed_step == 9


def test_decide_memory_untouched_when_off():
    memory = MemoryState(summary="s", step_count=8, last_refreshed_step=0)
    _, new_memory = decide(
        frame=_blank_frame(),
        history=(),
        memory=memory,
        dead_signatures=DeadSignatureTracker(),
        config=Config(memory_on=False),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert new_memory is memory


def test_decide_random_fallback_stays_within_legal_actions_and_never_raises():
    narrow_legal = frozenset({ActionName.ACTION7})
    decision, _ = decide(
        frame=_blank_frame(legal=narrow_legal),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=["garbage"]),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.ACTION7
    assert decision.source == "fallback_random"


def test_decide_records_target_object_label_on_heuristic_source():
    frame = _frame([[0, 0], [0, 5]])
    backend = FakeModelBackend(responses=[])
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(heuristic_first=True, heuristic_confidence_threshold=0.0),
        backend=backend,
        actions_taken=0,
    )
    assert decision.source == "heuristic"
    assert decision.target_object_label == "obj0"


def test_decide_leaves_target_object_label_none_on_model_source():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert decision.source == "model"
    assert decision.target_object_label is None


def test_build_prompt_lists_ranked_click_candidates():
    perception = PerceptionResult(
        ascii_grid="05",
        objects=(LabeledObject(label="obj0", color=5, cells=((1, 0),)),),
    )
    candidates = [ClickCandidate(x=1, y=0, object_label="obj0", score=0.5)]
    legal = frozenset({ActionName.ACTION6, ActionName.RESET})
    prompt = build_prompt(perception, MemoryState(), candidates, legal_actions=legal)
    assert "obj0" in prompt
    assert "x=1, y=0" in prompt


def test_build_prompt_without_candidates_says_so_when_action6_legal():
    perception = PerceptionResult(ascii_grid="0", objects=())
    legal = frozenset({ActionName.ACTION6, ActionName.RESET})
    prompt = build_prompt(perception, MemoryState(), legal_actions=legal)
    assert "no click candidates" in prompt


def test_build_prompt_omits_candidates_section_when_action6_not_legal():
    """Root cause of a real 0.0 Colab run (docs/HANDOFF.md item 6's live
    reproduction, and item 10's real gemma-4-31b-it trace): showing
    'Ranked click candidates' unconditionally -- even on a turn where
    ACTION6 isn't legal -- contradicts the 'Legal actions this turn' line
    right below it. Confirmed via the model's own raw output: it
    repeatedly proposed {"action": "ACTION6", ...} on ls20 turns where
    ACTION6 was never legal, got correctly rejected by parse_action(), and
    fell through to the fallback chain 76 of 101 steps. When ACTION6
    isn't legal there is nothing to click toward, so the section (and its
    "if you choose ACTION6" framing) must not appear at all.
    """
    perception = PerceptionResult(
        ascii_grid="05",
        objects=(LabeledObject(label="obj0", color=5, cells=((1, 0),)),),
    )
    candidates = [ClickCandidate(x=1, y=0, object_label="obj0", score=0.5)]
    legal = frozenset({ActionName.ACTION1, ActionName.RESET})  # no ACTION6
    prompt = build_prompt(perception, MemoryState(), candidates, legal_actions=legal)
    assert "Ranked click candidates" not in prompt
    assert "if you choose ACTION6" not in prompt
    assert "click (x=" not in prompt  # the candidate-list line format itself


def test_build_prompt_lists_legal_actions():
    perception = PerceptionResult(ascii_grid="0", objects=())
    legal = frozenset({ActionName.ACTION1, ActionName.ACTION6, ActionName.RESET})
    prompt = build_prompt(perception, MemoryState(), legal_actions=legal)
    assert "ACTION1" in prompt
    assert "ACTION6" in prompt
    assert "RESET" in prompt
    assert "ACTION2" not in prompt


def test_build_prompt_without_legal_actions_says_none():
    perception = PerceptionResult(ascii_grid="0", objects=())
    prompt = build_prompt(perception, MemoryState())
    assert "Legal actions this turn: (none)" in prompt


def test_build_prompt_includes_budget_signal():
    perception = PerceptionResult(ascii_grid="0", objects=())
    budget = BudgetSignal(actions_taken=5, soft_cap=10, should_favor_execution=False)
    prompt = build_prompt(perception, MemoryState(), budget=budget)
    assert "5" in prompt
    assert "10" in prompt
    assert "strategy signal only" in prompt


def test_build_prompt_without_budget_says_so():
    perception = PerceptionResult(ascii_grid="0", objects=())
    prompt = build_prompt(perception, MemoryState())
    assert "(no budget signal)" in prompt


def test_decide_budget_favoring_execution_triggers_heuristic_even_when_heuristic_first_off():
    """Fix 3a (final whole-branch review): the budget signal must be able to
    trigger heuristic use on its own, additively alongside heuristic_first,
    not just when heuristic_first is already on. actions_taken=45 against
    the default budget_soft_cap=50 gives ratio 0.9 >= the 0.8
    favor_threshold, so should_favor_execution is True; a clickable object
    gives a candidate with score > 0.0.
    """
    frame = _frame([[0, 0], [0, 5]])  # one clickable object -> a candidate
    backend = FakeModelBackend(responses=[])  # would raise if called
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(heuristic_first=False, budget_soft_cap=50),
        backend=backend,
        actions_taken=45,
    )
    assert decision.source == "heuristic"
    assert decision.action.name == ActionName.ACTION6
    assert backend.call_count == 0


def test_decide_budget_favoring_execution_without_candidates_falls_through_unchanged():
    """Negative case: same high actions_taken (budget favors execution), but
    no clickable object -> no candidates. The new budget branch must not
    invent a move when there is nothing sensible to execute; behavior falls
    through to the model/fallback chain exactly as before this change.
    """
    decision, _ = decide(
        frame=_blank_frame(),  # no objects -> no click candidates
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(heuristic_first=False, budget_soft_cap=50),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=45,
    )
    assert decision.source == "model"
    assert decision.action.name == ActionName.ACTION1


def test_decide_model_prompt_includes_ranked_click_candidates():
    frame = _frame([[0, 0], [0, 5]])
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'])
    decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=backend,
        actions_taken=0,
    )
    assert "obj0" in backend.last_prompt


def test_decide_multi_candidate_calls_backend_candidate_count_times():
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'] * 3)
    decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(candidate_count=3),
        backend=backend,
        actions_taken=0,
    )
    assert backend.call_count == 3


def test_decide_multi_candidate_uses_model_source_when_a_candidate_parses():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(candidate_count=2),
        backend=FakeModelBackend(
            responses=['{"action": "ACTION5"}', '{"action": "ACTION1"}']
        ),
        actions_taken=0,
    )
    assert decision.source == "model"
    assert decision.action.name in (ActionName.ACTION1, ActionName.ACTION5)


def test_decide_multi_candidate_falls_back_when_all_candidates_unparseable():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(candidate_count=2),
        backend=FakeModelBackend(responses=["garbage", "also garbage"]),
        actions_taken=0,
    )
    assert decision.source == "fallback_deterministic"


def test_decide_multi_candidate_prefers_higher_scored_non_reset_candidate():
    legal = frozenset({ActionName.RESET, ActionName.ACTION1, ActionName.ACTION5})
    decision, _ = decide(
        frame=_frame([[0, 0], [0, 0]], legal=legal),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(candidate_count=2),
        backend=FakeModelBackend(
            responses=['{"action": "RESET"}', '{"action": "ACTION1"}']
        ),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.ACTION1


def test_decide_default_candidate_count_still_calls_backend_exactly_once():
    """Regression guard: candidate_count's default (1) must take the
    original, untouched single-call path -- every other test in this file
    already exercises Config() with no candidate_count override, so this
    just makes the call-count invariant explicit.
    """
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'])
    decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=backend,
        actions_taken=0,
    )
    assert backend.call_count == 1


def test_decide_populates_raw_response_on_successful_model_action():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert decision.raw_response == '{"action": "ACTION1"}'


def test_decide_populates_raw_response_even_when_parse_fails():
    frame = _frame([[0, 0], [0, 5]])  # one clickable object -> fallback_heuristic
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION0"}']),  # invalid name
        actions_taken=0,
    )
    assert decision.source == "fallback_heuristic"
    assert decision.raw_response == '{"action": "ACTION0"}'


def test_decide_raw_response_is_none_when_no_model_call_happens():
    decision, _ = decide(
        frame=_blank_frame(is_game_over=True),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),
        actions_taken=0,
    )
    assert decision.raw_response is None


class _RaisingBackend:
    """Test double: every generate() call raises, simulating a real
    backend failure (auth, network, ...) rather than a bad-but-present
    response. Distinct from FakeModelBackend(responses=[]), which raises
    RuntimeError("no scripted responses left") -- this lets tests control
    the exact exception type/message surfaced via Decision.model_error.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def generate(self, prompt: str) -> str:
        raise self._exc


def test_decide_populates_model_error_when_backend_raises():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=_RaisingBackend(RuntimeError("boom")),
        actions_taken=0,
    )
    assert decision.raw_response is None
    assert decision.model_error == "RuntimeError: boom"


def test_decide_model_error_populated_on_fallback_heuristic_path_too():
    frame = _frame([[0, 0], [0, 5]])  # one clickable object -> fallback_heuristic
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=_RaisingBackend(ConnectionError("network down")),
        actions_taken=0,
    )
    assert decision.source == "fallback_heuristic"
    assert decision.model_error == "ConnectionError: network down"


def test_decide_model_error_is_none_when_model_call_succeeds():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert decision.model_error is None


def test_decide_model_error_is_none_when_no_model_call_happens():
    decision, _ = decide(
        frame=_blank_frame(is_game_over=True),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),
        actions_taken=0,
    )
    assert decision.model_error is None
