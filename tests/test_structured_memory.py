from zerx.memory import ConfirmedRule, Hypothesis, StructuredMemoryState


def test_structured_memory_state_defaults_are_empty():
    state = StructuredMemoryState()
    assert state.confirmed_rules == []
    assert state.working_hypotheses == []
    assert state.rejected_hypotheses == []
    assert state.open_questions == []
    assert state.current_goal == ""
    assert state.current_plan == []
    assert state.notable_failures == []
    assert state.step_count == 0
    assert state.last_refreshed_step == 0


def test_structured_memory_state_reset_clears_every_field():
    state = StructuredMemoryState(
        confirmed_rules=[ConfirmedRule(statement="lights turn on click", evidence_count=3)],
        working_hypotheses=[Hypothesis(statement="key opens door")],
        rejected_hypotheses=[Hypothesis(statement="red tile is lava", contradicting_evidence=2)],
        open_questions=["what does ACTION3 do"],
        current_goal="reach the exit",
        current_plan=["click door", "move right"],
        notable_failures=["clicked wall 4 times, no effect"],
        step_count=12,
        last_refreshed_step=10,
    )
    state.reset()
    assert state.confirmed_rules == []
    assert state.working_hypotheses == []
    assert state.rejected_hypotheses == []
    assert state.open_questions == []
    assert state.current_goal == ""
    assert state.current_plan == []
    assert state.notable_failures == []
    assert state.step_count == 0
    assert state.last_refreshed_step == 0
