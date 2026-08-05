from zerx.memory import (
    ConfirmedRule,
    Hypothesis,
    StructuredMemoryState,
    confirm_hypothesis,
    contradict_hypothesis,
    record_hypothesis,
)


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


def test_record_hypothesis_adds_new_working_hypothesis():
    state = StructuredMemoryState()
    new_state = record_hypothesis(state, "clicking the blue tile opens the door")
    assert len(new_state.working_hypotheses) == 1
    assert new_state.working_hypotheses[0].statement == "clicking the blue tile opens the door"
    assert new_state.working_hypotheses[0].supporting_evidence == 1
    assert new_state.working_hypotheses[0].contradicting_evidence == 0
    # other lists untouched
    assert new_state.confirmed_rules == []
    assert new_state.rejected_hypotheses == []


def test_record_hypothesis_does_not_mutate_input():
    state = StructuredMemoryState()
    record_hypothesis(state, "clicking the blue tile opens the door")
    assert state.working_hypotheses == []


def test_record_hypothesis_repeated_statement_increments_support_not_duplicate():
    state = StructuredMemoryState()
    state = record_hypothesis(state, "same object")
    state = record_hypothesis(state, "same object")
    assert len(state.working_hypotheses) == 1
    assert state.working_hypotheses[0].supporting_evidence == 2


def test_confirm_hypothesis_moves_matching_working_hypothesis_to_confirmed_rules():
    state = StructuredMemoryState(working_hypotheses=[Hypothesis(statement="key opens door", supporting_evidence=3)])
    new_state = confirm_hypothesis(state, "key opens door")
    assert new_state.working_hypotheses == []
    assert len(new_state.confirmed_rules) == 1
    assert new_state.confirmed_rules[0].statement == "key opens door"
    assert new_state.confirmed_rules[0].evidence_count == 3


def test_confirm_hypothesis_with_no_matching_working_hypothesis_confirms_directly():
    state = StructuredMemoryState()
    new_state = confirm_hypothesis(state, "reset always returns to level 1")
    assert new_state.working_hypotheses == []
    assert len(new_state.confirmed_rules) == 1
    assert new_state.confirmed_rules[0].statement == "reset always returns to level 1"
    assert new_state.confirmed_rules[0].evidence_count == 1


def test_confirm_hypothesis_does_not_mutate_input():
    state = StructuredMemoryState(working_hypotheses=[Hypothesis(statement="key opens door")])
    confirm_hypothesis(state, "key opens door")
    assert len(state.working_hypotheses) == 1
    assert state.confirmed_rules == []


def test_contradict_hypothesis_increments_contradicting_evidence_below_threshold():
    state = StructuredMemoryState(
        working_hypotheses=[Hypothesis(statement="green tile is safe", supporting_evidence=2, contradicting_evidence=0)]
    )
    new_state = contradict_hypothesis(state, "green tile is safe")
    assert len(new_state.working_hypotheses) == 1
    assert new_state.working_hypotheses[0].contradicting_evidence == 1
    assert new_state.rejected_hypotheses == []


def test_contradict_hypothesis_crossing_threshold_is_a_belief_reversal():
    state = StructuredMemoryState(
        working_hypotheses=[Hypothesis(statement="green tile is safe", supporting_evidence=1, contradicting_evidence=0)]
    )
    new_state = contradict_hypothesis(state, "green tile is safe")
    assert new_state.working_hypotheses == []
    assert len(new_state.rejected_hypotheses) == 1
    assert new_state.rejected_hypotheses[0].statement == "green tile is safe"
    assert new_state.rejected_hypotheses[0].contradicting_evidence == 1


def test_contradict_hypothesis_with_no_matching_hypothesis_is_a_no_op():
    state = StructuredMemoryState()
    new_state = contradict_hypothesis(state, "never asserted")
    assert new_state.working_hypotheses == []
    assert new_state.rejected_hypotheses == []


def test_contradict_hypothesis_does_not_mutate_input():
    state = StructuredMemoryState(
        working_hypotheses=[Hypothesis(statement="green tile is safe", supporting_evidence=1)]
    )
    contradict_hypothesis(state, "green tile is safe")
    assert state.working_hypotheses[0].contradicting_evidence == 0


from zerx.memory import (
    add_open_question,
    record_notable_failure,
    set_current_goal,
    set_current_plan,
)


def test_add_open_question_appends_and_dedupes():
    state = StructuredMemoryState()
    state = add_open_question(state, "what does ACTION3 do")
    state = add_open_question(state, "what does ACTION3 do")
    state = add_open_question(state, "is there a timer")
    assert state.open_questions == ["what does ACTION3 do", "is there a timer"]


def test_set_current_goal_replaces_value():
    state = StructuredMemoryState(current_goal="old goal")
    new_state = set_current_goal(state, "reach the exit")
    assert new_state.current_goal == "reach the exit"
    assert state.current_goal == "old goal"  # input not mutated


def test_set_current_plan_replaces_list():
    state = StructuredMemoryState(current_plan=["old step"])
    new_state = set_current_plan(state, ["click door", "move right"])
    assert new_state.current_plan == ["click door", "move right"]
    assert state.current_plan == ["old step"]  # input not mutated


def test_record_notable_failure_appends_without_deduping():
    state = StructuredMemoryState()
    state = record_notable_failure(state, "clicked wall, no effect")
    state = record_notable_failure(state, "clicked wall, no effect")
    assert state.notable_failures == ["clicked wall, no effect", "clicked wall, no effect"]
