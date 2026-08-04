from zerx.memory import MemoryState, maybe_refresh


def test_memory_state_reset_clears_all_fields():
    state = MemoryState(summary="learned stuff", step_count=5, last_refreshed_step=3)
    state.reset()
    assert state.summary == ""
    assert state.step_count == 0
    assert state.last_refreshed_step == 0


def test_maybe_refresh_not_due_keeps_summary_and_skips_summarizer():
    state = MemoryState(summary="old", step_count=0, last_refreshed_step=0)

    def boom(prev, ctx):
        raise AssertionError("summarizer should not be called")

    new_state = maybe_refresh(state, "context", boom, refresh_interval=10)
    assert new_state.summary == "old"
    assert new_state.step_count == 1
    assert new_state.last_refreshed_step == 0


def test_maybe_refresh_due_calls_summarizer_and_updates():
    state = MemoryState(summary="old", step_count=8, last_refreshed_step=0)
    new_state = maybe_refresh(
        state, "context", lambda prev, ctx: f"{prev}+{ctx}", refresh_interval=9
    )
    assert new_state.step_count == 9
    assert new_state.last_refreshed_step == 9
    assert new_state.summary == "old+context"


def test_maybe_refresh_does_not_mutate_input():
    state = MemoryState(summary="old", step_count=0, last_refreshed_step=0)
    maybe_refresh(state, "context", lambda prev, ctx: "new", refresh_interval=1)
    assert state.summary == "old"
    assert state.step_count == 0
