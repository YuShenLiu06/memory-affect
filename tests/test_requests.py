"""Stable request identity, read-only composition, and explicit commit boundaries."""

import pytest

from memory_affect.models import (
    AuxiliaryProposal,
    Emotion,
    RequestConflict,
    SearchQuery,
    VersionConflict,
)


def test_same_request_id_has_persistent_anchor_and_snapshot(library, tmp_path, settings, memory):
    from memory_affect import MemoryLibrary

    library.store.put_memory(memory())
    first = library.begin_request("stable", 1010)
    assert library.begin_request("stable", 1010) == first
    with pytest.raises(RequestConflict):
        library.begin_request("stable", 1011)
    second = MemoryLibrary(tmp_path / "memory.sqlite3", settings=settings, initial_time=9999)
    try:
        assert second.begin_request("stable", 1010) == first
        with pytest.raises(RequestConflict):
            second.begin_request("stable", 1011)
    finally:
        second.close()


def test_begin_search_compose_do_not_commit_affect_or_recall(library, memory):
    library.store.put_memory(memory(appraisal=Emotion(happy=8)))
    original_state = library.current_state()
    original_history = library.state_history()
    original_memory = library.store.get_memory("tea")
    snapshot = library.begin_request("read-only-boundaries", 1010)
    recalled = library.search(snapshot, SearchQuery(text="龙井"))
    first = library.compose(snapshot, recalled=recalled)
    second = library.compose(snapshot, recalled=recalled)
    assert first == second
    assert library.current_state() == original_state
    assert library.state_history() == original_history
    assert library.store.get_memory("tea") == original_memory


def test_empty_and_unknown_selection_do_not_reset_deadline(library):
    before = library.current_state()
    snapshot = library.begin_request("noop", 1010)
    result = library.commit(snapshot, selected_ids=("missing",), proposal=AuxiliaryProposal())
    assert result.status == "noop"
    assert library.current_state() == before
    assert library.current_state().next_due_at == before.next_due_at
    assert library.commit(snapshot).status == "duplicate"
    assert library.current_state() == before


def test_commit_is_idempotent_and_original_compose_stays_baseline(library, memory):
    snapshot = library.begin_request("commit-once", 1000)
    before_injection = library.compose(snapshot)
    result = library.commit(
        snapshot,
        proposal=AuxiliaryProposal(new_memories=(memory(appraisal=Emotion(happy=8)),)),
    )
    assert result.status == "committed"
    state = library.current_state()
    record = library.store.get_memory("tea")
    history = library.state_history()
    assert library.compose(snapshot) == before_injection
    after_injection = library.compose(snapshot, result=result)
    assert after_injection.state_version == state.version
    assert after_injection.state_version == snapshot.baseline.version + 1
    duplicate = library.commit(snapshot, selected_ids=("tea",))
    assert duplicate.status == "duplicate"
    assert duplicate.state == result.state
    assert library.current_state() == state
    assert library.state_history() == history
    assert library.store.get_memory("tea") == record


def test_expected_version_conflict_is_atomic(library, memory):
    stale = library.begin_request("stale", 1000)
    winner = library.begin_request("winner", 1000)
    library.commit(winner, proposal=AuxiliaryProposal(director="已提交"))
    before = (library.current_state(), library.store.list_memories(), library.audit())
    with pytest.raises(VersionConflict):
        library.commit(
            stale,
            expected_version=stale.baseline.version,
            proposal=AuxiliaryProposal(new_memories=(memory(),)),
        )
    assert before == (library.current_state(), library.store.list_memories(), library.audit())
    assert library.begin_request("stale", 1000) == stale


def test_commit_compose_keeps_memory_despite_recall_revision(library, memory):
    library.store.put_memory(memory())
    snapshot = library.begin_request("full-request-flow", 1010)
    candidates = library.search(snapshot, SearchQuery(text="龙井"))
    result = library.commit(snapshot, selected_ids=("tea",))
    injection = library.compose(snapshot, result=result, recalled=candidates)
    assert injection.memory_ids == ("tea",)
    assert "龙井" in injection.content
