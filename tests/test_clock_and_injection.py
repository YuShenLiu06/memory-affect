"""Deadline eligibility is anchored to user creation, never wall-clock execution."""

from memory_affect.models import AuxiliaryProposal, Emotion, SearchQuery


def test_wall_clock_advancing_without_request_does_not_mutate_state(library, monkeypatch):
    import time

    before = (library.current_state(), library.state_history(), library.audit())
    monkeypatch.setattr(time, "time", lambda: 10**12)
    assert before == (library.current_state(), library.state_history(), library.audit())
    snapshot = library.begin_request("host-anchor", 1010)
    assert not snapshot.periodic_due
    assert snapshot.created_at == 1010
    assert snapshot.baseline.evaluated_at == 1000


def test_not_due_request_does_not_gain_periodic_memory_after_deadline(library, memory, monkeypatch):
    import time

    library.store.put_memory(memory())
    snapshot = library.begin_request("before-deadline", 1099)
    assert not snapshot.periodic_due
    first = library.compose(snapshot)
    monkeypatch.setattr(time, "time", lambda: 99999999)
    assert library.begin_request("before-deadline", 1099) == snapshot
    assert library.compose(snapshot) == first
    assert not first.periodic_due
    assert first.memory_ids == ()
    assert library.store.get_memory("tea").recall_count == 0
    # On-demand retrieval remains available, independently of periodic eligibility.
    recalled = library.search(snapshot, SearchQuery(text="龙井"))
    assert [item.memory.event_id for item in recalled] == ["tea"]
    assert library.compose(snapshot, recalled=recalled).memory_ids == ("tea",)


def test_due_is_snapshot_eligibility_and_display_does_not_reinforce(library, memory):
    library.store.put_memory(memory(appraisal=Emotion(happy=8)))
    before = library.current_state()
    snapshot = library.begin_request("at-deadline", before.next_due_at)
    assert snapshot.periodic_due
    assert snapshot.periodic_memories
    injection = library.compose(snapshot)
    assert injection.periodic_due
    assert injection.memory_ids == ("tea",)
    assert library.store.get_memory("tea").recall_count == 0
    assert library.current_state().version == before.version + 1
    assert library.current_state().next_due_at == snapshot.created_at + 100
    assert library.current_state().emotion == before.emotion


def test_successful_update_resets_deadline_from_anchor_not_wall_clock(library, monkeypatch):
    import time

    snapshot = library.begin_request("valid-update", 1010)
    monkeypatch.setattr(time, "time", lambda: 10**12)
    result = library.commit(snapshot, proposal=AuxiliaryProposal(director="有效更新"))
    assert result.state.evaluated_at == 1010
    assert result.state.next_due_at == 1110


def test_unrelated_commit_does_not_reset_age_of_every_event(library, memory):
    library.store.put_memory(memory())
    first = library.begin_request("early", 1000)
    before = library.search(first, SearchQuery(text="龙井"))[0].retention
    library.commit(
        library.begin_request("unrelated", 1020),
        proposal=AuxiliaryProposal(director="并非召回茶事件"),
    )
    after = library.search(library.begin_request("later-read", 1020), SearchQuery(text="龙井"))[0]
    assert after.retention < before
    assert library.store.get_memory("tea").recall_count == 0
