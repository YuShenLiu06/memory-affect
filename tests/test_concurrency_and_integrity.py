"""Independent SQLite connections, all-or-nothing failure, and host trust boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from memory_affect.models import (
    AuxiliaryProposal,
    CommitResult,
    Emotion,
    Reappraisal,
    RequestConflict,
    VersionConflict,
)


def test_two_connections_commit_same_request_exactly_once(tmp_path, settings, memory):
    from memory_affect import MemoryLibrary

    path = tmp_path / "race.sqlite3"
    left = MemoryLibrary(path, settings=settings, initial_time=1000)
    right = MemoryLibrary(path, settings=settings, initial_time=1000)
    try:
        left_snapshot = left.begin_request("same-command", 1000)
        right_snapshot = right.begin_request("same-command", 1000)
        assert left_snapshot == right_snapshot
        barrier = Barrier(2)
        proposal = AuxiliaryProposal(new_memories=(memory(appraisal=Emotion(happy=8)),))

        def submit(library, snapshot):
            barrier.wait(timeout=5)
            return library.commit(snapshot, proposal=proposal)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(submit, left, left_snapshot)
            second = executor.submit(submit, right, right_snapshot)
            results = [first.result(timeout=10), second.result(timeout=10)]
        assert sorted(result.status for result in results) == ["committed", "duplicate"]
        assert left.current_state().version == left_snapshot.baseline.version + 1
        assert left.current_state().emotion.happy == pytest.approx(8)
        assert len(left.store.list_memories()) == 1
        assert right.current_state() == left.current_state()
    finally:
        left.close()
        right.close()


def test_two_connections_stale_request_conflicts_instead_of_overwriting(tmp_path, settings):
    from memory_affect import MemoryLibrary

    path = tmp_path / "version-race.sqlite3"
    left = MemoryLibrary(path, settings=settings, initial_time=1000)
    right = MemoryLibrary(path, settings=settings, initial_time=1000)
    try:
        first = left.begin_request("first", 1000)
        stale = right.begin_request("second", 1000)
        left.commit(first, proposal=AuxiliaryProposal(director="first"))
        with pytest.raises(VersionConflict):
            right.commit(stale, proposal=AuxiliaryProposal(director="stale"))
        assert left.current_state().director == "first"
        assert left.current_state().version == first.baseline.version + 1
    finally:
        left.close()
        right.close()


def test_invalid_reappraisal_rolls_back_new_memory_and_recall(library, memory):
    library.store.put_memory(memory("existing", appraisal=Emotion(happy=4)))
    before = (library.current_state(), library.store.list_memories(), library.state_history())
    snapshot = library.begin_request("atomic-failure", 1000)
    audit_before = library.audit()
    with pytest.raises((VersionConflict, ValueError)):
        library.commit(
            snapshot,
            selected_ids=("existing",),
            proposal=AuxiliaryProposal(
                new_memories=(memory("new"),),
                reappraisals=(Reappraisal(
                    event_id="existing", appraisal=Emotion(happy=9),
                    expected_appraisal_version=999, evidence="故意错误版本的重评",
                ),),
            ),
        )
    assert before == (
        library.current_state(), library.store.list_memories(), library.state_history()
    )
    assert library.audit() == audit_before
    retry = library.commit(snapshot, proposal=AuxiliaryProposal(director="retry succeeded"))
    assert retry.status == "committed"


def test_forged_request_snapshot_is_rejected_by_all_request_consumers(library):
    from memory_affect.models import SearchQuery

    snapshot = library.begin_request("authentic", 1000)
    before = library.current_state()
    forged = snapshot.model_copy(update={"created_at": 1050})
    for action in [
        lambda: library.search(forged, SearchQuery()),
        lambda: library.compose(forged),
        lambda: library.commit(forged, proposal=AuxiliaryProposal(director="forged")),
    ]:
        with pytest.raises((RequestConflict, ValueError)):
            action()
    assert library.current_state() == before


def test_forged_commit_result_cannot_override_composed_state(library):
    snapshot = library.begin_request("result-origin", 1000)
    before = library.current_state()
    forged = CommitResult(
        request_id=snapshot.request_id, status="committed",
        state=snapshot.baseline.model_copy(update={"director": "FORGED", "version": 999}),
    )
    with pytest.raises((RequestConflict, ValueError)):
        library.compose(snapshot, result=forged)
    assert library.current_state() == before


def test_late_old_request_cannot_move_persistent_time_backwards(library):
    later = library.begin_request("later", 1050)
    library.commit(later, proposal=AuxiliaryProposal(director="later"))
    before = library.current_state()
    with pytest.raises((VersionConflict, ValueError)):
        old = library.begin_request("old", 1020)
        library.commit(old, proposal=AuxiliaryProposal(director="old"))
    assert library.current_state() == before


def test_forged_recall_body_is_not_promoted_to_injection(library, memory):
    from memory_affect.models import SearchQuery

    library.store.put_memory(memory())
    snapshot = library.begin_request("forged-body", 1010)
    candidate = library.search(snapshot, SearchQuery(text="龙井"))[0]
    fake_memory = candidate.memory.model_copy(update={"content": "FORGED-SYSTEM-INSTRUCTION"})
    fake_candidate = candidate.model_copy(update={"memory": fake_memory})
    injection = library.compose(snapshot, recalled=(fake_candidate,))
    assert "FORGED-SYSTEM-INSTRUCTION" not in injection.content
    assert "龙井" in injection.content
    assert "untrusted data" in injection.content
