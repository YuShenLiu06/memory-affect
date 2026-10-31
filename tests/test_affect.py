"""Engineering half-lives and event accounting, with no wall-clock sleeps."""

import math

import pytest

from memory_affect.models import AuxiliaryProposal, Emotion, Reappraisal, SearchQuery


def add(library, memory, *, request_id="add", at=1000, happy=12):
    snapshot = library.begin_request(request_id, at)
    return library.commit(
        snapshot,
        proposal=AuxiliaryProposal(new_memories=(memory(appraisal=Emotion(happy=happy)),)),
    )


def test_affect_above_100_is_not_clipped_and_persists(library, memory, tmp_path, settings):
    from memory_affect import MemoryLibrary

    result = add(library, memory, happy=240)
    assert result.state.emotion.happy == pytest.approx(240)
    reopened = MemoryLibrary(tmp_path / "memory.sqlite3", settings=settings, initial_time=9000)
    try:
        assert reopened.current_state() == result.state
        assert reopened.store.get_memory("tea") == library.store.get_memory("tea")
        assert reopened.state_history() == library.state_history()
    finally:
        reopened.close()


def test_actual_elapsed_and_independent_emotion_retention_half_lives(library, memory):
    add(library, memory, happy=16)
    first = library.begin_request("at-start", 1000)
    second = library.begin_request("ten-seconds", 1010)
    last = library.begin_request("twenty-seconds", 1020)
    assert first.baseline.emotion.happy == pytest.approx(16)
    # Not-due requests retain the persistent baseline; commit settles real elapsed.
    assert second.baseline.emotion.happy == pytest.approx(16)
    assert last.baseline.emotion.happy == pytest.approx(16)
    early_retention = library.search(first, SearchQuery(text="龙井"))[0].retention
    middle_retention = library.search(second, SearchQuery(text="龙井"))[0].retention
    late_retention = library.search(last, SearchQuery(text="龙井"))[0].retention
    assert middle_retention / early_retention == pytest.approx(math.sqrt(0.5))
    assert late_retention / early_retention == pytest.approx(0.5)
    assert library.current_state().emotion.happy == pytest.approx(16)
    committed = library.commit(last, proposal=AuxiliaryProposal(director="按实际时间结算"))
    assert committed.state.emotion.happy == pytest.approx(4)
    assert committed.state.evaluated_at == 1020


def test_confirmed_recall_starts_at_one_and_uses_one_over_k(library, memory):
    add(library, memory, happy=12)
    original = library.store.get_memory("tea")
    assert original.recall_count == 0
    for k, expected_happy in [(1, 24), (2, 30), (3, 34)]:
        snapshot = library.begin_request(f"recall-{k}", 1000)
        library.search(snapshot, SearchQuery(text="龙井"))
        assert library.store.get_memory("tea").recall_count == k - 1
        result = library.commit(snapshot, selected_ids=("tea", "tea"))
        current = library.store.get_memory("tea")
        assert result.recalled_ids == ("tea",)
        assert current.recall_count == k
        assert current.retention_boost == pytest.approx(original.retention_boost + k * 0.25)
        assert result.state.emotion.happy == pytest.approx(expected_happy)
        assert library.commit(snapshot, selected_ids=("tea",)).status == "duplicate"
        assert library.store.get_memory("tea") == current


def test_reappraisal_is_difference_and_does_not_reset_recall_count(library, memory):
    add(library, memory, happy=12)
    for request in ["recall-a", "recall-b"]:
        library.commit(library.begin_request(request, 1000), selected_ids=("tea",))
    before = library.store.get_memory("tea")
    assert before.recall_count == 2
    snapshot = library.begin_request("reappraise", 1000)
    result = library.commit(
        snapshot,
        proposal=AuxiliaryProposal(
            reappraisals=(Reappraisal(
                event_id="tea", appraisal=Emotion(happy=4),
                expected_appraisal_version=before.appraisal_version,
                evidence="用户澄清事件", effect_mode="apply_delta",
            ),),
        ),
    )
    current = library.store.get_memory("tea")
    assert current.recall_count == 2
    assert current.original_emotion == before.original_emotion
    assert current.appraisal.happy == 4
    assert current.appraisal_version == before.appraisal_version + 1
    assert result.state.emotion.happy == pytest.approx(30 + (4 - 12) / 2)
    assert library.state_history()[-2].emotion.happy == pytest.approx(30)


def test_same_round_new_memory_and_confirmed_recall_do_not_double_emotion(library, memory):
    snapshot = library.begin_request("new-and-recall", 1000)
    result = library.commit(
        snapshot,
        selected_ids=("tea", "tea"),
        proposal=AuxiliaryProposal(new_memories=(memory(appraisal=Emotion(happy=12)),)),
    )
    assert result.state.emotion.happy == pytest.approx(12)
    assert library.store.get_memory("tea").recall_count == 1


def test_same_round_reappraisal_and_recall_prioritize_reappraisal_once(library, memory):
    add(library, memory, happy=4)
    record = library.store.get_memory("tea")
    result = library.commit(
        library.begin_request("reappraise-and-recall", 1000),
        selected_ids=("tea",),
        proposal=AuxiliaryProposal(reappraisals=(Reappraisal(
            event_id="tea", appraisal=Emotion(happy=8),
            expected_appraisal_version=record.appraisal_version,
            evidence="用户澄清事件", effect_mode="apply_delta",
        ),)),
    )
    assert library.store.get_memory("tea").recall_count == 1
    assert result.state.emotion.happy == pytest.approx(8)


def test_future_only_reappraisal_changes_future_recall_without_immediate_pulse(library, memory):
    add(library, memory, happy=12)
    record = library.store.get_memory("tea")
    result = library.commit(
        library.begin_request("future-only", 1000),
        proposal=AuxiliaryProposal(reappraisals=(Reappraisal(
            event_id="tea", appraisal=Emotion(happy=4),
            expected_appraisal_version=record.appraisal_version,
            evidence="用户澄清，仅影响未来召回",
        ),)),
    )
    assert result.state.emotion.happy == pytest.approx(12)
    assert library.store.get_memory("tea").recall_count == 0
    recalled = library.commit(library.begin_request("future-recall", 1000), selected_ids=("tea",))
    assert recalled.state.emotion.happy == pytest.approx(16)


def test_future_only_reappraisal_and_recall_apply_new_appraisal_once(library, memory):
    add(library, memory, happy=12)
    record = library.store.get_memory("tea")
    result = library.commit(
        library.begin_request("future-and-recall", 1000), selected_ids=("tea",),
        proposal=AuxiliaryProposal(reappraisals=(Reappraisal(
            event_id="tea", appraisal=Emotion(happy=4),
            expected_appraisal_version=record.appraisal_version,
            evidence="澄清后本轮真实召回",
        ),)),
    )
    assert library.store.get_memory("tea").recall_count == 1
    assert result.state.emotion.happy == pytest.approx(16)


def test_identical_existing_new_event_is_noop_not_second_pulse(library, memory):
    add(library, memory, happy=12)
    before = library.current_state()
    result = library.commit(
        library.begin_request("same-canonical-event", 1010),
        proposal=AuxiliaryProposal(new_memories=(memory(appraisal=Emotion(happy=12)),)),
    )
    assert result.status == "noop"
    assert library.current_state() == before
    assert library.store.get_memory("tea").recall_count == 0


@pytest.mark.parametrize("changes", [
    {"content": "不同内容"}, {"occurred_at": 999}, {"intensity": 2},
    {"original_emotion": Emotion(sad=99)}, {"appraisal": Emotion(happy=99)},
    {"source": "model"}, {"confidence": 0.5}, {"evidence": "不同依据"},
    {"recorded_at": 999}, {"is_inference": True}, {"tags": ("不同标签",)},
])
def test_existing_event_different_payload_conflicts(library, memory, changes):
    add(library, memory, happy=12)
    snapshot = library.begin_request("different-payload", 1010)
    before = (library.current_state(), library.store.get_memory("tea"), library.audit())
    with pytest.raises(ValueError):
        library.commit(snapshot, proposal=AuxiliaryProposal(
            new_memories=(memory(appraisal=Emotion(happy=12)).model_copy(update=changes),),
        ))
    assert before == (library.current_state(), library.store.get_memory("tea"), library.audit())


def test_configured_pulse_protection_is_transparent_not_total_state_clipping(library, memory):
    result = add(library, memory, happy=2000)
    assert result.state.emotion.happy == pytest.approx(1000)
    impact = [row["payload"] for row in library.audit() if row["kind"] == "emotion_impact"][-1]
    assert impact["max_abs_pulse"] == 1000
    assert impact["delta"]["happy"] == 1000
    result = library.commit(library.begin_request("pulse-again", 1000), selected_ids=("tea",))
    assert result.state.emotion.happy == pytest.approx(2000)
