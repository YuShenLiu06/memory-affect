"""Host-authorized metadata commands and deletion/correction cache safety."""

import json

import pytest

from memory_affect.models import AuxiliaryProposal, Emotion, SearchQuery


def test_management_crud_is_versioned_idempotent_and_does_not_reset_affect_clock(library, memory):
    before = library.current_state()
    result = library.manage_memory("manage-create", memory=memory(), evidence="用户授权录入")
    assert isinstance(result, dict)
    created = library.store.get_memory("tea")
    assert created.evidence == "用户授权录入"
    after = library.current_state()
    assert after.version == before.version + 1
    assert after.emotion == before.emotion
    assert after.evaluated_at == before.evaluated_at
    assert after.next_due_at == before.next_due_at
    library.manage_memory("manage-create", memory=memory(), evidence="用户授权录入")
    assert library.current_state() == after
    assert library.store.get_memory("tea") == created
    library.manage_memory(
        "manage-correct", memory=memory(content="我在成都吃火锅"),
        expected_revision=created.revision, evidence="用户提供更正依据",
    )
    corrected = library.store.get_memory("tea")
    assert corrected.content == "我在成都吃火锅"
    assert corrected.evidence == "用户提供更正依据"
    snapshot = library.begin_request("after-correction", 1010)
    assert not library.search(snapshot, SearchQuery(text="龙井"))
    assert library.search(snapshot, SearchQuery(text="火锅"))
    library.manage_memory(
        "manage-delete", delete_id="tea", expected_revision=corrected.revision,
        evidence="用户授权删除",
    )
    assert library.store.get_memory("tea") is None
    assert not library.search(snapshot, SearchQuery(text="火锅"))
    assert library.current_state().next_due_at == before.next_due_at


def test_old_periodic_snapshot_does_not_reintroduce_deleted_or_corrected_text(library, memory):
    library.manage_memory("add", memory=memory(), evidence="录入原事实")
    snapshot = library.begin_request("due-snapshot", 1100)
    assert "龙井" in library.compose(snapshot).content
    record = library.store.get_memory("tea")
    library.manage_memory(
        "correct", memory=memory(content="我在成都吃火锅"),
        expected_revision=record.revision, evidence="更正原事实",
    )
    corrected_injection = library.compose(snapshot)
    assert "龙井" not in corrected_injection.content
    record = library.store.get_memory("tea")
    library.manage_memory(
        "delete", delete_id="tea", expected_revision=record.revision,
        evidence="删除原事实",
    )
    deleted_injection = library.compose(snapshot)
    assert "龙井" not in deleted_injection.content
    assert "火锅" not in deleted_injection.content
    assert "tea" not in deleted_injection.memory_ids
    assert "龙井" not in json.dumps(library.audit(), ensure_ascii=False)
    assert "火锅" not in json.dumps(library.audit(), ensure_ascii=False)


def test_metadata_correction_and_deletion_do_not_rewrite_historical_emotion(library, memory):
    snapshot = library.begin_request("emotional-event", 1000)
    library.commit(snapshot, proposal=AuxiliaryProposal(
        new_memories=(memory(appraisal=Emotion(sad=20)),),
    ))
    before = library.current_state()
    record = library.store.get_memory("tea")
    library.manage_memory(
        "metadata-correction", memory=memory(content="更正事实"),
        expected_revision=record.revision, evidence="用户事实修正",
    )
    record = library.store.get_memory("tea")
    assert library.current_state().emotion == before.emotion
    library.manage_memory(
        "metadata-deletion", delete_id="tea", expected_revision=record.revision,
        evidence="用户删除",
    )
    assert library.current_state().emotion == before.emotion
    assert library.state_history()[-3].emotion == before.emotion


def test_manage_conflict_rolls_back_without_losing_prior_state(library, memory):
    library.manage_memory("create", memory=memory(), evidence="用户录入")
    before = (library.current_state(), library.store.get_memory("tea"), library.audit())
    with pytest.raises((ValueError, RuntimeError)):
        library.manage_memory(
            "bad-correction", memory=memory(content="错误覆盖"),
            expected_revision=999, evidence="错误版本",
        )
    assert before == (library.current_state(), library.store.get_memory("tea"), library.audit())


def test_same_management_id_with_different_payload_cannot_overwrite(library, memory):
    from memory_affect.models import RequestConflict

    library.manage_memory("stable-command", memory=memory(), evidence="用户录入")
    before = (library.current_state(), library.store.get_memory("tea"))
    with pytest.raises(RequestConflict):
        library.manage_memory(
            "stable-command", memory=memory(content="不同参数不得覆盖"), evidence="用户录入"
        )
    assert before == (library.current_state(), library.store.get_memory("tea"))


def test_auxiliary_correction_persists_outer_evidence(library, memory):
    from memory_affect.models import MemoryCorrection

    library.manage_memory("evidence-create", memory=memory(), evidence="原始用户证据")
    original = library.store.get_memory("tea")
    library.commit(
        library.begin_request("evidence-correct", 1010),
        proposal=AuxiliaryProposal(corrections=(MemoryCorrection(
            memory=memory(content="用户更正的新事实"),
            expected_revision=original.revision,
            evidence="本次更正的用户证据",
        ),)),
    )
    assert library.store.get_memory("tea").evidence == "本次更正的用户证据"


def test_delete_removes_persisted_periodic_body_and_search_derivatives(library, memory):
    secret = "独特隐私事实：青铜钥匙藏在杏花树下"
    library.manage_memory("private-create", memory=memory(content=secret), evidence="用户录入")
    snapshot = library.begin_request("private-periodic", 1100)
    assert secret in library.compose(snapshot).content
    record = library.store.get_memory("tea")
    library.manage_memory(
        "private-delete", delete_id="tea", expected_revision=record.revision,
        evidence="用户授权删除",
    )
    connection = library.store.connection
    for statement in [
        "SELECT payload FROM memories",
        "SELECT snapshot FROM engine_requests",
        "SELECT payload FROM audit_log",
        "SELECT evidence FROM appraisal_history",
    ]:
        assert all(secret not in row[0] for row in connection.execute(statement))
    assert connection.execute(
        "SELECT count(*) FROM memory_fts WHERE event_id='tea'"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT count(*) FROM appraisal_history WHERE event_id='tea'"
    ).fetchone()[0] == 0
