"""Storage assertions against the frozen public facade, not private SQL layout."""

import pytest

from memory_affect.models import Emotion, SearchQuery, VersionConflict


def test_chinese_crud_reindexes_corrections_and_deletions(library, memory):
    library.store.put_memory(memory())
    snapshot = library.begin_request("chinese", 1000)
    def query(text):
        return library.search(snapshot, SearchQuery(text=text, limit=10))

    assert [item.memory.event_id for item in query("龙井茶")] == ["tea"]
    original = library.store.get_memory("tea")
    assert original is not None
    revised = library.store.put_memory(
        memory(content="我在成都吃火锅"), expected_revision=original.revision
    )
    assert revised.revision > original.revision
    assert not query("龙井茶")
    assert [item.memory.event_id for item in query("火锅")] == ["tea"]
    assert library.store.delete_memory("tea", expected_revision=revised.revision)
    assert library.store.get_memory("tea") is None
    assert not query("火锅")
    assert library.store.list_memories() == []


def test_correction_preserves_original_history_fields(library, memory):
    original = library.store.put_memory(
        memory(original_emotion=Emotion(happy=4), appraisal=Emotion(happy=2), intensity=3)
    )
    corrected = library.store.put_memory(
        memory(
            content="修正后的事件内容",
            occurred_at=2000,
            original_emotion=Emotion(angry=999),
            appraisal=Emotion(angry=999),
            intensity=999,
            source="corrected",
            importance=0.8,
            tags=("修正",),
        ),
        expected_revision=original.revision,
    )
    assert corrected.original_emotion == original.original_emotion
    assert corrected.appraisal == original.appraisal
    assert corrected.occurred_at == original.occurred_at
    assert corrected.recall_count == original.recall_count
    assert corrected.source == "corrected"
    assert corrected.importance == 0.8
    assert corrected.tags == ("修正",)


def test_search_is_read_only_and_filters(library, memory):
    library.store.put_memory(memory(source="journal", tags=("旅行",)))
    library.store.put_memory(memory("other", source="message", tags=("工作",)))
    snapshot = library.begin_request("read-only", 1000)
    before = (
        library.current_state(), library.store.list_memories(), library.audit(),
        library.state_history(),
    )
    results = library.search(
        snapshot, SearchQuery(text="龙井", source="journal", tags=("旅行",), since=999, until=1001)
    )
    assert [item.memory.event_id for item in results] == ["tea"]
    assert not library.search(snapshot, SearchQuery(text="不存在的外星词语"))
    assert before == (
        library.current_state(), library.store.list_memories(), library.audit(),
        library.state_history(),
    )
    assert library.store.get_memory("tea").recall_count == 0


def test_sql_and_fts_syntax_inputs_cannot_mutate_database(library, memory):
    library.store.put_memory(memory(event_id="x'; DROP TABLE memories; --"))
    snapshot = library.begin_request("injection", 1000)
    for text in ["' OR 1=1 --", '"', "茶 OR *", "NEAR(茶) - 龙井", ";DROP TABLE memories;"]:
        library.search(snapshot, SearchQuery(text=text))
    assert len(library.store.list_memories()) == 1
    assert library.store.get_memory("x'; DROP TABLE memories; --") is not None


def test_revision_conflicts_leave_record_unchanged(library, memory):
    current = library.store.put_memory(memory())
    with pytest.raises(VersionConflict):
        library.store.put_memory(
            memory(content="错误覆盖"), expected_revision=current.revision + 10
        )
    assert library.store.get_memory("tea") == current
    with pytest.raises(VersionConflict):
        library.store.delete_memory("tea", expected_revision=current.revision + 10)
    assert library.store.get_memory("tea") == current


def test_explicit_transaction_rolls_back_records_and_index(library, memory):
    with pytest.raises(RuntimeError, match="abort"):
        with library.store.transaction() as connection:
            library.store.put_memory(memory(), conn=connection)
            raise RuntimeError("abort")
    assert library.store.list_memories() == []
    snapshot = library.begin_request("rolled-back", 1000)
    assert library.search(snapshot, SearchQuery(text="龙井")) == []


@pytest.mark.parametrize("changes", [{"content": ""}, {"importance": float("nan")}])
def test_forged_model_cannot_persist_invalid_correction(library, memory, changes):
    from pydantic import ValidationError

    original = library.store.put_memory(memory())
    forged = memory().model_copy(update=changes)
    with pytest.raises((ValidationError, ValueError)):
        library.store.put_memory(forged, expected_revision=original.revision)
    assert library.store.get_memory("tea") == original


def test_forged_full_record_cannot_bypass_finite_validation(library, memory):
    from pydantic import ValidationError

    original = library.store.put_memory(memory())
    forged = original.model_copy(update={"retention_boost": float("inf")})
    with pytest.raises((ValidationError, ValueError)):
        with library.store.transaction() as connection:
            library.store.save_record(forged, connection)
    assert library.store.get_memory("tea") == original
