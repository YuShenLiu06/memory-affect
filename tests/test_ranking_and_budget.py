"""Relevance first, explained curves, bounded and deduplicated injection."""

from memory_affect.models import SearchQuery


def test_relevance_cannot_be_displaced_by_unrelated_strong_events(library, memory):
    library.store.put_memory(memory("relevant", "杭州龙井茶", importance=0.0, intensity=0.0))
    library.store.put_memory(memory("irrelevant", "巴黎演唱会", importance=1.0, intensity=10000))
    snapshot = library.begin_request("relevant-query", 1010)
    results = library.search(snapshot, SearchQuery(text="龙井", limit=1))
    assert [candidate.memory.event_id for candidate in results] == ["relevant"]
    candidate = results[0]
    assert candidate.relevance > 0
    assert candidate.retention > 0
    assert candidate.total_score > 0
    assert "retention" in candidate.explanation
    assert "relevance" in candidate.explanation


def test_recent_and_high_weight_union_can_include_old_events(tmp_path, settings, memory):
    from memory_affect import MemoryLibrary

    lib = MemoryLibrary(tmp_path / "union.sqlite3", settings=settings, initial_time=1000)
    try:
        lib.store.put_memory(
            memory("old-strong", "古老但重要的记忆", occurred_at=-10**9, importance=1)
        )
        lib.store.put_memory(memory("recent", "近期记忆", occurred_at=1000))
        snapshot = lib.begin_request("first-user-request", 1000)
        assert snapshot.periodic_due
        assert {item.event_id for item in snapshot.periodic_memories} == {"old-strong", "recent"}
    finally:
        lib.close()


def test_shared_character_budget_limits_and_deduplicates(library, tmp_path, settings, memory):
    from memory_affect import MemoryLibrary

    limited = settings.model_copy(
        update={"summary_m": 2, "recall_n": 1, "injection_char_budget": 400}
    )
    lib = MemoryLibrary(tmp_path / "budget.sqlite3", settings=limited, initial_time=1000)
    try:
        for index in range(5):
            lib.store.put_memory(memory(str(index), "长内容测试" * 100))
        snapshot = lib.begin_request("budget", 1000)
        recalled = lib.search(snapshot, SearchQuery(text="测试", limit=100))
        before = lib.store.list_memories()
        injection = lib.compose(snapshot, recalled=(*recalled, *recalled))
        assert len(injection.content) <= limited.injection_char_budget
        assert len(snapshot.periodic_memories) <= limited.summary_m
        assert len(injection.memory_ids) == len(set(injection.memory_ids))
        assert len(injection.memory_ids) <= limited.summary_m + limited.recall_n
        assert lib.store.list_memories() == before
    finally:
        lib.close()


def test_clipped_preview_is_not_recalled_and_only_confirmed_final_ids_reinforce(
    tmp_path, settings, memory,
):
    from memory_affect import MemoryLibrary

    limited = settings.model_copy(update={"injection_char_budget": 400})
    lib = MemoryLibrary(tmp_path / "selection.sqlite3", settings=limited, initial_time=1000)
    try:
        lib.begin_request("initial", 1000)
        lib.store.put_memory(memory("short", "龙井"))
        lib.store.put_memory(memory("long", "龙井" * 500, importance=1))
        snapshot = lib.begin_request("select-after-budget", 1010)
        candidates = lib.search(snapshot, SearchQuery(text="龙井", limit=10))
        preview = lib.compose(snapshot, recalled=candidates)
        assert preview.memory_ids == ("short",)
        assert lib.store.get_memory("long").recall_count == 0
        result = lib.commit(snapshot, selected_ids=preview.memory_ids)
        assert result.recalled_ids == ("short",)
        assert lib.store.get_memory("short").recall_count == 1
        assert lib.store.get_memory("long").recall_count == 0
    finally:
        lib.close()


def test_exact_budget_edges_keep_selected_ids_after_large_new_director(settings, memory):
    """599 real transactions, not a mocked estimate of string lengths."""
    from memory_affect import AuxiliaryProposal, MemoryLibrary

    limited = settings.model_copy(update={"injection_char_budget": 800})
    for length in range(1, 600):
        lib = MemoryLibrary(":memory:", settings=limited, initial_time=1000)
        try:
            lib.begin_request("initial", 1000)
            lib.store.put_memory(memory(content="长" * length))
            snapshot = lib.begin_request("budget-edge", 1010)
            candidates = lib.search(snapshot, SearchQuery(text="长"))
            preview = lib.compose(snapshot, recalled=candidates)
            result = lib.commit(
                snapshot, selected_ids=preview.memory_ids,
                proposal=AuxiliaryProposal(director="导演" * 1000),
            )
            final = lib.compose(snapshot, result=result, recalled=candidates)
            assert preview.memory_ids == final.memory_ids, length
            assert len(preview.content) <= 800
            assert len(final.content) <= 800
            assert lib.store.get_memory("tea").recall_count == int("tea" in final.memory_ids)
        finally:
            lib.close()


def test_minimum_budget_omits_memory_and_does_not_reinforce(tmp_path, settings, memory):
    from memory_affect import MemoryLibrary

    limited = settings.model_copy(update={"injection_char_budget": 256})
    lib = MemoryLibrary(tmp_path / "minimum-budget.sqlite3", settings=limited, initial_time=1000)
    try:
        lib.begin_request("initial", 1000)
        lib.store.put_memory(memory(content="茶"))
        snapshot = lib.begin_request("no-memory-space", 1010)
        preview = lib.compose(snapshot, recalled=("tea",))
        assert preview.memory_ids == ()
        result = lib.commit(snapshot, selected_ids=preview.memory_ids)
        assert result.recalled_ids == ()
        assert lib.store.get_memory("tea").recall_count == 0
    finally:
        lib.close()
