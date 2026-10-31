"""Deterministic host clocks and temporary, isolated databases; no real model calls."""

import pytest

from memory_affect.models import MemoryInput, Settings


@pytest.fixture
def settings():
    return Settings(
        interval_seconds=100,
        emotion_half_life_seconds=10,
        retention_half_life_seconds=20,
        summary_m=3,
        recall_n=3,
    )


@pytest.fixture
def library(tmp_path, settings):
    from memory_affect import MemoryLibrary

    lib = MemoryLibrary(tmp_path / "memory.sqlite3", settings=settings, initial_time=1000)
    # Consume the explicitly due initialization request; regular tests start not-due.
    lib.begin_request("fixture-initial-user-request", 1000)
    yield lib
    lib.close()


@pytest.fixture
def memory():
    def make(event_id="tea", content="我在杭州喝龙井茶", **overrides):
        values = dict(event_id=event_id, content=content, occurred_at=1000)
        values.update(overrides)
        return MemoryInput(**values)

    return make
