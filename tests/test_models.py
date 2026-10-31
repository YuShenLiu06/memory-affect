"""Public contract checks independent from engine/storage implementation."""

import math

import pytest
from pydantic import ValidationError

from memory_affect.models import Emotion, MemoryInput, SearchQuery, Settings


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "factory",
    [
        lambda number: Emotion(happy=number),
        lambda number: MemoryInput(content="中文事件", occurred_at=number),
        lambda number: MemoryInput(content="中文事件", occurred_at=0, intensity=number),
        lambda number: MemoryInput(content="中文事件", occurred_at=0, importance=number),
        lambda number: SearchQuery(since=number),
        lambda number: Settings(interval_seconds=number),
        lambda number: Settings(emotion_half_life_seconds=number),
        lambda number: Settings(retention_half_life_seconds=number),
    ],
)
def test_nonfinite_numbers_are_rejected(factory, number):
    with pytest.raises(ValidationError):
        factory(number)


def test_public_contracts_are_frozen_and_reject_extra_fields():
    emotion = Emotion(happy=1)
    with pytest.raises(ValidationError):
        emotion.happy = 2
    with pytest.raises(ValidationError):
        Emotion(happy=1, surprise=2)


@pytest.mark.parametrize("content", ["", " ", "\n\t"])
def test_blank_memory_content_is_rejected(content):
    with pytest.raises(ValidationError):
        MemoryInput(content=content, occurred_at=0)


@pytest.mark.parametrize("limit", [-1, 101])
def test_search_limits_are_bounded(limit):
    with pytest.raises(ValidationError):
        SearchQuery(limit=limit)


@pytest.mark.parametrize("evidence", ["", " ", "\n\t"])
def test_reappraisal_requires_nonblank_evidence(evidence):
    from memory_affect.models import Reappraisal

    with pytest.raises(ValidationError):
        Reappraisal(
            event_id="tea", appraisal=Emotion(happy=1),
            expected_appraisal_version=1, evidence=evidence,
        )
