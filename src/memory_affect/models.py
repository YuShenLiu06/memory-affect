"""Frozen public contracts. All time values are UTC Unix seconds supplied by the host."""

from __future__ import annotations

import math
import time
import uuid
from string import Formatter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @field_validator("evidence", check_fields=False)
    @classmethod
    def evidence_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence must not be blank")
        return value


class Emotion(Contract):
    happy: float = 0.0
    angry: float = 0.0
    sad: float = 0.0

    def scaled(self, factor: float) -> Emotion:
        return Emotion(happy=self.happy * factor, angry=self.angry * factor, sad=self.sad * factor)

    def plus(self, other: Emotion) -> Emotion:
        return Emotion(
            happy=self.happy + other.happy, angry=self.angry + other.angry, sad=self.sad + other.sad
        )


class Settings(Contract):
    interval_seconds: float = Field(default=3600, gt=0)
    emotion_half_life_seconds: float = Field(default=21600, gt=0)
    retention_half_life_seconds: float = Field(default=604800, gt=0)
    reinforcement_gain: float = Field(default=0.25, ge=0)
    relevance_base_weight: float = Field(default=0.55, ge=0, le=1)
    retention_weight: float = Field(default=0.25, ge=0, le=1)
    importance_weight: float = Field(default=0.15, ge=0, le=1)
    intensity_weight: float = Field(default=0.05, ge=0, le=1)
    max_abs_pulse: float = Field(default=1000.0, gt=0)
    recent_window_seconds: float = Field(default=2592000, gt=0)
    summary_m: int = Field(default=5, ge=0, le=100)
    recall_n: int = Field(default=5, ge=0, le=100)
    injection_char_budget: int = Field(default=6000, ge=256, le=100000)
    director_enabled: bool = True
    director_thresholds: tuple[float, float] = (30.0, 50.0)
    director_template: str | None = Field(default=None, min_length=1, max_length=4000)
    external_director_policy: Literal["append_once", "ignore"] = "append_once"

    @field_validator("director_thresholds")
    @classmethod
    def validate_director_thresholds(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not 0 < value[0] < value[1] < 100:
            raise ValueError(
                "director thresholds require 0 < low < high < 100; extreme is fixed at 100"
            )
        return value

    @field_validator("director_template")
    @classmethod
    def validate_director_template(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = {
            "happy",
            "angry",
            "sad",
            "happy_level",
            "angry_level",
            "sad_level",
            "summary",
            "guidance",
        }
        if not value.strip():
            raise ValueError("director_template must not be blank")
        try:
            parts = list(Formatter().parse(value))
        except ValueError as exc:
            raise ValueError("director_template has invalid braces") from exc
        for _, name, specification, conversion in parts:
            if name is not None and (name not in allowed or specification or conversion):
                raise ValueError(
                    "director_template permits only plain whitelist fields: "
                    + ", ".join(sorted(allowed))
                )
        return value


class MemoryInput(Contract):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=50000)
    occurred_at: float
    source: str = Field(default="user", min_length=1, max_length=200)
    importance: float = Field(default=0.5, ge=0, le=1)
    intensity: float = Field(default=1.0, ge=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    recorded_at: float | None = None
    evidence: str = Field(default="host-provided statement", min_length=1, max_length=4000)
    is_inference: bool = False
    original_emotion: Emotion = Field(default_factory=Emotion)
    appraisal: Emotion = Field(default_factory=Emotion)
    tags: tuple[str, ...] = Field(default=(), max_length=100)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class MemoryRecord(MemoryInput):
    appraisal_version: int = Field(default=1, ge=1)
    revision: int = Field(default=1, ge=1)
    recall_count: int = Field(default=0, ge=0)
    last_recalled_at: float | None = None
    retention_boost: float = Field(default=1.0, gt=0)


class SearchQuery(Contract):
    text: str = Field(default="", max_length=20000)
    source: str | None = None
    tags: tuple[str, ...] = Field(default=(), max_length=100)
    since: float | None = None
    until: float | None = None
    limit: int = Field(default=5, ge=0, le=100)


class RecallCandidate(Contract):
    memory: MemoryRecord
    relevance: float
    retention: float
    importance_score: float
    intensity_score: float
    total_score: float
    explanation: str


class StateSnapshot(Contract):
    version: int = 0
    emotion: Emotion = Field(default_factory=Emotion)
    evaluated_at: float
    next_due_at: float
    director: str = ""


class RequestSnapshot(Contract):
    request_id: str
    created_at: float
    baseline: StateSnapshot
    periodic_due: bool
    periodic_memories: tuple[MemoryRecord, ...] = ()


class Reappraisal(Contract):
    event_id: str
    appraisal: Emotion
    expected_appraisal_version: int = Field(ge=1)
    evidence: str = Field(min_length=1, max_length=4000)
    effect_mode: Literal["future_only", "apply_delta"] = "future_only"


class MemoryCorrection(Contract):
    memory: MemoryInput
    expected_revision: int = Field(ge=1)
    evidence: str = Field(min_length=1, max_length=4000)


class MemoryDeletion(Contract):
    event_id: str
    expected_revision: int = Field(ge=1)
    evidence: str = Field(min_length=1, max_length=4000)


class AuxiliaryProposal(Contract):
    should_recall: bool = False
    search: SearchQuery | None = None
    new_memories: tuple[MemoryInput, ...] = Field(default=(), max_length=50)
    reappraisals: tuple[Reappraisal, ...] = Field(default=(), max_length=50)
    corrections: tuple[MemoryCorrection, ...] = Field(default=(), max_length=50)
    deletions: tuple[MemoryDeletion, ...] = Field(default=(), max_length=50)
    director: str | None = Field(default=None, max_length=2000)


class CommitResult(Contract):
    command_id: str | None = Field(default=None, min_length=1, max_length=200)
    request_id: str
    status: Literal["committed", "noop", "duplicate"]
    state: StateSnapshot
    recalled_ids: tuple[str, ...] = ()
    director_note: str | None = Field(default=None, max_length=2000)


class Injection(Contract):
    request_id: str
    baseline_version: int
    state_version: int
    content: str
    periodic_content: str = ""
    recalled_content: str = ""
    state_content: str = ""
    periodic_memory_content: str = ""
    periodic_due: bool
    memory_ids: tuple[str, ...] = ()


class VersionConflict(RuntimeError):
    """This request cannot rebase silently; skip its write, retry on a real next request."""


class RequestConflict(ValueError):
    """A stable request ID was reused with a different creation anchor."""


def utc_now() -> float:
    return time.time()


def finite_timestamp(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("timestamp must be finite")
    return value
