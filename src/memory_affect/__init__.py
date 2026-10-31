"""Host-driven knowledge, recall, and affect. No background work or agent loops."""

from .auxiliary import (
    DeterministicStub,
    ExternalJudge,
    build_auxiliary_prompt,
    parse_auxiliary_response,
)
from .director import render_director
from .engine import MemoryLibrary
from .models import (
    AuxiliaryProposal,
    CommitResult,
    Emotion,
    Injection,
    MemoryCorrection,
    MemoryDeletion,
    MemoryInput,
    MemoryRecord,
    Reappraisal,
    RecallCandidate,
    RequestConflict,
    RequestSnapshot,
    SearchQuery,
    Settings,
    StateSnapshot,
    VersionConflict,
)
from .storage import EmbeddingProvider, SQLiteStore

__all__ = [
    "MemoryLibrary",
    "render_director",
    "SQLiteStore",
    "EmbeddingProvider",
    "Settings",
    "Emotion",
    "MemoryCorrection",
    "MemoryDeletion",
    "MemoryInput",
    "MemoryRecord",
    "SearchQuery",
    "RecallCandidate",
    "StateSnapshot",
    "RequestSnapshot",
    "AuxiliaryProposal",
    "Reappraisal",
    "CommitResult",
    "Injection",
    "VersionConflict",
    "RequestConflict",
    "DeterministicStub",
    "ExternalJudge",
    "build_auxiliary_prompt",
    "parse_auxiliary_response",
]
