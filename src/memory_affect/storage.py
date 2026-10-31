"""Transactional SQLite knowledge storage and side-effect-free retrieval.

The host owns time. Audit ``at`` values for direct knowledge operations are the
record's event time, not a wall-clock mutation timestamp. Engine operations can
write their request anchor to the same audit table in their enclosing transaction.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from .models import (
    MemoryInput,
    MemoryRecord,
    RecallCandidate,
    SearchQuery,
    Settings,
    VersionConflict,
    finite_timestamp,
)


class EmbeddingProvider(Protocol):
    """Optional host adapter; the library ships no network provider or API key."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Keep Latin words and add Han unigrams/bigrams for SQLite's tokenizer."""
    normalized = text.casefold()
    tokens = _WORD.findall(_HAN.sub(" ", normalized))
    for match in _HAN.finditer(normalized):
        run = match.group()
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _query_tokens(text: str) -> list[str]:
    # For multi-character Chinese queries require bigram evidence rather than
    # considering every document with a common single character relevant.
    normalized = text.casefold()
    tokens = _WORD.findall(_HAN.sub(" ", normalized))
    for match in _HAN.finditer(normalized):
        run = match.group()
        tokens.extend([run] if len(run) == 1 else (run[i : i + 2] for i in range(len(run) - 1)))
    # SQLite FTS expression depth is finite; bound lexical expansion even when
    # a host supplies a very long query. Semantic adapters still see full text.
    return list(dict.fromkeys(tokens))[:64]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have one consistent nonzero dimension")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("embedding vectors must contain finite numbers")
    scale_left = max(abs(value) for value in left)
    scale_right = max(abs(value) for value in right)
    if not scale_left or not scale_right:
        return 0.0
    # Scale first: even hypot of several finite 1e308 values may be infinite.
    normalized_left = [value / scale_left for value in left]
    normalized_right = [value / scale_right for value in right]
    norm_left = math.hypot(*normalized_left)
    norm_right = math.hypot(*normalized_right)
    return max(
        0.0,
        min(
            1.0,
            sum(
                (a / norm_left) * (b / norm_right)
                for a, b in zip(normalized_left, normalized_right)
            ),
        ),
    )


def _retention(record: MemoryRecord, *, at: float, settings: Settings) -> float:
    age = max(0.0, at - record.occurred_at)
    # Confirmation raises the strength (and therefore effective half-life), but
    # never sets the event age to zero. last_recalled_at remains audit evidence.
    strength = max(record.retention_boost, 1e-12)
    return math.exp(-math.log(2.0) * age / settings.retention_half_life_seconds / strength)


class SQLiteStore:
    """Low-level persistence, not a substitute for the engine commit API.

    Hosts should use engine management commands for knowledge edits so request
    anchors, evidence and state history remain unified. All storage writes,
    including FTS and audit, are atomic.

    ``connection`` is exposed for the engine's unified commit. Callers sharing
    that connection must use ``transaction()`` or ``lock``. Separate instances
    and processes are serialized by SQLite's BEGIN IMMEDIATE and busy timeout.
    Optional embeddings are calculated from current eligible contents at search
    time; no vectors or credentials are persisted or silently reused.
    """

    def __init__(self, path: str | Path, *, embedding: EmbeddingProvider | None = None) -> None:
        self.lock = threading.RLock()
        self.embedding = embedding
        self.connection = sqlite3.connect(
            str(path), timeout=30.0, isolation_level=None, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        try:
            with self.transaction() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memories "
                    "(event_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
                    "event_id UNINDEXED, terms, tokenize='unicode61 remove_diacritics 2')"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS audit_log ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, "
                    "kind TEXT NOT NULL, payload TEXT NOT NULL)"
                )
                # Only repair missing legacy index entries on open, never search.
                missing = conn.execute(
                    "SELECT m.payload FROM memories m LEFT JOIN memory_fts f "
                    "ON m.event_id=f.event_id WHERE f.event_id IS NULL"
                ).fetchall()
                for row in missing:
                    self._index(MemoryRecord.model_validate_json(row[0]), conn)
        except BaseException:
            self.connection.close()
            raise

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield an atomic write scope; nested scopes use rollback-safe savepoints."""
        with self.lock:
            conn = self.connection
            nested = conn.in_transaction
            savepoint = "sp_" + uuid.uuid4().hex
            conn.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                if nested:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                else:
                    conn.rollback()
                raise
            else:
                try:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}") if nested else conn.commit()
                except BaseException:
                    if not nested:
                        conn.rollback()
                    raise

    @contextmanager
    def _write_connection(self, conn: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
        if conn is not None and conn is not self.connection:
            raise ValueError("conn must be this store's connection")
        # An explicit engine transaction is not committed here. Standalone calls
        # always get a transaction, including callers passing an idle connection.
        with self.lock:
            if conn is not None and conn.in_transaction:
                yield conn
            else:
                with self.transaction() as active:
                    yield active

    @contextmanager
    def _read_transaction(self, conn: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
        """Keep multi-statement retrieval consistent across other store instances."""
        with self.lock:
            active = self._read_connection(conn)
            owns_transaction = not active.in_transaction
            if owns_transaction:
                active.execute("BEGIN")
            try:
                yield active
            except BaseException:
                if owns_transaction:
                    active.rollback()
                raise
            else:
                if owns_transaction:
                    active.commit()

    def _read_connection(self, conn: sqlite3.Connection | None) -> sqlite3.Connection:
        if conn is not None and conn is not self.connection:
            raise ValueError("conn must be this store's connection")
        return self.connection if conn is None else conn

    @staticmethod
    def _audit(conn: sqlite3.Connection, record: MemoryRecord, kind: str) -> None:
        conn.execute(
            "INSERT INTO audit_log(at,kind,payload) VALUES(?,?,?)",
            (
                record.occurred_at,
                kind,
                json.dumps(
                    {
                        "event_id": record.event_id,
                        "revision": record.revision,
                        "time_basis": "event_occurred_at",
                    },
                    ensure_ascii=False,
                ),
            ),
        )

    @staticmethod
    def _index(record: MemoryRecord, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM memory_fts WHERE event_id=?", (record.event_id,))
        searchable = " ".join((record.content, record.source, *record.tags))
        conn.execute(
            "INSERT INTO memory_fts(event_id,terms) VALUES(?,?)",
            (record.event_id, " ".join(_tokens(searchable))),
        )

    @staticmethod
    def _purge_derived_body(
        event_id: str, conn: sqlite3.Connection, *, deleting: bool = False, old_content: str = ""
    ) -> None:
        """Privacy exception to snapshots: erase invalid derived event bodies.

        The engine owns these optional tables. Baseline versions and request
        anchors remain untouched; live caller-held objects cannot be revoked.
        """
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('engine_requests','appraisal_history','request_journal',"
                "'engine_request_commands','engine_state')"
            )
        }
        # Privacy-only text redaction is explicitly audited; state version,
        # elapsed-time anchor and numeric emotion history are never recomputed.
        if old_content:
            for table, id_column, payload_column, director_path in (
                ("engine_state", "version", "payload", ("director",)),
                ("engine_requests", "request_id", "snapshot", ("baseline", "director")),
            ):
                if table not in tables:
                    continue
                for row in conn.execute(
                    f"SELECT {id_column},{payload_column} FROM {table}"
                ).fetchall():
                    payload = json.loads(row[1])
                    state = payload if len(director_path) == 1 else payload.get("baseline", {})
                    director = state.get("director", "")
                    if not isinstance(director, str) or old_content not in director:
                        continue
                    state["director"] = director.replace(old_content, "[privacy-redacted]")
                    conn.execute(
                        f"UPDATE {table} SET {payload_column}=? WHERE {id_column}=?",
                        (json.dumps(payload, ensure_ascii=False), row[0]),
                    )
                    metadata = {
                        "table": table,
                        id_column: row[0],
                        "event_id": event_id,
                        "reason": "memory_deleted" if deleting else "memory_corrected",
                        "field": ".".join(director_path),
                        "state_version": state.get("version"),
                    }
                    conn.execute(
                        "INSERT INTO audit_log(at,kind,payload) VALUES(?,?,?)",
                        (
                            state.get("evaluated_at", 0),
                            "privacy_redaction",
                            json.dumps(metadata, ensure_ascii=False),
                        ),
                    )
        if "engine_requests" in tables:
            rows = conn.execute(
                "SELECT request_id,snapshot FROM engine_requests WHERE EXISTS("
                "SELECT 1 FROM json_each(snapshot,'$.periodic_memories') item "
                "WHERE json_extract(item.value,'$.event_id')=?)",
                (event_id,),
            ).fetchall()
            for row in rows:
                snapshot = json.loads(row[1])
                snapshot["periodic_memories"] = [
                    item
                    for item in snapshot.get("periodic_memories", [])
                    if item.get("event_id") != event_id
                ]
                conn.execute(
                    "UPDATE engine_requests SET snapshot=? WHERE request_id=?",
                    (json.dumps(snapshot, ensure_ascii=False, allow_nan=False), row[0]),
                )
        # Keep valid CommitResult contracts and command fingerprints for retries.
        for table, keys in (
            ("engine_request_commands", ("request_id", "command_id")),
            ("engine_requests", ("request_id",)),
        ):
            if table not in tables:
                continue
            columns = ",".join(keys)
            for row in conn.execute(
                f"SELECT {columns},result FROM {table} WHERE result IS NOT NULL"
            ).fetchall():
                result = json.loads(row[len(keys)])
                serialized = json.dumps(result, ensure_ascii=False)
                if event_id not in serialized and (
                    not old_content or old_content not in serialized
                ):
                    continue
                state = result.get("state", {})
                state["director"] = ""
                result["director_note"] = None
                where = " AND ".join(f"{key}=?" for key in keys)
                conn.execute(
                    f"UPDATE {table} SET result=? WHERE {where}",
                    (json.dumps(result, ensure_ascii=False), *row[: len(keys)]),
                )
                metadata = {
                    "table": table,
                    "event_id": event_id,
                    "reason": "memory_deleted" if deleting else "memory_corrected",
                    "state_version": state.get("version"),
                    **dict(zip(keys, row[: len(keys)])),
                }
                conn.execute(
                    "INSERT INTO audit_log(at,kind,payload) VALUES(?,?,?)",
                    (
                        state.get("evaluated_at", 0),
                        "privacy_redaction",
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )
        if "request_journal" in tables:
            # This privacy exception is explicit, not a silent rewrite of history.
            # Match canonical IDs and old body, including escaped raw JSON responses.
            needles = [event_id]
            if old_content:
                needles.extend(
                    [
                        old_content,
                        json.dumps(old_content)[1:-1],
                        json.dumps(old_content, ensure_ascii=False)[1:-1],
                    ]
                )
            for row in conn.execute("SELECT request_id,payload FROM request_journal").fetchall():
                payload = json.loads(row[1])
                serialized = json.dumps(payload, ensure_ascii=False)

                def references(value: object) -> bool:
                    if isinstance(value, str):
                        return any(needle in value for needle in needles)
                    if isinstance(value, dict):
                        return any(references(item) for item in value.values())
                    if isinstance(value, list):
                        return any(references(item) for item in value)
                    return False

                if not references(payload) and event_id not in serialized:
                    continue
                versions = {}
                for field in ("snapshot", "commit_receipt", "final_injection"):
                    item = payload.get(field)
                    if isinstance(item, dict):
                        versions[field] = {
                            key: value
                            for key, value in item.items()
                            if key in {"baseline_version", "state_version", "status"}
                        }
                        state = item.get("baseline", item.get("state"))
                        if isinstance(state, dict):
                            versions[field]["version"] = state.get("version")
                clean = {
                    key: payload[key]
                    for key in ("started_at", "finished_at", "mock")
                    if key in payload
                }
                if isinstance(payload.get("dsh_context"), dict):
                    clean["dsh_context"] = {
                        key: value
                        for key, value in payload["dsh_context"].items()
                        if key in {"session_id", "agent_id", "turn_id"}
                    }
                if isinstance(payload.get("events"), list):
                    clean["events"] = [
                        {
                            "event_id": item.get("event_id"),
                            "kind": item.get("kind"),
                            "at": item.get("at"),
                            "payload": {"privacy_redacted": True},
                        }
                        for item in payload["events"]
                        if isinstance(item, dict)
                    ]
                clean["privacy_redaction"] = {
                    "reason": "memory_deleted" if deleting else "memory_corrected",
                    "event_id": event_id,
                    "scope": "referencing_request_bodies_and_derived_content",
                    "versions": versions,
                    "cleared_fields": sorted(set(payload) - set(clean)),
                }
                conn.execute(
                    "UPDATE request_journal SET payload=? WHERE request_id=?",
                    (json.dumps(clean, ensure_ascii=False), row[0]),
                )
        if deleting and "appraisal_history" in tables:
            conn.execute("DELETE FROM appraisal_history WHERE event_id=?", (event_id,))

    def get_memory(
        self, event_id: str, *, conn: sqlite3.Connection | None = None
    ) -> MemoryRecord | None:
        with self.lock:
            row = (
                self._read_connection(conn)
                .execute("SELECT payload FROM memories WHERE event_id=?", (event_id,))
                .fetchone()
            )
            return None if row is None else MemoryRecord.model_validate_json(row[0])

    def list_memories(self, *, conn: sqlite3.Connection | None = None) -> list[MemoryRecord]:
        with self.lock:
            rows = (
                self._read_connection(conn)
                .execute("SELECT payload FROM memories ORDER BY event_id")
                .fetchall()
            )
            return [MemoryRecord.model_validate_json(row[0]) for row in rows]

    def put_memory(
        self,
        memory: MemoryInput,
        *,
        expected_revision: int | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> MemoryRecord:
        memory = MemoryInput.model_validate(
            memory.model_dump(include=set(MemoryInput.model_fields))
        )
        with self._write_connection(conn) as active:
            old = self.get_memory(memory.event_id, conn=active)
            actual_revision = 0 if old is None else old.revision
            if expected_revision is not None and expected_revision != actual_revision:
                raise VersionConflict(
                    f"memory {memory.event_id!r}: expected revision {expected_revision}, "
                    f"found {actual_revision}"
                )
            if old is None:
                values = memory.model_dump(include=set(MemoryInput.model_fields))
                if values["recorded_at"] is None:
                    values["recorded_at"] = memory.occurred_at
                record = MemoryRecord.model_validate(values)
            else:
                # Ordinary knowledge editing cannot rewrite history, appraisal,
                # intensity, or reinforcement state. Reappraisal is engine-only.
                changes = {
                    "content": memory.content,
                    "source": memory.source,
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "evidence": memory.evidence,
                    "is_inference": memory.is_inference,
                    "tags": memory.tags,
                }
                if all(getattr(old, name) == value for name, value in changes.items()):
                    return old
                if expected_revision is None:
                    raise VersionConflict(
                        f"memory {memory.event_id!r}: a correction requires expected_revision"
                    )
                record = old.model_copy(update={**changes, "revision": old.revision + 1})
            active.execute(
                "INSERT INTO memories(event_id,payload) VALUES(?,?) "
                "ON CONFLICT(event_id) DO UPDATE SET payload=excluded.payload",
                (record.event_id, record.model_dump_json()),
            )
            self._index(record, active)
            if old is not None:
                self._purge_derived_body(record.event_id, active, old_content=old.content)
            self._audit(active, record, "memory.created" if old is None else "memory.updated")
            return record

    def save_record(self, record: MemoryRecord, conn: sqlite3.Connection) -> None:
        """Engine-only full record write; do not manufacture a knowledge revision."""
        record = MemoryRecord.model_validate(record.model_dump())
        with self._write_connection(conn) as active:
            if self.get_memory(record.event_id, conn=active) == record:
                return
            active.execute(
                "INSERT INTO memories(event_id,payload) VALUES(?,?) "
                "ON CONFLICT(event_id) DO UPDATE SET payload=excluded.payload",
                (record.event_id, record.model_dump_json()),
            )
            old_terms = active.execute(
                "SELECT terms FROM memory_fts WHERE event_id=?", (record.event_id,)
            ).fetchone()
            new_terms = " ".join(_tokens(" ".join((record.content, record.source, *record.tags))))
            # Recall/appraisal changes do not rewrite the unchanged full-text index.
            if old_terms is None or old_terms[0] != new_terms:
                self._index(record, active)
            self._audit(active, record, "memory.saved")

    def delete_memory(
        self,
        event_id: str,
        *,
        expected_revision: int | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        with self._write_connection(conn) as active:
            old = self.get_memory(event_id, conn=active)
            actual_revision = 0 if old is None else old.revision
            if expected_revision is not None and expected_revision != actual_revision:
                raise VersionConflict(
                    f"memory {event_id!r}: expected revision {expected_revision}, "
                    f"found {actual_revision}"
                )
            if old is None:
                return False
            active.execute("DELETE FROM memories WHERE event_id=?", (event_id,))
            active.execute("DELETE FROM memory_fts WHERE event_id=?", (event_id,))
            self._purge_derived_body(event_id, active, deleting=True, old_content=old.content)
            self._audit(active, old, "memory.deleted")
            return True

    def search(
        self,
        query: SearchQuery,
        *,
        at: float,
        settings: Settings,
        conn: sqlite3.Connection | None = None,
    ) -> list[RecallCandidate]:
        """Retrieve and rank without changing memory, FTS, audit, or affect state.

        Filters restrict the candidate set before ranking. With no embedding
        provider FTS supplies lexical candidates. A host embedding adapter adds
        semantic candidates from the filtered corpus using fresh vectors only;
        this simple adapter is intended for small corpora, not an ANN service.
        """
        query = SearchQuery.model_validate(query.model_dump())
        settings = Settings.model_validate(settings.model_dump())
        finite_timestamp(at)
        if not query.limit:
            return []
        terms = _query_tokens(query.text)
        if query.text.strip() and not terms:
            return []
        predicates: list[str] = []
        parameters: list[str | float] = []
        if query.source is not None:
            predicates.append("json_extract(m.payload,'$.source')=?")
            parameters.append(query.source)
        if query.since is not None:
            predicates.append("json_extract(m.payload,'$.occurred_at')>=?")
            parameters.append(query.since)
        if query.until is not None:
            predicates.append("json_extract(m.payload,'$.occurred_at')<=?")
            parameters.append(query.until)
        if query.tags:
            # A single bound JSON array avoids SQLite variable/expression limits
            # while retaining all-tags (rather than any-tag) filter semantics.
            predicates.append(
                "NOT EXISTS(SELECT 1 FROM json_each(?) q WHERE NOT EXISTS("
                "SELECT 1 FROM json_each(m.payload,'$.tags') t WHERE t.value=q.value))"
            )
            parameters.append(json.dumps(query.tags, ensure_ascii=False))
        lexical_ids: set[str] = set()
        with self._read_transaction(conn) as active:
            if terms:
                # Quoting each token removes all user-controlled MATCH syntax;
                # bindings independently protect the SQL statement itself.
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                lexical_ids = {
                    row[0]
                    for row in active.execute(
                        "SELECT event_id FROM memory_fts WHERE memory_fts MATCH ?", (expression,)
                    )
                }
                if self.embedding is None:
                    predicates.append(
                        "m.event_id IN (SELECT event_id FROM memory_fts WHERE memory_fts MATCH ?)"
                    )
                    parameters.append(expression)
            sql = "SELECT m.payload FROM memories m"
            if predicates:
                sql += " WHERE " + " AND ".join(predicates)
            sql += " ORDER BY m.event_id"
            records = [
                MemoryRecord.model_validate_json(row[0])
                for row in active.execute(sql, parameters).fetchall()
            ]
        if not records:
            return []
        semantic: dict[str, float] = {}
        if terms and self.embedding is not None:
            # No writes and no cache: corrections/deletions cannot leave stale
            # vectors, and switching embedding providers cannot mix dimensions.
            vectors = self.embedding.embed([query.text] + [r.content for r in records])
            if len(vectors) != len(records) + 1:
                raise ValueError("embedding provider returned an incorrect vector count")
            for record, vector in zip(records, vectors[1:]):
                semantic[record.event_id] = _cosine(vectors[0], vector)
        candidates: list[RecallCandidate] = []
        term_set = set(terms)
        for record in records:
            lexical = 0.0
            if record.event_id in lexical_ids:
                document_tokens = set(
                    _tokens(" ".join((record.content, record.source, *record.tags)))
                )
                # FTS unicode61 may fold accents not represented by Python
                # token equality. An FTS hit always proves at least one match.
                lexical = max(1, len(term_set & document_tokens)) / len(term_set)
            relevance = max(lexical, semantic.get(record.event_id, 0.0)) if terms else 1.0
            if terms and relevance <= 0.0:
                continue
            retention = _retention(record, at=at, settings=settings)
            importance = record.importance
            intensity = record.intensity / (1.0 + record.intensity)
            total = relevance * (
                settings.relevance_base_weight
                + settings.retention_weight * retention
                + settings.importance_weight * importance
                + settings.intensity_weight * intensity
            )
            candidates.append(
                RecallCandidate(
                    memory=record,
                    relevance=relevance,
                    retention=retention,
                    importance_score=importance,
                    intensity_score=intensity,
                    total_score=total,
                    explanation=(
                        f"relevance={relevance:.6f} (lexical={lexical:.6f}, "
                        f"semantic={semantic.get(record.event_id, 0.0):.6f}); "
                        f"retention={retention:.6f} (event_age_seconds="
                        f"{max(0.0, at - record.occurred_at):.3f}, "
                        f"strength={record.retention_boost:.6f}); "
                        f"importance={importance:.6f}; intensity={intensity:.6f}; "
                        f"total=relevance*({settings.relevance_base_weight:g}+"
                        f"{settings.retention_weight:g}*retention+"
                        f"{settings.importance_weight:g}*importance+"
                        f"{settings.intensity_weight:g}*intensity)={total:.6f}"
                    ),
                )
            )
        candidates.sort(
            key=lambda item: (-item.total_score, -item.memory.occurred_at, item.memory.event_id)
        )
        return candidates[: query.limit]

    def close(self) -> None:
        with self.lock:
            self.connection.close()
