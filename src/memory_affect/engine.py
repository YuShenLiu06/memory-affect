"""Host-driven request snapshots and atomic memory/affect transactions.

No method starts a worker, schedules a callback, or reads time during a request.
The request creation anchor is the only clock used by request operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Iterable
from decimal import Decimal, localcontext
from itertools import islice
from pathlib import Path
from typing import Any

from .director import render_director
from .models import (
    AuxiliaryProposal,
    CommitResult,
    Emotion,
    Injection,
    MemoryInput,
    MemoryRecord,
    RecallCandidate,
    RequestConflict,
    RequestSnapshot,
    SearchQuery,
    Settings,
    StateSnapshot,
    VersionConflict,
    finite_timestamp,
    utc_now,
)
from .storage import SQLiteStore


class MemoryLibrary:
    """Persistent library whose work is explicitly driven by its host."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        settings: Settings = Settings(),
        initial_time: float | None = None,
    ) -> None:
        self.settings = Settings.model_validate(settings.model_dump())
        anchor = finite_timestamp(utc_now() if initial_time is None else initial_time)
        self.store = SQLiteStore(Path(path))
        try:
            with self.store.transaction() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS engine_state "
                    "(version INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS engine_requests "
                    "(request_id TEXT PRIMARY KEY, created_at REAL NOT NULL, "
                    "snapshot TEXT NOT NULL, result TEXT)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS engine_commands "
                    "(command_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS engine_request_commands "
                    "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, "
                    "command_id TEXT NOT NULL, fingerprint TEXT NOT NULL, result TEXT NOT NULL, "
                    "UNIQUE(request_id, command_id))"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS engine_request_events "
                    "(request_id TEXT NOT NULL, event_id TEXT NOT NULL, "
                    "recalled INTEGER NOT NULL DEFAULT 0, impacted INTEGER NOT NULL DEFAULT 0, "
                    "PRIMARY KEY(request_id,event_id))"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS appraisal_history "
                    "(event_id TEXT NOT NULL, appraisal_version INTEGER NOT NULL, "
                    "appraisal TEXT NOT NULL, evidence TEXT NOT NULL, at REAL NOT NULL, "
                    "effect_mode TEXT NOT NULL, PRIMARY KEY(event_id, appraisal_version))"
                )
                if conn.execute("SELECT 1 FROM engine_state LIMIT 1").fetchone() is None:
                    state = StateSnapshot(
                        evaluated_at=anchor,
                        next_due_at=anchor,
                    )
                    self._save_state(state, conn)
                    self._audit(conn, anchor, "state_initialized", state.model_dump())
        except BaseException:
            self.store.close()
            raise

    @staticmethod
    def _save_state(state: StateSnapshot, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO engine_state(version, payload) VALUES (?, ?)",
            (state.version, state.model_dump_json()),
        )

    @staticmethod
    def _load_state(conn: sqlite3.Connection) -> StateSnapshot:
        row = conn.execute(
            "SELECT payload FROM engine_state ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("library state is missing")
        return StateSnapshot.model_validate_json(row[0])

    @staticmethod
    def _audit(conn: sqlite3.Connection, at: float, kind: str, payload: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO audit_log(at, kind, payload) VALUES (?, ?, ?)",
            (at, kind, json.dumps(payload, ensure_ascii=False, allow_nan=False)),
        )

    @staticmethod
    def _stored_snapshot(snapshot: RequestSnapshot, conn: sqlite3.Connection) -> RequestSnapshot:
        # A frozen Pydantic model can still be forged with model_construct/copy;
        # compare every field with the immutable database copy, not just its ID.
        value = RequestSnapshot.model_validate(snapshot.model_dump())
        row = conn.execute(
            "SELECT snapshot FROM engine_requests WHERE request_id = ?", (value.request_id,)
        ).fetchone()
        if row is None:
            raise RequestConflict("request was not begun by this library")
        stored = RequestSnapshot.model_validate_json(row[0])
        identity = ("request_id", "created_at", "baseline", "periodic_due")
        if any(getattr(stored, key) != getattr(value, key) for key in identity):
            raise RequestConflict("snapshot differs from its persistent request")
        return stored

    def begin_request(self, request_id: str, created_at: float) -> RequestSnapshot:
        """Freeze the request; settle elapsed emotion only when the period is due."""
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200:
            raise ValueError("request_id must be a nonblank string of at most 200 characters")
        anchor = finite_timestamp(created_at)
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT created_at, snapshot FROM engine_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is not None:
                if row[0] != anchor:
                    raise RequestConflict("request_id already has a different creation anchor")
                return RequestSnapshot.model_validate_json(row[1])
            current = self._load_state(conn)
            if anchor < current.evaluated_at:
                raise RequestConflict("new request anchor predates the latest committed state")
            decay = math.exp(
                -math.log(2.0)
                * (anchor - current.evaluated_at)
                / self.settings.emotion_half_life_seconds
            )
            baseline = current
            due = anchor >= current.next_due_at
            periodic: tuple[MemoryRecord, ...] = ()
            if due and self.settings.summary_m:
                recent = self.store.search(
                    SearchQuery(
                        since=anchor - self.settings.recent_window_seconds,
                        until=anchor,
                        limit=self.settings.summary_m,
                    ),
                    at=anchor,
                    settings=self.settings,
                    conn=conn,
                )
                strongest = self.store.search(
                    SearchQuery(until=anchor, limit=self.settings.summary_m),
                    at=anchor,
                    settings=self.settings,
                    conn=conn,
                )
                union = {item.memory.event_id: item for item in (*recent, *strongest)}
                ranked = sorted(
                    union.values(),
                    key=lambda item: (
                        -item.total_score,
                        -item.memory.occurred_at,
                        item.memory.event_id,
                    ),
                )
                periodic = tuple(item.memory for item in ranked[: self.settings.summary_m])
            if due:
                baseline = StateSnapshot(
                    **{
                        **current.model_dump(),
                        "emotion": current.emotion.scaled(decay).model_dump(),
                        "evaluated_at": anchor,
                        "version": current.version + 1,
                        "next_due_at": finite_timestamp(anchor + self.settings.interval_seconds),
                    }
                )
                self._save_state(baseline, conn)
                self._audit(
                    conn,
                    anchor,
                    "periodic_refreshed",
                    {
                        "request_id": request_id,
                        "version": baseline.version,
                        "previous_version": current.version,
                        "emotion": baseline.emotion.model_dump(),
                        "next_due_at": baseline.next_due_at,
                    },
                )
            snapshot = RequestSnapshot(
                request_id=request_id,
                created_at=anchor,
                baseline=baseline,
                periodic_due=due,
                periodic_memories=periodic,
            )
            conn.execute(
                "INSERT INTO engine_requests(request_id, created_at, snapshot) VALUES (?, ?, ?)",
                (request_id, anchor, snapshot.model_dump_json()),
            )
            self._audit(
                conn,
                anchor,
                "request_begun",
                {
                    "request_id": request_id,
                    "baseline_version": baseline.version,
                    "periodic_due": due,
                },
            )
            return snapshot

    def search(self, snapshot: RequestSnapshot, query: SearchQuery) -> list[RecallCandidate]:
        """Retrieve candidates without affecting state, deadlines, or reinforcement."""
        validated_query = SearchQuery.model_validate(query.model_dump())
        with self.store.lock:
            fixed = self._stored_snapshot(snapshot, self.store.connection)
            return self.store.search(
                validated_query,
                at=fixed.created_at,
                settings=self.settings,
                conn=self.store.connection,
            )

    def commit(
        self,
        snapshot: RequestSnapshot,
        *,
        selected_ids: Iterable[str] = (),
        proposal: AuxiliaryProposal | None = None,
        expected_version: int | None = None,
        command_id: str | None = None,
    ) -> CommitResult:
        """Atomically apply confirmed recall/proposals once, using the fixed anchor.

        Periodic refresh was already settled by begin_request. An empty commit
        is a durable, idempotent noop and never resets the deadline. Optional named
        commands permit same-request tool recall after the first receipt: they
        require the exact preceding receipt version, and continuations are recall-only.
        """
        if command_id is not None and (
            not isinstance(command_id, str) or not command_id.strip() or len(command_id) > 200
        ):
            raise ValueError("command_id must be a nonblank string of at most 200 characters")
        if command_id is not None and expected_version is None:
            raise ValueError("named commands require explicit expected_version")
        if isinstance(selected_ids, (str, bytes)):
            raise ValueError("selected_ids must be an iterable of canonical event IDs")
        selected = tuple(dict.fromkeys(selected_ids))
        if any(not isinstance(value, str) or not value.strip() for value in selected):
            raise ValueError("selected_ids must contain nonblank strings")
        value = (
            None if proposal is None else AuxiliaryProposal.model_validate(proposal.model_dump())
        )
        if expected_version is not None and (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 0
        ):
            raise ValueError("expected_version must be a nonnegative integer")
        with self.store.transaction() as conn:
            fixed = self._stored_snapshot(snapshot, conn)
            row = conn.execute(
                "SELECT result FROM engine_requests WHERE request_id = ?", (fixed.request_id,)
            ).fetchone()
            legacy = (
                CommitResult.model_validate_json(row[0])
                if row is not None and row[0] is not None
                else None
            )
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "selected": selected,
                        "proposal": None if value is None else value.model_dump(mode="json"),
                        "expected_version": expected_version,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if command_id is None and legacy is not None:
                return legacy.model_copy(update={"status": "duplicate"})
            commands = conn.execute(
                "SELECT command_id,fingerprint,result FROM engine_request_commands "
                "WHERE request_id=? ORDER BY sequence",
                (fixed.request_id,),
            ).fetchall()
            if command_id is None and commands:
                raise RequestConflict("legacy commit cannot start after named commands")
            if command_id is not None:
                for existing in commands:
                    if existing[0] == command_id:
                        if existing[1] != fingerprint:
                            raise RequestConflict("command_id reused with different input")
                        return CommitResult.model_validate_json(existing[2]).model_copy(
                            update={"status": "duplicate"}
                        )
                if (legacy is not None or commands) and value is not None:
                    raise ValueError("continuation commands support selected recall only")
            chain_previous = (
                CommitResult.model_validate_json(commands[-1][2]) if commands else legacy
            )
            chain_version = (
                chain_previous.state.version if chain_previous else fixed.baseline.version
            )
            current = self._load_state(conn)
            required = fixed.baseline.version if expected_version is None else expected_version
            allowed_version = chain_version if command_id is not None else fixed.baseline.version
            if current.version != allowed_version or current.version != required:
                raise VersionConflict(
                    "state version changed; command must follow this request's receipt chain"
                )
            seen_recalled = {
                item[0]
                for item in conn.execute(
                    "SELECT event_id FROM engine_request_events WHERE request_id=? AND recalled=1",
                    (fixed.request_id,),
                )
            }
            seen_impacted = {
                item[0]
                for item in conn.execute(
                    "SELECT event_id FROM engine_request_events WHERE request_id=? AND impacted=1",
                    (fixed.request_id,),
                )
            }
            # Old databases have only the legacy receipt and immutable impact audit.
            if legacy is not None:
                seen_recalled.update(legacy.recalled_ids)
                for impact_row in conn.execute(
                    "SELECT payload FROM audit_log WHERE kind='emotion_impact'"
                ):
                    impact_data = json.loads(impact_row[0])
                    if impact_data.get("request_id") == fixed.request_id:
                        seen_impacted.add(impact_data["event_id"])
            if fixed.created_at < current.evaluated_at:
                raise RequestConflict("request anchor predates the latest committed state")
            changed = False
            decay = math.exp(
                -math.log(2.0)
                * (fixed.created_at - current.evaluated_at)
                / self.settings.emotion_half_life_seconds
            )
            emotion = current.emotion.scaled(decay)
            director = current.director
            # One source per canonical event: new > reappraisal > recall.
            impacts: dict[str, tuple[Emotion, str, Emotion | None]] = {}
            records: dict[str, MemoryRecord] = {}
            redacted_bodies: list[str] = []
            if value is not None:
                evidences = (
                    [memory.evidence for memory in value.new_memories]
                    + [item.evidence for item in value.reappraisals]
                    + [item.evidence for item in value.corrections]
                    + [item.evidence for item in value.deletions]
                )
                if any(not evidence.strip() for evidence in evidences):
                    raise ValueError("proposal evidence must be nonblank")
                new_ids = [memory.event_id for memory in value.new_memories]
                reappraisal_ids = [item.event_id for item in value.reappraisals]
                if len(set(new_ids)) != len(new_ids):
                    raise ValueError("duplicate new event_id in proposal")
                if len(set(reappraisal_ids)) != len(reappraisal_ids):
                    raise ValueError("duplicate reappraisal event_id in proposal")
                for memory in value.new_memories:
                    existing = self.store.get_memory(memory.event_id, conn=conn)
                    if existing is not None:
                        original = memory.model_dump()
                        original["recorded_at"] = (
                            memory.occurred_at if memory.recorded_at is None else memory.recorded_at
                        )
                        stored = existing.model_dump(include=set(MemoryInput.model_fields))
                        if original == stored:
                            continue
                        raise ValueError("new event_id exists with different data")
                    record = self.store.put_memory(memory, conn=conn)
                    records[record.event_id] = record
                    self._appraisal_history(
                        record, memory.evidence, fixed.created_at, "initial", conn
                    )
                    impacts[record.event_id] = (
                        record.appraisal,
                        "new_memory",
                        None,
                    )
                    changed = True
                for item in value.reappraisals:
                    previous = self.store.get_memory(item.event_id, conn=conn)
                    if previous is None:
                        raise ValueError("reappraisal event does not exist")
                    record = previous
                    if record.appraisal_version != item.expected_appraisal_version:
                        raise VersionConflict(
                            "appraisal version changed; retry on the next real request"
                        )
                    if record.appraisal == item.appraisal:
                        continue
                    self._appraisal_history(
                        record, record.evidence, fixed.created_at, "existing", conn
                    )
                    revised = MemoryRecord.model_validate(
                        {
                            **record.model_dump(),
                            "appraisal": item.appraisal.model_dump(),
                            "appraisal_version": record.appraisal_version + 1,
                            "revision": record.revision + 1,
                        }
                    )
                    self.store.save_record(revised, conn)
                    records[revised.event_id] = revised
                    if revised.event_id in new_ids:
                        impacts[revised.event_id] = (
                            revised.appraisal,
                            "new_memory",
                            None,
                        )
                    elif item.effect_mode == "apply_delta":
                        impacts[revised.event_id] = (
                            revised.appraisal,
                            "reappraisal",
                            record.appraisal,
                        )
                    self._appraisal_history(
                        revised, item.evidence, fixed.created_at, item.effect_mode, conn
                    )
                    self._audit(
                        conn,
                        fixed.created_at,
                        "memory_reappraised",
                        {
                            "event_id": record.event_id,
                            "old_appraisal_version": record.appraisal_version,
                            "appraisal_version": revised.appraisal_version,
                            "old_appraisal": record.appraisal.model_dump(),
                            "appraisal": revised.appraisal.model_dump(),
                        },
                    )
                    changed = True
                correction_ids = [item.memory.event_id for item in value.corrections]
                deletion_ids = [item.event_id for item in value.deletions]
                if len(set(correction_ids)) != len(correction_ids):
                    raise ValueError("duplicate correction event_id")
                if len(set(deletion_ids)) != len(deletion_ids):
                    raise ValueError("duplicate deletion event_id")
                if set(deletion_ids) & (set(new_ids) | set(reappraisal_ids) | set(correction_ids)):
                    raise ValueError("cannot delete and otherwise modify the same event")
                for correction in value.corrections:
                    correction_memory = MemoryInput.model_validate(
                        {
                            **correction.memory.model_dump(),
                            "evidence": correction.evidence,
                        }
                    )
                    old = self.store.get_memory(correction_memory.event_id, conn=conn)
                    if old is None:
                        raise ValueError("correction event does not exist")
                    if old.revision != correction.expected_revision:
                        raise VersionConflict("correction revision is stale")
                    if not self._same_knowledge(old, correction_memory):
                        redacted_bodies.append(old.content)
                        revised = self.store.put_memory(
                            correction_memory,
                            expected_revision=correction.expected_revision,
                            conn=conn,
                        )
                        records[revised.event_id] = revised
                        self._audit(
                            conn,
                            fixed.created_at,
                            "memory_corrected",
                            {
                                "event_id": revised.event_id,
                                "revision": revised.revision,
                                "evidence_present": bool(correction.evidence.strip()),
                            },
                        )
                        changed = True
                for deletion in value.deletions:
                    deleted_record = self.store.get_memory(deletion.event_id, conn=conn)
                    deleted = self.store.delete_memory(
                        deletion.event_id, expected_revision=deletion.expected_revision, conn=conn
                    )
                    if deleted:
                        if deleted_record is not None:
                            redacted_bodies.append(deleted_record.content)
                        self._redact_event(deletion.event_id, conn, deleted=True)
                        self._audit(
                            conn,
                            fixed.created_at,
                            "memory_deleted",
                            {
                                "event_id": deletion.event_id,
                                "evidence_present": bool(deletion.evidence.strip()),
                            },
                        )
                        changed = True
                # Storage privacy cleanup can sanitize historical/current directors.
                # Never resurrect the pre-cleanup local copy in the next state.
                if redacted_bodies:
                    director = self._load_state(conn).director
                if value.director is not None:
                    proposed_director = value.director
                    if any(body and body in proposed_director for body in redacted_bodies):
                        proposed_director = ""
                        self._audit(
                            conn,
                            fixed.created_at,
                            "privacy_redaction",
                            {
                                "request_id": fixed.request_id,
                                "command_id": command_id,
                                "reason": "same_commit_director_contains_invalidated_body",
                            },
                        )
                    if proposed_director != director:
                        director = proposed_director
                        changed = True
            recalled: list[str] = []
            for event_id in selected:
                if event_id in seen_recalled:
                    continue
                found = records.get(event_id) or self.store.get_memory(event_id, conn=conn)
                if found is None:
                    continue
                record = MemoryRecord.model_validate(
                    {
                        **found.model_dump(),
                        "recall_count": found.recall_count + 1,
                        "last_recalled_at": fixed.created_at,
                        "retention_boost": found.retention_boost + self.settings.reinforcement_gain,
                    }
                )
                self.store.save_record(record, conn)
                records[event_id] = record
                recalled.append(event_id)
                if event_id not in impacts:
                    impacts[event_id] = (record.appraisal, "recall", None)
                self._audit(
                    conn,
                    fixed.created_at,
                    "memory_recalled",
                    {
                        "event_id": event_id,
                        "appraisal_version": record.appraisal_version,
                        "recall_count": record.recall_count,
                        "retention_boost": record.retention_boost,
                    },
                )
                changed = True
            for event_id, (impact, reason, old_appraisal) in impacts.items():
                if event_id in seen_impacted:
                    continue
                record = records[event_id]
                divisor = 1 if reason == "new_memory" else max(1, record.recall_count)
                raw, delta = self._pulse(impact, record.intensity, divisor, old_appraisal)
                emotion = emotion.plus(delta)
                self._audit(
                    conn,
                    fixed.created_at,
                    "emotion_impact",
                    {
                        "request_id": fixed.request_id,
                        "event_id": event_id,
                        "appraisal_version": record.appraisal_version,
                        "recall_count": record.recall_count,
                        "divisor": divisor,
                        "reason": reason,
                        "raw": raw,
                        "applied": delta.model_dump(),
                        "delta": delta.model_dump(),
                        "max_abs_pulse": self.settings.max_abs_pulse,
                    },
                )
            if changed:
                state = StateSnapshot(
                    version=current.version + 1,
                    emotion=emotion,
                    evaluated_at=fixed.created_at,
                    next_due_at=finite_timestamp(fixed.created_at + self.settings.interval_seconds),
                    director=director,
                )
                self._save_state(state, conn)
                result = CommitResult(
                    command_id=command_id,
                    request_id=fixed.request_id,
                    status="committed",
                    state=state,
                    recalled_ids=tuple(recalled),
                )
                self._audit(
                    conn,
                    fixed.created_at,
                    "state_committed",
                    {
                        "request_id": fixed.request_id,
                        "version": state.version,
                        "baseline_version": fixed.baseline.version,
                        "periodic_refresh": False,
                        "emotion": state.emotion.model_dump(),
                        "next_due_at": state.next_due_at,
                    },
                )
            else:
                result = CommitResult(
                    command_id=command_id,
                    request_id=fixed.request_id,
                    status="noop",
                    state=current,
                )
                self._audit(
                    conn,
                    fixed.created_at,
                    "request_noop",
                    {
                        "request_id": fixed.request_id,
                        "version": current.version,
                    },
                )
            for event_id in set(recalled) | set(impacts):
                conn.execute(
                    "INSERT INTO engine_request_events(request_id,event_id,recalled,impacted) "
                    "VALUES(?,?,?,?) ON CONFLICT(request_id,event_id) DO UPDATE SET "
                    "recalled=max(recalled,excluded.recalled), "
                    "impacted=max(impacted,excluded.impacted)",
                    (
                        fixed.request_id,
                        event_id,
                        int(event_id in recalled),
                        int(event_id in impacts),
                    ),
                )
            if command_id is None:
                conn.execute(
                    "UPDATE engine_requests SET result = ? WHERE request_id = ?",
                    (result.model_dump_json(), fixed.request_id),
                )
            else:
                conn.execute(
                    "INSERT INTO engine_request_commands(request_id,command_id,fingerprint,result) "
                    "VALUES(?,?,?,?)",
                    (fixed.request_id, command_id, fingerprint, result.model_dump_json()),
                )
            return result

    def _pulse(
        self, appraisal: Emotion, intensity: float, divisor: int, old: Emotion | None
    ) -> tuple[dict[str, float | str], Emotion]:
        raw: dict[str, float | str] = {}
        applied: dict[str, float] = {}
        bound = Decimal(str(self.settings.max_abs_pulse))
        with localcontext() as context:
            context.prec = 50
            for axis in ("happy", "angry", "sad"):
                base = Decimal(str(getattr(appraisal, axis)))
                if old is not None:
                    base -= Decimal(str(getattr(old, axis)))
                value = base * Decimal(str(intensity)) / Decimal(divisor)
                numeric = float(value)
                raw[axis] = numeric if math.isfinite(numeric) else str(value)
                applied[axis] = float(max(-bound, min(bound, value)))
        return raw, Emotion(**applied)

    def compose(
        self,
        snapshot: RequestSnapshot,
        *,
        result: CommitResult | None = None,
        recalled: Iterable[RecallCandidate | MemoryRecord | str] = (),
    ) -> Injection:
        """Build a budgeted view without changing the immutable request baseline.

        Whole memory entries only: IDs not present in ``memory_ids`` were not
        included. The host must intersect its *on-demand* selections with these
        IDs; periodic display alone must never be confirmed as effective recall.
        """
        if isinstance(recalled, (str, bytes)):
            raise ValueError("recalled must be an iterable, not one string")
        recalled_items = tuple(islice(recalled, self.settings.recall_n))
        with self.store.lock:
            fixed = self._stored_snapshot(snapshot, self.store.connection)
            state = fixed.baseline
            successful_result = False
            if result is not None:
                supplied = CommitResult.model_validate(result.model_dump())
                if supplied.command_id is None:
                    row = self.store.connection.execute(
                        "SELECT result FROM engine_requests WHERE request_id = ?",
                        (fixed.request_id,),
                    ).fetchone()
                else:
                    row = self.store.connection.execute(
                        "SELECT result FROM engine_request_commands "
                        "WHERE request_id=? AND command_id=?",
                        (fixed.request_id, supplied.command_id),
                    ).fetchone()
                if row is None or row[0] is None:
                    raise RequestConflict("result was not committed by this library")
                prior = CommitResult.model_validate_json(row[0])
                if (
                    supplied.request_id != fixed.request_id
                    or supplied.command_id != prior.command_id
                    or supplied.state != prior.state
                    or supplied.recalled_ids != prior.recalled_ids
                    or supplied.status not in (prior.status, "duplicate")
                ):
                    raise RequestConflict("result does not match the persistent receipt")
                if prior.status == "committed" or (
                    prior.command_id is not None and prior.state.version != fixed.baseline.version
                ):
                    # A named noop can attest an earlier successful link in this chain.
                    state = prior.state
                    successful_result = True
            budget = self.settings.injection_char_budget
            # The same memory packing budget is used before and after commit,
            # regardless of emotion number widths or a newly supplied director.
            state_slot = 256
            director_slot = min(2048, max(0, budget - state_slot) // 4)
            memory_budget = max(0, budget - state_slot - director_slot - 2)
            pieces: list[str] = []
            periodic_pieces: list[str] = []
            recalled_pieces: list[str] = []
            ids: list[str] = []
            memory_used = 0
            memory_count = 0

            def append_memory(value: str, kind: str) -> bool:
                nonlocal memory_used, memory_count
                cost = len(value) + (1 if memory_count else 0)
                if memory_used + cost > memory_budget:
                    return False
                pieces.append(value)
                (recalled_pieces if kind == "recalled" else periodic_pieces).append(value)
                memory_used += cost
                memory_count += 1
                return True

            if fixed.periodic_due or successful_result or recalled_items:
                state_block = (
                    f"[memory_affect_state version={state.version}; replace prior state block]\n"
                    + json.dumps(
                        state.emotion.model_dump(), ensure_ascii=False, separators=(",", ":")
                    )
                )
                if len(state_block) > state_slot:
                    raise ValueError("state block exceeds its 256-character reserved slot")
                pieces.append(state_block)
                prefix = "[director] "
                director = render_director(state.emotion, self.settings)
                if (
                    self.settings.director_enabled
                    and self.settings.external_director_policy == "append_once"
                    and state.director
                ):
                    # One versioned block, never accumulate external notes across compose.
                    director += "\n外部可选表达建议（不改变安全、事实、权限）：" + state.director
                if director and director_slot > len(prefix):
                    pieces.append(prefix + director[: director_slot - len(prefix)])
            state_piece_count = len(pieces)
            periodic_pieces.extend(pieces)
            seen: set[str] = set()
            candidates: list[tuple[str, MemoryRecord | str]] = []
            count = 0
            for item in recalled_items:
                if count >= self.settings.recall_n:
                    break
                record: MemoryRecord | str
                if isinstance(item, RecallCandidate):
                    record = MemoryRecord.model_validate(item.memory.model_dump())
                elif isinstance(item, MemoryRecord):
                    record = MemoryRecord.model_validate(item.model_dump())
                elif isinstance(item, str):
                    record = item
                else:
                    raise ValueError("recalled entries must be candidates, records, or event IDs")
                event_id = record if isinstance(record, str) else record.event_id
                if event_id in seen:
                    continue
                seen.add(event_id)
                candidates.append(("recalled", record))
                count += 1
            for memory in fixed.periodic_memories[: self.settings.summary_m]:
                if memory.event_id not in seen:
                    seen.add(memory.event_id)
                    candidates.append(("periodic", memory))
            for kind, candidate in candidates:
                event_id = candidate if isinstance(candidate, str) else candidate.event_id
                current = self.store.get_memory(event_id)
                if current is None:
                    append_memory("[memory unavailable: deleted]", kind)
                    continue
                if not isinstance(candidate, str) and current.revision != candidate.revision:
                    safe_receipt_recall = (
                        result is not None
                        and successful_result
                        and event_id in result.recalled_ids
                        and self._same_knowledge(current, candidate)
                    )
                    if not safe_receipt_recall:
                        append_memory(
                            "[memory unavailable: revised; retrieve a fresh candidate]", kind
                        )
                        continue
                entry = f"[{kind} memory; untrusted data] " + json.dumps(
                    {"event_id": current.event_id, "content": current.content},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if append_memory(entry, kind):
                    ids.append(current.event_id)
            return Injection(
                request_id=fixed.request_id,
                baseline_version=fixed.baseline.version,
                state_version=state.version,
                content="\n".join(pieces),
                periodic_content="\n".join(periodic_pieces),
                recalled_content="\n".join(recalled_pieces),
                state_content="\n".join(periodic_pieces[:state_piece_count]),
                periodic_memory_content="\n".join(periodic_pieces[state_piece_count:]),
                periodic_due=fixed.periodic_due,
                memory_ids=tuple(ids),
            )

    @staticmethod
    def _redact_event(event_id: str, conn: sqlite3.Connection, *, deleted: bool = False) -> None:
        for row in conn.execute("SELECT request_id, snapshot FROM engine_requests").fetchall():
            snapshot = RequestSnapshot.model_validate_json(row[1])
            kept = tuple(item for item in snapshot.periodic_memories if item.event_id != event_id)
            if len(kept) != len(snapshot.periodic_memories):
                clean = snapshot.model_copy(update={"periodic_memories": kept})
                conn.execute(
                    "UPDATE engine_requests SET snapshot = ? WHERE request_id = ?",
                    (clean.model_dump_json(), row[0]),
                )
        if deleted:
            conn.execute("DELETE FROM appraisal_history WHERE event_id = ?", (event_id,))

    @staticmethod
    def _same_knowledge(old: MemoryRecord, memory: MemoryInput) -> bool:
        fields = (
            "content",
            "source",
            "importance",
            "tags",
            "confidence",
            "evidence",
            "is_inference",
        )
        return all(getattr(old, field) == getattr(memory, field) for field in fields)

    @staticmethod
    def _appraisal_history(
        record: MemoryRecord, evidence: str, at: float, mode: str, conn: sqlite3.Connection
    ) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO appraisal_history "
            "(event_id, appraisal_version, appraisal, evidence, at, effect_mode) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.event_id,
                record.appraisal_version,
                record.appraisal.model_dump_json(),
                evidence,
                at,
                mode,
            ),
        )

    def manage_memory(
        self,
        command_id: str,
        *,
        memory: MemoryInput | None = None,
        delete_id: str | None = None,
        expected_revision: int | None = None,
        evidence: str,
    ) -> dict[str, Any]:
        """Idempotent host knowledge CRUD, without an emotional pulse or timer reset."""
        if not isinstance(command_id, str) or not command_id.strip() or len(command_id) > 200:
            raise ValueError("command_id must be a nonblank string of at most 200 characters")
        if (memory is None) == (delete_id is None):
            raise ValueError("supply exactly one of memory or delete_id")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 4000:
            raise ValueError("nonblank evidence is required (at most 4000 characters)")
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a nonnegative integer")
        if delete_id is not None and (not isinstance(delete_id, str) or not delete_id.strip()):
            raise ValueError("delete_id must be a nonblank string")
        value = None if memory is None else MemoryInput.model_validate(memory.model_dump())
        payload = {
            "memory": None if value is None else value.model_dump(mode="json"),
            "delete_id": delete_id,
            "expected_revision": expected_revision,
            "evidence": evidence,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if value is not None:
            value = MemoryInput.model_validate({**value.model_dump(), "evidence": evidence})
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT fingerprint, result FROM engine_commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is not None:
                if row[0] != fingerprint:
                    raise RequestConflict("command_id reused with different inputs")
                result: dict[str, Any] = json.loads(row[1])
                return result
            current = self._load_state(conn)
            event_id = value.event_id if value is not None else str(delete_id)
            old = self.store.get_memory(event_id, conn=conn)
            actual_revision = 0 if old is None else old.revision
            if expected_revision is not None and expected_revision != actual_revision:
                raise VersionConflict("knowledge revision changed")
            changed = False
            revision: int | None = actual_revision or None
            action = "deleted" if value is None else ("created" if old is None else "updated")
            if value is None:
                changed = self.store.delete_memory(
                    event_id, expected_revision=expected_revision, conn=conn
                )
            elif old is None or not self._same_knowledge(old, value):
                record = self.store.put_memory(
                    value, expected_revision=expected_revision, conn=conn
                )
                revision = record.revision
                if old is None:
                    self._appraisal_history(record, evidence, current.evaluated_at, "initial", conn)
                changed = True
            version = current.version
            if changed:
                self._redact_event(event_id, conn, deleted=value is None)
                current = self._load_state(conn)
                state = StateSnapshot(**{**current.model_dump(), "version": current.version + 1})
                self._save_state(state, conn)
                version = state.version
            receipt = {
                "command_id": command_id,
                "status": "committed" if changed else "noop",
                "action": action,
                "event_id": event_id,
                "revision": revision,
                "state_version": version,
            }
            self._audit(
                conn,
                current.evaluated_at,
                "knowledge_command",
                {
                    **receipt,
                    "evidence_present": True,
                    "time_basis": "current_state_anchor",
                },
            )
            conn.execute(
                "INSERT INTO engine_commands(command_id, fingerprint, result) VALUES (?, ?, ?)",
                (command_id, fingerprint, json.dumps(receipt, ensure_ascii=False)),
            )
            return receipt

    def current_state(self) -> StateSnapshot:
        """Return the last committed state, without wall-clock projection."""
        with self.store.lock:
            return self._load_state(self.store.connection)

    def state_history(self) -> list[StateSnapshot]:
        """Return immutable state versions in ascending version order."""
        with self.store.lock:
            rows = self.store.connection.execute(
                "SELECT payload FROM engine_state ORDER BY version"
            ).fetchall()
            return [StateSnapshot.model_validate_json(row[0]) for row in rows]

    def audit(self) -> list[dict[str, Any]]:
        with self.store.lock:
            rows = self.store.connection.execute(
                "SELECT id, at, kind, payload FROM audit_log ORDER BY id"
            ).fetchall()
            return [
                {"id": row[0], "at": row[1], "kind": row[2], "payload": json.loads(row[3])}
                for row in rows
            ]

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> MemoryLibrary:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
