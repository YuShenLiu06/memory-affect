"""Host-owned request history. Reads never run requests; missing data stays missing.

This optional journal uses its own schema version, never SQLite user_version or
engine tables. Credentials are intentionally excluded, but request content itself
is sensitive: keep the local database private and do not enter secrets in prompts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import RequestConflict, finite_timestamp
from .storage import SQLiteStore

FIELDS = (
    "request_body",
    "started_at",
    "finished_at",
    "auxiliary_request",
    "auxiliary_response",
    "parsed_result",
    "recall_scores",
    "snapshot",
    "commit_receipt",
    "final_injection",
    "error",
    "mock",
    "user_envelope",
    "periodic_trigger",
    "recall_decision",
    "main_request",
    "main_response",
    "dsh_context",
    "lifecycle",
    "tool_steps",
)
MAX_EVENTS = 1000
MAX_EVENT_BYTES = 262144
MAX_REQUEST_BYTES = 8_000_000

_SECRET = re.compile(r"(?i)(bearer\s+)[^\s\"']+|\bsk-[A-Za-z0-9_-]+")
_SENSITIVE = {"authorization", "apikey", "api_key", "access_token", "secret", "password"}


def safe_data(value: Any) -> Any:
    """JSON-safe credential redaction; does not pretend to detect arbitrary prose secrets."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if str(key).lower().replace("-", "_") in _SENSITIVE
            else safe_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [safe_data(item) for item in value]
    if isinstance(value, str):
        return _SECRET.sub("[REDACTED]", value)
    return value


class RequestJournal:
    def __init__(self, store: SQLiteStore, *, secrets: tuple[str, ...] = ()) -> None:
        self.store = store
        self._secrets = tuple(secret for secret in secrets if secret)
        with store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS request_journal_meta "
                "(name TEXT PRIMARY KEY, value INTEGER NOT NULL)"
            )
            row = conn.execute(
                "SELECT value FROM request_journal_meta WHERE name='schema'"
            ).fetchone()
            if row is not None and row[0] > 2:
                raise ValueError("request journal schema is newer than this application")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS request_journal "
                "(request_id TEXT PRIMARY KEY, created_at REAL NOT NULL, payload TEXT NOT NULL)"
            )
            conn.execute("INSERT OR REPLACE INTO request_journal_meta VALUES('schema',2)")

    def _clean(self, value: Any) -> Any:
        encoded = json.dumps(safe_data(value), ensure_ascii=False, allow_nan=False)
        for secret in self._secrets:
            # Replace inside JSON strings safely, including credentials containing quotes.
            encoded = encoded.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
        return json.loads(encoded)

    def begin(
        self, request_id: str, created_at: float, request_body: Any, started_at: float
    ) -> dict[str, Any]:
        finite_timestamp(created_at)
        finite_timestamp(started_at)
        body = self._clean(request_body)
        # Serialize before entering transaction: invalid input cannot create partial history.
        payload = json.dumps(
            {"request_body": body, "started_at": started_at}, ensure_ascii=False, allow_nan=False
        )
        if len(payload.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("request journal size limit exceeded")
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT created_at,payload FROM request_journal WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is not None:
                if row[0] != created_at or json.loads(row[1]).get("request_body") != body:
                    raise RequestConflict(
                        "request journal ID reused with different input or anchor"
                    )
            else:
                conn.execute(
                    "INSERT INTO request_journal VALUES(?,?,?)", (request_id, created_at, payload)
                )
        result = self.get_request(request_id)
        assert result is not None
        return result

    def record(self, request_id: str, **fields: Any) -> None:
        if set(fields) - set(FIELDS):
            raise ValueError(
                "unknown journal fields: " + ",".join(sorted(set(fields) - set(FIELDS)))
            )
        if "request_body" in fields or "started_at" in fields:
            raise RequestConflict("request input and start time are immutable")
        if fields.get("finished_at") is not None:
            finite_timestamp(fields["finished_at"])
        cleaned = self._clean(fields)
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT payload FROM request_journal WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(request_id)
            data = json.loads(row[0])
            if "privacy_redaction" in data:
                raise RequestConflict("redacted history cannot accept new bodies")
            for key, value in cleaned.items():
                if key in data and data[key] != value:
                    raise RequestConflict(f"journal field {key} is already recorded")
                data[key] = value
            encoded = json.dumps(data, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode("utf-8")) > MAX_REQUEST_BYTES:
                raise ValueError("request journal size limit exceeded")
            conn.execute(
                "UPDATE request_journal SET payload=? WHERE request_id=?", (encoded, request_id)
            )

    def append_event(
        self, request_id: str, event_id: str, kind: str, at: float, payload: Any
    ) -> dict[str, Any]:
        """Record a host-observed step/tool event once; no business/model execution.

        IDs are stable per request. Same ID/data returns the original event, even
        at capacity; changed data is a conflict. Limits reject, never silently trim.
        """
        finite_timestamp(at)
        if not event_id or len(event_id) > 200 or not kind or len(kind) > 100:
            raise ValueError("invalid event ID or kind")
        event = {"event_id": event_id, "kind": kind, "at": at, "payload": self._clean(payload)}
        if len(json.dumps(event, ensure_ascii=False).encode("utf-8")) > MAX_EVENT_BYTES:
            raise ValueError("journal event size limit exceeded")
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT payload FROM request_journal WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(request_id)
            data = json.loads(row[0])
            if "privacy_redaction" in data:
                raise RequestConflict("redacted history cannot accept new event bodies")
            events = data.setdefault("events", [])
            for old in events:
                if old["event_id"] == event_id:
                    if old != event:
                        raise RequestConflict("journal event ID reused with different data")
                    return dict(old)
            if len(events) >= MAX_EVENTS:
                raise ValueError("journal event count limit exceeded")
            events.append(event)
            encoded = json.dumps(data, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode("utf-8")) > MAX_REQUEST_BYTES:
                raise ValueError("request journal size limit exceeded")
            conn.execute(
                "UPDATE request_journal SET payload=? WHERE request_id=?", (encoded, request_id)
            )
        return event

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self.store.lock:
            conn = self.store.connection
            row = conn.execute(
                "SELECT created_at,payload FROM request_journal WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is not None:
                data = json.loads(row[1])
                return {
                    **data,
                    "events": data.get("events", []),
                    "request_id": request_id,
                    "created_at": row[0],
                    "legacy": False,
                    "missing_fields": [k for k in FIELDS if k not in data],
                }
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='engine_requests'"
            ).fetchone()
            if exists:
                old = conn.execute(
                    "SELECT created_at,snapshot,result FROM engine_requests WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if old:
                    data = {"snapshot": json.loads(old[1])}
                    if old[2] is not None:
                        data["commit_receipt"] = json.loads(old[2])
                    return {
                        **data,
                        "events": [],
                        "request_id": request_id,
                        "created_at": old[0],
                        "legacy": True,
                        "missing_fields": [k for k in FIELDS if k not in data],
                    }
        return None

    def list_requests(self) -> list[dict[str, Any]]:
        with self.store.lock:
            conn = self.store.connection
            ids = {r[0] for r in conn.execute("SELECT request_id FROM request_journal")}
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='engine_requests'").fetchone():
                ids.update(r[0] for r in conn.execute("SELECT request_id FROM engine_requests"))
            records = [self.get_request(request_id) for request_id in ids]
        return sorted(
            (r for r in records if r is not None),
            key=lambda r: (r["created_at"], r["request_id"]),
            reverse=True,
        )
