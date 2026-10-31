"""Optional DSH host bridge, JSON-lines RPC; the core never imports this module.

No scheduler and no model calls. DSH owns the auxiliary call, main loop, and tools.
Each RPC opens the same durable library and preserves its request anchor/receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from .auxiliary import build_auxiliary_prompt, parse_auxiliary_response
from .engine import MemoryLibrary
from .models import (
    CommitResult,
    RecallCandidate,
    RequestConflict,
    RequestSnapshot,
    SearchQuery,
    VersionConflict,
)
from .request_journal import RequestJournal, safe_data


class DSHBridge:
    def __init__(self, library: MemoryLibrary) -> None:
        self.library = library
        self.journal = RequestJournal(library.store)

    def dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        op = message["op"]
        if op == "begin":
            request_id, anchor = message["request_id"], message["created_at"]
            body = message["request_body"]
            prior = self.journal.get_request(request_id)
            if prior and not prior.get("legacy"):
                if (
                    prior["created_at"] != anchor
                    or prior.get("user_envelope") != message["envelope"]
                ):
                    raise RequestConflict("DSH user envelope changed on retry")
                if "final_injection" in prior:
                    return {
                        "snapshot": prior["snapshot"],
                        "prepared": {
                            "injection": prior["final_injection"],
                            "receipt": prior.get("commit_receipt"),
                            "error": prior.get("error"),
                            "manifest": self._manifest(
                                [
                                    RecallCandidate.model_validate(c)
                                    for c in prior.get("recall_scores", [])
                                ],
                                prior.get("recall_decision", {}).get("candidate_selected_ids", []),
                            ),
                        },
                    }
                body = prior["request_body"]
            self.journal.begin(request_id, anchor, body, time.time())
            snapshot = self.library.begin_request(request_id, anchor)
            self.journal.record(
                request_id,
                snapshot=snapshot.model_dump(mode="json"),
                user_envelope=message["envelope"],
                periodic_trigger={
                    "triggered": snapshot.periodic_due,
                    "reason": "deadline_reached_at_user_entry"
                    if snapshot.periodic_due
                    else "deadline_not_reached_at_user_entry",
                    "anchor_source": "dsh_inbox_received_at",
                },
            )
            return {
                "snapshot": snapshot.model_dump(mode="json"),
                "auxiliary_spec": build_auxiliary_prompt(
                    snapshot, json.dumps(body, ensure_ascii=False)
                ),
            }
        if op == "prepare":
            snapshot = RequestSnapshot.model_validate(message["snapshot"])
            proposal = parse_auxiliary_response(message.get("proposal"))
            candidates = (
                self.library.search(snapshot, proposal.search or SearchQuery())
                if (proposal is not None and proposal.should_recall)
                else []
            )
            preview = self.library.compose(snapshot, recalled=candidates)
            selected = [
                c.memory.event_id for c in candidates if c.memory.event_id in preview.memory_ids
            ]
            error = None
            result = None
            try:
                result = self.library.commit(
                    snapshot,
                    selected_ids=(),
                    proposal=proposal,
                    command_id="auxiliary",
                    expected_version=snapshot.baseline.version,
                )
            except VersionConflict:
                error = "version_conflict: skipped; fixed baseline retained"
                selected = []
            injection = self.library.compose(snapshot, result=result, recalled=candidates)
            try:
                self.journal.record(
                    snapshot.request_id,
                    auxiliary_request=message.get("auxiliary_request"),
                    auxiliary_response=message.get("auxiliary_response"),
                    parsed_result=None if proposal is None else proposal.model_dump(mode="json"),
                    recall_decision={
                        "requested": bool(proposal and proposal.should_recall),
                        "reason": "invalid_or_failed_auxiliary"
                        if proposal is None
                        else "auxiliary_requested"
                        if proposal.should_recall
                        else "auxiliary_not_needed",
                        "candidate_selected_ids": selected,
                        "effective_confirmation": "deferred_until_frozen_main_request",
                    },
                    recall_scores=[c.model_dump(mode="json") for c in candidates],
                    commit_receipt=None if result is None else result.model_dump(mode="json"),
                    final_injection=injection.model_dump(mode="json"),
                    mock=bool(message.get("mock", False)),
                    error=error,
                )
            except RequestConflict:
                # A privacy tombstone must never be repopulated after a successful commit.
                if not (self.journal.get_request(snapshot.request_id) or {}).get(
                    "privacy_redaction"
                ):
                    raise
                error = "history_privacy_redacted; business_commit_not_replayed"
            return {
                "injection": injection.model_dump(mode="json"),
                "error": error,
                "receipt": None if result is None else result.model_dump(mode="json"),
                "manifest": self._manifest(candidates, selected),
            }
        if op == "search":
            snapshot = RequestSnapshot.model_validate(message["snapshot"])
            candidates = self.library.search(snapshot, SearchQuery.model_validate(message["query"]))
            injection = self.library.compose(snapshot, recalled=candidates)
            selected = [
                c.memory.event_id for c in candidates if c.memory.event_id in injection.memory_ids
            ]
            return {
                "content": injection.content,
                "memory_ids": selected,
                "manifest": self._manifest(candidates, selected),
                "effective_recall_committed": False,
            }
        if op == "compose":
            snapshot = RequestSnapshot.model_validate(message["snapshot"])
            receipt = message.get("receipt")
            return {
                "injection": self.library.compose(
                    snapshot,
                    result=None if receipt is None else CommitResult.model_validate(receipt),
                ).model_dump(mode="json")
            }
        if op == "confirm":
            snapshot = RequestSnapshot.model_validate(message["snapshot"])
            try:
                result = self.library.commit(
                    snapshot,
                    selected_ids=message["selected_ids"],
                    command_id=message["command_id"],
                    expected_version=message["expected_version"],
                )
                return {
                    "receipt": result.model_dump(mode="json"),
                    "injection": self.library.compose(
                        snapshot, result=result, recalled=message["selected_ids"]
                    ).model_dump(mode="json"),
                }
            except VersionConflict:
                return {"error": "version_conflict; fixed baseline retained"}
        if op == "event":
            self.journal.append_event(
                message["request_id"],
                message["event_id"],
                message["kind"],
                message["at"],
                message.get("payload"),
            )
            return {"recorded": True}
        raise ValueError("unknown bridge operation")

    @staticmethod
    def _manifest(candidates: Any, selected: list[str]) -> list[dict[str, str]]:
        return [
            {
                "event_id": c.memory.event_id,
                "entry": json.dumps(
                    {"event_id": c.memory.event_id, "content": c.memory.content},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            for c in candidates
            if c.memory.event_id in selected
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    args = parser.parse_args()
    with MemoryLibrary(args.database, initial_time=0) as library:
        bridge = DSHBridge(library)
        for line in sys.stdin:
            message: Any = None
            try:
                message = json.loads(line)
                result = bridge.dispatch(message)
                reply = {"id": message.get("id"), "ok": True, "result": result}
            except Exception as exc:
                reply = {
                    "id": message.get("id") if isinstance(message, dict) else None,
                    "ok": False,
                    "error": type(exc).__name__,
                }
            print(json.dumps(safe_data(reply), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
