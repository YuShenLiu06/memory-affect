"""Optional HOST adapter: one explicit Chat Completions call, no agent loop.

No environment/config credentials are read, stored, or cached. HTTP redirects are
rejected so Authorization cannot silently cross origins. No calls occur on import.
"""

from __future__ import annotations

import copy
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .auxiliary import DeterministicStub, build_auxiliary_prompt, parse_auxiliary_response
from .models import AuxiliaryProposal, RequestSnapshot
from .request_journal import safe_data


@dataclass(frozen=True)
class OpenAIChatConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 10.0
    max_response_bytes: int = 2_000_000
    response_format: str = "json_schema"

    def __post_init__(self) -> None:
        if self.response_format not in {"json_schema", "json_object", "none"}:
            raise ValueError("unsupported response format")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain credentials, query or fragment")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("unencrypted HTTP is permitted only for explicit local endpoints")
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive and finite")
        if not 1 <= self.max_response_bytes <= 20_000_000:
            raise ValueError("invalid response size limit")


@dataclass(frozen=True)
class ChatCallResult:
    request: dict[str, Any]
    response: Any = None
    proposal: AuxiliaryProposal | None = None
    error: str | None = None
    mock: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def _transport(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float, limit: int
) -> str:
    request = urllib.request.Request(
        url, data=json.dumps(body, allow_nan=False).encode(), headers=headers, method="POST"
    )
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        raw: bytes = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("response exceeded configured size limit")
        return raw.decode("utf-8")


class OpenAIChatAdapter:
    def __init__(
        self, config: OpenAIChatConfig, *, transport: Callable[..., str] | None = None
    ) -> None:
        self.config = config
        self._transport = transport or _transport

    def build_request(
        self, snapshot: RequestSnapshot, request_body: dict[str, Any]
    ) -> dict[str, Any]:
        messages = request_body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("request_body.messages must be a nonempty list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}
                or "content" not in message
            ):
                raise ValueError("invalid host message")
        # Preserve the original message objects and non-text content, not flattened text.
        user_text = json.dumps(request_body, ensure_ascii=False, allow_nan=False)
        spec = build_auxiliary_prompt(
            snapshot, user_text, timeout_seconds=self.config.timeout_seconds
        )
        body = {
            "model": self.config.model,
            "messages": copy.deepcopy(messages)
            + [
                {
                    "role": "system",
                    "content": spec["additional_prompt"]
                    + "\nJSON schema:\n"
                    + json.dumps(spec["response_schema"], ensure_ascii=False),
                }
            ],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "memory_auxiliary",
                    "strict": False,
                    "schema": spec["response_schema"],
                },
            },
        }
        if self.config.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif self.config.response_format == "none":
            del body["response_format"]
        return body

    def judge(
        self, snapshot: RequestSnapshot, request_body: dict[str, Any], *, api_key: str = ""
    ) -> ChatCallResult:
        body = self.build_request(snapshot, request_body)
        if not api_key and urllib.parse.urlsplit(self.config.base_url).hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            return ChatCallResult(safe_data(body), error="remote API key is not configured")
        logged = safe_data(body)

        # A supplied session credential must not occur even if a server echoes it.
        def clean(value: Any) -> Any:
            safe = safe_data(value)
            if not api_key:
                return safe
            encoded = json.dumps(safe, ensure_ascii=False).replace(
                json.dumps(api_key, ensure_ascii=False)[1:-1], "[REDACTED]"
            )
            return json.loads(encoded)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        response: Any = None
        try:
            raw = self._transport(
                self.config.base_url.rstrip("/") + "/chat/completions",
                body,
                headers,
                self.config.timeout_seconds,
                self.config.max_response_bytes,
            )
            response = raw
            response = json.loads(raw)
            choice = response["choices"][0]
            if choice.get("finish_reason") not in {None, "stop"} or choice["message"].get(
                "refusal"
            ):
                return ChatCallResult(
                    clean(logged), clean(response), error="auxiliary response refused or incomplete"
                )
            content = choice["message"]["content"]
            proposal = parse_auxiliary_response(content)
            if (
                proposal is not None
                and api_key
                and api_key in json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False)
            ):
                return ChatCallResult(
                    clean(logged),
                    clean(response),
                    error="auxiliary output contained the session credential",
                )
            return ChatCallResult(
                clean(logged),
                clean(response),
                proposal,
                None if proposal is not None else "invalid auxiliary contract",
            )
        except urllib.error.HTTPError as exc:
            # Never serialize exception text, URL, headers or an untrusted error body.
            return ChatCallResult(clean(logged), error=f"HTTP error {exc.code}")
        except (TimeoutError, urllib.error.URLError):
            return ChatCallResult(clean(logged), error="connection failed or timed out")
        except (ValueError, KeyError, IndexError, TypeError, OSError):
            return ChatCallResult(
                clean(logged), clean(response), error="invalid or unreadable response"
            )

    def mock_judge(self, snapshot: RequestSnapshot, request_body: dict[str, Any]) -> ChatCallResult:
        body = self.build_request(snapshot, request_body)
        text = next(
            (
                m.get("content", "")
                for m in reversed(request_body["messages"])
                if m.get("role") == "user"
            ),
            "",
        )
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        proposal = DeterministicStub().judge(text, occurred_at=snapshot.created_at)
        return ChatCallResult(
            safe_data(body),
            {"mock": True, "content": proposal.model_dump()},
            proposal,
            mock=True,
            metadata={"provider": "deterministic-stub"},
        )
