"""The live conversation: one Claude call per candidate turn."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .candidates import Candidate
from .config import Settings
from .prompts import candidate_context, live_system_prompt

log = logging.getLogger(__name__)

# Server-side refusal fallback: if the model declines a turn, the API re-runs
# it on Anthropic's recommended fallback model inside the same request.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "say": {"type": "string"},
        "end_call": {"type": "boolean"},
        "recording_consent": {"type": "string", "enum": ["unknown", "yes", "no"]},
        "current_question_id": {"type": "string"},
        "callback_request": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["say", "end_call", "recording_consent", "current_question_id", "callback_request", "note"],
    "additionalProperties": False,
}

REPEAT_LINE = "Sorry, could you say that again?"


@dataclass
class AgentTurn:
    say: str
    end_call: bool = False
    recording_consent: str = "unknown"
    current_question_id: str = ""
    callback_request: str = ""
    note: str = ""

    @classmethod
    def from_json(cls, text: str) -> "AgentTurn":
        data = json.loads(text)
        return cls(
            say=str(data.get("say", "")).strip(),
            end_call=bool(data.get("end_call", False)),
            recording_consent=str(data.get("recording_consent", "unknown")),
            current_question_id=str(data.get("current_question_id", "")),
            callback_request=str(data.get("callback_request", "")),
            note=str(data.get("note", "")),
        )


@dataclass
class Brain:
    settings: Settings
    candidate: Candidate
    client: Any = None  # anthropic.AsyncAnthropic; injectable for tests
    messages: list[dict[str, Any]] = field(default_factory=list)
    consent: str = "unknown"
    notes: list[str] = field(default_factory=list)
    callback_request: str = ""

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = anthropic.AsyncAnthropic()
        self._system = [
            # Same for every call -> cached across the whole calling session.
            {"type": "text", "text": live_system_prompt(self.settings), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": candidate_context(self.settings, self.candidate)},
        ]

    async def respond(self, user_text: str) -> AgentTurn:
        """Sends one candidate turn and returns what the agent says next.

        History is only appended once a reply arrives, so a cancelled call
        (e.g. the candidate kept talking) leaves no trace and can be retried
        with the combined text.
        """
        request_messages = self.messages + [{"role": "user", "content": user_text}]
        try:
            response = await self.client.beta.messages.create(
                model=self.settings.get("claude.live_model", "claude-opus-5-5"),
                max_tokens=4000,
                system=self._system,
                messages=request_messages,
                output_config={
                    "effort": self.settings.get("claude.live_effort", "low"),
                    "format": {"type": "json_schema", "schema": TURN_SCHEMA},
                },
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIConnectionError:
            log.exception("Claude connection error")
            return AgentTurn(say=REPEAT_LINE)
        except anthropic.RateLimitError:
            log.exception("Claude rate limit")
            return AgentTurn(say=REPEAT_LINE)
        except anthropic.APIStatusError as exc:
            log.error("Claude API error %s: %s", exc.status_code, exc.message)
            return AgentTurn(say=REPEAT_LINE)

        if response.stop_reason == "refusal":
            log.warning("Turn declined: %s", getattr(response, "stop_details", None))
            return AgentTurn(say=REPEAT_LINE)

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            turn = AgentTurn.from_json(text)
        except (json.JSONDecodeError, TypeError):
            log.error("Could not parse agent reply: %r", text[:300])
            return AgentTurn(say=REPEAT_LINE)

        # Keep the full content (including thinking blocks) so history stays valid.
        self.messages = request_messages + [{"role": "assistant", "content": response.content}]
        if turn.recording_consent in ("yes", "no"):
            self.consent = turn.recording_consent
        if turn.note:
            self.notes.append(f"[{turn.current_question_id}] {turn.note}")
        if turn.callback_request:
            self.callback_request = turn.callback_request
        return turn
