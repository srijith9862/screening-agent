"""Turns a call transcript into a structured scorecard."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from .brain import model_options
from .candidates import Candidate
from .config import Settings
from .prompts import scoring_system_prompt

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "section_id": {"type": "string"},
        "title": {"type": "string"},
        "score": {"type": "integer", "description": "0 to 5"},
        "reason": {"type": "string"},
        "evidence_quote": {"type": "string"},
    },
    "required": ["section_id", "title", "score", "reason", "evidence_quote"],
    "additionalProperties": False,
}

FACT_KEYS = [
    "hr_experience",
    "recruitment_experience",
    "office_commute_ok",
    "notice_period",
    "notice_flexibility",
    "current_ctc",
    "expected_ctc",
    "within_budget",
    "ctc_flexibility",
]

SCORECARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "call_outcome": {
            "type": "string",
            "enum": [
                "completed",
                "partial",
                "declined_recording",
                "call_back_later",
                "not_interested",
                "wrong_person",
                "dropped",
                "no_answer",
            ],
        },
        "recording_consent": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "confirmed_facts": {
            "type": "object",
            "description": "What the candidate said on the call; empty string if not discussed.",
            "properties": {key: {"type": "string"} for key in FACT_KEYS},
            "required": FACT_KEYS,
            "additionalProperties": False,
        },
        "sections": {"type": "array", "items": SECTION_SCHEMA},
        "knockouts": {"type": "array", "items": {"type": "string"}},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "unanswered_candidate_questions": {"type": "array", "items": {"type": "string"}},
        "callback_request": {"type": "string"},
        "recommendation": {"type": "string", "enum": ["advance", "hold", "reject"]},
        "recommendation_reason": {"type": "string"},
        "summary": {"type": "string", "description": "3-5 sentence summary for the recruiter"},
    },
    "required": [
        "call_outcome",
        "recording_consent",
        "confirmed_facts",
        "sections",
        "knockouts",
        "strengths",
        "red_flags",
        "unanswered_candidate_questions",
        "callback_request",
        "recommendation",
        "recommendation_reason",
        "summary",
    ],
    "additionalProperties": False,
}


def format_transcript(turns: list[dict[str, Any]], agent_name: str) -> str:
    lines = []
    for turn in turns:
        stamp = f"{int(turn.get('t', 0)) // 60:02d}:{int(turn.get('t', 0)) % 60:02d}"
        speaker = {"agent": agent_name, "candidate": "Candidate", "system": "Call software"}.get(
            turn["speaker"], turn["speaker"]
        )
        lines.append(f"[{stamp}] {speaker}: {turn['text']}")
    return "\n".join(lines)


def build_scoring_message(candidate: Candidate, transcript: str, agent_notes: list[str]) -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in candidate.application_facts().items()) or "- (none)"
    notes = "\n".join(f"- {n}" for n in agent_notes) or "- (none)"
    return (
        f"Candidate: {candidate.name}\n\n"
        f"Naukri application details (compare with what they said on the call):\n{facts}\n\n"
        f"Notes the call agent made during the call:\n{notes}\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        "Write the scorecard."
    )


async def score_call(
    settings: Settings,
    candidate: Candidate,
    transcript: str,
    agent_notes: list[str],
    client: Any = None,
) -> dict[str, Any]:
    client = client or anthropic.AsyncAnthropic()
    model = settings.get("claude.score_model", "claude-sonnet-5-5")
    async with client.beta.messages.stream(
        model=model,
        max_tokens=32000,
        system=scoring_system_prompt(settings),
        messages=[{"role": "user", "content": build_scoring_message(candidate, transcript, agent_notes)}],
        output_config={
            "effort": settings.get("claude.score_effort", "medium"),
            "format": {"type": "json_schema", "schema": SCORECARD_SCHEMA},
        },
        **model_options(model),
    ) as stream:
        response = await stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError(f"Scoring was declined: {getattr(response, 'stop_details', None)}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Scorecard was cut off (max_tokens). Try again.")
    text = next((b.text for b in response.content if b.type == "text"), "")
    return json.loads(text)


def scorecard_markdown(candidate: Candidate, card: dict[str, Any]) -> str:
    lines = [
        f"# {candidate.name}",
        "",
        f"**Recommendation:** {card['recommendation'].upper()} - {card['recommendation_reason']}",
        f"**Call outcome:** {card['call_outcome']}  |  **Recording consent:** {card['recording_consent']}",
        "",
        card["summary"],
        "",
        "## Scores",
        "",
        "| Section | Score | Why | Evidence |",
        "|---|---|---|---|",
    ]
    for s in card["sections"]:
        lines.append(
            f"| {s['section_id']} {s['title']} | {s['score']}/5 | {s['reason']} | \"{s['evidence_quote']}\" |"
        )
    lines += ["", "## Confirmed on the call", ""]
    lines += [f"- **{k.replace('_', ' ')}:** {v or '-'}" for k, v in card["confirmed_facts"].items()]
    for title, key in [
        ("Knockouts", "knockouts"),
        ("Strengths", "strengths"),
        ("Red flags", "red_flags"),
        ("Questions to answer personally", "unanswered_candidate_questions"),
    ]:
        lines += ["", f"## {title}", ""]
        lines += [f"- {item}" for item in card[key]] or ["- none"]
    if card.get("callback_request"):
        lines += ["", f"**Call back:** {card['callback_request']}"]
    return "\n".join(lines) + "\n"
