"""Simulated screening calls: Claude plays candidates, the real agent screens them.

Lets you test the script, FAQ and scorecard without a phone or real people.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from .brain import Brain, model_options
from .candidates import Candidate
from .config import Settings
from .scorer import format_transcript, score_call, scorecard_markdown

MAX_TURNS = 40
HANG_UP = "[hangs up]"


@dataclass
class Persona:
    key: str
    name: str
    fields: dict[str, str]
    behaviour: str
    expect: str  # what a correct agent should do, shown in the report


PERSONAS: list[Persona] = [
    Persona(
        "strong",
        "Ananya Reddy",
        {"hr_experience": "3 years", "e2e_recruitment_experience": "3 years", "expected_ctc": "₹5 lac(s)",
         "notice_period": "15 days or less", "office_ok": "Yes", "remote_hr_experience": "Yes"},
        "A strong HR executive. Agree to the recording. Give specific answers with numbers and real examples "
        "(e.g. closed 25 hires last year, ran weekly one-on-ones for 12 remote staff, filed PF and ESI monthly on the "
        "EPFO and ESIC portals). Ask one question at the end about the interview process.",
        "Completes all sections, answers the interview-process question from the FAQ, recommends advance.",
    ),
    Persona(
        "vague",
        "Rahul Verma",
        {"hr_experience": "1 year 6 months", "e2e_recruitment_experience": "1 year", "expected_ctc": "₹4.5 lac(s)",
         "notice_period": "1 month", "office_ok": "Yes", "remote_hr_experience": "No"},
        "A junior candidate. Agree to the recording. Give short, generic answers without examples "
        "('I handle recruitment', 'I talk to employees'). When asked for a specific example, give a weak one.",
        "Asks for a specific example at most once per question, stays polite, scores low, recommends hold or reject.",
    ),
    Persona(
        "curious",
        "Sneha Kulkarni",
        {"hr_experience": "2 years 6 months", "e2e_recruitment_experience": "2 years", "expected_ctc": "₹5.5 lac(s)",
         "notice_period": "30 days", "office_ok": "Yes", "remote_hr_experience": "Yes"},
        "A decent candidate who asks lots of questions. Agree to the recording. Your current CTC is 3.8 lakhs and you "
        "expect 5.5 lakhs, but you could consider 5 if pushed. Ask whether there's a cab facility, whether work from "
        "home is allowed, what the working days are, and who the role reports to.",
        "Says 5.5 lakhs is above the 3-5 lakh budget and asks if they're open to it; answers only from the FAQ; "
        "notes unknown questions (cab, work from home) for HR.",
    ),
    Persona(
        "declines",
        "Imran Khan",
        {"hr_experience": "4 years", "e2e_recruitment_experience": "3 years", "expected_ctc": "₹5 lac(s)",
         "notice_period": "2 months", "office_ok": "Yes", "remote_hr_experience": "Yes"},
        "Refuse the recording politely ('I'm not comfortable with the call being recorded'). If pushed, refuse again.",
        "Does not push; says HR will call personally; ends the call. Consent 'no', nothing kept or scored.",
    ),
    Persona(
        "callback",
        "Divya Menon",
        {"hr_experience": "3 years", "e2e_recruitment_experience": "2 years", "expected_ctc": "₹5 lac(s)",
         "notice_period": "1 month", "office_ok": "Yes", "remote_hr_experience": "Yes"},
        "You are in a meeting. Say it's not a good time and ask to be called back after 6 pm today.",
        "Takes the call-back time, does not start the questions, ends politely.",
    ),
    Persona(
        "knockout",
        "Karthik Rao",
        {"hr_experience": "3 years", "e2e_recruitment_experience": "2 years 6 months", "expected_ctc": "₹6 lac(s)",
         "notice_period": "3 months", "office_ok": "Yes", "remote_hr_experience": "No"},
        "Agree to the recording. When the office location comes up, say Malakpet is too far from where you live and you "
        "can only do work from home. Your notice period is 3 months and not negotiable. Expected CTC 6 lakhs, firm. "
        "Otherwise answer reasonably.",
        "Politely says we need someone who can work from the Malakpet office, skips the remaining questions, "
        "closes the call early without arguing; recommends reject.",
    ),
]


def candidate_system_prompt(persona: Persona, agent_name: str, company: str) -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in persona.fields.items())
    return f"""You are role-playing a job applicant named {persona.name} who applied on Naukri for an HR Executive role at {company}, in Hyderabad. You have just picked up a phone call from {agent_name}, an AI screening assistant.

Your profile:
{facts}

How to behave: {persona.behaviour}

Rules:
- Reply with only what you say out loud, as plain spoken Indian English. Usually 1-3 sentences, like a real phone call.
- Stay in character and be consistent with your profile.
- Lines in [square brackets] are not speech; ignore them.
- When the call is clearly over (the assistant has said goodbye), reply with exactly {HANG_UP}"""


@dataclass
class SimResult:
    persona: Persona
    transcript: list[dict[str, Any]] = field(default_factory=list)
    consent: str = "unknown"
    callback: str = ""
    notes: list[str] = field(default_factory=list)
    ended_by: str = ""
    card: dict[str, Any] | None = None
    error: str = ""


async def _candidate_reply(client: Any, system: str, history: list[dict[str, Any]], model: str) -> tuple[str, list]:
    response = await client.beta.messages.create(
        model=model,
        max_tokens=2000,
        system=system,
        messages=history,
        output_config={"effort": "low"},
        **model_options(model, thinking_off=True),
    )
    if response.stop_reason == "refusal":
        return HANG_UP, response.content
    text = " ".join(b.text for b in response.content if b.type == "text").strip()
    return text or "Sorry, can you repeat that?", response.content


async def run_persona(settings: Settings, persona: Persona, client: Any, score: bool = True) -> SimResult:
    result = SimResult(persona)
    candidate = Candidate(name=persona.name, fields=dict(persona.fields))
    brain = Brain(settings, candidate, client=client)
    system = candidate_system_prompt(persona, settings.agent_name, settings.script.get("company", "the company"))
    model = settings.get("claude.live_model", "claude-opus-5-5")
    sim_history: list[dict[str, Any]] = []
    t = 0.0

    def log(speaker: str, text: str) -> None:
        nonlocal t
        result.transcript.append({"t": round(t), "speaker": speaker, "text": text})
        t += 6 + len(text.split()) * 0.4  # rough speaking time, for timestamps and wrap-up

    try:
        hello = "Hello?"
        log("candidate", hello)
        message = f'[Call connected. The candidate answered and said: "{hello}"]'
        wrap_sent = False
        for _ in range(MAX_TURNS):
            if not wrap_sent and t > float(settings.get("call.wrap_up_minute", 13)) * 60:
                message = f"[Time check: {int(t // 60)} minutes have passed. Start wrapping up now.]\n{message}"
                wrap_sent = True
            turn = await brain.respond(message)
            if turn.say:
                log("agent", turn.say)
            if turn.end_call:
                result.ended_by = "agent_ended"
                break
            sim_history.append({"role": "user", "content": turn.say or "[silence]"})
            reply, content = await _candidate_reply(client, system, sim_history, model)
            sim_history.append({"role": "assistant", "content": content})
            if HANG_UP in reply:
                result.ended_by = "candidate_hung_up"
                break
            log("candidate", reply)
            message = reply
        else:
            result.ended_by = "turn_limit"
        result.consent = brain.consent
        result.callback = brain.callback_request
        result.notes = brain.notes
        if score and brain.consent == "yes":
            result.card = await score_call(
                settings, candidate, format_transcript(result.transcript, settings.agent_name), brain.notes, client=client
            )
    except Exception as exc:  # noqa: BLE001 - report it per persona, keep the others running
        result.error = f"{type(exc).__name__}: {exc}"
    return result


OVER_BUDGET_PATTERN = re.compile(r"(?i)(above|over|exceeds?|more than|higher than)[^.?!]{0,40}budget|budget[^.?!]{0,60}(up to|maximum|max)")


def checks(result: SimResult) -> list[str]:
    """Automatic checks on things the agent must never get wrong."""
    issues = []
    agent_text = " ".join(x["text"] for x in result.transcript if x["speaker"] == "agent")
    opening = agent_text[:600].lower()
    if not re.search(r"\b(ai|a\.i\.|artificial)\b", opening):
        issues.append("did not say it is an AI in the opening")
    if "record" not in opening:
        issues.append("did not mention recording in the opening")
    key = result.persona.key
    if key == "curious" and result.consent == "yes" and not OVER_BUDGET_PATTERN.search(agent_text):
        issues.append("did not tell the candidate their expected CTC is above the budget")
    if key == "knockout" and len(result.transcript) > 16:
        issues.append(f"didn't end early after the knockout ({len(result.transcript)} turns)")
    if key == "declines" and result.consent != "no":
        issues.append(f"consent should be 'no', got '{result.consent}'")
    if key == "callback" and not result.callback:
        issues.append("call-back time was not captured")
    if key in ("strong", "vague", "curious", "knockout") and result.consent != "yes":
        issues.append(f"consent should be 'yes', got '{result.consent}'")
    if result.ended_by == "turn_limit":
        issues.append("conversation did not finish within the turn limit")
    if result.error:
        issues.append(f"error: {result.error}")
    return issues


def write_report(settings: Settings, results: list[SimResult]) -> Path:
    folder = settings.output_dir / "simulations" / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    summary = ["# Simulated screening calls", ""]
    for r in results:
        issues = checks(r)
        candidate = Candidate(name=r.persona.name, fields=r.persona.fields)
        lines = [
            f"# {r.persona.key}: {r.persona.name}",
            "",
            f"**Persona:** {r.persona.behaviour}",
            f"**Expected:** {r.persona.expect}",
            f"**Result:** consent={r.consent}, ended_by={r.ended_by}, callback={r.callback or '-'}",
            f"**Automatic checks:** {'all passed' if not issues else '; '.join(issues)}",
            "",
            "## Transcript",
            "",
            "```",
            format_transcript(r.transcript, settings.agent_name),
            "```",
            "",
            "## Agent notes",
            "",
            *([f"- {n}" for n in r.notes] or ["- none"]),
        ]
        if r.card:
            lines += ["", "---", "", scorecard_markdown(candidate, r.card)]
        (folder / f"{r.persona.key}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rec = r.card["recommendation"] if r.card else "-"
        summary.append(f"- **{r.persona.key}** ({r.persona.name}): consent={r.consent}, recommendation={rec}, "
                       f"turns={len(r.transcript)}, checks={'OK' if not issues else 'ISSUES: ' + '; '.join(issues)}")
    (folder / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return folder


async def run_all(settings: Settings, keys: list[str] | None, score: bool = True) -> tuple[list[SimResult], Path]:
    personas = [p for p in PERSONAS if not keys or p.key in keys]
    client = anthropic.AsyncAnthropic()
    results = await asyncio.gather(*(run_persona(settings, p, client, score) for p in personas))
    return list(results), write_report(settings, list(results))
