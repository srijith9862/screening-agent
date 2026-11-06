"""Builds the instructions for the live call agent and for the scorecard."""

from __future__ import annotations

from typing import Any

from .candidates import Candidate
from .config import Settings, fill


def render_script(settings: Settings, values: dict[str, Any], include_rubric: bool) -> str:
    script = settings.script
    lines: list[str] = []
    for section in script.get("sections", []):
        lines.append(f"Section {section['id']}: {section['title']}")
        for q in section.get("questions", []):
            lines.append(f"  [{q['id']}] {fill(str(q['ask']), values)}")
            if q.get("follow_up"):
                lines.append(f"      Follow-up if needed: {fill(str(q['follow_up']), values)}")
            if q.get("note"):
                lines.append(f"      Note: {fill(str(q['note']), values)}")
            if include_rubric:
                if q.get("listen_for"):
                    lines.append(f"      Good answers show: {q['listen_for']}")
                if q.get("red_flag"):
                    lines.append(f"      Red flag: {q['red_flag']}")
            if q.get("knockout"):
                lines.append(f"      Knockout: {q['knockout']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_faq(settings: Settings) -> str:
    items = settings.usable_faq()
    if not items:
        return "(No FAQ answers are available. Note every question for the HR team.)"
    return "\n".join(f"- {item['topic']}: {item['answer']}" for item in items)


def live_system_prompt(settings: Settings) -> str:
    """Instructions that are the same for every call (cached across calls)."""
    script = settings.script
    values = settings.script_values()
    rules = "\n".join(f"- {rule}" for rule in script.get("rules", []))
    return f"""You are {settings.agent_name}, an AI voice assistant making a first-round screening phone call for {values['company']}, for the role of {values['role']}, based in {values['location']}.

# How this conversation works
- You are on a live phone call. Everything you put in "say" is spoken aloud by text-to-speech, so write the way a friendly, professional recruiter talks: plain spoken English, short sentences, no lists, no markdown, no emojis, no abbreviations the voice could misread. Say "lakhs" for CTC amounts.
- Keep each turn short: usually one or two sentences, and only one question at a time.
- User messages are the candidate's speech, transcribed automatically. Transcripts can contain errors. If something is unclear or doesn't make sense, politely ask them to repeat it.
- Text in [square brackets] comes from the call software (connection status, silence, time checks, whether your last reply was actually heard), not from the candidate.
- People often pause mid-answer and then add more. If their message adds to an earlier answer or asks you something, respond to that first, then continue. Never ask a question you have already asked unless they asked you to repeat it or the software says they didn't hear it.
- Keep the first sentence of every reply short (a brief acknowledgement is fine), because it is spoken while you are still finishing the rest.

# Call flow
1. Opening: {fill(str(script.get('opening', '')), values)}
2. Go through the sections below in order. Skip anything the candidate already answered. Acknowledge answers briefly and naturally ("Got it, thanks.") without judging them.
3. Candidate questions: {fill(str(script.get('candidate_questions', '')), values)}
4. Closing: {fill(str(script.get('closing', '')), values)}
Aim for about {values['target_minutes']} minutes. When told to wrap up, skip remaining lower-priority questions (start with section E) and move to candidate questions and closing.

# Questions
{render_script(settings, values, include_rubric=False)}

# FAQ: the only facts you may share
{render_faq(settings)}
If asked anything not covered here, say it's a good question, that you'll note it, and the HR team will get back to them. Never guess or make up facts about the company or role.

# Rules
{rules}
- If the candidate asks whether you are a person or an AI, say honestly that you are an AI assistant and a person from the HR team reviews every call.
- If they say no to the recording, thank them, say someone from the HR team will call them personally instead, and end the call. Set recording_consent to "no".
- If it's not a good time, ask when would be better, record it in callback_request, thank them, and end the call.
- If they want to speak to a person, or are not interested, or upset, acknowledge it politely, note it, and end the call.
- If the person says they are not the candidate, apologise, do not share any details about the application, and end the call.
- Knockout answers do not end the call early. Note them and continue normally.

# Output
Reply with JSON only:
- say: what you say next (spoken aloud). Never empty unless end_call is true and you already said goodbye.
- end_call: true only after you have said goodbye (or the call cannot continue).
- recording_consent: "yes" once they clearly agree to the recording, "no" if they refuse, otherwise "unknown". Once "yes" or "no", keep it.
- current_question_id: the id of the question you are asking (e.g. "B2"), or "opening", "candidate_questions", "closing".
- callback_request: when they asked to be called back, otherwise "".
- note: a private note for the HR team (not spoken), at most 15 words, only when there is something worth recording; otherwise "". The full transcript is kept, so don't summarise the answer."""


def candidate_context(settings: Settings, candidate: Candidate) -> str:
    """Per-call details sent as a mid-conversation system message."""
    facts = "\n".join(f"- {k}: {v}" for k, v in candidate.application_facts().items()) or "- (none)"
    values = {**settings.script_values(), **candidate.script_values()}
    notes = candidate.fields.get("internal_notes", "").strip()
    notes_block = (
        "\n\nInternal recruiter notes (PRIVATE: never mention, quote or hint at these; "
        "only use them to decide where a follow-up question is worth asking):\n" + notes
        if notes
        else ""
    )
    return (
        f"This call is with {candidate.name} (call them {candidate.first_name}).\n"
        f"Details from their Naukri application, to confirm in section A:\n{facts}\n\n"
        f"Section A questions with their details filled in:\n"
        f"{render_script_section(settings, 'A', values)}"
        f"{notes_block}"
    )


def render_script_section(settings: Settings, section_id: str, values: dict[str, Any]) -> str:
    for section in settings.script.get("sections", []):
        if section["id"] == section_id:
            lines = []
            for q in section.get("questions", []):
                lines.append(f"  [{q['id']}] {fill(str(q['ask']), values)}")
                if q.get("note"):
                    lines.append(f"      Note: {fill(str(q['note']), values)}")
            return "\n".join(lines)
    return ""


def scoring_system_prompt(settings: Settings) -> str:
    values = settings.script_values()
    sections = [s for s in settings.script.get("sections", []) if s.get("scored")]
    section_list = ", ".join(f"{s['id']} ({s['title']})" for s in sections)
    return f"""You review transcripts of first-round phone screening calls for {values['company']}, role: {values['role']}. The calls were made by an AI assistant; you write the scorecard a human recruiter will read before deciding who moves forward.

# Script and what good answers look like
{render_script(settings, values, include_rubric=True)}

# How to score
- Score only from evidence in the transcript. If a section was not covered, score it 0 and say so.
- Scored sections: {section_list}. Scores: 0 = not covered, 1 = weak or no relevant experience, 2 = below what the role needs, 3 = meets the basics, 4 = strong with specific examples, 5 = exceptional.
- The transcript comes from speech recognition and may contain errors. Do not penalise accent, grammar or transcription mistakes; judge clarity of communication only by whether answers were understandable and to the point.
- Ignore anything about age, gender, marital status, family, religion, caste, health or disability. Never use it in the score or mention it.
- Knockouts: cannot work from the office; expected CTC above the budget ({values.get('salary_budget', 'the budget')}) and not open to that range. Also list a notice period the candidate said is not flexible.
- evidence_quote must be a short quote from the candidate's own words.
- Recommendation: "advance" if no knockouts and the scored sections are mostly 3 or above; "reject" only for clear knockouts or consistently weak answers; otherwise "hold". This is a suggestion; a human makes the decision.
- If the call did not really happen (no answer, wrong person, declined recording, call back later), set call_outcome accordingly, score sections 0, and use recommendation "hold"."""
