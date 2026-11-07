import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from screener.audio import Player, Recorder
from screener.brain import REPEAT_LINE, Brain
from screener.candidates import Candidate, load_candidates, normalize_phone
from screener.config import TimeWindow, fill, load_settings
from screener.phone import count_telecom_active, parse_call_state
from screener.prompts import candidate_context, live_system_prompt, scoring_system_prompt
from screener.scorer import SCORECARD_SCHEMA, format_transcript, scorecard_markdown
from screener.speech import split_sentences
from screener import storage

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings():
    return load_settings()


# --- candidates ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("9876543210", "+919876543210"),
        ("+91 98765 43210", "+919876543210"),
        ("09876543210", "+919876543210"),
        ("919876543210", "+919876543210"),
        ("0091 9876543210", "+919876543210"),
        ("12345", ""),
        ("", ""),
        ("+91 9876543210 (WhatsApp: +91 9123456789)", "+919876543210"),
        ("Not shown on resume", ""),
        ("Not shown on resume (resume shows only partial 987654321)", ""),
    ],
)
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_load_naukri_style_csv(tmp_path):
    csv_path = tmp_path / "c.csv"
    csv_path.write_text(
        "Name,Category,Rating_out_of_10,HR_experience,E2E_recruitment_experience,Expected_CTC,"
        "Notice_period,Malakpet_office,Reason_strengths_and_gaps,Phone Number\n"
        '"Asha Rao","Top fit","8","3 years","2 years","₹5 lac(s)","15 days or less","Yes","Gap: no PF","98765 43210"\n'
        '"No Phone","Okish","5","1 year","1 year","₹4 lac(s)","1 month","Yes","",""\n',
        encoding="utf-8",
    )
    people = load_candidates(csv_path)
    assert [p.name for p in people] == ["Asha Rao", "No Phone"]
    asha = people[0]
    assert asha.phone == "+919876543210"
    assert asha.first_name == "Asha"
    assert asha.fields["notice_period"] == "15 days or less"
    # Ratings are never passed to the AI.
    assert "Top fit" not in json.dumps(asha.fields)
    assert people[1].phone == ""


def test_sample_csv_loads():
    people = load_candidates(ROOT / "data" / "sample_candidates.csv")
    assert people[0].phone == "+919999999999"


# --- config and prompts ---------------------------------------------------------


def test_local_settings_override(tmp_path):
    import shutil

    config = tmp_path / "config"
    shutil.copytree(ROOT / "config", config)
    (config / "settings.local.yaml").write_text("phone:\n  adb_path: 'C:\\x\\adb.exe'\n", encoding="utf-8")
    s = load_settings(config)
    assert s.get("phone.adb_path") == "C:\\x\\adb.exe"
    assert s.get("phone.default_country_code") == "+91"  # other keys kept


def test_fill_leaves_unknown_placeholders():
    assert fill("Hi {first_name}, {missing}", {"first_name": "Asha"}) == "Hi Asha, {missing}"


def test_todo_faq_entries_are_hidden(settings):
    settings.faq.append({"topic": "Cab facility", "answer": "TODO confirm with admin"})
    topics = [item["topic"] for item in settings.usable_faq()]
    assert "Office location" in topics and "Cab facility" not in topics
    assert settings.todo_warnings() == ["faq.yaml: 'Cab facility' still has a TODO answer"]


def test_shipped_config_is_complete():
    assert load_settings().todo_warnings() == []


def test_prompts_render(settings):
    system = live_system_prompt(settings)
    assert "Datasphere Solutions" in system and "[C2]" in system
    assert "TODO" not in system.split("# FAQ")[1].split("# Rules")[0]
    candidate = Candidate(name="Asha Rao", fields={"notice_period": "1 month", "internal_notes": "budget 3-5L"})
    ctx = candidate_context(settings, candidate)
    assert "Asha" in ctx and "1 month" in ctx and "PRIVATE" in ctx
    assert "Good answers show" in scoring_system_prompt(settings)


def test_voice_endpoints_and_wav_decoding():
    import io
    import wave

    from screener.speech import DG_SPEAK_URL, DG_SPEAK_V2_URL, decode_pcm16, speak_request

    url, params = speak_request("flux-priya-en")
    assert url == DG_SPEAK_V2_URL and params["model"] == "flux-priya-en" and "container" not in params
    url, params = speak_request("aura-2-thalia-en")
    assert url == DG_SPEAK_URL and params["container"] == "none"

    raw = (np.array([0, 16384, -16384], dtype="<i2")).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(raw)
    assert np.allclose(decode_pcm16(buf.getvalue()), [0, 0.5, -0.5])
    assert np.allclose(decode_pcm16(raw), [0, 0.5, -0.5])


def test_calling_hours():
    window = TimeWindow.parse("10:00-19:00")
    assert window.contains(10, 0) and window.contains(18, 59)
    assert not window.contains(19, 0) and not window.contains(9, 59)


# --- phone ---------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="uses a shell-script stand-in for adb")
def test_adb_output_with_non_ascii_bytes(tmp_path):
    from screener.phone import Phone

    fake = tmp_path / "adb"
    fake.write_text("#!/bin/sh\nprintf 'state: ACTIVE \\342\\200\\235 caf\\351\\n'\n")
    fake.chmod(0o755)
    out = Phone(str(fake))._run("shell", "dumpsys", "telecom")
    assert out.startswith("state: ACTIVE") and count_telecom_active(out) == 1


def test_find_bluetooth_button_in_ui_dump():
    from screener.phone import BLUETOOTH_PATTERN, find_node, parse_ui_nodes

    xml = (
        "UI hierchary dumped to: /sdcard/screener_ui.xml\n"
        "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><hierarchy rotation=\"0\">"
        '<node text="+91 90591 99427" content-desc="" clickable="false" bounds="[0,200][1080,300]" />'
        '<node text="" content-desc="Speaker" clickable="true" checked="false" bounds="[100,1500][300,1700]" />'
        '<node text="Bluetooth" content-desc="" clickable="false" bounds="[420,1660][660,1700]" />'
        '<node text="" content-desc="Bluetooth" clickable="true" checked="true" bounds="[400,1500][680,1700]" />'
        '<node text="Srijith9862" content-desc="" clickable="true" bounds="[0,800][1080,900]" />'
        "</hierarchy>"
    )
    nodes = parse_ui_nodes(xml)
    button = find_node(nodes, BLUETOOTH_PATTERN)
    assert button is not None and button.clickable and button.selected
    assert button.center == (540, 1600)
    laptop = find_node(nodes, re.compile("Srijith9862", re.I))
    assert laptop is not None and laptop.center == (540, 850)
    assert parse_ui_nodes("ERROR: could not get idle state.") == []


def test_phonelink_script_is_built_safely():
    from screener.phonelink import build_script

    script = build_script("+919121763791", timeout_s=8)
    assert "'+919121763791'" in script and "AddSeconds(8)" in script and "__" not in script
    with pytest.raises(ValueError):
        build_script("123; Remove-Item C:\\")


def test_parse_call_state():
    assert parse_call_state("mCallState=0\n  mCallState=2\n") == 2
    assert parse_call_state("nothing") == 0
    assert count_telecom_active("Call TC@1 state: ACTIVE\nSET_ACTIVE\nstate=DIALING") == 1


# --- audio ---------------------------------------------------------------------


def test_pick_device_prefers_name_prefix():
    from screener.audio import pick_device

    devices = [
        {"name": "Speakers (Realtek)", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Hi-Fi Cable Output (VB-Audio Hi-Fi Cable)", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Hi-Fi Cable Input (VB-Audio Hi-Fi Cable)", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_input_channels": 0, "max_output_channels": 2},
    ]
    assert pick_device("CABLE Output", devices, "input") == 2
    assert pick_device("CABLE Input", devices, "output") == 4
    assert pick_device("Hi-Fi Cable Input", devices, "output") == 3
    assert pick_device("Realtek", devices, "output") == 0  # substring fallback
    assert pick_device("CABLE Output", devices, "output") is None


def test_player_pull_and_clear():
    player = Player()
    player.enqueue(np.ones(30, np.float32))
    out = player.pull(20)
    assert out.sum() == 20 and not player.is_idle()
    player.clear()
    assert player.is_idle() and player.pull(5).sum() == 0


def test_recorder_save_and_discard(tmp_path):
    rec = Recorder()
    rec.add_candidate(np.zeros(1600, np.float32))
    rec.add_agent(np.zeros(800, np.float32))
    assert rec.save(tmp_path / "r.wav")
    rec.discard()
    rec.add_candidate(np.zeros(10, np.float32))
    assert not rec.save(tmp_path / "r2.wav")


def test_split_sentences():
    assert split_sentences("Got it, thanks. How many hires? Great!") == ["Got it, thanks.", "How many hires?", "Great!"]


# --- brain with a fake Claude client ---------------------------------------------


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, str) and reply == "refusal":
            return SimpleNamespace(stop_reason="refusal", stop_details=None, content=[])
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(reply))],
        )


def _turn(**overrides):
    base = {
        "say": "Hi Asha, this is Maya.",
        "end_call": False,
        "recording_consent": "unknown",
        "current_question_id": "opening",
        "callback_request": "",
        "note": "",
    }
    base.update(overrides)
    return base


def test_brain_turns_and_consent(settings):
    fake = FakeMessages([_turn(), _turn(recording_consent="yes", note="ok"), "refusal"])
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    brain = Brain(settings, Candidate(name="Asha Rao"), client=client)

    first = asyncio.run(brain.respond("[Call connected]"))
    assert first.say.startswith("Hi Asha") and brain.consent == "unknown"
    asyncio.run(brain.respond("Yes that's fine"))
    assert brain.consent == "yes" and brain.notes
    assert len(brain.messages) == 4  # user, assistant, user, assistant

    third = asyncio.run(brain.respond("something"))
    assert third.say == REPEAT_LINE
    assert len(brain.messages) == 4  # a failed turn leaves history unchanged

    request = fake.calls[0]
    assert request["model"] == "claude-haiku-5-5"
    assert request["thinking"] == {"type": "disabled"} and "fallbacks" not in request
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["output_config"]["format"]["type"] == "json_schema"


def test_model_options_per_model():
    from screener.brain import model_options

    assert model_options("claude-haiku-5-5") == {}
    assert model_options("claude-haiku-5-5", thinking_off=True) == {"thinking": {"type": "disabled"}}
    assert model_options("claude-sonnet-5-5")["fallbacks"] == "default"
    assert model_options("claude-sonnet-5-5", thinking_off=True)["thinking"] == {"type": "between_tools"}
    assert "thinking" not in model_options("claude-opus-5-5", thinking_off=True)  # can't be disabled


def test_simulation_loop_and_checks(settings, tmp_path):
    from screener.simulate import PERSONAS, checks, run_persona, write_report

    class SimMessages:
        """Agent calls (with a JSON schema) get scripted turns; candidate calls get canned speech."""

        def __init__(self):
            self.agent = [
                _turn(say="Hi, I'm Priya, an AI assistant from Datasphere. This call is recorded. Is that okay?"),
                _turn(say="Thanks. Sorry to bother you, goodbye.", recording_consent="no", end_call=True),
            ]

        async def create(self, **kwargs):
            if "format" in kwargs.get("output_config", {}):
                reply = json.dumps(self.agent.pop(0))
            else:
                reply = "No, I'm not comfortable with recording."
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=reply)])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimMessages()))
    persona = next(p for p in PERSONAS if p.key == "declines")
    result = asyncio.run(run_persona(settings, persona, client, score=False))
    assert result.consent == "no" and result.ended_by == "agent_ended" and not result.error
    assert checks(result) == []
    assert [x["speaker"] for x in result.transcript] == ["candidate", "agent", "candidate", "agent"]

    settings.raw.setdefault("storage", {})["output_dir"] = str(tmp_path)
    folder = write_report(settings, [result])
    assert (folder / "SUMMARY.md").exists() and (folder / "declines.md").exists()


def test_over_budget_check():
    from screener.simulate import OVER_BUDGET_PATTERN

    from screener.simulate import CEILING_PATTERN

    assert OVER_BUDGET_PATTERN.search("That's a bit above what we've budgeted for this role.")
    assert CEILING_PATTERN.search("Our budget goes up to 5 lakhs.")
    assert CEILING_PATTERN.search("The range is three to five lakhs.")
    assert not CEILING_PATTERN.search("That's well within our range, and there's room for a raise.")
    assert OVER_BUDGET_PATTERN.search("That exceeds our budget.")
    assert not OVER_BUDGET_PATTERN.search("That fits within our budget.")


def test_salary_question_gets_candidate_figures(settings):
    candidate = Candidate(name="Asha Rao", fields={"current_ctc": "₹3 lac(s)", "expected_ctc": "₹6 lac(s)"})
    ctx = candidate_context(settings, candidate)
    assert "3 to 5 lakhs per annum" in ctx and "₹3 lac(s)" in ctx and "₹6 lac(s)" in ctx
    assert "3 to 5 lakhs" in live_system_prompt(settings)  # FAQ and question note
    assert "Never reveal the salary budget" not in live_system_prompt(settings)


def test_partial_say_while_streaming():
    from screener.brain import partial_say

    assert partial_say('{"sa') == ""
    assert partial_say('{"say": "Got it, thanks. How many hi') == "Got it, thanks. How many hi"
    assert partial_say('{"say": "She said \\"hi\\"') == 'She said "hi"'
    assert partial_say('{"say": "ends with escape \\') == "ends with escape "
    assert partial_say('{"say": "Done.", "end_call": false') == "Done."


def test_complete_sentences_end():
    from screener.call_session import complete_sentences_end

    assert complete_sentences_end("Got it, thanks. How many hi") == len("Got it, thanks. ")
    assert complete_sentences_end("No full stop yet") == 0
    assert complete_sentences_end("One. Two? Three") == len("One. Two? ")


def _fake_session(settings, brain):
    """A CallSession with fake speech, voice, audio and phone, for turn-taking tests."""
    from screener.call_session import CallSession

    session = object.__new__(CallSession)
    session.s = settings
    session.candidate = Candidate(name="Ruthwik Reddy")
    session.phone = None
    session.echo = False
    session.recorder = Recorder()
    session.transcript = []
    session._t0 = 0.0
    session.show_steps = False
    session.brain = brain
    session.stt = SimpleNamespace(events=asyncio.Queue())

    class FakeTTS:
        async def synthesize(self, text):
            await asyncio.sleep(0.01)
            return np.zeros(10, np.float32)

    session.tts = FakeTTS()

    async def wait_until_played():
        return None

    session.engine = SimpleNamespace(
        player=SimpleNamespace(is_idle=lambda: True, clear=lambda: None),
        play=lambda audio, rate: None,
        wait_until_played=wait_until_played,
        input_level=0.0,
    )
    return session


class ScriptedBrain:
    def __init__(self, turns, think_s=0.3):
        from screener.brain import AgentTurn

        self.turns = [AgentTurn(**t) for t in turns]
        self.think_s = think_s
        self.messages = []
        self.consent = "unknown"
        self.notes = []
        self.callback_request = ""

    async def respond(self, message, on_say=None):
        self.messages.append(message)
        await asyncio.sleep(self.think_s)
        turn = self.turns.pop(0)
        if on_say:
            on_say(turn.say)
        return turn


def test_candidate_adding_to_answer_gets_one_reply(settings):
    """'Yeah, that's correct.' <pause> 'And it's three years two months.' -> one combined reply."""
    from screener.speech import SpeechEvent

    brain = ScriptedBrain([
        {"say": "Hi Ruthwik, is that experience correct?"},
        {"say": "Thanks, noted three years two months. Does the commute work?"},
        {"say": "Great, goodbye.", "end_call": True},
    ])
    session = _fake_session(settings, brain)

    async def scenario():
        conversation = asyncio.create_task(session._conversation("[Call connected]"))
        await asyncio.sleep(0.5)  # opening reply spoken
        await session.stt.events.put(SpeechEvent("turn_end", "Yeah, that's correct."))
        await asyncio.sleep(0.1)  # agent is thinking...
        await session.stt.events.put(SpeechEvent("partial", "And it's actually"))
        await asyncio.sleep(0.1)
        await session.stt.events.put(SpeechEvent("turn_end", "And it's actually three years two months."))
        await asyncio.sleep(0.6)
        await session.stt.events.put(SpeechEvent("turn_end", "Yes it works."))
        return await asyncio.wait_for(conversation, timeout=5)

    ended_by = asyncio.run(scenario())
    assert ended_by == "agent_ended"
    # The first reply attempt was dropped; the retry saw both fragments together.
    assert "Yeah, that's correct. And it's actually three years two months." in brain.messages[2]
    agent_lines = [t["text"] for t in session.transcript if t["speaker"] == "agent"]
    assert agent_lines == [
        "Hi Ruthwik, is that experience correct?",
        "Thanks, noted three years two months. Does the commute work?",
        "Great, goodbye.",
    ]


def test_pickup_greeting_detection():
    from screener.call_session import is_pickup_greeting

    for text in ["Hello?", "Hello. Hello?", "Hi.", "Yes, tell me.", "Haan ji.", "Who is this?", "Yeah, speaking."]:
        assert is_pickup_greeting(text), text
    for text in ["", "Tum hi ho, ab tum hi ho, zindagi ab tum hi ho", "The number you have dialled is busy"]:
        assert not is_pickup_greeting(text), text


def test_filler_and_trailing_off_detection():
    from screener.call_session import is_filler_only, trails_off

    for text in ["Umm.", "Hmm...", "Uh, um.", "Mm.", "Let me think.", "So...", "Hmm, let me think."]:
        assert is_filler_only(text), text
    for text in ["Um, I'd say about 50 hires.", "Yes.", "Three years.", "Okay.", "Yeah.", "Mm-hmm.", "Uh-huh."]:
        assert not is_filler_only(text), text
    assert trails_off("I posted it on Naukri and")
    assert trails_off("We did it because um")
    assert not trails_off("We closed fifty hires last year.")


def test_thinking_pause_does_not_trigger_a_reply(settings):
    """'Umm.' <pause> 'I think about fifty.' -> one reply with the real answer."""
    from screener.speech import SpeechEvent

    settings.raw.setdefault("call", {})["thinking_wait_s"] = 0.8
    brain = ScriptedBrain([
        {"say": "How many hires did you close?"},
        {"say": "Thanks. Goodbye.", "end_call": True},
    ], think_s=0.1)
    session = _fake_session(settings, brain)

    async def scenario():
        conversation = asyncio.create_task(session._conversation("[Call connected]"))
        await asyncio.sleep(0.3)
        await session.stt.events.put(SpeechEvent("turn_end", "Umm."))
        await asyncio.sleep(0.5)  # thinking...
        assert len(brain.messages) == 1  # no reply to "Umm."
        await session.stt.events.put(SpeechEvent("turn_end", "I think about fifty."))
        return await asyncio.wait_for(conversation, timeout=5)

    asyncio.run(scenario())
    assert brain.messages[1] == "Umm. I think about fifty."


def test_after_premade_greeting_agent_waits_for_candidate(settings):
    from screener.speech import SpeechEvent

    brain = ScriptedBrain([
        {"say": "Hi Rajitha, I'm Priya, an AI assistant. This call is recorded. Is that okay?"},
        {"say": "Thanks, goodbye.", "end_call": True},
    ], think_s=0.1)
    session = _fake_session(settings, brain)

    async def scenario():
        conversation = asyncio.create_task(session._conversation("[Call connected. You already greeted them.]", reply_now=False))
        await asyncio.sleep(0.4)
        assert brain.messages == []  # no Claude call until they reply
        await session.stt.events.put(SpeechEvent("turn_end", "Yes, speaking."))
        await asyncio.sleep(0.5)
        await session.stt.events.put(SpeechEvent("turn_end", "Yes, that's fine."))
        return await asyncio.wait_for(conversation, timeout=5)

    asyncio.run(scenario())
    assert brain.messages[0] == "[Call connected. You already greeted them.]\nYes, speaking."


def test_repeated_hello_does_not_restart_the_reply(settings):
    from screener.speech import SpeechEvent

    brain = ScriptedBrain([
        {"say": "Hi, can you hear me? This is Priya."},
        {"say": "Bye.", "end_call": True},
    ], think_s=0.5)
    session = _fake_session(settings, brain)

    async def scenario():
        conversation = asyncio.create_task(session._conversation("[greeted]", reply_now=False))
        await session.stt.events.put(SpeechEvent("turn_end", "Hello?"))
        await asyncio.sleep(0.2)
        await session.stt.events.put(SpeechEvent("turn_end", "Hello?"))
        await asyncio.sleep(0.2)
        await session.stt.events.put(SpeechEvent("turn_end", "Hello?"))
        await asyncio.sleep(0.6)
        assert any(t["speaker"] == "agent" for t in session.transcript)  # answered within ~0.5 s of the first
        await session.stt.events.put(SpeechEvent("turn_end", "Okay bye."))
        return await asyncio.wait_for(conversation, timeout=5)

    asyncio.run(scenario())
    assert brain.messages[0] == "[greeted]\nHello?"


def test_recent_attempts_are_not_redialled(tmp_path, settings):
    from datetime import datetime, timedelta

    settings.raw.setdefault("storage", {})["output_dir"] = str(tmp_path)
    now = datetime.now()
    storage.append_result(settings, {"phone": "+911", "outcome": "no_answer", "called_at": now.strftime("%Y-%m-%d %H:%M")})
    old = (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
    storage.append_result(settings, {"phone": "+912", "outcome": "no_answer", "called_at": old})
    storage.append_result(settings, {"phone": "+913", "outcome": "completed", "called_at": old})
    assert storage.already_screened(settings, retry_after_hours=3) == {"+911", "+913"}
    assert storage.already_screened(settings) == {"+913"}


def test_words_after_reply_was_decided_are_not_resent(settings):
    """If Claude already answered the words, they must not be sent again."""
    from screener.speech import SpeechEvent

    brain = ScriptedBrain([
        {"say": "Hello, is now a good time?"},
        {"say": "Great. First question?"},
        {"say": "Thanks, goodbye.", "end_call": True},
    ], think_s=0.2)
    session = _fake_session(settings, brain)

    async def scenario():
        conversation = asyncio.create_task(session._conversation("[Call connected]"))
        await asyncio.sleep(0.4)
        await session.stt.events.put(SpeechEvent("turn_end", "Yes, sure."))
        await asyncio.sleep(0.6)
        await session.stt.events.put(SpeechEvent("turn_end", "My answer."))
        return await asyncio.wait_for(conversation, timeout=5)

    asyncio.run(scenario())
    assert brain.messages[1].endswith("Yes, sure.")
    assert brain.messages[2] == "My answer."


# --- scorecard -----------------------------------------------------------------


def _card():
    return {
        "call_outcome": "completed",
        "recording_consent": "yes",
        "confirmed_facts": {k: "" for k in SCORECARD_SCHEMA["properties"]["confirmed_facts"]["required"]},
        "sections": [
            {"section_id": "B", "title": "Recruitment", "score": 4, "reason": "Specific", "evidence_quote": "We hired 20"}
        ],
        "knockouts": [],
        "strengths": ["Clear examples"],
        "red_flags": [],
        "unanswered_candidate_questions": ["Is there a cab facility?"],
        "callback_request": "",
        "recommendation": "advance",
        "recommendation_reason": "Strong recruiter",
        "summary": "Good call.",
    }


def test_scorecard_outputs(tmp_path, settings):
    md = scorecard_markdown(Candidate(name="Asha Rao"), _card())
    assert "ADVANCE" in md and "cab facility" in md
    row = storage.scorecard_row(_card())
    assert row["score_B_recruitment"] == 4

    settings.raw.setdefault("storage", {})["output_dir"] = str(tmp_path)
    storage.append_result(settings, {"name": "Asha", "phone": "+919876543210", "outcome": "completed"})
    storage.append_result(settings, {"name": "Bo", "phone": "+919000000000", "outcome": "no_answer"})
    assert storage.already_screened(settings) == {"+919876543210"}


def test_format_transcript():
    text = format_transcript(
        [{"t": 0, "speaker": "candidate", "text": "Hello?"}, {"t": 65, "speaker": "agent", "text": "Hi"}], "Maya"
    )
    assert text == "[00:00] Candidate: Hello?\n[01:05] Maya: Hi"


# --- Deepgram Voice Agent engine (against a local fake server) -------------------


def _fake_agent_session(settings, url):
    from screener import voice_agent
    from screener.voice_agent import AgentCallSession

    voice_agent.AGENT_URL = url
    session = object.__new__(AgentCallSession)
    session.s = settings
    session.candidate = Candidate(name="Ruthwik Reddy")
    session.phone = None
    session.echo = False
    session.recorder = Recorder()
    session.transcript = []
    session._t0 = 0.0
    session.show_steps = False
    session._key = "test-key"
    session._loop = None
    session._outgoing = asyncio.Queue()
    session.consent = "unknown"
    session.notes = []
    session.callback_request = ""
    session.end_reason = ""
    played = []

    async def wait_until_played():
        return None

    session.engine = SimpleNamespace(
        player=SimpleNamespace(is_idle=lambda: True, clear=lambda: None),
        play=lambda audio, rate: played.append(len(audio)),
        wait_until_played=wait_until_played,
        input_level=0.0,
        start=lambda: None,
        stop=lambda: None,
    )

    class FakeTTS:
        async def close(self):
            return None

    session.tts = FakeTTS()
    session._played = played
    return session


def test_voice_agent_engine_happy_path(settings):
    from websockets.asyncio.server import serve

    received = {}
    settings.raw.setdefault("call", {})["greeting_delay_s"] = 0

    async def fake_deepgram(ws):
        await ws.send(json.dumps({"type": "Welcome"}))
        received["settings"] = json.loads(await ws.recv())
        received["auth"] = ws.request.headers.get("Authorization")
        await ws.send(json.dumps({"type": "SettingsApplied"}))
        await ws.send(json.dumps({"type": "ConversationText", "role": "assistant", "content": "Hello, am I speaking with Ruthwik?"}))
        await ws.send(b"\x00\x00" * 2400)  # 0.1 s of agent audio
        await ws.send(json.dumps({"type": "ConversationText", "role": "user", "content": "Yes, and the recording is fine."}))
        await ws.send(json.dumps({"type": "FunctionCallRequest", "functions": [
            {"id": "f1", "name": "record_consent", "arguments": "{\"consent\": \"yes\"}", "client_side": True},
            {"id": "f2", "name": "save_note", "arguments": "{\"question_id\": \"A4\", \"note\": \"Expects 6 lakhs\"}", "client_side": True},
        ]}))
        received["responses"] = [json.loads(await ws.recv()), json.loads(await ws.recv())]
        await ws.send(json.dumps({"type": "ConversationText", "role": "assistant", "content": "Thanks, goodbye!"}))
        await ws.send(json.dumps({"type": "FunctionCallRequest", "functions": [
            {"id": "f3", "name": "end_call", "arguments": "{\"reason\": \"completed\"}", "client_side": True},
        ]}))
        received["end"] = json.loads(await ws.recv())
        try:
            async for _ in ws:  # audio from the call, until the client hangs up
                pass
        except Exception:
            pass

    async def scenario():
        async with serve(fake_deepgram, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = _fake_agent_session(settings, f"ws://127.0.0.1:{port}")
            session._loop = asyncio.get_running_loop()
            session._t0 = time.monotonic()
            ended = await asyncio.wait_for(session._agent_conversation("Hello, am I speaking with Ruthwik?", 0), 10)
            return session, ended

    import time

    session, ended = asyncio.run(scenario())
    assert ended == "agent_ended"
    assert received["auth"] == "Token test-key"
    s = received["settings"]
    assert s["type"] == "Settings" and s["agent"]["greeting"] == "Hello, am I speaking with Ruthwik?"
    assert s["agent"]["think"]["provider"] == {"type": "anthropic", "model": "claude-haiku-4-5"}
    assert s["agent"]["speak"]["provider"]["model"] == "flux-priya-en"
    assert "Ruthwik" in s["agent"]["think"]["prompt"] and "record_consent" in s["agent"]["think"]["prompt"]
    assert [r["id"] for r in received["responses"]] == ["f1", "f2"] and received["end"]["id"] == "f3"
    assert session.consent == "yes" and session.notes == ["[A4] Expects 6 lakhs"] and session.end_reason == "completed"
    assert [t["speaker"] for t in session.transcript] == ["agent", "candidate", "agent"]
    assert session._played  # agent audio went to the call


def test_voice_agent_rejected_settings_raise_setup_error(settings):
    from websockets.asyncio.server import serve

    from screener.voice_agent import AgentSetupError

    async def rejecting(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "Error", "description": "INVALID_SETTINGS: bad model"}))

    async def scenario():
        async with serve(rejecting, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = _fake_agent_session(settings, f"ws://127.0.0.1:{port}")
            session._loop = asyncio.get_running_loop()
            with pytest.raises(AgentSetupError, match="INVALID_SETTINGS"):
                await session._agent_conversation("Hello?", 0)

    asyncio.run(scenario())


def test_voice_agent_hangs_up_after_goodbye_without_end_call(settings):
    from websockets.asyncio.server import serve

    async def forgetful(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "SettingsApplied"}))
        await ws.send(json.dumps({"type": "ConversationText", "role": "assistant",
                                  "content": "Thanks so much for your time. All the best!"}))
        try:
            async for _ in ws:
                pass
        except Exception:
            pass

    async def scenario():
        async with serve(forgetful, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = _fake_agent_session(settings, f"ws://127.0.0.1:{port}")
            session._loop = asyncio.get_running_loop()
            return await asyncio.wait_for(session._agent_conversation("Hello?", 0), 15)

    assert asyncio.run(scenario()) == "agent_ended"


def test_parse_callback_times():
    from datetime import datetime

    from screener.callbacks import parse_callback

    now = datetime(2026, 11, 7, 10, 12)
    assert parse_callback("Can you please call me after five minutes?", now) == datetime(2026, 11, 7, 10, 17)
    assert parse_callback("in an hour", now) == datetime(2026, 11, 7, 11, 12)
    assert parse_callback("after 6 pm", now) == datetime(2026, 11, 7, 18, 0)
    assert parse_callback("at 4", now) == datetime(2026, 11, 7, 16, 0)
    assert parse_callback("tomorrow 11 am", now) == datetime(2026, 11, 8, 11, 0)
    assert parse_callback("I'm in a meeting", now) is None


def test_callbacks_are_skipped_until_due_then_called(tmp_path, settings):
    from datetime import datetime, timedelta

    settings.raw.setdefault("storage", {})["output_dir"] = str(tmp_path)
    now = datetime.now()
    fmt = "%Y-%m-%d %H:%M"
    storage.append_result(settings, {"phone": "+911", "outcome": "call_back_later", "called_at": now.strftime(fmt),
                                     "callback_due": (now + timedelta(hours=1)).strftime(fmt)})
    storage.append_result(settings, {"phone": "+912", "outcome": "call_back_later", "called_at": now.strftime(fmt),
                                     "callback_due": (now - timedelta(minutes=1)).strftime(fmt)})
    done = storage.already_screened(settings, retry_after_hours=3)
    assert "+911" in done  # not due yet
    assert "+912" not in done  # due now, even though it was called minutes ago
    assert set(storage.pending_callbacks(settings)) == {"+911", "+912"}


def test_callback_note_reaches_the_agent(settings):
    candidate = Candidate(name="Rajitha B", fields={"is_callback": "yes"})
    assert "calling back as they asked" in candidate_context(settings, candidate)


def test_voice_agent_waits_for_a_pause_before_replying(settings):
    from websockets.asyncio.server import serve

    settings.raw.setdefault("voice_agent", {})["min_reply_gap_s"] = 0.6
    timeline = {}

    async def quick_replier(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "SettingsApplied"}))
        await asyncio.sleep(0.1)
        await ws.send(b"\x00\x00" * 2400)  # reply arrives while the candidate is still audible
        await asyncio.sleep(1.5)
        await ws.send(json.dumps({"type": "FunctionCallRequest", "functions": [
            {"id": "e", "name": "end_call", "arguments": "{\"reason\": \"completed\"}", "client_side": True}]}))
        try:
            async for _ in ws:
                pass
        except Exception:
            pass

    async def scenario():
        async with serve(quick_replier, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = _fake_agent_session(settings, f"ws://127.0.0.1:{port}")
            session._loop = asyncio.get_running_loop()
            session.engine.input_level = 0.05  # candidate still making sound
            task = asyncio.create_task(session._agent_conversation("Hello?", 0))
            await asyncio.sleep(0.4)
            timeline["while_talking"] = len(session._played)
            session.engine.input_level = 0.0  # they go quiet
            await asyncio.sleep(0.3)
            timeline["just_after"] = len(session._played)
            await asyncio.sleep(0.6)
            timeline["after_gap"] = len(session._played)
            return await asyncio.wait_for(task, 10)

    assert asyncio.run(scenario()) == "agent_ended"
    assert timeline == {"while_talking": 0, "just_after": 0, "after_gap": 1}


def test_voice_agent_says_goodbye_before_hanging_up(settings):
    from websockets.asyncio.server import serve

    injected = []

    async def abrupt(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "SettingsApplied"}))
        await ws.send(json.dumps({"type": "ConversationText", "role": "user", "content": "No. No. At the moment."}))
        await ws.send(json.dumps({"type": "FunctionCallRequest", "functions": [
            {"id": "e", "name": "end_call", "arguments": "{\"reason\": \"completed\"}", "client_side": True}]}))
        async for raw in ws:
            if isinstance(raw, str) and json.loads(raw).get("type") == "InjectAgentMessage":
                injected.append(json.loads(raw)["message"])
                await ws.send(b"\x00\x00" * 2400)  # the closing line's audio

    async def scenario():
        async with serve(abrupt, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = _fake_agent_session(settings, f"ws://127.0.0.1:{port}")
            session._loop = asyncio.get_running_loop()
            return await asyncio.wait_for(session._agent_conversation("Hello?", 0), 15)

    assert asyncio.run(scenario()) == "agent_ended"
    assert len(injected) == 1
    assert "Ruthwik" in injected[0] and "next round" in injected[0] and "Have a great day" in injected[0]
