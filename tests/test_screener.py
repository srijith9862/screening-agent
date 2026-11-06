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


def test_budget_check_catches_ranges():
    from screener.simulate import BUDGET_PATTERN

    assert BUDGET_PATTERN.search("The budget is around 5 lakhs")
    assert BUDGET_PATTERN.search("we offer 3 to 5 lakhs")
    assert not BUDGET_PATTERN.search("Salary is discussed in the next round.")


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
