import asyncio
import json
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
    topics = [item["topic"] for item in settings.usable_faq()]
    assert "Office location" in topics
    assert all(not item["answer"].upper().startswith("TODO") for item in settings.usable_faq())
    assert settings.todo_warnings()  # shipped config still has TODOs to fill


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
    assert request["fallbacks"] == "default"
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["output_config"]["format"]["type"] == "json_schema"


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
