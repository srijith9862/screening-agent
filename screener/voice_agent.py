"""Live call through Deepgram's Voice Agent API: listening, Claude and Priya's voice
in one WebSocket. Deepgram handles turn-taking (Flux end-of-turn detection),
interruptions and streaming; this module wires it to the call audio and keeps
the transcript, consent, notes and call-back for the scorecard.

Dialing, Phone Link, recording and the scorecard are shared with CallSession.
If Deepgram rejects the setup at the start of a call, the call falls back to
the classic engine so the candidate isn't left in silence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from .audio import STT_RATE, pcm16_to_float
from .call_session import CallResult, CallSession
from .config import Settings
from .phone import CALL_STATE_IDLE, PhoneError
from .prompts import voice_agent_prompt

log = logging.getLogger(__name__)

AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"
AGENT_OUTPUT_RATE = 24000

FUNCTIONS: list[dict[str, Any]] = [
    {
        "name": "record_consent",
        "description": "Record whether the candidate agreed to the call recording. Call it as soon as they clearly say yes or no.",
        "parameters": {
            "type": "object",
            "properties": {"consent": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["consent"],
        },
    },
    {
        "name": "save_note",
        "description": "Save a short private note for the HR team (max 15 words). Not spoken.",
        "parameters": {
            "type": "object",
            "properties": {
                "question_id": {"type": "string", "description": "e.g. A4, B2, or 'general'"},
                "note": {"type": "string"},
            },
            "required": ["note"],
        },
    },
    {
        "name": "schedule_callback",
        "description": "Record when the candidate asked to be called back.",
        "parameters": {
            "type": "object",
            "properties": {"when": {"type": "string"}},
            "required": ["when"],
        },
    },
    {
        "name": "end_call",
        "description": "Hang up. Only call this after you have said goodbye, or if the call cannot continue.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": [
                        "completed",
                        "declined_recording",
                        "call_back_later",
                        "not_interested",
                        "wrong_person",
                        "no_response",
                        "other",
                    ],
                }
            },
            "required": ["reason"],
        },
        "defer_until_eot": True,  # never hang up while the candidate is still talking
    },
]

DEFAULT_KEYTERMS = [
    "Naukri", "Malakpet", "Datasphere", "Registrar Corp", "hires", "hired", "onboarding",
    "CTC", "lakhs", "notice period", "buyout", "PF", "ESI", "EPFO", "ESIC", "professional tax",
    "payroll", "full and final", "PIP", "food safety", "regulatory", "LinkedIn",
]


def build_settings(settings: Settings, prompt: str, greeting: str, think_live: bool | None = None) -> dict[str, Any]:
    """The Voice Agent `Settings` message."""
    va = settings.get("voice_agent", {}) or {}
    listen: dict[str, Any] = {
        "type": "deepgram",
        "model": va.get("listen_model", "flux-general-en"),
        "version": "v2",
        "keyterms": va.get("keyterms") or DEFAULT_KEYTERMS,
        "eot_threshold": float(va.get("eot_threshold", 0.9)),
        "eot_timeout_ms": int(va.get("eot_timeout_ms", 5000)),
    }
    if va.get("eager_eot_threshold") is not None:
        listen["eager_eot_threshold"] = float(va["eager_eot_threshold"])
    think: dict[str, Any] = {
        "provider": {"type": "anthropic", "model": va.get("think_model", "claude-haiku-4-5")},
        "prompt": prompt,
        "functions": FUNCTIONS,
    }
    speak_provider: dict[str, Any] = {
        "type": "deepgram",
        "version": "v2",
        "model": settings.get("deepgram.tts_voice", "flux-priya-en"),
    }
    expressivity = int(settings.get("deepgram.tts_expressivity", 0))
    if expressivity:
        speak_provider["expressivity"] = expressivity
    return {
        "type": "Settings",
        "tags": ["screening-agent"],
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": STT_RATE},
            "output": {"encoding": "linear16", "sample_rate": AGENT_OUTPUT_RATE},
        },
        "agent": {
            "listen": {"provider": listen},
            "think": think,
            "speak": {"provider": speak_provider},
            **({"think_live": {"provider": think_live_provider(settings)}}
               if (think_live_enabled(settings) if think_live is None else think_live) else {}),
            "greeting": greeting,
        },
    }


_GOODBYE = re.compile(r"(?i)\b(good ?bye|bye|all the best|have a (great|good|nice|lovely) (day|evening))\b")


def is_goodbye(text: str) -> bool:
    return bool(_GOODBYE.search(text))


def closing_line(settings: Settings, first_name: str, reason: str) -> str:
    """What Priya says if Claude hangs up without saying goodbye."""
    if reason in ("completed", "other", ""):
        timeline = settings.script_values().get("reply_timeline", "soon")
        return (f"Thank you for your time, {first_name}. Our team will review this call and get back to you "
                f"{timeline} if you're selected for the next round. Have a great day!")
    if reason == "call_back_later":
        return "Thank you, I'll call you back then. Have a good day!"
    return f"Thank you for your time, {first_name}. Have a good day!"


def think_live_enabled(settings: Settings) -> bool:
    return bool((settings.get("voice_agent", {}) or {}).get("think_live", True))


def think_live_provider(settings: Settings) -> dict[str, Any]:
    """Deepgram's Think Live: treats 'mhmm'/'yes'/noise while Priya talks as backchannel,
    not an interruption, and checks on long silences."""
    va = settings.get("voice_agent", {}) or {}
    return {
        "type": "deepgram",
        "version": "v1",
        "agent_backchannel_sensitivity": float(va.get("backchannel_sensitivity", 0.15)),
        "presence_check_max_count": int(va.get("presence_check_max_count", 2)),
        "presence_check_interval_ms": int(va.get("presence_check_interval_ms", 6000)),
        "restricted_terms": ["human", "transfer", "lakhs", "salary", "budget"],
    }


class AgentSetupError(RuntimeError):
    """Deepgram rejected or never applied the settings (before anything was said)."""


class AgentCallSession(CallSession):
    """CallSession whose conversation runs on Deepgram's Voice Agent API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._key = args[3] if len(args) > 3 else kwargs["deepgram_key"]
        self._loop: asyncio.AbstractEventLoop | None = None
        self._outgoing: asyncio.Queue[bytes] = asyncio.Queue()
        self._audio_sink = self._feed_agent  # candidate audio goes to the Voice Agent
        self.consent = "unknown"
        self.notes: list[str] = []
        self.callback_request = ""
        self.end_reason = ""

    # --- audio from the call -------------------------------------------------

    def _feed_agent(self, pcm16: bytes) -> None:
        """Called from the audio thread with 16 kHz PCM from the call."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._outgoing.put_nowait, pcm16)

    # --- the call ---------------------------------------------------------------

    async def run(self) -> CallResult:
        self._loop = asyncio.get_running_loop()
        greeting = f"Hello, am I speaking with {self.candidate.first_name}?"
        self.engine.start()
        try:
            if self.phone is not None:
                # While it rings, plain speech-to-text listens for "Hello?" to spot the pickup.
                self._audio_sink = self.stt.feed
                await self.stt.start()
                try:
                    answered = await self._wait_for_answer()
                except PhoneError as exc:
                    log.error("Dialing failed: %s", exc)
                    return CallResult(outcome="dial_failed", ended_by=str(exc))
                finally:
                    self._audio_sink = self._feed_agent
                    await self.stt.stop()
                if answered is None:
                    await self._hang_up()
                    return CallResult(outcome="no_answer")
            self._t0 = time.monotonic()
            delay = float(self.s.get("call.greeting_delay_s", 2))
            try:
                ended_by = await self._agent_conversation(greeting, delay)
            except AgentSetupError as exc:
                self._say(f"! Deepgram Voice Agent couldn't start ({exc}). Continuing with the classic engine.")
                ended_by = await self._classic_fallback()
        finally:
            await self._hang_up()
            self.engine.stop()
            await self.tts.close()
        return CallResult(
            outcome="connected",
            consent=self.consent,
            transcript=self.transcript,
            notes=self.notes,
            callback_request=self.callback_request,
            recorder=self.recorder,
            duration_s=round(self._elapsed(), 1),
            ended_by=ended_by,
        )

    async def _classic_fallback(self) -> str:
        """Run the rest of the call on the classic engine (separate STT, Claude, TTS)."""
        self._audio_sink = self.stt.feed
        await self.stt.start()
        try:
            ended = await self._conversation(
                "[Call connected. The candidate picked up; greet them and start the opening.]"
            )
        finally:
            await self.stt.stop()
        self.consent = self.brain.consent
        self.notes = self.brain.notes
        self.callback_request = self.brain.callback_request
        return ended

    async def _agent_conversation(self, greeting: str, greeting_delay: float) -> str:
        self._think_live = think_live_enabled(self.s)
        try:
            return await self._agent_conversation_once(greeting, greeting_delay)
        except AgentSetupError as exc:
            if not self._think_live:
                raise
            self._say(f"! Think Live wasn't accepted ({exc}); continuing without it.")
            self._think_live = False
            return await self._agent_conversation_once(greeting, 0)

    async def _agent_conversation_once(self, greeting: str, greeting_delay: float) -> str:
        from websockets.asyncio.client import connect

        started = time.monotonic()
        prompt = voice_agent_prompt(self.s, self.candidate, greeting)
        try:
            ws = await asyncio.wait_for(
                connect(AGENT_URL, additional_headers={"Authorization": f"Token {self._key}"}, max_size=None),
                timeout=8,
            )
        except Exception as exc:  # noqa: BLE001
            raise AgentSetupError(f"couldn't connect: {exc}") from exc

        async with ws:
            # Settings first; the greeting is spoken as soon as they're applied.
            # Wait out the rest of the natural pause after pickup before sending them.
            remaining = greeting_delay - (time.monotonic() - started)
            if remaining > 0:
                await asyncio.sleep(remaining)
            await ws.send(json.dumps(build_settings(self.s, prompt, greeting, think_live=self._think_live)))
            await self._wait_for_settings(ws)
            self._drain_outgoing()  # don't send audio from before the agent was ready
            sender = asyncio.create_task(self._send_audio(ws))
            try:
                return await self._receive(ws, prompt)
            finally:
                sender.cancel()

    async def _wait_for_settings(self, ws: Any) -> None:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
            except asyncio.TimeoutError:
                break
            except Exception as exc:  # noqa: BLE001
                raise AgentSetupError(f"connection closed: {exc}") from exc
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "SettingsApplied":
                return
            if kind == "Error":
                raise AgentSetupError(msg.get("description") or msg.get("message") or json.dumps(msg)[:300])
        raise AgentSetupError("settings were not confirmed in time")

    def _drain_outgoing(self) -> None:
        while not self._outgoing.empty():
            self._outgoing.get_nowait()

    async def _send_audio(self, ws: Any) -> None:
        buffer = bytearray()
        while True:
            chunk = await self._outgoing.get()
            buffer.extend(chunk)
            if len(buffer) >= 3200:  # ~100 ms at 16 kHz
                await ws.send(bytes(buffer))
                buffer.clear()

    async def _receive(self, ws: Any, prompt: str) -> str:
        max_s = float(self.s.get("call.max_minutes", 20)) * 60
        wrap_s = float(self.s.get("call.wrap_up_minute", 17)) * 60
        silence_s = float(self.s.get("call.silence_timeout_s", 10))
        max_silences = int(self.s.get("call.max_silences", 2))
        wrap_sent = False
        silences = 0
        last_activity = time.monotonic()
        last_state_check = 0.0
        ending_since: float | None = None
        agent_audio_at = 0.0  # when the agent last sent audio
        candidate_speaking_since: float | None = None  # between UserStartedSpeaking and their transcript
        goodbye_at: float | None = None  # when the agent said goodbye (backup hang-up if end_call never comes)
        said_goodbye_this_turn = False  # since the candidate last spoke
        closing_sent_at: float | None = None  # we made Priya say the closing line ourselves
        closing_check_at: float | None = None  # when to say the closing ourselves if Claude hasn't
        # Barge-in: only stop Priya if the candidate's sound is sustained (not a cough or background noise).
        barge_since: float | None = None
        loud_s = 0.0
        last_tick = time.monotonic()
        barge_level = float((self.s.get("voice_agent", {}) or {}).get("interrupt_level", 0.01))
        barge_needed_s = float((self.s.get("voice_agent", {}) or {}).get("interrupt_after_s", 1.2))
        # Until the candidate has answered the recording question, Priya always finishes
        # speaking, so everyone hears who's calling and why.
        protect_opening = bool((self.s.get("voice_agent", {}) or {}).get("protect_opening", True))
        # Minimum pause between the candidate going quiet and Priya starting to reply.
        min_gap_s = float((self.s.get("voice_agent", {}) or {}).get("min_reply_gap_s", 1.5))
        candidate_voice_at = 0.0  # last moment the candidate's line was loud like speech
        held: list[Any] = []  # start of Priya's reply, waiting for the pause
        held_loud_s = 0.0

        while True:
            now = time.monotonic()
            tick, last_tick = now - last_tick, now
            if self._elapsed() > max_s + 60:
                return "time_limit"

            if self.engine.input_level > barge_level:
                candidate_voice_at = now
                if held:
                    held_loud_s += tick
            # Release a held reply once they've been quiet long enough; drop it if they kept talking.
            if held:
                if held_loud_s >= barge_needed_s:
                    held.clear()  # they carried on talking; Deepgram will reply to the full answer
                elif now - candidate_voice_at >= min_gap_s:
                    for audio in held:
                        self.engine.play(audio, AGENT_OUTPUT_RATE)
                    held.clear()
                    agent_audio_at = now

            if barge_since is not None:
                if self.engine.input_level > barge_level:
                    loud_s += tick
                if protect_opening and self.consent == "unknown":
                    barge_since = None  # never cut the introduction short
                elif loud_s >= barge_needed_s:
                    self.engine.player.clear()  # real interruption: stop talking over them
                    barge_since = None
                elif now - barge_since > barge_needed_s + 1.0:
                    barge_since = None  # just noise; keep talking

            if self.phone is not None and now - last_state_check > 1.5:
                last_state_check = now
                try:
                    if await self.phone.call_state() == CALL_STATE_IDLE:
                        return "candidate_hung_up"
                except PhoneError:
                    pass

            # Hang up once the goodbye has finished playing.
            if closing_check_at is not None and now >= closing_check_at:
                closing_check_at = None
                if not said_goodbye_this_turn:  # never hang up without a proper goodbye
                    line = closing_line(self.s, self.candidate.first_name, self.end_reason)
                    await ws.send(json.dumps({"type": "InjectAgentMessage", "message": line}))
                    closing_sent_at = now

            if ending_since is not None:
                if closing_check_at is not None:
                    quiet = False
                else:
                    quiet = now - agent_audio_at > 1.2 and self.engine.player.is_idle() and not held
                if closing_sent_at is not None and agent_audio_at < closing_sent_at and now - closing_sent_at < 6:
                    quiet = False  # wait for the closing line we injected to arrive and play
                if quiet or now - ending_since > 20:
                    await asyncio.sleep(0.5)
                    return "agent_ended"

            # Backup hang-up: the agent said goodbye but never called end_call.
            agent_quiet = now - agent_audio_at > 1.5 and self.engine.player.is_idle() and not held
            if (
                goodbye_at is not None
                and ending_since is None
                and candidate_speaking_since is None
                and agent_quiet
                and now - goodbye_at > 4
            ):
                return "agent_ended"

            if not wrap_sent and self._elapsed() > wrap_s:
                wrap_sent = True
                note = f"[Time check: about {int(self._elapsed() // 60)} minutes have passed. Start wrapping up now.]"
                await ws.send(json.dumps({"type": "UpdatePrompt", "prompt": prompt + "\n\n" + note}))
            if self._elapsed() > max_s and ending_since is None:
                await ws.send(json.dumps({"type": "InjectAgentMessage", "message":
                    "I'm sorry, we're out of time. The HR team will be in touch. Thank you, goodbye!"}))
                ending_since = now

            # Silence: nobody has spoken for a while.
            agent_busy = not self.engine.player.is_idle() or now - agent_audio_at < 1.0 or bool(held)
            candidate_busy = candidate_speaking_since is not None and now - candidate_speaking_since < 90
            if agent_busy or candidate_busy:
                last_activity = now
            elif self._think_live:
                if ending_since is None and now - last_activity > max(30.0, silence_s * 3):
                    ending_since = now  # Think Live already checked in; nobody's there
                    await ws.send(json.dumps({"type": "InjectAgentMessage", "message":
                        "I can't hear you clearly, so the team will try again later. Thank you, goodbye!"}))
            elif ending_since is None and now - last_activity > silence_s:
                silences += 1
                last_activity = now
                if silences > max_silences:
                    return "silence"
                message = (
                    "I can't hear you clearly, so the team will try again later. Thank you, goodbye!"
                    if silences == max_silences
                    else "Hello, can you hear me?"
                )
                await ws.send(json.dumps({"type": "InjectAgentMessage", "message": message}))
                if silences == max_silences:
                    ending_since = now

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                self._say(f"! Voice Agent connection closed: {exc}")
                return "agent_disconnected"

            if isinstance(raw, bytes):
                audio = pcm16_to_float(raw)
                now = time.monotonic()
                starting_reply = self.engine.player.is_idle() and not held
                if held or (starting_reply and candidate_voice_at and now - candidate_voice_at < min_gap_s):
                    if not held:
                        held_loud_s = 0.0
                    held.append(audio)  # wait for a natural pause before speaking
                else:
                    self.engine.play(audio, AGENT_OUTPUT_RATE)
                agent_audio_at = now
                continue

            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "ConversationText":
                speaker = "agent" if msg.get("role") == "assistant" else "candidate"
                text = (msg.get("content") or "").strip()
                if text:
                    self._log(speaker, text)
                if speaker == "candidate":
                    silences = 0
                    last_activity = time.monotonic()
                    candidate_speaking_since = None
                    said_goodbye_this_turn = False
                    if len(text.split()) > 4:
                        goodbye_at = None  # they've raised something new; let the agent answer
                elif text and is_goodbye(text):
                    goodbye_at = time.monotonic()
                    said_goodbye_this_turn = True
            elif kind == "UserStartedSpeaking":
                if not self.engine.player.is_idle():
                    barge_since, loud_s = time.monotonic(), 0.0  # decide shortly if it's real speech
                candidate_speaking_since = time.monotonic()
                silences = 0
                last_activity = time.monotonic()
            elif kind == "FunctionCallRequest":
                for call in msg.get("functions", []):
                    if call.get("client_side", True):
                        await self._handle_function(ws, call)
                        if call.get("name") == "end_call" and ending_since is None:
                            ending_since = time.monotonic()
                            if not said_goodbye_this_turn:
                                closing_check_at = time.monotonic() + 3.0  # give Claude's own goodbye a moment
            elif kind == "Error":
                self._say(f"! Voice Agent error: {msg.get('description') or msg}")
                return "agent_error"
            elif kind == "Warning":
                description = msg.get("description") or str(msg)
                if "InjectAgentMessage" in description:
                    log.debug("Voice Agent: %s", description)  # harmless: candidate was talking
                else:
                    log.warning("Voice Agent warning: %s", description)

    async def _handle_function(self, ws: Any, call: dict[str, Any]) -> None:
        name = call.get("name", "")
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        result = "ok"
        if name == "record_consent":
            consent = str(args.get("consent", "")).lower()
            if consent in ("yes", "no"):
                self.consent = consent
            if self.consent == "no" and self.recorder.enabled:
                self.recorder.discard()
                log.info("Recording consent refused: recording discarded")
        elif name == "save_note":
            note = str(args.get("note", "")).strip()
            if note:
                self.notes.append(f"[{args.get('question_id', 'general')}] {note}")
                if self.echo:
                    print(f"           (note: {note})", flush=True)
        elif name == "schedule_callback":
            self.callback_request = str(args.get("when", "")).strip()
        elif name == "end_call":
            self.end_reason = str(args.get("reason", ""))
        else:
            result = f"unknown function {name}"
        await ws.send(json.dumps({"type": "FunctionCallResponse", "id": call.get("id"), "name": name, "content": result}))
