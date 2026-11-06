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


def build_settings(settings: Settings, prompt: str, greeting: str) -> dict[str, Any]:
    """The Voice Agent `Settings` message."""
    va = settings.get("voice_agent", {}) or {}
    listen: dict[str, Any] = {
        "type": "deepgram",
        "model": va.get("listen_model", "flux-general-en"),
        "version": "v2",
        "keyterms": va.get("keyterms") or DEFAULT_KEYTERMS,
        "eot_threshold": float(va.get("eot_threshold", 0.8)),
        "eot_timeout_ms": int(va.get("eot_timeout_ms", 4000)),
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
            "greeting": greeting,
        },
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
                    await self.phone.hang_up()
                    return CallResult(outcome="no_answer")
            self._t0 = time.monotonic()
            delay = float(self.s.get("call.greeting_delay_s", 2))
            try:
                ended_by = await self._agent_conversation(greeting, delay)
            except AgentSetupError as exc:
                self._say(f"! Deepgram Voice Agent couldn't start ({exc}). Continuing with the classic engine.")
                ended_by = await self._classic_fallback()
        finally:
            if self.phone is not None:
                await self.phone.hang_up()
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
            await ws.send(json.dumps(build_settings(self.s, prompt, greeting)))
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

        while True:
            now = time.monotonic()
            if self._elapsed() > max_s + 60:
                return "time_limit"

            if self.phone is not None and now - last_state_check > 1.5:
                last_state_check = now
                try:
                    if await self.phone.call_state() == CALL_STATE_IDLE:
                        return "candidate_hung_up"
                except PhoneError:
                    pass

            # Hang up once the goodbye has finished playing.
            if ending_since is not None:
                quiet = now - agent_audio_at > 1.2 and self.engine.player.is_idle()
                if quiet or now - ending_since > 12:
                    await asyncio.sleep(0.5)
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
            agent_busy = not self.engine.player.is_idle() or now - agent_audio_at < 1.0
            if agent_busy:
                last_activity = now
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
                raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                self._say(f"! Voice Agent connection closed: {exc}")
                return "agent_disconnected"

            if isinstance(raw, bytes):
                self.engine.play(pcm16_to_float(raw), AGENT_OUTPUT_RATE)
                agent_audio_at = time.monotonic()
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
            elif kind == "UserStartedSpeaking":
                self.engine.player.clear()  # stop talking over the candidate
                silences = 0
                last_activity = time.monotonic()
            elif kind == "FunctionCallRequest":
                for call in msg.get("functions", []):
                    if call.get("client_side", True):
                        await self._handle_function(ws, call)
                        if call.get("name") == "end_call" and ending_since is None:
                            ending_since = time.monotonic()
            elif kind == "Error":
                self._say(f"! Voice Agent error: {msg.get('description') or msg}")
                return "agent_error"
            elif kind == "Warning":
                log.warning("Voice Agent warning: %s", msg.get("description") or msg)

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
