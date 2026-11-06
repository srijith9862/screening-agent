"""Runs one live screening call: phone + audio + speech + Claude."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .audio import AudioEngine, Recorder
from .brain import AgentTurn, Brain
from .candidates import Candidate
from .config import Settings
from .phone import CALL_STATE_IDLE, Phone, PhoneError
from .speech import TTS_RATE, SpeechEvent, SpeechToText, TextToSpeech, split_sentences

log = logging.getLogger(__name__)

BARGE_IN_MIN_WORDS = 3


@dataclass
class CallResult:
    outcome: str  # "connected" | "no_answer" | "dial_failed"
    consent: str = "unknown"
    transcript: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    callback_request: str = ""
    recorder: Recorder | None = None
    duration_s: float = 0.0
    ended_by: str = ""


class CallSession:
    def __init__(
        self,
        settings: Settings,
        candidate: Candidate,
        phone: Phone | None,
        deepgram_key: str,
        in_device: str | None,
        out_device: str | None,
        brain: Brain | None = None,
        echo: bool = True,
    ) -> None:
        self.s = settings
        self.candidate = candidate
        self.phone = phone
        self.echo = echo
        self.recorder = Recorder()
        self.stt = SpeechToText(
            deepgram_key,
            model=settings.get("deepgram.stt_model", "nova-3"),
            language=settings.get("deepgram.stt_language", "en-IN"),
            endpointing_ms=int(settings.get("deepgram.endpointing_ms", 400)),
            utterance_end_ms=int(settings.get("deepgram.utterance_end_ms", 1200)),
        )
        self.tts = TextToSpeech(
            deepgram_key,
            settings.get("deepgram.tts_voice", "flux-priya-en"),
            int(settings.get("deepgram.tts_expressivity", 0)),
        )
        self.engine = AudioEngine(in_device, out_device, self.stt.feed, self.recorder)
        self.brain = brain or Brain(settings, candidate)
        self.transcript: list[dict[str, Any]] = []
        self._t0 = 0.0
        self._interrupted = False
        self._speaking_text = ""

    # --- helpers -----------------------------------------------------------

    def _elapsed(self) -> float:
        return time.monotonic() - self._t0 if self._t0 else 0.0

    def _log(self, speaker: str, text: str, **extra: Any) -> None:
        self.transcript.append({"t": round(self._elapsed(), 1), "speaker": speaker, "text": text, **extra})
        if self.echo:
            label = {"agent": self.s.agent_name, "candidate": self.candidate.first_name}.get(speaker, speaker)
            print(f"  [{int(self._elapsed()) // 60:02d}:{int(self._elapsed()) % 60:02d}] {label}: {text}", flush=True)

    async def _speak(self, text: str) -> None:
        sentences = split_sentences(text)
        if not sentences:
            return
        self._interrupted = False
        self._speaking_text = text
        # Synthesize all sentences in parallel, play them in order as each is ready.
        jobs = [asyncio.create_task(self.tts.synthesize(sentence)) for sentence in sentences]
        try:
            for job in jobs:
                audio = await job
                if self._interrupted:
                    break
                self.engine.play(audio, TTS_RATE)
        finally:
            for job in jobs:
                job.cancel()
        self._log("agent", text)

    def _barge_in(self) -> None:
        if not self.engine.player.is_idle():
            self.engine.player.clear()
            self._interrupted = True
            if self.transcript and self.transcript[-1]["speaker"] == "agent":
                self.transcript[-1]["interrupted"] = True

    def _apply_turn(self, turn: AgentTurn) -> None:
        if self.brain.consent == "no" and self.recorder.enabled:
            self.recorder.discard()
            log.info("Recording consent refused: recording discarded")

    # --- phases ------------------------------------------------------------

    async def _wait_for_answer(self) -> str | None:
        """Returns the first thing the candidate said ("" if nothing yet), or None if not answered."""
        assert self.phone is not None
        mode = self.s.get("phone.answer_detection", "telecom")
        timeout = float(self.s.get("phone.ring_timeout_s", 45))
        baseline = None
        if mode == "telecom":
            try:
                baseline = await self.phone.active_count()
            except PhoneError as exc:
                log.warning("Telecom answer detection unavailable (%s); using speech", exc)
        await self.phone.dial(self.candidate.phone)
        start = time.monotonic()
        if self.s.get("phone.auto_bluetooth", True):
            await self._route_audio_to_laptop()
        heard = ""
        while time.monotonic() - start < timeout:
            await asyncio.sleep(0.5)
            if time.monotonic() - start > 4 and await self.phone.call_state() == CALL_STATE_IDLE:
                return None  # rejected, busy, or failed
            if baseline is not None:
                if await self.phone.active_count() > baseline:
                    self._drain_events()
                    return ""
            else:
                while not self.stt.events.empty():
                    event = self.stt.events.get_nowait()
                    if event.kind in ("final", "turn_end"):
                        heard = event.text
                if heard:
                    return heard
        return None

    async def _call_audio_arriving(self, seconds: float, threshold: float = 0.002) -> bool:
        """True if any call audio (e.g. the ringing tone) reaches the laptop within `seconds`."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.engine.input_level > threshold:
                return True
            await asyncio.sleep(0.1)
        return False

    def _say(self, text: str) -> None:
        if self.echo:
            print(f"  {text}", flush=True)

    async def _route_audio_to_laptop(self) -> None:
        """Calls dialed on the phone stay on its earpiece; switch them to the laptop (Bluetooth)."""
        assert self.phone is not None
        try:
            button = await self.phone.bluetooth_button()
            if button is not None and button.selected:
                self._say("Call audio: already on Bluetooth.")
                return
            if await self._call_audio_arriving(4):
                self._say("Call audio: reaching the laptop.")
                return
            outcome = await self.phone.tap_bluetooth(str(self.s.get("phone.bluetooth_device_name") or ""))
            self._say(f"Call audio: {outcome}.")
            if not await self._call_audio_arriving(6):
                self._say("! No call audio on the laptop yet. If the candidate can't hear the agent, tap Bluetooth on the phone.")
        except PhoneError as exc:
            self._say(f"! Couldn't switch call audio automatically ({exc}). Tap Bluetooth on the phone.")

    def _drain_events(self) -> None:
        while not self.stt.events.empty():
            self.stt.events.get_nowait()

    async def _first_words(self, wait_s: float = 4.0) -> str:
        """After pickup, give the candidate a moment to say hello."""
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            try:
                event = await asyncio.wait_for(self.stt.events.get(), timeout=end - time.monotonic())
            except asyncio.TimeoutError:
                break
            if event.kind in ("turn_end", "final"):
                return event.text
        return ""

    async def run(self) -> CallResult:
        await self.stt.start()
        self.engine.start()
        result = CallResult(outcome="connected", recorder=self.recorder)
        try:
            if self.phone is not None:
                try:
                    hello = await self._wait_for_answer()
                except PhoneError as exc:
                    log.error("Dialing failed: %s", exc)
                    return CallResult(outcome="dial_failed", ended_by=str(exc))
                if hello is None:
                    await self.phone.hang_up()
                    return CallResult(outcome="no_answer")
            else:
                print("  Rehearsal: say hello into your microphone to start.", flush=True)
                hello = await self._first_words(wait_s=30)
            self._t0 = time.monotonic()
            if not hello:
                hello = await self._first_words()
            if hello:
                self._log("candidate", hello)
            opening = (
                f'[Call connected. The candidate answered and said: "{hello}"]'
                if hello
                else "[Call connected. The candidate picked up but hasn't spoken yet.]"
            )
            result.ended_by = await self._conversation(opening)
        finally:
            if self.phone is not None:
                await self.phone.hang_up()
            self.engine.stop()
            await self.stt.stop()
            await self.tts.close()
        result.duration_s = round(self._elapsed(), 1)
        result.consent = self.brain.consent
        result.transcript = self.transcript
        result.notes = self.brain.notes
        result.callback_request = self.brain.callback_request
        return result

    async def _conversation(self, first_message: str) -> str:
        max_s = float(self.s.get("call.max_minutes", 16)) * 60
        wrap_s = float(self.s.get("call.wrap_up_minute", 13)) * 60
        silence_s = float(self.s.get("call.silence_timeout_s", 10))
        max_silences = int(self.s.get("call.max_silences", 2))

        pending_text = ""
        respond_task: asyncio.Task[AgentTurn] | None = None
        speak_task: asyncio.Task[None] | None = None
        wrap_sent = False
        silences = 0
        last_activity = time.monotonic()
        last_state_check = 0.0
        ending = False

        def start_respond(text: str) -> asyncio.Task[AgentTurn]:
            nonlocal wrap_sent
            notes = []
            if self._interrupted:
                notes.append("[The candidate started talking while you were speaking, so they may not have heard all of your last message.]")
                self._interrupted = False
            if not wrap_sent and self._elapsed() > wrap_s:
                notes.append(f"[Time check: {int(self._elapsed() // 60)} minutes have passed. Start wrapping up now.]")
                wrap_sent = True
            if self._elapsed() > max_s:
                notes.append("[Time is up. Thank the candidate, say goodbye and end the call now.]")
            message = "\n".join(notes + ([text] if text else []))
            return asyncio.create_task(self.brain.respond(message))

        respond_task = start_respond(first_message)

        while True:
            now = time.monotonic()
            # Hard limit, with one minute of grace for the goodbye.
            if self._elapsed() > max_s + 60:
                return "time_limit"

            # Did the candidate hang up?
            if self.phone is not None and now - last_state_check > 1.5:
                last_state_check = now
                try:
                    if await self.phone.call_state() == CALL_STATE_IDLE:
                        if respond_task:
                            respond_task.cancel()
                        return "candidate_hung_up"
                except PhoneError:
                    pass

            # A reply is ready: speak it.
            if respond_task is not None and respond_task.done():
                try:
                    turn = respond_task.result()
                except asyncio.CancelledError:
                    turn = None
                respond_task = None
                if turn is not None:
                    pending_text = ""
                    self._apply_turn(turn)
                    if turn.note and self.echo:
                        print(f"           (note: {turn.note})", flush=True)
                    speak_task = asyncio.create_task(self._speak(turn.say)) if turn.say else None
                    if turn.end_call:
                        ending = True

            if ending and respond_task is None:
                if speak_task is not None:
                    await speak_task
                await self.engine.wait_until_played()
                await asyncio.sleep(0.8)
                return "agent_ended"

            # Speech events.
            try:
                event: SpeechEvent | None = await asyncio.wait_for(self.stt.events.get(), timeout=0.2)
            except asyncio.TimeoutError:
                event = None

            if event is not None:
                if event.kind == "error":
                    return "speech_error"
                last_activity = time.monotonic()
                silences = 0
                if event.kind == "partial" and len(event.text.split()) >= BARGE_IN_MIN_WORDS:
                    self._barge_in()
                elif event.kind == "turn_end":
                    self._log("candidate", event.text)
                    pending_text = f"{pending_text} {event.text}".strip()
                    # Still thinking about the previous words? Restart with everything they said.
                    if respond_task is not None:
                        respond_task.cancel()
                    respond_task = start_respond(pending_text)
                continue

            # Silence handling (only when nobody is talking and nothing is pending).
            agent_busy = respond_task is not None or not self.engine.player.is_idle() or (
                speak_task is not None and not speak_task.done()
            )
            if agent_busy:
                last_activity = time.monotonic()
            elif time.monotonic() - last_activity > silence_s:
                silences += 1
                last_activity = time.monotonic()
                if silences > max_silences:
                    return "silence"
                if silences == max_silences:
                    note = "[Still no response. Say the line seems unclear, that the team will try again later, say goodbye and end the call.]"
                else:
                    note = f"[The candidate has been silent for {int(silence_s)} seconds. Check if they can hear you.]"
                respond_task = start_respond(note)
