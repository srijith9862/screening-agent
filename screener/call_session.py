"""Runs one live screening call: phone + audio + speech + Claude."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from . import phonelink
from .audio import AudioEngine, Recorder
from .brain import AgentTurn, Brain
from .candidates import Candidate
from .config import Settings
from .phone import BLUETOOTH_PATTERN, CALL_STATE_IDLE, Phone, PhoneError, find_node
from .speech import TTS_RATE, SpeechEvent, SpeechToText, TextToSpeech, split_sentences

log = logging.getLogger(__name__)

# Words the candidate must say while the agent is talking to interrupt it.
BARGE_IN_MIN_WORDS = 3
# Words that show the candidate is still talking while the agent prepares a reply.
STILL_TALKING_MIN_WORDS = 2
# After the candidate's last partial words, wait this long for their turn to end
# before replying anyway (in case the end-of-speech signal never arrives).
CANDIDATE_TALKING_GRACE_S = 2.5
_SENTENCE_END = re.compile(r"[.!?][\"')]?\s+")


def complete_sentences_end(text: str) -> int:
    """Index just after the last complete sentence in `text` (0 if none yet)."""
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    return ends[-1] if ends else 0


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


@dataclass
class Reply:
    """One agent turn: Claude (streamed) -> sentences -> voice -> call."""

    message: str  # what was sent to Claude
    consumed_text: str  # candidate words included in the message
    consumed_notes: list[str]
    started_at: float
    task: asyncio.Task[None] | None = None
    brain_done: bool = False  # Claude finished: history now contains this reply
    handled: bool = False  # turn applied by the conversation loop
    turn: AgentTurn | None = None
    queued: list[str] = field(default_factory=list)  # sentences sent to the voice
    played: int = 0  # sentences handed to the speaker
    first_audio_at: float | None = None

    @property
    def spoken(self) -> str:
        return " ".join(self.queued[: self.played])


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

    # --- helpers -----------------------------------------------------------

    def _elapsed(self) -> float:
        return time.monotonic() - self._t0 if self._t0 else 0.0

    def _log(self, speaker: str, text: str, **extra: Any) -> None:
        self.transcript.append({"t": round(self._elapsed(), 1), "speaker": speaker, "text": text, **extra})
        if self.echo:
            label = {"agent": self.s.agent_name, "candidate": self.candidate.first_name}.get(speaker, speaker)
            if "latency_s" in extra:
                label += f" ({extra['latency_s']}s)"
            if extra.get("interrupted"):
                text += "  [cut off]"
            print(f"  [{int(self._elapsed()) // 60:02d}:{int(self._elapsed()) % 60:02d}] {label}: {text}", flush=True)

    def _say(self, text: str) -> None:
        if self.echo:
            print(f"  {text}", flush=True)

    # --- one agent reply -----------------------------------------------------

    async def _run_reply(self, reply: Reply) -> None:
        """Streams Claude's reply and speaks each sentence as soon as it is complete."""
        jobs: asyncio.Queue[asyncio.Task[Any] | None] = asyncio.Queue()
        emitted = ""

        def queue_sentence(sentence: str) -> None:
            reply.queued.append(sentence)
            jobs.put_nowait(asyncio.create_task(self.tts.synthesize(sentence)))

        def on_say(say_so_far: str) -> None:
            nonlocal emitted
            if not say_so_far.startswith(emitted):
                return
            cut = complete_sentences_end(say_so_far[len(emitted):])
            if cut:
                for sentence in split_sentences(say_so_far[len(emitted): len(emitted) + cut]):
                    queue_sentence(sentence)
                emitted = say_so_far[: len(emitted) + cut]

        async def play_in_order() -> None:
            while True:
                job = await jobs.get()
                if job is None:
                    return
                try:
                    audio = await job
                except Exception as exc:  # noqa: BLE001 - skip a sentence rather than drop the call
                    log.error("Voice synthesis failed: %s", exc)
                    continue
                if reply.first_audio_at is None:
                    reply.first_audio_at = time.monotonic()
                self.engine.play(audio, TTS_RATE)
                reply.played += 1

        player = asyncio.create_task(play_in_order())
        try:
            turn = await self.brain.respond(reply.message, on_say=on_say)
            reply.turn = turn
            reply.brain_done = True
            rest = turn.say[len(emitted):] if turn.say.startswith(emitted) else ("" if emitted else turn.say)
            for sentence in split_sentences(rest):
                queue_sentence(sentence)
            jobs.put_nowait(None)
            await player
        finally:
            if not player.done():
                player.cancel()
            while not jobs.empty():
                job = jobs.get_nowait()
                if job is not None:
                    job.cancel()

    def _log_reply(self, reply: Reply, interrupted: bool) -> None:
        text = reply.spoken if interrupted else " ".join(reply.queued)
        if not text:
            return
        extra: dict[str, Any] = {}
        if reply.first_audio_at is not None:
            extra["latency_s"] = round(reply.first_audio_at - reply.started_at, 1)
        if interrupted:
            extra["interrupted"] = True
        self._log("agent", text, **extra)

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
        via_phone_link = await self._dial()
        start = time.monotonic()
        # While it rings, move the call audio to the laptop if it started on the phone.
        route_task = None if via_phone_link else asyncio.create_task(self._get_audio_to_laptop())

        async def answered(result: str) -> str:
            if route_task is not None and not route_task.done():
                try:  # let the transfer finish before the agent speaks
                    await asyncio.wait_for(asyncio.shield(route_task), timeout=8)
                except asyncio.TimeoutError:
                    self._say("! Call audio may still be on the phone. Click 'Transfer' on the laptop if you see it.")
            return result

        try:
            heard = ""
            while time.monotonic() - start < timeout:
                await asyncio.sleep(0.5)
                if time.monotonic() - start > 4 and await self.phone.call_state() == CALL_STATE_IDLE:
                    return None  # rejected, busy, or failed
                if baseline is not None:
                    if await self.phone.active_count() > baseline:
                        self._drain_events()
                        return await answered("")
                else:
                    while not self.stt.events.empty():
                        event = self.stt.events.get_nowait()
                        if event.kind in ("final", "turn_end"):
                            heard = event.text
                    if heard:
                        return await answered(heard)
            return None
        finally:
            if route_task is not None and not route_task.done():
                route_task.cancel()

    async def _get_audio_to_laptop(self) -> None:
        """The call started on the phone: click Phone Link's Transfer popup, else tap Bluetooth on the phone."""
        clicked, details = await phonelink.click_transfer(timeout_s=12)
        if clicked:
            self._say(f"Call audio: clicked '{details.removeprefix('CLICKED ').strip()}' in Phone Link.")
            return
        if await self._call_audio_arriving(1):
            self._say("Call audio: reaching the laptop.")
            return
        log.info("Transfer popup not found: %s", details)
        if self.s.get("phone.auto_bluetooth", True):
            await self._route_audio_to_laptop()

    async def _dial(self) -> bool:
        """Dials the candidate. Returns True if the call was placed through Phone Link."""
        assert self.phone is not None
        if self.s.get("phone.dial_via", "phonelink") == "phonelink":
            clicked, details = await phonelink.dial(self.candidate.phone)
            if clicked:
                self._say(f"Dialing through Phone Link ({details.removeprefix('CLICKED ').strip()}).")
                return True
            # Phone Link may have placed the call without a button: never dial twice.
            await asyncio.sleep(2)
            if await self.phone.call_state() != CALL_STATE_IDLE:
                self._say("Dialing (call already started).")
                return False  # still check the audio reached the laptop
            self._say(f"! Phone Link didn't place the call ({details[:300]}). Dialing from the phone instead.")
        await self.phone.dial(self.candidate.phone)
        return False

    async def _call_audio_arriving(self, seconds: float, threshold: float = 0.002) -> bool:
        """True if any call audio (e.g. the ringing tone) reaches the laptop within `seconds`."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.engine.input_level > threshold:
                return True
            await asyncio.sleep(0.1)
        return False

    async def _route_audio_to_laptop(self, wait_s: float = 10.0) -> None:
        """Calls dialed on the phone may stay on its earpiece; switch them to the laptop (Bluetooth).

        The in-call screen takes a moment to appear after dialing, so keep
        looking for its Bluetooth button for up to `wait_s` seconds.
        """
        assert self.phone is not None
        device = str(self.s.get("phone.bluetooth_device_name") or "")
        deadline = time.monotonic() + wait_s
        labels: list[str] = []
        try:
            while time.monotonic() < deadline:
                if self.engine.input_level > 0.002:
                    self._say("Call audio: reaching the laptop.")
                    return
                nodes = await self.phone.screen_nodes()
                labels = [n.label for n in nodes]
                button = find_node(nodes, BLUETOOTH_PATTERN)
                if button is not None:
                    if button.selected:
                        self._say("Call audio: already on Bluetooth.")
                        return
                    outcome = await self.phone.tap_bluetooth(device)
                    self._say(f"Call audio: {outcome}.")
                    return
                await asyncio.sleep(0.7)
        except PhoneError as exc:
            self._say(f"! Couldn't check the call audio route ({exc}). Click 'Transfer' on the laptop if asked.")
            return
        seen = ", ".join(dict.fromkeys(labels))[:300] or "nothing (is the phone unlocked?)"
        self._say("! Couldn't find the Bluetooth button on the call screen. Click 'Transfer' on the laptop popup.")
        self._say(f"  (phone screen showed: {seen})")

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

        pending_text = ""  # candidate words the agent hasn't replied to yet
        pending_notes: list[str] = [first_message]  # [bracketed] notes for the next reply
        reply: Reply | None = None
        wrap_sent = False
        silences = 0
        last_activity = time.monotonic()
        last_state_check = 0.0
        last_partial_at = 0.0  # when the candidate was last heard mid-sentence
        ending = False

        def start_reply() -> None:
            nonlocal reply, wrap_sent
            notes = list(pending_notes)
            extra = []
            if not wrap_sent and self._elapsed() > wrap_s:
                extra.append(f"[Time check: {int(self._elapsed() // 60)} minutes have passed. Start wrapping up now.]")
                wrap_sent = True
            if self._elapsed() > max_s:
                extra.append("[Time is up. Thank the candidate, say goodbye and end the call now.]")
            message = "\n".join(notes + extra + ([pending_text] if pending_text else []))
            reply = Reply(message=message, consumed_text=pending_text, consumed_notes=notes, started_at=time.monotonic())
            reply.task = asyncio.create_task(self._run_reply(reply))

        def cancel_reply() -> None:
            """Stops the current reply (candidate kept talking or interrupted)."""
            nonlocal reply
            if reply is None:
                return
            assert reply.task is not None
            if reply.brain_done and not reply.handled:
                handle_finished_thinking(reply)  # its words are in the history now
            reply.task.cancel()
            self.engine.player.clear()
            if reply.brain_done and reply.turn is not None:
                if reply.played == 0:
                    pending_notes.append(
                        f'[Your last reply was not spoken because the candidate kept talking: "{reply.turn.say}". '
                        "They have not heard it.]"
                    )
                else:
                    pending_notes.append(
                        f'[The candidate started talking while you were speaking. They only heard: "{reply.spoken}"]'
                    )
            elif reply.played:
                pending_notes.append(f'[You were interrupted after saying: "{reply.spoken}". The rest was not heard.]')
            self._log_reply(reply, interrupted=True)
            reply = None

        def handle_finished_thinking(r: Reply) -> None:
            """Claude has replied: the words and notes it saw are now answered."""
            nonlocal pending_text, ending
            r.handled = True
            if pending_text.startswith(r.consumed_text):
                pending_text = pending_text[len(r.consumed_text):].strip()
            for note in r.consumed_notes:
                if note in pending_notes:
                    pending_notes.remove(note)
            assert r.turn is not None
            if self.brain.consent == "no" and self.recorder.enabled:
                self.recorder.discard()
                log.info("Recording consent refused: recording discarded")
            if r.turn.note and self.echo:
                print(f"           (note: {r.turn.note})", flush=True)
            if r.turn.end_call:
                ending = True

        start_reply()

        while True:
            now = time.monotonic()
            if self._elapsed() > max_s + 60:  # one minute of grace for the goodbye
                return "time_limit"

            # Did the candidate hang up?
            if self.phone is not None and now - last_state_check > 1.5:
                last_state_check = now
                try:
                    if await self.phone.call_state() == CALL_STATE_IDLE:
                        if reply is not None and reply.task is not None:
                            reply.task.cancel()
                        return "candidate_hung_up"
                except PhoneError:
                    pass

            # Progress of the current reply.
            if reply is not None:
                if reply.brain_done and not reply.handled:
                    handle_finished_thinking(reply)
                assert reply.task is not None
                if reply.task.done():
                    if not reply.task.cancelled() and reply.task.exception() is not None:
                        log.error("Reply failed: %s", reply.task.exception())
                    self._log_reply(reply, interrupted=False)
                    reply = None

            if ending and reply is None:
                await self.engine.wait_until_played()
                await asyncio.sleep(0.8)
                return "agent_ended"

            # Reply to whatever the candidate said, once both sides have finished talking.
            candidate_talking = time.monotonic() - last_partial_at < CANDIDATE_TALKING_GRACE_S
            if reply is None and pending_text and self.engine.player.is_idle() and not candidate_talking:
                start_reply()

            # Speech events.
            try:
                event: SpeechEvent | None = await asyncio.wait_for(self.stt.events.get(), timeout=0.1)
            except asyncio.TimeoutError:
                event = None

            if event is not None:
                if event.kind == "error":
                    return "speech_error"
                last_activity = time.monotonic()
                silences = 0
                words = len(event.text.split())
                if event.kind == "partial" and words:
                    last_partial_at = time.monotonic()
                if event.kind == "partial" and reply is not None and not ending:
                    agent_audible = reply.played > 0 or not self.engine.player.is_idle()
                    if agent_audible and words >= BARGE_IN_MIN_WORDS:
                        cancel_reply()  # candidate is talking over the agent
                    elif not agent_audible and words >= STILL_TALKING_MIN_WORDS:
                        cancel_reply()  # candidate is still talking; reply once they finish
                elif event.kind == "turn_end":
                    last_partial_at = 0.0  # they've finished this stretch of speech
                    self._log("candidate", event.text)
                    pending_text = f"{pending_text} {event.text}".strip()
                    if reply is not None and reply.played == 0 and not ending:
                        cancel_reply()  # not spoken yet: redo it with everything they said
                continue

            # Silence handling (only when nobody is talking and nothing is pending).
            if reply is not None or not self.engine.player.is_idle() or pending_text:
                last_activity = time.monotonic()
            elif time.monotonic() - last_activity > silence_s:
                silences += 1
                last_activity = time.monotonic()
                if silences > max_silences:
                    return "silence"
                if silences == max_silences:
                    pending_notes.append(
                        "[Still no response. Say the line seems unclear, that the team will try again later, "
                        "say goodbye and end the call.]"
                    )
                else:
                    pending_notes.append(
                        f"[The candidate has been silent for {int(silence_s)} seconds. Check if they can hear you.]"
                    )
                start_reply()
