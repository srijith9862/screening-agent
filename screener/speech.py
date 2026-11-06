"""Deepgram speech-to-text (streaming) and text-to-speech."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx
import numpy as np

from .audio import pcm16_to_float

log = logging.getLogger(__name__)

DG_LISTEN_URL = "wss://api.deepgram.com/v1/listen"
DG_SPEAK_URL = "https://api.deepgram.com/v1/speak"
TTS_RATE = 24000


@dataclass
class SpeechEvent:
    kind: str  # "partial" | "final" | "turn_end" | "speech_started"
    text: str = ""


class SpeechToText:
    """Streams 16 kHz PCM to Deepgram and emits SpeechEvents on a queue."""

    def __init__(self, api_key: str, model: str, language: str, endpointing_ms: int, utterance_end_ms: int) -> None:
        params = {
            "model": model,
            "language": language,
            "encoding": "linear16",
            "sample_rate": 16000,
            "channels": 1,
            "interim_results": "true",
            "smart_format": "true",
            "punctuate": "true",
            "endpointing": endpointing_ms,
            "utterance_end_ms": utterance_end_ms,
            "vad_events": "true",
        }
        self._url = f"{DG_LISTEN_URL}?{urlencode(params)}"
        self._api_key = api_key
        self.events: asyncio.Queue[SpeechEvent] = asyncio.Queue()
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._ws: Any = None

    def feed(self, pcm16: bytes) -> None:
        """Called from the audio thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._audio.put_nowait, pcm16)

    async def start(self) -> None:
        from websockets.asyncio.client import connect

        self._loop = asyncio.get_running_loop()
        self._ws = await connect(self._url, additional_headers={"Authorization": f"Token {self._api_key}"})
        self._tasks = [asyncio.create_task(self._sender()), asyncio.create_task(self._receiver())]

    async def stop(self) -> None:
        await self._audio.put(None)
        try:
            if self._ws is not None:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:  # noqa: BLE001 - socket may already be closed
            pass
        for task in self._tasks:
            task.cancel()
        if self._ws is not None:
            await self._ws.close()

    async def _sender(self) -> None:
        buffer = bytearray()
        while True:
            try:
                chunk = await asyncio.wait_for(self._audio.get(), timeout=5)
            except asyncio.TimeoutError:
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
                continue
            if chunk is None:
                return
            buffer.extend(chunk)
            if len(buffer) >= 3200:  # ~100 ms
                await self._ws.send(bytes(buffer))
                buffer.clear()

    async def _receiver(self) -> None:
        pending: list[str] = []
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "SpeechStarted":
                    await self.events.put(SpeechEvent("speech_started"))
                elif kind == "Results":
                    alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
                    text = (alt.get("transcript") or "").strip()
                    if not msg.get("is_final"):
                        if text:
                            await self.events.put(SpeechEvent("partial", " ".join(pending + [text])))
                        continue
                    if text:
                        pending.append(text)
                        await self.events.put(SpeechEvent("final", text))
                    if msg.get("speech_final") and pending:
                        await self.events.put(SpeechEvent("turn_end", " ".join(pending)))
                        pending = []
                elif kind == "UtteranceEnd" and pending:
                    await self.events.put(SpeechEvent("turn_end", " ".join(pending)))
                    pending = []
        except Exception as exc:  # noqa: BLE001
            log.error("Speech-to-text connection closed: %s", exc)
            await self.events.put(SpeechEvent("error", str(exc)))


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class TextToSpeech:
    def __init__(self, api_key: str, voice: str) -> None:
        self._client = httpx.AsyncClient(timeout=20)
        self._api_key = api_key
        self._voice = voice

    async def synthesize(self, text: str) -> np.ndarray:
        response = await self._client.post(
            DG_SPEAK_URL,
            params={"model": self._voice, "encoding": "linear16", "sample_rate": TTS_RATE, "container": "none"},
            headers={"Authorization": f"Token {self._api_key}", "Content-Type": "application/json"},
            json={"text": text},
        )
        response.raise_for_status()
        return pcm16_to_float(response.content)

    async def close(self) -> None:
        await self._client.aclose()
