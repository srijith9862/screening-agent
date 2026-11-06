"""Audio in/out for the call, plus the two-track recording.

Call audio path on Windows (see README):
  Phone Link speaker -> "CABLE Input"   -> we read  "CABLE Output"   (candidate's voice)
  we write "Hi-Fi Cable Input" -> "Hi-Fi Cable Output" -> Phone Link mic   (agent's voice)
"""

from __future__ import annotations

import asyncio
import threading
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soxr

STT_RATE = 16000  # what we send to speech-to-text and store in the recording
BLOCK_SECONDS = 0.02


def _sd() -> Any:
    import sounddevice  # imported lazily so tests run without PortAudio

    return sounddevice


def list_devices() -> str:
    sd = _sd()
    lines = []
    for index, dev in enumerate(sd.query_devices()):
        kinds = []
        if dev["max_input_channels"] > 0:
            kinds.append("in")
        if dev["max_output_channels"] > 0:
            kinds.append("out")
        host = sd.query_hostapis(dev["hostapi"])["name"]
        lines.append(f"{index:3d}  [{'/'.join(kinds):6s}] {dev['name']}  ({host}, {int(dev['default_samplerate'])} Hz)")
    return "\n".join(lines)


def find_device(name_part: str | None, kind: str) -> int | None:
    """Finds a device whose name contains name_part. None means the system default."""
    if not name_part:
        return None
    sd = _sd()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if name_part.lower() in d["name"].lower() and d[key] > 0
    ]
    if not matches:
        raise RuntimeError(f"No {kind} device matching '{name_part}'. Run `python -m screener devices`.")
    # Prefer the Windows WASAPI/MME variant consistently: take the first match.
    return matches[0][0]


def to_int16_bytes(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or samples.size == 0:
        return samples.astype(np.float32)
    return soxr.resample(samples.astype(np.float32), from_rate, to_rate).astype(np.float32)


class Recorder:
    """Stereo 16 kHz recording: left = candidate, right = agent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._left: list[np.ndarray] = []
        self._right: list[np.ndarray] = []
        self.enabled = True

    def add_candidate(self, samples_16k: np.ndarray) -> None:
        if self.enabled:
            with self._lock:
                self._left.append(samples_16k)

    def add_agent(self, samples_16k: np.ndarray) -> None:
        if self.enabled:
            with self._lock:
                self._right.append(samples_16k)

    def discard(self) -> None:
        """Deletes everything recorded so far and stops recording (consent refused)."""
        with self._lock:
            self.enabled = False
            self._left.clear()
            self._right.clear()

    def save(self, path: Path) -> bool:
        with self._lock:
            if not self._left and not self._right:
                return False
            left = np.concatenate(self._left) if self._left else np.zeros(0, np.float32)
            right = np.concatenate(self._right) if self._right else np.zeros(0, np.float32)
        length = max(left.size, right.size)
        stereo = np.zeros((length, 2), dtype=np.float32)
        stereo[: left.size, 0] = left
        stereo[: right.size, 1] = right
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(STT_RATE)
            wav.writeframes(to_int16_bytes(stereo.reshape(-1)))
        return True


class Player:
    """Thread-safe playback queue read by the output stream callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._current = np.zeros(0, np.float32)
        self.samples_played = 0

    def enqueue(self, samples: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(samples.astype(np.float32))

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._current = np.zeros(0, np.float32)

    def is_idle(self) -> bool:
        with self._lock:
            return self._current.size == 0 and not self._chunks

    def pull(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, np.float32)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._current.size == 0:
                    if not self._chunks:
                        break
                    self._current = self._chunks.popleft()
                take = min(frames - filled, self._current.size)
                out[filled : filled + take] = self._current[:take]
                self._current = self._current[take:]
                filled += take
        self.samples_played += filled
        return out


class AudioEngine:
    """Opens the call-in and call-out devices and wires them to STT, playback and recording."""

    def __init__(
        self,
        in_device: str | None,
        out_device: str | None,
        on_candidate_audio: Callable[[bytes], None],
        recorder: Recorder,
    ) -> None:
        sd = _sd()
        self._in_index = find_device(in_device, "input")
        self._out_index = find_device(out_device, "output")
        self.in_rate = int(sd.query_devices(self._in_index, "input")["default_samplerate"])
        self.out_rate = int(sd.query_devices(self._out_index, "output")["default_samplerate"])
        self._on_candidate_audio = on_candidate_audio
        self.recorder = recorder
        self.player = Player()
        self._in_resampler = soxr.ResampleStream(self.in_rate, STT_RATE, 1, dtype="float32")
        self._out_resampler = soxr.ResampleStream(self.out_rate, STT_RATE, 1, dtype="float32")
        self.input_level = 0.0
        self._streams: list[Any] = []

    def _input_callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        mono = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.astype(np.float32)
        self.input_level = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        samples = self._in_resampler.resample_chunk(mono)
        self.recorder.add_candidate(samples)
        self._on_candidate_audio(to_int16_bytes(samples))

    def _output_callback(self, outdata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        samples = self.player.pull(frames)
        outdata[:] = samples.reshape(-1, 1).repeat(outdata.shape[1], axis=1)
        self.recorder.add_agent(self._out_resampler.resample_chunk(samples))

    def start(self) -> None:
        sd = _sd()
        in_stream = sd.InputStream(
            device=self._in_index,
            channels=1,
            samplerate=self.in_rate,
            dtype="float32",
            blocksize=int(self.in_rate * BLOCK_SECONDS),
            callback=self._input_callback,
        )
        out_stream = sd.OutputStream(
            device=self._out_index,
            channels=1,
            samplerate=self.out_rate,
            dtype="float32",
            blocksize=int(self.out_rate * BLOCK_SECONDS),
            callback=self._output_callback,
        )
        in_stream.start()
        out_stream.start()
        self._streams = [in_stream, out_stream]

    def stop(self) -> None:
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
        self._streams = []

    def play(self, samples: np.ndarray, rate: int) -> None:
        self.player.enqueue(resample(samples, rate, self.out_rate))

    async def wait_until_played(self, poll: float = 0.05) -> None:
        while not self.player.is_idle():
            await asyncio.sleep(poll)


def tone(seconds: float = 1.0, rate: int = 48000, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
