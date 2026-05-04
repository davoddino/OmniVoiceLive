from __future__ import annotations

import io
import math
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Literal

import numpy as np


@dataclass
class AudioEvent:
    type: Literal["speech_start", "speech_end", "meter"]
    rms: float = 0.0
    samples: np.ndarray | None = None
    sample_rate: int | None = None
    duration_ms: int = 0


def ensure_mono_float32(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=0) if data.shape[0] <= data.shape[1] else data.mean(axis=1)
    return np.clip(data.reshape(-1), -1.0, 1.0).astype(np.float32, copy=False)


def float32_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    mono = ensure_mono_float32(samples)
    pcm = (mono * 32767.0).clip(-32768, 32767).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out.getvalue()


def pcm16_bytes_from_float32(samples: np.ndarray) -> bytes:
    mono = ensure_mono_float32(samples)
    pcm = (mono * 32767.0).clip(-32768, 32767).astype("<i2")
    return pcm.tobytes()


def iter_pcm_frames(pcm: bytes, sample_rate: int, frame_ms: int) -> Iterable[bytes]:
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    frame_bytes = frame_samples * 2
    for start in range(0, len(pcm), frame_bytes):
        chunk = pcm[start : start + frame_bytes]
        if chunk:
            yield chunk


def trim_low_amplitude_edges(
    samples: np.ndarray,
    sample_rate: int,
    threshold: float = 2e-4,
    max_trim_ms: int = 140,
    keep_ms: int = 8,
) -> np.ndarray:
    mono = ensure_mono_float32(samples)
    if mono.size == 0:
        return mono

    active = np.flatnonzero(np.abs(mono) > threshold)
    if active.size == 0:
        return mono

    max_trim = int(sample_rate * max_trim_ms / 1000)
    keep = int(sample_rate * keep_ms / 1000)
    start = max(0, min(int(active[0]), max_trim) - keep)
    end_edge = mono.size - int(active[-1]) - 1
    end_trim = min(max(0, end_edge), max_trim)
    end = mono.size - max(0, end_trim - keep)
    return mono[start:end].astype(np.float32, copy=False)


def apply_edge_fade(samples: np.ndarray, sample_rate: int, fade_ms: int = 8) -> np.ndarray:
    mono = ensure_mono_float32(samples).copy()
    n = min(int(sample_rate * fade_ms / 1000), mono.size // 2)
    if n <= 0:
        return mono
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    mono[:n] *= fade_in
    mono[-n:] *= fade_out
    return mono


class AudioTurnDetector:
    def __init__(
        self,
        sample_rate: int,
        speech_threshold: float,
        start_ms: int,
        end_ms: int,
        min_turn_ms: int,
        preroll_ms: int,
        max_turn_s: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.min_turn_ms = min_turn_ms
        self.preroll_ms = preroll_ms
        self.max_turn_s = max_turn_s
        self._last_meter_at = 0.0
        self.reset()

    def set_sample_rate(self, sample_rate: int) -> None:
        if sample_rate > 0 and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._captured: list[np.ndarray] = []
        self._captured_ms = 0.0
        self._preroll: Deque[np.ndarray] = deque()
        self._preroll_ms = 0.0

    def accept(self, chunk: np.ndarray) -> list[AudioEvent]:
        mono = ensure_mono_float32(chunk)
        if mono.size == 0:
            return []

        duration_ms = mono.size / self.sample_rate * 1000.0
        rms = float(math.sqrt(float(np.mean(mono * mono)) + 1e-12))
        is_voice = rms >= self.speech_threshold
        events: list[AudioEvent] = []

        now = time.monotonic()
        if now - self._last_meter_at >= 0.1:
            self._last_meter_at = now
            events.append(AudioEvent(type="meter", rms=rms))

        if is_voice:
            self._speech_ms += duration_ms
            self._silence_ms = 0.0
        else:
            self._speech_ms = 0.0
            self._silence_ms += duration_ms

        if self.in_speech:
            self._captured.append(mono.copy())
            self._captured_ms += duration_ms
            if self._should_end_turn():
                events.append(self._end_event())
            return events

        self._push_preroll(mono, duration_ms)
        if is_voice and self._speech_ms >= self.start_ms:
            self.in_speech = True
            self._captured = [part.copy() for part in self._preroll]
            self._captured_ms = self._preroll_ms
            self._silence_ms = 0.0
            events.append(AudioEvent(type="speech_start", rms=rms))

        return events

    def _push_preroll(self, mono: np.ndarray, duration_ms: float) -> None:
        self._preroll.append(mono.copy())
        self._preroll_ms += duration_ms
        while self._preroll_ms > self.preroll_ms and self._preroll:
            old = self._preroll.popleft()
            self._preroll_ms -= old.size / self.sample_rate * 1000.0

    def _should_end_turn(self) -> bool:
        if self._captured_ms >= self.max_turn_s * 1000.0:
            return True
        return self._captured_ms >= self.min_turn_ms and self._silence_ms >= self.end_ms

    def _end_event(self) -> AudioEvent:
        samples = (
            np.concatenate(self._captured).astype(np.float32, copy=False)
            if self._captured
            else np.zeros(0, dtype=np.float32)
        )
        duration_ms = int(self._captured_ms)
        event = AudioEvent(
            type="speech_end",
            samples=samples,
            sample_rate=self.sample_rate,
            duration_ms=duration_ms,
        )
        self.reset()
        return event
