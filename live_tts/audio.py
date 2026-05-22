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
    noise_rms: float = 0.0
    start_threshold: float = 0.0
    continue_threshold: float = 0.0
    in_speech: bool = False
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


def pad_audio_edges(
    samples: np.ndarray,
    sample_rate: int,
    lead_ms: int = 0,
    tail_ms: int = 0,
) -> np.ndarray:
    mono = ensure_mono_float32(samples)
    lead_samples = max(0, int(sample_rate * lead_ms / 1000))
    tail_samples = max(0, int(sample_rate * tail_ms / 1000))
    if lead_samples <= 0 and tail_samples <= 0:
        return mono

    parts: list[np.ndarray] = []
    if lead_samples > 0:
        parts.append(np.zeros(lead_samples, dtype=np.float32))
    parts.append(mono)
    if tail_samples > 0:
        parts.append(np.zeros(tail_samples, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32, copy=False)


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


def trim_tts_onset_noise(
    samples: np.ndarray,
    sample_rate: int,
    threshold: float = 0.010,
    window_ms: int = 8,
    keep_ms: int = 3,
    max_trim_ms: int = 80,
) -> np.ndarray:
    mono = ensure_mono_float32(samples)
    if mono.size == 0:
        return mono

    max_trim = min(mono.size, int(sample_rate * max_trim_ms / 1000))
    window = max(1, int(sample_rate * window_ms / 1000))
    keep = max(0, int(sample_rate * keep_ms / 1000))
    if max_trim <= window:
        return mono

    start = 0
    for idx in range(0, max_trim - window + 1):
        frame = mono[idx : idx + window]
        rms = math.sqrt(float(np.mean(frame * frame)) + 1e-12)
        peak = float(np.max(np.abs(frame)))
        if rms >= threshold or peak >= threshold * 2.5:
            start = max(0, idx - keep)
            break
    else:
        return mono

    return mono[start:].astype(np.float32, copy=False)


def apply_edge_fade(
    samples: np.ndarray,
    sample_rate: int,
    fade_ms: int = 8,
    fade_in_ms: int | None = None,
    fade_out_ms: int | None = None,
) -> np.ndarray:
    mono = ensure_mono_float32(samples).copy()
    if fade_in_ms is None and fade_out_ms is None:
        fade_in_ms = fade_ms
        fade_out_ms = fade_ms
    fade_in_ms = max(0, int(fade_in_ms or 0))
    fade_out_ms = max(0, int(fade_out_ms or 0))
    if mono.size == 0 or (fade_in_ms <= 0 and fade_out_ms <= 0):
        return mono
    fade_in_samples = min(int(sample_rate * fade_in_ms / 1000), mono.size // 2)
    if fade_in_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        mono[:fade_in_samples] *= fade_in
    fade_out_samples = min(int(sample_rate * fade_out_ms / 1000), mono.size // 2)
    if fade_out_samples > 0:
        fade_out = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
        mono[-fade_out_samples:] *= fade_out
    return mono


def rms_loudness_gain(
    samples: np.ndarray,
    target_lufs: float = -16.0,
    max_gain_db: float = 9.0,
    min_gain_db: float = -12.0,
) -> float:
    mono = ensure_mono_float32(samples)
    if mono.size == 0:
        return 1.0
    active = mono[np.abs(mono) > 1e-4]
    if active.size < max(16, mono.size // 100):
        active = mono
    rms = float(math.sqrt(float(np.mean(active * active)) + 1e-12))
    if rms <= 1e-6:
        return 1.0

    target_rms = 10.0 ** (target_lufs / 20.0)
    gain = target_rms / rms
    min_gain = 10.0 ** (min_gain_db / 20.0)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    return min(max(gain, min_gain), max_gain)


def normalize_loudness_rms(
    samples: np.ndarray,
    target_lufs: float = -16.0,
    enabled: bool = True,
    max_gain_db: float = 9.0,
    min_gain_db: float = -12.0,
    peak_limit: float = 0.98,
) -> np.ndarray:
    mono = ensure_mono_float32(samples)
    if not enabled or mono.size == 0:
        return mono

    gain = rms_loudness_gain(
        mono,
        target_lufs=target_lufs,
        max_gain_db=max_gain_db,
        min_gain_db=min_gain_db,
    )
    normalized = mono * gain
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > peak_limit:
        normalized = normalized * (peak_limit / peak)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


class AudioLoudnessSmoother:
    def __init__(
        self,
        target_lufs: float = -16.0,
        enabled: bool = True,
        smoothing: float = 0.72,
        max_gain_db: float = 9.0,
        min_gain_db: float = -12.0,
        peak_limit: float = 0.98,
    ) -> None:
        self.target_lufs = target_lufs
        self.enabled = enabled
        self.smoothing = min(max(smoothing, 0.0), 0.98)
        self.max_gain_db = max_gain_db
        self.min_gain_db = min_gain_db
        self.peak_limit = peak_limit
        self._gain: float | None = None

    def process(self, samples: np.ndarray) -> np.ndarray:
        mono = ensure_mono_float32(samples)
        if not self.enabled or mono.size == 0:
            return mono

        desired_gain = rms_loudness_gain(
            mono,
            target_lufs=self.target_lufs,
            max_gain_db=self.max_gain_db,
            min_gain_db=self.min_gain_db,
        )
        if self._gain is None:
            gain = desired_gain
        else:
            gain = self._gain * self.smoothing + desired_gain * (1.0 - self.smoothing)

        normalized = mono * gain
        peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        if peak > self.peak_limit:
            limiter_gain = self.peak_limit / peak
            normalized = normalized * limiter_gain
            gain *= limiter_gain

        self._gain = gain
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


class AudioCrossfader:
    def __init__(self, sample_rate: int, crossfade_ms: int) -> None:
        self.sample_rate = sample_rate
        self.crossfade_samples = max(0, int(sample_rate * crossfade_ms / 1000))
        self._tail: np.ndarray | None = None

    def process(self, samples: np.ndarray) -> np.ndarray:
        mono = ensure_mono_float32(samples)
        n = self.crossfade_samples
        if n <= 0 or mono.size <= n * 2:
            if self._tail is None:
                return mono
            out = np.concatenate([self._tail, mono]).astype(np.float32, copy=False)
            self._tail = None
            return out

        if self._tail is None:
            self._tail = mono[-n:].copy()
            return mono[:-n].astype(np.float32, copy=False)

        n = min(n, self._tail.size, mono.size // 2)
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
        cross = self._tail[-n:] * fade_out + mono[:n] * fade_in
        body = mono[n:-n]
        self._tail = mono[-n:].copy()
        if body.size:
            return np.concatenate([cross, body]).astype(np.float32, copy=False)
        return cross.astype(np.float32, copy=False)

    def flush(self) -> np.ndarray:
        if self._tail is None:
            return np.zeros(0, dtype=np.float32)
        tail = self._tail
        self._tail = None
        return tail.astype(np.float32, copy=False)


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
        adaptive: bool = True,
        noise_calibration_ms: int = 1000,
        start_multiplier: float = 2.2,
        continue_multiplier: float = 1.4,
    ) -> None:
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.min_turn_ms = min_turn_ms
        self.preroll_ms = preroll_ms
        self.max_turn_s = max_turn_s
        self.adaptive = adaptive
        self.noise_calibration_ms = noise_calibration_ms
        self.start_multiplier = start_multiplier
        self.continue_multiplier = continue_multiplier
        self._last_meter_at = 0.0
        self.noise_rms = max(1e-5, speech_threshold / max(start_multiplier, 1.0))
        self.start_threshold = speech_threshold
        self.continue_threshold = max(speech_threshold * 0.6, self.noise_rms)
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
        self._calibrated_ms = 0.0

    def accept(self, chunk: np.ndarray) -> list[AudioEvent]:
        mono = ensure_mono_float32(chunk)
        if mono.size == 0:
            return []

        duration_ms = mono.size / self.sample_rate * 1000.0
        rms = float(math.sqrt(float(np.mean(mono * mono)) + 1e-12))
        self._update_thresholds(rms, duration_ms)
        threshold = self.continue_threshold if self.in_speech else self.start_threshold
        is_voice = rms >= threshold
        events: list[AudioEvent] = []

        now = time.monotonic()
        if now - self._last_meter_at >= 0.1:
            self._last_meter_at = now
            events.append(
                AudioEvent(
                    type="meter",
                    rms=rms,
                    noise_rms=self.noise_rms,
                    start_threshold=self.start_threshold,
                    continue_threshold=self.continue_threshold,
                    in_speech=self.in_speech,
                )
            )

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
            events.append(
                AudioEvent(
                    type="speech_start",
                    rms=rms,
                    noise_rms=self.noise_rms,
                    start_threshold=self.start_threshold,
                    continue_threshold=self.continue_threshold,
                    in_speech=True,
                )
            )

        return events

    def _update_thresholds(self, rms: float, duration_ms: float) -> None:
        if not self.adaptive:
            self.noise_rms = max(1e-5, self.speech_threshold / max(self.start_multiplier, 1.0))
            self.start_threshold = self.speech_threshold
            self.continue_threshold = max(self.speech_threshold * 0.6, self.noise_rms)
            return

        previous_start = self.start_threshold
        can_update_noise = not self.in_speech and rms < max(previous_start, self.speech_threshold * 1.4)
        if can_update_noise:
            if self._calibrated_ms < self.noise_calibration_ms:
                total_ms = self._calibrated_ms + duration_ms
                old_weight = self._calibrated_ms / max(total_ms, 1e-6)
                new_weight = duration_ms / max(total_ms, 1e-6)
                self.noise_rms = self.noise_rms * old_weight + rms * new_weight
                self._calibrated_ms = total_ms
            else:
                self.noise_rms = self.noise_rms * 0.98 + rms * 0.02

        floor = max(1e-5, self.noise_rms)
        self.start_threshold = max(
            self.speech_threshold,
            floor * max(1.0, self.start_multiplier),
        )
        self.continue_threshold = max(
            self.speech_threshold * 0.55,
            floor * max(1.0, self.continue_multiplier),
        )

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
