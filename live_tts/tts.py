from __future__ import annotations

import asyncio
import math
import threading
from abc import ABC, abstractmethod

import numpy as np

from live_tts.audio import apply_edge_fade, trim_low_amplitude_edges
from live_tts.config import LiveTTSConfig


class BaseTTS(ABC):
    sample_rate: int = 24000

    async def start(self) -> None:
        return None

    @abstractmethod
    async def synthesize(self, text: str, first: bool) -> np.ndarray:
        raise NotImplementedError


class MockTTS(BaseTTS):
    sample_rate = 24000

    async def synthesize(self, text: str, first: bool) -> np.ndarray:
        duration = min(3.2, max(0.45, len(text) / 38.0))
        samples = int(self.sample_rate * duration)
        t = np.arange(samples, dtype=np.float32) / self.sample_rate
        base = 185.0 if first else 165.0
        wave = 0.12 * np.sin(2.0 * math.pi * base * t)
        wave += 0.045 * np.sin(2.0 * math.pi * base * 2.01 * t)
        envelope = np.ones_like(wave)
        fade = min(samples // 4, int(self.sample_rate * 0.035))
        if fade > 0:
            envelope[:fade] = np.linspace(0.0, 1.0, fade)
            envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        await asyncio.sleep(0.04 if first else 0.08)
        return (wave * envelope).astype(np.float32)


class OmniVoiceTTS(BaseTTS):
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config
        self.model = None
        self.sample_rate = 24000
        self._lock = threading.Lock()

    async def start(self) -> None:
        await asyncio.to_thread(self._load_model)
        if self.config.tts_warmup_enabled:
            await self.synthesize(self.config.tts_warmup_text, first=True)

    async def synthesize(self, text: str, first: bool) -> np.ndarray:
        return await asyncio.to_thread(self._synthesize_sync, text, first)

    def _load_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from omnivoice import OmniVoice

        dtype = self._resolve_dtype(torch)
        self.model = OmniVoice.from_pretrained(
            self.config.tts_model,
            device_map=self.config.tts_device_map,
            dtype=dtype,
        )
        self.sample_rate = int(self.model.sampling_rate or 24000)

    def _resolve_dtype(self, torch_module):
        value = self.config.tts_dtype.strip().lower()
        if value in {"float16", "fp16", "half"}:
            return torch_module.float16
        if value in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        if value in {"float32", "fp32"}:
            return torch_module.float32
        raise ValueError(f"Unsupported LIVE_TTS_DTYPE: {self.config.tts_dtype}")

    def _synthesize_sync(self, text: str, first: bool) -> np.ndarray:
        if self.model is None:
            self._load_model()
        assert self.model is not None

        num_step = (
            self.config.tts_num_step_first if first else self.config.tts_num_step_next
        )
        with self._lock:
            audio = self.model.generate(
                text=text,
                language=self.config.tts_language,
                instruct=self.config.tts_instruct,
                num_step=num_step,
                speed=self.config.tts_speed,
                guidance_scale=2.0,
                position_temperature=0.0,
                class_temperature=0.0,
                postprocess_output=False,
            )

        waveform = np.asarray(audio[0], dtype=np.float32).reshape(-1)
        waveform = trim_low_amplitude_edges(waveform, self.sample_rate)
        waveform = apply_edge_fade(waveform, self.sample_rate)
        return np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)


def create_tts(config: LiveTTSConfig) -> BaseTTS:
    backend = config.tts_backend.strip().lower()
    if backend == "mock":
        return MockTTS()
    if backend == "omnivoice":
        return OmniVoiceTTS(config)
    raise ValueError(f"Unsupported LIVE_TTS_TTS_BACKEND: {config.tts_backend}")
