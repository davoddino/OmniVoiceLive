from __future__ import annotations

import asyncio
import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from live_tts.audio import apply_edge_fade, trim_low_amplitude_edges
from live_tts.config import LiveTTSConfig


logger = logging.getLogger(__name__)


@dataclass
class TTSTurnState:
    voice_prompt: Any | None = None
    anchor_text: str = ""
    anchor_duration_s: float = 0.0


class BaseTTS(ABC):
    sample_rate: int = 24000

    async def start(self) -> None:
        return None

    def create_turn_state(self) -> TTSTurnState:
        return TTSTurnState()

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class MockTTS(BaseTTS):
    sample_rate = 24000

    async def synthesize(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None = None,
    ) -> np.ndarray:
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
        logger.info(
            "omnivoice loading model=%s device_map=%s",
            self.config.tts_model,
            self.config.tts_device_map,
        )
        await asyncio.to_thread(self._load_model)
        if self.config.tts_warmup_enabled:
            logger.info("omnivoice warmup text=%r", self.config.tts_warmup_text)
            await self.synthesize(self.config.tts_warmup_text, first=True)
        logger.info("omnivoice ready sample_rate=%s", self.sample_rate)

    async def synthesize(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None = None,
    ) -> np.ndarray:
        return await asyncio.to_thread(self._synthesize_sync, text, first, state)

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
        logger.info("omnivoice model loaded sample_rate=%s", self.sample_rate)

    def _resolve_dtype(self, torch_module):
        value = self.config.tts_dtype.strip().lower()
        if value in {"float16", "fp16", "half"}:
            return torch_module.float16
        if value in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        if value in {"float32", "fp32"}:
            return torch_module.float32
        raise ValueError(f"Unsupported LIVE_TTS_DTYPE: {self.config.tts_dtype}")

    def _synthesize_sync(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None,
    ) -> np.ndarray:
        if self.model is None:
            self._load_model()
        assert self.model is not None

        num_step = (
            self.config.tts_num_step_first if first else self.config.tts_num_step_next
        )
        use_anchor = (
            self.config.tts_self_condition
            and state is not None
            and state.voice_prompt is not None
            and not first
        )
        with self._lock:
            kwargs = {
                "text": text,
                "language": self.config.tts_language,
                "num_step": num_step,
                "speed": self.config.tts_speed,
                "guidance_scale": 2.0,
                "position_temperature": 0.0,
                "class_temperature": 0.0,
                "postprocess_output": False,
            }
            if use_anchor:
                kwargs["voice_clone_prompt"] = state.voice_prompt
            else:
                kwargs["instruct"] = self.config.tts_instruct

            audio = self.model.generate(**kwargs)

            raw_waveform = np.asarray(audio[0], dtype=np.float32).reshape(-1)
            self._maybe_create_anchor(state, text, raw_waveform)

        waveform = raw_waveform
        waveform = trim_low_amplitude_edges(waveform, self.sample_rate)
        waveform = apply_edge_fade(waveform, self.sample_rate)
        return np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)

    def _maybe_create_anchor(
        self,
        state: TTSTurnState | None,
        text: str,
        waveform: np.ndarray,
    ) -> None:
        if not self.config.tts_self_condition or state is None:
            return
        if state.voice_prompt is not None:
            return

        duration_s = waveform.size / self.sample_rate
        if duration_s < self.config.tts_anchor_min_seconds:
            logger.info(
                "omnivoice anchor skipped duration_s=%.2f min_s=%.2f chars=%s",
                duration_s,
                self.config.tts_anchor_min_seconds,
                len(text),
            )
            return

        try:
            state.voice_prompt = self.model.create_voice_clone_prompt(
                ref_audio=(waveform.astype(np.float32, copy=False), self.sample_rate),
                ref_text=text,
                preprocess_prompt=False,
            )
            state.anchor_text = text
            state.anchor_duration_s = duration_s
            logger.info(
                "omnivoice anchor created duration_s=%.2f chars=%s",
                duration_s,
                len(text),
            )
        except Exception:
            logger.exception("omnivoice anchor creation failed; continuing without it")


def create_tts(config: LiveTTSConfig) -> BaseTTS:
    backend = config.tts_backend.strip().lower()
    if backend == "mock":
        return MockTTS()
    if backend == "omnivoice":
        return OmniVoiceTTS(config)
    raise ValueError(f"Unsupported LIVE_TTS_TTS_BACKEND: {config.tts_backend}")
