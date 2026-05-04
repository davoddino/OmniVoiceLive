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
    anchor_source: str = ""


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
        self._startup_voice_prompt: Any | None = None
        self._startup_anchor_text = ""
        self._startup_anchor_duration_s = 0.0

    async def start(self) -> None:
        voice_mode = self._voice_mode()
        logger.info(
            "omnivoice loading model=%s device_map=%s voice_mode=%s instruct=%r",
            self.config.tts_model,
            self.config.tts_device_map,
            voice_mode,
            self.config.tts_instruct,
        )
        await asyncio.to_thread(self._load_model)
        if voice_mode == "session_anchor" and self.config.tts_startup_voice_anchor:
            anchor_text = self.config.tts_startup_anchor_text.strip()
            if anchor_text:
                logger.info("omnivoice startup voice anchor text=%r", anchor_text)
                anchor_state = TTSTurnState()
                await self.synthesize(anchor_text, first=True, state=anchor_state)
                if anchor_state.voice_prompt is not None:
                    self._startup_voice_prompt = anchor_state.voice_prompt
                    self._startup_anchor_text = anchor_state.anchor_text
                    self._startup_anchor_duration_s = anchor_state.anchor_duration_s
                    logger.info(
                        "omnivoice startup voice anchor ready duration_s=%.2f chars=%s",
                        self._startup_anchor_duration_s,
                        len(self._startup_anchor_text),
                    )
        elif self.config.tts_warmup_enabled:
            logger.info("omnivoice warmup text=%r", self.config.tts_warmup_text)
            await self.synthesize(self.config.tts_warmup_text, first=True)
        logger.info("omnivoice ready sample_rate=%s", self.sample_rate)

    def create_turn_state(self) -> TTSTurnState:
        if (
            self._voice_mode() == "session_anchor"
            and self._startup_voice_prompt is not None
        ):
            return TTSTurnState(
                voice_prompt=self._startup_voice_prompt,
                anchor_text=self._startup_anchor_text,
                anchor_duration_s=self._startup_anchor_duration_s,
                anchor_source="startup",
            )
        return TTSTurnState()

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
        voice_mode = self._voice_mode()
        use_anchor = (
            voice_mode in {"turn_anchor", "session_anchor"}
            and state is not None
            and state.voice_prompt is not None
        )
        with self._lock:
            kwargs = {
                "text": text,
                "language": self.config.tts_language,
                "num_step": num_step,
                "speed": self.config.tts_speed,
                "guidance_scale": self.config.tts_guidance_scale,
                "position_temperature": self.config.tts_position_temperature,
                "class_temperature": self.config.tts_class_temperature,
                "postprocess_output": self.config.tts_postprocess_output,
                "denoise": self.config.tts_denoise,
            }
            if self.config.tts_instruct.strip():
                kwargs["instruct"] = self.config.tts_instruct
            if use_anchor:
                kwargs["voice_clone_prompt"] = state.voice_prompt

            audio = self.model.generate(**kwargs)

            raw_waveform = np.asarray(audio[0], dtype=np.float32).reshape(-1)
            self._maybe_create_anchor(state, text, raw_waveform, "generated", voice_mode)

        waveform = raw_waveform
        waveform = trim_low_amplitude_edges(waveform, self.sample_rate)
        waveform = apply_edge_fade(waveform, self.sample_rate)
        return np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)

    def _maybe_create_anchor(
        self,
        state: TTSTurnState | None,
        text: str,
        waveform: np.ndarray,
        source: str,
        voice_mode: str,
    ) -> None:
        if voice_mode not in {"turn_anchor", "session_anchor"} or state is None:
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
            state.anchor_source = source
            logger.info(
                "omnivoice anchor created source=%s duration_s=%.2f chars=%s",
                source,
                duration_s,
                len(text),
            )
        except Exception:
            logger.exception("omnivoice anchor creation failed; continuing without it")

    def _voice_mode(self) -> str:
        mode = self.config.tts_voice_mode.strip().lower().replace("-", "_")
        if mode in {"design", "voice_design", "instruct"}:
            return "voice_design"
        if mode in {"turn_anchor", "self_condition", "turn_self_condition"}:
            return "turn_anchor"
        if mode in {"session_anchor", "anchor", "session_self_condition"}:
            return "session_anchor"
        raise ValueError(
            "Unsupported LIVE_TTS_VOICE_MODE. Use voice_design, turn_anchor, or session_anchor."
        )


def create_tts(config: LiveTTSConfig) -> BaseTTS:
    backend = config.tts_backend.strip().lower()
    if backend == "mock":
        return MockTTS()
    if backend == "omnivoice":
        return OmniVoiceTTS(config)
    raise ValueError(f"Unsupported LIVE_TTS_TTS_BACKEND: {config.tts_backend}")
