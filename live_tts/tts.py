from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
import math
import random
import os
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from live_tts.audio import apply_edge_fade, trim_low_amplitude_edges
from live_tts.config import LiveTTSConfig
from live_tts.voice import VoiceSessionConfig


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

    async def close(self) -> None:
        return None

    def create_turn_state(self) -> TTSTurnState:
        return TTSTurnState()

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None = None,
        language: str | None = None,
        voice_config: VoiceSessionConfig | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class MockTTS(BaseTTS):
    sample_rate = 24000

    async def synthesize(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None = None,
        language: str | None = None,
        voice_config: VoiceSessionConfig | None = None,
    ) -> np.ndarray:
        duration = min(3.2, max(0.45, len(text) / 38.0))
        samples = int(self.sample_rate * duration)
        t = np.arange(samples, dtype=np.float32) / self.sample_rate
        base = 175.0
        wave = 0.12 * np.sin(2.0 * math.pi * base * t)
        wave += 0.045 * np.sin(2.0 * math.pi * base * 2.01 * t)
        envelope = np.ones_like(wave)
        fade = min(samples // 4, int(self.sample_rate * 0.035))
        if fade > 0:
            envelope[:fade] = np.linspace(0.0, 1.0, fade)
            envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        await asyncio.sleep(0.04 if first else 0.08)
        return (wave * envelope).astype(np.float32)


@dataclass(frozen=True)
class TTSEngineDefinition:
    id: str
    label: str
    description: str
    kind: str


class OmniVoiceTTS(BaseTTS):
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config
        self.model = None
        self.sample_rate = 24000
        self._lock = threading.Lock()
        self._startup_voice_prompt: Any | None = None
        self._startup_anchor_text = ""
        self._startup_anchor_duration_s = 0.0
        self._fixed_reference_voice_prompt: Any | None = None
        self._fixed_reference_text = ""
        self._fixed_reference_duration_s = 0.0

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
        if voice_mode == "fixed_reference":
            await asyncio.to_thread(self._load_fixed_reference_voice_prompt)
            if self.config.tts_warmup_enabled:
                logger.info(
                    "omnivoice fixed reference warmup text=%r",
                    self.config.tts_warmup_text,
                )
                voice_config = VoiceSessionConfig.from_config(
                    self.config,
                    self.sample_rate,
                )
                await self.synthesize(
                    self.config.tts_warmup_text,
                    first=True,
                    state=self.create_turn_state(),
                    voice_config=voice_config,
                )
        elif voice_mode == "session_anchor" and self.config.tts_startup_voice_anchor:
            anchor_text = self.config.tts_startup_anchor_text.strip()
            if anchor_text:
                logger.info("omnivoice startup voice anchor text=%r", anchor_text)
                anchor_state = TTSTurnState()
                voice_config = VoiceSessionConfig.from_config(
                    self.config,
                    self.sample_rate,
                )
                await self.synthesize(
                    anchor_text,
                    first=True,
                    state=anchor_state,
                    voice_config=voice_config,
                )
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
            voice_config = VoiceSessionConfig.from_config(self.config, self.sample_rate)
            await self.synthesize(
                self.config.tts_warmup_text,
                first=True,
                voice_config=voice_config,
            )
        logger.info("omnivoice ready sample_rate=%s", self.sample_rate)

    async def close(self) -> None:
        self.model = None
        self._startup_voice_prompt = None
        self._fixed_reference_voice_prompt = None
        self._startup_anchor_text = ""
        self._fixed_reference_text = ""
        self._startup_anchor_duration_s = 0.0
        self._fixed_reference_duration_s = 0.0
        await asyncio.to_thread(_release_torch_memory)

    def create_turn_state(self) -> TTSTurnState:
        if (
            self._voice_mode() == "fixed_reference"
            and self._fixed_reference_voice_prompt is not None
        ):
            return TTSTurnState(
                voice_prompt=self._fixed_reference_voice_prompt,
                anchor_text=self._fixed_reference_text,
                anchor_duration_s=self._fixed_reference_duration_s,
                anchor_source="fixed_reference",
            )
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
        language: str | None = None,
        voice_config: VoiceSessionConfig | None = None,
    ) -> np.ndarray:
        return await asyncio.to_thread(
            self._synthesize_sync,
            text,
            first,
            state,
            language,
            voice_config,
        )

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

    def _load_fixed_reference_voice_prompt(self) -> None:
        if self.model is None:
            self._load_model()
        assert self.model is not None
        if self._fixed_reference_voice_prompt is not None:
            return

        ref_audio_path = _resolve_path(self.config.tts_reference_audio)
        ref_text = self.config.tts_reference_text.strip()
        if not ref_audio_path.is_file():
            raise RuntimeError(
                "LIVE_TTS_VOICE_MODE=fixed_reference requires "
                f"LIVE_TTS_REFERENCE_AUDIO={ref_audio_path}. "
                "Generate candidates first with: "
                "uv run python scripts/generate_voice_candidates.py"
            )
        if not ref_text:
            raise RuntimeError(
                "LIVE_TTS_REFERENCE_TEXT is required for fixed_reference mode."
            )

        logger.info(
            "omnivoice fixed reference loading audio=%s chars=%s",
            ref_audio_path,
            len(ref_text),
        )
        self._fixed_reference_voice_prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
            preprocess_prompt=self.config.tts_reference_preprocess,
        )
        self._clear_cuda_cache()
        self._fixed_reference_text = ref_text
        self._fixed_reference_duration_s = _audio_duration_seconds(ref_audio_path)
        logger.info(
            "omnivoice fixed reference ready duration_s=%.2f chars=%s",
            self._fixed_reference_duration_s,
            len(ref_text),
        )

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
        language: str | None,
        voice_config: VoiceSessionConfig | None,
    ) -> np.ndarray:
        if self.model is None:
            self._load_model()
        assert self.model is not None

        voice_config = voice_config or VoiceSessionConfig.from_config(
            self.config,
            self.sample_rate,
            language=language,
        )

        num_step = (
            voice_config.num_step_first if first else voice_config.num_step_next
        )
        voice_mode = self._voice_mode()
        use_anchor = (
            voice_mode in {"fixed_reference", "turn_anchor", "session_anchor"}
            and state is not None
            and state.voice_prompt is not None
        )
        with self._lock:
            self._apply_seed(voice_config.seed)
            kwargs = {
                "text": text,
                "language": language or voice_config.language or self.config.tts_language,
                "num_step": num_step,
                "speed": voice_config.speed,
                "guidance_scale": voice_config.guidance_scale,
                "position_temperature": voice_config.position_temperature,
                "class_temperature": voice_config.class_temperature,
                "postprocess_output": self.config.tts_postprocess_output,
                "denoise": self.config.tts_denoise,
            }
            if voice_config.instruct and voice_mode != "fixed_reference":
                kwargs["instruct"] = voice_config.instruct
            if use_anchor:
                kwargs["voice_clone_prompt"] = state.voice_prompt

            audio = self.model.generate(**kwargs)

            raw_waveform = np.asarray(audio[0], dtype=np.float32).reshape(-1)
            self._maybe_create_anchor(state, text, raw_waveform, "generated", voice_mode)

        waveform = raw_waveform
        waveform = trim_low_amplitude_edges(waveform, self.sample_rate)
        waveform = apply_edge_fade(waveform, self.sample_rate)
        return np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)

    def _apply_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            logger.debug("torch seed setup failed", exc_info=True)

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

        ref_text = text
        ref_waveform = waveform.astype(np.float32, copy=False)
        if duration_s > self.config.tts_anchor_max_seconds:
            ref_text, ref_waveform, duration_s = self._crop_anchor_reference(
                text,
                ref_waveform,
                duration_s,
            )

        try:
            state.voice_prompt = self.model.create_voice_clone_prompt(
                ref_audio=(ref_waveform, self.sample_rate),
                ref_text=ref_text,
                preprocess_prompt=False,
            )
            state.anchor_text = ref_text
            state.anchor_duration_s = duration_s
            state.anchor_source = source
            logger.info(
                "omnivoice anchor created source=%s duration_s=%.2f chars=%s",
                source,
                duration_s,
                len(ref_text),
            )
        except Exception:
            logger.exception("omnivoice anchor creation failed; continuing without it")
            self._clear_cuda_cache()

    def _crop_anchor_reference(
        self,
        text: str,
        waveform: np.ndarray,
        duration_s: float,
    ) -> tuple[str, np.ndarray, float]:
        max_seconds = max(0.5, self.config.tts_anchor_max_seconds)
        max_samples = min(waveform.size, int(max_seconds * self.sample_rate))
        ref_waveform = waveform[:max_samples].astype(np.float32, copy=False)
        ref_duration_s = ref_waveform.size / self.sample_rate

        ratio = ref_duration_s / max(duration_s, 1e-6)
        max_chars = max(24, min(len(text), int(len(text) * ratio)))
        ref_text = text[:max_chars].rstrip()
        cut = max(ref_text.rfind("."), ref_text.rfind(","), ref_text.rfind(" "))
        if cut >= 24:
            ref_text = ref_text[:cut].rstrip(" ,.")
        if ref_text and ref_text[-1] not in ".!?":
            ref_text += "."

        logger.info(
            "omnivoice anchor cropped original_duration_s=%.2f ref_duration_s=%.2f original_chars=%s ref_chars=%s",
            duration_s,
            ref_duration_s,
            len(text),
            len(ref_text),
        )
        return ref_text, ref_waveform, ref_duration_s

    def _clear_cuda_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("torch cuda cache cleanup failed", exc_info=True)

    def _voice_mode(self) -> str:
        mode = self.config.tts_voice_mode.strip().lower().replace("-", "_")
        if mode in {"design", "voice_design", "instruct"}:
            return "voice_design"
        if mode in {"fixed_reference", "reference", "reference_voice", "voice_reference"}:
            return "fixed_reference"
        if mode in {"turn_anchor", "self_condition", "turn_self_condition"}:
            return "turn_anchor"
        if mode in {"session_anchor", "anchor", "session_self_condition"}:
            return "session_anchor"
        raise ValueError(
            "Unsupported LIVE_TTS_VOICE_MODE. Use voice_design, fixed_reference, turn_anchor, or session_anchor."
        )


class Qwen3TTS(BaseTTS):
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config
        self.sample_rate = 24000
        self._process: subprocess.Popen | None = None
        self._owns_process = False
        self._url = self._resolve_url()

    async def start(self) -> None:
        if self.config.qwen_tts_auto_start and not self.config.qwen_tts_url.strip():
            await asyncio.to_thread(self._ensure_worker_process)
        await self._wait_ready()
        if self.config.tts_warmup_enabled:
            logger.info("qwen3_tts worker warmup text=%r", self.config.tts_warmup_text)
            await self.synthesize(self.config.tts_warmup_text, first=True)
        logger.info("qwen3_tts worker ready sample_rate=%s", self.sample_rate)

    async def close(self) -> None:
        if self._owns_process and self._process is not None:
            process = self._process
            self._process = None
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 20)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
        await asyncio.to_thread(_release_torch_memory)

    async def synthesize(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None = None,
        language: str | None = None,
        voice_config: VoiceSessionConfig | None = None,
    ) -> np.ndarray:
        return await asyncio.to_thread(self._synthesize_sync, text, language)

    def _ensure_worker_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        worker_python = self._worker_python_path()
        worker_dir = _resolve_path(self.config.qwen_tts_worker_dir)
        worker_dir.mkdir(parents=True, exist_ok=True)
        if self.config.qwen_tts_worker_install:
            self._ensure_worker_environment(worker_python)

        env = self._worker_env()
        env.update(
            {
                "LIVE_TTS_QWEN_MODEL": self.config.qwen_tts_model,
                "LIVE_TTS_QWEN_MODE": self.config.qwen_tts_mode,
                "LIVE_TTS_QWEN_SPEAKER": self.config.qwen_tts_speaker,
                "LIVE_TTS_QWEN_INSTRUCT": self.config.qwen_tts_instruct,
                "LIVE_TTS_QWEN_DEVICE_MAP": self.config.qwen_tts_device_map,
                "LIVE_TTS_QWEN_DTYPE": self.config.qwen_tts_dtype,
                "LIVE_TTS_QWEN_ATTN_IMPLEMENTATION": (
                    self.config.qwen_tts_attn_implementation
                ),
                "LIVE_TTS_QWEN_WORKER_HOST": self.config.qwen_tts_worker_host,
                "LIVE_TTS_QWEN_WORKER_PORT": str(self.config.qwen_tts_worker_port),
            }
        )
        command = [
            str(worker_python),
            "-m",
            "live_tts.qwen_worker",
            "--host",
            self.config.qwen_tts_worker_host,
            "--port",
            str(self.config.qwen_tts_worker_port),
        ]
        logger.info("qwen3_tts starting worker command=%s", " ".join(command))
        self._process = subprocess.Popen(command, env=env, cwd=Path.cwd())
        self._owns_process = True

    def _ensure_worker_environment(self, worker_python: Path) -> None:
        requirements = Path.cwd() / "more_requirement.txt"
        if not requirements.is_file():
            raise RuntimeError(f"Missing Qwen worker requirements file: {requirements}")
        if not worker_python.exists():
            venv_dir = worker_python.parents[1]
            logger.info("qwen3_tts creating worker venv=%s", venv_dir)
            subprocess.run(
                ["uv", "venv", str(venv_dir), "--python", sys.executable],
                check=True,
            )
        marker = worker_python.parents[1] / ".live_tts_qwen_installed"
        if marker.exists():
            return
        import_check = [
            str(worker_python),
            "-c",
            (
                "from live_tts.qwen_worker import install_transformers_compat_shim; "
                "install_transformers_compat_shim(); "
                "from qwen_tts import Qwen3TTSModel; "
                "print('ok')"
            ),
        ]
        try:
            subprocess.run(import_check, check=True, env=self._worker_env())
            marker.write_text("installed\n", encoding="utf-8")
            return
        except subprocess.CalledProcessError:
            pass

        logger.info("qwen3_tts installing worker requirements=%s", requirements)
        subprocess.run(
            [str(worker_python), "-m", "pip", "install", "-r", str(requirements)],
            check=True,
        )
        subprocess.run(import_check, check=True, env=self._worker_env())
        marker.write_text("installed\n", encoding="utf-8")

    def _worker_env(self) -> dict[str, str]:
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[1])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            repo_root if not existing else repo_root + os.pathsep + existing
        )
        return env

    def _synthesize_sync(self, text: str, language: str | None) -> np.ndarray:
        import requests

        response = requests.post(
            f"{self._url}/synthesize",
            json={
                "text": text,
                "language": language or self.config.tts_language,
                "mode": self.config.qwen_tts_mode,
                "speaker": self.config.qwen_tts_speaker,
                "instruct": self.config.qwen_tts_instruct,
            },
            timeout=self.config.qwen_tts_worker_request_timeout_s,
        )
        response.raise_for_status()
        waveform, sample_rate = _decode_tts_worker_response(response)
        if sample_rate:
            self.sample_rate = int(sample_rate)
        waveform = trim_low_amplitude_edges(waveform, self.sample_rate)
        waveform = apply_edge_fade(waveform, self.sample_rate)
        return np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)

    async def _wait_ready(self) -> None:
        import requests

        deadline = (
            asyncio.get_running_loop().time()
            + self.config.qwen_tts_worker_start_timeout_s
        )
        last_error = ""
        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Qwen worker exited with code {self._process.returncode}."
                )
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    f"{self._url}/health",
                    timeout=5,
                )
                if response.ok and response.json().get("ok"):
                    self.sample_rate = int(response.json().get("sample_rate") or 24000)
                    return
                last_error = response.text[:300]
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(1.0)
        raise RuntimeError(f"Qwen worker did not become ready: {last_error}")

    def _resolve_url(self) -> str:
        configured = self.config.qwen_tts_url.strip().rstrip("/")
        if configured:
            return configured
        return (
            f"http://{self.config.qwen_tts_worker_host}:"
            f"{self.config.qwen_tts_worker_port}"
        )

    def _worker_python_path(self) -> Path:
        configured = self.config.qwen_tts_worker_python.strip()
        if configured:
            return _resolve_path(configured)
        worker_dir = _resolve_path(self.config.qwen_tts_worker_dir)
        return worker_dir / ".venv" / "bin" / "python"


class CTCTTSWorker(BaseTTS):
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config
        self.sample_rate = max(1, config.ctc_tts_sample_rate)

    async def start(self) -> None:
        if not self.config.ctc_tts_url.strip():
            raise RuntimeError(
                "CTC-TTS is configured as an external worker. Set "
                "LIVE_TTS_CTC_URL to enable it."
            )
        if self.config.tts_warmup_enabled:
            logger.info("ctc_tts worker warmup text=%r", self.config.tts_warmup_text)
            await self.synthesize(self.config.tts_warmup_text, first=True)
        logger.info("ctc_tts worker ready sample_rate=%s", self.sample_rate)

    async def synthesize(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None = None,
        language: str | None = None,
        voice_config: VoiceSessionConfig | None = None,
    ) -> np.ndarray:
        return await asyncio.to_thread(self._synthesize_sync, text, language)

    def _synthesize_sync(self, text: str, language: str | None) -> np.ndarray:
        import requests

        payload = {
            "text": text,
            "language": language or self.config.tts_language,
            "voice": self.config.ctc_tts_voice,
            "sample_rate": self.sample_rate,
        }
        response = requests.post(
            self.config.ctc_tts_url,
            json=payload,
            timeout=self.config.ctc_tts_timeout_s,
        )
        response.raise_for_status()
        waveform, sample_rate = _decode_tts_worker_response(response)
        if sample_rate:
            self.sample_rate = int(sample_rate)
        waveform = trim_low_amplitude_edges(waveform, self.sample_rate)
        waveform = apply_edge_fade(waveform, self.sample_rate)
        return np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)


class TTSEngineManager(BaseTTS):
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config
        self.sample_rate = 24000
        self._active_name = ""
        self._active: BaseTTS | None = None
        self._status = "unloaded"
        self._error = ""
        self._loading_started_at = 0.0
        self._lock = asyncio.Lock()
        self._synthesize_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.select_engine(self.config.tts_backend)

    async def close(self) -> None:
        async with self._lock:
            await self._close_active()
            self._status = "unloaded"

    def create_turn_state(self) -> TTSTurnState:
        if self._active is None:
            return TTSTurnState()
        return self._active.create_turn_state()

    async def synthesize(
        self,
        text: str,
        first: bool,
        state: TTSTurnState | None = None,
        language: str | None = None,
        voice_config: VoiceSessionConfig | None = None,
    ) -> np.ndarray:
        async with self._synthesize_lock:
            if self._active is None or self._status != "ready":
                raise RuntimeError("TTS engine is not ready.")
            waveform = await self._active.synthesize(
                text,
                first,
                state=state,
                language=language,
                voice_config=voice_config,
            )
            self.sample_rate = self._active.sample_rate
            return waveform

    async def select_engine(self, name: str) -> dict[str, object]:
        engine_name = normalize_tts_engine(name)
        async with self._lock:
            if (
                self._active is not None
                and self._active_name == engine_name
                and self._status == "ready"
            ):
                return self.status()

            async with self._synthesize_lock:
                await self._close_active()
                self._active_name = engine_name
                self._status = "loading"
                self._error = ""
                self._loading_started_at = asyncio.get_running_loop().time()
                logger.info("tts engine loading engine=%s", engine_name)
                try:
                    engine = create_tts_engine(self.config, engine_name)
                    await engine.start()
                except Exception as exc:
                    self._active = None
                    self._status = "failed"
                    self._error = str(exc)
                    logger.exception("tts engine failed engine=%s", engine_name)
                    raise
                self._active = engine
                self.sample_rate = engine.sample_rate
                self._status = "ready"
                logger.info(
                    "tts engine ready engine=%s sample_rate=%s",
                    engine_name,
                    self.sample_rate,
                )
                return self.status()

    def status(self) -> dict[str, object]:
        return {
            "engine": self._active_name or normalize_tts_engine(self.config.tts_backend),
            "status": self._status,
            "sample_rate": self.sample_rate,
            "error": self._error,
        }

    def engines(self) -> list[dict[str, str]]:
        active = self._active_name or normalize_tts_engine(self.config.tts_backend)
        result = []
        for definition in TTS_ENGINE_DEFINITIONS:
            result.append(
                {
                    "id": definition.id,
                    "label": definition.label,
                    "description": definition.description,
                    "kind": definition.kind,
                    "status": self._status if definition.id == active else "unloaded",
                }
            )
        return result

    async def _close_active(self) -> None:
        if self._active is None:
            return
        old_name = self._active_name
        old = self._active
        self._active = None
        self._status = "unloading"
        logger.info("tts engine unloading engine=%s", old_name)
        await old.close()
        await asyncio.to_thread(_release_torch_memory)
        logger.info("tts engine unloaded engine=%s", old_name)


TTS_ENGINE_DEFINITIONS = [
    TTSEngineDefinition(
        id="omnivoice",
        label="OmniVoice",
        description="Default locale con reference sintetica scelta.",
        kind="local",
    ),
    TTSEngineDefinition(
        id="qwen3_tts",
        label="Qwen3-TTS",
        description="Engine opzionale Qwen, caricato solo quando selezionato.",
        kind="local_optional",
    ),
    TTSEngineDefinition(
        id="ctc_tts",
        label="CTC-TTS",
        description="Worker esterno sperimentale per dual-streaming CTC.",
        kind="external_worker",
    ),
    TTSEngineDefinition(
        id="mock",
        label="Mock",
        description="Senoide locale per test di trasporto.",
        kind="local_test",
    ),
]


def create_tts(config: LiveTTSConfig) -> BaseTTS:
    return TTSEngineManager(config)


def create_tts_engine(config: LiveTTSConfig, engine: str) -> BaseTTS:
    backend = normalize_tts_engine(engine)
    if backend == "qwen3_tts":
        return Qwen3TTS(config)
    if backend == "ctc_tts":
        return CTCTTSWorker(config)
    if backend == "mock":
        return MockTTS()
    if backend == "omnivoice":
        return OmniVoiceTTS(config)
    raise ValueError(
        "Unsupported TTS engine. Use omnivoice, qwen3_tts, ctc_tts, or mock."
    )


def normalize_tts_engine(value: str) -> str:
    engine = str(value or "omnivoice").strip().lower().replace("-", "_")
    aliases = {
        "qwen": "qwen3_tts",
        "qwen_tts": "qwen3_tts",
        "qwen3": "qwen3_tts",
        "ctc": "ctc_tts",
        "ctc_tts_worker": "ctc_tts",
        "omni": "omnivoice",
        "omni_voice": "omnivoice",
    }
    return aliases.get(engine, engine)


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        logger.debug("unable to read reference audio duration", exc_info=True)
    return 0.0


def _decode_tts_worker_response(response: Any) -> tuple[np.ndarray, int | None]:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        data = response.json()
        sample_rate = int(data.get("sample_rate") or 0) or None
        if data.get("pcm16_base64"):
            raw = base64.b64decode(data["pcm16_base64"])
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            return pcm.astype(np.float32, copy=False), sample_rate
        encoded_audio = data.get("audio_base64") or data.get("wav_base64")
        if encoded_audio:
            return _read_audio_bytes(base64.b64decode(encoded_audio))
        if data.get("samples") is not None:
            return np.asarray(data["samples"], dtype=np.float32).reshape(-1), sample_rate
        raise RuntimeError(
            "CTC-TTS worker JSON must include audio_base64, wav_base64, "
            "pcm16_base64, or samples."
        )
    return _read_audio_bytes(response.content)


def _read_audio_bytes(payload: bytes) -> tuple[np.ndarray, int | None]:
    try:
        import soundfile as sf

        data, sample_rate = sf.read(
            io.BytesIO(payload),
            dtype="float32",
            always_2d=False,
        )
        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform.reshape(-1), int(sample_rate)
    except Exception:
        if len(payload) % 2 != 0:
            raise RuntimeError("TTS worker returned unsupported audio bytes.")
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        return pcm.astype(np.float32, copy=False), None


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        logger.debug("torch memory cleanup failed", exc_info=True)
