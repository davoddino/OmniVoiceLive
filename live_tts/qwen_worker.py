from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class SynthesisRequest(BaseModel):
    text: str
    language: str = "it"
    mode: str = "custom_voice"
    speaker: str = "Aiden"
    instruct: str = ""


class QwenWorkerState:
    def __init__(self) -> None:
        self.model = None
        self.sample_rate = 24000
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        with self.lock:
            if self.model is not None:
                return
            import torch
            from qwen_tts import Qwen3TTSModel

            model_id = os.environ["LIVE_TTS_QWEN_MODEL"]
            dtype_name = os.environ.get("LIVE_TTS_QWEN_DTYPE", "bfloat16")
            device_map = os.environ.get("LIVE_TTS_QWEN_DEVICE_MAP", "cuda:0")
            attn_impl = os.environ.get("LIVE_TTS_QWEN_ATTN_IMPLEMENTATION", "")

            kwargs: dict[str, Any] = {
                "device_map": device_map,
                "dtype": resolve_torch_dtype(torch, dtype_name),
            }
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl

            logger.info(
                "qwen worker loading model=%s device_map=%s dtype=%s",
                model_id,
                device_map,
                dtype_name,
            )
            self.model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
            logger.info("qwen worker model loaded")

    def synthesize(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        self.load()
        assert self.model is not None
        mode = request.mode.strip().lower().replace("-", "_")
        language = qwen_language(request.language)

        with self.lock:
            if mode in {"voice_design", "design"}:
                wavs, sample_rate = self.model.generate_voice_design(
                    text=request.text,
                    language=language,
                    instruct=request.instruct,
                )
            elif mode in {"custom_voice", "voice", "speaker"}:
                wavs, sample_rate = self.model.generate_custom_voice(
                    text=request.text,
                    language=language,
                    speaker=request.speaker,
                    instruct=request.instruct,
                )
            else:
                raise RuntimeError(
                    "Unsupported mode. Use custom_voice or voice_design."
                )

        self.sample_rate = int(sample_rate or self.sample_rate)
        return first_waveform(wavs), self.sample_rate


state = QwenWorkerState()
app = FastAPI(title="Live TTS Qwen Worker")


@app.on_event("startup")
async def startup() -> None:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LIVE_TTS_LOG_LEVEL", "info").upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    state.load()


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "ok": state.model is not None,
        "engine": "qwen3_tts",
        "sample_rate": state.sample_rate,
    }


@app.post("/synthesize")
async def synthesize(request: SynthesisRequest) -> dict[str, object]:
    waveform, sample_rate = state.synthesize(request)
    wav_bytes = encode_wav(waveform, sample_rate)
    return {
        "sample_rate": sample_rate,
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
    }


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    out = io.BytesIO()
    sf.write(out, samples.astype(np.float32, copy=False), sample_rate, format="WAV")
    return out.getvalue()


def first_waveform(wavs: Any) -> np.ndarray:
    first = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    if hasattr(first, "detach"):
        first = first.detach().cpu().numpy()
    waveform = np.asarray(first, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = (
            waveform.mean(axis=0)
            if waveform.shape[0] <= waveform.shape[1]
            else waveform.mean(axis=1)
        )
    return waveform.reshape(-1).astype(np.float32, copy=False)


def resolve_torch_dtype(torch_module, value: str):
    normalized = value.strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch_module.float16
    if normalized in {"bfloat16", "bf16"}:
        if torch_module.cuda.is_available():
            return torch_module.bfloat16
        logger.warning("bfloat16 requested without CUDA; falling back to float32")
        return torch_module.float32
    if normalized in {"float32", "fp32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {value}")


def qwen_language(value: str) -> str:
    normalized = str(value or "").strip().lower()
    mapping = {
        "it": "Italian",
        "italian": "Italian",
        "italiano": "Italian",
        "en": "English",
        "english": "English",
        "es": "Spanish",
        "spanish": "Spanish",
        "fr": "French",
        "french": "French",
        "de": "German",
        "german": "German",
        "zh": "Chinese",
        "cn": "Chinese",
        "ja": "Japanese",
        "jp": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "pt": "Portuguese",
    }
    return mapping.get(normalized, value or "Italian")


def main() -> None:
    import uvicorn

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("LIVE_TTS_QWEN_WORKER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LIVE_TTS_QWEN_WORKER_PORT", "8031")),
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
