from __future__ import annotations

import asyncio
import tempfile
import threading

from live_tts.config import LiveTTSConfig


class STTService:
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config
        self._model = None
        self._lock = threading.Lock()

    async def transcribe(
        self, wav_bytes: bytes, cancel_event: threading.Event
    ) -> dict[str, object]:
        if cancel_event.is_set():
            return {"text": "", "language": None, "language_probability": 0.0}

        backend = self.config.stt_backend
        if backend == "mock":
            return {"text": "Vorrei informazioni sui vostri servizi.", "language": "it"}
        if self.config.stt_url and backend in {"auto", "http"}:
            return await asyncio.to_thread(self._transcribe_http, wav_bytes)
        return await asyncio.to_thread(self._transcribe_local, wav_bytes)

    def _transcribe_http(self, wav_bytes: bytes) -> dict[str, object]:
        import requests

        files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
        response = requests.post(self.config.stt_url, files=files, timeout=60)
        response.raise_for_status()
        return response.json()

    def _transcribe_local(self, wav_bytes: bytes) -> dict[str, object]:
        model = self._load_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            segments, info = model.transcribe(
                tmp.name,
                language=self.config.stt_language,
                vad_filter=True,
                beam_size=1,
                temperature=0.0,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
        }

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is required for local STT. "
                    "Set LIVE_TTS_STT_URL to use whisper.py over HTTP, or "
                    "LIVE_TTS_STT_BACKEND=mock for transport tests."
                ) from exc
            self._model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )
            return self._model
