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
        self,
        wav_bytes: bytes,
        cancel_event: threading.Event,
        language: str | None = None,
    ) -> dict[str, object]:
        if cancel_event.is_set():
            return {"text": "", "language": None, "language_probability": 0.0}

        backend = self.config.stt_backend
        if self.config.stt_url and backend in {"auto", "http"}:
            return await asyncio.to_thread(self._transcribe_http, wav_bytes, language)
        return await asyncio.to_thread(self._transcribe_local, wav_bytes, language)

    def _transcribe_http(
        self, wav_bytes: bytes, language: str | None
    ) -> dict[str, object]:
        import requests

        files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
        data = {"language": self._requested_language(language)}
        response = requests.post(self.config.stt_url, files=files, data=data, timeout=60)
        response.raise_for_status()
        return response.json()

    def _transcribe_local(
        self, wav_bytes: bytes, language: str | None
    ) -> dict[str, object]:
        model = self._load_model()
        requested_language = self._requested_language(language)
        language_arg = None if requested_language == "auto" else requested_language
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            segments, info = model.transcribe(
                tmp.name,
                language=language_arg,
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
                    "Set LIVE_TTS_STT_URL to use whisper.py over HTTP."
                ) from exc
            self._model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )
            return self._model

    def _requested_language(self, language: str | None) -> str:
        requested = str(language or "").strip().lower()
        if requested:
            return requested
        return str(self.config.stt_language or "it").strip().lower() or "it"
