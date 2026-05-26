from __future__ import annotations

import asyncio
import tempfile
import threading
from urllib.parse import urlsplit, urlunsplit

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

        backend = self._backend()
        if self._url() and backend in {"auto", "http"}:
            return await asyncio.to_thread(self._transcribe_http, wav_bytes, language)
        return await asyncio.to_thread(self._transcribe_local, wav_bytes, language)

    async def status(self) -> dict[str, object]:
        backend = self._backend()
        url = self._url()
        if url and backend in {"auto", "http"}:
            return await asyncio.to_thread(self._http_status, url, self.health_url(url))
        return {
            "ok": True,
            "mode": "local",
            "backend": backend,
            "url": "",
            "health_url": "",
        }

    @staticmethod
    def health_url(stt_url: str) -> str:
        url = stt_url.strip()
        if not url:
            return ""
        parts = urlsplit(url)
        path = parts.path.rstrip("/")
        if path.endswith("/health") or path.endswith("/healthz"):
            return url
        if path.endswith("/transcribe"):
            path = path[: -len("/transcribe")]
        path = f"{path}/healthz" if path else "/healthz"
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    def _transcribe_http(
        self, wav_bytes: bytes, language: str | None
    ) -> dict[str, object]:
        import requests

        files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
        data = {"language": self._requested_language(language)}
        response = requests.post(self._url(), files=files, data=data, timeout=60)
        response.raise_for_status()
        return response.json()

    def _http_status(self, url: str, health_url: str) -> dict[str, object]:
        import requests

        status: dict[str, object] = {
            "ok": False,
            "mode": "http",
            "backend": self._backend(),
            "url": url,
            "health_url": health_url,
        }
        try:
            response = requests.get(health_url, timeout=2.5)
        except requests.RequestException as exc:
            status["error"] = str(exc)
            return status

        status["status_code"] = response.status_code
        status["ok"] = response.ok
        if not response.ok:
            status["detail"] = response.text[:300]
        return status

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
        fallback_language = getattr(self.config, "stt_language", "it") or "it"
        return str(fallback_language).strip().lower() or "it"

    def _backend(self) -> str:
        backend = getattr(self.config, "stt_backend", "auto") or "auto"
        return str(backend).strip().lower()

    def _url(self) -> str:
        return str(getattr(self.config, "stt_url", "") or "").strip()
