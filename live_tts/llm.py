from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import AsyncIterator

from live_tts.config import LiveTTSConfig
from live_tts.segmenter import clean_stream_text


class LLMStreamer:
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config

    async def stream(
        self,
        prompt: str,
        history: list[dict[str, str]],
        cancel_event: threading.Event,
    ) -> AsyncIterator[str]:
        if self.config.llm_backend == "mock":
            async for piece in self._mock_stream(prompt, cancel_event):
                yield piece
            return

        if not self.config.llm_url:
            raise RuntimeError("LIVE_TTS_LLM_URL is required when LLM backend is openai")

        async for piece in self._openai_compatible_stream(prompt, history, cancel_event):
            yield piece

    async def _mock_stream(
        self, prompt: str, cancel_event: threading.Event
    ) -> AsyncIterator[str]:
        text = (
            "Certo, ti aiuto subito. "
            "Ho ricevuto la tua richiesta e posso guidarti passo per passo. "
            "Dimmi pure qual e' il dettaglio piu' importante da cui vuoi partire."
        )
        if "prezzo" in prompt.lower():
            text = (
                "Certo. Per darti un prezzo corretto devo capire volume, canali "
                "e integrazioni richieste. Possiamo partire dal numero di chiamate mensili."
            )
        for token in text.split(" "):
            if cancel_event.is_set():
                return
            yield token + " "
            await asyncio.sleep(0.035)

    async def _openai_compatible_stream(
        self,
        prompt: str,
        history: list[dict[str, str]],
        cancel_event: threading.Event,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

        def worker() -> None:
            try:
                import requests

                payload = {
                    "model": self.config.llm_model,
                    "stream": True,
                    "temperature": self.config.llm_temperature,
                    "top_p": self.config.llm_top_p,
                    "messages": [
                        {"role": "system", "content": self.config.system_prompt},
                        *history[-10:],
                        {"role": "user", "content": prompt},
                    ],
                }
                timeout = (5, self.config.llm_timeout_s)
                with requests.post(
                    self.config.llm_url,
                    json=payload,
                    stream=True,
                    timeout=timeout,
                ) as response:
                    response.raise_for_status()
                    response.encoding = "utf-8"
                    for raw_line in response.iter_lines(decode_unicode=False):
                        if cancel_event.is_set():
                            break
                        if not raw_line:
                            continue
                        piece = self._parse_sse_line(raw_line)
                        if piece:
                            loop.call_soon_threadsafe(queue.put_nowait, piece)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, name="llm-stream", daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise RuntimeError(f"LLM stream failed: {item}") from item
            yield item

    @staticmethod
    def _parse_sse_line(raw_line: bytes) -> str:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            line = raw_line.decode("latin-1")

        if line.startswith("data: "):
            line = line[6:]
        if line == "[DONE]":
            return ""

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return ""

        delta = data.get("choices", [{}])[0].get("delta", {})
        text = delta.get("content")
        if not text:
            return ""
        return clean_stream_text(text)


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)
