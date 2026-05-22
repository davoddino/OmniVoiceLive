from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import AsyncIterator

from live_tts.config import LiveTTSConfig
from live_tts.languages import language_english_name, source_language_name
from live_tts.rag import rag_system_message
from live_tts.segmenter import clean_stream_text


class LLMStreamer:
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config

    async def stream(
        self,
        prompt: str,
        history: list[dict[str, str]],
        cancel_event: threading.Event,
        language: str | None = None,
        rag_context: str = "",
        mode: str = "agent",
        source_language: str | None = None,
        target_language: str | None = None,
        live_translation: bool = False,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if not self.config.llm_url:
            raise RuntimeError("LIVE_TTS_LLM_URL is required when LLM backend is openai")

        async for piece in self._openai_compatible_stream(
            prompt,
            history,
            cancel_event,
            language,
            rag_context,
            mode,
            source_language,
            target_language,
            live_translation,
            system_prompt,
        ):
            yield piece

    async def _openai_compatible_stream(
        self,
        prompt: str,
        history: list[dict[str, str]],
        cancel_event: threading.Event,
        language: str | None,
        rag_context: str,
        mode: str,
        source_language: str | None,
        target_language: str | None,
        live_translation: bool,
        system_prompt: str | None,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

        def worker() -> None:
            try:
                import requests

                messages = llm_messages(
                    self.config,
                    prompt,
                    history,
                    language,
                    rag_context,
                    mode=mode,
                    source_language=source_language,
                    target_language=target_language,
                    live_translation=live_translation,
                    system_prompt=system_prompt,
                )
                payload = {
                    "model": self.config.llm_model,
                    "stream": True,
                    "temperature": self.config.llm_temperature,
                    "top_p": self.config.llm_top_p,
                    "messages": messages,
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


def language_instruction(language: str | None) -> str:
    name = language_english_name(language)
    return (
        f"Session language is fixed to {name}. "
        f"Answer only in {name}. Do not auto-detect or switch language."
    )


def translation_instruction(
    source_language: str | None,
    target_language: str | None,
    *,
    live_translation: bool,
) -> str:
    source = source_language_name(source_language)
    target = language_english_name(target_language)
    live_rule = (
        "This is live translation: prefer natural target-language fragments, "
        "not isolated words. If the source fragment is incomplete, translate it "
        "naturally without adding missing facts."
        if live_translation
        else "Translate the complete turn naturally and faithfully."
    )
    return (
        "You are a professional live speech translator.\n"
        f"Translate from {source} into {target}.\n"
        f"{live_rule}\n"
        "Rules:\n"
        "- Preserve meaning faithfully; do not summarize, answer, explain, or add details.\n"
        "- Preserve numbers, names, places, times, codes, measurements, URLs, and product names.\n"
        "- If the speech is incomplete, produce a natural partial translation and do not invent.\n"
        f"- Output only in {target}. No labels, quotes, notes, markdown, or source-language text."
    )


def llm_messages(
    config: LiveTTSConfig,
    prompt: str,
    history: list[dict[str, str]],
    language: str | None,
    rag_context: str,
    *,
    mode: str = "agent",
    source_language: str | None = None,
    target_language: str | None = None,
    live_translation: bool = False,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    if mode == "translator":
        return [
            {
                "role": "system",
                "content": translation_instruction(
                    source_language,
                    target_language or language,
                    live_translation=live_translation,
                ),
            },
            {"role": "user", "content": prompt},
        ]

    return [
        {"role": "system", "content": system_prompt or config.system_prompt},
        {"role": "system", "content": language_instruction(language)},
        *([rag_system_message(rag_context)] if rag_context else []),
        *history[-10:],
        {"role": "user", "content": prompt},
    ]


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)
