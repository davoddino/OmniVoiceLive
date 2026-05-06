from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from live_tts.config import LiveTTSConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGChunk:
    source: str
    text: str
    score: int


@dataclass(frozen=True)
class RAGResult:
    used: bool
    context: str
    chunks: list[RAGChunk]
    latency_ms: int
    timed_out: bool = False
    error: str = ""


class RAGRetriever:
    def __init__(self, config: LiveTTSConfig) -> None:
        self.config = config
        self.enabled = config.rag_enabled
        self.docs_dir = Path(config.rag_docs_dir)
        self._chunks: list[tuple[str, str]] | None = None
        self._lock = threading.Lock()

    async def retrieve(
        self,
        query: str,
        cancel_event: threading.Event | None = None,
    ) -> RAGResult:
        start = time.monotonic()
        if not self.enabled or not should_try_rag(query):
            return RAGResult(False, "", [], 0)
        try:
            chunks = await asyncio.wait_for(
                asyncio.to_thread(self._retrieve_sync, query, cancel_event),
                timeout=max(1, self.config.rag_timeout_ms) / 1000,
            )
        except asyncio.TimeoutError:
            latency_ms = elapsed_ms(start)
            logger.warning("rag timeout latency_ms=%s query=%r", latency_ms, query[:80])
            return RAGResult(False, "", [], latency_ms, timed_out=True)
        except Exception as exc:
            latency_ms = elapsed_ms(start)
            logger.exception("rag failed latency_ms=%s", latency_ms)
            return RAGResult(False, "", [], latency_ms, error=str(exc))

        context = format_context(chunks, self.config.rag_max_context_chars)
        return RAGResult(bool(context), context, chunks, elapsed_ms(start))

    def _retrieve_sync(
        self,
        query: str,
        cancel_event: threading.Event | None = None,
    ) -> list[RAGChunk]:
        if cancel_event is not None and cancel_event.is_set():
            return []
        chunks = self._load_chunks()
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        scored: list[RAGChunk] = []
        for source, text in chunks:
            if cancel_event is not None and cancel_event.is_set():
                return []
            terms = set(tokenize(text))
            score = len(query_terms & terms)
            if score > 0:
                scored.append(RAGChunk(source=source, text=text, score=score))

        scored.sort(key=lambda item: (item.score, len(item.text)), reverse=True)
        return scored[: max(1, self.config.rag_max_chunks)]

    def _load_chunks(self) -> list[tuple[str, str]]:
        if self._chunks is not None:
            return self._chunks
        with self._lock:
            if self._chunks is not None:
                return self._chunks
            self._chunks = load_document_chunks(self.docs_dir)
            logger.info("rag loaded chunks=%s dir=%s", len(self._chunks), self.docs_dir)
            return self._chunks


def should_try_rag(query: str) -> bool:
    text = query.strip().lower()
    if len(text) < 12:
        return False
    small_talk = {"ciao", "buongiorno", "buonasera", "grazie", "ok", "va bene"}
    return text not in small_talk


def load_document_chunks(docs_dir: Path) -> list[tuple[str, str]]:
    if not docs_dir.exists() or not docs_dir.is_dir():
        return []
    chunks: list[tuple[str, str]] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".json"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        for index, chunk in enumerate(split_chunks(text)):
            chunks.append((f"{path.name}#{index + 1}", chunk))
    return chunks


def split_chunks(text: str, max_chars: int = 900) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", normalized)
    chunks: list[str] = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if current and len(current) + len(part) + 1 > max_chars:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def format_context(chunks: list[RAGChunk], max_chars: int) -> str:
    lines: list[str] = []
    used = 0
    for chunk in chunks:
        line = f"[{chunk.source}] {chunk.text}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(line) > remaining:
            line = line[:remaining].rstrip()
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines).strip()


def tokenize(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term]


def elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def rag_system_message(context: str) -> dict[str, Any]:
    return {
        "role": "system",
        "content": (
            "Contesto recuperato per questa risposta. Usalo solo se rilevante. "
            "Non citare file o fonti se l'utente non lo chiede. Se il contesto "
            "non basta, chiedi chiarimento senza inventare.\n\n"
            f"{context}"
        ),
    }
