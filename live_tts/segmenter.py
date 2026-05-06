from __future__ import annotations

import re
import unicodedata


PROTECTED_PATTERNS = [
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE),
    re.compile(r"(?<!\w)\d+(?:[.,:/-]\d+)+(?!\w)"),
    re.compile(r"(?<!\w)[A-Z]{1,6}[- ]?\d{2,}[A-Z0-9-]*(?!\w)"),
    re.compile(r"(?<!\w)\d{2,}[A-Z][A-Z0-9-]*(?!\w)"),
]


def remove_think_only(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE)


def fix_encoding(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def clean_stream_text(text: str) -> str:
    return remove_think_only(fix_encoding(text))


def normalize_tts_text(text: str, language: str | None = None) -> str:
    text = clean_stream_text(text)
    if not text:
        return ""

    text = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", text)
    text = re.sub(r"`{1,3}([^`]+)`{1,3}", r"\1", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", ". ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", ". ", text)
    text = re.sub(r"[*_~#>`]+", " ", text)
    text = re.sub(r"\b(?:https?://\S+|www\.\S+)", " link web ", text)
    text = re.sub(
        r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
        _speak_email,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<=\d)\s*(?:\u20ac|eur|euro)\b", " euro", text, flags=re.I)
    text = re.sub(r"\b(\d+(?:[.,]\d+)?)\s*(?:\u20ac|eur|euro)\b", r"\1 euro", text, flags=re.I)
    text = re.sub(r"\b(?:tel\.?|telefono|phone)\s*[:.-]?\s*", "telefono ", text, flags=re.I)
    text = re.sub(r"(?<![\w/])\+?(?:\d[\s.-]?){7,}\d(?![\w/])", _speak_digits, text)
    text = text.replace("&", " e ")
    text = text.replace("%", " per cento")
    text = re.sub(r"(?<=\w)/(?=\w)", " o ", text)
    text = re.sub(r"(?<!\.)\.{2,}", ".", text)
    text = re.sub(r"([!?;:,]){2,}", r"\1", text)
    text = "".join(ch for ch in text if _is_tts_char(ch))
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _speak_email(match: re.Match[str]) -> str:
    value = match.group(0)
    value = value.replace("@", " chiocciola ")
    value = value.replace(".", " punto ")
    value = value.replace("_", " underscore ")
    value = value.replace("-", " trattino ")
    return value


def _speak_digits(match: re.Match[str]) -> str:
    value = match.group(0)
    prefix = "piu " if value.strip().startswith("+") else ""
    digits = re.sub(r"\D", "", value)
    return prefix + " ".join(digits)


def _is_tts_char(ch: str) -> bool:
    if ch in "\n\r\t":
        return True
    category = unicodedata.category(ch)
    if category in {"So", "Cs", "Co"}:
        return False
    if category.startswith("C"):
        return False
    return True


class SentenceAccumulator:
    def __init__(
        self,
        min_first_chars: int = 90,
        max_first_chars: int = 280,
        min_next_chars: int = 120,
        max_next_chars: int = 380,
        normalize_segments: bool = True,
        language: str | None = None,
    ) -> None:
        self.min_first_chars = min_first_chars
        self.max_first_chars = max_first_chars
        self.min_next_chars = min_next_chars
        self.max_next_chars = max_next_chars
        self.normalize_segments = normalize_segments
        self.language = language
        self.buffer = ""
        self.first = True
        self.strong_delimiters = ".?!;:"
        self.soft_delimiters = ","

    def push(self, piece: str) -> list[str]:
        piece = clean_stream_text(piece)
        if not piece:
            return []
        self.buffer += piece
        return self._drain(force=False)

    def flush(self) -> list[str]:
        return self._drain(force=True)

    def _drain(self, force: bool) -> list[str]:
        out: list[str] = []
        while True:
            segment = self._next_segment(force=force)
            if segment is None:
                break
            out.append(segment)
            self.first = False
            force = False
        return out

    def _next_segment(self, force: bool) -> str | None:
        buffer = clean_stream_text(self.buffer).strip()
        self.buffer = buffer
        if not buffer:
            return None

        min_chars = self.min_first_chars if self.first else self.min_next_chars
        max_chars = self.max_first_chars if self.first else self.max_next_chars
        lookahead_chars = min(len(buffer), int(max_chars * 1.25))

        strong_cut = self._find_delimiter_cut(
            buffer,
            self.strong_delimiters,
            min_chars,
            lookahead_chars,
        )
        if strong_cut is not None:
            return self._cut(strong_cut)

        if len(buffer) >= max_chars:
            soft_cut = self._find_delimiter_cut(
                buffer,
                self.soft_delimiters,
                min_chars,
                max_chars,
            )
            if soft_cut is not None:
                return self._cut(soft_cut)

            space_cut = self._find_space_cut(buffer, min_chars, max_chars)
            if space_cut is not None:
                return self._cut(space_cut)

            fallback_cut = self._find_space_cut(buffer, min_chars, lookahead_chars)
            return self._cut(fallback_cut or max_chars)

        if force:
            return self._cut(len(buffer))

        return None

    def _cut(self, index: int) -> str:
        segment = clean_stream_text(self.buffer[:index]).strip()
        self.buffer = self.buffer[index:].strip()
        if self.normalize_segments:
            segment = normalize_tts_text(segment, self.language)
        return segment

    def _find_delimiter_cut(
        self,
        buffer: str,
        delimiters: str,
        min_chars: int,
        limit: int,
    ) -> int | None:
        spans = _protected_spans(buffer)
        cuts: list[int] = []
        for index, ch in enumerate(buffer[:limit]):
            if ch not in delimiters:
                continue
            cut = index + 1
            if cut >= min_chars and _valid_cut(cut, spans):
                cuts.append(cut)
        return cuts[-1] if cuts else None

    def _find_space_cut(
        self,
        buffer: str,
        min_chars: int,
        limit: int,
    ) -> int | None:
        spans = _protected_spans(buffer)
        for index in range(min(len(buffer), limit) - 1, min_chars - 1, -1):
            if buffer[index] == " " and _valid_cut(index + 1, spans):
                return index + 1
        return None


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in PROTECTED_PATTERNS:
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    return spans


def _valid_cut(index: int, spans: list[tuple[int, int]]) -> bool:
    return not any(start < index < end for start, end in spans)


class LiveTextSegmenter(SentenceAccumulator):
    pass
