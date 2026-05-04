from __future__ import annotations

import re


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


class LiveTextSegmenter:
    def __init__(
        self,
        min_first_chars: int = 28,
        max_first_chars: int = 70,
        min_next_chars: int = 90,
        max_next_chars: int = 220,
    ) -> None:
        self.min_first_chars = min_first_chars
        self.max_first_chars = max_first_chars
        self.min_next_chars = min_next_chars
        self.max_next_chars = max_next_chars
        self.buffer = ""
        self.first = True
        self.strong_delimiters = ".?!..."
        self.soft_delimiters = ",;:"

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

        strong_cut = max(
            (buffer.rfind(d, 0, lookahead_chars + 1) for d in self.strong_delimiters),
            default=-1,
        )
        if strong_cut + 1 >= min_chars:
            return self._cut(strong_cut + 1)

        if len(buffer) >= max_chars:
            soft_cut = max(
                (buffer.rfind(d, 0, max_chars) for d in self.soft_delimiters),
                default=-1,
            )
            if soft_cut + 1 >= min_chars:
                return self._cut(soft_cut + 1)

            space_cut = buffer.rfind(" ", 0, max_chars)
            if space_cut + 1 >= min_chars:
                return self._cut(space_cut + 1)

            return self._cut(max_chars)

        if force:
            return self._cut(len(buffer))

        return None

    def _cut(self, index: int) -> str:
        segment = clean_stream_text(self.buffer[:index]).strip()
        self.buffer = self.buffer[index:].strip()
        return segment
