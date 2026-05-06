from __future__ import annotations

import asyncio


def drain_segment_queue(queue: asyncio.Queue[str | None]) -> int:
    count = 0
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return count
        if item is not None:
            count += 1
