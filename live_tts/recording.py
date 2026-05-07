from __future__ import annotations

import asyncio
import json
import logging
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from live_tts.audio import ensure_mono_float32, pcm16_bytes_from_float32
from live_tts.config import LiveTTSConfig
from live_tts.voice import VoiceSessionConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingPaths:
    root: Path
    input_wav: Path
    output_wav: Path
    transcript_jsonl: Path
    metadata_json: Path
    latency_metrics_json: Path
    crm_events_jsonl: Path
    crm_payload_json: Path


class AsyncSessionRecorder:
    def __init__(self, config: LiveTTSConfig, session_id: str) -> None:
        self.config = config
        self.session_id = session_id
        self.enabled = config.recording_enabled
        self.paths: RecordingPaths | None = None
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=max(8, config.recording_queue_size)
        )
        self._task: asyncio.Task | None = None
        self._started_at: str | None = None
        self._started_monotonic = time.monotonic()
        self._input_sample_rate = 48000
        self._output_sample_rate = 24000
        self._metadata: dict[str, Any] = {}
        self._prebuffer: deque[np.ndarray] = deque()
        self._prebuffer_samples = 0
        self._dropped_items = 0
        self.metrics: dict[str, Any] = {
            "session_id": session_id,
            "turns": 0,
            "stt_ms": [],
            "first_llm_delta_ms": [],
            "first_segment_ms": [],
            "llm_ms": [],
            "tts_ms": [],
            "turn_ttfa_ms": [],
            "time_to_first_audio_ms": None,
            "barge_in_count": 0,
            "tts_chunks": 0,
            "tts_chunks_cancelled": 0,
            "rag_used": False,
            "rag_ms": [],
            "provider_errors": [],
            "fallbacks": [],
            "escalation_requested": False,
            "outcome_estimate": "unknown",
        }

    async def start(
        self,
        input_sample_rate: int,
        output_sample_rate: int,
        voice_config: VoiceSessionConfig,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._task is not None:
            return

        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._started_at = utc_now()
        self._started_monotonic = time.monotonic()
        day = self._started_at[:10]
        root = Path(self.config.recording_dir) / day / f"call_{self.session_id}"
        root.mkdir(parents=True, exist_ok=True)
        self.paths = RecordingPaths(
            root=root,
            input_wav=root / "input.wav",
            output_wav=root / "output_tts.wav",
            transcript_jsonl=root / "transcript.jsonl",
            metadata_json=root / "metadata.json",
            latency_metrics_json=root / "latency_metrics.json",
            crm_events_jsonl=root / "crm_events.jsonl",
            crm_payload_json=root / "crm_payload.json",
        )
        self._metadata = {
            "session_id": self.session_id,
            "started_at": self._started_at,
            "status": "active",
            "input_sample_rate": input_sample_rate,
            "output_sample_rate": output_sample_rate,
            "voice_session_config": voice_config.as_dict(),
            "metadata": metadata or {},
        }
        self._write_json(self.paths.metadata_json, self._metadata)
        self._task = asyncio.create_task(self._writer_loop(), name=f"rec-{self.session_id}")
        while self._prebuffer:
            self.record_input(self._prebuffer.popleft())
        self._prebuffer_samples = 0
        self.event("session_started", input_sample_rate=input_sample_rate)

    def record_input(self, samples: np.ndarray) -> None:
        if not self.enabled:
            return
        mono = ensure_mono_float32(samples).copy()
        if self._task is None:
            self._add_prebuffer(mono)
            return
        self._enqueue({"kind": "input", "samples": mono})

    def record_output(self, samples: np.ndarray) -> None:
        if not self.enabled or self._task is None:
            return
        self._enqueue({"kind": "output", "samples": ensure_mono_float32(samples).copy()})

    def transcript(self, speaker: str, text: str, **fields: Any) -> None:
        if not self.enabled or self._task is None:
            return
        entry = {"ts": utc_now(), "speaker": speaker, "text": text, **fields}
        self._enqueue({"kind": "transcript", "entry": entry})

    def event(self, event: str, **fields: Any) -> None:
        if not self.enabled or self._task is None:
            return
        entry = {"ts": utc_now(), "event": event, **fields}
        self._enqueue({"kind": "crm_event", "entry": entry})

    def note_turn(self) -> None:
        self.metrics["turns"] += 1

    def note_latency(self, key: str, latency_ms: int) -> None:
        values = self.metrics.setdefault(key, [])
        if isinstance(values, list):
            values.append(int(latency_ms))

    def note_first_audio(self) -> None:
        if self.metrics["time_to_first_audio_ms"] is None:
            self.metrics["time_to_first_audio_ms"] = int(
                (time.monotonic() - self._started_monotonic) * 1000
            )

    def note_tts_chunk(self) -> None:
        self.metrics["tts_chunks"] += 1

    def note_tts_cancelled(self, count: int = 1) -> None:
        self.metrics["tts_chunks_cancelled"] += max(0, int(count))

    def note_barge_in(self) -> None:
        self.metrics["barge_in_count"] += 1

    def note_rag(self, used: bool, latency_ms: int, fallback: bool = False) -> None:
        self.metrics["rag_used"] = bool(self.metrics["rag_used"] or used)
        self.note_latency("rag_ms", latency_ms)
        if fallback:
            self.metrics["fallbacks"].append("rag_timeout")

    def note_error(self, error: str) -> None:
        self.metrics["provider_errors"].append(error)

    def request_escalation(self, reason: str) -> None:
        self.metrics["escalation_requested"] = True
        self.metrics["outcome_estimate"] = "escalation"
        self.event("escalation_requested", reason=reason)

    async def close(self) -> None:
        if not self.enabled:
            return
        if self._task is not None:
            await self._queue.put(None)
            await self._task
            self._task = None
        if self.paths is None:
            return
        ended_at = utc_now()
        duration_s = max(0.0, time.monotonic() - self._started_monotonic)
        if (
            self.metrics["outcome_estimate"] == "unknown"
            and self.metrics["turns"] > 0
            and not self.metrics["provider_errors"]
        ):
            self.metrics["outcome_estimate"] = "resolved"
        metadata = {
            **self._metadata,
            "session_id": self.session_id,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "duration_s": round(duration_s, 3),
            "status": "closed",
            "dropped_recording_items": self._dropped_items,
        }
        self._write_json(self.paths.metadata_json, metadata)
        self._write_json(self.paths.latency_metrics_json, self._metrics_payload(duration_s))
        self._write_json(
            self.paths.crm_payload_json,
            self._crm_payload(ended_at),
        )

    async def _writer_loop(self) -> None:
        assert self.paths is not None
        with wave.open(str(self.paths.input_wav), "wb") as input_wav, wave.open(
            str(self.paths.output_wav), "wb"
        ) as output_wav, self.paths.transcript_jsonl.open(
            "a", encoding="utf-8"
        ) as transcript_file, self.paths.crm_events_jsonl.open(
            "a", encoding="utf-8"
        ) as event_file:
            input_wav.setnchannels(1)
            input_wav.setsampwidth(2)
            input_wav.setframerate(self._input_sample_rate)
            output_wav.setnchannels(1)
            output_wav.setsampwidth(2)
            output_wav.setframerate(self._output_sample_rate)

            while True:
                item = await self._queue.get()
                if item is None:
                    break
                kind = item.get("kind")
                if kind == "input":
                    input_wav.writeframes(pcm16_bytes_from_float32(item["samples"]))
                elif kind == "output":
                    output_wav.writeframes(pcm16_bytes_from_float32(item["samples"]))
                elif kind == "transcript":
                    transcript_file.write(json.dumps(item["entry"], ensure_ascii=False) + "\n")
                    transcript_file.flush()
                elif kind == "crm_event":
                    event_file.write(json.dumps(item["entry"], ensure_ascii=False) + "\n")
                    event_file.flush()

    def _enqueue(self, item: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._dropped_items += 1
            logger.debug("recording queue full session_id=%s", self.session_id)

    def _add_prebuffer(self, mono: np.ndarray) -> None:
        max_samples = int(
            max(0.0, self.config.recording_prebuffer_seconds) * self._input_sample_rate
        )
        if max_samples <= 0:
            return
        self._prebuffer.append(mono)
        self._prebuffer_samples += mono.size
        while self._prebuffer and self._prebuffer_samples > max_samples:
            old = self._prebuffer.popleft()
            self._prebuffer_samples -= old.size

    def _metrics_payload(self, duration_s: float) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "duration_s": round(duration_s, 3),
            "turns": self.metrics["turns"],
            "avg_stt_ms": average(self.metrics["stt_ms"]),
            "avg_first_llm_delta_ms": average(self.metrics["first_llm_delta_ms"]),
            "avg_first_segment_ms": average(self.metrics["first_segment_ms"]),
            "avg_llm_ms": average(self.metrics["llm_ms"]),
            "avg_tts_ms": average(self.metrics["tts_ms"]),
            "avg_turn_ttfa_ms": average(self.metrics["turn_ttfa_ms"]),
            "time_to_first_audio_ms": self.metrics["time_to_first_audio_ms"],
            "barge_in_count": self.metrics["barge_in_count"],
            "tts_chunks": self.metrics["tts_chunks"],
            "tts_chunks_cancelled": self.metrics["tts_chunks_cancelled"],
            "rag_used": self.metrics["rag_used"],
            "avg_rag_ms": average(self.metrics["rag_ms"]),
            "provider_errors": self.metrics["provider_errors"],
            "fallbacks": self.metrics["fallbacks"],
            "outcome_estimate": self.metrics["outcome_estimate"],
            "escalation_requested": self.metrics["escalation_requested"],
            "dropped_recording_items": self._dropped_items,
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _crm_payload(self, created_at: str) -> dict[str, Any]:
        assert self.paths is not None
        turns = []
        if self.paths.transcript_jsonl.exists():
            for line in self.paths.transcript_jsonl.read_text(encoding="utf-8").splitlines():
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        user_turns = [item for item in turns if item.get("speaker") == "user"]
        assistant_turns = [item for item in turns if item.get("speaker") == "assistant"]
        summary = {
            "turns": len(user_turns),
            "last_user_text": user_turns[-1]["text"] if user_turns else "",
            "last_assistant_text": assistant_turns[-1]["text"] if assistant_turns else "",
            "outcome_estimate": self.metrics["outcome_estimate"],
            "escalation_requested": self.metrics["escalation_requested"],
        }
        return {
            "session_id": self.session_id,
            "status": "pending",
            "created_at": created_at,
            "crm_sync": "manual_or_background",
            "events_source": str(self.paths.crm_events_jsonl),
            "transcript_source": str(self.paths.transcript_jsonl),
            "summary": summary,
        }


def average(values: list[int]) -> int | None:
    if not values:
        return None
    return int(sum(values) / len(values))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
