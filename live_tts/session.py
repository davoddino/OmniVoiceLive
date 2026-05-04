from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from live_tts.audio import (
    AudioTurnDetector,
    float32_to_wav_bytes,
    iter_pcm_frames,
    pcm16_bytes_from_float32,
    trim_low_amplitude_edges,
)
from live_tts.config import LiveTTSConfig
from live_tts.llm import LLMStreamer
from live_tts.segmenter import LiveTextSegmenter
from live_tts.stt import STTService
from live_tts.tts import BaseTTS, TTSTurnState


logger = logging.getLogger(__name__)


@dataclass
class TurnRuntime:
    turn_id: int
    cancel: threading.Event
    task: asyncio.Task | None = None


class RealtimeSession:
    def __init__(
        self,
        websocket: WebSocket,
        config: LiveTTSConfig,
        stt: STTService,
        llm: LLMStreamer,
        tts: BaseTTS,
    ) -> None:
        self.websocket = websocket
        self.config = config
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.session_id = str(uuid.uuid4())
        self.send_lock = asyncio.Lock()
        self.detector = AudioTurnDetector(
            sample_rate=48000,
            speech_threshold=config.vad_speech_threshold,
            start_ms=config.vad_start_ms,
            end_ms=config.vad_end_ms,
            min_turn_ms=config.vad_min_turn_ms,
            preroll_ms=config.vad_preroll_ms,
            max_turn_s=config.vad_max_turn_s,
        )
        self.history: list[dict[str, str]] = []
        self.turn_counter = 0
        self.current: TurnRuntime | None = None
        self.tts_session_state: TTSTurnState | None = (
            tts.create_turn_state() if config.tts_session_voice_anchor else None
        )
        self.closed = False

    async def run(self) -> None:
        await self.websocket.accept()
        client = self.websocket.client
        logger.info(
            "session connected session_id=%s client=%s",
            self.session_id,
            f"{client.host}:{client.port}" if client else "unknown",
        )
        await self.send_event(
            "session.ready",
            session_id=self.session_id,
            tts_sample_rate=self.tts.sample_rate,
            tts_frame_ms=self.config.tts_frame_ms,
            client_barge_threshold=self.config.client_barge_threshold,
            client_barge_stop_ms=self.config.client_barge_stop_ms,
            client_barge_commit_ms=self.config.client_barge_commit_ms,
            client_barge_cooldown_ms=self.config.client_barge_cooldown_ms,
        )

        try:
            while not self.closed:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self._handle_audio_bytes(message["bytes"])
                elif message.get("text") is not None:
                    await self._handle_text_message(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            self.closed = True
            await self.cancel_current_turn("disconnect")
            logger.info("session closed session_id=%s", self.session_id)

    async def _handle_text_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self.send_event("error", message="Invalid JSON message")
            return

        msg_type = data.get("type")
        if msg_type == "session.start":
            sample_rate = int(data.get("sample_rate") or 48000)
            self.detector.set_sample_rate(sample_rate)
            logger.info(
                "session started session_id=%s sample_rate=%s",
                self.session_id,
                sample_rate,
            )
            await self.send_event("session.started", sample_rate=sample_rate)
        elif msg_type == "barge_in":
            logger.info(
                "barge_in received session_id=%s active_turn_id=%s",
                self.session_id,
                self.active_turn_id(),
            )
            await self.cancel_current_turn("barge_in")
            self.detector.reset()
        elif msg_type == "session.stop":
            logger.info("session stop requested session_id=%s", self.session_id)
            await self.cancel_current_turn("session_stop")
            self.closed = True
            await self.websocket.close()
        else:
            await self.send_event("error", message=f"Unsupported message type: {msg_type}")

    async def _handle_audio_bytes(self, raw: bytes) -> None:
        if len(raw) < 4 or len(raw) % 4 != 0:
            return
        chunk = np.frombuffer(raw, dtype="<f4")
        for event in self.detector.accept(chunk):
            if event.type == "meter":
                await self.send_event("audio.meter", rms=event.rms)
            elif event.type == "speech_start":
                logger.info(
                    "speech_start session_id=%s active_turn_id=%s",
                    self.session_id,
                    self.active_turn_id(),
                )
                await self.send_event("vad.speech_start", turn_id=self.active_turn_id())
                if self.current and not self.current.cancel.is_set():
                    await self.cancel_current_turn("barge_in")
            elif event.type == "speech_end" and event.samples is not None:
                logger.info(
                    "speech_end session_id=%s duration_ms=%s samples=%s",
                    self.session_id,
                    event.duration_ms,
                    event.samples.size,
                )
                await self.send_event(
                    "vad.speech_end",
                    duration_ms=event.duration_ms,
                )
                await self._start_user_turn(event.samples, event.sample_rate or 48000)

    async def _start_user_turn(self, samples: np.ndarray, sample_rate: int) -> None:
        if self.current and self.current.task and not self.current.task.done():
            await self.cancel_current_turn("new_user_turn")

        self.turn_counter += 1
        cancel = threading.Event()
        runtime = TurnRuntime(turn_id=self.turn_counter, cancel=cancel)
        self.current = runtime
        logger.info(
            "turn queued session_id=%s turn_id=%s sample_rate=%s samples=%s",
            self.session_id,
            runtime.turn_id,
            sample_rate,
            samples.size,
        )
        runtime.task = asyncio.create_task(
            self._process_turn(runtime, samples, sample_rate),
            name=f"live-tts-turn-{runtime.turn_id}",
        )

    async def _process_turn(
        self, runtime: TurnRuntime, samples: np.ndarray, sample_rate: int
    ) -> None:
        turn_id = runtime.turn_id
        start = time.monotonic()
        try:
            logger.info("turn started session_id=%s turn_id=%s", self.session_id, turn_id)
            await self.send_event("turn.started", turn_id=turn_id)
            clean_samples = trim_low_amplitude_edges(
                samples,
                sample_rate,
                threshold=1e-4,
                max_trim_ms=300,
                keep_ms=40,
            )
            wav_bytes = float32_to_wav_bytes(clean_samples, sample_rate)

            stt_started = time.monotonic()
            stt_result = await self.stt.transcribe(wav_bytes, runtime.cancel)
            if runtime.cancel.is_set():
                return

            user_text = str(stt_result.get("text") or "").strip()
            stt_latency_ms = int((time.monotonic() - stt_started) * 1000)
            logger.info(
                "stt final session_id=%s turn_id=%s latency_ms=%s text=%r",
                self.session_id,
                turn_id,
                stt_latency_ms,
                _preview(user_text),
            )
            await self.send_event(
                "stt.final",
                turn_id=turn_id,
                text=user_text,
                language=stt_result.get("language"),
                latency_ms=stt_latency_ms,
            )
            if not user_text:
                logger.info("turn empty session_id=%s turn_id=%s", self.session_id, turn_id)
                await self.send_event("turn.empty", turn_id=turn_id)
                return

            self.history.append({"role": "user", "content": user_text})
            assistant_text = await self._respond(runtime, user_text)
            if assistant_text:
                self.history.append({"role": "assistant", "content": assistant_text})
                self.history = self.history[-12:]

            if not runtime.cancel.is_set():
                total_latency_ms = int((time.monotonic() - start) * 1000)
                logger.info(
                    "assistant done session_id=%s turn_id=%s latency_ms=%s",
                    self.session_id,
                    turn_id,
                    total_latency_ms,
                )
                await self.send_event(
                    "assistant.done",
                    turn_id=turn_id,
                    latency_ms=total_latency_ms,
                )
        except asyncio.CancelledError:
            runtime.cancel.set()
            raise
        except Exception as exc:
            logger.exception(
                "turn failed session_id=%s turn_id=%s error=%s",
                self.session_id,
                turn_id,
                exc,
            )
            await self.send_event("error", turn_id=turn_id, message=str(exc))
        finally:
            if self.current is runtime:
                self.current = None

    async def _respond(self, runtime: TurnRuntime, user_text: str) -> str:
        turn_id = runtime.turn_id
        segment_queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_text: list[str] = []

        async def produce_text() -> None:
            segmenter = LiveTextSegmenter()
            try:
                await self.send_event("assistant.thinking", turn_id=turn_id)
                async for piece in self.llm.stream(user_text, self.history, runtime.cancel):
                    if runtime.cancel.is_set() or self.current is not runtime:
                        break
                    full_text.append(piece)
                    await self.send_event(
                        "assistant.text_delta",
                        turn_id=turn_id,
                        text=piece,
                    )
                    for segment in segmenter.push(piece):
                        logger.info(
                            "llm segment session_id=%s turn_id=%s chars=%s text=%r",
                            self.session_id,
                            turn_id,
                            len(segment),
                            _preview(segment),
                        )
                        await segment_queue.put(segment)
                for segment in segmenter.flush():
                    logger.info(
                        "llm segment final session_id=%s turn_id=%s chars=%s text=%r",
                        self.session_id,
                        turn_id,
                        len(segment),
                        _preview(segment),
                    )
                    await segment_queue.put(segment)
            finally:
                await segment_queue.put(None)

        async def consume_tts() -> None:
            first = True
            segment_index = 0
            tts_state = self.tts_session_state or self.tts.create_turn_state()
            while True:
                segment = await segment_queue.get()
                if segment is None:
                    break
                if runtime.cancel.is_set() or self.current is not runtime:
                    break

                logger.info(
                    "tts start session_id=%s turn_id=%s segment_index=%s first=%s chars=%s",
                    self.session_id,
                    turn_id,
                    segment_index,
                    first,
                    len(segment),
                )
                await self.send_event(
                    "assistant.audio_start",
                    turn_id=turn_id,
                    segment_index=segment_index,
                    text=segment,
                    sample_rate=self.tts.sample_rate,
                )
                synth_started = time.monotonic()
                waveform = await self.tts.synthesize(
                    segment,
                    first=first,
                    state=tts_state,
                )
                if runtime.cancel.is_set() or self.current is not runtime:
                    break

                pcm = pcm16_bytes_from_float32(waveform)
                tts_latency_ms = int((time.monotonic() - synth_started) * 1000)
                duration_ms = int(len(waveform) / self.tts.sample_rate * 1000)
                logger.info(
                    "tts ready session_id=%s turn_id=%s segment_index=%s latency_ms=%s duration_ms=%s bytes=%s",
                    self.session_id,
                    turn_id,
                    segment_index,
                    tts_latency_ms,
                    duration_ms,
                    len(pcm),
                )
                await self.send_event(
                    "assistant.audio_ready",
                    turn_id=turn_id,
                    segment_index=segment_index,
                    duration_ms=duration_ms,
                    latency_ms=tts_latency_ms,
                )
                for frame in iter_pcm_frames(
                    pcm, self.tts.sample_rate, self.config.tts_frame_ms
                ):
                    if runtime.cancel.is_set() or self.current is not runtime:
                        break
                    await self.send_bytes(frame)
                    await asyncio.sleep(0)

                await self.send_event(
                    "assistant.audio_end",
                    turn_id=turn_id,
                    segment_index=segment_index,
                )
                first = False
                segment_index += 1

        producer = asyncio.create_task(produce_text(), name=f"llm-{turn_id}")
        consumer = asyncio.create_task(consume_tts(), name=f"tts-{turn_id}")
        try:
            await asyncio.gather(producer, consumer)
        finally:
            producer.cancel()
            consumer.cancel()

        return "".join(full_text).strip()

    async def cancel_current_turn(self, reason: str) -> None:
        runtime = self.current
        if runtime is None:
            return
        runtime.cancel.set()
        logger.info(
            "turn cancelled session_id=%s turn_id=%s reason=%s",
            self.session_id,
            runtime.turn_id,
            reason,
        )
        if runtime.task and not runtime.task.done():
            runtime.task.cancel()
        await self.send_event(
            "turn.cancelled",
            turn_id=runtime.turn_id,
            reason=reason,
        )

    def active_turn_id(self) -> int | None:
        return self.current.turn_id if self.current else None

    async def send_event(self, event_type: str, **payload: Any) -> None:
        if self.closed:
            return
        data = {"type": event_type, **payload}
        async with self.send_lock:
            try:
                await self.websocket.send_text(json.dumps(data, ensure_ascii=False))
            except RuntimeError:
                self.closed = True

    async def send_bytes(self, payload: bytes) -> None:
        if self.closed:
            return
        async with self.send_lock:
            try:
                await self.websocket.send_bytes(payload)
            except RuntimeError:
                self.closed = True


def _preview(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."
