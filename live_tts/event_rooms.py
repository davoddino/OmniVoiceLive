from __future__ import annotations

import asyncio
import json
import logging
import secrets
import string
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from live_tts.audio import (
    AudioCrossfader,
    AudioLoudnessSmoother,
    AudioTurnDetector,
    float32_to_wav_bytes,
    iter_pcm_frames,
    pad_audio_edges,
    pcm16_bytes_from_float32,
)
from live_tts.config import LiveTTSConfig
from live_tts.languages import (
    SOURCE_LANGUAGE_OPTIONS,
    SUPPORTED_LANGUAGES,
    language_label,
    normalize_language_code,
)
from live_tts.llm import LLMStreamer
from live_tts.segmenter import SentenceAccumulator
from live_tts.stt import STTService
from live_tts.tts import BaseTTS
from live_tts.voice import VoiceSessionConfig


logger = logging.getLogger(__name__)

EVENT_CODE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


@dataclass
class QueuedEventTurn:
    samples: np.ndarray
    sample_rate: int


class EventWebSocketPeer:
    def __init__(
        self,
        websocket: WebSocket,
        *,
        role: str,
        target_language: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.client_id = str(uuid.uuid4())
        self.role = role
        self.target_language = target_language
        self.send_lock = asyncio.Lock()
        self.closed = False

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

    async def close(self, code: int = 1000) -> None:
        self.closed = True
        try:
            await self.websocket.close(code=code)
        except RuntimeError:
            pass


class EventRoomManager:
    def __init__(
        self,
        config: LiveTTSConfig,
        stt: STTService,
        llm: LLMStreamer,
        tts: BaseTTS,
    ) -> None:
        self.config = config
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self._rooms: dict[str, EventRoom] = {}
        self._lock = asyncio.Lock()

    async def handle_speaker(self, websocket: WebSocket) -> None:
        await websocket.accept()
        peer = EventWebSocketPeer(websocket, role="speaker")
        room: EventRoom | None = None
        await peer.send_event(
            "event.speaker.ready",
            languages=SUPPORTED_LANGUAGES,
            source_languages=SOURCE_LANGUAGE_OPTIONS,
            tts_engine=self.tts_status(),
            tts_engines=self.tts_engines(),
        )

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if room is not None:
                        await room.handle_audio_bytes(message["bytes"])
                    continue
                if message.get("text") is None:
                    continue
                data = self._parse_json(message["text"])
                if data is None:
                    await peer.send_event("error", message="Invalid JSON message")
                    continue
                msg_type = str(data.get("type") or "")
                if msg_type == "event.host.start":
                    if room is not None:
                        await peer.send_event(
                            "error",
                            message="This speaker is already hosting an event.",
                        )
                        continue
                    engine = str(data.get("tts_engine") or "").strip()
                    if engine and not await self._select_tts_engine(peer, engine):
                        continue
                    source_language = normalize_language_code(
                        data.get("source_language"),
                        default="auto",
                        allow_auto=True,
                    )
                    sample_rate = int(data.get("sample_rate") or 48000)
                    room = await self.create_room(
                        peer,
                        source_language=source_language,
                        speaker_sample_rate=sample_rate,
                    )
                    await peer.send_event(
                        "event.host.started",
                        event_code=room.code,
                        source_language=room.source_language,
                        source_language_label=(
                            SOURCE_LANGUAGE_OPTIONS.get(
                                room.source_language,
                                room.source_language,
                            )
                        ),
                        tts_sample_rate=self.tts.sample_rate,
                        tts_engine=self.tts_status(),
                        listener_count=0,
                        limits=room.limits(),
                    )
                elif msg_type in {"event.host.stop", "session.stop"}:
                    break
                else:
                    await peer.send_event(
                        "error",
                        message=f"Unsupported event speaker message: {msg_type}",
                    )
        except WebSocketDisconnect:
            pass
        finally:
            if room is not None:
                await self.close_room(room.code, reason="host_left")
            else:
                await peer.close()

    async def handle_listener(self, websocket: WebSocket) -> None:
        await websocket.accept()
        peer = EventWebSocketPeer(websocket, role="listener")
        room: EventRoom | None = None
        await peer.send_event(
            "event.listener.ready",
            languages=SUPPORTED_LANGUAGES,
            tts_sample_rate=self.tts.sample_rate,
        )

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("text") is None:
                    continue
                data = self._parse_json(message["text"])
                if data is None:
                    await peer.send_event("error", message="Invalid JSON message")
                    continue
                msg_type = str(data.get("type") or "")
                if msg_type == "event.listener.join":
                    event_code = normalize_event_code(data.get("event_code"))
                    target_language = normalize_language_code(
                        data.get("target_language"),
                        default=self.config.translator_target_language
                        or self.config.tts_language
                        or "en",
                    )
                    room = await self.get_room(event_code)
                    if room is None:
                        await peer.send_event(
                            "error",
                            message="Event not found or already closed.",
                        )
                        break
                    peer.target_language = target_language
                    await room.add_listener(peer)
                    await peer.send_event(
                        "event.listener.joined",
                        event_code=room.code,
                        target_language=target_language,
                        target_language_label=language_label(target_language),
                        tts_sample_rate=self.tts.sample_rate,
                        listener_count=await room.listener_count(),
                        limits=room.limits(),
                    )
                elif msg_type == "event.listener.update_language":
                    if room is None:
                        await peer.send_event("error", message="Join an event first.")
                        continue
                    target_language = normalize_language_code(
                        data.get("target_language"),
                        default=peer.target_language or self.config.tts_language or "en",
                    )
                    await room.update_listener_language(peer.client_id, target_language)
                    await peer.send_event(
                        "event.listener.updated",
                        target_language=target_language,
                        target_language_label=language_label(target_language),
                    )
                elif msg_type in {"event.listener.leave", "session.stop"}:
                    break
                else:
                    await peer.send_event(
                        "error",
                        message=f"Unsupported event listener message: {msg_type}",
                    )
        except WebSocketDisconnect:
            pass
        finally:
            if room is not None:
                await room.remove_listener(peer.client_id)
                await peer.close()
            else:
                await peer.close()

    async def create_room(
        self,
        speaker: Any,
        *,
        source_language: str,
        speaker_sample_rate: int,
        code: str | None = None,
    ) -> "EventRoom":
        async with self._lock:
            event_code = normalize_event_code(code) if code else ""
            if not event_code:
                event_code = self._new_code()
            while event_code in self._rooms:
                event_code = self._new_code()
            room = EventRoom(
                event_code,
                speaker,
                self.config,
                self.stt,
                self.llm,
                self.tts,
                source_language=source_language,
                speaker_sample_rate=speaker_sample_rate,
            )
            self._rooms[event_code] = room
            return room

    async def get_room(self, code: str) -> "EventRoom | None":
        event_code = normalize_event_code(code)
        async with self._lock:
            return self._rooms.get(event_code)

    async def close_room(self, code: str, *, reason: str) -> None:
        event_code = normalize_event_code(code)
        async with self._lock:
            room = self._rooms.pop(event_code, None)
        if room is not None:
            await room.close(reason=reason)

    def tts_status(self) -> dict[str, object]:
        status = getattr(self.tts, "status", None)
        if status is not None:
            return status()
        return {
            "engine": self.config.tts_backend,
            "status": "ready",
            "sample_rate": self.tts.sample_rate,
            "error": "",
        }

    def tts_engines(self) -> list[dict[str, object]]:
        engines = getattr(self.tts, "engines", None)
        if engines is not None:
            return engines()
        return [
            {
                "id": self.config.tts_backend,
                "label": self.config.tts_backend,
                "description": "Configured TTS backend.",
                "kind": "local",
                "status": "ready",
            }
        ]

    async def _select_tts_engine(self, peer: Any, engine: str) -> bool:
        selector = getattr(self.tts, "select_engine", None)
        await peer.send_event(
            "tts.engine.loading",
            engine=engine,
            tts_engines=self.tts_engines(),
        )
        if selector is None:
            await peer.send_event(
                "tts.engine.ready",
                **self.tts_status(),
                tts_sample_rate=self.tts.sample_rate,
                tts_engines=self.tts_engines(),
            )
            return True
        try:
            status = await selector(engine)
        except Exception as exc:
            await peer.send_event(
                "tts.engine.error",
                engine=engine,
                message=str(exc),
                tts_engines=self.tts_engines(),
            )
            return False
        await peer.send_event(
            "tts.engine.ready",
            **status,
            tts_sample_rate=self.tts.sample_rate,
            tts_engines=self.tts_engines(),
        )
        return True

    def _new_code(self) -> str:
        return "".join(secrets.choice(EVENT_CODE_ALPHABET) for _ in range(6))

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


class EventRoom:
    def __init__(
        self,
        code: str,
        speaker: Any,
        config: LiveTTSConfig,
        stt: STTService,
        llm: LLMStreamer,
        tts: BaseTTS,
        *,
        source_language: str,
        speaker_sample_rate: int,
    ) -> None:
        self.code = normalize_event_code(code)
        self.speaker = speaker
        self.config = config
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.source_language = normalize_language_code(
            source_language,
            default="auto",
            allow_auto=True,
        )
        self.detector = AudioTurnDetector(
            sample_rate=speaker_sample_rate,
            speech_threshold=config.vad_speech_threshold,
            start_ms=config.vad_start_ms,
            end_ms=config.vad_end_ms,
            min_turn_ms=config.vad_min_turn_ms,
            preroll_ms=config.vad_preroll_ms,
            max_turn_s=config.vad_max_turn_s,
            adaptive=config.vad_adaptive,
            noise_calibration_ms=config.vad_noise_calibration_ms,
            start_multiplier=config.vad_start_multiplier,
            continue_multiplier=config.vad_continue_multiplier,
        )
        self.listeners: dict[str, Any] = {}
        self.turn_counter = 0
        self.closed = False
        self._lock = asyncio.Lock()
        self._turn_queue: asyncio.Queue[QueuedEventTurn] = asyncio.Queue()
        self._turn_worker_task: asyncio.Task | None = None
        self._active_cancel: threading.Event | None = None

    def limits(self) -> dict[str, object]:
        return {
            "speaker_count": 1,
            "listener_count": "many",
            "translation": (
                "Target-language groups are translated and synthesized one at a "
                "time in this MVP."
            ),
        }

    async def add_listener(self, listener: Any) -> None:
        async with self._lock:
            self.listeners[listener.client_id] = listener
            listener_count = len(self.listeners)
        await self._send_speaker_event(
            "event.listener_count",
            listener_count=listener_count,
        )
        await self.broadcast_event(
            "event.listener_count",
            listener_count=listener_count,
        )

    async def remove_listener(self, client_id: str) -> None:
        async with self._lock:
            listener = self.listeners.pop(client_id, None)
            listener_count = len(self.listeners)
        if listener is not None:
            listener.closed = True
        await self._send_speaker_event(
            "event.listener_count",
            listener_count=listener_count,
        )
        await self.broadcast_event(
            "event.listener_count",
            listener_count=listener_count,
        )

    async def update_listener_language(
        self,
        client_id: str,
        target_language: str,
    ) -> None:
        async with self._lock:
            listener = self.listeners.get(client_id)
            if listener is not None:
                listener.target_language = target_language

    async def listener_count(self) -> int:
        async with self._lock:
            return len(self.listeners)

    async def handle_audio_bytes(self, raw: bytes) -> None:
        if self.closed or len(raw) < 4 or len(raw) % 4 != 0:
            return
        chunk = np.frombuffer(raw, dtype="<f4")
        for event in self.detector.accept(chunk):
            if event.type == "meter":
                await self._send_speaker_event(
                    "audio.meter",
                    rms=event.rms,
                    noise_rms=event.noise_rms,
                    start_threshold=event.start_threshold,
                    continue_threshold=event.continue_threshold,
                    in_speech=event.in_speech,
                )
            elif event.type == "speech_start":
                await self._send_speaker_event("event.speech_start")
                await self.broadcast_event("event.speech_start")
            elif event.type == "speech_end" and event.samples is not None:
                await self._send_speaker_event(
                    "event.speech_end",
                    duration_ms=event.duration_ms,
                )
                await self.broadcast_event(
                    "event.speech_end",
                    duration_ms=event.duration_ms,
                )
                await self._turn_queue.put(
                    QueuedEventTurn(event.samples, event.sample_rate or 48000)
                )
                self._ensure_turn_worker()

    async def process_turn(self, samples: np.ndarray, sample_rate: int) -> None:
        if self.closed:
            return
        self.turn_counter += 1
        turn_id = self.turn_counter
        cancel = threading.Event()
        self._active_cancel = cancel
        started_at = time.monotonic()
        await self._send_speaker_event("event.turn.started", turn_id=turn_id)
        await self.broadcast_event("event.turn.started", turn_id=turn_id)
        try:
            stt_samples = pad_audio_edges(
                samples,
                sample_rate,
                lead_ms=self.config.stt_lead_padding_ms,
                tail_ms=self.config.stt_tail_padding_ms,
            )
            stt_started = time.monotonic()
            stt_result = await self.stt.transcribe(
                float32_to_wav_bytes(stt_samples, sample_rate),
                cancel,
                language=self.source_language,
            )
            if cancel.is_set() or self.closed:
                return
            source_text = str(stt_result.get("text") or "").strip()
            stt_latency_ms = int((time.monotonic() - stt_started) * 1000)
            await self._send_speaker_event(
                "event.source_text",
                turn_id=turn_id,
                text=source_text,
                language=stt_result.get("language"),
                requested_language=self.source_language,
                latency_ms=stt_latency_ms,
            )
            await self.broadcast_event(
                "event.source_text",
                turn_id=turn_id,
                text=source_text,
                language=stt_result.get("language"),
                requested_language=self.source_language,
                latency_ms=stt_latency_ms,
            )
            if not source_text:
                await self._send_speaker_event("event.turn.empty", turn_id=turn_id)
                await self.broadcast_event("event.turn.empty", turn_id=turn_id)
                return

            target_languages = await self.target_languages()
            if not target_languages:
                await self._send_speaker_event(
                    "event.no_listeners",
                    turn_id=turn_id,
                    message="No listeners are connected.",
                )
                return

            for target_language in target_languages:
                if cancel.is_set() or self.closed:
                    return
                await self._translate_and_speak(
                    turn_id,
                    source_text,
                    target_language,
                    cancel,
                    turn_started_at=started_at,
                )
            total_latency_ms = int((time.monotonic() - started_at) * 1000)
            await self._send_speaker_event(
                "event.turn.done",
                turn_id=turn_id,
                latency_ms=total_latency_ms,
            )
            await self.broadcast_event(
                "event.turn.done",
                turn_id=turn_id,
                latency_ms=total_latency_ms,
            )
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception as exc:
            logger.exception("event turn failed code=%s turn_id=%s", self.code, turn_id)
            await self._send_speaker_event(
                "error",
                turn_id=turn_id,
                message=str(exc),
            )
            await self.broadcast_event(
                "error",
                turn_id=turn_id,
                message=str(exc),
            )
        finally:
            if self._active_cancel is cancel:
                self._active_cancel = None

    async def target_languages(self) -> list[str]:
        async with self._lock:
            languages = {
                normalize_language_code(
                    getattr(listener, "target_language", None),
                    default=self.config.tts_language or "en",
                )
                for listener in self.listeners.values()
                if not getattr(listener, "closed", False)
            }
        return sorted(languages)

    async def broadcast_event(self, event_type: str, **payload: Any) -> None:
        listeners = await self._listeners_snapshot()
        await asyncio.gather(
            *(listener.send_event(event_type, **payload) for listener in listeners),
            return_exceptions=True,
        )
        await self._drop_closed_listeners(listeners)

    async def close(self, *, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        if self._active_cancel is not None:
            self._active_cancel.set()
        if self._turn_worker_task and not self._turn_worker_task.done():
            self._turn_worker_task.cancel()
        await self._send_speaker_event("event.ended", reason=reason)
        await self.broadcast_event("event.ended", event_code=self.code, reason=reason)
        await self.speaker.close()
        for listener in await self._listeners_snapshot():
            await listener.close()
        async with self._lock:
            self.listeners.clear()

    def _ensure_turn_worker(self) -> None:
        if self._turn_worker_task and not self._turn_worker_task.done():
            return
        self._turn_worker_task = asyncio.create_task(
            self._process_turn_queue(),
            name=f"event-room-{self.code}-turns",
        )

    async def _process_turn_queue(self) -> None:
        try:
            while not self.closed:
                try:
                    queued = await asyncio.wait_for(self._turn_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    if self._turn_queue.empty():
                        break
                    continue
                try:
                    await self.process_turn(queued.samples, queued.sample_rate)
                finally:
                    self._turn_queue.task_done()
        finally:
            if self._turn_worker_task is asyncio.current_task():
                self._turn_worker_task = None

    async def _translate_and_speak(
        self,
        turn_id: int,
        source_text: str,
        target_language: str,
        cancel: threading.Event,
        *,
        turn_started_at: float,
    ) -> None:
        segment_queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_text: list[str] = []
        first_audio_logged = False

        async def produce_text() -> None:
            segmenter = SentenceAccumulator(
                min_first_chars=self.config.segment_min_first_chars,
                max_first_chars=self.config.segment_max_first_chars,
                min_next_chars=self.config.segment_min_next_chars,
                max_next_chars=self.config.segment_max_next_chars,
                normalize_segments=True,
                language=target_language,
            )
            await self._broadcast_to_language(
                target_language,
                "event.translation.started",
                turn_id=turn_id,
                target_language=target_language,
                target_language_label=language_label(target_language),
            )
            try:
                async for piece in self.llm.stream(
                    source_text,
                    [],
                    cancel,
                    language=target_language,
                    mode="translator",
                    source_language=self.source_language,
                    target_language=target_language,
                    live_translation=False,
                ):
                    if cancel.is_set() or self.closed:
                        break
                    full_text.append(piece)
                    await self._broadcast_to_language(
                        target_language,
                        "event.translation_delta",
                        turn_id=turn_id,
                        target_language=target_language,
                        text=piece,
                    )
                    for segment in segmenter.push(piece):
                        await segment_queue.put(segment)
                for segment in segmenter.flush():
                    await segment_queue.put(segment)
            finally:
                await segment_queue.put(None)

        async def consume_tts() -> None:
            nonlocal first_audio_logged
            first = True
            segment_index = 0
            tts_state = self.tts.create_turn_state()
            voice_config = VoiceSessionConfig.from_config(
                self.config,
                self.tts.sample_rate,
                language=target_language,
                provider=str(self._tts_status().get("engine") or self.config.tts_backend),
                voice_id=str(self._tts_status().get("engine") or self.config.tts_backend),
            )
            loudness = AudioLoudnessSmoother(
                target_lufs=voice_config.loudness_target_lufs,
                enabled=voice_config.loudness_enabled,
            )
            crossfader = AudioCrossfader(
                self.tts.sample_rate,
                voice_config.crossfade_ms,
            )
            while True:
                segment = await segment_queue.get()
                if segment is None:
                    break
                if cancel.is_set() or self.closed:
                    break
                await self._broadcast_to_language(
                    target_language,
                    "event.audio_start",
                    turn_id=turn_id,
                    segment_index=segment_index,
                    target_language=target_language,
                    text=segment,
                    sample_rate=self.tts.sample_rate,
                )
                synth_started = time.monotonic()
                waveform = await self.tts.synthesize(
                    segment,
                    first=first,
                    state=tts_state,
                    language=target_language,
                    voice_config=voice_config,
                )
                if cancel.is_set() or self.closed:
                    break
                waveform = loudness.process(waveform)
                output_waveform = crossfader.process(waveform)
                latency_ms = int((time.monotonic() - synth_started) * 1000)
                duration_ms = int(len(waveform) / self.tts.sample_rate * 1000)
                await self._broadcast_to_language(
                    target_language,
                    "event.audio_ready",
                    turn_id=turn_id,
                    segment_index=segment_index,
                    target_language=target_language,
                    latency_ms=latency_ms,
                    duration_ms=duration_ms,
                )
                ttfa_started_at = None
                if segment_index == 0 and not first_audio_logged:
                    ttfa_started_at = turn_started_at
                    first_audio_logged = True
                await self._send_waveform_to_language(
                    target_language,
                    output_waveform,
                    turn_id=turn_id,
                    turn_started_at=ttfa_started_at,
                )
                await self._broadcast_to_language(
                    target_language,
                    "event.audio_end",
                    turn_id=turn_id,
                    segment_index=segment_index,
                    target_language=target_language,
                )
                first = False
                segment_index += 1

            if not cancel.is_set() and not self.closed:
                tail = crossfader.flush()
                if tail.size:
                    await self._send_waveform_to_language(target_language, tail)

        producer = asyncio.create_task(produce_text(), name=f"event-llm-{turn_id}")
        consumer = asyncio.create_task(consume_tts(), name=f"event-tts-{turn_id}")
        try:
            await asyncio.gather(producer, consumer)
        finally:
            producer.cancel()
            consumer.cancel()

        translation = "".join(full_text).strip()
        await self._send_speaker_event(
            "event.translation_final",
            turn_id=turn_id,
            target_language=target_language,
            target_language_label=language_label(target_language),
            text=translation,
        )
        await self._broadcast_to_language(
            target_language,
            "event.translation_final",
            turn_id=turn_id,
            target_language=target_language,
            target_language_label=language_label(target_language),
            text=translation,
        )

    async def _send_waveform_to_language(
        self,
        target_language: str,
        waveform: np.ndarray,
        *,
        turn_id: int | None = None,
        turn_started_at: float | None = None,
    ) -> None:
        if waveform.size == 0:
            return
        pcm = pcm16_bytes_from_float32(waveform)
        for frame in iter_pcm_frames(pcm, self.tts.sample_rate, self.config.tts_frame_ms):
            if self.closed:
                return
            if turn_started_at is not None:
                latency_ms = int((time.monotonic() - turn_started_at) * 1000)
                await self._send_speaker_event(
                    "event.first_audio",
                    turn_id=turn_id,
                    target_language=target_language,
                    latency_ms=latency_ms,
                )
                turn_started_at = None
            listeners = await self._listeners_snapshot(target_language=target_language)
            await asyncio.gather(
                *(listener.send_bytes(frame) for listener in listeners),
                return_exceptions=True,
            )
            await self._drop_closed_listeners(listeners)
            await asyncio.sleep(0)

    async def _broadcast_to_language(
        self,
        language_code: str,
        event_type: str,
        **payload: Any,
    ) -> None:
        listeners = await self._listeners_snapshot(target_language=language_code)
        await asyncio.gather(
            *(listener.send_event(event_type, **payload) for listener in listeners),
            return_exceptions=True,
        )
        await self._drop_closed_listeners(listeners)

    async def _listeners_snapshot(
        self,
        *,
        target_language: str | None = None,
    ) -> list[Any]:
        async with self._lock:
            listeners = list(self.listeners.values())
        if target_language is None:
            return [listener for listener in listeners if not listener.closed]
        return [
            listener
            for listener in listeners
            if not listener.closed and listener.target_language == target_language
        ]

    async def _drop_closed_listeners(self, listeners: list[Any]) -> None:
        closed_ids = [
            listener.client_id for listener in listeners if getattr(listener, "closed", False)
        ]
        if not closed_ids:
            return
        async with self._lock:
            for client_id in closed_ids:
                self.listeners.pop(client_id, None)

    async def _send_speaker_event(self, event_type: str, **payload: Any) -> None:
        await self.speaker.send_event(event_type, event_code=self.code, **payload)

    def _tts_status(self) -> dict[str, object]:
        status = getattr(self.tts, "status", None)
        if status is not None:
            return status()
        return {
            "engine": self.config.tts_backend,
            "status": "ready",
            "sample_rate": self.tts.sample_rate,
            "error": "",
        }


def normalize_event_code(value: object) -> str:
    raw = str(value or "").upper()
    return "".join(ch for ch in raw if ch in string.ascii_uppercase + string.digits)[:12]
