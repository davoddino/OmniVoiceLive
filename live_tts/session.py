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
    AudioCrossfader,
    AudioTurnDetector,
    float32_to_wav_bytes,
    iter_pcm_frames,
    normalize_loudness_rms,
    pad_audio_edges,
    pcm16_bytes_from_float32,
)
from live_tts.config import LiveTTSConfig
from live_tts.llm import LLMStreamer
from live_tts.playback import drain_segment_queue
from live_tts.rag import RAGResult, RAGRetriever
from live_tts.recording import AsyncSessionRecorder
from live_tts.segmenter import SentenceAccumulator
from live_tts.stt import STTService
from live_tts.tts import BaseTTS, TTSTurnState
from live_tts.voice import VoiceSessionConfig


logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = {
    "it": "Italiano",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
}


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
        rag: RAGRetriever | None = None,
    ) -> None:
        self.websocket = websocket
        self.config = config
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.rag = rag
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
            adaptive=config.vad_adaptive,
            noise_calibration_ms=config.vad_noise_calibration_ms,
            start_multiplier=config.vad_start_multiplier,
            continue_multiplier=config.vad_continue_multiplier,
        )
        self.history: list[dict[str, str]] = []
        self.language = self._normalize_language(config.tts_language)
        self.voice_config = VoiceSessionConfig.from_config(
            config,
            tts.sample_rate,
            language=self.language,
        )
        self.recorder = AsyncSessionRecorder(config, self.session_id)
        self.turn_counter = 0
        self.current: TurnRuntime | None = None
        self.tts_session_state: TTSTurnState | None = (
            tts.create_turn_state()
            if config.tts_voice_mode.strip().lower().replace("-", "_")
            in {
                "fixed_reference",
                "reference",
                "reference_voice",
                "voice_reference",
                "session_anchor",
                "anchor",
                "session_self_condition",
            }
            else None
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
            tts_engine=self._tts_engine_status(),
            tts_engines=self._tts_engines(),
            language=self.language,
            languages=SUPPORTED_LANGUAGES,
            client_barge_threshold=self.config.client_barge_threshold,
            client_barge_stop_ms=self.config.client_barge_stop_ms,
            client_barge_commit_ms=self.config.client_barge_commit_ms,
            client_barge_cooldown_ms=self.config.client_barge_cooldown_ms,
            voice_config=self.voice_config.as_dict(),
            rag_enabled=self.config.rag_enabled,
            recording_enabled=self.config.recording_enabled,
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
            await self.recorder.close()
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
            self.language = self._normalize_language(data.get("language"))
            selected_engine = str(data.get("tts_engine") or "").strip()
            if selected_engine and not await self._select_tts_engine(selected_engine):
                return
            self._refresh_tts_runtime_state()
            self.detector.set_sample_rate(sample_rate)
            await self.recorder.start(
                input_sample_rate=sample_rate,
                output_sample_rate=self.tts.sample_rate,
                voice_config=self.voice_config,
                metadata={
                    "client": str(self.websocket.client or "unknown"),
                    "language": self.language,
                },
            )
            logger.info(
                "session started session_id=%s sample_rate=%s language=%s",
                self.session_id,
                sample_rate,
                self.language,
            )
            await self.send_event(
                "session.started",
                sample_rate=sample_rate,
                language=self.language,
                language_label=SUPPORTED_LANGUAGES[self.language],
                tts_sample_rate=self.tts.sample_rate,
                tts_engine=self._tts_engine_status(),
                recording_dir=(
                    str(self.recorder.paths.root)
                    if self.recorder.paths is not None
                    else ""
                ),
            )
        elif msg_type == "tts.engine.select":
            engine = str(data.get("engine") or "").strip()
            if not engine:
                await self.send_event("error", message="Missing TTS engine id.")
                return
            await self.cancel_current_turn("tts_engine_select")
            self.detector.reset()
            await self._select_tts_engine(engine)
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

    async def _select_tts_engine(self, engine: str) -> bool:
        selector = getattr(self.tts, "select_engine", None)
        if selector is None:
            await self.send_event(
                "tts.engine.ready",
                **self._tts_engine_status(),
                tts_sample_rate=self.tts.sample_rate,
                tts_engines=self._tts_engines(),
            )
            return True

        await self.send_event(
            "tts.engine.loading",
            engine=engine,
            tts_engines=self._tts_engines(),
        )
        try:
            status = await selector(engine)
        except Exception as exc:
            await self.send_event(
                "tts.engine.error",
                engine=engine,
                message=str(exc),
                tts_engines=self._tts_engines(),
            )
            return False

        self._refresh_tts_runtime_state()
        self.recorder.event(
            "tts_engine_selected",
            engine=status.get("engine"),
            sample_rate=self.tts.sample_rate,
        )
        await self.send_event(
            "tts.engine.ready",
            **status,
            tts_sample_rate=self.tts.sample_rate,
            tts_engines=self._tts_engines(),
            voice_config=self.voice_config.as_dict(),
        )
        return True

    def _refresh_tts_runtime_state(self) -> None:
        engine_status = self._tts_engine_status()
        engine = str(engine_status.get("engine") or self.config.tts_backend)
        self.voice_config = VoiceSessionConfig.from_config(
            self.config,
            self.tts.sample_rate,
            language=self.language,
            provider=engine,
            voice_id=engine,
        )
        self.tts_session_state = self.tts.create_turn_state()

    def _tts_engine_status(self) -> dict[str, object]:
        status = getattr(self.tts, "status", None)
        if status is not None:
            return status()
        return {
            "engine": self.config.tts_backend,
            "status": "ready",
            "sample_rate": self.tts.sample_rate,
            "error": "",
        }

    def _tts_engines(self) -> list[dict[str, object]]:
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

    async def _handle_audio_bytes(self, raw: bytes) -> None:
        if len(raw) < 4 or len(raw) % 4 != 0:
            return
        chunk = np.frombuffer(raw, dtype="<f4")
        self.recorder.record_input(chunk)
        for event in self.detector.accept(chunk):
            if event.type == "meter":
                await self.send_event(
                    "audio.meter",
                    rms=event.rms,
                    noise_rms=event.noise_rms,
                    start_threshold=event.start_threshold,
                    continue_threshold=event.continue_threshold,
                    in_speech=event.in_speech,
                )
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
            self.recorder.note_turn()
            self.recorder.event("turn_started", turn_id=turn_id)
            await self.send_event("turn.started", turn_id=turn_id)
            stt_samples = pad_audio_edges(
                samples,
                sample_rate,
                lead_ms=self.config.stt_lead_padding_ms,
                tail_ms=self.config.stt_tail_padding_ms,
            )
            logger.info(
                "stt input session_id=%s turn_id=%s raw_duration_ms=%s padded_duration_ms=%s lead_padding_ms=%s tail_padding_ms=%s",
                self.session_id,
                turn_id,
                int(samples.size / sample_rate * 1000),
                int(stt_samples.size / sample_rate * 1000),
                self.config.stt_lead_padding_ms,
                self.config.stt_tail_padding_ms,
            )
            wav_bytes = float32_to_wav_bytes(stt_samples, sample_rate)

            stt_started = time.monotonic()
            stt_result = await self.stt.transcribe(
                wav_bytes,
                runtime.cancel,
                language=self.language,
            )
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
                requested_language=self.language,
                latency_ms=stt_latency_ms,
            )
            self.recorder.note_latency("stt_ms", stt_latency_ms)
            if not user_text:
                logger.info("turn empty session_id=%s turn_id=%s", self.session_id, turn_id)
                await self.send_event("turn.empty", turn_id=turn_id)
                return
            self.recorder.transcript(
                "user",
                user_text,
                turn_id=turn_id,
                stt_ms=stt_latency_ms,
            )
            if needs_human_escalation(user_text):
                self.recorder.request_escalation("user_requested_human")

            history_before_turn = list(self.history)
            assistant_text = await self._respond(runtime, user_text, history_before_turn)
            if not runtime.cancel.is_set():
                self.history.append({"role": "user", "content": user_text})
                if assistant_text:
                    self.history.append(
                        {"role": "assistant", "content": assistant_text}
                    )
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
            self.recorder.note_error(str(exc))
            await self.send_event("error", turn_id=turn_id, message=str(exc))
        finally:
            if self.current is runtime:
                self.current = None

    async def _respond(
        self,
        runtime: TurnRuntime,
        user_text: str,
        history: list[dict[str, str]],
    ) -> str:
        turn_id = runtime.turn_id
        segment_queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_text: list[str] = []
        rag_result = RAGResult(False, "", [], 0)
        llm_latency_ms = 0

        async def produce_text() -> None:
            nonlocal rag_result, llm_latency_ms
            segmenter = SentenceAccumulator(
                min_first_chars=self.config.segment_min_first_chars,
                max_first_chars=self.config.segment_max_first_chars,
                min_next_chars=self.config.segment_min_next_chars,
                max_next_chars=self.config.segment_max_next_chars,
                normalize_segments=True,
                language=self.language,
            )
            try:
                await self.send_event("assistant.thinking", turn_id=turn_id)
                rag_result = await self._retrieve_rag(user_text, runtime)
                llm_started = time.monotonic()
                async for piece in self.llm.stream(
                    user_text,
                    history,
                    runtime.cancel,
                    language=self.language,
                    rag_context=rag_result.context,
                ):
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
                llm_latency_ms = int((time.monotonic() - llm_started) * 1000)
                if not runtime.cancel.is_set():
                    self.recorder.note_latency("llm_ms", llm_latency_ms)
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
            cancelled_pending = 0
            tts_state = self.tts_session_state or self.tts.create_turn_state()
            crossfader = AudioCrossfader(
                self.tts.sample_rate,
                self.voice_config.crossfade_ms,
            )
            try:
                while True:
                    segment = await segment_queue.get()
                    if segment is None:
                        break
                    if runtime.cancel.is_set() or self.current is not runtime:
                        cancelled_pending += 1 + drain_segment_queue(segment_queue)
                        break

                    logger.info(
                        "tts start session_id=%s turn_id=%s segment_index=%s first=%s chars=%s",
                        self.session_id,
                        turn_id,
                        segment_index,
                        first,
                        len(segment),
                    )
                    self.recorder.event(
                        "tts_chunk_started",
                        turn_id=turn_id,
                        chunk_id=segment_index,
                        chars=len(segment),
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
                        language=self.language,
                        voice_config=self.voice_config,
                    )
                    if runtime.cancel.is_set() or self.current is not runtime:
                        cancelled_pending += 1 + drain_segment_queue(segment_queue)
                        break

                    waveform = normalize_loudness_rms(
                        waveform,
                        target_lufs=self.voice_config.loudness_target_lufs,
                        enabled=self.voice_config.loudness_enabled,
                    )
                    output_waveform = crossfader.process(waveform)
                    tts_latency_ms = int((time.monotonic() - synth_started) * 1000)
                    duration_ms = int(len(waveform) / self.tts.sample_rate * 1000)
                    pcm_size = len(pcm16_bytes_from_float32(output_waveform))
                    logger.info(
                        "tts ready session_id=%s turn_id=%s segment_index=%s latency_ms=%s duration_ms=%s bytes=%s",
                        self.session_id,
                        turn_id,
                        segment_index,
                        tts_latency_ms,
                        duration_ms,
                        pcm_size,
                    )
                    self.recorder.note_latency("tts_ms", tts_latency_ms)
                    self.recorder.note_tts_chunk()
                    self.recorder.transcript(
                        "assistant",
                        segment,
                        turn_id=turn_id,
                        llm_ms=llm_latency_ms or None,
                        tts_ms=tts_latency_ms,
                        chunk_id=segment_index,
                        rag_used=rag_result.used,
                    )
                    await self.send_event(
                        "assistant.audio_ready",
                        turn_id=turn_id,
                        segment_index=segment_index,
                        duration_ms=duration_ms,
                        latency_ms=tts_latency_ms,
                    )
                    sent = await self._send_waveform(runtime, output_waveform)
                    if not sent:
                        cancelled_pending += drain_segment_queue(segment_queue)
                        break

                    await self.send_event(
                        "assistant.audio_end",
                        turn_id=turn_id,
                        segment_index=segment_index,
                    )
                    self.recorder.event(
                        "tts_chunk_played",
                        turn_id=turn_id,
                        chunk_id=segment_index,
                    )
                    first = False
                    segment_index += 1

                if not runtime.cancel.is_set() and self.current is runtime:
                    tail = crossfader.flush()
                    if tail.size:
                        await self._send_waveform(runtime, tail)
            finally:
                if runtime.cancel.is_set() or self.current is not runtime:
                    cancelled_pending += drain_segment_queue(segment_queue)
                if cancelled_pending:
                    self.recorder.note_tts_cancelled(cancelled_pending)
                    self.recorder.event(
                        "tts_chunks_cancelled",
                        turn_id=turn_id,
                        count=cancelled_pending,
                    )

        producer = asyncio.create_task(produce_text(), name=f"llm-{turn_id}")
        consumer = asyncio.create_task(consume_tts(), name=f"tts-{turn_id}")
        try:
            await asyncio.gather(producer, consumer)
        finally:
            producer.cancel()
            consumer.cancel()

        return "".join(full_text).strip()

    async def _retrieve_rag(
        self,
        user_text: str,
        runtime: TurnRuntime,
    ) -> RAGResult:
        if self.rag is None:
            return RAGResult(False, "", [], 0)
        result = await self.rag.retrieve(user_text, runtime.cancel)
        fallback = bool(result.timed_out or result.error)
        self.recorder.note_rag(result.used, result.latency_ms, fallback=fallback)
        self.recorder.event(
            "rag_result",
            turn_id=runtime.turn_id,
            used=result.used,
            latency_ms=result.latency_ms,
            timed_out=result.timed_out,
            error=result.error,
        )
        await self.send_event(
            "rag.result",
            turn_id=runtime.turn_id,
            used=result.used,
            latency_ms=result.latency_ms,
            timed_out=result.timed_out,
        )
        if fallback and not self.config.rag_fallback_to_llm:
            raise RuntimeError("RAG retrieval failed and fallback is disabled")
        return result if result.used else RAGResult(False, "", [], result.latency_ms)

    async def _send_waveform(self, runtime: TurnRuntime, waveform: np.ndarray) -> bool:
        if waveform.size == 0:
            return True
        self.recorder.record_output(waveform)
        pcm = pcm16_bytes_from_float32(waveform)
        for frame in iter_pcm_frames(pcm, self.tts.sample_rate, self.config.tts_frame_ms):
            if runtime.cancel.is_set() or self.current is not runtime:
                return False
            self.recorder.note_first_audio()
            await self.send_bytes(frame)
            await asyncio.sleep(0)
        return True

    async def cancel_current_turn(self, reason: str) -> None:
        runtime = self.current
        if runtime is None:
            return
        runtime.cancel.set()
        if reason == "barge_in":
            self.recorder.note_barge_in()
            self.recorder.event(
                "barge_in",
                turn_id=runtime.turn_id,
                reason=reason,
            )
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

    def _normalize_language(self, value: object) -> str:
        language = str(value or self.config.tts_language or "it").strip().lower()
        return language if language in SUPPORTED_LANGUAGES else "it"

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


def needs_human_escalation(text: str) -> bool:
    lowered = text.lower()
    triggers = (
        "operatore",
        "persona",
        "umano",
        "parlare con qualcuno",
        "assistenza umana",
        "responsabile",
    )
    return any(trigger in lowered for trigger in triggers)
